# scout/cli.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.ranking.composite import CompositeRanking
from scout.ranking.recency import RecencyRanking
from scout.explain import explain_query
from scout.state.signals import IndexState
from scout.state.persistence import AutoSaver
from scout.data.loader import load_records

from scout.benchmarks.index import build_benchmark_index
from scout.benchmarks.run import run_benchmark, BenchmarkQuery
from scout.benchmarks.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank
from scout.benchmarks.utils import export_benchmark_results, aggregate_metrics
from scout.benchmarks.regression import detect_regression

# ---------------- Rankings ---------------- #
RANKINGS = {
    "robust": RobustRanking,
    "bm25": BM25Ranking,
    "fusion": lambda: CompositeRanking(
        strategies=[BM25Ranking(), RobustRanking()],
        weights=[0.5, 0.5],
        recency=RecencyRanking(decay_days=30.0, max_boost=1.0),
    ),
}

# ---------------- CLI Parser ---------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("scout", description="ScoutSearch CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # SEARCH
    search = sub.add_parser("search")
    search.add_argument("--records-file", type=Path, required=True)
    search.add_argument("--limit-docs", type=int)
    search.add_argument("--ranking", choices=RANKINGS, default="robust")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--json", action="store_true")
    search.add_argument("--explain", action="store_true")
    search.add_argument("--field-weights", type=str)
    search.add_argument("--persist", type=Path)

    # INTERACTIVE
    interactive = sub.add_parser("interactive")
    interactive.add_argument("--records-file", type=Path, required=True)
    interactive.add_argument("--ranking", choices=RANKINGS, default="robust")

    # BENCHMARK
    benchmark = sub.add_parser("benchmark")
    bench_sub = benchmark.add_subparsers(dest="bench_cmd", required=True)

    # Build index
    build = bench_sub.add_parser("index")
    build.add_argument("--config", type=Path, required=True)

    # Run benchmark
    run = bench_sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, help="Export benchmark results (JSON/CSV)")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--previous", type=Path, help="Previous benchmark JSON for regression detection")

    return parser

# ---------------- Engine Builder ---------------- #
def build_engine(
    *,
    records_file: Path,
    ranking_name: str,
    limit_docs: Optional[int] = None,
    field_weights: Optional[str] = None,
    persist: Optional[Path] = None,
) -> SearchEngine:
    records = list(load_records(records_file, limit=limit_docs))
    ranking = RANKINGS[ranking_name]()
    state = IndexState()
    if persist:
        AutoSaver(state)
    weights = json.loads(field_weights) if field_weights else None
    return SearchEngine.from_records(records, ranking=ranking, state=state, field_weights=weights)

# ---------------- Commands ---------------- #
def cmd_search(args: argparse.Namespace) -> None:
    engine = build_engine(
        records_file=args.records_file,
        ranking_name=args.ranking,
        limit_docs=args.limit_docs,
        field_weights=args.field_weights,
        persist=args.persist,
    )
    query = input("Query: ")
    results = engine.search(query, limit=args.limit)
    if args.explain:
        results = explain_query(engine, query, limit=args.limit)
    if args.json:
        print(json.dumps([{"doc_id": doc_id, "score": r.score, "components": r.components, "per_term": r.per_term} for doc_id, r in results], indent=2))
        return
    for doc_id, r in results:
        print(f"{doc_id}  score={r.score:.4f}")
        if r.per_term:
            print(f"  per_term={r.per_term}")

def cmd_interactive(args: argparse.Namespace) -> None:
    engine = build_engine(records_file=args.records_file, ranking_name=args.ranking)
    print("Interactive mode. Type 'exit' to quit.")
    while True:
        q = input(">> ").strip()
        if q == "exit":
            break
        for doc_id, r in engine.search(q):
            print(f"{doc_id}: {r.score:.4f}")

def cmd_benchmark_index(args: argparse.Namespace) -> None:
    cfg = json.loads(args.config.read_text())
    build_benchmark_index(**cfg["index"])
    print("Benchmark index built.")

def cmd_benchmark_run(args: argparse.Namespace) -> None:
    cfg = json.loads(args.config.read_text())
    index = build_benchmark_index(**cfg["index"])
    queries = [BenchmarkQuery(query=q["query"], relevant_doc_ids=frozenset(q["relevant_doc_ids"])) for q in cfg["queries"]]
    engine = build_engine(
        records_file=Path(cfg["index"]["dataset_path"]),
        ranking_name=cfg.get("ranking", "robust"),
        limit_docs=cfg["index"].get("limit"),
    )
    results = run_benchmark(
        engine=engine,
        index=index,
        queries=queries,
        k=cfg["k"],
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # Per-query metrics
    for q, r in zip(queries, results):
        print(f"\nQuery: {q.query}")
        print(f"P@{cfg['k']}: {precision_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=cfg['k']):.3f}")
        print(f"R@{cfg['k']}: {recall_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=cfg['k']):.3f}")
        print(f"MRR: {mean_reciprocal_rank(retrieved=r.retrieved, relevant=q.relevant_doc_ids):.3f}")
        print(f"Latency(ms): {r.latency_ms:.2f}")

    # Aggregate metrics
    agg = aggregate_metrics(results, queries, cfg["k"])
    print("\n--- Aggregated Metrics ---")
    for k, v in agg.items():
        print(f"{k}: {v:.4f}")

    # Export results
    if args.output:
        export_benchmark_results(results, args.output)
        print(f"Benchmark exported to {args.output}.json/.csv")

    # Regression detection
    if args.previous:
        detect_regression(new_results=results, queries=queries, previous_results_file=args.previous, k=cfg["k"])

# ---------------- Main ---------------- #
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "search":
        cmd_search(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "benchmark":
        if args.bench_cmd == "index":
            cmd_benchmark_index(args)
        elif args.bench_cmd == "run":
            cmd_benchmark_run(args)

if __name__ == "__main__":
    main()
