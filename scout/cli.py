from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

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
from scout.benchmarks.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
)
from scout.benchmarks.utils import export_benchmark_results, aggregate_metrics
from scout.benchmarks.regression import detect_regression
from scout.benchmarks.synthetic import (
    generate_synthetic_index,
    generate_synthetic_queries,
)

console = Console()


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
    search.add_argument("--ranking", choices=RANKINGS, default="robust")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--json", action="store_true")
    search.add_argument("--explain", action="store_true")

    # BENCHMARK
    benchmark = sub.add_parser("benchmark")
    bench = benchmark.add_subparsers(dest="bench_cmd", required=True)

    bench_index = bench.add_parser("index")
    bench_index.add_argument("--config", type=Path, required=True)

    bench_run = bench.add_parser("run")
    bench_run.add_argument("--config", type=Path, required=True)
    bench_run.add_argument("--output", type=Path)
    bench_run.add_argument("--repeats", type=int, default=3)
    bench_run.add_argument("--warmup", type=int, default=1)
    bench_run.add_argument("--previous", type=Path)

    # SYNTHETIC
    synth = bench.add_parser("synthetic")
    synth.add_argument("--docs", type=int, default=10_000)
    synth.add_argument("--queries", type=int, default=500)
    synth.add_argument("--k", type=int, default=10)

    return parser


# ---------------- Engine Builder ---------------- #
def build_engine(
    *,
    records_file: Path,
    ranking_name: str,
) -> SearchEngine:
    records = list(load_records(records_file))
    ranking = RANKINGS[ranking_name]()
    state = IndexState()
    AutoSaver(state)
    return SearchEngine.from_records(records, ranking=ranking, state=state)


# ---------------- Commands ---------------- #
def cmd_search(args: argparse.Namespace) -> None:
    engine = build_engine(
        records_file=args.records_file,
        ranking_name=args.ranking,
    )
    query = input("Query: ")
    results = engine.search(query, limit=args.limit)

    if args.explain:
        results = explain_query(engine, query, limit=args.limit)

    if args.json:
        console.print_json(json.dumps([
            {"doc_id": doc_id, "score": r.score}
            for doc_id, r in results
        ]))
        return

    table = Table(title="Search Results")
    table.add_column("Doc ID")
    table.add_column("Score", justify="right")
    for doc_id, r in results:
        table.add_row(str(doc_id), f"{r.score:.4f}")
    console.print(table)


def cmd_benchmark_run(args: argparse.Namespace) -> None:
    cfg = json.loads(args.config.read_text())

    index = build_benchmark_index(**cfg["index"])
    queries = [
        BenchmarkQuery(
            query=q["query"],
            relevant_doc_ids=frozenset(q["relevant_doc_ids"]),
        )
        for q in cfg["queries"]
    ]

    engine = build_engine(
        records_file=Path(cfg["index"]["dataset_path"]),
        ranking_name=cfg.get("ranking", "robust"),
    )

    results = run_benchmark(
        engine=engine,
        index=index,
        queries=queries,
        k=cfg["k"],
        warmup=args.warmup,
        repeats=args.repeats,
    )

    table = Table(title="Benchmark Results")
    table.add_column("Query")
    table.add_column("P@K")
    table.add_column("R@K")
    table.add_column("MRR")
    table.add_column("Latency (ms)")

    for q, r in zip(queries, results):
        table.add_row(
            q.query,
            f"{precision_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=cfg['k']):.3f}",
            f"{recall_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=cfg['k']):.3f}",
            f"{mean_reciprocal_rank(retrieved=r.retrieved, relevant=q.relevant_doc_ids):.3f}",
            f"{r.latency_ms:.2f}",
        )

    console.print(table)

    agg = aggregate_metrics(results, queries, cfg["k"])
    console.print("[bold yellow]Aggregated Metrics[/bold yellow]")
    for k, v in agg.items():
        console.print(f"{k}: {v:.4f}")

    if args.output:
        export_benchmark_results(results, args.output)

    if args.previous:
        detect_regression(
            new_results=results,
            queries=queries,
            previous_results_file=args.previous,
            k=cfg["k"],
        )


def cmd_benchmark_synthetic(args: argparse.Namespace) -> None:
    index = generate_synthetic_index(name="synthetic", num_docs=args.docs)
    queries = generate_synthetic_queries(index=index, num_queries=args.queries)

    console.print(
        f"[green]Generated synthetic benchmark:[/green] "
        f"{args.docs} docs, {args.queries} queries"
    )


# ---------------- Main ---------------- #
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "benchmark":
        if args.bench_cmd == "run":
            cmd_benchmark_run(args)
        elif args.bench_cmd == "synthetic":
            cmd_benchmark_synthetic(args)


if __name__ == "__main__":
    main()
