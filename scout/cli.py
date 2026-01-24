# scout/cli.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scout.data.loader import load_records
from scout.search.engine import SearchEngine
from scout.explain import explain_query

from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.ranking.composite import CompositeRanking
from scout.ranking.recency import RecencyRanking

from scout.benchmarks.config_loader import load_benchmark_config
from scout.benchmarks.run import run_benchmark, BenchmarkQuery
from scout.benchmarks.aggregate import aggregate_metrics
from scout.benchmarks.artifacts import write_benchmark_artifact
from scout.benchmarks.index import build_benchmark_index

console = Console()


# ---------------- Ranking Factory ---------------- #

def build_ranking(cfg: dict):
    rtype = cfg["type"]

    if rtype == "robust":
        return RobustRanking(**cfg.get("params", {}))
    if rtype == "bm25":
        return BM25Ranking(**cfg.get("params", {}))
    if rtype == "fusion":
        return CompositeRanking(
            strategies=[BM25Ranking(), RobustRanking()],
            weights=[0.5, 0.5],
            recency=RecencyRanking(**cfg.get("recency", {})),
        )

    raise ValueError(f"Unknown ranking type: {rtype}")


def build_engine(records_file: Path, ranking):
    records = list(load_records(records_file))
    return SearchEngine.from_records(records, ranking=ranking)


# ---------------- CLI ---------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("scout", description="ScoutSearch CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # SEARCH
    search = sub.add_parser("search", help="Run interactive search")
    search.add_argument("--records-file", type=Path, required=True)
    search.add_argument("--ranking", choices=["robust", "bm25"], default="robust")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--explain", action="store_true")
    search.add_argument("--json", action="store_true")

    # BENCHMARK
    bench = sub.add_parser("benchmark", help="Run benchmark from config")
    bench.add_argument("--config", type=Path, required=True)

    return parser


# ---------------- Commands ---------------- #

def cmd_search(args) -> int:
    ranking = RobustRanking() if args.ranking == "robust" else BM25Ranking()
    engine = build_engine(args.records_file, ranking)

    query = input("Query: ").strip()
    if not query:
        console.print("[red]Empty query[/red]")
        return 1

    results = (
        explain_query(engine, query, limit=args.limit)
        if args.explain
        else engine.search(query, limit=args.limit)
    )

    if args.json:
        console.print_json(json.dumps([
            {"doc_id": doc_id, "score": r.score}
            for doc_id, r in results
        ]))
        return 0

    table = Table(title="Search Results")
    table.add_column("Doc ID")
    table.add_column("Score", justify="right")

    for doc_id, r in results:
        table.add_row(str(doc_id), f"{r.score:.4f}")

    console.print(table)
    return 0


def cmd_benchmark(args) -> int:
    try:
        cfg = load_benchmark_config(args.config)
    except Exception as e:
        console.print(f"[red]Invalid benchmark config:[/red] {e}")
        return 1

    cfg = load_benchmark_config(args.config)

    # Build benchmark index (authoritative corpus snapshot)
    index = build_benchmark_index(
        name=cfg["name"],
        dataset_path=cfg["index"]["dataset_path"],
        id_field=cfg["index"]["id_field"],
        content_field=cfg["index"]["content_field"],
        metadata_fields=cfg["index"].get("metadata_fields"),
        limit=cfg["index"].get("limit"),
    )

    # Build search engine over same dataset
    records = list(load_records(cfg["index"]["dataset_path"]))
    ranking = build_ranking(cfg["ranking"])
    engine = SearchEngine.from_records(records, ranking=ranking)

    queries = [
        BenchmarkQuery(
            query=q["query"],
            relevant_doc_ids=frozenset(q["relevant_doc_ids"]),
        )
        for q in cfg["queries"]
    ]

    results = run_benchmark(
        engine=engine,
        index=index,  # ✅ FIXED — no None
        queries=queries,
        k=cfg["benchmark"]["k"],
        warmup=cfg["benchmark"].get("warmup", 0),
        repeats=cfg["benchmark"].get("repeats", 1),
        seed=cfg["benchmark"].get("seed"),
    )

    metrics = aggregate_metrics(
        results=results,
        queries=queries,
        k=cfg["benchmark"]["k"],
    )

    write_benchmark_artifact(
        path=Path(cfg["output"]),
        results=results,
        metadata=cfg,
    )

    console.print("[bold green]Benchmark complete[/bold green]")
    for k, v in metrics.items():
        console.print(f"{k}: {v:.4f}")

    return 0



# ---------------- Main ---------------- #

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "search":
        sys.exit(cmd_search(args))
    elif args.command == "benchmark":
        sys.exit(cmd_benchmark(args))


if __name__ == "__main__":
    main()
