# scout/cli.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scout.benchmarks.aggregate import aggregate_metrics
from scout.benchmarks.artifacts import load_benchmark_artifact, write_benchmark_artifact
from scout.benchmarks.config_loader import load_benchmark_config
from scout.benchmarks.index import build_benchmark_index
from scout.benchmarks.regression import RegressionReport, compare_benchmarks
from scout.benchmarks.run import BenchmarkQuery, run_benchmark
from scout.benchmarks.thresholds import RegressionThresholds
from scout.data.loader import load_records
from scout.explain import explain_query
from scout.ranking.bm25 import BM25Ranking
from scout.ranking.composite import CompositeRanking
from scout.ranking.recency import RecencyRanking
from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine

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

def build_engine(records_file: Path, ranking) -> SearchEngine:
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

    # REGRESSION
    regress = sub.add_parser("benchmark-regress", help="Run regression comparison")
    regress.add_argument("--baseline", type=Path, required=True)
    regress.add_argument("--candidate", type=Path, required=True)

    return parser

# ---------------- Commands ---------------- #
def cmd_search(args) -> int:
    ranking = RobustRanking() if args.ranking == "robust" else BM25Ranking()
    engine = build_engine(args.records_file, ranking)
    query = input("Query: ").strip()
    if not query:
        console.print("[red]Empty query[/red]")
        return 1
    results = explain_query(engine, query, limit=args.limit) if args.explain else engine.search(query, limit=args.limit)
    if args.json:
        console.print_json(json.dumps([{"doc_id": doc_id, "score": r.score} for doc_id, r in results]))
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
    index = build_benchmark_index(
        name=cfg.name,
        dataset_path=cfg.index.dataset_path,
        id_field=cfg.index.id_field,
        content_field=cfg.index.content_field,
        metadata_fields=cfg.index.metadata_fields,
        limit=cfg.index.limit,
    )
    ranking = build_ranking(cfg.ranking.model_dump())
    engine = build_engine(Path(cfg.index.dataset_path), ranking)
    queries = [BenchmarkQuery(query=q.query, relevant_doc_ids=frozenset(q.relevant_doc_ids)) for q in cfg.queries]
    results = run_benchmark(
        engine=engine,
        index=index,
        queries=queries,
        k=cfg.benchmark.k,
        warmup=cfg.benchmark.warmup,
        repeats=cfg.benchmark.repeats,
        seed=cfg.benchmark.seed,
    )
    metrics = aggregate_metrics(results=results, queries=queries, k=cfg.benchmark.k)
    artifact_path = Path(cfg.output)
    write_benchmark_artifact(path=artifact_path, results=results, metadata=cfg.model_dump())
    console.print("[bold green]Benchmark complete[/bold green]")
    console.print(f"Artifact written to [cyan]{artifact_path}[/cyan]")
    for k, v in metrics.items():
        console.print(f"{k}: {v:.4f}")
    return 0

def cmd_benchmark_regress(args) -> int:
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    if not baseline_path.exists():
        console.print(f"[red]Baseline artifact not found: {baseline_path}[/red]")
        return 1
    if not candidate_path.exists():
        console.print(f"[red]Candidate artifact not found: {candidate_path}[/red]")
        return 1
    baseline_data = load_benchmark_artifact(baseline_path)
    candidate_data = load_benchmark_artifact(candidate_path)
    report: RegressionReport = compare_benchmarks(
        baseline_results=baseline_data["results"],
        candidate_results=candidate_data["results"],
        queries=baseline_data["queries"],
        k=baseline_data.get("k", 10),
        thresholds=RegressionThresholds(),
    )
    if report.failed:
        console.print("[red]Regression detected:[/red]")
        for r in report.reasons:
            console.print(f"- {r}")
    else:
        console.print("[green]No regressions detected[/green]")
    # Exit code
    exit_code = 0
    relevance_failed = any("precision" in r or "recall" in r or "mrr" in r or "nDCG" in r for r in report.reasons)
    latency_failed = any("Latency" in r for r in report.reasons)
    if relevance_failed and latency_failed:
        exit_code = 4
    elif relevance_failed:
        exit_code = 2
    elif latency_failed:
        exit_code = 3
    return exit_code

# ---------------- Main ---------------- #
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "search":
        sys.exit(cmd_search(args))
    if args.command == "benchmark":
        sys.exit(cmd_benchmark(args))
    if args.command == "benchmark-regress":
        sys.exit(cmd_benchmark_regress(args))

if __name__ == "__main__":
    main()
