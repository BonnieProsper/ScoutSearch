# scout/benchmarks/utils.py

import json
import csv
from pathlib import Path
from typing import List
from scout.benchmarks.run import BenchmarkResult, BenchmarkQuery
from scout.benchmarks.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank

def export_benchmark_results(results: List[BenchmarkResult], output_path: Path) -> None:
    """Export benchmark results to JSON and CSV."""
    # JSON
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    # CSV
    with open(output_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "retrieved", "latency_ms"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "query": r.query,
                "retrieved": ";".join(r.retrieved),
                "latency_ms": f"{r.latency_ms:.2f}",
            })


def aggregate_metrics(results: List[BenchmarkResult], queries: List[BenchmarkQuery], k: int):
    """Aggregate metrics across all queries."""
    precisions = [
        precision_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=k)
        for r, q in zip(results, queries)
    ]
    recalls = [
        recall_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=k)
        for r, q in zip(results, queries)
    ]
    mrrs = [
        mean_reciprocal_rank(retrieved=r.retrieved, relevant=q.relevant_doc_ids)
        for r, q in zip(results, queries)
    ]
    return {
        "P@k": sum(precisions)/len(precisions),
        "R@k": sum(recalls)/len(recalls),
        "MRR": sum(mrrs)/len(mrrs),
        "queries": len(queries),
    }
