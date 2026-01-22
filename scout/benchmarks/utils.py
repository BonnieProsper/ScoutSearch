# scout/benchmarks/utils.py
import json
import csv
from pathlib import Path
from typing import List
from scout.benchmarks.run import BenchmarkResult, BenchmarkQuery
from scout.benchmarks.metrics import (
    precision_at_k, recall_at_k, mean_reciprocal_rank,
    average_precision, ndcg_at_k, f1_at_k, latency_percentiles
)


def export_benchmark_results(results: List[BenchmarkResult], output_path: Path) -> None:
    """Export benchmark results to JSON, CSV, and HTML."""
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

    # HTML (simple table)
    html_path = output_path.with_suffix(".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><body><table border='1'><tr><th>Query</th><th>Retrieved</th><th>Latency(ms)</th></tr>")
        for r in results:
            f.write(f"<tr><td>{r.query}</td><td>{','.join(r.retrieved)}</td><td>{r.latency_ms:.2f}</td></tr>")
        f.write("</table></body></html>")


def aggregate_metrics(results: List[BenchmarkResult], queries: List[BenchmarkQuery], k: int):
    """Aggregate metrics across all queries."""
    precisions = [precision_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=k)
                  for r, q in zip(results, queries)]
    recalls = [recall_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=k)
               for r, q in zip(results, queries)]
    mrrs = [mean_reciprocal_rank(retrieved=r.retrieved, relevant=q.relevant_doc_ids)
            for r, q in zip(results, queries)]
    f1s = [f1_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=k)
           for r, q in zip(results, queries)]
    ndcgs = [ndcg_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=k)
             for r, q in zip(results, queries)]
    aps = [average_precision(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=k)
           for r, q in zip(results, queries)]
    latencies = [r.latency_ms for r in results]

    latency_pct = latency_percentiles(latencies)

    return {
        "P@k": sum(precisions)/len(precisions),
        "R@k": sum(recalls)/len(recalls),
        "MRR": sum(mrrs)/len(mrrs),
        "F1@k": sum(f1s)/len(f1s),
        "nDCG@k": sum(ndcgs)/len(ndcgs),
        "MAP@k": sum(aps)/len(aps),
        "queries": len(queries),
        "latency_ms": {
            "p50": latency_pct[50],
            "p95": latency_pct[95],
            "p99": latency_pct[99]
        }
    }
