# scout/benchmarks/regression.py
import json
from pathlib import Path
from scout.benchmarks.utils import aggregate_metrics
from scout.benchmarks.run import BenchmarkResult, BenchmarkQuery

def detect_regression(
    new_results: list[BenchmarkResult],
    queries: list[BenchmarkQuery],
    previous_results_file: Path,
    k: int,
) -> None:
    """
    Compares new benchmark results to previous run and prints warnings if metrics drop.
    """
    if not previous_results_file.exists():
        print(f"No previous benchmark file found at {previous_results_file}. Skipping regression detection.")
        return

    prev_data = json.loads(previous_results_file.read_text())
    prev_results = [
        BenchmarkResult(
            query=d["query"],
            retrieved=d["retrieved"],
            latency_ms=d.get("latency_ms", 0.0)
        ) for d in prev_data
    ]
    prev_metrics = aggregate_metrics(prev_results, queries, k)
    new_metrics = aggregate_metrics(new_results, queries, k)

    print("\n--- Regression Detection ---")
    for metric in ["P@k", "R@k", "MRR"]:
        prev_val = prev_metrics[metric]
        new_val = new_metrics[metric]
        if new_val < prev_val:
            print(f"WARNING: {metric} dropped from {prev_val:.4f} → {new_val:.4f}")
        else:
            print(f"{metric}: {prev_val:.4f} → {new_val:.4f} (ok)")
