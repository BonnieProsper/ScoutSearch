# scout/benchmarks/regression.py
import json
from pathlib import Path
from typing import Optional
from scout.benchmarks.utils import aggregate_metrics
from scout.benchmarks.run import BenchmarkResult, BenchmarkQuery


def detect_regression(
    new_results: list[BenchmarkResult],
    queries: list[BenchmarkQuery],
    previous_results_file: Path,
    k: int,
    threshold_drop: Optional[float] = 0.01,  # Warn if metric drops more than 1%
) -> None:
    """
    Detect regressions between new benchmark and previous benchmark.
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
    for metric in ["P@k", "R@k", "MRR", "F1@k", "nDCG@k", "MAP@k"]:
        prev_val = prev_metrics.get(metric, 0.0)
        new_val = new_metrics.get(metric, 0.0)
        drop = prev_val - new_val
        if drop > threshold_drop:
            print(f"WARNING: {metric} dropped by {drop:.4f} ({prev_val:.4f} → {new_val:.4f})")
        else:
            print(f"{metric}: {prev_val:.4f} → {new_val:.4f} (ok)")

    # Optional: regression on latency
    prev_latency = prev_metrics["latency_ms"]["p95"]
    new_latency = new_metrics["latency_ms"]["p95"]
    if new_latency > prev_latency * 1.05:  # 5% slower
        print(f"WARNING: 95th percentile latency increased from {prev_latency:.2f}ms → {new_latency:.2f}ms")
