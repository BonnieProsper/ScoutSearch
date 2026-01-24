# scout/benchmarks/regression.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from scout.benchmarks.aggregate import aggregate_metrics
from scout.benchmarks.thresholds import RegressionThresholds
from scout.benchmarks.run import BenchmarkResult, BenchmarkQuery

@dataclass(frozen=True)
class RegressionReport:
    deltas: Dict[str, float]
    failed: bool
    reasons: List[str]

def compare_benchmarks(
    *,
    baseline_results: List[BenchmarkResult],
    candidate_results: List[BenchmarkResult],
    queries: List[BenchmarkQuery],
    k: int,
    thresholds: RegressionThresholds,
) -> RegressionReport:

    base_metrics = aggregate_metrics(results=baseline_results, queries=queries, k=k)
    cand_metrics = aggregate_metrics(results=candidate_results, queries=queries, k=k)

    deltas: Dict[str, float] = {}
    reasons: List[str] = []
    failed = False

    # Relevance metrics
    for metric_name, threshold in [
        ("precision@k_mean", thresholds.min_precision_delta),
        ("recall@k_mean", thresholds.min_recall_delta),
        ("mrr_mean", thresholds.min_mrr_delta),
        ("ndcg_mean", thresholds.min_ndcg_delta)  # must exist in aggregate_metrics
    ]:
        if metric_name not in base_metrics:
            continue  # skip if metric not calculated
        delta = cand_metrics[metric_name] - base_metrics[metric_name]
        deltas[metric_name] = delta
        if delta < threshold:
            failed = True
            reasons.append(f"{metric_name} regressed by {delta:.4f}")

    # Latency
    base_latency = base_metrics.get("latency_ms_mean", 0.0)
    cand_latency = cand_metrics.get("latency_ms_mean", 0.0)
    delta_latency = cand_latency - base_latency
    deltas["latency_ms_mean"] = delta_latency
    if delta_latency > thresholds.max_latency_regression_ms:
        failed = True
        reasons.append(f"Latency regressed by {delta_latency:.2f}ms ({base_latency:.2f} → {cand_latency:.2f})")

    return RegressionReport(deltas=deltas, failed=failed, reasons=reasons)
