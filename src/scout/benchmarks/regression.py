from __future__ import annotations

from dataclasses import dataclass

from scout.benchmarks.aggregate import aggregate_metrics
from scout.benchmarks.run import BenchmarkQuery, BenchmarkResult
from scout.benchmarks.thresholds import RegressionThresholds


@dataclass(frozen=True)
class RegressionReport:
    deltas: dict[str, float]
    failed: bool
    reasons: list[str]


def compare_benchmarks(
    *,
    baseline_results: list[BenchmarkResult],
    candidate_results: list[BenchmarkResult],
    queries: list[BenchmarkQuery],
    k: int,
    thresholds: RegressionThresholds,
) -> RegressionReport:
    base_metrics = aggregate_metrics(
        results=baseline_results, queries=queries, k=k
    )
    cand_metrics = aggregate_metrics(
        results=candidate_results, queries=queries, k=k
    )

    deltas: dict[str, float] = {}
    reasons: list[str] = []
    failed = False

    for metric_name, threshold in [
        ("precision@k_mean", thresholds.min_precision_delta),
        ("recall@k_mean", thresholds.min_recall_delta),
        ("mrr_mean", thresholds.min_mrr_delta),
        ("ndcg_mean", thresholds.min_ndcg_delta),
    ]:
        if metric_name not in base_metrics:
            continue

        delta = cand_metrics[metric_name] - base_metrics[metric_name]
        deltas[metric_name] = delta

        if delta < threshold:
            failed = True
            reasons.append(
                f"{metric_name} regressed by {delta:.4f}"
            )

    base_latency = base_metrics.get("latency_ms_mean", 0.0)
    cand_latency = cand_metrics.get("latency_ms_mean", 0.0)
    delta_latency = cand_latency - base_latency

    deltas["latency_ms_mean"] = delta_latency

    if delta_latency > thresholds.max_latency_regression_ms:
        failed = True
        reasons.append(
            f"Latency regressed by {delta_latency:.2f}ms "
            f"({base_latency:.2f} → {cand_latency:.2f})"
        )

    return RegressionReport(
        deltas=deltas,
        failed=failed,
        reasons=reasons,
    )
