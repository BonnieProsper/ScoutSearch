from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Mapping

from scout.benchmarks.aggregate import aggregate_metrics
from scout.benchmarks.run import BenchmarkResult
from scout.benchmarks.thresholds import RegressionThresholds


@dataclass(frozen=True)
class RegressionReport:
    deltas: dict[str, float]
    failed: bool
    reasons: list[str]


_DISPLAY = {
    "precision@k_mean": "Precision@K",
    "recall@k_mean": "Recall@K",
    "mrr_mean": "MRR",
    "ndcg_mean": "nDCG",
    "ndcg@k": "nDCG",
}


def _extract_metrics(
    results: Iterable[BenchmarkResult] | Mapping[str, object],
    *,
    queries,
    k: int,
) -> dict[str, float]:
    if isinstance(results, Mapping):
        metrics_obj = results["metrics"]
        assert isinstance(metrics_obj, Mapping)

        normalized: dict[str, float] = {}

        for key, value in metrics_obj.items():
            if key.startswith("ndcg@"):
                normalized["ndcg@k"] = float(value)
            else:
                normalized[str(key)] = float(value)

        return normalized

    return aggregate_metrics(
        results=results,
        queries=queries,
        k=k,
    )


def compare_benchmarks(
    *,
    baseline_results,
    candidate_results,
    queries,
    k: int,
    thresholds: RegressionThresholds,
) -> RegressionReport:
    base_metrics = _extract_metrics(
        baseline_results,
        queries=queries,
        k=k,
    )
    cand_metrics = _extract_metrics(
        candidate_results,
        queries=queries,
        k=k,
    )

    deltas: dict[str, float] = {}
    reasons: list[str] = []
    failed = False

    checks = [
        ("precision@k_mean", thresholds.min_precision_delta),
        ("recall@k_mean", thresholds.min_recall_delta),
        ("mrr_mean", thresholds.min_mrr_delta),
        ("ndcg_mean", thresholds.min_ndcg_delta),
        ("ndcg@k", thresholds.min_ndcg_delta),
    ]

    for metric_name, threshold in checks:
        if metric_name not in base_metrics:
            continue

        delta = cand_metrics.get(metric_name, 0.0) - base_metrics[metric_name]
        deltas[metric_name] = delta

        if delta < threshold:
            failed = True
            display = _DISPLAY.get(metric_name, metric_name)
            reasons.append(f"{display} regressed by {delta:.4f}")

    return RegressionReport(
        deltas=deltas,
        failed=failed,
        reasons=reasons,
    )
