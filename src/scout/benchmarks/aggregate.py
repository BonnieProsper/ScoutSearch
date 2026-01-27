# scout/benchmarks/aggregate.py

from __future__ import annotations

import math
from collections.abc import Iterable

from scout.benchmarks.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from scout.benchmarks.run import BenchmarkResult


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def stddev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def aggregate_metrics(
    *,
    results: Iterable[BenchmarkResult],
    queries,
    k: int,
) -> dict[str, float]:
    p_vals = []
    r_vals = []
    mrr_vals = []
    latency_vals = []

    for q, r in zip(queries, results):
        p_vals.append(
            precision_at_k(
                retrieved=r.retrieved,
                relevant=q.relevant_doc_ids,
                k=k,
            )
        )
        r_vals.append(
            recall_at_k(
                retrieved=r.retrieved,
                relevant=q.relevant_doc_ids,
                k=k,
            )
        )
        mrr_vals.append(
            mean_reciprocal_rank(
                retrieved=r.retrieved,
                relevant=q.relevant_doc_ids,
            )
        )
        latency_vals.append(r.latency_ms)

    return {
        "precision@k_mean": mean(p_vals),
        "precision@k_std": stddev(p_vals),
        "recall@k_mean": mean(r_vals),
        "recall@k_std": stddev(r_vals),
        "mrr_mean": mean(mrr_vals),
        "mrr_std": stddev(mrr_vals),
        "latency_ms_mean": mean(latency_vals),
        "latency_ms_std": stddev(latency_vals),
    }
