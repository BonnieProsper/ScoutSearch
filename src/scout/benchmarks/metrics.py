# scout/benchmarks/metrics.py
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def precision_at_k(*, retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return float(hits / k)


def recall_at_k(*, retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return float(hits / len(relevant_set))


def mean_reciprocal_rank(*, retrieved: list[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return float(1.0 / rank)
    return 0.0


def average_precision(*, retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    score = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            hits += 1
            score += hits / i
    return float(score / min(len(relevant_set), k))


def ndcg_at_k(*, retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1 / math.log2(i + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def f1_at_k(*, retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    p = precision_at_k(retrieved=retrieved, relevant=relevant, k=k)
    r = recall_at_k(retrieved=retrieved, relevant=relevant, k=k)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def latency_percentiles(latencies_ms: list[float], percentiles: list[int] | None = None) -> dict[int, float]:
    arr = np.array(latencies_ms, dtype=float)
    if percentiles is None:
        percentiles = [50, 95, 99]
    return {p: float(np.percentile(arr, p)) for p in percentiles}
