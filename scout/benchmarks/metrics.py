# scout/benchmarks/metrics.py

from __future__ import annotations

from typing import Iterable, List


def precision_at_k(
    *,
    retrieved: List[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    relevant_set = set(relevant)
    if k == 0:
        return 0.0

    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return hits / k


def recall_at_k(
    *,
    retrieved: List[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(
    *,
    retrieved: List[str],
    relevant: Iterable[str],
) -> float:
    relevant_set = set(relevant)

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0
