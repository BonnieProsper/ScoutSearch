# scout/benchmarks/baselines.py

from __future__ import annotations

import random
from typing import Iterable, List

from scout.benchmarks.run import BenchmarkQuery, BenchmarkResult
from scout.benchmarks.index import BenchmarkIndex


def random_baseline(
    *,
    index: BenchmarkIndex,
    queries: Iterable[BenchmarkQuery],
    k: int,
    seed: int = 42,
) -> List[BenchmarkResult]:
    """
    Random retrieval baseline.

    Provides a deterministic lower bound for evaluation.
    """
    rng = random.Random(seed)

    doc_ids = [record.doc_id for record in index.records]

    results: List[BenchmarkResult] = []

    for q in queries:
        shuffled = doc_ids[:]
        rng.shuffle(shuffled)

        results.append(
            BenchmarkResult(
                query=q.query,
                retrieved=shuffled[:k],
                latency_ms=0.0,
            )
        )

    return results
