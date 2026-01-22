from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Optional

from scout.benchmarks.index import BenchmarkIndex
from scout.search.engine import SearchEngine


@dataclass(frozen=True)
class BenchmarkQuery:
    query: str
    relevant_doc_ids: frozenset[str]


@dataclass(frozen=True)
class BenchmarkResult:
    query: str
    retrieved: List[str]
    latency_ms: float


def run_benchmark(
    *,
    engine: SearchEngine,
    index: BenchmarkIndex,
    queries: Iterable[BenchmarkQuery],
    k: int,
    warmup: int = 0,
    repeats: int = 1,
) -> List[BenchmarkResult]:
    """
    Run a deterministic benchmark against an already-built SearchEngine.

    Assumptions:
    - engine has already indexed `index.records`
    - ordering of results is meaningful
    - caller controls randomness (if any) outside this function
    """
    results: List[BenchmarkResult] = []

    for q in queries:
        # warmup runs (not measured)
        for _ in range(warmup):
            engine.search(q.query, limit=k)

        latencies: List[float] = []
        retrieved_ids: List[str] = []

        for _ in range(repeats):
            start = perf_counter()
            hits = engine.search(q.query, limit=k)
            latencies.append((perf_counter() - start) * 1000.0)

            # last run's ordering is what we evaluate
            retrieved_ids = [str(doc_id) for doc_id, _ in hits]

        results.append(
            BenchmarkResult(
                query=q.query,
                retrieved=retrieved_ids,
                latency_ms=sum(latencies) / len(latencies),
            )
        )

    return results
