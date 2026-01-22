# scout/benchmarks/run.py

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List

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
) -> List[BenchmarkResult]:
    """
    Run a deterministic benchmark against an already-built SearchEngine.

    Assumptions:
    - engine has already indexed `index.records`
    - ordering of results is meaningful
    """
    results: List[BenchmarkResult] = []

    for q in queries:
        start = perf_counter()
        hits = engine.search(q.query, limit=k)
        latency_ms = (perf_counter() - start) * 1000.0

        retrieved_ids = [str(doc_id) for doc_id, _ in hits]

        results.append(
            BenchmarkResult(
                query=q.query,
                retrieved=retrieved_ids,
                latency_ms=latency_ms,
            )
        )

    return results
