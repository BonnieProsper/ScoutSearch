from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Protocol

from scout.benchmarks.index import BenchmarkIndex, BenchmarkRecord


class SearchEngine(Protocol):
    """
    Minimal interface required for benchmarking.
    """

    def index(self, records: Iterable[BenchmarkRecord]) -> None:
        ...

    def search(self, query: str, *, k: int) -> List[str]:
        """
        Returns ordered list of doc_ids.
        """
        ...


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
    Run a deterministic benchmark.
    """
    engine.index(index.records)

    results: List[BenchmarkResult] = []

    for q in queries:
        start = perf_counter()
        retrieved = engine.search(q.query, k=k)
        elapsed_ms = (perf_counter() - start) * 1000

        results.append(
            BenchmarkResult(
                query=q.query,
                retrieved=retrieved,
                latency_ms=elapsed_ms,
            )
        )

    return results
