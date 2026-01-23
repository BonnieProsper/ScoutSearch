# scout/benchmarks/run.py
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Optional, Dict
import random

from rich.progress import track

from scout.benchmarks.index import BenchmarkIndex
from scout.search.engine import SearchEngine
from scout.benchmarks.metrics import latency_percentiles


@dataclass(frozen=True)
class BenchmarkQuery:
    query: str
    relevant_doc_ids: frozenset[str]


@dataclass(frozen=True)
class BenchmarkResult:
    query: str
    retrieved: List[str]
    latency_ms: float
    latency_stats: Optional[Dict[int, float]] = None  # e.g., 50th, 95th percentiles


def run_benchmark(
    *,
    engine: SearchEngine,
    index: BenchmarkIndex,
    queries: Iterable[BenchmarkQuery],
    k: int,
    warmup: int = 0,
    repeats: int = 1,
    seed: Optional[int] = None,
) -> List[BenchmarkResult]:
    """
    Run a benchmark with warmup and repeated measurements.

    Returns:
        List[BenchmarkResult] with average latency and percentile stats
    """
    if seed is not None:
        random.seed(seed)

    results: List[BenchmarkResult] = []

    for q in track(queries, description="[bold green]Running benchmark..."):
        # Warmup runs
        for _ in range(warmup):
            engine.search(q.query, limit=k)

        latencies: List[float] = []
        retrieved_ids: List[str] = []

        for _ in range(repeats):
            start = perf_counter()
            hits = engine.search(q.query, limit=k)
            elapsed = (perf_counter() - start) * 1000.0
            latencies.append(elapsed)
            retrieved_ids = [str(doc_id) for doc_id, _ in hits]  # last run determines ordering

        results.append(
            BenchmarkResult(
                query=q.query,
                retrieved=retrieved_ids,
                latency_ms=sum(latencies) / len(latencies),
                latency_stats=latency_percentiles(latencies),
            )
        )

    return results
