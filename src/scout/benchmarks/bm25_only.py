# scout/benchmarks/bm25_only.py

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from scout.benchmarks.index import BenchmarkIndex
from scout.benchmarks.run import BenchmarkQuery, BenchmarkResult
from scout.ranking.bm25 import BM25Ranking
from scout.search.engine import SearchEngine


def bm25_only_baseline(
    *,
    index: BenchmarkIndex,
    queries: Iterable[BenchmarkQuery],
    k: int,
) -> list[BenchmarkResult]:
    """
    BM25-only baseline using the standard SearchEngine pipeline.

    This ensures:
    - identical tokenization
    - identical indexing
    - fair comparison vs composite rankings
    """

    # Build a fresh engine from benchmark records
    records = [
        {
            "id": record.doc_id,
            "text": record.content,
            **record.metadata,
        }
        for record in index.records
    ]

    engine = SearchEngine.from_records(
        records,
        ranking=BM25Ranking(),
    )

    results: list[BenchmarkResult] = []

    for q in queries:
        start = perf_counter()
        hits = engine.search(q.query, limit=k)
        latency_ms = (perf_counter() - start) * 1000.0

        results.append(
            BenchmarkResult(
                query=q.query,
                retrieved=[str(doc_id) for doc_id, _ in hits],
                latency_ms=latency_ms,
            )
        )

    return results
