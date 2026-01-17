# scout/benchmark.py

import time
from typing import List, Dict
from scout.search.engine import SearchEngine
from scout.ranking.base import RankingStrategy

def benchmark_engine(
    records: List[Dict],
    queries: List[str],
    ranking_strategy: RankingStrategy,
    limit: int = 10
) -> float:
    """
    Benchmark a SearchEngine with given records, queries, and ranking strategy.

    Prints per-query time and average time.
    """
    print(f"\nBenchmarking {ranking_strategy.__class__.__name__} with {len(records)} documents...\n")

    engine = SearchEngine.from_records(records, ranking=ranking_strategy)

    times: List[float] = []

    for query in queries:
        start = time.time()
        results = engine.search(query, limit=limit)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Query: '{query}' - {len(results)} results in {elapsed:.6f}s")

    avg_time = sum(times) / len(times) if times else 0.0
    print(f"\nAverage query time: {avg_time:.6f}s\n")
    return avg_time
