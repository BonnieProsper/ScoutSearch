# scout/benchmark.py

import time
from typing import List, Dict
import matplotlib.pyplot as plt
from scout.search.engine import SearchEngine
from scout.ranking.base import RankingStrategy


def benchmark_engine(
    records: List[Dict],
    queries: List[str],
    ranking_strategy: RankingStrategy,
    limit: int = 10,
    plot: bool = True
) -> float:
    print(f"\nBenchmarking {ranking_strategy.__class__.__name__} with {len(records)} documents...\n")

    engine = SearchEngine.from_records(records, ranking=ranking_strategy)
    times: List[float] = []

    for query in queries:
        start = time.perf_counter()
        results = engine.search(query, limit=limit)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Query: '{query}' - {len(results)} results in {elapsed:.6f}s")

    avg_time = sum(times) / len(times) if times else 0.0
    print(f"\nAverage query time: {avg_time:.6f}s\n")

    if plot:
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(queries)), times, tick_label=queries)
        plt.ylabel("Query Time (s)")
        plt.title(f"Benchmark: {ranking_strategy.__class__.__name__}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return avg_time
