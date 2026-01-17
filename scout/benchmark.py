# scout/benchmark.py

import time
import json
from typing import List, Dict
from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking

def benchmark_engine(records: List[Dict], queries: List[str], ranking_strategy):
    print(f"\nBenchmarking with {ranking_strategy.__class__.__name__}...\n")
    
    engine = SearchEngine.from_records(records, ranking=ranking_strategy)

    times = []
    for query in queries:
        start = time.time()
        results = engine.search(query)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Query: '{query}' - {len(results)} results in {elapsed:.6f}s")

    avg_time = sum(times) / len(times) if times else 0
    print(f"\nAverage query time: {avg_time:.6f}s\n")
    return avg_time

if __name__ == "__main__":
    # Example usage with synthetic dataset
    records = [{"id": i, "text": f"Document {i} with sample text"} for i in range(1, 101)]
    queries = ["sample", "document", "text"]

    benchmark_engine(records, queries, RobustRanking())
    benchmark_engine(records, queries, BM25Ranking())
