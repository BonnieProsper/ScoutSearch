# tests/test_benchmark.py

import unittest
from scout.benchmark import benchmark_engine
from scout.ranking.robust import RobustRanking

class TestBenchmark(unittest.TestCase):
    def test_benchmark_runs(self):
        records = [{"id": i, "text": f"Document {i} sample text"} for i in range(1, 21)]
        queries = ["sample", "document", "text"]

        avg_time = benchmark_engine(records, queries, RobustRanking())

        # Ensure avg_time is numeric and >= 0
        self.assertIsNotNone(avg_time)

        # Type-safe check for Pylance
        self.assertTrue(isinstance(avg_time, (int, float)) and avg_time >= 0.0)

if __name__ == "__main__":
    unittest.main()
