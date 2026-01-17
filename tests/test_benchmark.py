# tests/test_benchmark.py

import unittest
from scout.benchmark import benchmark_engine
from scout.ranking.robust import RobustRanking


class TestBenchmark(unittest.TestCase):
    def test_benchmark_runs(self):
        records = [{"id": i, "text": f"Document {i} sample text"} for i in range(1, 21)]
        queries = ["sample", "document", "text"]

        avg_time = benchmark_engine(records, queries, RobustRanking())

        # Ensure numeric and >= 0 safely
        self.assertIsNotNone(avg_time)
        self.assertIsInstance(avg_time, (int, float))

        # Now safe for type checker
        if avg_time is not None:
            self.assertGreaterEqual(avg_time, 0.0)


if __name__ == "__main__":
    unittest.main()
