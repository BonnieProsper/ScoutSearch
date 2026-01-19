from scout.benchmark import benchmark_engine
from scout.ranking.robust import RobustRanking


def test_benchmark_runs():
    records = [{"id": i, "text": f"Document {i} sample text"} for i in range(1, 21)]
    queries = ["sample", "document", "text"]

    avg_time = benchmark_engine(records, queries, RobustRanking())

    assert avg_time is not None
    assert isinstance(avg_time, (int, float))
    assert avg_time >= 0.0
