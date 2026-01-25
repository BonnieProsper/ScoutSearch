import time
import pytest
from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking


@pytest.mark.performance
def test_query_latency_under_budget():
    records = [{"id": i, "text": "token"} for i in range(1000)]
    engine = SearchEngine.from_records(records, ranking=RobustRanking())

    query = " ".join(["token"] * 1000)
    start = time.perf_counter()
    engine.search(query)
    duration = time.perf_counter() - start

    assert duration < 0.05
