import time
import pytest
from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
import sys

@pytest.mark.performance
@pytest.mark.skipif(sys.platform == "win32", reason="Timing unstable on Windows")
def test_query_latency_under_budget():
    records = [{"id": i, "text": "token"} for i in range(1000)]
    engine = SearchEngine.from_records(records, ranking=RobustRanking())

    query = " ".join(["token"] * 1000)
    start = time.perf_counter()
    engine.search(query)
    duration = time.perf_counter() - start

    assert duration < 0.05


# TODO