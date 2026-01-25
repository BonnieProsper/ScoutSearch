from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine


def test_index_builder_basic():
    records = [{"id": 1, "text": "hello world"}]
    engine = SearchEngine.from_records(
        records,
        ranking=RobustRanking(),
    )
    results = engine.search("hello")
    assert results
