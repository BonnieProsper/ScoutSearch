from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine


def test_deterministic_ranking():
    records = [
        {"id": 1, "text": "fox"},
        {"id": 2, "text": "fox"},
    ]
    engine = SearchEngine.from_records(records, ranking=RobustRanking())

    r1 = engine.search("fox")
    r2 = engine.search("fox")

    assert r1 == r2
