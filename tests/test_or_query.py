from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking


def test_or_query_returns_either():
    records = [
        {"id": "1", "text": "apple banana"},
        {"id": "2", "text": "orange pear"},
    ]

    engine = SearchEngine.from_records(
        records,
        ranking=RobustRanking(),
    )

    results = engine.search("apple OR orange")
    assert len(results) == 2
