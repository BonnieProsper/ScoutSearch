from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking

def test_or_query_returns_either():
    records = [
        {"text": "apple banana"},
        {"text": "orange pear"},
    ]

    engine = SearchEngine.from_records(
        records,
        ranking=RobustRanking(),
    )

    results = engine.search("apple OR orange")
    assert len(results) == 2
