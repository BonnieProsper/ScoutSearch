from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine


def test_phrase_query_filters():
    records = [
        {"id": "1", "text": "the quick brown fox"},
        {"id": "2", "text": "quick fox brown"},
    ]

    engine = SearchEngine.from_records(
        records,
        ranking=RobustRanking(),
    )

    results = engine.search('"quick brown fox"')
    assert len(results) == 1
    assert results[0][0] == "1"
