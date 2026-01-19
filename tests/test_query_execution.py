from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking


def test_phrase_query_filters():
    records = [
        {"text": "the quick brown fox"},
        {"text": "quick fox brown"},
    ]

    engine = SearchEngine.from_records(
        records,
        ranking=RobustRanking(),
    )

    results = engine.search('"quick brown fox"')
    assert len(results) == 1
