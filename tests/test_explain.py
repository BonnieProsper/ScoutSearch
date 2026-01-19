from scout.search.engine import SearchEngine
from scout.ranking.bm25 import BM25Ranking
from scout.explain import explain_query


def test_explainability_contains_all_tokens():
    records = [
        {"text": "apple banana"},
        {"text": "apple orange"},
    ]

    engine = SearchEngine.from_records(
        records,
        ranking=BM25Ranking(),
    )

    results = explain_query(engine, "apple banana", limit=2)

    assert results
    for _, result in results:
        assert result.score > 0
        assert "apple" in result.per_term
        assert "banana" in result.per_term
