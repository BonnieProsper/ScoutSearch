import pytest

from scout.explain import explain_query
from scout.ranking.bm25 import BM25Ranking
from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine


@pytest.fixture
def engine(sample_records):
    return SearchEngine.from_records(sample_records, ranking=RobustRanking())


def test_explainability_contains_all_tokens():
    records = [
        {"id": "1", "text": "apple banana"},
        {"id": "2", "text": "apple orange"},
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


def test_explain_query_does_not_mutate_results(engine):
    original = engine.search("apple", limit=3)
    explained = explain_query(engine, "apple", limit=3)

    for (_, o), (_, e) in zip(original, explained):
        assert o is not e
        assert o.score == e.score
        assert o.components == e.components
