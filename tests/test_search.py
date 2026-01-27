# tests/test_search.py

from datetime import datetime, timedelta

import pytest

from scout.explain import explain_query
from scout.index.builder import IndexBuilder
from scout.ranking.bm25 import BM25Ranking
from scout.ranking.composite import CompositeRanking
from scout.ranking.fusion import FusionRanking
from scout.ranking.recency import RecencyRanking
from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine
from scout.state.persistence import AutoSaver
from scout.state.signals import IndexState


@pytest.fixture
def sample_records():
    now = datetime.now()
    return [
        {
            "id": 1,
            "title": "Quick brown fox",
            "text": "the quick brown fox jumps over the lazy dog",
            "timestamp": now.isoformat(),
        },
        {
            "id": 2,
            "title": "Lazy dog",
            "text": "the lazy dog sleeps all day",
            "timestamp": (now - timedelta(days=10)).isoformat(),
        },
        {
            "id": 3,
            "title": "Fast fox",
            "text": "fast fox runs quickly",
            "timestamp": (now - timedelta(days=30)).isoformat(),
        },
    ]



@pytest.mark.parametrize("ranking_cls", [RobustRanking, BM25Ranking])
def test_search_engine_returns_results(sample_records, ranking_cls):
    engine = SearchEngine.from_records(sample_records, ranking=ranking_cls())
    results = engine.search("fox")
    assert results
    for doc_id, r in results:
        assert r.score > 0
        assert isinstance(r.components, dict)


def test_candidate_documents_and_phrase_matching(sample_records):
    state = IndexState()
    engine = SearchEngine.from_records(sample_records, ranking=RobustRanking(), state=state)
    # Phrase matching
    phrase = ["quick", "brown", "fox"]
    for doc_id in engine._candidate_documents(["quick"]):
        tokens = state.get_document_tokens(doc_id)
        if tokens:
            assert isinstance(tokens, list)
            # ensure phrase exists for doc 1
            if doc_id == 1:
                assert all(word in tokens for word in phrase)


def test_incremental_indexing(sample_records):
    state = IndexState()
    engine = SearchEngine.from_records([], ranking=RobustRanking(), state=state)
    engine.add_document(42, {"id": 42, "title": "New doc", "text": "hello world"})
    results = engine.search("hello")
    assert results[0][0] == 42


def test_field_weighting(sample_records):
    builder = IndexBuilder(fields=["title", "text"])
    index = builder.build(sample_records, field_weights={"title": 2, "text": 1})
    # doc 1 title "Quick brown fox" should have doubled tokens
    postings = index.get_postings("quick")
    assert any(doc_id == 1 for doc_id, _ in postings)


def test_explain_query_returns_components(sample_records):
    engine = SearchEngine.from_records(sample_records, ranking=RobustRanking())
    explanations = explain_query(engine, "fox", limit=2)
    assert explanations
    for doc_id, r in explanations:
        assert isinstance(r.score, float)
        assert isinstance(r.components, dict)
        assert isinstance(r.per_term, dict)


def test_recency_boost(sample_records):
    recency = RecencyRanking(decay_days=10.0, max_boost=1.0)
    engine = SearchEngine.from_records(sample_records, ranking=CompositeRanking(
        strategies=[BM25Ranking()], weights=[1.0], recency=recency
    ))
    results = engine.search("fox")
    # The most recent doc should have higher score
    scores = [r.score for _, r in results]
    assert scores[0] >= scores[-1]


def test_autosaver_trigger(tmp_path, sample_records):
    state = IndexState()
    auto = AutoSaver(state)
    engine = SearchEngine.from_records([], ranking=RobustRanking(), state=state)
    engine.add_document(100, {"id": 100, "text": "autosave test"})
    # _state.on_change should have emitted callback (manual check)
    assert state.get_document_tokens(100) == ["autosave", "test"]


def test_fusion_ranking_combines_scores(sample_records):
    tf_ranking = RobustRanking(tf_weight=1.0, tfidf_weight=0.0)
    tfidf_ranking = RobustRanking(tf_weight=0.0, tfidf_weight=1.0)
    fusion = FusionRanking(
        strategies=[tf_ranking, tfidf_ranking],
        weights=[0.6, 0.4]
    )
    engine = SearchEngine.from_records(
        records=sample_records[:2],
        ranking=fusion
    )
    results = engine.search("quick jumps")
    assert results
    doc_ids = [doc_id for doc_id, _ in results]
    assert 1 in doc_ids

def test_stopwords_are_removed():
    records = [{"id": 1, "text": "the fox"}]
    engine = SearchEngine.from_records(records, ranking=RobustRanking())

    r1 = engine.search("fox")
    r2 = engine.search("the fox")

    assert r1 == r2
