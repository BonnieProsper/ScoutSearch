# tests/test_search.py

import json
from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.ranking.fusion import FusionRanking
from scout.state.signals import IndexState
from scout.state.persistence import AutoSaver


def test_search_returns_results():
    records = [
        {"id": 1, "text": "the quick brown fox"},
        {"id": 2, "text": "jumps over the lazy dog"},
        {"id": 3, "text": "quick fox jumps high"},
    ]

    engine = SearchEngine.from_records(
        records=records,
        ranking=RobustRanking(),
    )

    results = engine.search("quick fox")

    assert results
    for doc_id, result in results:
        assert result.score > 0
        assert isinstance(result.components, dict)


def test_candidate_documents_subset():
    records = [
        {"id": 1, "text": "quick fox"},
        {"id": 2, "text": "lazy dog"},
    ]

    engine = SearchEngine.from_records(
        records=records,
        ranking=RobustRanking(),
    )

    candidates = set(engine._candidate_documents(["quick"]))
    assert candidates.issubset({1, 2})
    assert 1 in candidates


def test_index_state_signal_emitted(tmp_path):
    # Setup state + autosave
    state = IndexState()
    triggered = []

    def callback(doc_id: int):
        triggered.append(doc_id)

    state.on_change.subscribe(callback)

    # Autosave ensures that saving happens on change
    class DummyStore:
        saved = []

        @staticmethod
        def save(index):
            DummyStore.saved.append(list(index.doc_freqs.keys()))

    from scout.state import persistence
    original_store = persistence.Store
    persistence.Store = DummyStore  # monkey patch

    AutoSaver(state)
    state.add_document(42, ["hello", "world"], metadata={"text": "hello world"})

    # Signal triggered
    assert triggered == [42]

    # Autosave triggered
    assert DummyStore.saved
    assert "hello" in DummyStore.saved[0]
    assert "world" in DummyStore.saved[0]

    # Restore original Store
    persistence.Store = original_store


def test_fusion_ranking_combines_scores():
    records = [
        {"id": 1, "text": "quick fox jumps"},
        {"id": 2, "text": "lazy dog sleeps"},
    ]

    tf_ranking = RobustRanking(tf_weight=1.0, tfidf_weight=0.0)
    tfidf_ranking = RobustRanking(tf_weight=0.0, tfidf_weight=1.0)

    fusion = FusionRanking(
        strategies=[tf_ranking, tfidf_ranking],
        weights=[0.6, 0.4]
    )

    engine = SearchEngine.from_records(
        records=records,
        ranking=fusion
    )

    results = engine.search("quick jumps")
    assert results
    doc_ids = [doc_id for doc_id, _ in results]
    assert 1 in doc_ids
