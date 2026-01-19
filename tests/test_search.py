from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
from scout.state.signals import IndexState


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


def test_index_state_signal_emitted():
    state = IndexState()
    triggered = []

    def callback(doc_id: int):
        triggered.append(doc_id)

    state.on_change.subscribe(callback)
    state.add_document(42, ["hello", "world"])

    assert triggered == [42]
