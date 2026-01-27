from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine


def test_incremental_document_becomes_searchable():
    records = [
        {"id": 1, "text": "hello world"},
    ]

    engine = SearchEngine.from_records(
        records=records,
        ranking=RobustRanking(),
    )

    # Add new document AFTER engine creation
    engine.add_document(
        doc_id=2,
        record={"text": "hello scout"},
    )

    results = engine.search("scout")

    assert results
    assert any(doc_id == 2 for doc_id, _ in results)
