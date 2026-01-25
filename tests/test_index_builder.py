from scout.search.engine import SearchEngine


def test_ignores_malformed_records():
    records = [
        {"id": "1", "text": "valid"},
        {"id": "2"},                 # missing text
        {"text": "no id"},           # missing id
        {"id": "3", "text": 123},    # invalid text
    ]

    engine = SearchEngine.from_records(records)
    results = engine.search("valid")

    assert len(results) == 1
    assert results[0][0] == "1"