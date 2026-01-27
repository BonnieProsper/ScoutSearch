from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine


def test_index_snapshot_roundtrip(tmp_path):
    records = [{"id": 1, "text": "fox"}]
    engine = SearchEngine.from_records(records, ranking=RobustRanking())

    path = tmp_path / "index.json"
    engine.save(path)

    engine2 = SearchEngine.load(path, ranking=RobustRanking())
    assert engine2.search("fox") == engine.search("fox")
