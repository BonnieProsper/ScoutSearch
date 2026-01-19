from scout.ranking.tf import TermFrequencyRanking
from scout.ranking.tfidf import TFIDFRanking
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.index.builder import IndexBuilder


def _build_index():
    records = [
        {"id": 1, "text": "the quick brown fox"},
        {"id": 2, "text": "jumps over the lazy dog"},
        {"id": 3, "text": "quick fox jumps high"},
    ]
    builder = IndexBuilder(fields=["text"], ngram=1)
    return builder.build(records)


def test_tf_ranking():
    index = _build_index()
    tf = TermFrequencyRanking()
    result = tf.score(["quick", "fox"], index, doc_id=1)

    assert result.score > 0
    assert all(token in result.components for token in ["quick", "fox"])


def test_tfidf_ranking():
    index = _build_index()
    tfidf = TFIDFRanking()
    result = tfidf.score(["quick", "fox"], index, doc_id=1)

    assert result.score > 0
    assert all(token in result.components for token in ["quick", "fox"])


def test_robust_ranking():
    index = _build_index()
    robust = RobustRanking(tf_weight=0.5, tfidf_weight=0.5)
    result = robust.score(["quick", "fox"], index, doc_id=1)

    assert result.score > 0
    assert result.components


def test_bm25_ranking():
    index = _build_index()
    bm25 = BM25Ranking()
    result = bm25.score(["quick", "fox"], index, doc_id=1)

    assert result.score > 0
    assert all(token in result.components for token in ["quick", "fox"])
