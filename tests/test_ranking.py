# tests/test_ranking.py
import unittest
from collections import defaultdict
from scout.ranking.tf import TermFrequencyRanking
from scout.ranking.tfidf import TFIDFRanking
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.index.builder import IndexBuilder

class TestRanking(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": 1, "text": "the quick brown fox"},
            {"id": 2, "text": "jumps over the lazy dog"},
            {"id": 3, "text": "quick fox jumps high"}
        ]
        self.builder = IndexBuilder(fields=["text"], ngram=1)
        self.index = self.builder.build(self.records)
        self.query_tokens = ["quick", "fox"]

    def test_tf_ranking(self):
        tf = TermFrequencyRanking()
        result = tf.score(self.query_tokens, self.index, doc_id=1)
        self.assertTrue(result.score > 0)
        self.assertTrue(all(token in result.components for token in self.query_tokens))

    def test_tfidf_ranking(self):
        tfidf = TFIDFRanking()
        result = tfidf.score(self.query_tokens, self.index, doc_id=1)
        self.assertTrue(result.score > 0)
        self.assertTrue(all(token in result.components for token in self.query_tokens))

    def test_robust_ranking(self):
        robust = RobustRanking(tf_weight=0.5, tfidf_weight=0.5)
        result = robust.score(self.query_tokens, self.index, doc_id=1)
        self.assertTrue(result.score > 0)
        self.assertTrue(len(result.components) > 0)

    def test_bm25_ranking(self):
        bm25 = BM25Ranking()
        result = bm25.score(self.query_tokens, self.index, doc_id=1)
        self.assertTrue(result.score > 0)
        self.assertTrue(all(token in result.components for token in self.query_tokens))

if __name__ == "__main__":
    unittest.main()
