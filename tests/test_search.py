# tests/test_search.py
import unittest
from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
from scout.index.builder import IndexBuilder
from scout.index.tokens import Tokenizer
from scout.state.signals import IndexState

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": 1, "text": "the quick brown fox"},
            {"id": 2, "text": "jumps over the lazy dog"},
            {"id": 3, "text": "quick fox jumps high"}
        ]
        self.ranking = RobustRanking()
        self.engine = SearchEngine.from_records(records=self.records, ranking=self.ranking)

    def test_search_returns_results(self):
        results = self.engine.search("quick fox", limit=10)
        self.assertTrue(len(results) > 0)
        for doc_id, ranking_result in results:
            self.assertTrue(ranking_result.score > 0)
            self.assertIsInstance(ranking_result.components, dict)

    def test_candidate_documents(self):
        tokens = ["quick", "fox"]
        candidates = list(self.engine._candidate_documents(tokens))
        self.assertTrue(set(candidates).issubset({1, 2, 3}))
        self.assertIn(1, candidates)

    def test_signals_on_index_change(self):
        state = IndexState()
        triggered = []

        def callback(doc_id: int):
            triggered.append(doc_id)

        state.subscribe_to_changes(callback)
        state.notify_changes(42)  # Public method
        self.assertEqual(triggered, [42])

if __name__ == "__main__":
    unittest.main()
