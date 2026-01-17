# tests/test_explain.py

import unittest
from scout.explain import explain_query
from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking


class TestExplain(unittest.TestCase):
    def setUp(self):
        self.records = [{"id": i, "text": f"Document {i} sample text"} for i in range(1, 6)]
        self.engine = SearchEngine.from_records(self.records, ranking=RobustRanking())

    def test_explain_query(self):
        explanations = explain_query(self.engine, "sample text", limit=3)
        self.assertEqual(len(explanations), 3)

        for doc_id, result in explanations:
            self.assertTrue(result.score > 0)
            self.assertIn("sample", result.components)
            self.assertIn("text", result.components)


if __name__ == "__main__":
    unittest.main()
