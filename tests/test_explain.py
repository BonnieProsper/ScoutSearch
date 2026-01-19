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

# tests/test_explain.py

from scout.search.engine import SearchEngine
from scout.ranking.bm25 import BM25Ranking
from scout.explain import explain_query


def test_explainability_contains_all_tokens():
    records = [
        {"text": "apple banana"},
        {"text": "apple orange"},
    ]

    engine = SearchEngine.from_records(
        records,
        ranking=BM25Ranking(),
    )

    results = explain_query(engine, "apple banana", limit=2)

    assert results
    for _, result in results:
        assert "apple" in result.per_term
        assert "banana" in result.per_term
