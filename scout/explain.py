# scout/explain.py

from typing import List, Dict, Tuple
from scout.search.engine import SearchEngine
from scout.ranking.base import RankingResult

def explain_query(engine: SearchEngine, query: str, limit: int = 5) -> List[Tuple[int, RankingResult]]:
    """
    Returns top results along with component-wise scoring.
    """
    results = engine.search(query, limit=limit)
    explanations = []
    for doc_id, result in results:
        explanations.append((doc_id, result))
    return explanations

if __name__ == "__main__":
    from scout.ranking.robust import RobustRanking
    from scout.index.builder import IndexBuilder
    records = [{"id": i, "text": f"Document {i} sample text"} for i in range(1, 11)]
    ranking = RobustRanking()
    engine = SearchEngine.from_records(records, ranking=ranking)
    query = "sample text"
    for doc_id, result in explain_query(engine, query):
        print(f"Doc {doc_id}: score={result.score:.4f}, components={result.components}")
