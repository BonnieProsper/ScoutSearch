# scout/explain.py

from typing import List, Tuple
from scout.search.engine import SearchEngine
from scout.ranking.base import RankingResult

def explain_query(
    engine: SearchEngine,
    query: str,
    limit: int = 5
) -> List[Tuple[int, RankingResult]]:
    """
    Explain query results by returning top documents along with token-wise scoring.

    Args:
        engine (SearchEngine): Engine to execute query.
        query (str): Query string.
        limit (int): Max number of results to return.

    Returns:
        List of tuples: (doc_id, RankingResult)
    """
    results = engine.search(query, limit=limit)
    explanations: List[Tuple[int, RankingResult]] = [(doc_id, result) for doc_id, result in results]
    return explanations
