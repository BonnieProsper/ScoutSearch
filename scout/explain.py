# scout/explain.py

from typing import List, Tuple
from scout.search.engine import SearchEngine
from scout.ranking.base import RankingResult


def explain_query(
    engine: SearchEngine,
    query: str,
    limit: int = 5,
) -> List[Tuple[int, RankingResult]]:
    """
    Explain query results by returning top documents along with
    token-level scoring components.

    This function is PURE:
    - It does not mutate engine state
    - It does not mutate RankingResult objects returned by search()
    """
    results = engine.search(query, limit=limit)

    explanations: List[Tuple[int, RankingResult]] = []
    query_tokens = query.lower().split()

    for doc_id, result in results:
        # Copy components defensively
        components = dict(result.components)

        # Ensure each query token is represented
        for token in query_tokens:
            components.setdefault(token, 0.0)

        explained = RankingResult(
            score=result.score,
            components=components,
        )
        explanations.append((doc_id, explained))

    return explanations
