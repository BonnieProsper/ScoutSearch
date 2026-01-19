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
    query_tokens = query.lower().split()

    explanations: List[Tuple[int, RankingResult]] = []

    for doc_id, result in results:
        # Defensive copies
        components = dict(result.components)
        per_term = dict(result.per_term)

        # Ensure every query token appears in explanations
        for token in query_tokens:
            components.setdefault(token, 0.0)
            per_term.setdefault(token, {})

        explained = RankingResult(
            score=result.score,
            components=components,
            per_term=per_term,
        )
        explanations.append((doc_id, explained))

    return explanations
