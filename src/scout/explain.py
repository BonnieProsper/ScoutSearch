# scout/explain.py


from scout.ranking.base import RankingResult
from scout.search.engine import SearchEngine


def explain_query(
    engine: SearchEngine,
    query: str,
    limit: int = 5,
) -> list[tuple[int, RankingResult]]:
    """
    Explain query results by returning top documents along with
    token-level scoring components.

    PURE FUNCTION:
    - It does not mutate engine state
    - It does not mutate RankingResult objects returned by search()
    """
    results = engine.search(query, limit=limit)
    query_tokens = query.lower().split()

    explanations: list[tuple[int, RankingResult]] = []

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
