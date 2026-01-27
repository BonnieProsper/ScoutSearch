# scout/ranking/composite.py


from scout.index.inverted import InvertedIndex

from .base import RankingResult, RankingStrategy
from .recency import RecencyRanking


class CompositeRanking(RankingStrategy):
    """
    Combines multiple ranking strategies (linear fusion) and optional recency boost.
    """

    def __init__(self, strategies: list[RankingStrategy], weights: list[float], recency: RecencyRanking | None = None):
        assert len(strategies) == len(weights)
        self.strategies = strategies
        self.weights = weights
        self.recency = recency

    def score(self, query_tokens: list[str], index: InvertedIndex, doc_id: int) -> RankingResult:
        total_score = 0.0
        components = {}
        per_term = {}

        for strategy, weight in zip(self.strategies, self.weights):
            result = strategy.score(query_tokens, index, doc_id)
            total_score += result.score * weight
            components[strategy.__class__.__name__] = result.score * weight
            for token, breakdown in result.per_term.items():
                per_term.setdefault(token, {}).update(breakdown)

        if self.recency:
            rec_result = self.recency.score(query_tokens, index, doc_id)
            total_score += rec_result.score
            components["recency"] = rec_result.score

        return RankingResult(score=total_score, components=components, per_term=per_term)
