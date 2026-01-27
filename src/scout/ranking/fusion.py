# scout/ranking/fusion.py


from scout.index.inverted import InvertedIndex
from scout.ranking.base import RankingResult, RankingStrategy


class FusionRanking(RankingStrategy):
    """
    Linear weighted fusion of multiple ranking strategies.
    """

    def __init__(self, strategies: list[RankingStrategy], weights: list[float]):
        assert len(strategies) == len(weights)
        self._strategies = strategies
        self._weights = weights

    def score(self, query_tokens, index: InvertedIndex, doc_id: int) -> RankingResult:
        total = 0.0
        components = {}
        per_term = {}

        for strategy, weight in zip(self._strategies, self._weights, strict=True):
            result = strategy.score(query_tokens, index, doc_id)
            total += result.score * weight
            components[strategy.__class__.__name__] = result.score * weight

            for token, breakdown in result.per_term.items():
                per_term.setdefault(token, {}).update(breakdown)

        return RankingResult(
            score=total,
            components=components,
            per_term=per_term,
        )
