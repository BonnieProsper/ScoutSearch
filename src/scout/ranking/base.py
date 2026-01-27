# scout/ranking/base.py

from abc import ABC, abstractmethod

from scout.index.inverted import InvertedIndex


class RankingResult:
    """
    Result of scoring a single document.

    Immutable by convention.
    """

    def __init__(
        self,
        score: float,
        components: dict[str, float],
        per_term: dict[str, dict[str, float]] | None = None,
    ):
        self.score = score
        self.components = components
        self.per_term = per_term or {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RankingResult):
            return False
        return (
            self.score == other.score
            and self.components == other.components
            and self.per_term == other.per_term
        )


class RankingStrategy(ABC):
    @abstractmethod
    def score(
        self,
        query_tokens: list[str],
        index: InvertedIndex,
        doc_id: int,
    ) -> RankingResult:
        raise NotImplementedError
