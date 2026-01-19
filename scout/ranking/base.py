# scout/ranking/base.py

from abc import ABC, abstractmethod
from typing import Dict, List
from scout.index.inverted import InvertedIndex


class RankingResult:
    """
    Result of scoring a single document.

    - score: final numeric score
    - components: high-level score components (strategy-defined)
    - per_term: token-level explanations (optional, strategy-defined)
    """

    def __init__(
        self,
        score: float,
        components: Dict[str, float],
        per_term: Dict[str, Dict[str, float]] | None = None,
    ):
        self.score = score
        self.components = components
        self.per_term = per_term or {}


class RankingStrategy(ABC):
    """
    Base interface for ranking strategies.
    """

    @abstractmethod
    def score(
        self,
        query_tokens: List[str],
        index: InvertedIndex,
        doc_id: int,
    ) -> RankingResult:
        raise NotImplementedError
