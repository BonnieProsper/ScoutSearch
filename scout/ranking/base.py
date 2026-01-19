# scout/ranking/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from scout.index.inverted import InvertedIndex


class RankingResult:
    """
    Result of scoring a single document for a query.
    """

    def __init__(
        self,
        score: float,
        components: Dict[str, float],
        per_term: Optional[Dict[str, Dict[str, float]]] = None,
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
