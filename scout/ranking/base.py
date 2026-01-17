# scout/ranking/base.py

from abc import ABC, abstractmethod
from typing import Dict, List
from scout.index.inverted import InvertedIndex

class RankingResult:
    def __init__(self, score: float, components: Dict[str, float]):
        self.score = score
        self.components = components

class RankingStrategy(ABC):
    """
    Base interface for ranking strategies.
    """

    @abstractmethod
    def score(
        self,
        query_tokens: List[str],
        index: InvertedIndex,
        doc_id: int
    ) -> RankingResult:
        pass
