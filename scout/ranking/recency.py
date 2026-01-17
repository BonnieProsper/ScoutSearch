# scout/ranking/recency.py

from typing import List
from datetime import datetime, timedelta
from .base import RankingStrategy, RankingResult
from scout.index.inverted import InvertedIndex

class RecencyRanking(RankingStrategy):
    """
    Boosts documents based on recency.
    Requires `documents` to have a 'timestamp' field (datetime or ISO string).
    """

    def __init__(self, decay_days: float = 30.0, max_boost: float = 1.0):
        """
        :param decay_days: number of days for score to decay to ~0.37 (1/e)
        :param max_boost: maximum score boost applied to very recent docs
        """
        self.decay_days = decay_days
        self.max_boost = max_boost

    def score(
        self,
        query_tokens: List[str],
        index: InvertedIndex,
        doc_id: int
    ) -> RankingResult:
        doc = index.get_document(doc_id)
        ts = doc.get("timestamp")
        if ts is None:
            return RankingResult(score=0.0, components={"recency": 0.0})

        # Convert ISO string to datetime if needed
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        age_days = (datetime.now() - ts).total_seconds() / 86400
        recency_score = self.max_boost * pow(2.71828, -age_days / self.decay_days)

        return RankingResult(score=recency_score, components={"recency": recency_score})
