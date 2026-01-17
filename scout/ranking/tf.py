# scout/ranking/tf.py

from collections import defaultdict
from typing import List
from .base import RankingStrategy, RankingResult
from scout.index.inverted import InvertedIndex


class TermFrequencyRanking(RankingStrategy):
    """
    Pure term-frequency scoring.
    """

    def score(
        self,
        query_tokens: List[str],
        index: InvertedIndex,
        doc_id: int
    ) -> RankingResult:
        score = 0.0
        components = defaultdict(float)

        for token in query_tokens:
            postings = index.get_postings(token)
            for d_id, freq in postings:
                if d_id == doc_id:
                    score += freq
                    components[token] += freq

        return RankingResult(score=score, components=dict(components))
