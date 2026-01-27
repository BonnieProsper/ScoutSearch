# scout/ranking/tf.py

from collections import defaultdict

from scout.index.inverted import InvertedIndex

from .base import RankingResult, RankingStrategy


class TermFrequencyRanking(RankingStrategy):
    """
    Pure term-frequency scoring.
    """

    def score(
        self,
        query_tokens: list[str],
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
