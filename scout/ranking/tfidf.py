# scout/ranking/tfidf.py

import math
from collections import defaultdict
from typing import List
from .base import RankingStrategy, RankingResult
from scout.index.inverted import InvertedIndex


class TFIDFRanking(RankingStrategy):
    """
    TF-IDF scoring.
    """

    def score(
        self,
        query_tokens: List[str],
        index: InvertedIndex,
        doc_id: int
    ) -> RankingResult:
        score = 0.0
        components = defaultdict(float)

        N = index.stats.total_docs

        for token in query_tokens:
            df = index.doc_freqs.get(token, 0)
            if df == 0:
                continue

            idf = math.log((N + 1) / (df + 1)) + 1.0
            postings = index.get_postings(token)

            for d_id, tf in postings:
                if d_id == doc_id:
                    tfidf = tf * idf
                    score += tfidf
                    components[token] += tfidf

        return RankingResult(score=score, components=dict(components))
