# scout/ranking/bm25.py

import math
from collections import defaultdict
from typing import List, Dict
from .base import RankingStrategy, RankingResult
from scout.index.inverted import InvertedIndex

class BM25Ranking(RankingStrategy):
    """
    Okapi BM25 ranking strategy.

    Attributes:
        k1 (float): Term frequency saturation parameter.
        b (float): Length normalization parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(
        self,
        query_tokens: List[str],
        index: InvertedIndex,
        doc_id: int
    ) -> RankingResult:
        score: float = 0.0
        components: Dict[str, float] = defaultdict(float)

        N = index.stats.total_docs
        avg_dl = index.stats.avg_doc_length
        doc_len = index.stats.doc_lengths.get(doc_id, avg_dl)

        for token in query_tokens:
            postings = index.get_postings(token)
            tf = 0
            for d_id, freq in postings:
                if d_id == doc_id:
                    tf = freq
                    break

            df = index.doc_freqs.get(token, 0)
            if df == 0:
                continue

            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_dl))
            token_score = idf * (tf * (self.k1 + 1) / denom if denom > 0 else 0)

            score += token_score
            components[token] = token_score

        return RankingResult(score=score, components=dict(components))
