# scout/ranking/bm25.py

import math
from collections import defaultdict
from typing import List
from .base import RankingStrategy, RankingResult
from scout.index.inverted import InvertedIndex

class BM25Ranking(RankingStrategy):
    """
    BM25 ranking strategy.
    """
    def __init__(self, k: float = 1.5, b: float = 0.75):
        self.k = k
        self.b = b

    def score(
        self,
        query_tokens: List[str],
        index: InvertedIndex,
        doc_id: int
    ) -> RankingResult:
        score = 0.0
        components = defaultdict(float)

        N = index.stats.total_docs
        doc_length = index.stats.doc_lengths.get(doc_id, 0)
        avg_doc_length = index.stats.avg_doc_length

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
            denom = tf + self.k * (1 - self.b + self.b * (doc_length / avg_doc_length))
            score_token = idf * ((tf * (self.k + 1)) / denom if denom != 0 else 0)
            score += score_token
            components[token] += score_token

        return RankingResult(score=score, components=dict(components))
