# scout/ranking/bm25.py

import math
from collections import defaultdict
from typing import Dict, List

from scout.index.inverted import InvertedIndex
from scout.ranking.base import RankingResult, RankingStrategy


class BM25Ranking(RankingStrategy):
    """
    Okapi BM25 ranking strategy.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(
        self,
        query_tokens: List[str],
        index: InvertedIndex,
        doc_id: int,
    ) -> RankingResult:
        total_score = 0.0
        per_term: Dict[str, Dict[str, float]] = {}

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
            if df == 0 or tf == 0:
                continue

            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_dl))
            score = idf * (tf * (self.k1 + 1) / denom)

            total_score += score
            per_term[token] = {
                "tf": float(tf),
                "df": float(df),
                "idf": float(idf),
                "score": float(score),
            }

        components = {
            "bm25": total_score,
            "k1": self.k1,
            "b": self.b,
        }

        return RankingResult(
            score=total_score,
            components=components,
            per_term=per_term,
        )
