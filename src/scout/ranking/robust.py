# scout/ranking/robust.py


from scout.index.inverted import InvertedIndex

from .base import RankingResult, RankingStrategy
from .tf import TermFrequencyRanking
from .tfidf import TFIDFRanking


class RobustRanking(RankingStrategy):
    """
    Composite ranking strategy combining TF and TF-IDF.
    """

    def __init__(self, tf_weight: float = 0.4, tfidf_weight: float = 0.6):
        self.tf = TermFrequencyRanking()
        self.tfidf = TFIDFRanking()
        self.tf_weight = tf_weight
        self.tfidf_weight = tfidf_weight

    def score(
        self,
        query_tokens: list[str],
        index: InvertedIndex,
        doc_id: int
    ) -> RankingResult:
        tf_result = self.tf.score(query_tokens, index, doc_id)
        tfidf_result = self.tfidf.score(query_tokens, index, doc_id)

        score = (
            self.tf_weight * tf_result.score
            + self.tfidf_weight * tfidf_result.score
        )

        components = {
            "tf": tf_result.score,
            "tfidf": tfidf_result.score
        }

        return RankingResult(score=score, components=components)
