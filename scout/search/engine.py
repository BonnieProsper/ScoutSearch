# scout/search/engine.py

from typing import Dict, Iterable, List, Optional, Tuple

from scout.index.builder import IndexBuilder
from scout.index.inverted import InvertedIndex
from scout.index.tokens import Tokenizer
from scout.ranking.base import RankingResult, RankingStrategy
from scout.state.signals import IndexState

class SearchEngine:
    """
    High-level search façade coordinating tokenization, ranking,
    and result aggregation over an inverted index.
    """

    def __init__(
        self,
        index: InvertedIndex,
        ranking: RankingStrategy,
        tokenizer: Tokenizer,
        state: Optional[IndexState] = None, # optional reactive state
    ):
        self._index = index
        self._ranking = ranking
        self._tokenizer = tokenizer
        self._state = state
        if self._state:
            self._state.subscribe_to_changes(self._on_index_change)
    @classmethod
    def from_records(
        cls,
        records: List[Dict],
        *,
        fields: List[str] | None = None,
        ngram: int | None = None,
        ranking: RankingStrategy,
    ) -> "SearchEngine":
        """
        Factory constructor for building an index directly from records.
        """
        builder = IndexBuilder(fields=fields, ngram=ngram)
        index = builder.build(records)
        tokenizer = Tokenizer(ngram=ngram)

        return cls(
            index=index,
            ranking=ranking,
            tokenizer=tokenizer,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> List[Tuple[int, RankingResult]]:
        """
        Execute a ranked search over the indexed documents.

        Returns:
            List of (doc_id, RankingResult), sorted by score descending.
        """
        query_tokens = self._tokenizer.tokenize(query)

        if not query_tokens:
            return []

        scores: Dict[int, RankingResult] = {}

        for doc_id in self._candidate_documents(query_tokens):
            result = self._ranking.score(
                query_tokens=query_tokens,
                index=self._index,
                doc_id=doc_id,
            )

            if result.score > 0.0:
                scores[doc_id] = result

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1].score,
            reverse=True,
        )

        return ranked[:limit]

    def _candidate_documents(self, query_tokens: List[str]) -> Iterable[int]:
        """
        Collect candidate document IDs by unioning postings lists
        for all query tokens.
        """
        candidates = set()

        for token in query_tokens:
            for doc_id, _ in self._index.get_postings(token):
                candidates.add(doc_id)

        return candidates
    
    def _on_index_change(self, doc_id: int):
        # For example: log or re-cache candidates
        print(f"[Signal] Document {doc_id} added to index.")
