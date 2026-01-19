# scout/search/engine.py

from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime

from scout.index.builder import IndexBuilder
from scout.index.inverted import InvertedIndex
from scout.index.tokens import Tokenizer
from scout.ranking.base import RankingResult, RankingStrategy
from scout.state.signals import IndexState


class SearchEngine:
    """
    High-level search façade coordinating tokenization, ranking,
    field/phrase weighting, and result aggregation over an inverted index.
    """

    def __init__(
        self,
        index: InvertedIndex,
        ranking: RankingStrategy,
        tokenizer: Tokenizer,
        state: Optional[IndexState] = None,
        field_weights: Optional[Dict[str, float]] = None,
    ):
        self._index = index
        self._ranking = ranking
        self._tokenizer = tokenizer
        self._state = state
        self._field_weights = field_weights or {}

        if self._state:
            self._state.on_change.subscribe(self._on_index_change)

    @classmethod
    def from_records(
        cls,
        records: List[Dict],
        *,
        fields: Optional[List[str]] = None,
        ngram: Optional[int] = None,
        ranking: RankingStrategy,
        state: Optional[IndexState] = None,
        field_weights: Optional[Dict[str, float]] = None,
    ) -> "SearchEngine":
        builder = IndexBuilder(fields=fields, ngram=ngram)
        index = builder.build(records, field_weights=field_weights)
        tokenizer = Tokenizer(ngram=ngram)

        if state:
            state.index = index

        return cls(
            index=index,
            ranking=ranking,
            tokenizer=tokenizer,
            state=state,
            field_weights=field_weights,
        )

    def add_document(
        self,
        doc_id: int,
        record: Dict,
        *,
        fields: Optional[List[str]] = None,
    ) -> None:
        """
        Incrementally add a document to the index, supporting field weighting.
        """
        tokens: List[str] = []
        used_fields = fields or list(self._field_weights.keys()) or ["text"]
        for field in used_fields:
            field_text = str(record.get(field, ""))
            weight = self._field_weights.get(field, 1.0)
            field_tokens = self._tokenizer.tokenize(field_text)
            tokens.extend(field_tokens * int(weight))

        if self._state:
            self._state.add_document(doc_id, tokens, metadata=record)
        else:
            self._index.add_document(doc_id, tokens, metadata=record)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> List[Tuple[int, RankingResult]]:
        from scout.search.query import parse_query

        parsed = parse_query(query)
        query_tokens = list(parsed.required | parsed.optional)
        if not query_tokens:
            return []

        scores: Dict[int, RankingResult] = {}

        for doc_id in self._candidate_documents(query_tokens):
            # Exclude terms
            if parsed.exclude and any(
                self._index.document_contains(doc_id, t) for t in parsed.exclude
            ):
                continue

            # Required terms
            if parsed.required and not all(
                self._index.document_contains(doc_id, t) for t in parsed.required
            ):
                continue

            # Phrase filtering
            if parsed.phrases:
                if self._state:
                    doc_tokens = self._state.get_document_tokens(doc_id)
                else:
                    doc_tokens = []
                if not self._matches_phrases(doc_tokens, parsed.phrases):
                    continue

            result = self._ranking.score(query_tokens=query_tokens, index=self._index, doc_id=doc_id)

            if result.score > 0.0:
                scores[doc_id] = result

        # Sort descending by score
        return sorted(scores.items(), key=lambda x: x[1].score, reverse=True)[:limit]

    def _candidate_documents(self, query_tokens: List[str]) -> Iterable[int]:
        candidates = set()
        for token in query_tokens:
            for doc_id, _ in self._index.get_postings(token):
                candidates.add(doc_id)
        return sorted(candidates)  # deterministic order

    def _on_index_change(self, doc_id: int) -> None:
        # Hook for caching invalidation, logging, persistence
        pass

    @staticmethod
    def _matches_phrases(tokens: List[str], phrases: List[List[str]]) -> bool:
        for phrase in phrases:
            plen = len(phrase)
            found = any(tokens[i:i+plen] == phrase for i in range(len(tokens) - plen + 1))
            if not found:
                return False
        return True
