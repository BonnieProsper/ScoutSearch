# scout/search/engine.py

from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime
import json

from scout.index.builder import IndexBuilder
from scout.index.inverted import InvertedIndex
from scout.index.tokens import Tokenizer
from scout.ranking.base import RankingResult, RankingStrategy
from scout.state.signals import IndexState

DEFAULT_STOPWORDS = {"the", "a", "an", "and", "or"}


class SearchEngine:
    """
    High-level search façade coordinating tokenization, ranking,
    phrase filtering, and deterministic result aggregation.
    """

    def __init__(
        self,
        index: InvertedIndex,
        ranking: RankingStrategy,
        tokenizer: Tokenizer,
        *,
        stopwords: Optional[set[str]] = None,
        state: Optional[IndexState] = None,
        field_weights: Optional[Dict[str, float]] = None,
    ):
        self._index = index
        self._ranking = ranking
        self._tokenizer = tokenizer
        self._state = state
        self._field_weights = field_weights or {}
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS

        if self._state:
            self._state.on_change.subscribe(self._on_index_change)

    @classmethod
    def from_records(
        cls,
        records: List[Dict],
        *,
        ranking: RankingStrategy,
        fields: Optional[List[str]] = None,
        ngram: Optional[int] = None,
        stopwords: Optional[set[str]] = None,
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
            stopwords=stopwords,
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
        tokens: List[str] = []
        used_fields = fields or list(self._field_weights.keys()) or ["text"]

        for field in used_fields:
            value = record.get(field)
            if not isinstance(value, str):
                continue

            field_tokens = self._tokenizer.tokenize(value)
            weight = int(self._field_weights.get(field, 1))
            tokens.extend(field_tokens * weight)

        tokens = [t for t in tokens if t not in self.stopwords]

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
        raw_tokens = list(parsed.required | parsed.optional)
        query_tokens = [t for t in raw_tokens if t not in self.stopwords]

        if not query_tokens:
            return []

        scores: Dict[int, RankingResult] = {}

        for doc_id in self._candidate_documents(query_tokens):
            if parsed.exclude and any(
                self._index.document_contains(doc_id, t) for t in parsed.exclude
            ):
                continue

            if parsed.required and not all(
                self._index.document_contains(doc_id, t) for t in parsed.required
            ):
                continue

            if parsed.phrases:
                tokens = self._state.get_document_tokens(doc_id) if self._state else []
                if not self._matches_phrases(tokens, parsed.phrases):
                    continue

            result = self._ranking.score(
                query_tokens=query_tokens,
                index=self._index,
                doc_id=doc_id,
            )

            if result.score > 0.0:
                scores[doc_id] = result

        return sorted(
            scores.items(),
            key=lambda x: (-x[1].score, x[0]),
        )[:limit]

    def _candidate_documents(self, query_tokens: List[str]) -> Iterable[int]:
        candidates = set()
        for token in query_tokens:
            for doc_id, _ in self._index.get_postings(token):
                candidates.add(doc_id)
        return sorted(candidates)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._index.to_dict(), f)

    @classmethod
    def load(cls, path, *, ranking: RankingStrategy) -> "SearchEngine":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        index = InvertedIndex.from_dict(data)
        tokenizer = Tokenizer()

        return cls(
            index=index,
            ranking=ranking,
            tokenizer=tokenizer,
        )

    def _on_index_change(self, doc_id: int) -> None:
        pass

    @staticmethod
    def _matches_phrases(tokens: List[str], phrases: List[List[str]]) -> bool:
        for phrase in phrases:
            plen = len(phrase)
            if not any(tokens[i:i + plen] == phrase for i in range(len(tokens) - plen + 1)):
                return False
        return True
