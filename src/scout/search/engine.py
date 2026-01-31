# scout/search/engine.py

from __future__ import annotations

import json
from collections.abc import Iterable

from scout.index.builder import IndexBuilder
from scout.index.inverted import InvertedIndex
from scout.index.tokens import Tokenizer
from scout.ranking.base import RankingResult, RankingStrategy
from scout.search.query import parse_query
from scout.state.signals import IndexState

DEFAULT_STOPWORDS = {"the", "a", "an", "and", "or"}


class SearchEngine:
    """
    High-level search façade coordinating tokenization, ranking,
    boolean logic, phrase filtering, and deterministic result ordering.
    """

    def __init__(
        self,
        index: InvertedIndex,
        ranking: RankingStrategy,
        tokenizer: Tokenizer,
        *,
        stopwords: set[str] | None = None,
        state: IndexState | None = None,
        field_weights: dict[str, float] | None = None,
    ) -> None:
        self._index = index
        self._ranking = ranking
        self._tokenizer = tokenizer
        self._state = state
        self._field_weights = field_weights or {}
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS

        if self._state is not None:
            self._state.on_change.subscribe(self._on_index_change)

    @classmethod
    def from_records(
        cls,
        records: list[dict],
        *,
        ranking: RankingStrategy,
        fields: list[str] | None = None,
        ngram: int | None = None,
        stopwords: set[str] | None = None,
        state: IndexState | None = None,
        field_weights: dict[str, float] | None = None,
    ) -> SearchEngine:
        builder = IndexBuilder(fields=fields, ngram=ngram)
        index = builder.build(records, field_weights=field_weights)
        tokenizer = Tokenizer(ngram=ngram)

        # Initialize IndexState with tokens if not provided
        if state is None:
            tokens_by_doc: dict[int, list[str]] = {}
            for doc_id, record in enumerate(records):
                if fields is not None:
                    used_fields = fields
                elif field_weights is not None:
                    used_fields = list(field_weights)
                else:
                    used_fields = ["text"]

                tokens: list[str] = []
                for field in used_fields:
                    value = record.get(field)
                    if not isinstance(value, str):
                        continue
                    field_tokens = tokenizer.tokenize(value)
                    weight = field_weights.get(field, 1.0) if field_weights else 1.0
                    repeat = max(1, int(weight))
                    tokens.extend(field_tokens * repeat)
                tokens = [t for t in tokens if t not in (stopwords or DEFAULT_STOPWORDS)]
                tokens_by_doc[doc_id] = tokens

            state = IndexState(tokens_by_doc)

        else:
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
        record: dict,
        *,
        fields: list[str] | None = None,
    ) -> None:
        tokens: list[str] = []

        if fields is not None:
            used_fields = fields
        elif self._field_weights:
            used_fields = list(self._field_weights)
        else:
            used_fields = ["text"]


        for field in used_fields:
            value = record.get(field)
            if not isinstance(value, str):
                continue

            field_tokens = self._tokenizer.tokenize(value)
            weight = self._field_weights.get(field, 1.0)

            repeat = max(1, int(weight))
            tokens.extend(field_tokens * repeat)

        tokens = [t for t in tokens if t not in self.stopwords]

        if self._state is not None:
            self._state.add_document(doc_id, tokens, metadata=record)
        else:
            self._index.add_document(doc_id, tokens, metadata=record)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[tuple[int, RankingResult]]:
        parsed = parse_query(query)

        raw_tokens = list(parsed.required | parsed.optional)

        if not raw_tokens and parsed.phrases:
            raw_tokens = list({t for phrase in parsed.phrases for t in phrase})

        query_tokens = [t for t in raw_tokens if t not in self.stopwords]

        if not query_tokens:
            return []

        results: dict[int, RankingResult] = {}

        for doc_id in self._candidate_documents(query_tokens):
            if parsed.exclude and any(
                self._index.document_contains(doc_id, t)
                for t in parsed.exclude
            ):
                continue

            if parsed.has_or:
                if not any(
                    self._index.document_contains(doc_id, t)
                    for t in (parsed.required | parsed.optional)
                ):
                    continue
            else:
                if parsed.required and not all(
                    self._index.document_contains(doc_id, t)
                    for t in parsed.required
                ):
                    continue

            if parsed.phrases:
                if self._state is None:
                    raise RuntimeError(
                        "Phrase queries require IndexState with document tokens"
                    )

                tokens = self._state.get_document_tokens(doc_id)
                if not self._matches_phrases(tokens, parsed.phrases):
                    continue

            ranking_result = self._ranking.score(
                query_tokens=query_tokens,
                index=self._index,
                doc_id=doc_id,
            )

            if ranking_result.score > 0.0:
                results[doc_id] = ranking_result

        return sorted(
            results.items(),
            key=lambda item: (-item[1].score, item[0]),
        )[:limit]

    def _candidate_documents(self, query_tokens: list[str]) -> Iterable[int]:
        candidates: set[int] = set()

        for token in query_tokens:
            for doc_id, _ in self._index.get_postings(token):
                candidates.add(doc_id)

        return candidates

    def save(self, path: str) -> None:
        payload = {
            "index": self._index.to_dict(),
            "config": {
                "stopwords": sorted(self.stopwords),
                "field_weights": self._field_weights,
                "ngram": self._tokenizer.ngram,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str, *, ranking: RankingStrategy) -> SearchEngine:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        index = InvertedIndex.from_dict(data["index"])
        config = data["config"]

        tokenizer = Tokenizer(ngram=config["ngram"])

        return cls(
            index=index,
            ranking=ranking,
            tokenizer=tokenizer,
            stopwords=set(config["stopwords"]),
            field_weights=config["field_weights"],
        )

    def _on_index_change(self, doc_id: int) -> None:
        pass

    @staticmethod
    def _matches_phrases(
        tokens: list[str],
        phrases: list[list[str]],
    ) -> bool:
        for phrase in phrases:
            plen = len(phrase)
            if not any(
                tokens[i : i + plen] == phrase
                for i in range(len(tokens) - plen + 1)
            ):
                return False
        return True
