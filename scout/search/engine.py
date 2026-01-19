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
        state: Optional[IndexState] = None,
    ):
        self._index = index
        self._ranking = ranking
        self._tokenizer = tokenizer
        self._state = state

        if self._state:
            self._state.on_change.subscribe(self._on_index_change)


    @classmethod
    def from_records(
        cls,
        records: List[Dict],
        *,
        fields: List[str] | None = None,
        ngram: int | None = None,
        ranking: RankingStrategy,
        state: Optional[IndexState] = None,
    ) -> "SearchEngine":
        builder = IndexBuilder(fields=fields, ngram=ngram)
        index = builder.build(records)
        tokenizer = Tokenizer(ngram=ngram)

        if state:
            state.index = index

        return cls(
            index=index,
            ranking=ranking,
            tokenizer=tokenizer,
            state=state,
        )

    def add_document(
        self,
        doc_id: int,
        record: Dict,
        *,
        fields: List[str] | None = None,
    ) -> None:
        """
        Incrementally add a document to the index.
        """
        tokens = self._tokenizer.tokenize_record(record, fields=fields)

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
        query_tokens = list(parsed.include | parsed.optional)

        if not query_tokens:
            return []

        scores: Dict[int, RankingResult] = {}

        for doc_id in self._candidate_documents(query_tokens):
            # Exclusion filter (CORRECT)
            if parsed.exclude:
                if any(
                    self._index.document_contains(doc_id, token)
                    for token in parsed.exclude
                ):
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
            key=lambda item: item[1].score,
            reverse=True,
        )[:limit]


    def _candidate_documents(self, query_tokens: List[str]) -> Iterable[int]:
        candidates = set()
        for token in query_tokens:
            for doc_id, _ in self._index.get_postings(token):
                candidates.add(doc_id)
        return candidates

    def _on_index_change(self, doc_id: int) -> None:
        # Hook for caching, persistence, logging
        pass
