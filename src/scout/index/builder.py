# scout/index/builder.py

from __future__ import annotations

from .inverted import InvertedIndex
from .tokens import Tokenizer


class IndexBuilder:
    """
    Deterministically builds an inverted index from structured records.
    Supports optional per-field weighting.
    """

    def __init__(
        self,
        fields: list[str] | None = None,
        ngram: int | None = None,
    ) -> None:
        self.fields = fields if fields is not None else ["text"]
        self.tokenizer = Tokenizer(ngram=ngram)

    def build(
        self,
        records: list[dict],
        field_weights: dict[str, float] | None = None,
    ) -> InvertedIndex:
        index = InvertedIndex()
        field_weights = field_weights or {f: 1.0 for f in self.fields}

        for record in records:
            if not isinstance(record, dict):
                continue

            if "id" not in record:
                continue

            doc_id = record["id"]

            all_tokens: list[str] = []

            for field in self.fields:
                value = record.get(field)
                if not isinstance(value, str):
                    continue

                tokens = self.tokenizer.tokenize(value)
                weight = int(field_weights.get(field, 1))
                all_tokens.extend(tokens * weight)

            if not all_tokens:
                continue

            index.add_document(
                doc_id,
                all_tokens,
                metadata=record,
            )

        return index
