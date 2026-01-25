# scout/index/builder.py

from __future__ import annotations
from typing import Dict, List, Optional

from .tokens import Tokenizer
from .inverted import InvertedIndex


class IndexBuilder:
    """
    Deterministically builds an inverted index from structured records.
    Supports optional per-field weighting.
    """

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        ngram: Optional[int] = None,
    ) -> None:
        self.fields = fields if fields is not None else ["text"]
        self.tokenizer = Tokenizer(ngram=ngram)

    def build(
        self,
        records: List[Dict],
        field_weights: Optional[Dict[str, float]] = None,
    ) -> InvertedIndex:
        index = InvertedIndex()
        field_weights = field_weights or {f: 1.0 for f in self.fields}

        for record in records:
            if not isinstance(record, dict):
                continue

            if "id" not in record:
                continue

            doc_id = record["id"]

            all_tokens: List[str] = []

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
