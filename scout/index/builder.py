# scout/index/builder.py

from __future__ import annotations

from typing import Dict, List, Optional

from .tokens import Tokenizer
from .inverted import InvertedIndex


class IndexBuilder:
    """
    Deterministically builds an inverted index from structured records.
    """

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        ngram: Optional[int] = None,
    ) -> None:
        self.fields = fields if fields is not None else ["text"]
        self.tokenizer = Tokenizer(ngram=ngram)

    def build(self, records: List[Dict]) -> InvertedIndex:
        index = InvertedIndex()

        for record in records:
            doc_id = record["id"]

            text = " ".join(
                str(record.get(field, ""))
                for field in self.fields
            )

            tokens = self.tokenizer.tokenize(text)
            index.add_document(
                doc_id,
                tokens,
                metadata=record,
            )

        return index
