# scout/index/tokens.py

from __future__ import annotations

import re
from collections.abc import Iterable


class Tokenizer:
    """
    Deterministic tokenizer used for indexing and querying.

    - Unicode-aware
    - Lowercases text
    - Extracts word tokens
    - Optionally generates n-grams
    """

    _token_re = re.compile(r"\w+", flags=re.UNICODE)

    def __init__(self, ngram: int | None = None) -> None:
        if ngram is not None and ngram < 1:
            raise ValueError("ngram must be >= 1")

        self.ngram = ngram

    def tokenize(self, text: str) -> list[str]:
        tokens = self._token_re.findall(text.lower())

        if self.ngram and self.ngram > 1:
            return self._generate_ngrams(tokens)

        return tokens

    def tokenize_record(
        self,
        record: dict,
        *,
        fields: Iterable[str] | None = None,
    ) -> list[str]:
        texts: list[str] = []

        if fields:
            for field in fields:
                value = record.get(field)
                if isinstance(value, str):
                    texts.append(value)
        else:
            for value in record.values():
                if isinstance(value, str):
                    texts.append(value)

        combined = " ".join(texts)
        return self.tokenize(combined)

    def _generate_ngrams(self, tokens: list[str]) -> list[str]:
        n = self.ngram
        assert n is not None and n > 1

        return [
            "_".join(tokens[i : i + n])
            for i in range(len(tokens) - n + 1)
        ]
