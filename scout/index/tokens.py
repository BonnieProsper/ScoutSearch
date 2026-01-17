# scout/index/tokens.py

from __future__ import annotations

import re
from typing import List, Optional


class Tokenizer:
    """
    Deterministic tokenizer used for indexing and querying.

    - Lowercases text
    - Removes punctuation
    - Splits on whitespace
    - Optionally generates n-grams
    """

    def __init__(self, ngram: Optional[int] = None) -> None:
        if ngram is not None and ngram < 1:
            raise ValueError("ngram must be >= 1")

        self.ngram = ngram
        self._punct_re = re.compile(r"[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = self._punct_re.sub("", text)

        tokens = text.split()
        if self.ngram and self.ngram > 1:
            return self._generate_ngrams(tokens)

        return tokens

    def _generate_ngrams(self, tokens: List[str]) -> List[str]:
        n = self.ngram
        assert n is not None and n > 1

        return [
            "_".join(tokens[i : i + n])
            for i in range(len(tokens) - n + 1)
        ]
