# scout/index/inverted.py

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .stats import IndexStats


Posting = Tuple[int, int]  # (doc_id, term_frequency)


class InvertedIndex:
    """
    Inverted index mapping tokens to postings lists.

    token -> [(doc_id, term_frequency)]
    """

    def __init__(self) -> None:
        self.index: Dict[str, List[Posting]] = defaultdict(list)
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.documents: Dict[int, dict] = {}
        self.stats = IndexStats()

    def add_document(
        self,
        doc_id: int,
        tokens: List[str],
        metadata: Optional[dict] = None,
    ) -> None:
        self.documents[doc_id] = metadata or {}

        token_counts: Dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        for token, freq in token_counts.items():
            self.index[token].append((doc_id, freq))
            self.doc_freqs[token] += 1

        self.stats.add_document(doc_id, len(tokens))

    def get_postings(self, token: str) -> List[Posting]:
        return self.index.get(token, [])

    def get_document(self, doc_id: int) -> dict:
        return self.documents.get(doc_id, {})

    def document_contains(self, doc_id: int, token: str) -> bool:
        """
        Return True if the document contains the given token.
        """
        for posting_doc_id, _ in self.get_postings(token):
            if posting_doc_id == doc_id:
                return True
        return False

    # ----------------------------
    # Snapshot / persistence API
    # ----------------------------

    def to_dict(self) -> Dict:
        """
        Deterministic, JSON-serializable snapshot of the index.
        """
        return {
            "index": {
                term: list(postings)
                for term, postings in self.index.items()
            },
            "doc_freqs": dict(self.doc_freqs),
            "documents": self.documents,
            "stats": self.stats.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "InvertedIndex":
        index = cls()

        index.index = defaultdict(
            list,
            {
                term: [tuple(p) for p in postings]
                for term, postings in data["index"].items()
            },
        )
        index.doc_freqs = defaultdict(int, data["doc_freqs"])
        index.documents = data["documents"]
        index.stats = IndexStats.from_dict(data["stats"])

        return index
