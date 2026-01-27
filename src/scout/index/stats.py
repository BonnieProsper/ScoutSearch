# scout/index/stats.py

from typing import Any


class IndexStats:
    """
    Stores corpus-level statistics needed for ranking.
    """

    def __init__(self) -> None:
        self.doc_lengths: dict[int, int] = {}
        self.total_docs: int = 0

    def add_document(self, doc_id: int, length: int) -> None:
        self.doc_lengths[doc_id] = length
        self.total_docs += 1

    def get_doc_length(self, doc_id: int) -> int:
        return self.doc_lengths.get(doc_id, 0)

    @property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 1.0
        return sum(self.doc_lengths.values()) / self.total_docs

    # ----------------------------
    # Snapshot/persistence API
    # ----------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_lengths": self.doc_lengths,
            "total_docs": self.total_docs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexStats":
        stats = cls()
        stats.doc_lengths = data["doc_lengths"]
        stats.total_docs = data["total_docs"]
        return stats
