# scout/state/signals.py

from collections.abc import Callable

from scout.index.inverted import InvertedIndex


class Signal:
    """Reactive signal system."""

    def __init__(self):
        self._subscribers: list[Callable] = []

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, *args, **kwargs) -> None:
        for callback in self._subscribers:
            callback(*args, **kwargs)


class IndexState:
    """
    Wraps an index and emits signals when modified.
    Stores raw document tokens for phrase matching and autosave.
    """

    def __init__(self):
        self.index = InvertedIndex()
        self.on_change = Signal()
        self._doc_tokens: dict[int, list[str]] = {}  # NEW: store raw tokens

    def add_document(
        self,
        doc_id: int,
        tokens: list[str],
        metadata: dict | None = None,
    ) -> None:
        self.index.add_document(doc_id, tokens, metadata or {})
        self._doc_tokens[doc_id] = tokens  # save tokens for phrase matching
        self.on_change.emit(doc_id=doc_id)

    def get_document_tokens(self, doc_id: int) -> list[str]:
        """Retrieve raw tokens for a document (used for phrase matching)."""
        return self._doc_tokens.get(doc_id, [])
