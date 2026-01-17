# scout/state/signals.py

from typing import Callable, List
from scout.index.inverted import InvertedIndex

class Signal:
    """Reactive signal system for index updates."""

    def __init__(self):
        self._subscribers: List[Callable] = []

    def subscribe(self, callback: Callable):
        """Subscribe to signal notifications."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        """Remove a subscriber."""
        self._subscribers.remove(callback)

    def emit(self, *args, **kwargs):
        """Emit signal to all subscribers."""
        for callback in self._subscribers:
            callback(*args, **kwargs)


class IndexState:
    """
    Wraps an index and emits signals when modified.
    Provides a public notify_changes method for reactive updates.
    """

    def __init__(self):
        self.index = InvertedIndex()
        self.on_change = Signal()

    def add_document(self, doc_id: int, tokens: list[str], metadata: dict | None = None):
        """
        Add a document to the index and notify subscribers.
        """
        self.index.add_document(doc_id, tokens, metadata=metadata or {})
        self.notify_changes(doc_id)

    def subscribe_to_changes(self, callback: Callable[[int], None]):
        """
        Subscribe a callback to index changes.
        """
        self.on_change.subscribe(callback)

    def notify_changes(self, doc_id: int):
        """
        Notify all subscribers of a document change.
        """
        self.on_change.emit(doc_id=doc_id)
