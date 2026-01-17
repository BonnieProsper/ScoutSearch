# scout/state/signals.py

from typing import Callable, List
from scout.index.inverted import InvertedIndex

class Signal:
    """Reactive signal system for index updates."""

    def __init__(self):
        self._subscribers: List[Callable] = []

    def subscribe(self, callback: Callable):
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        self._subscribers.remove(callback)

    def emit(self, *args, **kwargs):
        for callback in self._subscribers:
            callback(*args, **kwargs)


class IndexState:
    """
    Wraps an index and emits signals when modified.
    """

    def __init__(self):
        self.index = InvertedIndex()
        self.on_change = Signal()

    def add_document(self, doc_id: int, tokens: list[str], metadata: dict | None = None):
        self.index.add_document(doc_id, tokens, metadata=metadata or {})
        self.on_change.emit(doc_id=doc_id)

    def subscribe_to_changes(self, callback: Callable):
        self.on_change.subscribe(callback)
