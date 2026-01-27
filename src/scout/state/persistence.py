# scout/state/persistence.py

from scout.state.signals import IndexState
from scout.state.store import Store


class AutoSaver:
    """
    Automatically persists the index whenever it changes.

    This is intentionally decoupled from SearchEngine.
    """

    def __init__(self, state: IndexState):
        self._state = state
        self._state.on_change.subscribe(self._on_change)

    def _on_change(self, doc_id: int) -> None:
        # Persist entire index snapshot (atomic save)
        Store.save(self._state.index)
