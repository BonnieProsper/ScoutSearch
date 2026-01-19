# scout/state/persistence.py

from scout.state.store import Store
from scout.state.signals import IndexState


class AutoSaver:
    """
    Automatically persists index when it changes.
    """

    def __init__(self, state: IndexState):
        self._state = state
        self._state.on_change.subscribe(self._on_change)

    def _on_change(self, doc_id: int) -> None:
        Store.save(self._state.index)
