from scout.state.persistence import AutoSaver
from scout.state.signals import IndexState


def test_autosave_triggers_on_index_change(monkeypatch):
    state = IndexState()
    saved = []

    def fake_save(index):
        saved.append(index)

    monkeypatch.setattr(
        "scout.state.store.Store.save",
        fake_save,
    )

    AutoSaver(state)

    state.add_document(1, ["hello", "world"])

    assert len(saved) == 1
    assert saved[0] is state.index
