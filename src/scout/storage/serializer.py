# scout/storage/serializer.py

import pickle
from typing import Any


def save(obj: Any, path: str) -> None:
    """Serialize object to disk using pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load(path: str) -> Any:
    """Load object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
