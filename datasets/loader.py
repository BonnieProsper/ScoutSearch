# scout/datasets/loader.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


def load_jsonl(
    path: str | Path,
    *,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Stream records from a JSONL file.

    Each line must be a single JSON object.
    Designed for large datasets (millions of records).
    """
    path = Path(path)
    count = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            yield json.loads(line)
            count += 1

            if limit is not None and count >= limit:
                break


def load_records(
    path: str | Path,
    *,
    limit: Optional[int] = None,
) -> Iterable[Dict]:
    """
    Load records from a dataset file.

    Currently supported:
    - .jsonl (streaming)
    - .json  (fully loaded; small datasets only)
    """
    path = Path(path)

    if path.suffix == ".jsonl":
        return load_jsonl(path, limit=limit)

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if limit is not None:
                return data[:limit]
            return data

    raise ValueError(f"Unsupported dataset format: {path.suffix}")
     
