# scout/benchmarks/artifacts.py
from __future__ import annotations
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Dict, Any

from scout.benchmarks.run import BenchmarkResult


def dataset_fingerprint(records: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()


def write_benchmark_artifact(
    *,
    path: Path,
    results: Iterable[BenchmarkResult],
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "metadata": metadata,
        "results": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
