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

# TODO: config_loader comparison
def load_benchmark_artifact(path: Path) -> dict:
    """Load benchmark artifact from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return json.loads(path.read_text())
