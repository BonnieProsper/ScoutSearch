# scout/benchmarks/config_loader.py
from __future__ import annotations
import json
from pathlib import Path


def load_benchmark_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())

    for key in ("dataset", "ranking", "benchmark", "output"):
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")

    return cfg
