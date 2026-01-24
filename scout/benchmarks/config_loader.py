# scout/benchmarks/config_loader.py
from __future__ import annotations

import json
from pathlib import Path

from scout.benchmarks.config_schema import FullBenchmarkConfig


def load_benchmark_config(path: Path) -> FullBenchmarkConfig:
    raw = json.loads(path.read_text())
    return FullBenchmarkConfig.model_validate(raw)
