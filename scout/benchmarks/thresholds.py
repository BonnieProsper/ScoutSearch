# scout/benchmarks/thresholds.py
from dataclasses import dataclass

@dataclass(frozen=True)
class RegressionThresholds:
    # Relevance metrics (negative means fail if candidate drops below baseline)
    min_precision_delta: float = -0.01
    min_recall_delta: float = -0.01
    min_mrr_delta: float = -0.01
    min_ndcg_delta: float = -0.01  # for your test

    # Latency metric
    max_latency_regression_ms: float = 5.0
