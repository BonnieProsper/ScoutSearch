# scout/benchmarks/explainability.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from scout.benchmarks.run import BenchmarkResult
from scout.search.engine import SearchEngine


def aggregate_components(
    *,
    engine: SearchEngine,
    benchmark_results: Iterable[BenchmarkResult],
) -> dict[str, float]:
    """
    Aggregate ranking component contributions across all benchmark queries.

    Returns:
        Dict[str, float]: component_name -> average contribution
    """
    totals = defaultdict(float)
    counts = defaultdict(int)

    for result in benchmark_results:
        hits = engine.search(result.query, limit=len(result.retrieved))
        for _, ranking_result in hits:
            for component, value in ranking_result.components.items():
                totals[component] += value
                counts[component] += 1

    return {
        component: totals[component] / counts[component]
        for component in totals
        if counts[component] > 0
    }
