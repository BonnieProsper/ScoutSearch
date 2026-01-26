# tests/benchmarks/test_regression.py

from scout.benchmarks.regression import compare_benchmarks
from scout.benchmarks.thresholds import RegressionThresholds

def test_detects_ndcg_regression(sample_baseline, sample_candidate, queries):
    report = compare_benchmarks(
        baseline_results=sample_baseline,
        candidate_results=sample_candidate,
        queries=queries,
        k=10,
        thresholds=RegressionThresholds(min_ndcg_delta=-0.0001),
    )

    assert report.failed
    assert any("nDCG" in r for r in report.reasons)

