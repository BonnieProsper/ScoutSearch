from pathlib import Path

from scout.benchmarks import (
    BenchmarkQuery,
    run_benchmark,
)
from scout.benchmarks.index import build_benchmark_index
from scout.ranking.robust import RobustRanking
from scout.search.engine import SearchEngine


def test_run_benchmark_executes(tmp_path: Path):
    """
    Integration-level benchmark test.

    Verifies that:
    - a benchmark index can be built from disk
    - a SearchEngine can execute benchmark queries
    - latency statistics are produced deterministically
    """

    # --- write a tiny dataset to disk (realistic benchmark input) ---
    dataset_path = tmp_path / "docs.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"id": "1", "text": "Document one sample text"}',
                '{"id": "2", "text": "Another document with sample content"}',
                '{"id": "3", "text": "More text data for benchmarking"}',
            ]
        )
    )

    # --- build deterministic benchmark index ---
    benchmark_index = build_benchmark_index(
        name="test",
        dataset_path=dataset_path,
        id_field="id",
        content_field="text",
    )

    # --- build search engine from benchmark records ---
    engine = SearchEngine.from_records(
        records=[
            {"id": r.doc_id, "text": r.content}
            for r in benchmark_index.records
        ],
        ranking=RobustRanking(),
    )

    # --- define benchmark queries ---
    queries = [
        BenchmarkQuery(query="sample", relevant_doc_ids=frozenset()),
        BenchmarkQuery(query="document", relevant_doc_ids=frozenset()),
    ]

    # --- run benchmark ---
    results = run_benchmark(
        engine=engine,
        index=benchmark_index,
        queries=queries,
        k=5,
        warmup=1,
        repeats=3,
        seed=123,
    )

    # --- assertions ---
    assert len(results) == len(queries)

    for result in results:
        assert result.latency_ms >= 0.0
        assert isinstance(result.retrieved, list)
        assert result.latency_stats is not None
        assert 50 in result.latency_stats
        assert 95 in result.latency_stats
