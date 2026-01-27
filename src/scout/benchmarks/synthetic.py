# scout/benchmarks/synthetic.py
from __future__ import annotations

import random

from scout.benchmarks.index import BenchmarkIndex, BenchmarkRecord
from scout.benchmarks.run import BenchmarkQuery


def generate_synthetic_index(
    *,
    name: str,
    num_docs: int,
) -> BenchmarkIndex:
    """
    Generate a deterministic synthetic benchmark corpus.
    """
    records = []
    for i in range(num_docs):
        records.append(
            BenchmarkRecord(
                doc_id=str(i),
                content=f"document {i} about topic {i % 10}",
                metadata={"topic": i % 10},
            )
        )

    return BenchmarkIndex(
        name=name,
        records=tuple(records),
    )


def generate_synthetic_queries(
    *,
    index: BenchmarkIndex,
    num_queries: int,
) -> list[BenchmarkQuery]:
    """
    Generate synthetic queries with relevance judgments.
    """
    queries: list[BenchmarkQuery] = []
    rng = random.Random(42)

    for _ in range(num_queries):
        topic = rng.randint(0, 9)
        relevant = {
            r.doc_id
            for r in index.records
            if r.metadata.get("topic") == topic
        }

        queries.append(
            BenchmarkQuery(
                query=f"topic {topic}",
                relevant_doc_ids=frozenset(relevant),
            )
        )

    return queries
