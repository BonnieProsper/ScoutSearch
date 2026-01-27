# scout/benchmarks/index.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scout.data.loader import load_records


@dataclass(frozen=True)
class BenchmarkRecord:
    """
    Canonical record used for benchmarking.

    Intentionally minimal, immutable, and deterministic.
    """
    doc_id: str
    content: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class BenchmarkIndex:
    """
    Deterministic snapshot of a benchmark corpus.
    """
    name: str
    records: tuple[BenchmarkRecord, ...]

    def __len__(self) -> int:
        return len(self.records)
    
    def doc_ids(self) -> set[str]:
        return {r.doc_id for r in self.records}




def build_benchmark_index(
    *,
    name: str,
    dataset_path: str | Path,
    id_field: str,
    content_field: str,
    metadata_fields: Sequence[str] | None = None,
    limit: int | None = None,
) -> BenchmarkIndex:
    """
    Build a deterministic benchmark index from a dataset.

    Guarantees:
    - Stable ordering (input order)
    - Unique document IDs
    - Explicit field copying
    """
    raw_records = load_records(dataset_path, limit=limit)

    seen_ids: set[str] = set()
    records: list[BenchmarkRecord] = []

    for raw in raw_records:
        try:
            doc_id = str(raw[id_field])
            content = str(raw[content_field])
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}") from e

        if doc_id in seen_ids:
            raise ValueError(f"Duplicate document id detected: {doc_id}")
        seen_ids.add(doc_id)

        metadata: dict[str, object] = {}
        if metadata_fields:
            for field in metadata_fields:
                if field in raw:
                    metadata[field] = raw[field]

        records.append(
            BenchmarkRecord(
                doc_id=doc_id,
                content=content,
                metadata=metadata,
            )
        )

    return BenchmarkIndex(
        name=name,
        records=tuple(records),  # frozen for determinism
    )

# TO ADD
"""
if not records:
    raise ValueError("Benchmark index is empty")
"""
