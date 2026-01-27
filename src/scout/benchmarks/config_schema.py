# scout/benchmarks/config_schema.py
from __future__ import annotations

from pydantic import BaseModel, Field


class IndexConfig(BaseModel):
    dataset_path: str
    id_field: str
    content_field: str
    metadata_fields: list[str] | None = None
    limit: int | None = None


class RankingConfig(BaseModel):
    type: str
    params: dict[str, float] = Field(default_factory=dict)
    recency: dict[str, float] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    k: int
    warmup: int = 0
    repeats: int = 1
    seed: int | None = None


class QueryConfig(BaseModel):
    query: str
    relevant_doc_ids: list[str]


class FullBenchmarkConfig(BaseModel):
    name: str
    index: IndexConfig
    ranking: RankingConfig
    benchmark: BenchmarkConfig
    queries: list[QueryConfig]
    output: str
