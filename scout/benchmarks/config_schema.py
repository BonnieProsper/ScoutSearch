# scout/benchmarks/config_schema.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class IndexConfig(BaseModel):
    dataset_path: str
    id_field: str
    content_field: str
    metadata_fields: Optional[List[str]] = None
    limit: Optional[int] = None


class RankingConfig(BaseModel):
    type: str
    params: Dict[str, float] = Field(default_factory=dict)
    recency: Dict[str, float] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    k: int
    warmup: int = 0
    repeats: int = 1
    seed: Optional[int] = None


class QueryConfig(BaseModel):
    query: str
    relevant_doc_ids: List[str]


class FullBenchmarkConfig(BaseModel):
    name: str
    index: IndexConfig
    ranking: RankingConfig
    benchmark: BenchmarkConfig
    queries: List[QueryConfig]
    output: str
