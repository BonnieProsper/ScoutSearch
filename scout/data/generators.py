# scout/data/generators.py
from pathlib import Path
import json
import random
import uuid
from typing import List, Dict, Optional


def generate_synthetic_records(
    num_records: int = 50000,
    vocab: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Generate synthetic records for stress-testing.
    Each record has:
      - id: unique string
      - title: random words
      - body: random words
    """
    if vocab is None:
        vocab = [f"word{i}" for i in range(1000)]

    records: List[Dict] = []
    for _ in range(num_records):
        title = " ".join(random.choices(vocab, k=5))
        body = " ".join(random.choices(vocab, k=20))
        records.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "body": body
        })
    return records


def save_records(records: List[Dict], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def generate_synthetic_queries(records: List[Dict], num_queries: int = 1000) -> List[Dict]:
    queries: List[Dict] = []
    for _ in range(num_queries):
        record = random.choice(records)
        words = record["title"].split()
        query_text = " ".join(random.sample(words, min(3, len(words))))
        queries.append({
            "query": query_text,
            "relevant_doc_ids": [record["id"]]
        })
    return queries
