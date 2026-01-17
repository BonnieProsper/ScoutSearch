# scout/ingest.py

import json
import csv
from typing import List, Dict

def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv(file_path: str, id_field: str = "id", text_fields: List[str] = ["text"]) -> List[Dict]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = { "id": row[id_field] }
            for field in text_fields:
                record[field] = row.get(field, "")
            records.append(record)
    return records
