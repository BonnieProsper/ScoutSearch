# scout/ingest.py

import csv
import json


def load_json(file_path: str) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def load_csv(file_path: str, id_field: str = "id", text_fields: list[str] | None = None,) -> list[dict]:
    records = []
    if text_fields is None:
        text_fields = ["text"]
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = { "id": row[id_field] }
            for field in text_fields:
                record[field] = row.get(field, "")
            records.append(record)
    return records
