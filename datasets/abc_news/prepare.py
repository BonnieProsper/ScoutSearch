#!/usr/bin/env python3
"""
Prepare the ABC News Headlines dataset for indexing.

Input:
    CSV file with columns:
        - publish_date (yyyyMMdd)
        - headline_text

Output:
    JSONL file with records shaped for scout indexing:
        {
            "id": str,
            "text": str,
            "date": "yyyy-mm-dd",
            "source": "abc_news"
        }

Design goals:
- Deterministic output
- Streaming (no full file load)
- Explicit failures on bad input
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Dict


REQUIRED_COLUMNS = {"publish_date", "headline_text"}
SOURCE_NAME = "abc_news"


def _parse_date(raw: str) -> str:
    try:
        return datetime.strptime(raw, "%Y%m%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid publish_date: {raw!r}") from exc


def iter_records(csv_path: Path) -> Iterable[Dict]:
    """
    Stream records from the CSV and yield normalized dicts.
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns {REQUIRED_COLUMNS}, "
                f"found {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=1):
            text = row["headline_text"].strip()
            if not text:
                continue  # skip empty headlines silently

            date_iso = _parse_date(row["publish_date"])

            yield {
                "id": f"abc_{row['publish_date']}_{row_num:07d}",
                "text": text,
                "date": date_iso,
                "source": SOURCE_NAME,
            }


def write_jsonl(records: Iterable[Dict], out_path: Path) -> int:
    """
    Write records to a JSONL file.
    Returns number of records written.
    """
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        print(
            "Usage: prepare.py <input.csv> <output.jsonl>",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_path = Path(argv[1])
    out_path = Path(argv[2])

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    records = iter_records(csv_path)
    count = write_jsonl(records, out_path)

    print(f"Wrote {count:,} records to {out_path}")


if __name__ == "__main__":
    main(sys.argv)
