# scout/cli.py

import argparse
from typing import List, Dict
from scout.search.engine import SearchEngine
from scout.index.tokens import Tokenizer
from scout.ranking.robust import RobustRanking
from scout.index.builder import IndexBuilder

def run_cli():
    parser = argparse.ArgumentParser(
        description="ScoutSearch: Lightweight, high-quality search engine CLI"
    )

    parser.add_argument(
        "--records-file",
        type=str,
        required=True,
        help="Path to JSON file containing documents (list of dicts with 'id' and text fields)"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query string to search for"
    )

    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=None,
        help="Document fields to index (default: all text fields)"
    )

    parser.add_argument(
        "--ngram",
        type=int,
        default=None,
        help="Optional n-gram tokenization"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to return"
    )

    args = parser.parse_args()

    # Load records
    import json
    with open(args.records_file, "r", encoding="utf-8") as f:
        records: List[Dict] = json.load(f)

    # Initialize engine
    ranking = RobustRanking()  # Composite TF + TF-IDF ranking
    engine = SearchEngine.from_records(
        records=records,
        fields=args.fields,
        ngram=args.ngram,
        ranking=ranking
    )

    # Execute search
    results = engine.search(args.query, limit=args.limit)

    if not results:
        print("No results found.")
        return

    print(f"Top {len(results)} results for query: '{args.query}'\n")
    for rank, (doc_id, r) in enumerate(results, start=1):
        print(f"{rank}. Doc ID: {doc_id}, Score: {r.score:.4f}, Components: {r.components}")

if __name__ == "__main__":
    run_cli()
