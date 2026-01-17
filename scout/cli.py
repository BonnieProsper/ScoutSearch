# scout/cli.py

import argparse
import json
from typing import List, Dict

from scout.search.engine import SearchEngine
from scout.index.tokens import Tokenizer
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.state.store import Store
from scout.explain import explain_query

def main():
    parser = argparse.ArgumentParser(description="ScoutSearch CLI")
    parser.add_argument("--records-file", type=str, help="JSON file with records")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--ranking", type=str, default="robust", choices=["robust", "bm25"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--load-index", action="store_true", help="Load persisted index")
    parser.add_argument("--ngram", type=int, default=1, help="Optional n-gram tokenization")
    args = parser.parse_args()

    # Load or build index
    if args.load_index:
        index = Store.load_index()
        tokenizer = Tokenizer(ngram=args.ngram)
    else:
        if not args.records_file:
            raise ValueError("Must provide --records-file when building a new index")
        with open(args.records_file, "r", encoding="utf-8") as f:
            records: List[Dict] = json.load(f)

        tokenizer = Tokenizer(ngram=args.ngram)
        ranking_strategy = RobustRanking() if args.ranking == "robust" else BM25Ranking()
        engine = SearchEngine.from_records(records, ngram=args.ngram, ranking=ranking_strategy)
        index = engine._index
        Store.save_index(index)

    ranking_strategy = RobustRanking() if args.ranking == "robust" else BM25Ranking()
    engine = SearchEngine(index=index, ranking=ranking_strategy, tokenizer=tokenizer)

    results = engine.search(args.query, limit=args.limit)
    print(f"Results for '{args.query}':")
    for doc_id, ranking_result in results:
        print(f"Doc {doc_id}: score={ranking_result.score:.4f}, components={ranking_result.components}")

    print("\nExplanations:")
    explanations = explain_query(engine, args.query, limit=args.limit)
    for doc_id, result in explanations:
        print(f"Doc {doc_id}: components={result.components}")

if __name__ == "__main__":
    main()
