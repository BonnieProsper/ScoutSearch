# scout/cli.py

import argparse
import json
from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.explain import explain_query

RANKINGS = {
    "robust": RobustRanking(),
    "bm25": BM25Ranking(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="ScoutSearch CLI")

    parser.add_argument("--records-file", type=str, help="JSON file with records")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--ranking", choices=RANKINGS.keys(), default="robust")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()

    if not args.records_file:
        raise ValueError("Must provide --records-file when building a new index")

    with open(args.records_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    engine = SearchEngine.from_records(
        records,
        ranking=RANKINGS[args.ranking],
    )

    results = engine.search(args.query, limit=args.limit)
    explanations = explain_query(engine, args.query, limit=args.limit)

    if args.json:
        output = {
            "query": args.query,
            "results": [
                {"doc_id": doc_id, "score": r.score, "components": r.components}
                for doc_id, r in results
            ],
            "explanations": [
                {"doc_id": doc_id, "components": r.components}
                for doc_id, r in explanations
            ],
        }
        print(json.dumps(output, indent=2))
        return

    print(f"Results for '{args.query}':")
    for doc_id, r in results:
        print(f"Doc {doc_id}: score={r.score:.4f}, components={r.components}")

    print("\nExplanations:")
    for doc_id, r in explanations:
        print(f"Doc {doc_id}: components={r.components}")


if __name__ == "__main__":
    main()
