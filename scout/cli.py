# scout/cli.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.ranking.composite import CompositeRanking
from scout.ranking.recency import RecencyRanking
from scout.explain import explain_query
from scout.benchmark import benchmark_engine
from scout.state.signals import IndexState
from scout.state.persistence import AutoSaver
from scout.data.loader import load_records


RANKINGS = {
    "robust": RobustRanking,
    "bm25": BM25Ranking,
    "fusion": lambda: CompositeRanking(
        strategies=[BM25Ranking(), RobustRanking()],
        weights=[0.5, 0.5],
        recency=RecencyRanking(decay_days=30.0, max_boost=1.0),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="ScoutSearch CLI")

    parser.add_argument(
        "--records-file",
        type=Path,
        required=True,
        help="Path to dataset (.json or .jsonl)",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optional cap on number of documents loaded",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory for persisted index",
    )
    parser.add_argument(
        "--ranking",
        choices=RANKINGS.keys(),
        default="robust",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode to add docs and query in real-time",
    )
    parser.add_argument(
        "--field-weights",
        type=str,
        default=None,
        help='Optional JSON dict of field weights, e.g. \'{"title":2,"text":1}\'',
    )

    args = parser.parse_args()

    records = list(
        load_records(
            args.records_file,
            limit=args.limit_docs,
        )
    )

    ranking = RANKINGS[args.ranking]()
    state = IndexState()

    if args.out is not None:
        AutoSaver(state) # add root later


    field_weights = json.loads(args.field_weights) if args.field_weights else None

    engine = SearchEngine.from_records(
        records,
        ranking=ranking,
        state=state,
        field_weights=field_weights,
    )

    if args.benchmark:
        queries_input = input("Enter benchmark queries, comma-separated: ")
        queries = [q.strip() for q in queries_input.split(",") if q.strip()]
        benchmark_engine(
            records=records,
            queries=queries,
            ranking_strategy=ranking,
        )
        return

    if args.interactive:
        print("Interactive mode: type 'exit' to quit, 'add <json>' to add doc")
        while True:
            cmd = input(">> ").strip()
            if cmd.lower() == "exit":
                break
            elif cmd.startswith("add "):
                try:
                    doc = json.loads(cmd[4:].strip())
                    engine.add_document(doc["id"], doc)
                    print(f"Added doc {doc['id']}.")
                    print("Interactive add is experimental; document will be indexed using default fields.")
                except Exception as e:
                    print(f"Error adding document: {e}")
            else:
                results = engine.search(cmd, limit=args.limit)
                for doc_id, r in results:
                    print(f"Doc {doc_id}: score={r.score:.4f}")
                    print(f"  components: {r.components}")
                    if r.per_term:
                        print(f"  per_term: {r.per_term}")
        return

    query = input("Enter search query: ") if not args.json else ""
    results = engine.search(query, limit=args.limit)

    if args.explain:
        results = explain_query(engine, query, limit=args.limit)

    if args.json:
        print(json.dumps(
            [
                {
                    "doc_id": doc_id,
                    "score": r.score,
                    "components": r.components,
                    "per_term": r.per_term,
                }
                for doc_id, r in results
            ],
            indent=2,
        ))
        return

    for doc_id, r in results:
        print(f"Doc {doc_id}: score={r.score:.4f}")
        print(f"  components: {r.components}")
        if r.per_term:
            print(f"  per_term: {r.per_term}")


if __name__ == "__main__":
    main()
