# scout/cli.py

import argparse
import json
from scout.search.engine import SearchEngine
from scout.ranking.robust import RobustRanking
from scout.ranking.bm25 import BM25Ranking
from scout.ranking.fusion import FusionRanking
from scout.explain import explain_query
from scout.benchmark import benchmark_engine
from scout.state.signals import IndexState
from scout.state.persistence import AutoSaver
from scout.ranking.composite import CompositeRanking
from scout.ranking.recency import RecencyRanking

RANKINGS = {
    "robust": RobustRanking,
    "bm25": BM25Ranking,
    "fusion": lambda: CompositeRanking(
        strategies=[BM25Ranking(), RobustRanking()],
        weights=[0.5, 0.5],
        recency=RecencyRanking(decay_days=30.0, max_boost=1.0)
    ),
    # Fusion example: combines BM25 + Robust
    "fusion": lambda: FusionRanking(
        strategies=[BM25Ranking(), RobustRanking()],
        weights=[0.5, 0.5]
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="ScoutSearch CLI")
    parser.add_argument("--records-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--ranking", choices=RANKINGS.keys(), default="robust")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()

    with open(args.records_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    ranking = RANKINGS[args.ranking]()

    state = IndexState()
    AutoSaver(state)

    engine = SearchEngine.from_records(
        records,
        ranking=ranking,
        state=state,
    )

    if args.benchmark:
        benchmark_engine(records, [args.query], ranking)
        return

    results = engine.search(args.query, limit=args.limit)

    if args.explain:
        results = explain_query(engine, args.query, limit=args.limit)

    if args.json:
        print(json.dumps([
            {
                "doc_id": doc_id,
                "score": r.score,
                "components": r.components,
                "per_term": r.per_term,
            }
            for doc_id, r in results
        ], indent=2))
        return

    for doc_id, r in results:
        print(f"Doc {doc_id}: score={r.score:.4f}")
        print(f"  components: {r.components}")
        if r.per_term:
            print(f"  per_term: {r.per_term}")


if __name__ == "__main__":
    main()
