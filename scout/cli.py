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
from scout.state.signals import IndexState
from scout.state.persistence import AutoSaver
from scout.data.loader import load_records

# ---- Benchmark imports (top-level for static analysis) ----
from scout.benchmarks.index import build_benchmark_index
from scout.benchmarks.run import run_benchmark, BenchmarkQuery
from scout.benchmarks.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
)

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

    # Benchmark entrypoint
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        help="Run a benchmark defined by a JSON config file",
    )

    args = parser.parse_args()

    # ---------- Load records ----------
    records = list(
        load_records(
            args.records_file,
            limit=args.limit_docs,
        )
    )

    ranking = RANKINGS[args.ranking]()
    state = IndexState()

    if args.out is not None:
        AutoSaver(state)

    field_weights = json.loads(args.field_weights) if args.field_weights else None

    engine = SearchEngine.from_records(
        records,
        ranking=ranking,
        state=state,
        field_weights=field_weights,
    )

    # ---------- BENCHMARK MODE ----------
    if args.benchmark_config:
        config = json.loads(args.benchmark_config.read_text())

        index = build_benchmark_index(
            name=config["name"],
            dataset_path=Path(config["dataset"]),
            id_field=config["id_field"],
            content_field=config["content_field"],
            metadata_fields=config.get("metadata_fields"),
            limit=config.get("limit_docs"),
        )

        queries = [
            BenchmarkQuery(
                query=q["query"],
                relevant_doc_ids=frozenset(q["relevant_doc_ids"]),
            )
            for q in config["queries"]
        ]

        results = run_benchmark(
            engine=engine,
            index=index,
            queries=queries,
            k=config["k"],
        )

        for q, r in zip(queries, results):
            print(f"\nQuery: {q.query}")
            print(
                f"  P@{config['k']}: "
                f"{precision_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=config['k']):.3f}"
            )
            print(
                f"  R@{config['k']}: "
                f"{recall_at_k(retrieved=r.retrieved, relevant=q.relevant_doc_ids, k=config['k']):.3f}"
            )
            print(
                "  MRR: "
                f"{mean_reciprocal_rank(retrieved=r.retrieved, relevant=q.relevant_doc_ids):.3f}"
            )
            print(f"  Latency: {r.latency_ms:.2f} ms")

        return

    # ---------- INTERACTIVE MODE ----------
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

    # ---------- NORMAL QUERY ----------
    query = input("Enter search query: ") if not args.json else ""
    results = engine.search(query, limit=args.limit)

    if args.explain:
        results = explain_query(engine, query, limit=args.limit)

    if args.json:
        print(
            json.dumps(
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
            )
        )
        return

    for doc_id, r in results:
        print(f"Doc {doc_id}: score={r.score:.4f}")
        print(f"  components: {r.components}")
        if r.per_term:
            print(f"  per_term: {r.per_term}")


if __name__ == "__main__":
    main()
