from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scirelay.benchmark import (
    build_index,
    load_dataset,
    make_shards,
    retrieval_metrics,
    select_minmax_score,
    select_raw_score,
    select_round_robin,
    select_rrf,
    select_scirelay,
    select_zscore_score,
    tokenize,
)
from scirelay.dense import create_mixed_bids, load_dense_index


POLICIES = {
    "raw_score": select_raw_score,
    "minmax_score": select_minmax_score,
    "zscore_score": select_zscore_score,
    "rrf": select_rrf,
    "round_robin": select_round_robin,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen SPECTER SciRelay studies")
    parser.add_argument("--config", type=Path, default=Path("configs/dense_development.yaml"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parents[1]
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    dataset = load_dataset(root / config["data_dir"])
    lexical_index = build_index(dataset.documents)
    dense_index = load_dense_index(root / config["embedding_dir"], dataset.documents)
    qrels = dataset.qrels_test if config["split"] == "test" else dataset.qrels_train
    query_ids = sorted(qrels, key=int)
    if args.quick:
        query_ids = query_ids[:20]
    elif config.get("query_limit") is not None:
        query_ids = query_ids[: int(config["query_limit"])]

    dense_query_positions = dense_index.query_position
    dense_rows = np.asarray(
        dense_index.query_matrix[[dense_query_positions[query_id] for query_id in query_ids]],
        dtype=np.float32,
    )
    dense_scores = dense_rows @ np.asarray(dense_index.document_matrix, dtype=np.float32).T
    word_queries = lexical_index.word_vectorizer.transform(
        [dataset.queries[query_id].text for query_id in query_ids]
    )
    word_scores = (word_queries @ lexical_index.word_matrix.T).toarray().astype(np.float32)
    document_terms = [frozenset(tokenize(document.full_text)) for document in dataset.documents]
    query_terms = {
        query_id: frozenset(tokenize(dataset.queries[query_id].text)) for query_id in query_ids
    }
    weights = {
        "rank_weight": float(config["method_weights"]["rank"]),
        "lexical_weight": float(config["method_weights"]["lexical"]),
        "coverage_weight": float(config["method_weights"]["coverage"]),
        "margin_weight": float(config["method_weights"].get("margin", 0.0)),
        "diversity_weight": float(config["method_weights"].get("diversity", 0.0)),
    }
    records: list[dict[str, object]] = []
    for seed in [int(value) for value in config["seeds"]]:
        shards = make_shards(dataset.documents, int(config["n_agents"]), seed)
        for scenario in config["scenarios"]:
            print(f"Running seed={seed}, scenario={scenario}", flush=True)
            for query_position, query_id in enumerate(query_ids):
                query = dataset.queries[query_id]
                bids = create_mixed_bids(
                    query=query,
                    documents=dataset.documents,
                    shards=shards,
                    word_scores=word_scores[query_position],
                    dense_scores=dense_scores[query_position],
                    document_terms=document_terms,
                    query_terms=query_terms[query_id],
                    bid_limit=int(config["bid_limit"]),
                    scenario=str(scenario),
                    seed=seed,
                )
                for budget in [int(value) for value in config["budgets"]]:
                    selected_by_policy = {
                        name: selector(bids, budget) for name, selector in POLICIES.items()
                    }
                    selected_by_policy["scirelay"] = select_scirelay(
                        bids, budget, **weights
                    )
                    for policy, selected in selected_by_policy.items():
                        doc_ids = [bid.doc_id for bid in selected]
                        metrics = retrieval_metrics(doc_ids, qrels[query_id])
                        payload_bytes = sum(
                            len(dataset.documents[bid.doc_index].full_text.encode("utf-8"))
                            for bid in selected
                        )
                        control_width = {
                            "raw_score": 12,
                            "minmax_score": 10,
                            "zscore_score": 10,
                            "rrf": 9,
                        }.get(policy, 20 if policy == "scirelay" else 0)
                        control_bytes = len(bids) * control_width
                        records.append(
                            {
                                "split": config["split"],
                                "scenario": scenario,
                                "seed": seed,
                                "query_id": query_id,
                                "n_agents": int(config["n_agents"]),
                                "bid_limit": int(config["bid_limit"]),
                                "budget": budget,
                                "policy": policy,
                                "selected_doc_ids": "|".join(doc_ids),
                                "payload_bytes": payload_bytes,
                                "control_bytes": control_bytes,
                                "total_bytes": payload_bytes + control_bytes,
                                **metrics,
                            }
                        )
    frame = pd.DataFrame.from_records(records)
    suffix = "quick" if args.quick else "full"
    result_path = output / f"results_{suffix}.csv"
    frame.to_csv(result_path, index=False)
    summary = (
        frame.groupby(["scenario", "budget", "policy"], as_index=False)
        .agg(
            recall=("recall", "mean"),
            hit_rate=("hit_rate", "mean"),
            mrr=("mrr", "mean"),
            ndcg=("ndcg", "mean"),
            total_bytes=("total_bytes", "mean"),
        )
        .sort_values(["scenario", "budget", "ndcg"], ascending=[True, True, False])
    )
    summary.to_csv(output / f"summary_{suffix}.csv", index=False)
    run_info = {
        "config": config,
        "quick": args.quick,
        "model_id": dense_index.model_id,
        "model_revision": dense_index.revision,
        "pooling": dense_index.pooling,
        "document_format": dense_index.document_format,
        "documents": len(dataset.documents),
        "queries": len(query_ids),
        "records": len(frame),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
        "result_path": str(result_path.relative_to(root)),
    }
    (output / f"run_info_{suffix}.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Results written to: {result_path}")


if __name__ == "__main__":
    main()
