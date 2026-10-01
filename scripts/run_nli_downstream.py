from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scirelay.benchmark import _read_jsonl, load_dataset


LABELS = {"CONTRADICT": 0, "SUPPORT": 1}
MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
COMPARATORS = {
    "dense_homogeneous": "raw_score",
    "lexical_dense_heterogeneous": "zscore_score",
    "dense_faulty": "zscore_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate relayed evidence with frozen NLI")
    parser.add_argument("--data", type=Path, default=Path("data/raw/scifact-beir"))
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path("results/dense/confirmation/results_full.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/downstream/nli"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reuse-pair-scores",
        action="store_true",
        help="Reuse a complete pair_scores.csv instead of repeating NLI inference",
    )
    return parser.parse_args()


def query_evidence_labels(path: Path) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    query_labels: dict[str, int] = {}
    evidence_labels: dict[str, dict[str, int]] = {}
    for row in _read_jsonl(path):
        query_id = str(row["_id"])
        per_document: dict[str, int] = {}
        for document_id, annotations in row.get("metadata", {}).items():
            values = {LABELS[item["label"]] for item in annotations if item.get("label") in LABELS}
            if len(values) > 1:
                raise ValueError(f"Conflicting evidence labels for query {query_id}")
            if values:
                per_document[str(document_id)] = next(iter(values))
        unique = set(per_document.values())
        if len(unique) > 1:
            raise ValueError(f"Conflicting query labels for query {query_id}")
        if unique:
            query_labels[query_id] = next(iter(unique))
            evidence_labels[query_id] = per_document
    return query_labels, evidence_labels


def bootstrap_difference(
    left: np.ndarray, right: np.ndarray, count: int, seed: int = 20260810
) -> tuple[float, float, float]:
    difference = left.astype(float) - right.astype(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(difference), size=(count, len(difference)))
    means = difference[sampled].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(difference.mean()), float(low), float(high)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dataset = load_dataset(args.data)
    document_by_id = {document.doc_id: document for document in dataset.documents}
    query_labels, evidence_labels = query_evidence_labels(args.data / "queries.jsonl")
    confirmation_queries = sorted(set(dataset.qrels_train) & set(query_labels), key=int)

    selections = pd.read_csv(args.selections, dtype={"query_id": str})
    allowed = {
        scenario: {"scirelay", comparator} for scenario, comparator in COMPARATORS.items()
    }
    selections = selections[
        (selections["budget"] == args.budget)
        & selections["query_id"].isin(confirmation_queries)
        & selections.apply(
            lambda row: row["scenario"] in allowed
            and row["policy"] in allowed[row["scenario"]],
            axis=1,
        )
    ].copy()

    pairs: set[tuple[str, str]] = set()
    for row in selections.itertuples(index=False):
        pairs.update(
            (row.query_id, document_id)
            for document_id in str(row.selected_doc_ids).split("|")
            if document_id
        )
    for query_id in confirmation_queries:
        pairs.update((query_id, document_id) for document_id in evidence_labels[query_id])
    ordered_pairs = sorted(pairs, key=lambda item: (int(item[0]), int(item[1])))

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    device = torch.device(args.device)
    score_path = args.output / "pair_scores.csv"
    if args.reuse_pair_scores:
        score_frame = pd.read_csv(
            score_path, dtype={"query_id": str, "document_id": str}
        )
        actual_pairs = set(zip(score_frame["query_id"], score_frame["document_id"]))
        if actual_pairs != pairs:
            raise ValueError("Existing pair_scores.csv does not cover the frozen pair set")
        print(f"Reused {len(score_frame)} frozen NLI pair scores", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, revision=args.revision, dtype=dtype
        )
        model.eval().to(device)
        label2id = {
            str(name).lower(): int(value) for name, value in model.config.label2id.items()
        }
        entailment_id = label2id["entailment"]
        contradiction_id = label2id["contradiction"]
        neutral_id = label2id["neutral"]
        score_records: list[dict[str, object]] = []
        for start in range(0, len(ordered_pairs), args.batch_size):
            batch = ordered_pairs[start : start + args.batch_size]
            premises = [document_by_id[document_id].full_text for _, document_id in batch]
            hypotheses = [dataset.queries[query_id].text for query_id, _ in batch]
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits.float()
                probabilities = torch.softmax(logits, dim=1)
            logits = logits.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            for (query_id, document_id), row_logits, row_probabilities in zip(
                batch, logits, probabilities
            ):
                score_records.append(
                    {
                        "query_id": query_id,
                        "document_id": document_id,
                        "entailment_logit": float(row_logits[entailment_id]),
                        "contradiction_logit": float(row_logits[contradiction_id]),
                        "neutral_probability": float(row_probabilities[neutral_id]),
                        "nonneutral_confidence": float(
                            max(
                                row_probabilities[entailment_id],
                                row_probabilities[contradiction_id],
                            )
                        ),
                        "support_margin": float(
                            row_logits[entailment_id] - row_logits[contradiction_id]
                        ),
                    }
                )
            if start == 0 or (start // args.batch_size + 1) % 100 == 0:
                print(
                    f"NLI pairs {min(start + args.batch_size, len(ordered_pairs))}/{len(ordered_pairs)}",
                    flush=True,
                )
        score_frame = pd.DataFrame.from_records(score_records)
        score_frame.to_csv(score_path, index=False)
    score_lookup = {
        (row.query_id, row.document_id): (row.support_margin, row.nonneutral_confidence)
        for row in score_frame.itertuples(index=False)
    }

    seed_records: list[dict[str, object]] = []
    for row in selections.itertuples(index=False):
        selected = [value for value in str(row.selected_doc_ids).split("|") if value]
        scored = [(document_id, *score_lookup[(row.query_id, document_id)]) for document_id in selected]
        chosen_document, margin, confidence = max(scored, key=lambda item: item[2])
        gold = query_labels[row.query_id]
        seed_records.append(
            {
                "query_id": row.query_id,
                "scenario": row.scenario,
                "seed": int(row.seed),
                "budget": int(row.budget),
                "policy": row.policy,
                "gold_label": gold,
                "chosen_document": chosen_document,
                "support_margin": margin,
                "nonneutral_confidence": confidence,
                "prediction": int(margin >= 0.0),
                "correct": int((margin >= 0.0) == bool(gold)),
                "gold_evidence_hit": int(
                    bool(set(selected) & dataset.qrels_train.get(row.query_id, set()))
                ),
            }
        )
    seed_frame = pd.DataFrame.from_records(seed_records)
    seed_frame.to_csv(args.output / "per_seed.csv", index=False)
    query_frame = (
        seed_frame.groupby(
            ["query_id", "scenario", "budget", "policy", "gold_label"], as_index=False
        )
        .agg(
            support_margin=("support_margin", "mean"),
            nonneutral_confidence=("nonneutral_confidence", "mean"),
            evidence_hit_rate=("gold_evidence_hit", "mean"),
        )
    )
    query_frame["prediction"] = (query_frame["support_margin"] >= 0.0).astype(int)
    query_frame["correct"] = (
        query_frame["prediction"] == query_frame["gold_label"]
    ).astype(int)
    query_frame.to_csv(args.output / "per_query.csv", index=False)

    summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for (scenario, budget, policy), cell in query_frame.groupby(
        ["scenario", "budget", "policy"]
    ):
        summaries.append(
            {
                "scenario": scenario,
                "budget": int(budget),
                "policy": policy,
                "queries": len(cell),
                "accuracy": accuracy_score(cell["gold_label"], cell["prediction"]),
                "balanced_accuracy": balanced_accuracy_score(
                    cell["gold_label"], cell["prediction"]
                ),
                "macro_f1": f1_score(
                    cell["gold_label"], cell["prediction"], average="macro"
                ),
                "evidence_hit_rate": float(cell["evidence_hit_rate"].mean()),
            }
        )
    summary = pd.DataFrame.from_records(summaries).sort_values(["scenario", "policy"])
    summary.to_csv(args.output / "summary.csv", index=False)
    for scenario, comparator in COMPARATORS.items():
        cell = query_frame[
            (query_frame["scenario"] == scenario)
            & query_frame["policy"].isin(["scirelay", comparator])
        ]
        paired = cell.pivot(index="query_id", columns="policy", values="correct")
        difference, low, high = bootstrap_difference(
            paired["scirelay"].to_numpy(), paired[comparator].to_numpy(), args.bootstrap
        )
        comparisons.append(
            {
                "scenario": scenario,
                "budget": args.budget,
                "comparator": comparator,
                "scirelay_accuracy": float(paired["scirelay"].mean()),
                "baseline_accuracy": float(paired[comparator].mean()),
                "accuracy_difference": difference,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    comparison_frame = pd.DataFrame.from_records(comparisons)
    comparison_frame.to_csv(args.output / "paired_comparisons.csv", index=False)

    oracle_records: list[tuple[int, int]] = []
    for query_id in confirmation_queries:
        scored = [
            (document_id, *score_lookup[(query_id, document_id)])
            for document_id in evidence_labels[query_id]
        ]
        _, margin, _ = max(scored, key=lambda item: item[2])
        oracle_records.append((query_labels[query_id], int(margin >= 0.0)))
    oracle_true = [value[0] for value in oracle_records]
    oracle_predicted = [value[1] for value in oracle_records]
    run_info = {
        "evaluation_split": "SciFact conventional train / 809-query held-out confirmation",
        "labeled_queries": len(confirmation_queries),
        "model_id": args.model,
        "model_revision": args.revision,
        "model_training_data": "MNLI, FEVER, and ANLI; no SciFact fine-tuning",
        "input_order": "premise=full title and abstract; hypothesis=claim",
        "aggregation": (
            "within each relay set choose the document with maximum entailment-or-contradiction "
            "probability; average its entailment-minus-contradiction margin across shard seeds"
        ),
        "budget": args.budget,
        "pair_count": len(score_frame),
        "gold_evidence_oracle": {
            "accuracy": accuracy_score(oracle_true, oracle_predicted),
            "balanced_accuracy": balanced_accuracy_score(oracle_true, oracle_predicted),
            "macro_f1": f1_score(oracle_true, oracle_predicted, average="macro"),
        },
        "device": str(device),
        "dtype": str(dtype),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checks = {
        "complete_query_count": bool(len(confirmation_queries) == 505),
        "gold_oracle_above_chance": bool(
            run_info["gold_evidence_oracle"]["balanced_accuracy"] > 0.60
        ),
        "all_scenarios_present": bool(set(summary["scenario"]) == set(COMPARATORS)),
    }
    (args.output / "claim_checks.json").write_text(
        json.dumps({"passed": all(checks.values()), "checks": checks}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(comparison_frame.to_string(index=False))
    print(json.dumps(run_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
