from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BASELINES = ["raw_score", "minmax_score", "zscore_score", "rrf", "round_robin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen dense-retrieval outputs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def bootstrap_difference(
    left: pd.Series, right: pd.Series, n_bootstrap: int, seed: int = 20260810
) -> tuple[float, float, float]:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    differences = (joined["left"] - joined["right"]).to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    means = differences[samples].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[position]))
        adjusted[position] = running
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, dtype={"query_id": str})
    aggregate = (
        data.groupby(["scenario", "budget", "policy"], as_index=False)
        .agg(
            recall=("recall", "mean"),
            hit_rate=("hit_rate", "mean"),
            mrr=("mrr", "mean"),
            ndcg=("ndcg", "mean"),
            payload_bytes=("payload_bytes", "mean"),
            control_bytes=("control_bytes", "mean"),
            total_bytes=("total_bytes", "mean"),
        )
    )
    aggregate.to_csv(args.output / "aggregate_metrics.csv", index=False)

    records: list[dict[str, object]] = []
    for (scenario, budget), cell in data.groupby(["scenario", "budget"]):
        means = cell.groupby("policy")["ndcg"].mean()
        baseline = max(BASELINES, key=lambda name: float(means.get(name, -np.inf)))
        by_query = (
            cell[cell["policy"].isin(["scirelay", baseline])]
            .groupby(["query_id", "policy"])["ndcg"]
            .mean()
            .unstack("policy")
        )
        difference, low, high = bootstrap_difference(
            by_query["scirelay"], by_query[baseline], args.bootstrap
        )
        paired = by_query["scirelay"] - by_query[baseline]
        try:
            p_value = float(wilcoxon(paired, zero_method="zsplit").pvalue)
        except ValueError:
            p_value = 1.0
        records.append(
            {
                "scenario": scenario,
                "budget": int(budget),
                "best_baseline": baseline,
                "scirelay_ndcg": float(means["scirelay"]),
                "baseline_ndcg": float(means[baseline]),
                "mean_difference": difference,
                "ci95_low": low,
                "ci95_high": high,
                "p_value": p_value,
            }
        )
    comparisons = pd.DataFrame.from_records(records).sort_values(["scenario", "budget"])
    comparisons["p_holm"] = holm_adjust(comparisons["p_value"].tolist())
    comparisons.to_csv(args.output / "paired_comparisons.csv", index=False)
    checks = {
        "model_scenarios_complete": {
            "passed": set(data["scenario"]) == {
                "dense_homogeneous",
                "lexical_dense_heterogeneous",
                "dense_faulty",
            },
            "scenarios": sorted(data["scenario"].unique().tolist()),
        },
        "all_budgets_complete": {
            "passed": set(data["budget"].astype(int)) == {2, 4, 8},
            "budgets": sorted(data["budget"].astype(int).unique().tolist()),
        },
        "finite_metrics": {
            "passed": bool(
                np.isfinite(data[["recall", "hit_rate", "mrr", "ndcg"]].to_numpy()).all()
            )
        },
    }
    (args.output / "claim_checks.json").write_text(
        json.dumps({"passed": all(item["passed"] for item in checks.values()), "checks": checks},
                   ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
