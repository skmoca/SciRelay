from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scirelay.benchmark import build_index, load_dataset, run_benchmark


BASE_WEIGHTS = {
    "rank": 0.35,
    "lexical": 0.29,
    "coverage": 0.16,
    "margin": 0.0,
    "diversity": 0.0,
}

WEIGHT_GRID = {
    "base": BASE_WEIGHTS,
    "rank_minus20": {"rank": 0.28, "lexical": 0.29, "coverage": 0.16, "margin": 0.0, "diversity": 0.0},
    "rank_plus20": {"rank": 0.42, "lexical": 0.29, "coverage": 0.16, "margin": 0.0, "diversity": 0.0},
    "lexical_minus20": {"rank": 0.35, "lexical": 0.232, "coverage": 0.16, "margin": 0.0, "diversity": 0.0},
    "lexical_plus20": {"rank": 0.35, "lexical": 0.348, "coverage": 0.16, "margin": 0.0, "diversity": 0.0},
    "coverage_minus20": {"rank": 0.35, "lexical": 0.29, "coverage": 0.128, "margin": 0.0, "diversity": 0.0},
    "coverage_plus20": {"rank": 0.35, "lexical": 0.29, "coverage": 0.192, "margin": 0.0, "diversity": 0.0},
}

REFERENCE_BASELINE = {
    "homogeneous": "raw_score",
    "heterogeneous": "rrf",
    "faulty": "zscore_score",
    "imbalanced": "raw_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SciRelay development sensitivity analysis.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sensitivity"))
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--part", choices=["all", "weights", "bid", "scale"], default="all")
    return parser.parse_args()


def bootstrap_difference(left: pd.Series, right: pd.Series, n_bootstrap: int = 10000) -> tuple[float, float, float]:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    difference = (joined["left"] - joined["right"]).to_numpy(dtype=float)
    rng = np.random.default_rng(20260810)
    samples = rng.integers(0, len(difference), size=(n_bootstrap, len(difference)))
    means = difference[samples].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(difference.mean()), float(low), float(high)


def add_meta(frame: pd.DataFrame, study: str, setting: str, bid_limit: int, n_agents: int) -> pd.DataFrame:
    result = frame.copy()
    result["sensitivity_study"] = study
    result["setting"] = setting
    result["sensitivity_bid_limit"] = bid_limit
    result["sensitivity_n_agents"] = n_agents
    return result


def run_weight_grid(dataset, index, query_limit: int | None) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for setting, weights in WEIGHT_GRID.items():
        for scenario in ["heterogeneous", "faulty"]:
            baseline = REFERENCE_BASELINE[scenario]
            frame = run_benchmark(
                dataset=dataset,
                index=index,
                split="test",
                n_agents=8,
                budgets=[4],
                seeds=[11, 47, 131],
                bid_limit=4,
                scenario=scenario,
                method_weights=weights,
                query_limit=query_limit,
                policies=["scirelay", baseline],
            )
            frames.append(add_meta(frame, "weight", setting, 4, 8))
    return frames


def run_bid_grid(dataset, index, query_limit: int | None) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for bid_limit in [2, 4, 8]:
        for scenario in ["heterogeneous", "faulty"]:
            baseline = REFERENCE_BASELINE[scenario]
            frame = run_benchmark(
                dataset=dataset,
                index=index,
                split="test",
                n_agents=8,
                budgets=[2, 4, 8],
                seeds=[11, 47, 131],
                bid_limit=bid_limit,
                scenario=scenario,
                method_weights=BASE_WEIGHTS,
                query_limit=query_limit,
                policies=["scirelay", baseline],
            )
            frames.append(add_meta(frame, "bid_limit", f"m={bid_limit}", bid_limit, 8))
    return frames


def run_scale_probe(dataset, index, query_limit: int | None) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for bid_limit in [2, 4, 8]:
        for scenario in ["heterogeneous", "faulty"]:
            baseline = REFERENCE_BASELINE[scenario]
            frame = run_benchmark(
                dataset=dataset,
                index=index,
                split="test",
                n_agents=16,
                budgets=[4],
                seeds=[11, 47, 131],
                bid_limit=bid_limit,
                scenario=scenario,
                method_weights=BASE_WEIGHTS,
                query_limit=query_limit,
                policies=["scirelay", baseline],
            )
            frames.append(add_meta(frame, "bid_limit_scale16", f"m={bid_limit}", bid_limit, 16))
    return frames


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, cell in data.groupby(
        ["sensitivity_study", "setting", "sensitivity_n_agents", "scenario", "budget", "sensitivity_bid_limit"]
    ):
        study, setting, n_agents, scenario, budget, bid_limit = keys
        baseline = REFERENCE_BASELINE[scenario]
        pair = cell[cell["policy"].isin(["scirelay", baseline])]
        query_means = pair.groupby(["query_id", "policy"])["ndcg"].mean().unstack("policy")
        delta, ci_low, ci_high = bootstrap_difference(query_means["scirelay"], query_means[baseline])
        means = cell.groupby("policy")["ndcg"].mean()
        records.append(
            {
                "sensitivity_study": study,
                "setting": setting,
                "n_agents": int(n_agents),
                "scenario": scenario,
                "budget": int(budget),
                "bid_limit": int(bid_limit),
                "reference_baseline": baseline,
                "scirelay_ndcg": float(means["scirelay"]),
                "baseline_ndcg": float(means[baseline]),
                "mean_difference": delta,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "queries": int(len(query_means)),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["sensitivity_study", "setting", "n_agents", "scenario", "budget", "bid_limit"]
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/main.yaml").read_text(encoding="utf-8"))
    dataset = load_dataset(root / config["data_dir"])
    index = build_index(dataset.documents)
    frames: list[pd.DataFrame] = []
    if args.part in {"all", "weights"}:
        frames.extend(run_weight_grid(dataset, index, args.query_limit))
    if args.part in {"all", "bid"}:
        frames.extend(run_bid_grid(dataset, index, args.query_limit))
    if args.part in {"all", "scale"}:
        frames.extend(run_scale_probe(dataset, index, args.query_limit))
    raw = pd.concat(frames, ignore_index=True)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "results_full.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(output_dir / "summary.csv", index=False)
    run_info = {
        "development_only": True,
        "part": args.part,
        "query_limit": args.query_limit,
        "weights": WEIGHT_GRID,
        "reference_baseline": REFERENCE_BASELINE,
        "seeds": [11, 47, 131],
        "weight_scenarios": ["heterogeneous", "faulty"],
        "bid_scenarios": ["heterogeneous", "faulty"],
        "scale_probe_agents": 16,
        "record_count": int(len(raw)),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
