from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from scirelay.benchmark import build_index, load_dataset, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SciRelay benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--quick", action="store_true", help="Run a small query subset for smoke testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parents[1]
    data_dir = root / config["data_dir"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    dataset = load_dataset(data_dir)
    index = build_index(dataset.documents)
    frames: list[pd.DataFrame] = []
    n_agents_value = config["n_agents"]
    n_agents_values = n_agents_value if isinstance(n_agents_value, list) else [n_agents_value]
    for n_agents in n_agents_values:
        for scenario in config["scenarios"]:
            print(f"Running agents={n_agents}, scenario={scenario}", flush=True)
            frames.append(
                run_benchmark(
                    dataset=dataset,
                    index=index,
                    split=config["split"],
                    n_agents=int(n_agents),
                    budgets=[int(value) for value in config["budgets"]],
                    seeds=[int(value) for value in config["seeds"]],
                    bid_limit=int(config["bid_limit"]),
                    method_weights=config.get("method_weights"),
                    scenario=scenario,
                    query_limit=20 if args.quick else config.get("query_limit"),
                    include_ablations=bool(config.get("include_ablations", False)),
                )
            )
    results = pd.concat(frames, ignore_index=True)
    suffix = "quick" if args.quick else "full"
    result_path = output_dir / f"results_{suffix}.csv"
    results.to_csv(result_path, index=False)

    summary = (
        results.groupby(["n_agents", "scenario", "budget", "policy"], as_index=False)
        .agg(
            recall=("recall", "mean"),
            hit_rate=("hit_rate", "mean"),
            mrr=("mrr", "mean"),
            ndcg=("ndcg", "mean"),
            total_bytes=("total_bytes", "mean"),
        )
        .sort_values(
            ["n_agents", "scenario", "budget", "ndcg"],
            ascending=[True, True, True, False],
        )
    )
    summary.to_csv(output_dir / f"summary_{suffix}.csv", index=False)

    run_info = {
        "config": config,
        "quick": args.quick,
        "documents": len(dataset.documents),
        "queries_train": len(dataset.qrels_train),
        "queries_test": len(dataset.qrels_test),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
        "result_path": str(result_path.relative_to(root)),
    }
    (output_dir / f"run_info_{suffix}.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nResults written to: {result_path}")


if __name__ == "__main__":
    main()
