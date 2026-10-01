from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def cell_values(frame: pd.DataFrame, scenario: str, column: str) -> list[float]:
    subset = frame[frame["scenario"] == scenario].sort_values("budget")
    if len(subset) != 3:
        raise AssertionError(f"{scenario} should have 3 budget cells, found {len(subset)}")
    return subset[column].astype(float).tolist()


def within(value: float, target: float, tolerance: float = 0.0006) -> bool:
    return abs(value - target) <= tolerance


def main() -> int:
    checks: dict[str, dict[str, object]] = {}

    main_pairs = paired("results/analysis/paired_comparisons.csv")
    confirmation = paired("results/confirmation/analysis/paired_comparisons.csv")
    scaling = paired("results/scaling/analysis/paired_comparisons.csv")
    external = paired("results/trec_covid/analysis/paired_comparisons.csv")
    ablation = pd.read_csv(ROOT / "results/ablation/component_drops.csv")
    query_diagnostics = pd.read_csv(
        ROOT / "results/analysis/query_level_diagnostics.csv"
    )
    weight_sensitivity = paired("results/sensitivity/weights/summary.csv")
    bid_sensitivity = paired("results/sensitivity/bid/summary.csv")
    scale_sensitivity = paired("results/sensitivity/scale/summary.csv")
    dense_development = paired(
        "results/dense/development/analysis/paired_comparisons.csv"
    )
    dense_confirmation = paired(
        "results/dense/confirmation/analysis/paired_comparisons.csv"
    )
    downstream_pairs = paired("results/downstream/nli/paired_comparisons.csv")
    downstream_summary = paired("results/downstream/nli/summary.csv")
    downstream_info = json.loads(
        (ROOT / "results/downstream/nli/run_info.json").read_text(encoding="utf-8")
    )

    main_hetero = cell_values(main_pairs, "heterogeneous", "mean_difference")
    main_faulty = cell_values(main_pairs, "faulty", "mean_difference")
    main_target = main_pairs[main_pairs["scenario"].isin(["heterogeneous", "faulty"])]
    checks["scifact_development"] = {
        "passed": (
            within(min(main_hetero), 0.038)
            and within(max(main_hetero), 0.049)
            and within(min(main_faulty), 0.108)
            and within(max(main_faulty), 0.150)
            and float(main_target["ci95_low"].min()) > 0
        ),
        "heterogeneous_range": [min(main_hetero), max(main_hetero)],
        "faulty_range": [min(main_faulty), max(main_faulty)],
        "minimum_ci95_low": float(main_target["ci95_low"].min()),
    }

    homogeneous = main_pairs[main_pairs["scenario"] == "homogeneous"]
    checks["scifact_homogeneous_boundary"] = {
        "passed": bool(
            (homogeneous["ci95_low"] < 0).all()
            and (homogeneous["ci95_high"] > 0).all()
        ),
        "differences": homogeneous.sort_values("budget")["mean_difference"].tolist(),
    }

    conf_hetero = cell_values(confirmation, "heterogeneous", "mean_difference")
    conf_faulty = cell_values(confirmation, "faulty", "mean_difference")
    checks["scifact_confirmation"] = {
        "passed": (
            within(min(conf_hetero), 0.042)
            and within(max(conf_hetero), 0.052)
            and within(min(conf_faulty), 0.125)
            and within(max(conf_faulty), 0.183)
            and float(confirmation["ci95_low"].min()) > 0
            and float(confirmation["p_holm"].max()) < 0.001
        ),
        "heterogeneous_range": [min(conf_hetero), max(conf_hetero)],
        "faulty_range": [min(conf_faulty), max(conf_faulty)],
        "maximum_holm_p": float(confirmation["p_holm"].max()),
    }

    ext_hetero = external[external["scenario"] == "heterogeneous"].sort_values("budget")
    ext_faulty = external[external["scenario"] == "faulty"].sort_values("budget")
    checks["trec_covid_transfer"] = {
        "passed": bool(
            (ext_hetero["mean_difference"] > 0).all()
            and (ext_hetero["ci95_low"] < 0).all()
            and within(float(ext_faulty["mean_difference"].min()), 0.087)
            and within(float(ext_faulty["mean_difference"].max()), 0.140)
            and float(ext_faulty["ci95_low"].min()) > 0.06
            and float(ext_faulty["p_holm"].max()) < 0.001
        ),
        "heterogeneous_differences": ext_hetero["mean_difference"].tolist(),
        "faulty_range": [
            float(ext_faulty["mean_difference"].min()),
            float(ext_faulty["mean_difference"].max()),
        ],
        "minimum_faulty_ci95_low": float(ext_faulty["ci95_low"].min()),
    }

    scale_target = scaling[scaling["scenario"] == "faulty"]
    scale_16_hetero = scaling[
        (scaling["scenario"] == "heterogeneous") & (scaling["n_agents"] == 16)
    ]
    checks["scaling_boundary"] = {
        "passed": bool(
            (scale_target["mean_difference"] > 0).all()
            and (scale_target["ci95_low"] > 0).all()
            and (scale_16_hetero["ci95_low"] < 0).all()
            and (scale_16_hetero["ci95_high"] > 0).all()
        ),
        "faulty_gain_range": [
            float(scale_target["mean_difference"].min()),
            float(scale_target["mean_difference"].max()),
        ],
    }

    drops = ablation.set_index("variant")["mean_drop"].to_dict()
    checks["ablation"] = {
        "passed": bool(
            within(float(drops["scirelay_no_coverage"]), 0.261)
            and within(float(drops["scirelay_no_calibration"]), 0.044)
            and within(float(drops["scirelay_with_raw_margin"]), 0.007)
            and within(float(drops["scirelay_with_diversity"]), 0.002)
        ),
        "mean_drops": drops,
    }

    diagnostic_b4 = query_diagnostics[
        (query_diagnostics["n_agents"] == 8)
        & (query_diagnostics["budget"] == 4)
    ].set_index("scenario")
    checks["query_level_diagnostics"] = {
        "passed": bool(
            int(diagnostic_b4.loc["faulty", "wins"]) == 175
            and int(diagnostic_b4.loc["faulty", "ties"]) == 66
            and int(diagnostic_b4.loc["faulty", "losses"]) == 59
            and within(
                float(diagnostic_b4.loc["faulty", "median_difference"]), 0.099
            )
            and int(diagnostic_b4.loc["heterogeneous", "wins"]) == 111
            and int(diagnostic_b4.loc["heterogeneous", "ties"]) == 112
            and int(diagnostic_b4.loc["heterogeneous", "losses"]) == 77
            and int(diagnostic_b4.loc["homogeneous", "losses"])
            > int(diagnostic_b4.loc["homogeneous", "wins"])
        ),
        "budget_4": diagnostic_b4[
            [
                "mean_difference",
                "median_difference",
                "wins",
                "ties",
                "losses",
                "rescued_from_zero",
                "lost_to_zero",
            ]
        ].to_dict(orient="index"),
    }

    relative20 = weight_sensitivity[
        weight_sensitivity["setting"].isin(
            [
                "rank_minus20",
                "rank_plus20",
                "lexical_minus20",
                "lexical_plus20",
                "coverage_minus20",
                "coverage_plus20",
            ]
        )
    ]
    relative20_heterogeneous = relative20[relative20["scenario"] == "heterogeneous"]
    relative20_faulty = relative20[relative20["scenario"] == "faulty"]
    checks["weight_sensitivity"] = {
        "passed": bool(
            len(relative20) == 12
            and relative20_heterogeneous["mean_difference"].between(0.062, 0.070).all()
            and relative20_faulty["mean_difference"].between(0.131, 0.138).all()
            and float(relative20["ci95_low"].min()) > 0
        ),
        "heterogeneous_range": [
            float(relative20_heterogeneous["mean_difference"].min()),
            float(relative20_heterogeneous["mean_difference"].max()),
        ],
        "faulty_range": [
            float(relative20_faulty["mean_difference"].min()),
            float(relative20_faulty["mean_difference"].max()),
        ],
    }

    checks["bid_limit_sensitivity"] = {
        "passed": bool(
            len(bid_sensitivity) == 18
            and (bid_sensitivity["ci95_low"] > 0).all()
            and set(bid_sensitivity["bid_limit"].astype(int)) == {2, 4, 8}
            and len(scale_sensitivity) == 6
            and (scale_sensitivity["ci95_low"] > 0).all()
        ),
        "eight_agent_cells": int(len(bid_sensitivity)),
        "sixteen_agent_cells": int(len(scale_sensitivity)),
        "minimum_ci95_low": float(
            min(bid_sensitivity["ci95_low"].min(), scale_sensitivity["ci95_low"].min())
        ),
    }

    dense_dev_homogeneous = cell_values(
        dense_development, "dense_homogeneous", "mean_difference"
    )
    dense_dev_heterogeneous = cell_values(
        dense_development, "lexical_dense_heterogeneous", "mean_difference"
    )
    dense_dev_faulty = cell_values(dense_development, "dense_faulty", "mean_difference")
    checks["dense_development"] = {
        "passed": bool(
            min(dense_dev_homogeneous) >= 0.077
            and max(dense_dev_homogeneous) <= 0.111
            and min(dense_dev_heterogeneous) >= 0.166
            and max(dense_dev_heterogeneous) <= 0.243
            and min(dense_dev_faulty) >= 0.197
            and max(dense_dev_faulty) <= 0.271
            and (dense_development["ci95_low"] > 0).all()
        ),
        "homogeneous_range": [min(dense_dev_homogeneous), max(dense_dev_homogeneous)],
        "lexical_dense_range": [min(dense_dev_heterogeneous), max(dense_dev_heterogeneous)],
        "faulty_range": [min(dense_dev_faulty), max(dense_dev_faulty)],
        "minimum_ci95_low": float(dense_development["ci95_low"].min()),
    }

    dense_conf_homogeneous = cell_values(
        dense_confirmation, "dense_homogeneous", "mean_difference"
    )
    dense_conf_heterogeneous = cell_values(
        dense_confirmation, "lexical_dense_heterogeneous", "mean_difference"
    )
    dense_conf_faulty = cell_values(dense_confirmation, "dense_faulty", "mean_difference")
    checks["dense_confirmation"] = {
        "passed": bool(
            min(dense_conf_homogeneous) >= 0.058
            and max(dense_conf_homogeneous) <= 0.076
            and min(dense_conf_heterogeneous) >= 0.163
            and max(dense_conf_heterogeneous) <= 0.224
            and min(dense_conf_faulty) >= 0.176
            and max(dense_conf_faulty) <= 0.239
            and (dense_confirmation["ci95_low"] > 0).all()
        ),
        "homogeneous_range": [min(dense_conf_homogeneous), max(dense_conf_homogeneous)],
        "lexical_dense_range": [min(dense_conf_heterogeneous), max(dense_conf_heterogeneous)],
        "faulty_range": [min(dense_conf_faulty), max(dense_conf_faulty)],
        "minimum_ci95_low": float(dense_confirmation["ci95_low"].min()),
    }

    downstream_by_scenario = downstream_pairs.set_index("scenario")
    downstream_summary_indexed = downstream_summary.set_index(["scenario", "policy"])
    fault_downstream = downstream_by_scenario.loc["dense_faulty"]
    homogeneous_downstream = downstream_by_scenario.loc["dense_homogeneous"]
    heterogeneous_downstream = downstream_by_scenario.loc[
        "lexical_dense_heterogeneous"
    ]
    checks["downstream_verification"] = {
        "passed": bool(
            int(downstream_summary["queries"].min()) == 505
            and within(float(fault_downstream["accuracy_difference"]), 0.038)
            and float(fault_downstream["ci95_low"]) > 0
            and float(homogeneous_downstream["ci95_low"]) < 0
            and float(homogeneous_downstream["ci95_high"]) > 0
            and float(heterogeneous_downstream["ci95_low"]) < 0
            and float(heterogeneous_downstream["ci95_high"]) > 0
            and within(
                float(downstream_info["gold_evidence_oracle"]["accuracy"]), 0.836
            )
            and within(
                float(
                    downstream_summary_indexed.loc[
                        ("lexical_dense_heterogeneous", "scirelay"),
                        "evidence_hit_rate",
                    ]
                ),
                0.782,
            )
        ),
        "accuracy_differences": downstream_by_scenario[
            "accuracy_difference"
        ].to_dict(),
        "fault_ci95": [
            float(fault_downstream["ci95_low"]),
            float(fault_downstream["ci95_high"]),
        ],
        "gold_evidence_oracle": downstream_info["gold_evidence_oracle"],
    }

    expected_rows = {
        "results/results_full.csv": 144_000,
        "results/scaling/results_full.csv": 216_000,
        "results/ablation/results_full.csv": 64_800,
        "results/confirmation/results_full.csv": 116_496,
        "results/trec_covid/results_full.csv": 24_000,
        "results/dense/development/results_full.csv": 81_000,
        "results/dense/confirmation/results_full.csv": 131_058,
        "results/downstream/nli/pair_scores.csv": 8_953,
        "results/downstream/nli/per_query.csv": 3_030,
        "results/downstream/nli/per_seed.csv": 9_090,
    }
    actual_rows = {
        relative: sum(1 for _ in (ROOT / relative).open("r", encoding="utf-8")) - 1
        for relative in expected_rows
    }
    checks["per_query_record_counts"] = {
        "passed": actual_rows == expected_rows,
        "expected": expected_rows,
        "actual": actual_rows,
    }

    manifest_path = ROOT / "configs/research_checksums.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected_digest in manifest.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected_digest:
            mismatches.append(
                {"path": relative, "expected": expected_digest, "actual": actual}
            )
    checks["sha256_manifest"] = {
        "passed": not mismatches,
        "checked_files": len(manifest),
        "mismatches": mismatches,
    }

    passed = all(bool(item["passed"]) for item in checks.values())
    report = {"passed": passed, "checks": checks}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
