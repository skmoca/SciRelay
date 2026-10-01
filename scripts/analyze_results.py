from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


BASELINES = ["raw_score", "minmax_score", "zscore_score", "rrf", "round_robin"]
DISPLAY = {
    "scirelay": "SciRelay",
    "raw_score": "Raw score",
    "minmax_score": "Local Min--Max",
    "zscore_score": "Local z-score",
    "rrf": "RRF",
    "round_robin": "Round robin",
    "centralized_word": "Centralized word",
}
COLORS = {
    "scirelay": "#8C2D3E",
    "raw_score": "#3E5964",
    "minmax_score": "#527A8C",
    "zscore_score": "#456E52",
    "rrf": "#71818A",
    "round_robin": "#AEB4B7",
    "centralized_word": "#3A3A3A",
}
SCENARIO_LABELS = {
    "homogeneous": "Homogeneous",
    "heterogeneous": "Heterogeneous",
    "faulty": "Score-inflated fault",
    "imbalanced": "Shard imbalance",
}
MARKERS = {
    "scirelay": "o",
    "raw_score": "s",
    "minmax_score": "v",
    "zscore_score": "X",
    "rrf": "^",
    "round_robin": "D",
    "centralized_word": "P",
}
LINESTYLES = {
    "scirelay": "-",
    "raw_score": "-",
    "minmax_score": "--",
    "zscore_score": "-.",
    "rrf": "--",
    "round_robin": ":",
    "centralized_word": "-.",
}


def set_publication_style() -> None:
    """Use a restrained, print-safe style shared by all exported figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SciRelay per-query outputs")
    parser.add_argument("--input", type=Path, default=Path("results/results_full.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def bootstrap_difference(
    left: pd.Series, right: pd.Series, n_bootstrap: int, seed: int = 20260810
) -> tuple[float, float, float]:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    difference = (joined["left"] - joined["right"]).to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(difference), size=(n_bootstrap, len(difference)))
    means = difference[samples].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(difference.mean()), float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, position in enumerate(order):
        value = min(1.0, (m - rank) * p_values[position])
        running = max(running, value)
        adjusted[position] = running
    return adjusted.tolist()


def comparison_table(data: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    records: list[dict] = []
    for (n_agents, scenario, budget), cell in data.groupby(
        ["n_agents", "scenario", "budget"]
    ):
        means = cell.groupby("policy")["ndcg"].mean()
        available = [name for name in BASELINES if name in means]
        best_baseline = max(available, key=lambda name: means[name])
        query_means = (
            cell[cell["policy"].isin(["scirelay", best_baseline])]
            .groupby(["query_id", "policy"])["ndcg"]
            .mean()
            .unstack("policy")
        )
        mean_diff, ci_low, ci_high = bootstrap_difference(
            query_means["scirelay"], query_means[best_baseline], n_bootstrap
        )
        difference = query_means["scirelay"] - query_means[best_baseline]
        try:
            p_value = float(wilcoxon(difference, zero_method="zsplit").pvalue)
        except ValueError:
            p_value = 1.0
        records.append(
            {
                "n_agents": n_agents,
                "scenario": scenario,
                "budget": budget,
                "best_baseline": best_baseline,
                "scirelay_ndcg": means["scirelay"],
                "baseline_ndcg": means[best_baseline],
                "mean_difference": mean_diff,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "relative_gain_percent": 100.0 * mean_diff / max(means[best_baseline], 1e-12),
                "p_value": p_value,
            }
        )
    result = pd.DataFrame.from_records(records)
    result["p_holm"] = holm_adjust(result["p_value"].tolist())
    return result


def plot_main(data: pd.DataFrame, output: Path) -> None:
    subset = data[data["n_agents"] == 8]
    scenarios = [
        name
        for name in ["homogeneous", "heterogeneous", "faulty", "imbalanced"]
        if name in set(subset["scenario"])
    ]
    policies = ["scirelay", "raw_score", "zscore_score", "rrf", "round_robin", "centralized_word"]
    summary = subset.groupby(["scenario", "budget", "policy"])["ndcg"].mean()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.05, 2.08), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for axis, scenario in zip(axes, scenarios):
        budgets = sorted(subset["budget"].unique())
        x = np.arange(len(budgets))
        for policy in policies:
            values = [summary.get((scenario, budget, policy), np.nan) for budget in budgets]
            axis.plot(
                x,
                values,
                marker=MARKERS[policy],
                linestyle=LINESTYLES[policy],
                linewidth=1.75 if policy == "scirelay" else 0.95,
                markersize=3.7 if policy == "scirelay" else 3.0,
                markeredgewidth=0.45,
                markeredgecolor="white",
                label=DISPLAY[policy],
                color=COLORS[policy],
            )
        axis.set_title(SCENARIO_LABELS.get(scenario, scenario), pad=4)
        axis.set_xticks(x, budgets)
        axis.set_xlabel("Payload budget $B$", labelpad=2)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.set_ylim(0, 0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("nDCG", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.05), handlelength=2.0, columnspacing=1.1)
    fig.tight_layout(rect=[0, 0, 1, 0.91], w_pad=0.7)
    fig.savefig(output / "main_ndcg.pdf", bbox_inches="tight")
    fig.savefig(output / "main_ndcg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gain_heatmap(comparisons: pd.DataFrame, output: Path) -> None:
    cells = comparisons[comparisons["n_agents"] == 8].copy()
    pivot = cells.pivot(index="scenario", columns="budget", values="mean_difference")
    preferred = ["homogeneous", "heterogeneous", "faulty", "imbalanced"]
    pivot = pivot.reindex([name for name in preferred if name in pivot.index])
    fig, axis = plt.subplots(figsize=(3.65, 2.15))
    muted_diverging = LinearSegmentedColormap.from_list(
        "muted_diverging", ["#557F9C", "#F4F1EA", "#8C2D3E"]
    )
    image = axis.imshow(
        pivot.to_numpy(),
        cmap=muted_diverging,
        norm=TwoSlopeNorm(vmin=-0.02, vcenter=0.0, vmax=0.23),
        aspect="auto",
    )
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), [SCENARIO_LABELS.get(name, name) for name in pivot.index])
    axis.set_xlabel("Payload budget $B$", labelpad=2)
    axis.set_title("$\u0394$ nDCG: SciRelay $-$ best decentralized", pad=5)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=7.4)
    cbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    cbar.set_label("Paired mean $\u0394$", labelpad=3)
    cbar.ax.tick_params(labelsize=6.8, width=0.45, length=2.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.55)
    fig.tight_layout(pad=0.6)
    fig.savefig(output / "gain_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output / "gain_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost(data: pd.DataFrame, output: Path) -> None:
    subset = data[
        (data["n_agents"] == 8)
        & (data["budget"] == 4)
        & data["scenario"].isin(["heterogeneous", "faulty"])
        & data["policy"].isin(["scirelay", "raw_score", "zscore_score", "rrf", "round_robin"])
    ]
    summary = subset.groupby(["scenario", "policy"], as_index=False).agg(
        ndcg=("ndcg", "mean"), total_bytes=("total_bytes", "mean")
    )
    scenarios = [name for name in ["heterogeneous", "faulty"] if name in set(summary["scenario"])]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.9, 2.1), sharex=True, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for axis, scenario in zip(axes, scenarios):
        cell = summary[summary["scenario"] == scenario]
        for row in cell.itertuples(index=False):
            axis.scatter(
                row.total_bytes / 1024.0,
                row.ndcg,
                s=38 if row.policy != "scirelay" else 48,
                color=COLORS[row.policy],
                marker=MARKERS[row.policy],
                edgecolor="white",
                linewidth=0.45,
                label=DISPLAY[row.policy],
            )
        axis.set_title(SCENARIO_LABELS.get(scenario, scenario), pad=4)
        axis.set_xlabel("Mean transmitted KiB/query", labelpad=2)
        axis.grid(color="#D8D8D8", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("nDCG", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.06), handlelength=1.0, columnspacing=1.0)
    fig.suptitle("Communication--quality trade-off ($B=4$, $N=8$)",
                 fontsize=8.5, y=1.16)
    fig.tight_layout(rect=[0, 0, 1, 0.86], w_pad=0.8)
    fig.savefig(output / "cost_quality.pdf", bbox_inches="tight")
    fig.savefig(output / "cost_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def query_level_diagnostics(
    data: pd.DataFrame, comparisons: pd.DataFrame
) -> pd.DataFrame:
    """Summarize seed-averaged query wins, ties, losses, rescues, and regressions."""
    records: list[dict[str, object]] = []
    tolerance = 1e-12
    for comparison in comparisons.itertuples(index=False):
        cell = data[
            (data["n_agents"] == comparison.n_agents)
            & (data["scenario"] == comparison.scenario)
            & (data["budget"] == comparison.budget)
            & data["policy"].isin(["scirelay", comparison.best_baseline])
        ]
        by_query = (
            cell.groupby(["query_id", "policy"])["ndcg"].mean().unstack("policy")
        )
        delta = by_query["scirelay"] - by_query[comparison.best_baseline]
        records.append(
            {
                "n_agents": comparison.n_agents,
                "scenario": comparison.scenario,
                "budget": comparison.budget,
                "best_baseline": comparison.best_baseline,
                "queries": len(delta),
                "mean_difference": delta.mean(),
                "median_difference": delta.median(),
                "q25_difference": delta.quantile(0.25),
                "q75_difference": delta.quantile(0.75),
                "wins": int((delta > tolerance).sum()),
                "ties": int((delta.abs() <= tolerance).sum()),
                "losses": int((delta < -tolerance).sum()),
                "rescued_from_zero": int(
                    (
                        (by_query[comparison.best_baseline] <= tolerance)
                        & (by_query["scirelay"] > tolerance)
                    ).sum()
                ),
                "lost_to_zero": int(
                    (
                        (by_query[comparison.best_baseline] > tolerance)
                        & (by_query["scirelay"] <= tolerance)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    set_publication_style()
    data = pd.read_csv(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = (
        data.groupby(["n_agents", "scenario", "budget", "policy"], as_index=False)
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
    summary.to_csv(args.output / "aggregate_metrics.csv", index=False)
    comparisons = comparison_table(data, args.bootstrap)
    comparisons.to_csv(args.output / "paired_comparisons.csv", index=False)
    diagnostics = query_level_diagnostics(data, comparisons)
    diagnostics.to_csv(args.output / "query_level_diagnostics.csv", index=False)

    scirelay = data[data["policy"] == "scirelay"]
    overhead_ratio = float(scirelay["control_bytes"].sum() / scirelay["payload_bytes"].sum())
    target = comparisons[
        comparisons["scenario"].isin(["heterogeneous", "faulty"])
        & (comparisons["n_agents"] == 8)
    ]
    claim_checks = {
        "C1": {
            "description": "SciRelay beats the strongest decentralized baseline at every budget in the eight-agent heterogeneous and faulty scenarios.",
            "passed": bool((target["mean_difference"] > 0).all()),
            "minimum_mean_difference": float(target["mean_difference"].min()),
        },
        "C2": {
            "description": "Every paired bootstrap 95% interval for those comparisons has a positive lower bound.",
            "passed": bool((target["ci95_low"] > 0).all()),
            "minimum_ci95_low": float(target["ci95_low"].min()),
        },
        "C3": {
            "description": "SciRelay bid-sketch bytes are below 25% of relayed payload bytes.",
            "passed": overhead_ratio < 0.25,
            "control_to_payload_ratio": overhead_ratio,
        },
    }
    (args.output / "claim_checks.json").write_text(
        json.dumps({"claims": claim_checks}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_main(data, args.output)
    plot_gain_heatmap(comparisons, args.output)
    plot_cost(data, args.output)
    print(comparisons.to_string(index=False))
    print(json.dumps(claim_checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
