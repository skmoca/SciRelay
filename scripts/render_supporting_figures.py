"""Render supporting figures for the SciRelay paper.

The script derives every plotted value from the submitted result records; it
does not hard-code query-level deltas.  It writes both PDF (paper) and PNG
(visual QA) versions using a restrained, vector-friendly Matplotlib style.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
RESULTS = ROOT / "results" / "results_full.csv"

COLORS = {
    "ink": "#26384D",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "control": "#486278",
    "payload": "#AFCDBB",
    "homogeneous": "#A8B5C6",
    "heterogeneous": "#6C86C7",
    "faulty": "#C9822B",
    "imbalanced": "#4F9C57",
}


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def save_pair(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG_DIR / f"{stem}_preview.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def communication_breakdown() -> None:
    usecols = ["split", "scenario", "n_agents", "bid_limit", "budget", "policy", "payload_bytes", "control_bytes"]
    df = pd.read_csv(RESULTS, usecols=usecols)
    df = df[
        (df["split"] == "test")
        & (df["n_agents"] == 8)
        & (df["bid_limit"] == 4)
        & (df["policy"] == "scirelay")
        & (df["budget"].isin([2, 4, 8]))
    ]
    grouped = df.groupby("budget", as_index=True)[["payload_bytes", "control_bytes"]].mean()
    budgets = np.array([2, 4, 8])
    if grouped.empty or not np.all(np.isin(budgets, grouped.index)):
        raise ValueError("Missing communication records for N=8, m=4, SciRelay, B in {2,4,8}")
    control = grouped.loc[budgets, "control_bytes"].to_numpy(dtype=float)
    payload = grouped.loc[budgets, "payload_bytes"].to_numpy(dtype=float)
    if not np.allclose(control, control[0]):
        raise ValueError("SciRelay control width is not fixed across relay budgets")
    total = control + payload
    # Keep the denominator identical to Table 4 and the surrounding prose:
    # control bytes divided by relayed payload bytes.
    percentages = control / payload * 100

    fig, ax = plt.subplots(figsize=(3.28, 1.38))
    x = np.arange(len(budgets))
    width = 0.56
    ax.bar(x, control, width, color=COLORS["control"], label=f"Fixed control ({control[0]:.0f} B)")
    ax.bar(x, payload, width, bottom=control, color=COLORS["payload"], label="Mean abstract payload")
    for i, (c, p, t, pct) in enumerate(zip(control, payload, total, percentages)):
        ax.text(i, c / 2, f"{c:,.0f}", ha="center", va="center", color="white", fontsize=7.5, weight="bold")
        ax.text(i, c + p / 2, f"{p:,.0f}", ha="center", va="center", color=COLORS["ink"], fontsize=7.5)
        ax.text(i, t + 260, f"{t:,.0f} ({pct:.1f}%)", ha="center", va="bottom", color=COLORS["ink"], fontsize=7.2)

    ax.set_xticks(x, ["B=2", "B=4", "B=8"])
    ax.set_xlabel("")
    ax.set_ylabel("Bytes per query")
    ax.set_ylim(0, 16000)
    ax.set_yticks([0, 4000, 8000, 12000, 16000])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["muted"])
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.92)
    save_pair(fig, "communication_breakdown")


def query_delta_distribution() -> None:
    usecols = ["scenario", "seed", "query_id", "n_agents", "bid_limit", "budget", "policy", "ndcg"]
    df = pd.read_csv(RESULTS, usecols=usecols)
    df = df[(df["n_agents"] == 8) & (df["bid_limit"] == 4) & (df["budget"] == 4)]
    baseline = {
        "homogeneous": "raw_score",
        "heterogeneous": "rrf",
        "faulty": "zscore_score",
        "imbalanced": "raw_score",
    }
    order = ["homogeneous", "heterogeneous", "faulty", "imbalanced"]
    labels = ["Homogeneous", "Heterogeneous", "Score fault", "Imbalanced"]
    values: list[np.ndarray] = []
    for scenario in order:
        pivot = (
            df[df["scenario"] == scenario]
            .pivot_table(index="query_id", columns="policy", values="ndcg", aggfunc="mean")
        )
        values.append((pivot["scirelay"] - pivot[baseline[scenario]]).to_numpy())

    fig, ax = plt.subplots(figsize=(3.28, 1.86))
    positions = np.arange(1, len(values) + 1)
    violin = ax.violinplot(values, positions=positions, widths=0.76, showmeans=False, showextrema=False, showmedians=False)
    for body, scenario in zip(violin["bodies"], order):
        body.set_facecolor(COLORS[scenario])
        body.set_edgecolor(COLORS[scenario])
        body.set_alpha(0.30)
        body.set_linewidth(0.6)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.2},
        whiskerprops={"color": COLORS["muted"], "linewidth": 0.8},
        capprops={"color": COLORS["muted"], "linewidth": 0.8},
        boxprops={"edgecolor": COLORS["ink"], "linewidth": 0.8},
    )
    for patch, scenario in zip(box["boxes"], order):
        patch.set_facecolor(COLORS[scenario])
        patch.set_alpha(0.78)

    for i, (scenario, arr) in enumerate(zip(order, values), start=1):
        q3 = float(np.quantile(arr, 0.75))
        med = float(np.median(arr))
        y = max(q3 + 0.025, 0.035)
        ax.text(i, y, f"med {med:+.3f}", ha="center", va="bottom", fontsize=6.7, color=COLORS["ink"])

    ax.axhline(0, color=COLORS["ink"], linewidth=0.8, zorder=0)
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelrotation=18)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel(r"$\Delta$ nDCG at $B=4$")
    ax.set_xlabel("")
    ax.set_ylim(-0.36, 0.42)
    ax.set_yticks([-0.3, -0.1, 0.0, 0.1, 0.3])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["muted"])
    ax.tick_params(axis="x", pad=3)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.28, top=0.98)
    save_pair(fig, "query_delta_distribution")


if __name__ == "__main__":
    paper_style()
    communication_breakdown()
    query_delta_distribution()
