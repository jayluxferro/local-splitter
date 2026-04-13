#!/usr/bin/env python3
"""Generate paper figures from eval summary data.

Reads both run summaries and produces averaged bar charts and a
cost-savings scatter plot as PDF files in paper/figures/.

Usage:
    uv run python scripts/gen_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data: averaged from two eval runs
# ---------------------------------------------------------------------------

# Run 1 savings (%)
R1 = {
    "T1":       {"wl1": 28.2, "wl2": 61.9, "wl3": 55.4, "wl4": 33.7},
    "T2":       {"wl1": 20.6, "wl2": 20.3, "wl3": -4.7, "wl4": 15.5},
    "T3":       {"wl1":  9.9, "wl2": -1.2, "wl3": -5.0, "wl4":  3.6},
    "T4":       {"wl1":-39.4, "wl2":-43.1, "wl3": 11.7, "wl4":-35.8},
    "T5":       {"wl1":  2.3, "wl2": -2.8, "wl3": -3.5, "wl4": 37.9},
    "T6":       {"wl1": -1.8, "wl2": -2.9, "wl3": -2.6, "wl4":  0.0},
    "T7":       {"wl1": -5.0, "wl2": 13.6, "wl3": -4.7, "wl4": 14.1},
    "T1+T3":    {"wl1": 34.1, "wl2": 65.6, "wl3": 56.8, "wl4": 34.8},
    "T1+T2":    {"wl1": 46.9, "wl2": 79.4, "wl3": 59.5, "wl4": 45.3},
    "T1+T2+T3": {"wl1": 42.4, "wl2": 79.1, "wl3": 60.6, "wl4": 45.6},
    "all":      {"wl1": 31.0, "wl2": 72.7, "wl3": 60.3, "wl4": 51.8},
}

# Run 2 savings (%) — T5-T7 filled after second clean pass
R2 = {
    "T1":       {"wl1": 30.3, "wl2": 75.8, "wl3": 62.4, "wl4": 42.4},
    "T2":       {"wl1": 24.1, "wl2": 18.3, "wl3": -0.5, "wl4": 22.3},
    "T3":       {"wl1":  9.3, "wl2": -0.8, "wl3": -2.6, "wl4":  1.1},
    "T4":       {"wl1":-30.7, "wl2":-37.8, "wl3": 13.5, "wl4":-26.4},
    "T5":       {"wl1":  7.8, "wl2": -4.0, "wl3": -5.2, "wl4": 40.7},
    "T6":       {"wl1": 11.7, "wl2": -8.1, "wl3":  3.1, "wl4": -3.3},
    "T7":       {"wl1":  2.4, "wl2": -0.9, "wl3":  1.3, "wl4": -0.1},
    "T1+T3":    {"wl1": 33.3, "wl2": 75.3, "wl3": 58.0, "wl4": 37.7},
    "T1+T2":    {"wl1": 43.1, "wl2": 78.6, "wl3": 55.2, "wl4": 43.4},
    "T1+T2+T3": {"wl1": 42.9, "wl2": 80.1, "wl3": 58.6, "wl4": 42.0},
    "all":      {"wl1": 27.9, "wl2": 70.5, "wl3": 58.0, "wl4": 50.4},
}

# Run 2 cost data ($) — includes T5-T7 from clean run
COST_R2 = {
    "wl1": {
        "baseline": 0.004254, "T1": 0.003084, "T2": 0.003671,
        "T3": 0.003912, "T4": 0.003715, "T5": 0.004220,
        "T6": 0.003950, "T7": 0.004149, "T1+T3": 0.003020,
        "T1+T2": 0.002881, "T1+T2+T3": 0.002896, "all": 0.002332,
    },
    "wl2": {
        "baseline": 0.005308, "T1": 0.001372, "T2": 0.005061,
        "T3": 0.005361, "T4": 0.004827, "T5": 0.005402,
        "T6": 0.005334, "T7": 0.004508, "T1+T3": 0.001405,
        "T1+T2": 0.001381, "T1+T2+T3": 0.001279, "all": 0.001009,
    },
    "wl3": {
        "baseline": 0.006988, "T1": 0.002620, "T2": 0.007024,
        "T3": 0.007171, "T4": 0.003453, "T5": 0.006974,
        "T6": 0.006912, "T7": 0.007056, "T1+T3": 0.002933,
        "T1+T2": 0.003120, "T1+T2+T3": 0.002889, "all": 0.001681,
    },
    "wl4": {
        "baseline": 0.005522, "T1": 0.003519, "T2": 0.006605,
        "T3": 0.005408, "T4": 0.005128, "T5": 0.005619,
        "T6": 0.005400, "T7": 0.004457, "T1+T3": 0.003988,
        "T1+T2": 0.005077, "T1+T2+T3": 0.005226, "all": 0.002567,
    },
}

WLS = ["wl1", "wl2", "wl3", "wl4"]
WL_LABELS = ["WL1\n(edit)", "WL2\n(explain)", "WL3\n(chat)", "WL4\n(RAG)"]

OUTDIR = Path("paper/figures")


def avg(r1: dict, r2: dict, subset: str, wl: str) -> float:
    return (r1[subset][wl] + r2[subset][wl]) / 2


def halfrange(r1: dict, r2: dict, subset: str, wl: str) -> float:
    return abs(r1[subset][wl] - r2[subset][wl]) / 2


# ---------------------------------------------------------------------------
# Figure 1: Singleton bar chart
# ---------------------------------------------------------------------------

def fig_singletons():
    subsets = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336",
              "#9C27B0", "#795548", "#607D8B"]
    x = np.arange(len(WLS))
    width = 0.11
    offset = (len(subsets) - 1) * width / 2

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (subset, color) in enumerate(zip(subsets, colors)):
        means = [avg(R1, R2, subset, wl) for wl in WLS]
        errs = [halfrange(R1, R2, subset, wl) for wl in WLS]
        ax.bar(x + i * width - offset, means, width, yerr=errs,
               label=subset, color=color, capsize=2, edgecolor="white",
               linewidth=0.5)

    ax.set_ylabel("Cloud token savings (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(WL_LABELS)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(loc="lower left", framealpha=0.9, ncol=4, fontsize=8)
    ax.set_title("Per-tactic savings in isolation")
    ax.set_ylim(-85, 85)
    fig.tight_layout()
    fig.savefig(OUTDIR / "singletons.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUTDIR / 'singletons.pdf'}")


# ---------------------------------------------------------------------------
# Figure 2: Combinations bar chart
# ---------------------------------------------------------------------------

def fig_combos():
    subsets = ["T1+T3", "T1+T2", "T1+T2+T3", "all"]
    colors = ["#7E57C2", "#2196F3", "#00BCD4", "#607D8B"]
    x = np.arange(len(WLS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (subset, color) in enumerate(zip(subsets, colors)):
        means = [avg(R1, R2, subset, wl) for wl in WLS]
        errs = [halfrange(R1, R2, subset, wl) for wl in WLS]
        bars = ax.bar(x + i * width, means, width, yerr=errs,
                      label=subset, color=color, capsize=3, edgecolor="white")

    ax.set_ylabel("Cloud token savings (%)")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(WL_LABELS)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Tactic combination savings")
    ax.set_ylim(0, 95)
    fig.tight_layout()
    fig.savefig(OUTDIR / "combos.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUTDIR / 'combos.pdf'}")


# ---------------------------------------------------------------------------
# Figure 3: Cost scatter — savings (%) vs dollar cost per workload
# ---------------------------------------------------------------------------

def fig_cost_scatter():
    subsets_ordered = ["T1", "T2", "T3", "T4", "T5", "T6", "T7",
                       "T1+T3", "T1+T2", "T1+T2+T3", "all"]
    markers = {"wl1": "o", "wl2": "s", "wl3": "^", "wl4": "D"}
    wl_colors = {"wl1": "#2196F3", "wl2": "#4CAF50", "wl3": "#FF9800", "wl4": "#F44336"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for wl, marker in markers.items():
        for subset in subsets_ordered:
            saving = avg(R1, R2, subset, wl)
            cost = COST_R2[wl][subset] * 1000  # to millicents for readability
            ax.scatter(saving, cost, marker=marker, color=wl_colors[wl],
                       s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

        # baseline reference
        base_cost = COST_R2[wl]["baseline"] * 1000
        ax.scatter(0, base_cost, marker=marker, color=wl_colors[wl],
                   s=100, alpha=0.4, edgecolors="black", linewidth=1.0)

    # Legend for workloads
    for wl, marker, label in zip(WLS, ["o", "s", "^", "D"], WL_LABELS):
        ax.scatter([], [], marker=marker, color=wl_colors[wl], s=60,
                   label=label.replace("\n", " "))

    ax.set_xlabel("Cloud token savings (%)")
    ax.set_ylabel("Cost per 10 samples ($×10³)")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Token savings vs. dollar cost")
    fig.tight_layout()
    fig.savefig(OUTDIR / "cost_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUTDIR / 'cost_scatter.pdf'}")


# ---------------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")
    fig_singletons()
    fig_combos()
    fig_cost_scatter()
    print("Done.")


if __name__ == "__main__":
    main()
