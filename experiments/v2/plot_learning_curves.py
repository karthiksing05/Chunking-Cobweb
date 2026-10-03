"""Small-multiple learning curves for a run of run_synthetic.py.

Usage:
    python experiments/v2/plot_learning_curves.py experiments/v2/results/main
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_synthetic import V1_ENDPOINTS  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
PARSE, GEN = "#2a78d6", "#eb6834"   # categorical slots 1-2 (validated, light mode)
ORDER = ["small", "med", "large", "term_low", "term_med", "term_high"]
TITLES = {"small": "SMALL grammar", "med": "MED grammar", "large": "LARGE grammar",
          "term_low": "MED, 11-word lexicon", "term_med": "MED, 22-word lexicon",
          "term_high": "MED, 39-word lexicon"}


def main(run_dir: str) -> str:
    rows = json.load(open(os.path.join(run_dir, "results.json")))["rows"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for ax, cond in zip(axes.flat, ORDER):
        rs = [r for r in rows if r["condition"] == cond]
        ns = sorted({r["n_train"] for r in rs})
        ax.set_facecolor(SURFACE)
        for key, color in (("omission", PARSE), ("gen_commission", GEN)):
            vals = np.array([[100 * (1 - r[key]) for r in rs if r["n_train"] == n] for n in ns])
            mean, sd = vals.mean(axis=1), vals.std(axis=1)
            ax.fill_between(ns, np.clip(mean - sd, 0, 100), np.clip(mean + sd, 0, 100),
                            color=color, alpha=0.12, linewidth=0)
            ax.plot(ns, mean, color=color, linewidth=2, solid_capstyle="round",
                    solid_joinstyle="round", marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        v1_om, v1_cm = V1_ENDPOINTS[cond]
        for value, color in ((100 * (1 - v1_om), PARSE), (100 * (1 - v1_cm), GEN)):
            ax.plot([300], [value], marker="D", markersize=7, markerfacecolor=SURFACE,
                    markeredgecolor=color, markeredgewidth=2, linestyle="none", zorder=4)
        final = [r for r in rs if r["n_train"] == max(ns)]
        om = 100 * np.mean([r["omission"] for r in final])
        cm = 100 * np.mean([r["gen_commission"] for r in final])
        ax.set_title(TITLES[cond], loc="left", color=INK, fontsize=10.5, pad=18)
        ax.text(0, 1.02, f"v2 at {max(ns)}: omission {om:.1f}%, commission {cm:.1f}%   "
                         f"(v1: {100 * v1_om:.1f}%, {100 * v1_cm:.1f}%)",
                transform=ax.transAxes, color=INK2, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    for ax in axes[1]:
        ax.set_xlabel("Training sentences (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("1 − error rate (%)")
    handles = [
        Line2D([], [], color=PARSE, linewidth=2, marker="o", markersize=6,
               markeredgecolor=SURFACE, label="Parsing: 1 − omission (held-out brackets)"),
        Line2D([], [], color=GEN, linewidth=2, marker="o", markersize=6,
               markeredgecolor=SURFACE, label="Generation: 1 − commission (500 samples)"),
        Line2D([], [], color=INK2, marker="D", markersize=7, markerfacecolor=SURFACE,
               markeredgewidth=2, linestyle="none", label="v1 (paper), 300 sentences"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965), labelcolor=INK2, fontsize=9)
    fig.suptitle("TRELLIS v2 learning curves (mean ± 1σ over five seeds; same corpora and splits as v1)",
                 x=0.5, y=0.995, color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(run_dir, "learning_curves.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    return out


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "main")))
