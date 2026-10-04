"""Small multiples for a run of run_incremental.py: learning by day and by
night versus batch learning, per condition.

Usage:
    python experiments/v2/plot_incremental.py experiments/v2/results/incremental
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
from plot_learning_curves import AXIS, GRID, INK, INK2, MUTED, ORDER, SURFACE, TITLES  # noqa: E402

INCR, BATCH = "#2a78d6", "#eb6834"   # categorical slots 1-2 (validated, light mode)
# Batch is dashed with square markers and drawn on top, so that where the two
# learners coincide both remain visible.
STYLE = {"incremental": dict(color=INCR, linestyle="-", marker="o", zorder=3),
         "batch": dict(color=BATCH, linestyle=(0, (4, 3)), marker="s", zorder=4)}
ROWS = [("gen_commission", "Generation: 1 − commission (%)", lambda v: 100 * (1 - v)),
        ("test_bits_per_sentence", "Held-out bits per sentence", lambda v: v),
        ("night_seconds", "Seconds per consolidation", lambda v: v)]


def main(run_dir: str) -> str:
    rows = json.load(open(os.path.join(run_dir, "results.json")))
    seeds = sorted({r["seed"] for r in rows})
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(ROWS), len(ORDER), figsize=(17, 9.5), sharex=True,
                             facecolor=SURFACE)
    for col, cond in enumerate(ORDER):
        for row, (key, label, f) in enumerate(ROWS):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            for mode in ("incremental", "batch"):
                color = STYLE[mode]["color"]
                rs = [r for r in rows if r["condition"] == cond and r["mode"] == mode]
                ns = sorted({r["sentences"] for r in rs})
                if not ns:
                    continue
                vals = [[f(r[key]) for r in rs if r["sentences"] == n] for n in ns]
                mean = np.array([np.mean(v) for v in vals])
                lo = np.array([np.min(v) for v in vals])
                hi = np.array([np.max(v) for v in vals])
                ax.fill_between(ns, lo, hi, color=color, alpha=0.12, linewidth=0)
                ax.plot(ns, mean, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        **STYLE[mode])
            if row == 0:
                ax.set_title(TITLES[cond], loc="left", color=INK, fontsize=10.5)
                ax.set_ylim(0, 102)
                ax.set_yticks([0, 25, 50, 75, 100])
            if row == 2:
                ax.set_yscale("log")
                ax.set_xlabel("Training sentences (log scale)")
            if col == 0:
                ax.set_ylabel(label)
            ax.set_xscale("log", base=2)
            ax.set_xticks([10, 20, 40, 80, 160, 320])
            ax.set_xticklabels(["10", "20", "40", "80", "160", "320"])
            ax.minorticks_off()
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)
    handles = [Line2D([], [], linewidth=2, markersize=6, markeredgecolor=SURFACE, label=t,
                      **{k: v for k, v in STYLE[m].items() if k != "zorder"})
               for m, t in (("incremental", "By day and by night: perceive each sentence, "
                                            "sleep at each checkpoint"),
                            ("batch", "Batch: a fresh learner sleeps once over the same sentences"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.965), labelcolor=INK2, fontsize=9.5)
    fig.suptitle(f"Unsupervised TRELLIS v2: incremental versus batch learning "
                 f"(mean over seeds {', '.join(map(str, seeds))}; band = range)",
                 x=0.5, y=0.995, color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(run_dir, "incremental_vs_batch.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "incremental")))
