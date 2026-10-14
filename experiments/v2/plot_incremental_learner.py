"""Learning curves of the continuously incremental learner on the synthetic
corpora: the total code along the stream (bits per sentence) and generation
commission, one panel per condition, both seeds; the night learner's final
values as reference marks.

Usage:
    python experiments/v2/plot_incremental_learner.py RESULTS_JSON NIGHT_JSON OUT_PNG
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, LINE, NIGHT = "#1f2328", "#6e7781", "#2f6fdd", "#c4432b"


def main(results, night, out):
    rows = json.load(open(results))
    nights = {(r["condition"], r["seed"]): r["final"] for r in json.load(open(night))}
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    fig, axes = plt.subplots(2, len(conds), figsize=(2.6 * len(conds), 5.2), squeeze=False)
    for j, cond in enumerate(conds):
        for r in rows:
            if r["condition"] != cond:
                continue
            xs = [p["sentences"] for p in r["curve"]]
            style = "-" if r["seed"] == 13 else "--"
            axes[0, j].plot(xs, [p["total_bits"] / p["sentences"] for p in r["curve"]], style, color=LINE, lw=2,
                            label=f"seed {r['seed']}")
            axes[1, j].plot(xs, [100 * p["gen_commission"] for p in r["curve"]], style, color=LINE, lw=2)
            n = nights.get((cond, r["seed"]))
            if n:
                axes[0, j].plot([320], [n["total_bits"] / 320], "o", color=NIGHT, ms=7, mfc="none" if r["seed"] == 17 else NIGHT)
                axes[1, j].plot([320], [100 * n["gen_commission"]], "o", color=NIGHT, ms=7, mfc="none" if r["seed"] == 17 else NIGHT)
        axes[0, j].set_title(cond, color=INK, fontsize=10)
        axes[1, j].set_xlabel("sentences seen", color=MUTED, fontsize=8)
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.tick_params(colors=MUTED, labelsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel("total code (bits / sentence)", color=MUTED, fontsize=8)
    axes[1, 0].set_ylabel("generation commission (%)", color=MUTED, fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, title="incremental (lines);\nnight at 320 (dots)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)


if __name__ == "__main__":
    main(*sys.argv[1:4])
