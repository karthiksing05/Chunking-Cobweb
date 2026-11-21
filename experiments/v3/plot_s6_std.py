"""chess_chunkers with the standard TRELLIS configuration (exact_alone on) added.

The learning-scale curve of plot_s6.py (300 to 2,000 positions from chess_scale_submit.sh, 4,000 from the transfer runs) is the better of
the offsets-only and offsets-plus-squares arms with exact_alone off. Learning-curve runs of the standard configuration (C5, exact_alone on)
exist only at 4,000 positions: ab-control-{control,shift} (logs/v3_r103_submit.sh) and r228-full-{family,wing} (seed 13). They are plotted as
separate markers. Writes confs/trellis-v2/paper/figures/chess_chunkers_std.png.
"""
import json
import os
import sys

sys.argv = sys.argv[:1]
import plot_s6 as m  # noqa: E402

plt = m.plt
CC = os.path.join(m.RES, "chess_compress")
STD_FILES = {"control": "ab-control-control", "shift": "ab-control-shift", "family": "r228-full-family", "wing": "r228-full-wing"}


def std_bits(s):
    d = json.load(open(os.path.join(CC, STD_FILES[s] + "_s13.json"), encoding="utf-8"))
    return min(r["bits per position"] for r in d["by levels"] if r.get("copy") is True)


fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.1), sharey=True)
for ax, (s, title) in zip(axes, m.SETS):
    for name, color, ls in m.METHODS:
        pts = [(n, m.bits(name, s, n)) for n in m.NS]
        pts = [(n, b) for n, b in pts if b is not None]
        if pts:
            lab = "TRELLIS, C7 (best variant)" if name == "TRELLIS" else name
            ax.plot(*zip(*pts), color=color, ls=ls, lw=2.2 if name == "TRELLIS" else 1.3,
                    marker="o" if name == "TRELLIS" else None, ms=3.5, mfc="white" if name == "TRELLIS" else None, label=lab)
    b = std_bits(s)
    ax.plot([4000], [b], "o", color="#D55E00", ms=6.5, zorder=5, label="TRELLIS, C5 (standard)")
    ax.annotate(f"{b:.1f}", (4000, b), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color="#D55E00")
    ax.set_xscale("log")
    ax.set_xticks(m.NS)
    ax.minorticks_off()
    ax.set_xticklabels(["300", "1k", "2k", "4k"])
    ax.set_title(title, loc="left")
    ax.set_xlabel("Positions learned")
axes[0].set_ylabel("Bits per held-out position")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = os.path.join(m.ROOT, "confs", "trellis-v2", "paper", "figures", "chess_chunkers_std.png")
fig.savefig(out)
print(out, {s: std_bits(s) for s in STD_FILES}, {s: m.bits("TRELLIS", s, 4000) for s in STD_FILES})
