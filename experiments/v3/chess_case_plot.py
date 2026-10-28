"""Figure for chess_case.py: one held-out position followed through experience.
Rows = checkpoints, columns = the real position and its scrambled twin. Pieces in
the units short-term memory keeps are drawn in full, each multi-piece unit ringed
in its own colour; pieces not recalled are faded.

Usage: chess_case_plot.py <results json> <case index> <checkpoints> <out png>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

d = json.load(open(sys.argv[1], encoding="utf-8"))
case = d["cases"][int(sys.argv[2])]
want = [int(x) for x in sys.argv[3].split(",")]
out = sys.argv[4]
GLYPH = {"wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
         "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟"}
LIGHT, DARK = "#efe6d6", "#c9b79c"
RING = ["#d9622b", "#2a6fc0", "#1a9068", "#8c4440", "#5a3a7a", "#b86508", "#3b5e8c"]


def sq(name):
    return "abcdefgh".index(name[0]), int(name[1]) - 1


cps = [c for c in case["checkpoints"] if c["learned"] in want]
fig, axes = plt.subplots(len(cps), 2, figsize=(5.6, 2.9 * len(cps)))
for row, cp in zip(axes, cps):
    for ax, which in zip(row, ("real", "scrambled")):
        for f in range(8):
            for r in range(8):
                ax.add_patch(Rectangle((f, r), 1, 1, color=LIGHT if (f + r) % 2 else DARK, lw=0))
        kept = cp[which]["kept"]
        recalled = {name for u in kept for name, _ in u}
        k = 0
        for u in kept:
            if len(u) < 2:
                continue
            for name, _ in u:
                f, r = sq(name)
                ax.add_patch(Rectangle((f + 0.06, r + 0.06), 0.88, 0.88, fill=False, ec=RING[k % len(RING)], lw=2.2))
            k += 1
        for name, tok in case[which].items():
            f, r = sq(name)
            ax.text(f + 0.5, r + 0.47, GLYPH[tok], ha="center", va="center", fontsize=12,
                    alpha=1.0 if name in recalled else 0.18, family="DejaVu Sans")
        ax.set_xlim(0, 8), ax.set_ylim(0, 8), ax.set_aspect("equal"), ax.axis("off")
        n_chunks = sum(len(u) > 1 for u in kept)
        ax.set_title(f"{which}, {cp['learned']} learned: recall {cp[which]['recall']:.2f}\n"
                     f"{len(recalled)} of {len(case[which])} pieces, {n_chunks} chunks", fontsize=8.5)
fig.suptitle(f"One held-out position through experience (7 units kept, θ {d['theta']:g})", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out, dpi=130)
