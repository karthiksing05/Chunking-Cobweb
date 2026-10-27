"""Figures for chess_inspect.py: chunk types by size, and nested analyses.

Usage: chess_inspect_plot.py <results json> <out prefix>
  <out prefix>_types.png    rows = chunk size, columns = the most frequent content
                            concepts of that size, each drawn by its most frequent
                            instance (count = committed chunks under the concept)
  <out prefix>_nested.png   held-out positions with their stored analysis: every
                            join drawn from one part's anchor to the other's, coloured
                            by the size of the chunk it makes; recognised chunks ringed
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

d = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
GLYPH = {"wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
         "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟"}
LIGHT, DARK, HL = "#efe6d6", "#c9b79c", "#e8a33d"


def sq(name):
    return "abcdefgh".index(name[0]), int(name[1]) - 1


def board(ax, pieces, highlight=(), alpha_other=1.0, size=14):
    for f in range(8):
        for r in range(8):
            ax.add_patch(Rectangle((f, r), 1, 1, color=LIGHT if (f + r) % 2 else DARK, lw=0))
    for name in highlight:
        f, r = sq(name)
        ax.add_patch(Rectangle((f + 0.04, r + 0.04), 0.92, 0.92, fill=False, ec=HL, lw=2))
    for name, tok in pieces:
        f, r = sq(name)
        ax.text(f + 0.5, r + 0.47, GLYPH[tok], ha="center", va="center", fontsize=size, color="black",
                alpha=1.0 if (not highlight or name in highlight) else alpha_other, family="DejaVu Sans")
    ax.set_xlim(0, 8), ax.set_ylim(0, 8), ax.set_aspect("equal"), ax.axis("off")


# chunk types by size
inv = {int(k): v for k, v in d["inventory"].items()}
sizes = sorted(inv)
cols = 5
fig, axes = plt.subplots(len(sizes), cols, figsize=(cols * 2.1, len(sizes) * 2.35))
for i, s in enumerate(sizes):
    rows = inv[s]["top"][:cols]
    for j in range(cols):
        ax = axes[i][j]
        if j >= len(rows):
            ax.axis("off")
            continue
        r = rows[j]
        names = [n for n, _ in r["example"]]
        board(ax, r["example"], highlight=names, size=11)
        ax.set_title(f"n={r['count']}  ({r['distinct']} distinct)", fontsize=7.5)
    axes[i][0].text(-0.6, 4, f"{s} pieces\n{inv[s]['committed']} stored\n{inv[s]['concepts']} concepts",
                    ha="right", va="center", fontsize=8.5, transform=axes[i][0].transData)
fig.suptitle(f"Chess chunks by size after {d['positions learned']} positions (max size {d['max size']}, seed {d['seed']}): "
             "the most frequent content concepts, each drawn by its most frequent instance", fontsize=10)
fig.tight_layout(rect=(0.06, 0, 1, 0.97))
fig.savefig(out + "_types.png", dpi=130)

# nested analyses
cmap = plt.get_cmap("viridis")
maxs = d["max size"]


def anchor(node):
    if "piece" in node:
        return sq(node["sq"])
    return min((sq(n) for n, _ in node["pieces"]), key=lambda t: (t[1], t[0]))


def joins(node, acc):
    if "piece" in node:
        return
    a, b = (anchor(p) for p in node["parts"])
    acc.append((a, b, node["size"]))
    for p in node["parts"]:
        joins(p, acc)


n = min(len(d["nested"]), 4)
fig, axes = plt.subplots(1, n, figsize=(n * 3.4, 3.9))
for ax, pos in zip(axes if n > 1 else [axes], d["nested"][:n]):
    pieces = sorted(pos["fen"].items())
    board(ax, pieces, size=13)
    acc = []
    for t in pos["top"]:
        joins(t, acc)
    for (fa, ra), (fb, rb), s in sorted(acc, key=lambda t: -t[2]):
        ax.plot([fa + 0.5, fb + 0.5], [ra + 0.5, rb + 0.5], color=cmap((s - 2) / max(maxs - 2, 1)),
                lw=1.0 + 2.2 * (s - 2) / max(maxs - 2, 1), alpha=0.85, solid_capstyle="round")
    rec = [c for c in pos["recognised"] if len(c) > 1]
    ax.set_title(f"{len(acc)} joins, {len(rec)} chunks recognised", fontsize=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(2, maxs))
cb = fig.colorbar(sm, ax=axes if n > 1 else [axes], fraction=0.02, pad=0.01)
cb.set_label("size of the chunk a join makes (pieces)", fontsize=8)
fig.suptitle("Held-out positions as nested chunks: small chunks (dark, thin) join into larger ones (light, thick)", fontsize=10)
fig.savefig(out + "_nested.png", dpi=130, bbox_inches="tight")

rec = d["recognised units per position"]
print(json.dumps(rec))
