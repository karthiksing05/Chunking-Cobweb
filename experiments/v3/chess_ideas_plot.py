"""Figures for chess_ideas.py: how the chess ideas in learned chunks change with learning.

Usage: chess_ideas_plot.py <out prefix> <results json> [more seeds' json ...]
  <out prefix>_curves.png   one panel per idea: share of chunks committed in each learning window
                            carrying the idea (mean over seeds, each seed a thin line), against the
                            same share in size-matched nearby piece groups (chance, dashed)
  <out prefix>_lift.png     heatmap, ideas x checkpoints: lift over nearby chance (log2), seed mean
  <out prefix>_cases.png    held-out positions (rows) x checkpoints (columns): the chunks the player
                            recognises, each ringed in its own colour, with their ideas listed
"""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

out = sys.argv[1]
runs = [json.load(open(p, encoding="utf-8")) for p in sys.argv[2:]]
IDEAS = runs[0]["ideas"]
cps = [c["learned"] for c in runs[0]["checkpoints"]]
INK, MUTED, GRID, ACC = "#1f1f1d", "#6b6a64", "#e4e3dc", "#2a78d6"
GLYPH = {"wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
         "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟"}
LIGHT, DARK = "#efe6d6", "#c9b79c"
RING = ["#2a78d6", "#d6336c", "#1baf7a", "#eda100", "#8a5cd6", "#eb6834", "#6b6a64"]


def series(key, idea):
    return [[c[key][idea] if c[key][idea] is not None else float("nan") for c in r["checkpoints"]] for r in runs]


def mean(cols):
    out_ = []
    for vals in zip(*cols):
        v = [x for x in vals if not math.isnan(x)]
        out_.append(sum(v) / len(v) if v else float("nan"))
    return out_


# curves
n = len(IDEAS)
cols = 4
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows), sharex=True)
for k, idea in enumerate(IDEAS):
    ax = axes.flat[k]
    s = series("rate", idea)
    for line in s:
        ax.plot(cps, line, color=ACC, lw=0.8, alpha=0.35)
    ax.plot(cps, mean(s), color=ACC, lw=2, marker="o", ms=4, label="learned chunks")
    ax.plot(cps, mean(series("chance nearby", idea)), color=MUTED, lw=1.5, ls="--", label="nearby chance")
    ax.set_title(idea, fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.grid(color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if k % cols == 0:
        ax.set_ylabel("share of chunks", fontsize=9)
for k in range(n, rows * cols):
    axes.flat[k].axis("off")
axes.flat[0].legend(fontsize=8, frameon=False)
for ax in axes[-1]:
    ax.set_xlabel("positions learned", fontsize=9)
fig.tight_layout()
fig.savefig(out + "_curves.png", dpi=150)
plt.close(fig)

# lift heatmap
M = [[math.log2(v) if v and v > 0 and not math.isnan(v) else float("nan") for v in
      mean([[c["lift nearby"][idea] if c["lift nearby"][idea] is not None else float("nan") for c in r["checkpoints"]]
            for r in runs])] for idea in IDEAS]
fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cps), 0.45 * n + 1.2))
im = ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(cps)), [str(c) for c in cps])
ax.set_yticks(range(n), IDEAS)
for i in range(n):
    for j in range(len(cps)):
        if not math.isnan(M[i][j]):
            ax.text(j, i, f"{2 ** M[i][j]:.1f}", ha="center", va="center", fontsize=8, color=INK)
ax.set_xlabel("positions learned")
ax.set_title("lift over nearby chance (×)", fontsize=10)
fig.colorbar(im, ax=ax, label="log2 lift", shrink=0.8)
fig.tight_layout()
fig.savefig(out + "_lift.png", dpi=150)
plt.close(fig)

# cases (first run)
cases = runs[0]["cases"]
fig, axes = plt.subplots(len(cases), len(cps), figsize=(2.4 * len(cps), 2.9 * len(cases)), squeeze=False)
for i, case in enumerate(cases):
    pieces = list(case["fen"].items())
    for j, snap in enumerate(case["by checkpoint"]):
        ax = axes[i][j]
        for f in range(8):
            for r in range(8):
                ax.add_patch(Rectangle((f, r), 1, 1, color=LIGHT if (f + r) % 2 else DARK, lw=0))
        labels = []
        for k, ch in enumerate(snap["chunks"][:len(RING)]):
            for name, _ in ch["pieces"]:
                f, r = "abcdefgh".index(name[0]), int(name[1]) - 1
                pad = 0.04 + 0.07 * (k % 3)
                ax.add_patch(Rectangle((f + pad, r + pad), 1 - 2 * pad, 1 - 2 * pad, fill=False, ec=RING[k], lw=1.6))
            labels.append(", ".join(ch["ideas"]) or "-")
        for name, tok in pieces:
            f, r = "abcdefgh".index(name[0]), int(name[1]) - 1
            ax.text(f + 0.5, r + 0.47, GLYPH[tok], ha="center", va="center", fontsize=10, family="DejaVu Sans")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if i == 0:
            ax.set_title(f"{snap['learned']} learned", fontsize=9)
        ax.set_xlabel("\n".join(f"{k + 1}: {t}" for k, t in enumerate(labels[:4])) or "no chunk recognised", fontsize=6,
                      color=MUTED)
fig.tight_layout()
fig.savefig(out + "_cases.png", dpi=150)
plt.close(fig)
print("wrote", out + "_curves.png", out + "_lift.png", out + "_cases.png")
