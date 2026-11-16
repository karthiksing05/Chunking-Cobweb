"""Figures for story_chunks.py: which phrases TRELLIS's story chunks encode as learning goes on.

Usage: story_chunks_plot.py <out prefix> <results json> [more seeds' json ...]
  <out prefix>_curves.png   one panel per label: share of chunks committed in each learning window
                            carrying the label (mean over seeds, each seed a thin line), against
                            the same share in size-matched random and same-tag spans (dashed, dotted),
                            and in held-out recognised chunks (green)
  <out prefix>_lift.png     heatmaps, labels x checkpoints: lift over random and over same-tag
                            chance (log2), seed mean
  <out prefix>_cases.png    held-out sentences (rows) x checkpoints (columns): the sentence with its
                            recognised chunks bracketed (chosen chunks below theta in grey parentheses)
"""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1]
runs = [json.load(open(p, encoding="utf-8")) for p in sys.argv[2:]]
LABELS = runs[0]["labels"]
cps = [c["learned"] for c in runs[0]["checkpoints"]]
INK, MUTED, GRID, ACC, HELD = "#1f1f1d", "#6b6a64", "#e4e3dc", "#2a78d6", "#1baf7a"


def series(key, lab):
    return [[c[key][lab] if c[key][lab] is not None else float("nan") for c in r["checkpoints"]] for r in runs]


def mean(cols):
    out_ = []
    for vals in zip(*cols):
        v = [x for x in vals if not math.isnan(x)]
        out_.append(sum(v) / len(v) if v else float("nan"))
    return out_


# curves
n = len(LABELS)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.5 * rows), sharex=True, squeeze=False)
for k, lab in enumerate(LABELS):
    ax = axes.flat[k]
    s = series("rate", lab)
    for line in s:
        ax.plot(cps, line, color=ACC, lw=0.8, alpha=0.35)
    ax.plot(cps, mean(s), color=ACC, lw=2, marker="o", ms=4, label="committed chunks")
    ax.plot(cps, mean(series("chance random", lab)), color=MUTED, lw=1.5, ls="--", label="random-span chance")
    ax.plot(cps, mean(series("chance pos", lab)), color=MUTED, lw=1.5, ls=":", label="same-tag chance")
    ax.plot(cps, mean(series("held-out rate", lab)), color=HELD, lw=1.5, marker="s", ms=3, label="held-out recognised")
    ax.set_title(lab, fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.grid(color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if k % cols == 0:
        ax.set_ylabel("share of chunks", fontsize=9)
for k in range(n, rows * cols):
    axes.flat[k].axis("off")
axes.flat[0].legend(fontsize=7, frameon=False)
for ax in axes[-1]:
    ax.set_xlabel("sentences learned", fontsize=9)
fig.tight_layout()
fig.savefig(out + "_curves.png", dpi=150)
plt.close(fig)

# lift heatmaps
fig, axs = plt.subplots(1, 2, figsize=(2.6 + 1.8 * len(cps), 0.45 * n + 1.4))
for ax, key, title in ((axs[0], "lift random", "over random spans"), (axs[1], "lift pos", "over same-tag spans")):
    M = [[math.log2(v) if v and v > 0 and not math.isnan(v) else float("nan") for v in
          mean([[c[key][lab] if c[key][lab] is not None else float("nan") for c in r["checkpoints"]] for r in runs])]
         for lab in LABELS]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cps)), [str(c) for c in cps])
    ax.set_yticks(range(n), LABELS if ax is axs[0] else [""] * n)
    for i in range(n):
        for j in range(len(cps)):
            if not math.isnan(M[i][j]):
                ax.text(j, i, f"{2 ** M[i][j]:.1f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlabel("sentences learned")
    ax.set_title(f"lift {title} (×)", fontsize=10)
fig.colorbar(im, ax=axs, label="log2 lift", shrink=0.8)
fig.savefig(out + "_lift.png", dpi=150, bbox_inches="tight")
plt.close(fig)


def bracketed(words, chunks):
    """The sentence with recognised chunks in [ ] and other chosen chunks in ( )."""
    opens, closes = [[] for _ in words], [[] for _ in words]
    for ch in sorted(chunks, key=lambda c: (c["span"][0], -c["span"][1])):
        i, j = ch["span"]
        o, c = ("[", "]") if ch["recognised"] else ("(", ")")
        opens[i].append(o)
        closes[j - 1].insert(0, c)
    return " ".join("".join(opens[k]) + w + "".join(closes[k]) for k, w in enumerate(words))


# cases (first run)
cases = runs[0]["cases"]
fig, ax = plt.subplots(figsize=(3.2 + 3.0 * len(cps), 0.9 + 0.95 * len(cases)))
ax.axis("off")
for j, cp in enumerate(cps):
    ax.text((j + 0.5) / len(cps), 1.0, f"{cp} learned", ha="center", va="bottom", fontsize=9, color=INK,
            transform=ax.transAxes, weight="bold")
for i, case in enumerate(cases):
    words = case["sentence"].split()
    y = 1 - (i + 0.6) / len(cases)
    for j, snap in enumerate(case["by checkpoint"]):
        rec = [c for c in snap["chunks"] if c["recognised"]]
        lab = "; ".join(f"{c['words']}: {', '.join(c['labels']) or '-'}" for c in sorted(rec, key=lambda c: -c["posterior"])[:2])
        ax.text((j + 0.5) / len(cps), y, bracketed(words, snap["chunks"]), ha="center", va="center", fontsize=8,
                color=INK, family="DejaVu Sans Mono", transform=ax.transAxes)
        ax.text((j + 0.5) / len(cps), y - 0.28 / len(cases), lab, ha="center", va="center", fontsize=6, color=MUTED,
                transform=ax.transAxes)
fig.text(0.01, 0.005, "[ ] recognised (span posterior >= theta)   ( ) chosen, below theta   grey: labels of the two most "
         "confident recognised chunks", fontsize=7, color=MUTED)
fig.savefig(out + "_cases.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", out + "_curves.png", out + "_lift.png", out + "_cases.png")
