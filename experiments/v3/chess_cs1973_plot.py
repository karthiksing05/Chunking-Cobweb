"""Figure for chess_cs1973_exact.py: the model's learning curve against Chase & Simon's three players.

Usage: chess_cs1973_plot.py <out png> <results json> [more seeds' json ...]
Panels (x = positions learned, log scale; seed mean, each seed a thin line; human B / A / M drawn
at the headline points 300 / 1000 / 4000 as open markers, not fitted):
  trial-1 pieces correct, middle game  K = 7 and K = 4 (STM size)
  trial-1 pieces correct, end game     K = 7 and K = 4
  trial-1 pieces correct, random       K = 7 and K = 4 (middle and end averaged)
  first chunk size and pieces per glance (copy task), K = 7
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1]
runs = [json.load(open(p, encoding="utf-8")) for p in sys.argv[2:]]
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e4e3dc"
COL = {"K=7": "#2a78d6", "K=4": "#d6336c"}
HEAD = [300, 1000, 4000]
HUMAN = {"middle": (4, 8, 16), "end": (4, 7, 8), "random": (3.0, 2.5, 2.0),
         "first chunk size": (2.4, 2.6, 3.8), "pieces per glance": (2.0, 2.8, 2.0)}


def pts(r):
    return sorted((int(k), v) for k, v in r["points"].items() if int(k) > 0)


def get(r, K, typ, key):
    out_ = []
    for n, p in pts(r):
        if typ == "random":
            a = p["memory"][K]["random middle"][key]["mean"]
            b = p["memory"][K]["random end"][key]["mean"]
            out_.append((n, (a + b) / 2))
        else:
            v = p["memory"][K][typ][key]["mean"]
            out_.append((n, v if v is not None else float("nan")))
    return out_


def draw(ax, series, colour, label):
    xs = [n for n, _ in series[0]]
    for s in series:
        ax.plot([n for n, _ in s], [v for _, v in s], color=colour, lw=0.7, alpha=0.35)
    m = [sum(s[i][1] for s in series) / len(series) for i in range(len(xs))]
    ax.plot(xs, m, color=colour, lw=2, marker="o", ms=4, label=label)


def human(ax, vals, label="human B / A / M"):
    ax.plot(HEAD, vals, ls="none", marker="s", ms=8, mfc="none", mec=INK, mew=1.5, label=label)
    for x, v, t in zip(HEAD, vals, "BAM"):
        ax.annotate(t, (x, v), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)


fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, typ, title in zip(axes[:3], ("middle", "end", "random"),
                          ("game middle (24-26 pieces)", "game end (12-15 pieces)", "random positions")):
    for K in ("K=7", "K=4"):
        draw(ax, [get(r, K, typ, "trial 1 correct") for r in runs], COL[K], f"model, STM {K}")
    human(ax, HUMAN[typ])
    ax.set_title(f"trial-1 recall, {title}", fontsize=10)
    ax.set_ylabel("pieces correct", fontsize=9)
ax = axes[3]
draw(ax, [get(r, "K=7", "middle", "first chunk size") for r in runs], "#1baf7a", "model first chunk size")
cp = [[(n, p["copy"]["middle"]["pieces per glance"]["mean"]) for n, p in pts(r)] for r in runs]
draw(ax, cp, "#eda100", "model pieces per glance (copy)")
ax.plot(HEAD, HUMAN["first chunk size"], ls="none", marker="s", ms=8, mfc="none", mec="#1baf7a", mew=1.5, label="human first chunk")
ax.plot(HEAD, HUMAN["pieces per glance"], ls="none", marker="D", ms=7, mfc="none", mec="#eda100", mew=1.5, label="human per glance")
ax.set_title("chunk size, game middle", fontsize=10)
ax.set_ylabel("pieces", fontsize=9)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("positions learned", fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("wrote", out)
