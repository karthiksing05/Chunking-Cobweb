"""Learning curves for the status page: chess recall, chess compression and
semantic generation, each against experience. Draws whatever results exist.

  docs/site/curve_recall.png       verified recall of real and scrambled positions (K 7, K 4)
  docs/site/curve_compress.png     bits per held-out position: code, and code + memory
  docs/site/curve_semantic.png     acceptable novel sentences per generation, and topic (cross share)

Usage: learning_curves.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "docs", "site")
# fixed categorical order (one hue per arm, the same arm keeps its hue in every figure)
HUE = {"flat, absolute squares, code-chosen cover": "#d6336c", "flat, join any": "#2a78d6", "flat, join interact, absolute squares": "#8a5cd6", "flat, join interact": "#eb6834", "nested, pieces in content": "#1baf7a",
       "nested, original": "#eda100", "co-occurrence chunks": "#6b6a64", "piece-square list": "#a8a79f"}
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e4e3dc"


def style(ax, xlabel, ylabel, title, ticks=(300, 1000, 2000, 4000)):
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def rows(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def logs(p):
    return os.path.join(ROOT, "logs", p)


# ---------------------------------------------------------------- recall
RECALL = {"nested, original": "chess-rc2-13.log", "nested, pieces in content": "chess-flat-base-13.log",
          "flat, join any": "chess-flat-any-m4-13.log", "flat, join interact": "chess-flat-interact-m4-13.log"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
bpe = None
for ax, k in zip(axes, ("K7", "K4")):
    for name, f in RECALL.items():
        if not os.path.exists(logs(f)):
            continue
        rs = [r for r in rows(logs(f)) if "positions learned" in r and r["positions learned"] > 0]
        key = lambda side: f"{side} verified theta 12: recall {k}" if f"real verified theta 12: recall {k}" in rs[0] \
            else (f"{side} verified theta 12: recall" if k == "K7" else None)
        if not rs or key("real") is None:
            continue
        x = [r["positions learned"] for r in rs]
        ax.plot(x, [r[key("real")] for r in rs], color=HUE[name], lw=2, marker="o", ms=5, label=f"{name}, real")
        ax.plot(x, [r[key("random")] for r in rs], color=HUE[name], lw=1.5, ls="--", marker="o", ms=4, mfc="white")
        if f"real bpe: recall {k}" in rs[0] and (bpe is None or len(rs) >= len(bpe[0])):
            bpe = (x, rs)
    if bpe:
        x, rs = bpe
        ax.plot(x, [r[f"real bpe: recall {k}"] for r in rs], color=HUE["co-occurrence chunks"], lw=2, marker="s", ms=5,
                label="co-occurrence chunks, real")
        ax.plot(x, [r[f"random bpe: recall {k}"] for r in rs], color=HUE["co-occurrence chunks"], lw=1.5, ls="--",
                marker="s", ms=4, mfc="white")
    ax.set_xscale("log")
    style(ax, "positions learned", "share of pieces recalled" if k == "K7" else "",
          f"Recall with {k[1]} units in short-term memory")
axes[0].text(0.02, 0.04, "solid: real positions   dashed: same pieces, random squares", transform=axes[0].transAxes,
             fontsize=7.5, color=MUTED)
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, [l.replace(", real", "") for l in lab], loc="lower center", ncol=5, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, "curve_recall.png"), dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- compression
res = os.path.join(ROOT, "experiments", "v3", "results", "chess_compress")
ARMS = {"nested, pieces in content": "leaf-nested", "flat, join interact": "leaf-flat-interact",
        "flat, join interact, absolute squares": "abs-interact", "flat, absolute squares, code-chosen cover": "recog-abs-interact"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ps = {}
for name, tag in ARMS.items():
    pts = []
    for f in glob.glob(os.path.join(res, f"{tag}-*_s13.json")):
        d = json.load(open(f))
        n = int(re.search(r"-(\d+)_s13", f).group(1))
        rows_ = [r for r in d["by levels"] if "copy" in r and r["bits per position"] is not None]
        copy = min(r["bits per position"] for r in rows_ if r["copy"])
        plain = min(r["bits per position"] for r in d["by levels"] if not r.get("copy"))
        pts.append((n, copy, plain))
        ps[n] = d["piece-square bits mean"]
    pts.sort()
    if pts:
        x = [p[0] for p in pts]
        axes[0].plot(x, [p[1] for p in pts], color=HUE[name], lw=2, marker="o", ms=5, label=name)
        axes[0].plot(x, [p[2] for p in pts], color=HUE[name], lw=1.5, ls="--", marker="o", ms=4, mfc="white")
# prequential: every position coded by what was learned before it, model included (cumulative mean)
PREQ = {"nested, pieces in content": "nested", "flat, join interact": "flat-interact",
        "flat, join interact, absolute squares": "abs-interact", "flat, absolute squares, code-chosen cover": "recog",
        "co-occurrence chunks": "bpe"}
pq_raw = {}
for name, t in PREQ.items():
    if not os.path.exists(logs(f"chess-preq-{t}.log")):
        continue
    pr = {r["positions"]: r for r in rows(logs(f"chess-preq-{t}.log")) if "positions" in r}
    xs = sorted(pr)
    if xs:
        axes[1].plot(xs, [pr[x]["mean bits per position"] for x in xs], color=HUE[name], lw=2,
                     marker="s" if t == "bpe" else "o", ms=5)
        pq_raw.update({x: pr[x]["piece-square mean"] for x in xs})
if pq_raw:
    xs = sorted(pq_raw)
    axes[1].plot(xs, [pq_raw[x] for x in xs], color=HUE["piece-square list"], lw=2, ls=":", marker="^", ms=5)
bp = []
for f in glob.glob(logs("chess-curve-bpe-*.log")):
    rr = [r for r in rows(f) if "merges" in r and "total per position" in r]
    if rr:
        n = int(re.search(r"bpe-(\d+)\.log", f).group(1))
        bp.append((n, min(r["bits per position"] for r in rr), min(r["total per position"] for r in rr)))
bp.sort()
if bp:
    axes[0].plot([p[0] for p in bp], [p[1] for p in bp], color=HUE["co-occurrence chunks"], lw=2, marker="s", ms=5,
                 label="co-occurrence chunks (best merge count)")
if ps:
    xs = sorted(ps)
    axes[0].plot(xs, [ps[x] for x in xs], color=HUE["piece-square list"], lw=2, ls=":", marker="^", ms=5,
                 label="piece-square list, entropy-coded")
for ax in axes:
    ax.set_xscale("log")
style(axes[0], "positions learned", "bits per held-out position", "Code (dashed: without copying from memory)")
style(axes[1], "positions seen", "mean bits per position so far", "Prequential: model included")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(os.path.join(OUT, "curve_compress.png"), dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- semantic generation
judge = os.path.join(ROOT, "experiments", "v3", "results", "story_gen", "curve", "judge.json")
if os.path.exists(judge):
    J = json.load(open(judge))
    SEM = {0: ("unconditioned", "#6b6a64"), 1: ("topic-directed (back-off β 64, both neighbours)", "#2a78d6"),
           2: ("topic-directed + parse critic (best of 4)", "#eb6834")}
    acc = {}
    for a in J["arms"]:
        m = re.match(r"story_s(\d+)_n(\d+)_arm(\d+)", a["file"])
        s, n, i = map(int, m.groups())
        acc.setdefault(i, {}).setdefault(n, []).append(a["grammatical novel per generation"])
    cross = {}
    for f in glob.glob(logs("v3-semcurve-*-*.log")):
        for i, r in enumerate(r for r in rows(f) if "arm" in r):
            if r.get("cross share") is not None:
                cross.setdefault(i, {}).setdefault(r["n"], []).append(r["cross share"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, data, ylab, title in ((axes[0], acc, "share of generations", "Acceptable novel sentences per generation"),
                                  (axes[1], cross, "cross share (0 random, 1 real next sentence)", "Topic carried from the prompt")):
        for i, (name, col) in SEM.items():
            if i not in data:
                continue
            xs = sorted(data[i])
            ax.plot(xs, [sum(data[i][x]) / len(data[i][x]) for x in xs], color=col, lw=2, marker="o", ms=5, label=name)
            for x in xs:
                ax.scatter([x] * len(data[i][x]), data[i][x], color=col, s=10, alpha=0.5, zorder=3)
        ax.set_xscale("log")
        style(ax, "sentences learned", ylab, title, ticks=(500, 1000, 2000))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, "curve_semantic.png"), dpi=140)
    plt.close(fig)
print("done")
