"""Deck revision 4b: picture text of at least 14 pt at slide size on slides 6, 10, 14, 17, 25 and 28.

Infographics are redrawn in the ACS first-paper look (numbered black title bars, pink word boxes, purple
candidates, blue chunks) with three words instead of five and large labelled boxes instead of mini-trees.
Each figure is W inches wide and is shown about 9 in wide, so a figure font F appears as F * 9 / W pt;
FS = 22 at W = 13.5 gives 14.7 pt. Plots use figure width = slide width and 16 pt text.
Writes PNGs *_talk.png in confs/trellis-v2/talk_figures/.  Run on a compute node:
  python confs/trellis-v2/make_talk_figures_rev4.py [parse_v1 parse_v2 gen_v2 learn_v2 cs28 regimes]"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_talk_infographics as ti  # noqa: E402

OUT = os.path.join(HERE, "talk_figures")
WANT = set(sys.argv[1:])
W, H, PW = 13.5, 4.3, 4.3
FS = 22
GATE, RANK, POST = ("#2f8a3e", "#e8f7ed"), ("#2b6cb0", "#eaf2fb"), ("#5d3f7d", "#f1eaf7")


def want(n):
    return not WANT or n in WANT


def canvas():
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    return fig, ax


def panel(ax, k, title):
    gap = (W - 0.1 - 3 * PW) / 2
    x = 0.05 + k * (PW + gap)
    ax.add_patch(FancyBboxPatch((x, 0.05), PW, H - 0.1, boxstyle="round,pad=0.005,rounding_size=0.04",
                                linewidth=1.2, edgecolor="#444", facecolor="white"))
    ax.add_patch(FancyBboxPatch((x + 0.06, H - 0.67), PW - 0.12, 0.62, boxstyle="round,pad=0.005,rounding_size=0.04",
                                linewidth=0, facecolor="#1a1a1a"))
    ax.add_patch(mpatches.Circle((x + 0.42, H - 0.36), 0.22, facecolor="white", edgecolor="#1a1a1a"))
    ax.text(x + 0.42, H - 0.36, str(k + 1), ha="center", va="center", fontsize=FS, fontweight="bold",
            family="serif", color="#1a1a1a")
    ax.text(x + 0.82, H - 0.36, title, ha="left", va="center", fontsize=FS, fontweight="bold", family="serif",
            color="white")
    if k < 2:
        ti._panel_arrow(ax, x + PW + 0.02, H / 2, x + PW + gap - 0.02, color="#444")
    return x + PW / 2


def edge(ax, a, b, ra, rb, col="#444", ls="-"):
    ax.add_line(Line2D([a[0], b[0]], [a[1] - ra, b[1] + rb], color=col, lw=1.8, ls=ls, zorder=2))


def node(ax, p, lab, r=0.36, fill=None, edgec=None):
    ti._circle_chunk(ax, p[0], p[1], r, label=lab, fontsize=FS - 3, fill=fill or ti.CHUNK_FILL,
                     edge=edgec or ti.CHUNK_EDGE)


def word(ax, p, w, fill=None, edgec=None):
    kw = {}
    if fill:
        kw = dict(fill=fill, edge=edgec)
    ti._primitive_box(ax, p[0], p[1], 1.05, 0.58, w, fontsize=FS, **kw)


def box(ax, cx, cy, w, text, col, fill, h=0.62, fs=FS):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.8, edgecolor=col, facecolor=fill, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=col, fontweight="bold", family="serif",
            zorder=4)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print("wrote", p)


def words3(cx, y):
    return [(cx - 1.25, y), (cx, y), (cx + 1.25, y)]


def parse_v1():
    fig, ax = canvas()
    cx = panel(ax, 0, "Form candidates")
    the, dog, eats = words3(cx, 0.95)
    np_ = (cx - 0.62, 1.95)
    cand = (cx + 0.1, 3.0)
    edge(ax, np_, the, .36, .29); edge(ax, np_, dog, .36, .29)
    edge(ax, cand, np_, .36, .36, col=ti.CAND_EDGE, ls=":"); edge(ax, cand, eats, .36, .29, col=ti.CAND_EDGE, ls=":")
    node(ax, np_, "NP"); node(ax, cand, "", fill=ti.CAND_FILL, edgec=ti.CAND_EDGE)
    for p, w in ((the, "the"), (dog, "dog"), (eats, "eats")):
        word(ax, p, w)
    cx = panel(ax, 1, "Gate and rank")
    box(ax, cx, 3.05, 3.8, "content taxonomy", "#2b6cb0", "#eaf2fb")
    box(ax, cx, 2.25, 3.8, "context taxonomy", "#3c8062", "#e8f7ed")
    box(ax, cx, 1.35, 3.8, "admit: 47 > 30", *GATE)
    box(ax, cx, 0.6, 3.8, "rank: -5.3", RANK[0], RANK[1])
    cx = panel(ax, 2, "Commit")
    the, dog, eats = words3(cx, 0.95)
    np_, s_ = (cx - 0.62, 1.95), (cx + 0.1, 3.0)
    edge(ax, np_, the, .36, .29); edge(ax, np_, dog, .36, .29)
    edge(ax, s_, np_, .36, .36); edge(ax, s_, eats, .36, .29)
    node(ax, np_, "NP"); node(ax, s_, "S")
    for p, w in ((the, "the"), (dog, "dog"), (eats, "eats")):
        word(ax, p, w)
    save(fig, "parse_v1_infographic_talk.png")


def parse_v2():
    fig, ax = canvas()
    cx = panel(ax, 0, "Score every span")
    the, dog, eats = words3(cx, 0.95)
    a, b, c = (cx - 0.62, 2.1), (cx + 0.62, 2.1), (cx, 3.15)
    for p, ch in ((a, the), (a, dog), (b, dog), (b, eats)):
        edge(ax, p, ch, .36, .29, col=ti.CAND_EDGE, ls=":")
    edge(ax, c, a, .36, .36, col=ti.CAND_EDGE, ls=":"); edge(ax, c, b, .36, .36, col=ti.CAND_EDGE, ls=":")
    for p in (a, b, c):
        node(ax, p, "", fill=ti.CAND_FILL, edgec=ti.CAND_EDGE)
    for p, w in ((the, "the"), (dog, "dog"), (eats, "eats")):
        word(ax, p, w)
    cx = panel(ax, 1, "Inside, outside")
    box(ax, cx, 3.0, 3.8, "inside: sub-spans", *GATE)
    box(ax, cx, 2.1, 3.8, "outside: context", RANK[0], RANK[1])
    box(ax, cx, 1.2, 3.8, "posterior: .82", POST[0], POST[1])
    cx = panel(ax, 2, "Best tree")
    the, dog, eats = words3(cx, 0.95)
    np_, s_ = (cx - 0.62, 2.1), (cx + 0.1, 3.15)
    edge(ax, np_, the, .36, .29); edge(ax, np_, dog, .36, .29)
    edge(ax, s_, np_, .36, .36); edge(ax, s_, eats, .36, .29)
    node(ax, np_, ".82"); node(ax, s_, "1.0")
    for p, w in ((the, "the"), (dog, "dog"), (eats, "eats")):
        word(ax, p, w)
    save(fig, "parse_v2_infographic_talk.png")


def gen_v2():
    fig, ax = canvas()
    cx = panel(ax, 0, "Next position")
    the, dog, eats = (cx - 1.52, 0.95), (cx - 0.5, 0.95), (cx + 0.52, 0.95)
    q = (cx + 1.5, 0.95)
    top, mid = (cx - 0.35, 3.1), (cx - 0.9, 2.05)
    edge(ax, top, mid, .36, .36, col=ti.CAND_EDGE, ls=":"); edge(ax, top, q, .36, .29, col=ti.CAND_EDGE, ls=":")
    edge(ax, mid, the, .36, .29, col=ti.CAND_EDGE, ls=":"); edge(ax, mid, dog, .36, .29, col=ti.CAND_EDGE, ls=":")
    for p in (top, mid):
        node(ax, p, "", fill=ti.CAND_FILL, edgec=ti.CAND_EDGE)
    for p, w in ((the, "the"), (dog, "dog"), (eats, "eats")):
        word(ax, p, w)
    ti._primitive_box(ax, q[0], q[1], 0.85, 0.58, "?", edge=ti.ANCHOR_EDGE, fill=ti.ANCHOR_FILL, fontsize=FS)
    cx = panel(ax, 1, "Score and gate")
    box(ax, cx, 2.8, 3.8, "outside score", RANK[0], RANK[1])
    box(ax, cx, 1.7, 3.8, "pair gate", *GATE)
    cx = panel(ax, 2, "Write, close")
    ws = [(cx - 1.55, 2.8), (cx - 0.55, 2.8), (cx + 0.5, 2.8), (cx + 1.55, 2.8)]
    for p, w in zip(ws, ("the", "dog", "eats", "a")):
        if w == "a":
            ti._primitive_box(ax, p[0], p[1], 0.85, 0.58, w, edge=ti.CAND_EDGE, fill=ti.CAND_FILL, fontsize=FS)
        else:
            ti._primitive_box(ax, p[0], p[1], 0.95, 0.58, w, fontsize=FS)
    box(ax, cx, 1.6, 3.8, "starts a sentence", *GATE)
    box(ax, cx, 0.8, 3.8, "ends a sentence", *GATE)
    save(fig, "generation_v2_infographic_talk.png")


def learn_v2():
    fig, ax = canvas()
    cx = panel(ax, 0, "Parse so far")
    the, dog, eats = words3(cx, 0.95)
    a, b, c = (cx - 0.62, 2.1), (cx + 0.62, 2.1), (cx, 3.15)
    for p, ch in ((a, the), (a, dog), (b, dog), (b, eats)):
        edge(ax, p, ch, .4, .29, col=ti.CAND_EDGE, ls=":")
    edge(ax, c, a, .4, .4, col=ti.CAND_EDGE, ls=":"); edge(ax, c, b, .4, .4, col=ti.CAND_EDGE, ls=":")
    for p, l in ((a, ".84"), (b, ".10"), (c, "1.0")):
        node(ax, p, l, r=0.42, fill=ti.CAND_FILL, edgec=ti.CAND_EDGE)
    for p, w in ((the, "the"), (dog, "dog"), (eats, "eats")):
        word(ax, p, w)
    cx = panel(ax, 1, "Weight spans")
    rows = ((2.95, "the dog", .84), (2.05, "dog eats", .10), (1.15, "all three", 1.0))
    for y, t, v in rows:
        ax.text(cx - 2.0, y, t, ha="left", va="center", fontsize=FS, family="serif", color="#1a1a1a")
        ax.add_patch(FancyBboxPatch((cx + 0.05, y - 0.17), 0.95 * v, 0.34, boxstyle="round,pad=0,rounding_size=0.04",
                                    linewidth=0, facecolor=ti.CAND_FILL, zorder=3))
        ax.text(cx + 1.2, y, ("%.2f" % v), ha="left", va="center", fontsize=FS - 3, family="monospace",
                color="#444")
    cx = panel(ax, 2, "Add to both")
    box(ax, cx, 2.9, 3.9, "content: the + dog", "#1f4e79", "#e4eef9")
    box(ax, cx, 2.0, 3.9, "context: __ eats", "#2e6e55", "#e6f4ed")
    ax.text(cx, 0.95, "each count +.84,\nnot +1", ha="center", va="center", fontsize=FS, family="serif",
            style="italic", color="#444")
    save(fig, "learning_v2_infographic_talk.png")


def plot_style(size=16):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": size, "axes.edgecolor": "#555",
                         "axes.labelcolor": "#222", "xtick.color": "#333", "ytick.color": "#333",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "axes.linewidth": 1.4, "xtick.labelsize": size,
                         "ytick.labelsize": size, "axes.labelsize": size})


def cs28():
    sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "v3"))
    argv = sys.argv
    sys.argv = [argv[0]]
    import plot_s6 as m
    sys.argv = argv
    plot_style(16)
    runs = m.load_cs()
    model = {p: m.avg(runs, lambda r, p=p: r["players"][p]["middle"]["pieces per trial"]) for p in ("M", "A", "B")}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), sharey=True)
    for ax, (title, data) in zip(axes, (("Chase & Simon", m.CS_FIG1["middle"]), ("TRELLIS", model))):
        for p in ("M", "A", "B"):
            ys = data[p]
            ax.plot(range(1, len(ys) + 1), ys, color=m.PLAYER[p], marker=m.MARK[p], ms=8, lw=3.2,
                    ls="-" if p == "M" else "--" if p == "A" else ":")
            if title != "TRELLIS":
                ax.text(len(ys) + 0.3, ys[-1] + (1.8 if p == "A" else -1.8 if p == "B" else 0), p, color=m.PLAYER[p], fontweight="bold", va="center", fontsize=16)
        ax.set_xlim(0.6, 8.3); ax.set_ylim(0, 27); ax.set_xticks([1, 3, 5, 7]); ax.set_yticks([0, 10, 20])
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold")
        ax.set_xlabel("trial")
        ax.grid(axis="y", color="#e4e3dc", lw=1)
    axes[0].set_ylabel("pieces correct")
    fig.tight_layout()
    save(fig, "cs_trials_talk.png")


def regimes():
    import make_figure_r230 as m
    plot_style(16)
    J = m.load()
    levels = ["leaf", 7, 6, 5, 4, 3, 2, 0]
    cols = [m.BLUE, m.ORANGE, m.GREEN]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 2.95), gridspec_kw={"width_ratios": [1.15, 1]})
    x = list(range(len(levels)))
    for j in range(3):
        mm, lo, hi = [], [], []
        for b in levels:
            v = [m.c6(J, s, 1000, b)[j] for s in m.SEEDS]
            mm.append(sum(v) / 2); lo.append(min(v)); hi.append(max(v))
        ax1.fill_between(x, lo, hi, color=cols[j], alpha=0.2, lw=0)
        ax1.plot(x, mm, "-o", color=cols[j], ms=6, lw=3)
    ax1.set_xticks(x); ax1.set_xticklabels(["leaf"] + [str(b) for b in levels[1:]])
    ax1.set_xlabel("word level $b$ (bits)"); ax1.set_ylabel("share")
    ax1.set_ylim(0, 1.0); ax1.set_xlim(-0.3, 9.6); ax1.set_yticks([0, .5, 1])
    ax1.text(7.3, 0.93, "R-non", color=m.GREEN, fontsize=16, va="center", fontweight="bold")
    ax1.text(7.3, 0.09, "R-sem", color=m.BLUE, fontsize=16, va="center", fontweight="bold")
    ax1.text(2.9, 0.80, "R-syn", color=m.ORANGE, fontsize=16, va="center", ha="center", fontweight="bold")
    ax1.set_title("One dial", fontsize=17, fontweight="bold", loc="left")
    ns = [250, 500, 1000, 2000, 4000]
    for b, j, c, lab, dy in (("leaf", 0, m.BLUE, "sem", 0.05), (6, 1, m.ORANGE, "syn", -0.07), (0, 2, m.GREEN, "non", 0)):
        v = [[m.c6(J, s, n, b)[j] for s in m.SEEDS] for n in ns]
        mm = [sum(t) / 2 for t in v]
        ax2.fill_between(ns, [min(t) for t in v], [max(t) for t in v], color=c, alpha=0.2, lw=0)
        ax2.plot(ns, mm, "-o", color=c, ms=6, lw=3)
        ax2.text(4600, mm[-1] + dy, lab, color=c, fontsize=16, va="center", fontweight="bold")
    ax2.set_xscale("log"); ax2.set_xticks([250, 1000, 4000]); ax2.set_xticklabels(["250", "1k", "4k"])
    ax2.minorticks_off(); ax2.set_xlim(220, 4000 * 2.6); ax2.set_ylim(0, 1.0); ax2.set_yticks([0, .5, 1])
    ax2.set_xlabel("training sentences"); ax2.set_ylabel("share")
    ax2.set_title("Learning curves", fontsize=17, fontweight="bold", loc="left")
    fig.tight_layout()
    save(fig, "controlled_regimes_talk.png")


if __name__ == "__main__":
    for n, f in (("parse_v1", parse_v1), ("parse_v2", parse_v2), ("gen_v2", gen_v2), ("learn_v2", learn_v2),
                 ("cs28", cs28), ("regimes", regimes)):
        if want(n):
            f()
