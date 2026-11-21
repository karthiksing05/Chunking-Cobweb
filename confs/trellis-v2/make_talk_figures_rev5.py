"""Deck revision 5: Chase and Simon trials from the paper's protocol (slide 25) and 14 pt text on slides 21, 23, 24.

Slide 25 data: experiments/v3/results/chess_cs1973_exact/cs73x_s{13,17,19}.json, memory K=7, middle-game positions,
"curve k" = pieces correct on trial k (repeated views, previous reconstruction kept, capped at seven chunk slots),
ten simulated subjects per seed sharing one memory, mean over seeds. Trial 1 is the paper's first-trial recall
(7.2, 9.7, 12.9 after 300, 1,000 and 4,000 positions; supplement Table S10, main text Section Chase and Simon).
People: the digitised Chase and Simon (1973) Figure 1 curves in plot_s6.CS_FIG1.
Run on a compute node:  python confs/trellis-v2/make_talk_figures_rev5.py [cs25 gen21 chess23 comp24]"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "talk_figures")
RES = os.path.join(ROOT, "experiments", "v3", "results")
WANT = set(sys.argv[1:])
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GRID, MUTED = "#e4e3dc", "#555450"


def want(n):
    return not WANT or n in WANT


def style(size=16):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": size, "axes.edgecolor": "#555",
                         "axes.labelcolor": "#222", "xtick.color": "#333", "ytick.color": "#333",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "axes.linewidth": 1.4, "xtick.labelsize": size,
                         "ytick.labelsize": size, "axes.labelsize": size})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print("wrote", p)


def cs25():
    sys.path.insert(0, os.path.join(ROOT, "experiments", "v3"))
    argv = sys.argv
    sys.argv = [argv[0]]
    import plot_s6 as m
    sys.argv = argv
    style(16)
    base = os.path.join(RES, "chess_cs1973_exact")
    runs = [json.load(open(os.path.join(base, f"cs73x_s{s}.json"), encoding="utf-8")) for s in (13, 17, 19)]
    NT = 7

    def curve(n):
        out = []
        for k in range(1, NT + 1):
            out.append(sum(r["points"][str(n)]["memory"]["K=7"]["middle"][f"curve {k}"]["mean"] for r in runs) / len(runs))
        return out
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    ax = axes[0]
    for p, lab in (("M", "master"), ("A", "class A"), ("B", "beginner")):
        ys = m.CS_FIG1["middle"][p]
        ax.plot(range(1, len(ys) + 1), ys, color=m.PLAYER[p], marker=m.MARK[p], ms=8, lw=3.2,
                ls="-" if p == "M" else "--" if p == "A" else ":", label=lab)
    ax.set_title("People", loc="left", fontsize=17, fontweight="bold")
    ax.legend(loc="lower right", fontsize=15, frameon=False, handlelength=2.6)
    ax = axes[1]
    spec = ((4000, BLUE, "o", "-", "4,000 positions"), (1000, ORANGE, "s", "--", "1,000 positions"),
            (300, GREEN, "^", ":", "300 positions"))
    for n, c, mk, ls, lab in spec:
        ax.plot(range(1, NT + 1), curve(n), color=c, marker=mk, ms=8, lw=3.2, ls=ls, label=lab)
    ax.set_title("TRELLIS, 7 chunks", loc="left", fontsize=17, fontweight="bold")
    ax.legend(loc="lower right", fontsize=15, frameon=False, handlelength=2.6)
    for a in axes:
        a.set_xlim(0.6, 7.4); a.set_ylim(0, 27); a.set_xticks([1, 3, 5, 7]); a.set_yticks([0, 10, 20])
        a.set_xlabel("trial"); a.grid(axis="y", color=GRID, lw=1)
    axes[0].set_ylabel("pieces correct")
    fig.tight_layout()
    save(fig, "cs_trials_talk.png")


def gen21():
    style(16)
    gn = [250, 500, 1000, 4000]
    fc = {"syntax": {"TRELLIS": [.3030, .3718, .4096, .4163], "LSTM": [.2877, .3731, .4200, .4729],
                     "GPT": [.2828, .3720, .4284, .4716]},
          "syntax and meaning": {"TRELLIS": [.1890, .2304, .2584, .2220], "LSTM": [.1828, .2435, .2840, .2916],
                                 "GPT": [.1671, .2265, .2711, .3042]}}
    cols = {"TRELLIS": BLUE, "LSTM": RED, "GPT": GREEN}
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.5), gridspec_kw={"wspace": 0.38})
    for ax, (title, series) in zip(axs, fc.items()):
        for name, y in series.items():
            ax.plot(gn[:3], y[:3], "-", color=cols[name], lw=3.4 if name == "TRELLIS" else 2.4, marker="o", ms=7,
                    label=name)
            ax.plot(gn[2:], y[2:], ":", color=cols[name], lw=2.4)
            ax.plot(gn[3:], y[3:], "o", color=cols[name], ms=8, mfc="white", mew=2)
        ax.set_xscale("log", base=2); ax.set_xticks(gn); ax.set_xticklabels(["250", "500", "1k", "4k"])
        ax.minorticks_off()
        ax.set_xlabel("training sentences"); ax.set_title("Judge: " + title, fontsize=16, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=1)
    axs[0].set_ylabel("fair count")
    h, l = axs[0].get_legend_handles_labels()
    h.append(Line2D([], [], color="#444", ls=":", marker="o", mfc="white", mew=2, ms=8, lw=2.4))
    l.append("4k: two seeds")
    axs[0].legend(h, l, frameon=False, fontsize=14, loc="lower right", handlelength=2.2)
    fig.tight_layout()
    save(fig, "generation_fair_count_talk.png")


def chess23():
    style(16)
    GLYPH = {"wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
             "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟"}
    LIGHT, DARK = "#efe6d6", "#c9b79c"
    RING = ["#2a78d6", "#d6336c", "#1baf7a", "#eda100", "#8a5cd6", "#eb6834", "#6b6a64"]
    d = json.load(open(os.path.join(RES, "chess_ideas", "ideas_s13.json"), encoding="utf-8"))
    cps = [c["learned"] for c in d["checkpoints"]]
    case = d["cases"][0]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.15))
    for ax, cp in zip(axes, (300, 1000, 4000)):
        snap = case["by checkpoint"][cps.index(cp)]
        chunks = snap["chunks"][:len(RING)]
        for f in range(8):
            for r in range(8):
                ax.add_patch(Rectangle((f, r), 1, 1, color=LIGHT if (f + r) % 2 else DARK, lw=0))
        for k, ch in enumerate(chunks):
            for name, _ in ch["pieces"]:
                f, r = "abcdefgh".index(name[0]), int(name[1]) - 1
                ax.add_patch(Rectangle((f + 0.07, r + 0.07), 0.86, 0.86, fill=False, ec=RING[k], lw=3))
        for name, tok in case["fen"].items():
            f, r = "abcdefgh".index(name[0]), int(name[1]) - 1
            ax.text(f + 0.5, r + 0.47, GLYPH[tok], ha="center", va="center", fontsize=17)
        ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{cp:,} learned", fontsize=17, fontweight="bold")
        labs = [(ch["ideas"][0] if ch["ideas"] else None) for ch in chunks][:3]
        ax.set_xlabel("\n".join(f"{k + 1}: {t}" for k, t in enumerate(labs) if t) or "no chunk", fontsize=16,
                      color=MUTED, labelpad=4, linespacing=1.1)
    fig.tight_layout()
    save(fig, "chess_ideas_cases_talk.png")


def comp24():
    sys.path.insert(0, os.path.join(ROOT, "experiments", "v3"))
    argv = sys.argv
    sys.argv = [argv[0]]
    import plot_s6 as m
    sys.argv = argv
    style(16)
    CC = os.path.join(m.RES, "chess_compress")
    STD = {"control": "ab-control-control", "shift": "ab-control-shift"}

    def std_bits(s):
        dd = json.load(open(os.path.join(CC, STD[s] + "_s13.json"), encoding="utf-8"))
        return min(r["bits per position"] for r in dd["by levels"] if r.get("copy") is True)
    names = {"BPE, squares": "BPE, squares", "BPE, shapes": "BPE, shapes", "TRELLIS": "TRELLIS, best"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    for ax, (s, title) in zip(axes, (("control", "Control positions"), ("shift", "Shifted one file"))):
        for name, color, ls in m.METHODS:
            if name not in names:
                continue
            pts = [(n, m.bits(name, s, n)) for n in m.NS]
            pts = [(n, b) for n, b in pts if b is not None]
            ax.plot(*zip(*pts), color=color, ls=ls, lw=3.6 if name == "TRELLIS" else 2.6,
                    marker="o" if name == "TRELLIS" else None, ms=8, mfc="white" if name == "TRELLIS" else None,
                    label=names[name])
        b = std_bits(s)
        ax.plot([4000], [b], "o", color="#D55E00", ms=11, zorder=5, label="TRELLIS, standard")
        ax.annotate(f"{b:.1f}", (4000, b), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=15,
                    color="#D55E00", fontweight="bold")
        ax.set_xscale("log"); ax.set_xticks(m.NS); ax.minorticks_off(); ax.set_xticklabels(["300", "1k", "2k", "4k"])
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold"); ax.set_xlabel("positions learned")
        ax.grid(color=GRID, lw=1)
    axes[0].set_ylabel("bits per position")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "chess_chunkers_talk.png")


if __name__ == "__main__":
    for n, f in (("cs25", cs25), ("gen21", gen21), ("chess23", chess23), ("comp24", comp24)):
        if want(n):
            f()
