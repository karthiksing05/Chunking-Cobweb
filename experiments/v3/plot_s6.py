"""Figures for study S6 (chess): Chase & Simon (1973) side by side with TRELLIS, and
the chunker comparison on the place-transfer sets as learning scales.

Chase & Simon values: Tables 6 and 8 are copied from the paper; Figs. 1, 2 and 5 are
read off the published figures (to about half a piece).

Usage: plot_s6.py [seed ...]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "docs", "site")
RES = os.path.join(ROOT, "experiments", "v3", "results")
SEEDS = [int(s) for s in sys.argv[1:]] or [13, 17]

INK, MUTED, RULE = "#1f2328", "#6b7280", "#d0d4da"
PLAYER = {"M": "#0072B2", "A": "#E69F00", "B": "#009E73"}          # Okabe-Ito
MARK = {"M": "o", "A": "s", "B": "^"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})

# ---------------------------------------------------------------- Chase & Simon (1973)
CS_FIG1 = {"middle": {"M": [16, 20.5, 24, 25], "A": [8.5, 14, 21, 23.5, 24.5, 24.5, 25], "B": [4.5, 9, 13.5, 17.5, 21, 22.5, 23.5]},
           "random middle": {"M": [2, 5, 6.5, 9, 12.5], "A": [3, 8, 12, 16.5], "B": [2.5, 7, 12.5, 15, 17.5, 21]},
           "end": {"M": [8, 12.5, 13, 13], "A": [7, 11.5, 13, 13], "B": [4.5, 8.5, 10.5, 12, 12.8, 13]},
           "random end": {"M": [3, 7.5, 10, 13], "A": [2, 7.5, 9.5], "B": [3, 7.5, 10, 11]}}
CS_T8 = {"middle": {"M": ([3.8, 3.0, 2.5, 2.3, 1.9, 1.5, 2.2, 2.0], 7.7), "A": ([2.6, 2.5, 1.8, 1.6, 1.7, 1.7, 2.1, 2.5], 5.7),
                    "B": ([2.4, 2.1, 2.0, 1.6, 1.4, 1.5, 1.0, 2.0], 5.3)},
         "end": {"M": ([2.6, 1.6, 1.4, 1.8, 1.8, 1.2, 2.3, 1.0], 7.6), "A": ([2.4, 1.4, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], 6.4),
                 "B": ([2.2, 2.4, 2.2, 1.0, 1.0, 1.0, 1.0, None], 4.2)}}
RELS = ["-", "A", "P", "C", "S", "AP", "AS", "DC", "PC", "PS", "CS", "APS", "DPC", "DCS", "PCS", "DPCS"]
CS_T6 = {"within": [.031, .010, .004, .089, .029, .015, .010, .054, .027, .008, .046, .004, .198, .079, .216, .179],
         "between": [.258, .018, .010, .224, .070, .023, .008, .063, .031, .003, .047, 0, .190, .016, .013, .026],
         "prior": [.320, .0201, .0057, .255, .148, .0077, .0025, .0423, .0159, .0075, .0939, .0022, .0469, .0057, .0105, .0162]}
CS_FIG5 = [5.8, 4.15, 3.25, 1.75, 1.0]


def load_cs():
    runs = []
    for s in SEEDS:
        p = os.path.join(RES, "chess_recall", f"cs73_s{s}.json")
        if os.path.exists(p):
            runs.append(json.load(open(p, encoding="utf-8")))
    return runs


def avg(runs, f):
    vals = [f(r) for r in runs]
    if isinstance(vals[0], list):
        return [sum(v[i] for v in vals) / len(vals) for i in range(len(vals[0]))]
    return sum(vals) / len(vals)


def curve_panel(ax, data, title, ymax, label_end=True):
    for p in ("M", "A", "B"):
        ys = data[p]
        ax.plot(range(1, len(ys) + 1), ys, color=PLAYER[p], marker=MARK[p], ms=4.5, lw=1.6,
                ls="-" if p == "M" else "--" if p == "A" else ":")
        if label_end:
            ax.annotate(p, (len(ys), ys[-1]), xytext=(4, 0), textcoords="offset points", color=PLAYER[p],
                        fontweight="bold", va="center", fontsize=8.5)
    ax.set_xlim(0.6, 7.6)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(1, 8))
    ax.set_title(title, loc="left")


def fig_recall(runs, phase, n_pieces, fname):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    model = {k: {p: avg(runs, lambda r: r["players"][p][k]["pieces per trial"]) for p in ("M", "A", "B")}
             for k in (phase, "random " + phase)}
    ymax = n_pieces + 2
    for col, (src, data) in enumerate((("Chase & Simon (1973)", CS_FIG1), ("TRELLIS", model))):
        curve_panel(axes[0, col], data[phase], f"{src} · {'middle' if phase == 'middle' else 'end'} games", ymax)
        curve_panel(axes[1, col], data["random " + phase], f"{src} · random {'middle' if phase == 'middle' else 'end'} games", ymax)
        for ax in axes[:, col]:
            ax.axhline(n_pieces, color=RULE, lw=1, zorder=0)
    for ax in axes[1]:
        ax.set_xlabel("Trial")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pieces placed correctly")
    fig.text(0.5, -0.02, "M master · A class A · B beginner.  TRELLIS: memory after 4,000 / 1,000 / 300 positions; "
             "7 units per 5-s view, previous reconstruction kept.", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def fig_chunks(runs):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    for row, phase in enumerate(("middle", "end")):
        for col, src in enumerate(("Chase & Simon (1973)", "TRELLIS")):
            ax = axes[row, col]
            for p in ("M", "A", "B"):
                if col == 0:
                    sizes, per = CS_T8[phase][p]
                else:
                    sizes = avg(runs, lambda r: r["players"][p][phase]["successive chunk sizes"])
                    per = avg(runs, lambda r: r["players"][p][phase]["chunks per trial"])
                xs = [i + 1 for i, v in enumerate(sizes) if v is not None]
                ys = [v for v in sizes if v is not None]
                ax.plot(xs, ys, color=PLAYER[p], marker=MARK[p], ms=4, lw=1.5)
                ax.annotate(f"{p}  {per:.1f}/trial", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                            color=PLAYER[p], fontsize=7.5, va="center")
            ax.set_title(f"{src} · {phase} games", loc="left")
            ax.set_xticks(range(1, 9))
            ax.set_xlim(0.6, 10.2)
    for ax in axes[1]:
        ax.set_xlabel("Successive chunk on trial 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pieces per chunk")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cs_chunks.png"))
    plt.close(fig)


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((y - mb) ** 2 for y in b) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float("nan")


def fig_relations(runs):
    model = {part: [avg(runs, lambda r: r["players"]["M"]["middle"][f"relations {part}"].get(k, 0.0)) for k in RELS]
             for part in ("within", "between", "prior")}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    ys = list(range(len(RELS)))[::-1]
    for ax, (src, d) in zip(axes, (("Chase & Simon (1973), memory task", CS_T6), ("TRELLIS (M, middle games)", model))):
        ax.barh(ys, d["prior"], color=RULE, height=0.7, label="a priori (all pairs)")
        ax.plot(d["within"], ys, "o", color="#0072B2", ms=4.5, label="within a chunk (< 2 s)")
        ax.plot(d["between"], ys, "D", color="#D55E00", ms=3.8, mfc="white", mew=1.3, label="between chunks (> 2 s)")
        ax.set_yticks(ys)
        ax.set_yticklabels(RELS, fontsize=7.5)
        ax.set_xlabel("Probability")
        ax.set_title(src, loc="left")
    axes[0].legend(loc="lower right", fontsize=7.5)
    r_w, r_b = corr(CS_T6["within"], model["within"]), corr(CS_T6["between"], model["between"])
    fig.text(0.5, -0.03, f"A attack · D defense · P proximity · C same colour · S same type.  Correlation with Chase & Simon: "
             f"within r = {r_w:.2f}, between r = {r_b:.2f}.", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cs_relations.png"))
    plt.close(fig)
    return r_w, r_b


def fig_relations_count(runs):
    share = {p: [avg(runs, lambda r: r["players"][p]["middle"]["share in one chunk by relations"][str(k)]) for k in range(5)]
             for p in ("M", "A", "B")}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    axes[0].plot(range(5), CS_FIG5, "o-", color=INK, ms=4, lw=1.5)
    axes[0].set_ylabel("Latency (s)")
    axes[0].set_title("Chase & Simon (1973), Fig. 5", loc="left")
    for p in ("M", "A", "B"):
        axes[1].plot(range(5), share[p], marker=MARK[p], color=PLAYER[p], ms=4, lw=1.5, label=p)
    axes[1].set_ylabel("Pairs held in one chunk")
    axes[1].set_title("TRELLIS (proxy: no latencies)", loc="left")
    axes[1].legend(fontsize=7.5, loc="upper left")
    for ax in axes:
        ax.set_xlabel("Number of relations")
        ax.set_xticks(range(5))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cs_fig5.png"))
    plt.close(fig)


# ---------------------------------------------------------------- chunkers as learning scales
SETS = [("control", "Control"), ("family", "Other openings"), ("wing", "Other wing"), ("shift", "Shifted one file")]
METHODS = [("piece-square list", "#999999", ":"), ("BPE, squares", "#E69F00", "-"), ("BPE, shapes", "#CC79A7", "-"),
           ("Krimp", "#56B4E9", "-"), ("xz", "#000000", "--"), ("TRELLIS", "#0072B2", "-")]
NS = [300, 1000, 2000, 4000]


def bits(method, s, n):
    cc = os.path.join(RES, "chess_compress")
    if method in ("piece-square list", "BPE, squares", "BPE, shapes", "Krimp", "xz"):
        m = {"piece-square list": "bpe-abs", "BPE, squares": "bpe-abs", "BPE, shapes": "bpe-rel", "Krimp": "krimp", "xz": "xz"}[method]
        p = os.path.join(cc, f"scale-{m}-{n}_s13.json")
        if not os.path.exists(p):
            return None
        d = json.load(open(p, encoding="utf-8"))["sets"][s]
        if method == "piece-square list":
            return d.get("merges 0")
        return d.get("best", d.get("bits per position"))
    best = None
    for arm in ("abs", "rel"):
        p = os.path.join(cc, f"xfer-{arm}-{s}_s13.json" if n == 4000 else f"scale-{arm}-{s}-{n}_s13.json")
        if os.path.exists(p):
            rows = [r["bits per position"] for r in json.load(open(p, encoding="utf-8"))["by levels"]
                    if r.get("copy") is True]
            if rows:
                best = min(rows) if best is None else min(best, min(rows))
    return best


def fig_scaling():
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True)
    table = {}
    for ax, (s, title) in zip(axes, SETS):
        for name, color, ls in METHODS:
            pts = [(n, bits(name, s, n)) for n in NS]
            pts = [(n, b) for n, b in pts if b is not None]
            table[(s, name)] = dict(pts)
            if pts:
                ax.plot(*zip(*pts), color=color, ls=ls, lw=2.2 if name == "TRELLIS" else 1.3,
                        marker="o" if name == "TRELLIS" else None, ms=3.5, label=name)
        ax.set_xscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels(["300", "1k", "2k", "4k"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Positions learned")
    axes[0].set_ylabel("Bits per held-out position")
    axes[-1].legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chess_chunkers.png"))
    plt.close(fig)
    return table


if __name__ == "__main__":
    runs = load_cs()
    if runs:
        fig_recall(runs, "middle", 25, "cs_fig1.png")
        fig_recall(runs, "end", 13, "cs_fig2.png")
        fig_chunks(runs)
        print("relations r (within, between):", fig_relations(runs))
        fig_relations_count(runs)
        for p in ("novice", "B", "A", "M"):
            r = runs[0]["players"][p]
            print(p, {k: r[k]["pieces per trial"][:3] for k in ("middle", "random middle", "end", "random end")})
    t = fig_scaling()
    print(json.dumps({f"{s}|{m}": v for (s, m), v in t.items()}))
