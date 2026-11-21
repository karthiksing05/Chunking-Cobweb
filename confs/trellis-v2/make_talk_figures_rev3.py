"""Talk versions of the figures that were too small to read on a projector (deck revision 3).

Writes PNGs with a _talk suffix to confs/trellis-v2/talk_figures/. The paper's figures are never touched.
Each figure uses fewer panels, larger fonts and thicker lines than its paper counterpart, and is
sized so its text stays at or above about 12 pt when the image fills the slide width.

Data and drawing helpers come from experiments/v3 (story_chunks_plot, chess_ideas_plot, chess_case_plot,
chess_cs1973_plot, plot_s6) and from confs/acs-26/make_paper_figures.py (hierarchies).
Run from the repo root on a compute node: python confs/trellis-v2/make_talk_figures_rev3.py [name ...]
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "talk_figures")
RES = os.path.join(ROOT, "experiments", "v3", "results")
os.makedirs(OUT, exist_ok=True)
WANT = set(sys.argv[1:])

INK, MUTED, GRID, ACC, HELD = "#1f1f1d", "#555450", "#e4e3dc", "#2a78d6", "#1baf7a"
GLYPH = {"wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
         "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟"}
LIGHT, DARK = "#efe6d6", "#c9b79c"
RING = ["#2a78d6", "#d6336c", "#1baf7a", "#eda100", "#8a5cd6", "#eb6834", "#6b6a64"]


def base_style(size=14):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": size, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": size + 1,
                         "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
                         "figure.facecolor": "white", "axes.linewidth": 1.3})


def save(fig, name, **kw):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=kw.pop("dpi", 200), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print("wrote", path)


def want(name):
    return not WANT or name in WANT


def mean(cols):
    out = []
    for vals in zip(*cols):
        v = [x for x in vals if x is not None and not math.isnan(x)]
        out.append(sum(v) / len(v) if v else float("nan"))
    return out


def board(ax, pieces, rings=None, fade=None, size=20, lw=3.0):
    for f in range(8):
        for r in range(8):
            ax.add_patch(Rectangle((f, r), 1, 1, color=LIGHT if (f + r) % 2 else DARK, lw=0))
    for k, group in enumerate(rings or []):
        for name in group:
            f, r = "abcdefgh".index(name[0]), int(name[1]) - 1
            ax.add_patch(Rectangle((f + 0.07, r + 0.07), 0.86, 0.86, fill=False, ec=RING[k % len(RING)], lw=lw))
    for name, tok in pieces:
        f, r = "abcdefgh".index(name[0]), int(name[1]) - 1
        ax.text(f + 0.5, r + 0.47, GLYPH[tok], ha="center", va="center", fontsize=size,
                alpha=0.2 if (fade is not None and name not in fade) else 1.0, family="DejaVu Sans")
    ax.set_xlim(0, 8), ax.set_ylim(0, 8), ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------- story chunks: cases
def story_cases():
    base_style(14)
    runs = [json.load(open(os.path.join(RES, "story_chunks", f"v2_s{s}.json"), encoding="utf-8")) for s in (13, 17)]
    cases = runs[0]["cases"]
    cps = [c["learned"] for c in runs[0]["checkpoints"]]
    keep_cp = [250, 4000]
    keep_rows = [1, 3]            # tim had an idea; i loved that toy car
    names = {1: "tim had an idea", 3: "i loved that toy car", 0: "tom was sad and scared"}

    def bracketed(words, chunks):
        opens, closes = [[] for _ in words], [[] for _ in words]
        for ch in sorted(chunks, key=lambda c: (c["span"][0], -c["span"][1])):
            i, j = ch["span"]
            o, c = ("[", "]") if ch["recognised"] else ("(", ")")
            opens[i].append(o)
            closes[j - 1].insert(0, c)
        return " ".join("".join(opens[k]) + w + "".join(closes[k]) for k, w in enumerate(words))

    fig, ax = plt.subplots(figsize=(10.0, 2.5))
    ax.axis("off")
    n = len(keep_rows)
    for j, cp in enumerate(keep_cp):
        ax.text((j + 0.5) / 2, 1.0, f"{cp:,} learned", ha="center", va="bottom", fontsize=16, weight="bold", color=INK,
                transform=ax.transAxes)
    for i, r in enumerate(keep_rows):
        case = cases[r]
        words = case["sentence"].split()
        y = 1 - (i + 0.55) / n
        for j, cp in enumerate(keep_cp):
            snap = case["by checkpoint"][cps.index(cp)]
            ax.text((j + 0.5) / 2, y, bracketed(words, snap["chunks"]), ha="center", va="center", fontsize=17,
                    color=INK, family="DejaVu Sans Mono", transform=ax.transAxes)
        ax.plot([0, 1], [1 - (i + 1) / n] * 2, color=GRID, lw=1, transform=ax.transAxes, clip_on=False)
    save(fig, "story_chunks_cases_talk.png")


# ---------------------------------------------------------------- story chunks: phrase curves
def story_phrases():
    base_style(15)
    runs = [json.load(open(os.path.join(RES, "story_chunks", f"v2_s{s}.json"), encoding="utf-8")) for s in (13, 17)]
    cps = [c["learned"] for c in runs[0]["checkpoints"]]

    def ser(key, lab):
        return [[c[key][lab] if c[key][lab] is not None else float("nan") for c in r["checkpoints"]] for r in runs]

    labs = ["crosses a boundary", "inside one phrase"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for k, (ax, lab) in enumerate(zip(axes, labs)):
        s = ser("rate", lab)
        for line in s:
            ax.plot(cps, line, color=ACC, lw=1.0, alpha=0.35)
        ax.plot(cps, mean(s), color=ACC, lw=3.4, marker="o", ms=8, label="committed chunks")
        ax.plot(cps, mean(ser("chance random", lab)), color=MUTED, lw=2.6, ls="--", label="random spans")
        ax.plot(cps, mean(ser("held-out rate", lab)), color=HELD, lw=2.6, marker="s", ms=6, label="held-out recognised")
        ax.set_title(lab, loc="left")
        ax.set_xscale("log")
        ax.set_xticks(cps)
        ax.set_xticklabels(["250", "500", "1k", "2k", "4k"])
        ax.minorticks_off()
        ax.grid(color=GRID, lw=0.8)
        ax.set_xlabel("sentences learned")
        if k == 0:
            ax.set_ylabel("share of committed chunks")
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncol=3, fontsize=14, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "story_chunks_phrases_talk.png")


# ---------------------------------------------------------------- chess ideas: cases
def chess_ideas_cases():
    base_style(14)
    d = json.load(open(os.path.join(RES, "chess_ideas", "ideas_s13.json"), encoding="utf-8"))
    cps = [c["learned"] for c in d["checkpoints"]]
    keep_cp, keep_rows = [300, 1000, 4000], [0]
    fig, axes = plt.subplots(len(keep_rows), 3, figsize=(11.5, 4.9), squeeze=False)
    for i, r in enumerate(keep_rows):
        case = d["cases"][r]
        for j, cp in enumerate(keep_cp):
            ax = axes[i][j]
            snap = case["by checkpoint"][cps.index(cp)]
            chunks = snap["chunks"][:len(RING)]
            board(ax, list(case["fen"].items()), rings=[[n for n, _ in ch["pieces"]] for ch in chunks], size=19, lw=2.8)
            if i == 0:
                ax.set_title(f"{cp:,} learned")
            lab = [", ".join(ch["ideas"]) or "-" for ch in chunks][:3]
            ax.set_xlabel("\n".join(f"{k + 1}: {t}" for k, t in enumerate(lab)) or "no chunk recognised", fontsize=13,
                          color=MUTED, labelpad=6)
    fig.tight_layout()
    save(fig, "chess_ideas_cases_talk.png")


# ---------------------------------------------------------------- chess ideas: curves
def chess_ideas_curves():
    base_style(15)
    runs = [json.load(open(os.path.join(RES, "chess_ideas", f"ideas_s{s}.json"), encoding="utf-8"))
            for s in (13, 17, 19, 23, 29)]
    cps = [c["learned"] for c in runs[0]["checkpoints"]]

    def ser(key, idea):
        return [[c[key][idea] if c[key][idea] is not None else float("nan") for c in r["checkpoints"]] for r in runs]

    ideas = ["pawn chain", "pawn ram", "attack"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.0))
    for k, (ax, idea) in enumerate(zip(axes, ideas)):
        s = ser("rate", idea)
        for line in s:
            ax.plot(cps, line, color=ACC, lw=1.0, alpha=0.3)
        ax.plot(cps, mean(s), color=ACC, lw=3.4, marker="o", ms=8, label="learned chunks")
        ax.plot(cps, mean(ser("chance nearby", idea)), color=MUTED, lw=2.6, ls="--", label="nearby chance")
        ax.set_title(idea, loc="left")
        ax.set_xscale("log")
        ax.set_xticks([100, 1000])
        ax.set_xticklabels(["100", "1k"])
        ax.minorticks_off()
        ax.grid(color=GRID, lw=0.8)
        ax.set_xlabel("positions learned")
        if k == 0:
            ax.set_ylabel("share of chunks")
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncol=2, fontsize=14, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "chess_ideas_curves_talk.png")


# ---------------------------------------------------------------- one position, real and scrambled
def chess_case():
    base_style(15)
    d = json.load(open(os.path.join(RES, "chess_case", "case_s13.json"), encoding="utf-8"))
    case = d["cases"][6]   # the case drawn in the site figure (recall .25, .82, .86 real; .57 scrambled)
    cps = {c["learned"]: c for c in case["checkpoints"]}
    panels = [("real", 0), ("real", 4000), ("scrambled", 0), ("scrambled", 4000)]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.1))
    for ax, (which, n) in zip(axes, panels):
        cp = cps[n][which] if "kept" in cps[n].get(which, {}) else cps[n][which]
        kept = cp["kept"]
        recalled = {name for u in kept for name, _ in u}
        groups = [[name for name, _ in u] for u in kept if len(u) > 1]
        board(ax, list(case[which].items()), rings=groups, fade=recalled, size=20, lw=2.8)
        ax.set_title(f"{which}, {n:,} learned\nrecall {cp['recall']:.2f}", fontsize=17)
    fig.tight_layout()
    save(fig, "chess_case_talk.png")


# ---------------------------------------------------------------- Chase and Simon
def _cs_module():
    here = sys.argv
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, os.path.join(ROOT, "experiments", "v3"))
    import plot_s6 as m
    sys.argv = here
    return m


def cs_trials():
    m = _cs_module()
    base_style(15)
    runs = m.load_cs()
    model = {k: {p: m.avg(runs, lambda r, k=k, p=p: r["players"][p][k]["pieces per trial"]) for p in ("M", "A", "B")}
             for k in ("middle", "random middle")}
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.1), sharex=True)
    for col, (src, data) in enumerate((("Chase & Simon (1973)", m.CS_FIG1), ("TRELLIS", model))):
        for row, (key, word) in enumerate((("middle", "middle-game positions"), ("random middle", "random positions"))):
            ax = axes[row, col]
            for p in ("M", "A", "B"):
                ys = data[key][p]
                ax.plot(range(1, len(ys) + 1), ys, color=m.PLAYER[p], marker=m.MARK[p], ms=7, lw=2.8,
                        ls="-" if p == "M" else "--" if p == "A" else ":")
                if col == 0:
                    ax.annotate(p, (len(ys), ys[-1]), xytext=(len(ys) + 0.2, ys[-1]), color=m.PLAYER[p],
                                fontweight="bold", va="center", fontsize=15)
            ax.axhline(25, color=GRID, lw=1.2, zorder=0)
            ax.set_xlim(0.6, 7.8)
            ax.set_ylim(0, 27)
            ax.set_xticks(range(1, 8))
            ax.set_title(f"{src}\n{word}", loc="left", fontsize=13)
            if col == 0:
                ax.set_ylabel("pieces correct")
            if row == 1:
                ax.set_xlabel("trial")
    fig.tight_layout()
    save(fig, "cs_trials_talk.png")


def cs_recall():
    base_style(15)
    from collections import OrderedDict
    base = os.path.join(RES, "chess_cs1973_exact")
    groups = OrderedDict()
    for tag, seeds in (("cs73x", (13, 17, 19)), ("cs73L", (13, 17))):
        groups[tag] = [json.load(open(os.path.join(base, f"{tag}_s{s}.json"), encoding="utf-8")) for s in seeds]
    HEAD = [300, 1000, 4000]
    HUMAN = {"middle": (4, 8, 16), "random": (3.0, 2.5, 2.0)}
    COL = {"K=7": "#2a78d6", "K=4": "#d6336c"}

    def pts(r):
        return sorted((int(k), v) for k, v in r["points"].items() if int(k) > 0)

    def get(r, K, typ):
        out = []
        for n, p in pts(r):
            if typ == "random":
                a = p["memory"][K]["random middle"]["trial 1 correct"]["mean"]
                b = p["memory"][K]["random end"]["trial 1 correct"]["mean"]
                out.append((n, (a + b) / 2))
            else:
                v = p["memory"][K][typ]["trial 1 correct"]["mean"]
                out.append((n, v if v is not None else float("nan")))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, typ, title in zip(axes, ("middle", "random"), ("middle game", "random positions")):
        for gi, (tag, runs) in enumerate(groups.items()):
            for K in ("K=7", "K=4"):
                series = [get(r, K, typ) for r in runs]
                L = min(len(s) for s in series)
                xs = [n for n, _ in series[0][:L]]
                mm = [sum(s[i][1] for s in series) / len(series) for i in range(L)]
                ax.plot(xs, mm, color=COL[K], lw=3.2 if gi == 0 else 2.6, ls="-" if gi == 0 else "--",
                        marker="o" if gi == 0 else "^", ms=7)
        ax.plot(HEAD, HUMAN[typ], ls="none", marker="s", ms=11, mfc="none", mec=INK, mew=2)
        for x, v, t in zip(HEAD, HUMAN[typ], "BAM"):
            ax.annotate(t, (x, v), xytext=(8, 5), textcoords="offset points", fontsize=14, color=INK, fontweight="bold")
        ax.set_title(title, loc="left", fontsize=16)
        ax.set_xscale("log")
        ax.set_xlabel("positions learned")
        ax.grid(color=GRID, lw=0.8)
    axes[0].set_ylabel("trial-1 pieces correct")
    handles = [Line2D([], [], color=COL["K=7"], lw=3.2, marker="o", ms=7, label="TRELLIS, 7 chunk slots"),
               Line2D([], [], color=COL["K=4"], lw=3.2, marker="o", ms=7, label="TRELLIS, 4 slots"),
               Line2D([], [], color=MUTED, lw=2.6, ls="--", marker="^", ms=7, label="long stream, 27,459 positions"),
               Line2D([], [], color=INK, ls="none", marker="s", ms=11, mfc="none", mew=2, label="people: B, A, M")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=14, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save(fig, "chess_cs1973_recall_talk.png")


# ---------------------------------------------------------------- compression
def chess_chunkers():
    m = _cs_module()
    base_style(15)
    CC = os.path.join(m.RES, "chess_compress")
    STD = {"control": "ab-control-control", "shift": "ab-control-shift"}

    def std_bits(s):
        d = json.load(open(os.path.join(CC, STD[s] + "_s13.json"), encoding="utf-8"))
        return min(r["bits per position"] for r in d["by levels"] if r.get("copy") is True)

    sets = [("control", "Control positions"), ("shift", "Shifted one file")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    names = {"BPE, squares": "BPE over squares", "BPE, shapes": "BPE over shapes", "TRELLIS": "TRELLIS, best variant",
             "piece-square list": "piece-square list"}
    for ax, (s, title) in zip(axes, sets):
        for name, color, ls in m.METHODS:
            if name not in ("BPE, squares", "BPE, shapes", "TRELLIS"):
                continue
            pts = [(n, m.bits(name, s, n)) for n in m.NS]
            pts = [(n, b) for n, b in pts if b is not None]
            if pts:
                ax.plot(*zip(*pts), color=color, ls=ls, lw=3.6 if name == "TRELLIS" else 2.6,
                        marker="o" if name == "TRELLIS" else None, ms=8, mfc="white" if name == "TRELLIS" else None,
                        label=names[name])
        b = std_bits(s)
        ax.plot([4000], [b], "o", color="#D55E00", ms=11, zorder=5, label="TRELLIS, standard")
        ax.annotate(f"{b:.1f}", (4000, b), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=14,
                    color="#D55E00", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(m.NS)
        ax.minorticks_off()
        ax.set_xticklabels(["300", "1k", "2k", "4k"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("positions learned")
        ax.grid(color=GRID, lw=0.8)
    axes[0].set_ylabel("bits per held-out position")
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncol=4, fontsize=13, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    save(fig, "chess_chunkers_talk.png")


# ---------------------------------------------------------------- hierarchies
def hierarchies():
    sys.path.insert(0, os.path.join(ROOT, "confs", "acs-26"))
    import types
    stub = types.ModuleType("parse_mh")
    stub.TRELLIS = stub.PrimitiveParseNode = stub.CompositeParseNode = object
    sys.modules["parse_mh"] = stub   # only the node-drawing helpers are used here
    import make_paper_figures as mp
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.15))
    W, H, FS = 3.45, 2.1, 1.32
    for ax in axes:
        ax.set_xlim(0, 7.2), ax.set_ylim(0, 5.15), ax.set_axis_off()
    # content taxonomy
    ax = axes[0]
    top, bot = 3.6, 1.2
    mp._cobweb_node(ax, 3.6, top, W, H, title="CNT_CONCEPT-300001", count=4, font_scale=FS,
                    attrs=[("content-left", [("CTX_CONCEPT-121001", 1)]), ("content-right", [("CTX_CONCEPT-111002", 1)])])
    mp._cobweb_node(ax, 1.7, bot, W, H, title="CNT_CONCEPT-310001", count=2, font_scale=FS,
                    attrs=[("content-left", [("CTX_CONCEPT-121001", 1)]), ("content-right", [("CTX_CONCEPT-111002", 1)])])
    mp._cobweb_node(ax, 5.5, bot, W, H, title="CNT_CONCEPT-320001", count=2, font_scale=FS,
                    attrs=[("content-left", [("CTX_CONCEPT-140287", 1)]), ("content-right", [("CTX_CONCEPT-145561", 1)])])
    for x in (1.7, 5.5):
        ax.add_line(Line2D([3.6, x], [top - H / 2, bot + H / 2], color="#888", lw=1.6))
    # context taxonomy
    ax = axes[1]
    ctx = lambda a, b: [("left-1", [(a, 1)]), ("right-1", [(b, 1)])]
    mp._cobweb_node(ax, 3.6, top, W, H, title="CTX_CONCEPT-100000", count=4, font_scale=FS,
                    attrs=ctx("big", "saw"), title_fill="#2f7a55")
    mp._cobweb_node(ax, 1.7, bot, W, H, title="CTX_CONCEPT-110001", count=2, font_scale=FS,
                    attrs=ctx("big", "saw"), title_fill="#2f7a55")
    mp._cobweb_node(ax, 5.5, bot, W, H, title="CTX_CONCEPT-120001", count=2, font_scale=FS,
                    attrs=ctx("cat", "a"), title_fill="#2f7a55")
    for x in (1.7, 5.5):
        ax.add_line(Line2D([3.6, x], [top - H / 2, bot + H / 2], color="#888", lw=1.6))
    axes[0].set_title("Content taxonomy", fontsize=22, fontweight="bold", color="#1f4e79", loc="center", y=0.965)
    axes[1].set_title("Context taxonomy", fontsize=22, fontweight="bold", color="#2f7a55", loc="center", y=0.965)
    fig.tight_layout()
    save(fig, "hierarchies_talk.png")




# ---------------------------------------------------------------- instances (R1, R2)
def instances():
    import types
    stub = types.ModuleType("parse_mh")
    stub.TRELLIS = stub.PrimitiveParseNode = stub.CompositeParseNode = object
    sys.modules["parse_mh"] = stub
    sys.path.insert(0, os.path.join(ROOT, "confs", "acs-26"))
    import make_paper_figures as mp
    from matplotlib.patches import Circle, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(15.4, 3.95))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 15.4), ax.set_ylim(0, 3.95), ax.set_axis_off()
    FS, W = 1.3, 3.4
    # context instance of the primitive "cat"
    mp._cobweb_node(ax, 1.75, 1.95, W, 3.4, title="cat", count=None, font_scale=FS, title_fill="#1f7a4d",
                    attrs=[("left-2", [("EMPTYNULL", 1)]), ("left-1", [("the", 1)]),
                           ("right+1", [("saw", 1)]), ("right+2", [("a", 1)])])
    # content instance and context instance of the composite "saw a dog"
    mp._cobweb_node(ax, 10.45, 2.3, W, 2.3, title="CNT_CONCEPT-321001", count=None, font_scale=FS,
                    attrs=[("content-left", [("CTX_CONCEPT-264738", 1)]), ("content-right", [("CTX_CONCEPT-247665", 1)])])
    mp._cobweb_node(ax, 13.75, 1.95, 3.1, 3.4, title="CTX_CONCEPT-198754", count=None, font_scale=FS, title_fill="#1f7a4d",
                    attrs=[("left-2", [("the", 1)]), ("left-1", [("cat", 1)]),
                           ("right+1", [("EMPTYNULL", 1)]), ("right+2", [("EMPTYNULL", 1)])])
    # parse tree of "the cat saw a dog"
    ly = 0.6
    xs = {"the": 4.5, "cat": 5.5, "saw": 6.5, "a": 7.35, "dog": 8.2}
    for w, x in xs.items():
        ec = "#1f7a4d" if w == "cat" else "#8c4440"
        fc = "#e8f6ee" if w == "cat" else "#f2a9a4"
        ax.add_patch(FancyBboxPatch((x - 0.36, ly - 0.2), 0.72, 0.4, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc=fc, ec=ec, lw=2.6 if w == "cat" else 1.6, zorder=3))
        ax.text(x, ly, w, ha="center", va="center", fontsize=17, family="serif", zorder=4)
    nodes = {"NP1": (5.0, 1.55, "NP"), "NP2": (7.78, 1.55, "NP"), "VP": (7.1, 2.45, "VP"), "S": (6.05, 3.35, "S")}
    edges = [("S", "NP1"), ("S", "VP"), ("VP", "NP2")]
    for a, b in edges:
        ax.add_line(Line2D([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color="#555", lw=1.6, zorder=1))
    for n, ws in (("NP1", ("the", "cat")), ("NP2", ("a", "dog"))):
        for w in ws:
            ax.add_line(Line2D([nodes[n][0], xs[w]], [nodes[n][1], ly + 0.2], color="#555", lw=1.6, zorder=1))
    ax.add_line(Line2D([nodes["VP"][0], xs["saw"]], [nodes["VP"][1], ly + 0.2], color="#555", lw=1.6, zorder=1))
    for n, (x, y, lab) in nodes.items():
        hi = n == "VP"
        ax.add_patch(Circle((x, y), 0.27, fc="#fdf0c8" if hi else "#a9c4ef", ec="#e08a00" if hi else "#3b5e8c",
                            lw=2.6 if hi else 1.6, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", fontsize=16, fontweight="bold", family="serif", zorder=4)
    # pointers: cat -> its context instance; VP -> its content and context instances
    g, o = "#1f7a4d", "#e08a00"
    ax.add_line(Line2D([5.5, 5.5, 1.75], [ly - 0.2, 0.1, 0.1], color=g, lw=2.2, zorder=2))
    ax.annotate("", xy=(1.75, 0.28), xytext=(1.75, 0.1), arrowprops=dict(arrowstyle="-|>", color=g, lw=2.2))
    ax.annotate("", xy=(8.75, 2.3), xytext=(7.37, 2.45), arrowprops=dict(arrowstyle="-|>", color=o, lw=2.2))
    ax.add_line(Line2D([7.1, 7.1, 13.75], [2.72, 3.86, 3.86], color=o, lw=2.2, zorder=2))
    ax.annotate("", xy=(13.75, 3.68), xytext=(13.75, 3.86), arrowprops=dict(arrowstyle="-|>", color=o, lw=2.2))
    save(fig, "instances_talk.png")

if __name__ == "__main__":
    for name, fn in (("story_cases", story_cases), ("story_phrases", story_phrases),
                     ("chess_ideas_cases", chess_ideas_cases), ("chess_ideas_curves", chess_ideas_curves),
                     ("chess_case", chess_case), ("cs_trials", cs_trials), ("cs_recall", cs_recall),
                     ("chess_chunkers", chess_chunkers), ("hierarchies", hierarchies), ("instances", instances)):
        if want(name):
            fn()
