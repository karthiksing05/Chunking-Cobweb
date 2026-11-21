"""Second readability pass for the talk (deck revision 3b): larger fonts for the parse-success curves,
the controlled-grammar figure and the three infographics on slides 6, 7 and 16.
Outputs *_talk.png in confs/trellis-v2/talk_figures/. Run on a compute node:
  python confs/trellis-v2/make_talk_figures_rev3b.py [parse_success r230 v1_parse v1_gen v2_gen]"""
import csv
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.axes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "talk_figures")
WANT = set(sys.argv[1:])
SCALE = float(os.environ.get("TALK_FONT_SCALE", "1.5"))
INK = "#22262e"


def want(n):
    return not WANT or n in WANT


SHORT = {"Inside and outside, both taxonomies": "Inside and outside", "inside: each span from its sub-spans": "inside: from sub-spans", "outside: each span from its surroundings": "outside: from surroundings", "Weight each span by its posterior": "Weight each span", "counts grow by the posterior, not by 1": "counts grow by the posterior",
         "Sample new content leaf": "Sample content leaf", "Sample new context leaves": "Sample context leaves",
         "Write it, close at the boundary": "Write it and close",
         "outside score: sum over chunks": "outside: sum over chunks",
         r"pair gate: (eats, a) seen side by side   $\checkmark$": r"pair gate: (eats, a) seen  $\checkmark$",
         "\"the\" began a stored sentence   $\\checkmark$": "\"the\" began a sentence  $\\checkmark$",
         "\"cat\" ended a stored sentence   $\\checkmark$": "\"cat\" ended a sentence  $\\checkmark$",
"Gate + rank through both hierarchies": "Gate and rank",
         "Commit to partial parse tree": "Commit to the tree",
         r"admit: path-count $=\ 47\ >\ \tau_{\mathrm{parse}}\ =\ 30$   $\checkmark$": r"admit: $47\ >\ 30$  $\checkmark$",
         r"rank: $s_c + s_x\ =\ -5.3$   (best-scoring wins)": r"rank: $-5.3$  (best wins)"}


def scale_text(factor):
    """Multiply every explicit fontsize drawn with Axes.text / annotate by factor."""
    orig_text, orig_ann = matplotlib.axes.Axes.text, matplotlib.axes.Axes.annotate

    def text(self, x, y, s, fontdict=None, **kw):
        s = SHORT.get(s, s)
        if "fontsize" in kw:
            is_title = kw.get("color") == "white" and kw.get("fontweight") == "bold"
            small = s in ("seed", "next seed", "new composite", "sampled leaf", "next element", "new")
            kw["fontsize"] = kw["fontsize"] * (1.25 if is_title else (1.1 if small else (1.0 if s == "expanded" else factor)))
        return orig_text(self, x, y, s, fontdict, **kw)

    def annotate(self, text, xy, *a, **kw):
        if "fontsize" in kw:
            kw["fontsize"] = kw["fontsize"] * factor
        return orig_ann(self, text, xy, *a, **kw)
    matplotlib.axes.Axes.text, matplotlib.axes.Axes.annotate = text, annotate


def parse_success():
    BLUE, GREEN, ORANGE, GREY = "#2a78d6", "#1aa57a", "#d9601a", "#8a8f98"
    plt.rcParams.update({"font.size": 16, "axes.edgecolor": "#6b7280", "axes.labelcolor": INK, "xtick.color": "#4b5563",
                         "ytick.color": "#4b5563", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "axes.linewidth": 1.4})
    rows = list(csv.DictReader(open(os.path.join(ROOT, "experiments/v3/results/story_parse_eval/r216_figure.tsv")), delimiter="\t"))

    def series(score, cls):
        return sorted((int(r["n"]), float(r["p_succ_set_mean"]), float(r["p_succ_gram_mean"]))
                      for r in rows if r["score"] == score and r["set"] == cls)
    LINES = [("TRELLIS parse", "logZ", BLUE, "-", "o"), ("TRELLIS generator", "gen_lp_gate", BLUE, "--", "s"),
             ("LSTM", "LSTM", GREEN, "-", "^"), ("GPT", "GPT", ORANGE, "-", "D"), ("5-gram", "5-gram", GREY, ":", "v")]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
    for ax, (cls, title) in zip(axs, [("C3scr", "Scrambled sentences"), ("C2full", "Content-swapped sentences")]):
        for lab, sc, col, ls, mk in LINES:
            p = series(sc, cls)
            ax.plot([a for a, _, _ in p], [b for _, b, _ in p], color=col, ls=ls, marker=mk, lw=3.2, ms=8, label=lab)
        real = series("logZ", cls)
        ax.axhline(sum(c for _, _, c in real) / len(real), color="#9ca3af", lw=1.6, ls=(0, (1, 2)))
        ax.text(125, 0.955, "real sentences accepted", fontsize=16, color="#4b5563", va="bottom")
        ax.set_xscale("log"); ax.set_xticks([125, 500, 1000, 4000])
        ax.set_xticklabels(["125", "500", "1k", "4k"]); ax.minorticks_off()
        ax.set_title(title, loc="left", fontweight="bold", fontsize=17, color=INK)
        ax.set_xlabel("Training sentences"); ax.grid(axis="y", color="#e5e7eb", lw=1); ax.set_ylim(0, 1.12)
    axs[0].set_ylabel("Share of broken sentences\naccepted (lower is better)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=15, frameon=False, bbox_to_anchor=(0.5, -0.07), handlelength=2.2,
               columnspacing=1.2)
    fig.tight_layout()
    p = os.path.join(OUT, "parse_success_curve_talk.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", p)


def r230():
    sys.path.insert(0, HERE)
    import make_figure_r230 as m
    BLUE, ORANGE, GREEN = m.BLUE, m.ORANGE, m.GREEN
    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
                         "axes.labelcolor": "#222222", "xtick.color": "#333333", "ytick.color": "#333333",
                         "figure.facecolor": "white", "axes.facecolor": "white", "axes.linewidth": 1.4})
    J = m.load()
    levels = ["leaf", 7, 6, 5, 4, 3, 2, 0]
    cols = [BLUE, ORANGE, GREEN]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1.2, 1]})
    x = list(range(len(levels)))
    for j in range(3):
        mm, lo, hi = [], [], []
        for b in levels:
            v = [m.c6(J, s, 1000, b)[j] for s in m.SEEDS]
            mm.append(sum(v) / 2); lo.append(min(v)); hi.append(max(v))
        ax1.fill_between(x, lo, hi, color=cols[j], alpha=0.2, lw=0)
        ax1.plot(x, mm, "-o", color=cols[j], ms=7, lw=3.4)
    ax1.set_xticks(x); ax1.set_xticklabels(["leaf" if b == "leaf" else str(b) for b in levels])
    ax1.set_xlabel("Level $b$ of the word taxonomy (bits)"); ax1.set_ylabel("Share of 1,600 generations")
    ax1.set_ylim(0, 1.0); ax1.set_xlim(-0.3, 9.4)
    ax1.text(7.25, 0.93, "R-non", color=GREEN, fontsize=18, va="center", fontweight="bold")
    ax1.text(7.25, 0.09, "R-sem", color=BLUE, fontsize=18, va="center", fontweight="bold")
    ax1.text(2.6, 0.79, "R-syn", color=ORANGE, fontsize=18, va="center", ha="center", fontweight="bold")
    ax1.set_title("One dial, n = 1,000", fontsize=17, fontweight="bold", loc="left")
    ns = [250, 500, 1000, 2000, 4000]
    spec = [("leaf", 0, BLUE, "R-sem\nleaf"), (6, 1, ORANGE, "R-syn\nb = 6"), (0, 2, GREEN, "R-non\nb = 0")]
    for b, j, c, lab in spec:
        v = [[m.c6(J, s, n, b)[j] for s in m.SEEDS] for n in ns]
        mm = [sum(t) / 2 for t in v]
        ax2.fill_between(ns, [min(t) for t in v], [max(t) for t in v], color=c, alpha=0.2, lw=0)
        ax2.plot(ns, mm, "-o", color=c, ms=7, lw=3.4)
        ax2.text(4500, mm[-1] + (0.0 if j == 2 else (0.07 if j == 0 else -0.09)), lab, color=c, fontsize=16, va="center", fontweight="bold")
    ax2.set_xscale("log"); ax2.set_xticks([250, 1000, 4000]); ax2.set_xticklabels(["250", "1k", "4k"])
    ax2.minorticks_off(); ax2.set_xlim(220, 4000 * 3.4); ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("Training sentences"); ax2.set_ylabel("Share of the target regime")
    ax2.set_title("Learning curves", fontsize=17, fontweight="bold", loc="left")
    fig.tight_layout()
    p = os.path.join(OUT, "controlled_regimes_talk.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", p)


def paper_infographic(fn_name, out_name):
    stub = types.ModuleType("parse_mh")
    stub.TRELLIS = stub.PrimitiveParseNode = stub.CompositeParseNode = object
    sys.modules["parse_mh"] = stub
    sys.path.insert(0, os.path.join(ROOT, "confs", "acs-26"))
    scale_text(SCALE)
    import make_paper_figures as mp
    getattr(mp, fn_name)(os.path.join(OUT, out_name))


def v2_gen():
    scale_text(SCALE)
    sys.path.insert(0, HERE)
    import make_talk_infographics as ti
    ti.make_generation_v2(os.path.join(OUT, "generation_v2_infographic_talk.png"))




def v1_gen3():
    """Three-panel, large-label version of the ACS generation infographic (slide 7)."""
    sys.path.insert(0, HERE)
    import make_talk_infographics as ti
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    W, H, PW = 13.2, 4.4, 4.1
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    FS = 20

    def panel(x, n, title):
        ax.add_patch(FancyBboxPatch((x, 0.05), PW, H - 0.1, boxstyle="round,pad=0.005,rounding_size=0.04",
                                    linewidth=1.2, edgecolor="#444", facecolor="white"))
        ax.add_patch(FancyBboxPatch((x + 0.06, H - 0.05 - 0.62), PW - 0.12, 0.62,
                                    boxstyle="round,pad=0.005,rounding_size=0.04", linewidth=0, facecolor="#1a1a1a"))
        ax.add_patch(mpatches.Circle((x + 0.42, H - 0.36), 0.22, facecolor="white", edgecolor="#1a1a1a"))
        ax.text(x + 0.42, H - 0.36, str(n), ha="center", va="center", fontsize=FS, fontweight="bold",
                family="serif", color="#1a1a1a")
        ax.text(x + 0.82, H - 0.36, title, ha="left", va="center", fontsize=FS, fontweight="bold",
                family="serif", color="white")
        return x + PW / 2

    def line(a, b, ra=0.3, rb=0.3, col="#444"):
        ax.add_line(Line2D([a[0], b[0]], [a[1] - ra, b[1] + rb], color=col, lw=1.8, zorder=2))

    def node(p, lab, r=0.34, fill=None, edge=None):
        ti._circle_chunk(ax, p[0], p[1], r, label=lab, fontsize=FS, fill=fill or ti.CHUNK_FILL,
                         edge=edge or ti.CHUNK_EDGE)

    def word(p, w):
        ti._primitive_box(ax, p[0], p[1], 0.9, 0.55, w, fontsize=FS)

    ly = 1.25
    # panel 1: the partial tree with the next seed
    cx = panel(0.05, 1, "Find the seed")
    s, np_, the, dog, vp = (cx, 3.0), (cx - 1.0, 2.1), (cx - 1.35, ly), (cx - 0.4, ly), (cx + 1.15, ly)
    line(s, np_, .34, .34); line(s, vp, .34, .4); line(np_, the, .34, .27); line(np_, dog, .34, .27)
    node(s, "S"); node(np_, "NP"); node(vp, "VP", r=0.4, fill=ti.ANCHOR_FILL, edge=ti.ANCHOR_EDGE)
    word(the, "the"); word(dog, "dog")
    ax.text(vp[0], vp[1] - 0.72, "next seed", ha="center", va="center", fontsize=FS, fontweight="bold",
            family="serif", color=ti.ANCHOR_EDGE)
    # panel 2: sample from both taxonomies
    cx = panel(0.05 + PW + 0.3, 2, "Recall its parts")
    seed = (cx, 3.1)
    node(seed, "VP", r=0.4, fill=ti.ANCHOR_FILL, edge=ti.ANCHOR_EDGE)
    for dx, lab, col, tc in ((-1.0, "content", "#2b6cb0", "#eaf2fb"), (1.0, "context", "#3c8062", "#e8f7ed")):
        ax.add_line(Line2D([seed[0], cx + dx], [seed[1] - 0.4, 2.3], color="#444", lw=1.8, zorder=2))
        ax.add_patch(FancyBboxPatch((cx + dx - 0.9, 1.75), 1.8, 0.55, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=1.8, edgecolor=col, facecolor=tc, zorder=3))
        ax.text(cx + dx, 2.025, lab, ha="center", va="center", fontsize=FS, fontweight="bold", family="serif",
                color=col, zorder=4)
    ax.text(cx - 1.0, 1.2, "pick a\nsplit", ha="center", va="center", fontsize=FS, family="serif", color="#2b6cb0")
    ax.text(cx + 1.0, 1.2, "fill each\nside", ha="center", va="center", fontsize=FS, family="serif", color="#3c8062")
    # panel 3: expand
    cx = panel(0.05 + 2 * (PW + 0.3), 3, "Expand the seed")
    s, np_, vp = (cx, 3.0), (cx - 1.05, 2.1), (cx + 1.05, 2.1)
    the, dog, eats, new = (cx - 1.5, ly), (cx - 0.52, ly), (cx + 0.46, ly), (cx + 1.42, ly)
    line(s, np_, .34, .34); line(s, vp, .34, .4); line(np_, the, .34, .27); line(np_, dog, .34, .27)
    line(vp, eats, .4, .27); line(vp, new, .4, .34)
    node(s, "S"); node(np_, "NP"); node(vp, "VP", r=0.4, fill=ti.ANCHOR_FILL, edge=ti.ANCHOR_EDGE)
    word(the, "the"); word(dog, "dog"); word(eats, "eats")
    node(new, "NP", fill=ti.CAND_FILL, edge=ti.CAND_EDGE)
    ax.text(new[0] - 0.3, new[1] - 0.72, "next seed", ha="center", va="center", fontsize=FS, fontweight="bold",
            family="serif", color=ti.CAND_EDGE)
    for x1 in (0.05 + PW + 0.02, 0.05 + 2 * PW + 0.32):
        ti._panel_arrow(ax, x1, H / 2, x1 + 0.26, color="#444")
    p = os.path.join(OUT, "generation_v1_infographic_talk.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.05); plt.close(fig); print("wrote", p)

if __name__ == "__main__":
    which = [n for n in ("parse_success", "r230", "v1_parse", "v1_gen", "v1_gen3", "v2_gen", "v2_parse", "v2_learn") if want(n)]
    for n in which:
        if n == "parse_success":
            parse_success()
        elif n == "v1_gen3":
            v1_gen3()
        elif n == "r230":
            r230()
        else:
            # one infographic per process: the text scaling is global and not reentrant
            if len(which) > 1 and n != which[-1] and False:
                pass
            if n == "v1_parse":
                paper_infographic("make_parse_infographic_figure", "parse_v1_infographic_talk.png")
            elif n == "v1_gen":
                paper_infographic("make_generation_infographic_figure", "generation_v1_infographic_talk.png")
            elif n == "v2_parse":
                scale_text(SCALE)
                sys.path.insert(0, HERE)
                import make_talk_infographics as ti
                ti.make_parse_v2(os.path.join(OUT, "parse_v2_infographic_talk.png"))
            elif n == "v2_learn":
                scale_text(SCALE)
                sys.path.insert(0, HERE)
                import make_talk_infographics as ti
                ti.make_learning_v2(os.path.join(OUT, "learning_v2_infographic_talk.png"))
            else:
                v2_gen()
