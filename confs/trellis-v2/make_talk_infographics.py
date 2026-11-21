"""Infographics for the TRELLIS v2 talk, in the visual style of the ACS 2026 paper figures.

The helpers (_panel_box, _panel_arrow, _primitive_box, _circle_chunk, _mini_hierarchy,
_cobweb_node, palette constants) are taken verbatim from confs/acs-26/make_paper_figures.py
by parsing that file and executing only those definitions, so the colors, fonts, panel
headers and node shapes are identical.  The numbers drawn in the figures are schematic,
as in the ACS paper's own infographics.

Run on a compute node:
  srun -A overcap -p overcap --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps \
       -c 2 --mem=8G -t 30 /coc/flash10/ksingara3/miniconda3/bin/python confs/trellis-v2/make_talk_infographics.py

Outputs: confs/trellis-v2/talk_figures/{parse_v2,generation_v2,learning_v2}_infographic.png
"""
import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "acs-26", "make_paper_figures.py")
OUT = os.path.join(HERE, "talk_figures")

# ---- pull the ACS helpers out of the paper-figure script, unchanged ----
_NEED_FUNCS = {"_cobweb_node", "_primitive_box", "_circle_chunk", "_mini_hierarchy",
               "_panel_box", "_panel_arrow"}
_NEED_CONSTS = {"PRIM_FILL", "PRIM_EDGE", "CHUNK_FILL", "CHUNK_EDGE", "CAND_FILL", "CAND_EDGE",
                "HIER_FILL", "HIER_EDGE", "HIER_GRAY", "LINK_COLOR", "ANCHOR_FILL", "ANCHOR_EDGE"}
_tree = ast.parse(open(SRC, encoding="utf-8").read())
_ns = {"plt": plt, "mpatches": mpatches, "FancyBboxPatch": FancyBboxPatch, "Line2D": Line2D}
for node in _tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in _NEED_FUNCS:
        exec(compile(ast.Module([node], []), SRC, "exec"), _ns)
    elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in _NEED_CONSTS for t in node.targets):
        exec(compile(ast.Module([node], []), SRC, "exec"), _ns)
globals().update({k: v for k, v in _ns.items() if k in _NEED_FUNCS | _NEED_CONSTS})

GATE_COLOR, GATE_FILL = "#2f8a3e", "#e8f7ed"
RANK_COLOR, RANK_FILL = "#2b6cb0", "#eaf2fb"
POST_COLOR, POST_FILL = "#5d3f7d", "#f1eaf7"
PANEL_Y, PANEL_H = 0.10, 4.30


def canvas():
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.5)
    ax.set_axis_off()
    return fig, ax


def three_panels(ax, titles):
    xs = [(0.10, 4.55), (4.95, 4.95), (10.20, 4.70)]
    areas = []
    for i, ((x, w), t) in enumerate(zip(xs, titles), 1):
        areas.append(_panel_box(ax, x=x, y=PANEL_Y, w=w, h=PANEL_H, number=i, title=t))
    arrow_y = PANEL_Y + PANEL_H / 2
    _panel_arrow(ax, 4.62, arrow_y, 4.97, color="#444")
    _panel_arrow(ax, 9.87, arrow_y, 10.22, color="#444")
    return areas


def label_box(ax, cx, cy, w, h, text, color, fill, fs=10.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.4, edgecolor=color, facecolor=fill, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold", family="serif", zorder=4)


def word_row(ax, x0, cw, y, words, margin=0.50, w=0.52, fs=11.0):
    n = len(words)
    xs = [x0 + margin + (cw - 2 * margin) * i / (n - 1) for i in range(n)]
    for x, wd in zip(xs, words):
        _primitive_box(ax, x, y, w, 0.40, wd, fontsize=fs)
    return xs


def span_chart(ax, xs, leaf_y, labels=None, r=0.19, dy=0.62, fill=CAND_FILL, edge=CAND_EDGE,
               fs=8.5, first=0.62):
    """Triangular chart of every span of length >= 2 over the words at xs.
    labels: dict (i, j) -> text drawn in the circle."""
    n = len(xs)
    pos = {(i, i + 1): (xs[i], leaf_y + 0.20) for i in range(n)}
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            pos[(i, j)] = ((xs[i] + xs[j - 1]) / 2, leaf_y + 0.20 + first + dy * (length - 2) + 0.0)
    for (i, j), (x, y) in pos.items():
        if j - i < 2:
            continue
        for child in ((i, j - 1), (i + 1, j)):
            cx_, cy_ = pos[child]
            off = 0.0 if child[1] - child[0] == 1 else r
            ax.add_line(Line2D([x, cx_], [y - r, cy_ + off - (0.0 if child[1] - child[0] == 1 else 0.0)],
                               color=CAND_EDGE, linewidth=0.8, linestyle=":", zorder=1))
    for (i, j), (x, y) in pos.items():
        if j - i < 2:
            continue
        ax.add_patch(mpatches.Circle((x, y), r, facecolor=fill, edgecolor=edge, linewidth=1.6, zorder=3))
        if labels and (i, j) in labels:
            ax.text(x, y, labels[(i, j)], ha="center", va="center", fontsize=fs, color="#1a1a1a",
                    family="serif", fontweight="bold", zorder=3.2)
    return pos


# =====================================================================
# P1: parsing, version 2
# =====================================================================
def make_parse_v2(path):
    fig, ax = canvas()
    (p1, p2, p3) = three_panels(ax, ["Score every span", "Inside and outside, both taxonomies",
                                     "Report the best tree"])
    # --- panel 1: every span as a chart over "the dog eats a cat"
    cx0, cy0, cw, ch = p1
    leaf_y = cy0 + 0.35
    xs = word_row(ax, cx0, cw, leaf_y, ["the", "dog", "eats", "a", "cat"], margin=0.45)
    span_chart(ax, xs, leaf_y, r=0.17, dy=0.64, first=0.60)

    # --- panel 2: two hierarchies, inside / outside / posterior
    cx0, cy0, cw, ch = p2
    HS, HCY = 0.36, cy0 + 2.50
    cn = _mini_hierarchy(ax, cx=cx0 + cw * 0.27, cy=HCY, title="Content", title_color="#2b6cb0",
                         highlight_path=["root", "L", "L_L", "L_L_R"], scale=HS)
    xn = _mini_hierarchy(ax, cx=cx0 + cw * 0.73, cy=HCY, title="Context", title_color="#3c8062",
                         highlight_path=["root", "R", "R_L"], scale=HS)
    bw = cw - 0.35
    gx = cx0 + cw / 2
    label_box(ax, gx, cy0 + 1.62, bw, 0.40, "inside: each span from its sub-spans", GATE_COLOR, GATE_FILL)
    label_box(ax, gx, cy0 + 1.10, bw, 0.40, "outside: each span from its surroundings", RANK_COLOR, RANK_FILL)
    label_box(ax, gx, cy0 + 0.58, bw, 0.40, r"posterior $\mu$(the dog) = 0.82", POST_COLOR, POST_FILL)
    for src, sign in ((cn["L_L_R"], -1), (xn["R_L"], +1)):
        ax.annotate("", xy=(gx + sign * 0.5, cy0 + 1.62 + 0.20),
                    xytext=(src[0], src[1] - 0.18 * HS),
                    arrowprops=dict(arrowstyle="->", color="#6e6e6e", lw=1.0))

    # --- panel 3: the reported tree, with posteriors
    cx0, cy0, cw, ch = p3
    leaf_y = cy0 + 0.35
    xs = word_row(ax, cx0, cw, leaf_y, ["the", "dog", "eats", "a", "cat"], margin=0.45)
    y1, y2, y3 = leaf_y + 0.95, leaf_y + 1.75, leaf_y + 2.55
    np1 = (xs[0] + xs[1]) / 2
    np2 = (xs[3] + xs[4]) / 2
    vp = (xs[2] + np2) / 2
    s_ = (np1 + vp) / 2
    rr = 0.21

    def edge(a, b, color="#444", ls="-", lw=1.0):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, linestyle=ls, zorder=2))
    for x in (xs[0], xs[1]):
        edge((np1, y1 - rr), (x, leaf_y + 0.20))
    for x in (xs[3], xs[4]):
        edge((np2, y1 - rr), (x, leaf_y + 0.20))
    edge((vp, y2 - rr), (xs[2], leaf_y + 0.20))
    edge((vp, y2 - rr), (np2, y1 + rr))
    edge((s_, y3 - rr), (np1, y1 + rr))
    edge((s_, y3 - rr), (vp, y2 + rr))
    for (x, y, t) in ((np1, y1, ".82"), (np2, y1, ".88"), (vp, y2, ".74"), (s_, y3, "1.0")):
        _circle_chunk(ax, x, y, rr, label=t, fontsize=8.5)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path)


# =====================================================================
# P2: generation, version 2
# =====================================================================
def make_generation_v2(path):
    fig, ax = canvas()
    (p1, p2, p3) = three_panels(ax, ["Pick the next position", "Outside score + pair gate",
                                     "Write it, close at the boundary"])
    # --- panel 1: the dog eats [ ? ] [ ? ]; chunks that could surround the slot
    cx0, cy0, cw, ch = p1
    leaf_y = cy0 + 0.70
    n = 5
    xs = [cx0 + 0.45 + (cw - 0.9) * i / (n - 1) for i in range(n)]
    for x, wd in zip(xs[:3], ["the", "dog", "eats"]):
        _primitive_box(ax, x, leaf_y, 0.52, 0.40, wd, fontsize=11.0)
    # slots
    ax.add_patch(FancyBboxPatch((xs[3] - 0.26, leaf_y - 0.20), 0.52, 0.40,
                                boxstyle="round,pad=0.015,rounding_size=0.04", linewidth=1.8,
                                edgecolor=ANCHOR_EDGE, facecolor=ANCHOR_FILL, zorder=3))
    ax.text(xs[3], leaf_y, "?", ha="center", va="center", fontsize=12, family="serif",
            fontweight="bold", color=ANCHOR_EDGE, zorder=3.1)
    ax.add_patch(FancyBboxPatch((xs[4] - 0.26, leaf_y - 0.20), 0.52, 0.40,
                                boxstyle="round,pad=0.015,rounding_size=0.04", linewidth=1.4,
                                edgecolor="#999", facecolor="#f4f4f4", linestyle="--", zorder=3))
    ax.text(xs[4], leaf_y, "?", ha="center", va="center", fontsize=12, family="serif",
            color="#888", zorder=3.1)
    ax.text(xs[3], leaf_y - 0.42, "next element", ha="center", va="top", fontsize=9, family="serif",
            fontweight="bold", color=ANCHOR_EDGE)
    # chunks that could surround the position
    y1, y2 = leaf_y + 1.15, leaf_y + 2.10
    a_x = (xs[2] + xs[3]) / 2
    b_x = (xs[3] + xs[4]) / 2
    c_x = (xs[2] + b_x) / 2
    r = 0.20

    def dot(p, q):
        ax.add_line(Line2D([p[0], q[0]], [p[1], q[1]], color=CAND_EDGE, linewidth=0.9,
                           linestyle=":", zorder=1))
    for x in (xs[2], xs[3]):
        dot((a_x, y1 - r), (x, leaf_y + 0.20))
    for x in (xs[3], xs[4]):
        dot((b_x, y1 - r), (x, leaf_y + 0.20))
    dot((c_x, y2 - r), (xs[2], leaf_y + 0.20))
    dot((c_x, y2 - r), (b_x, y1 + r))
    for (x, y) in ((a_x, y1), (b_x, y1), (c_x, y2)):
        _circle_chunk(ax, x, y, r, fill=CAND_FILL, edge=CAND_EDGE)

    # --- panel 2: context hierarchy, sampled leaf, outside score, pair gate
    cx0, cy0, cw, ch = p2
    HS, HCY = 0.55, cy0 + 2.05
    _mini_hierarchy(ax, cx=cx0 + cw / 2, cy=HCY, title="Context", title_color="#3c8062",
                    highlight_path=["root", "L", "L_R"], highlight_basic="L",
                    highlight_leaf="L_R", scale=HS,
                    leaf_labels={"L_R": ('"a"', 0.0, -0.20)})
    bw = cw - 0.35
    gx = cx0 + cw / 2
    label_box(ax, gx, cy0 + 1.05, bw, 0.42, "outside score: sum over chunks", RANK_COLOR, RANK_FILL, fs=10)
    label_box(ax, gx, cy0 + 0.45, bw, 0.42, r"pair gate: (eats, a) seen side by side   $\checkmark$", GATE_COLOR, GATE_FILL, fs=10)

    # --- panel 3: the written sentence and the exact-boundary rule
    cx0, cy0, cw, ch = p3
    leaf_y = cy0 + 2.55
    xs = word_row(ax, cx0, cw, leaf_y, ["the", "dog", "eats", "a", "cat"], margin=0.45)
    # new element drawn purple
    _primitive_box(ax, xs[3], leaf_y, 0.52, 0.40, "a", edge=CAND_EDGE, fill=CAND_FILL, fontsize=11.0)
    ax.text(xs[3], leaf_y - 0.42, "new", ha="center", va="top", fontsize=9, family="serif",
            fontweight="bold", color=CAND_EDGE)
    bw = cw - 0.35
    gx = cx0 + cw / 2
    label_box(ax, gx, cy0 + 1.25, bw, 0.42, '"the" began a stored sentence   $\\checkmark$', GATE_COLOR, GATE_FILL, fs=10)
    label_box(ax, gx, cy0 + 0.65, bw, 0.42, '"cat" ended a stored sentence   $\\checkmark$', GATE_COLOR, GATE_FILL, fs=10)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path)


# =====================================================================
# L1: learning, version 2
# =====================================================================
def make_learning_v2(path):
    fig, ax = canvas()
    (p1, p2, p3) = three_panels(ax, ["Parse with memory so far", "Weight each span by its posterior",
                                     "Add to both taxonomies"])
    words = ["the", "dog", "eats", "fish"]
    post = {(0, 2): .84, (1, 3): .10, (2, 4): .74, (0, 3): .13, (1, 4): .22, (0, 4): 1.0}
    # --- panel 1: span chart with posteriors
    cx0, cy0, cw, ch = p1
    leaf_y = cy0 + 0.35
    xs = word_row(ax, cx0, cw, leaf_y, words, margin=0.55)
    labs = {k: ("1.0" if v == 1.0 else ("%.2f" % v).lstrip("0")) for k, v in post.items()}
    span_chart(ax, xs, leaf_y, labels=labs, r=0.22, dy=0.85, first=0.85, fs=8.5)

    # --- panel 2: one row per span, bar = weight
    cx0, cy0, cw, ch = p2
    rows = [((0, 2), "the dog"), ((2, 4), "eats fish"), ((1, 4), "dog eats fish"),
            ((0, 3), "the dog eats"), ((1, 3), "dog eats"), ((0, 4), "the dog eats fish")]
    y = cy0 + ch - 0.45
    for (k, text) in rows:
        v = post[k]
        _circle_chunk(ax, cx0 + 0.35, y, 0.15, fill=CAND_FILL, edge=CAND_EDGE)
        ax.text(cx0 + 0.65, y, text, ha="left", va="center", fontsize=10.5, family="serif", color="#1a1a1a")
        bx = cx0 + 2.55
        bar_w = 1.55 * v
        ax.add_patch(FancyBboxPatch((bx, y - 0.09), bar_w, 0.18, boxstyle="round,pad=0.0,rounding_size=0.04",
                                    linewidth=0, facecolor=CAND_FILL, zorder=3))
        ax.text(bx + 1.65, y, ("%.2f" % v) if v < 1 else "1.00", ha="left", va="center", fontsize=10,
                family="monospace", color="#444")
        y -= 0.52

    # --- panel 3: two cards weighted by the posterior (hierarchies.png style)
    cx0, cy0, cw, ch = p3
    top = cy0 + ch - 0.12
    cardw, cardh = 2.1, 2.95
    cyc = top - cardh / 2
    _cobweb_node(ax, cx0 + cardw / 2 + 0.06, cyc, cardw, cardh, title="CNT_CONCEPT", count="+.84",
                 attrs=[("content-left", [("the", ".84")]), ("content-right", [("dog", ".84")])],
                 title_fill="#1f4e79", font_scale=0.8)
    _cobweb_node(ax, cx0 + cw - cardw / 2 - 0.06, cyc, cardw, cardh, title="CTX_CONCEPT", count="+.84",
                 attrs=[("left-1", [("EMPTYNULL", ".84")]), ("right+1", [("eats", ".84")]),
                        ("right+2", [("fish", ".84")])],
                 title_fill="#2e6e55", font_scale=0.8)
    ax.text(cx0 + cw / 2, cy0 + 0.30, "counts grow by the posterior, not by 1", ha="center", va="center",
            fontsize=10, family="serif", style="italic", color="#444")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_parse_v2(os.path.join(OUT, "parse_v2_infographic.png"))
    make_generation_v2(os.path.join(OUT, "generation_v2_infographic.png"))
    make_learning_v2(os.path.join(OUT, "learning_v2_infographic.png"))
