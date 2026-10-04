"""Figures for docs/FRAMEWORK.md.

Diagrams are drawn with Graphviz (``dot`` must be on the PATH); everything
that shows a learned structure or a measurement comes from models trained on
the paper's corpora (v1 split, seed 13) or from the result files of the
experiment scripts.

Usage:
    python docs/figures/make_figures.py              # all figures
    python docs/figures/make_figures.py overview cut # some figures
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from html import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "experiments", "v2", "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import Trellis2, load_corpus, v1_split  # noqa: E402
from trellis2.chart import Chart  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root, target_grammar  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.grammar import (UNK, TreeIndex, _compact, _elements, compile_grammar,  # noqa: E402
                              dm_code)
from trellis2.mdl_search import _State, word_classes  # noqa: E402
from trellis2.model import same_partition  # noqa: E402

# Palette (validated reference instance, light mode).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
BLUE_TINT, ORANGE_TINT, VIOLET_TINT, YELLOW_TINT = "#e6f0fc", "#fdeee6", "#ecebf7", "#fdf3dc"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
FONT = "Helvetica"
SEED = 13

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def out_path(name: str) -> str:
    return os.path.join(HERE, name)


def render_dot(source: str, name: str) -> str:
    path = out_path(name)
    subprocess.run(["dot", "-Tpng", "-Gdpi=180", "-o", path], input=source.encode(),
                   check=True)
    return path


def html(title: str, *lines: str, size: float = 9.5, color: str = INK2) -> str:
    """A Graphviz HTML label: a bold title and smaller lines below it."""
    body = "".join(f'<br/><font point-size="{size}" color="{color}">{line}</font>'
                   for line in lines)
    return f"<<b>{title}</b>{body}>"


def graph_header(rankdir: str = "TB", **extra) -> str:
    attrs = {"rankdir": rankdir, "bgcolor": SURFACE, "pad": "0.25", "nodesep": "0.35",
             "ranksep": "0.45", "fontname": FONT, "fontcolor": INK, **extra}
    a = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
    return (f"digraph G {{\n  graph [{a}];\n"
            f'  node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=11, '
            f'fontcolor="{INK}", color="{AXIS}", fillcolor="white", penwidth=1.3, '
            f'margin="0.16,0.07"];\n'
            f'  edge [color="{MUTED}", penwidth=1.2, arrowsize=0.7, fontname="{FONT}", '
            f'fontsize=9.5, fontcolor="{INK2}"];\n')


def titles(fig, title, subtitle, top=0.86):
    """Title and subtitle above the plotting area, left-aligned."""
    fig.suptitle(title, x=0.01, y=0.985, ha="left", va="top", fontsize=11.5, color=INK)
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=8.8, color=INK2)
    fig.subplots_adjust(top=top)


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def consolidate_keeping_trees(model: Trellis2):
    """Trellis2.consolidate, keeping the representation tree and the element
    leaves of the best round (the figures need them)."""
    mem, labels, rules, best = model.memory, None, None, None
    for _ in range(model.max_rounds):
        rtree, leaves = mem.build_hierarchy(labels, rules, seed=model.seed)
        g = compile_grammar(mem, rtree, leaves, alpha=model.alpha, seed=model.seed,
                            search=model.search, merge=model.merge)
        if best is None or g.info["bits (factored grammar)"] < best[2].info["bits (factored grammar)"]:
            best = (rtree, leaves, g)
        if labels is not None and same_partition(labels[0], g.elem_symbol):
            break
        labels = [g.elem_symbol, g.elem_fine][:mem.granularities]
        rules = [g.elem_rule, g.elem_rule_fine][:mem.granularities]
    model._grammar = best[2]
    return best


@lru_cache(maxsize=None)
def supervised(condition: str, n: int = 320):
    examples = load_corpus(os.path.join(default_data_root(), CONDITIONS[condition]))
    train, test = v1_split(examples, SEED)
    model = Trellis2(seed=SEED)
    for e in train[:n]:
        model.learn(e.tokens, e.tree)
    rtree, leaves, g = consolidate_keeping_trees(model)
    return model, rtree, leaves, g, train[:n], test


def symbol_names(g) -> dict:
    """Short display names for the symbols: S1, S2, ... by frequency."""
    order = np.argsort([-sum(y.values()) for y in g.symbol_yields], kind="stable")
    return {int(s): f"S{i + 1}" for i, s in enumerate(order)}


def gold_majority(model, g, examples, condition) -> dict:
    """For orientation only: the gold category most elements of each symbol
    carry (the learner never sees these names)."""
    cfg = CFG(target_grammar(condition))
    gold = [cfg.gold_labels(e.tokens, e.tree) for e in examples]
    mem = model.memory
    votes = defaultdict(Counter)
    for e, (sid, span) in enumerate(zip(mem.sentence_of, mem.span)):
        votes[int(g.elem_symbol[e])][gold[sid].get(span, "?")] += mem.weight[e]
    return {s: c.most_common(1)[0][0] for s, c in votes.items()}


def yield_text(mem, e) -> str:
    i, j = mem.span[e]
    toks = mem.sentences[mem.sentence_of[e]][i:j]
    return " ".join(toks) if len(toks) <= 3 else " ".join(toks[:2] + ["…"] + toks[-1:])


# ---------------------------------------------------------------------- #
# Figure 2: one element, two descriptions
# ---------------------------------------------------------------------- #
def fig_element():
    model, rtree, leaves, g, train, _ = supervised("small")
    names, gold = symbol_names(g), gold_majority(model, g, train, "small")
    mem = model.memory
    sid = 0
    elems = [e for e in range(len(mem)) if mem.sentence_of[e] == sid]
    by_span = {mem.span[e]: e for e in elems}
    tokens = mem.sentences[sid]
    n = len(tokens)
    focus = by_span[(n - 2, n)]                       # the object noun phrase
    labels = [g.elem_symbol, g.elem_fine]
    x = mem.instance(focus, labels)

    def sym(v):
        return names.get(int(v[1:]), v) if isinstance(v, str) and v.startswith("S") and v[1:].isdigit() else v

    d = graph_header("TB", nodesep="0.25", ranksep="0.4", ordering="out")
    d += '  subgraph cluster_tree { label=<<b>An analysed experience</b>>; labeljust="l"; style="rounded"; color="#e1e0d9"; fontsize=12;\n'
    for e in elems:
        i, j = mem.span[e]
        s = int(g.elem_symbol[e])
        word = f'<br/><font point-size="10" color="{INK2}">{escape(yield_text(mem, e))}</font>'
        hl = (f', color="{BLUE}", fillcolor="{BLUE_TINT}", penwidth=2.2' if e == focus else "")
        d += f'    e{e} [label=<<b>{names[s]}</b>{word}>{hl}];\n'
    for e in elems:
        if mem.left[e] >= 0:
            d += f"    e{e} -> e{mem.left[e]}; e{e} -> e{mem.right[e]};\n"
    d += "  }\n"
    rows = [("<b>surface</b>", ""), ("l1 (token before)", x["l1"]), ("r1 (token after)", x["r1"]),
            ("f (first token)", x["f"]), ("e (last token)", x["e"]),
            ("k (kind)", "composite" if x["k"] == "C" else "primitive"),
            ("<b>chunk context</b>", ""), ("cl, cr (children)", f'{sym(x["cl.0"])}, {sym(x["cr.0"])}'),
            ("a1 (parent)", sym(x["a1.0"])), ("sl1, sr1 (sibling)", f'{sym(x["sl1.0"])}, {sym(x["sr1.0"])}'),
            ("a2 (grandparent)", sym(x["a2.0"])), ("sl2, sr2 (its sibling)", f'{sym(x["sl2.0"])}, {sym(x["sr2.0"])}')]
    cells = "".join(
        f'<tr><td align="left" bgcolor="{BLUE_TINT if not v else "white"}" colspan="{2 if not v else 1}">'
        f'<font point-size="10" color="{INK if not v else INK2}">{k}</font></td>'
        + (f'<td align="left"><font point-size="10">{escape(str(v))}</font></td>' if v else "")
        + "</tr>" for k, v in rows)
    d += (f'  rep [shape=plain, label=<<table border="1" cellborder="1" cellspacing="0" cellpadding="4" '
          f'color="{BLUE}"><tr><td colspan="2" bgcolor="{BLUE}"><font color="white"><b>Representation instance</b>'
          f'</font></td></tr><tr><td colspan="2"><font point-size="9.5" color="{INK2}">how it behaves → the representation hierarchy'
          f'</font></td></tr>{cells}<tr><td colspan="2"><font point-size="9" color="{MUTED}">chunk context is also written at a finer'
          f' granularity</font></td></tr></table>>];\n')
    comp = [("kind", "composite"), ("left part", sym(x["cl.0"])), ("right part", sym(x["cr.0"]))]
    ccells = "".join(f'<tr><td align="left"><font point-size="10" color="{INK2}">{k}</font></td>'
                     f'<td align="left"><font point-size="10">{escape(str(v))}</font></td></tr>' for k, v in comp)
    d += (f'  comp [shape=plain, label=<<table border="1" cellborder="1" cellspacing="0" cellpadding="4" '
          f'color="{ORANGE}"><tr><td colspan="2" bgcolor="{ORANGE}"><font color="white"><b>Composition instance</b>'
          f'</font></td></tr><tr><td colspan="2"><font point-size="9.5" color="{INK2}">what it is made of → the composition hierarchy'
          f'</font></td></tr>{ccells}</table>>];\n')
    d += f'  e{focus} -> rep [style=dashed, color="{BLUE}", penwidth=1.5, minlen=2];\n'
    d += f'  e{focus} -> comp [style=dashed, color="{ORANGE}", penwidth=1.5, minlen=2];\n'
    legend = ", ".join(f"{names[s]} ≈ {gold[s]}" for s in sorted(names, key=lambda s: names[s]))
    d += (f'  label=<<font point-size="10" color="{MUTED}">Symbols learned from the SMALL corpus. '
          f'For orientation only, their majority gold categories: {escape(legend)}</font>>; labelloc="b";\n')
    d += "}\n"
    return render_dot(d, "element_two_descriptions.png")


# ---------------------------------------------------------------------- #
# Figure 3: the two hierarchies and their cuts
# ---------------------------------------------------------------------- #
def _drawn_tree(cut_nodes):
    """The cut nodes plus all their ancestors (a complete top of the tree)."""
    keep = {}
    for node in cut_nodes:
        a = node
        while a is not None:
            keep[a.id] = a
            a = a.parent
    return keep


def fig_hierarchies():
    model, rtree, leaves, g, train, _ = supervised("small")
    names, gold = symbol_names(g), gold_majority(model, g, train, "small")
    mem = model.memory
    no_orange = [c for c in CATEGORICAL if c != ORANGE]      # orange marks composition
    color_of = {s: no_orange[i % len(no_orange)] for i, s in
                enumerate(sorted(names, key=lambda s: names[s]))}
    cut_symbol = {node.id: s for s, nodes in enumerate(g.symbol_nodes) for node in nodes}
    # Yields under each cut node.
    yields = defaultdict(Counter)
    for e, leaf in enumerate(leaves):
        a = leaf
        while a.id not in cut_symbol:
            a = a.parent
        yields[a.id][yield_text(mem, e)] += mem.weight[e]
    keep = _drawn_tree([node for nodes in g.symbol_nodes for node in nodes])

    d = graph_header("TB", nodesep="0.18", ranksep="0.35")
    d += ('  subgraph cluster_r { label=<<b>Representation hierarchy</b>: concepts of behaviour, cut into symbols>; '
          f'labeljust="l"; style="rounded"; color="{BLUE}"; penwidth=1.4; fontsize=12;\n')
    for nid, node in keep.items():
        if nid in cut_symbol:
            s = cut_symbol[nid]
            top = ", ".join(t for t, _ in yields[nid].most_common(2))
            d += (f'    r{nid} [shape=box, style="rounded,filled", fillcolor="white", color="{color_of[s]}", '
                  f'penwidth=2.4, label=<<b>{names[s]}</b> <font color="{MUTED}">≈ {escape(gold[s])}</font>'
                  f'<br/><font point-size="9" color="{INK2}">n = {node.count:.0f}</font>'
                  f'<br/><font point-size="9" color="{INK2}">{escape(top)}</font>>];\n')
        else:
            d += (f'    r{nid} [shape=circle, style=filled, fillcolor="{AXIS}", color="{AXIS}", width=0.32, '
                  f'fixedsize=true, label=<<font point-size="7" color="{INK}">{node.count:.0f}</font>>];\n')
    for nid, node in keep.items():
        if nid not in cut_symbol:
            for ch in node.children:
                d += f"    r{nid} -> r{ch.id} [arrowhead=none];\n"
    d += "  }\n"

    # Composition hierarchy.
    rule_ids = {node.id: c for c, node in enumerate(g.rule_nodes)}
    ckeep = _drawn_tree(list(g.rule_nodes))

    def rename(label):
        parts = label.split()
        if len(parts) == 2 and all(p.startswith("S") and p[1:].isdigit() for p in parts):
            return " ".join(names[int(p[1:])] for p in parts)
        return label
    d += ('  subgraph cluster_c { label=<<b>Composition hierarchy</b>: concepts of make-up, cut into rule classes>; '
          f'labeljust="l"; style="rounded"; color="{ORANGE}"; penwidth=1.4; fontsize=12;\n')
    for nid, node in ckeep.items():
        if nid in rule_ids:
            c = rule_ids[nid]
            keys = g.rule_keys[c]
            composite = g.pk[c] < 0.5
            top = ", ".join(rename(k) for k, _ in keys.most_common(3))
            more = f" +{len(keys) - 3}" if len(keys) > 3 else ""
            d += (f'    c{nid} [shape=box, style="rounded,filled", '
                  f'fillcolor="{ORANGE_TINT if composite else "white"}", color="{ORANGE if composite else AXIS}", '
                  f'penwidth={2.4 if composite else 1.2}, label=<<font point-size="10">{escape(top)}{more}</font>'
                  f'<br/><font point-size="8.5" color="{INK2}">n = {node.count:.0f}</font>>];\n')
        else:
            d += (f'    c{nid} [shape=circle, style=filled, fillcolor="{AXIS}", color="{AXIS}", width=0.32, '
                  f'fixedsize=true, label=<<font point-size="7" color="{INK}">{node.count:.0f}</font>>];\n')
    for nid, node in ckeep.items():
        if nid not in rule_ids:
            for ch in node.children:
                d += f"    c{nid} -> c{ch.id} [arrowhead=none];\n"
    d += "  }\n"
    # Stack the composition hierarchy below the representation hierarchy.
    first_cut = next(nid for nid in keep if nid in cut_symbol)
    croot = next(nid for nid, node in ckeep.items() if node.parent is None)
    d += f"  r{first_cut} -> c{croot} [style=invis, minlen=2];\n"
    d += (f'  label=<<font point-size="10" color="{MUTED}">SMALL corpus, 320 analysed sentences (2,880 elements). '
          f'Grey dots: concepts above the cut (with their counts). Boxes: the cut. Orange boxes: composite rule classes '
          f'(chunk types).</font>>; labelloc="b";\n')
    d += "}\n"
    return render_dot(d, "two_hierarchies.png")


# ---------------------------------------------------------------------- #
# Figure 4: description length picks the level of generalization
# ---------------------------------------------------------------------- #
def _ml_code(groups, keys, weights, stride) -> float:
    """-log likelihood (nats) under the maximum-likelihood parameters."""
    gk = groups.astype(np.int64) * np.int64(stride) + keys.astype(np.int64)
    uniq, inv = np.unique(gk, return_inverse=True)
    cnt = np.bincount(inv, weights=weights)
    _, ginv = np.unique(uniq // stride, return_inverse=True)
    tot = np.bincount(ginv, weights=cnt)
    return float(-np.sum(cnt * np.log(cnt / tot[ginv])))


def fig_mdl_curve():
    model, rtree, leaves, g, _, _ = supervised("med")
    mem = model.memory
    vocab = mem.vocabulary() + [UNK]
    tok_index = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    index = TreeIndex(rtree)
    el = _elements(mem, leaves, index, tok_index)
    prim, comp, alpha = el.prim, ~el.prim, model.alpha
    LN2 = np.log(2)

    def codes(cut):
        _, sym = _compact(index.assign(cut)[el.leafpos])
        K = int(sym.max()) + 1
        key = np.empty_like(sym)
        key[prim] = el.tok[prim]
        key[comp] = V + sym[el.left[comp]] * K + sym[el.right[comp]]
        zeros = np.zeros(int(el.root.sum()), dtype=np.int64)
        total = (dm_code(zeros, sym[el.root], el.w[el.root], K, alpha)
                 + dm_code(sym, key, el.w, V + K * K, alpha))
        data = (_ml_code(zeros, sym[el.root], el.w[el.root], K)
                + _ml_code(sym, key, el.w, V + K * K))
        return K, total / LN2, data / LN2

    # Refine from the root: always split the largest concept on the cut.
    cut, path = [0], []
    while len(cut) < 400:
        path.append(codes(cut))
        splittable = [x for x in cut if index.children[x]]
        if not splittable:
            break
        x = max(splittable, key=lambda y: index.nodes[y].count)
        cut = [y for y in cut if y != x] + index.children[x]
    K, total, data = map(np.array, zip(*path))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(K, total / 1000, color=INK, linewidth=2.2, label="total")
    ax.plot(K, data / 1000, color=BLUE, linewidth=2, label="data bits (analyses, given the grammar)")
    ax.plot(K, (total - data) / 1000, color=ORANGE, linewidth=2, label="model bits (learning the grammar)")
    best = int(np.argmin(total))
    ax.plot([K[best]], [total[best] / 1000], marker="o", markersize=8, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.annotate(f"shortest on this path:\n{K[best]} symbols, {total[best]:,.0f} bits",
                (K[best], total[best] / 1000), xytext=(K[best] * 1.6, 15.5), textcoords="data",
                color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    chosen = g.info["bits (plain PCFG, after merges)"]
    ax.plot([g.K], [chosen / 1000], marker="D", markersize=8, color=BLUE, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=5)
    ax.annotate(f"chosen cut after search and merging:\n{g.K} symbols, {chosen:,.0f} bits",
                (g.K, chosen / 1000), xytext=(1.5, 2.2), textcoords="data", ha="left",
                color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for k_, t_, label in ((K[-1], total[-1], "total"), (K[-1], data[-1], "data"),
                          (K[-1], total[-1] - data[-1], "model")):
        ax.text(k_ * 1.06, t_ / 1000, label, color=INK2, fontsize=9, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.9, K[-1] * 1.5)
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of symbols (concepts on the cut; log scale)")
    ax.set_ylabel("Bits (thousands)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    titles(fig, "Description length picks the level of generalization",
           "MED, 320 analysed sentences. Cuts refine the representation hierarchy from its root, "
           "always splitting the largest concept.", top=0.86)
    path_out = out_path("mdl_cut_curve.png")
    fig.savefig(path_out, dpi=180)
    plt.close(fig)
    return path_out


# ---------------------------------------------------------------------- #
# Figure 5: parsing = inside-outside posteriors + minimum-risk tree
# ---------------------------------------------------------------------- #
def fig_chart(condition: str = "large", early: int = 20):
    """The test sentence the early grammar is least sure about, parsed by
    the early grammar and by the grammar learned from 320 sentences."""
    _, _, _, g_early, _, test = supervised(condition, early)

    def uncertainty(e):
        post = Chart(g_early, e.tokens).span_posteriors()
        L = len(e.tokens)
        return sum(post[i, j] * (1 - post[i, j]) for i in range(L) for j in range(i + 2, L + 1))
    example = max((e for e in test if 6 <= len(e.tokens) <= 10), key=uncertainty)
    tokens, n = example.tokens, len(example.tokens)
    gold = example.tree.brackets()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd"] + BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, size in zip(axes, (early, 320)):
        _, _, _, g, _, _ = supervised(condition, size)
        chart = Chart(g, tokens)
        post = chart.span_posteriors()
        mbr = chart.mbr_tree().brackets()
        for L in range(2, n + 1):
            for i in range(n - L + 1):
                j = i + L
                p = float(post[i, j])
                x0, y0 = i + (L - 1) / 2, L - 1
                ax.add_patch(plt.Rectangle((x0 + 0.04, y0 + 0.04), 0.92, 0.92, facecolor=cmap(p),
                                           edgecolor=INK if (i, j) in mbr else "none",
                                           linewidth=2 if (i, j) in mbr else 0))
                if p >= 0.05:
                    ax.text(x0 + 0.5, y0 + 0.5, f"{p:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if p > 0.55 else INK)
                if (i, j) in gold:
                    ax.plot([x0 + 0.85], [y0 + 0.17], marker="o", markersize=3.5,
                            color=ORANGE, zorder=5)
        for i, t in enumerate(tokens):
            ax.text(i + 0.5, -0.35, t, ha="center", va="top", fontsize=9, color=INK)
        ax.set_xlim(-0.2, n + 0.2)
        ax.set_ylim(-1.1, n)
        ax.set_axis_off()
        ax.set_title(f"Grammar learned from {size} analysed sentences", loc="left", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, linewidth=2,
                             label="span of the minimum-risk tree"),
               plt.Line2D([], [], marker="o", color=ORANGE, linestyle="none", markersize=5,
                          label="span of the gold tree")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("P(span is a chunk | sentence)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Parsing: inside-outside span posteriors and the minimum-risk tree "
                 f"({condition.upper()}; the test sentence the {early}-sentence grammar is least sure of)",
                 x=0.02, ha="left", fontsize=11.5, color=INK)
    fig.subplots_adjust(left=0.01, right=0.9, top=0.86, bottom=0.12, wspace=0.05)
    path_out = out_path("chart_posteriors.png")
    fig.savefig(path_out, dpi=180)
    plt.close(fig)
    return path_out


# ---------------------------------------------------------------------- #
# Figure 7: the structure search, from flat sentences to a grammar
# ---------------------------------------------------------------------- #
def _draw_analysis(ax, tops, names, new_label=None):
    """Draw a symbolic analysis (a list of top-level nodes) as a forest."""
    leaves = []

    def collect(node, depth):
        lab, body = node
        if isinstance(body, str):
            leaves.append(node)
            return {"node": node, "x": len(leaves) - 1, "h": 0, "kids": []}
        a, b = collect(body[0], depth + 1), collect(body[1], depth + 1)
        return {"node": node, "x": (a["x"] + b["x"]) / 2, "h": max(a["h"], b["h"]) + 1, "kids": [a, b]}
    roots = [collect(t, 0) for t in tops]

    def draw(r):
        lab, body = r["node"]
        y = r["h"]
        for k in r["kids"]:
            ax.plot([r["x"], k["x"]], [y - 0.18, k["h"] + 0.2], color=MUTED, linewidth=1.2, zorder=1)
            draw(k)
        text = names[lab] + (f"\n{body}" if isinstance(body, str) else "")
        fresh = lab == new_label
        ax.text(r["x"], y, text, ha="center", va="center", fontsize=9, color=INK, zorder=3,
                fontweight="bold" if not isinstance(body, str) else "normal",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=ORANGE_TINT if fresh else "white",
                          edgecolor=ORANGE if fresh else AXIS, linewidth=1.8 if fresh else 1))
    for r in roots:
        draw(r)
    ax.set_xlim(-0.7, len(leaves) - 0.3)
    ax.set_ylim(-0.7, 3.6)
    ax.set_axis_off()


def fig_search():
    examples = load_corpus(os.path.join(default_data_root(), CONDITIONS["small"]))
    train, _ = v1_split(examples, SEED)
    sentences = [e.tokens for e in train]
    n_tokens = len({w for s_ in sentences for w in s_}) + 1
    cls = word_classes(sentences, 0.001)[-1]
    state = _State([[(("w", cls[w]), w) for w in s_] for s_ in sentences], n_tokens, 0.001)
    steps = [(None, state)]
    while True:
        moves = list(state.scored_moves())
        if not moves:
            break
        nats, move = min(moves, key=lambda m: m[0])
        if nats >= state.nats - 1e-9:
            break
        state = state.apply(move)
        steps.append((move, state))
    example = 0
    names = {}
    for top in steps[0][1].analyses[example]:
        names.setdefault(top[0], f"W{len(names) + 1}")
    k = 0
    for move, st in steps[1:]:
        labels = {lab for lab in st.rows if lab not in names}
        for lab in sorted(labels, key=str):
            k += 1
            names[lab] = f"C{k}"
    fig, axes = plt.subplots(1, len(steps), figsize=(3.3 * len(steps), 3.4))
    for i, (ax, (move, st)) in enumerate(zip(axes, steps)):
        new = None
        if move is None:
            title = "Flat sentences in word classes"
        else:
            new = next(lab for lab in st.rows if names.get(lab) == f"C{i}")
            title = f"Step {i}: chunk ({names[move[1]]}, {names[move[2]]}) → {names[new]}"
        _draw_analysis(ax, st.analyses[example], names, new)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.text(0, -0.02, f"corpus code: {st.bits:,.0f} bits", transform=ax.transAxes,
                color=INK2, fontsize=9, va="top")
    fig.suptitle("Unsupervised structure search on SMALL: every move is global and must shorten the "
                 "code of all 320 sentences (one sentence shown)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    path_out = out_path("search_trajectory.png")
    fig.savefig(path_out, dpi=180)
    plt.close(fig)
    return path_out


# ---------------------------------------------------------------------- #
# Figure 8: a shorter code is a better grammar
# ---------------------------------------------------------------------- #
def fig_code_vs_commission():
    from scipy.stats import spearmanr
    results = json.load(open(os.path.join(RESULTS, "search", "results.json")))
    order = ["med", "large", "term_low", "term_med", "term_high"]
    cond_titles = {"med": "MED", "large": "LARGE", "term_low": "MED, 11-word lexicon",
                   "term_med": "MED, 22-word lexicon", "term_high": "MED, 39-word lexicon"}
    styles = {1: (BLUE, "o", "greedy"), 4: (ORANGE, "s", "beam 4"), 16: (AQUA, "^", "beam 16")}
    fig, axes = plt.subplots(1, len(order), figsize=(15, 3.7))
    for ax, cond in zip(axes, order):
        r = next(x for x in results if x["condition"] == cond)
        for beam, (color, marker, label) in styles.items():
            runs = [u for u in r["runs"] if u["beam"] == beam]
            ax.scatter([u["bits"] for u in runs], [100 * u["commission"] for u in runs], s=34,
                       color=color, marker=marker, edgecolor=SURFACE, linewidth=0.8, label=label,
                       zorder=3)
        ax.axvline(r["gold_bits"], color=INK2, linestyle=(0, (3, 3)), linewidth=1)
        ax.text(r["gold_bits"], 103, " gold trees", color=INK2, fontsize=8, va="bottom")
        rho = spearmanr([u["bits"] for u in r["runs"]], [u["commission"] for u in r["runs"]]).correlation
        ax.set_title(cond_titles[cond], loc="left", fontsize=10)
        ax.text(0.98, 0.04, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, ha="right",
                color=INK2, fontsize=8.5)
        ax.set_ylim(0, 110)
        ax.set_xlabel("Code length (bits)")
        style_axes(ax)
    axes[0].set_ylabel("Generation commission (%)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    titles(fig, "A shorter code is a better grammar",
           "Each point is one structure search (12 starting partitions × 3 search widths) on 320 sentences; "
           "commission of the maximum-likelihood grammar read off its analyses.", top=0.8)
    path_out = out_path("code_vs_commission.png")
    fig.savefig(path_out, dpi=180)
    plt.close(fig)
    return path_out


# ---------------------------------------------------------------------- #
# Figure 9: unsupervised learning against the supervised model
# ---------------------------------------------------------------------- #
def fig_unsupervised():
    rows = json.load(open(os.path.join(RESULTS, "unsupervised", "results.json")))
    try:
        old = json.loads(subprocess.run(
            ["git", "show", "f70b4a3c:experiments/v2/results/unsupervised/results.json"],
            cwd=ROOT, capture_output=True, check=True).stdout)
    except (subprocess.CalledProcessError, ValueError):
        old = []
    order = ["small", "med", "large", "term_low", "term_med", "term_high"]
    labels = {"small": "SMALL", "med": "MED", "large": "LARGE", "term_low": "MED, 11-word lexicon",
              "term_med": "MED, 22-word lexicon", "term_high": "MED, 39-word lexicon"}

    def mean(rs, side):
        return 100 * np.mean([r[side]["gen_commission"] for r in rs]) if rs else None
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for y, cond in enumerate(reversed(order)):
        unsup = mean([r for r in rows if r["condition"] == cond], "unsupervised")
        gold = mean([r for r in rows if r["condition"] == cond], "supervised")
        before = mean([r for r in old if r["condition"] == cond], "unsupervised")
        if before is not None and before - unsup > 1:
            ax.annotate("", xy=(unsup + 0.8, y), xytext=(before - 0.8, y),
                        arrowprops=dict(arrowstyle="->", color=AXIS, lw=1.4))
            ax.plot([before], [y], marker="o", markersize=8, markerfacecolor=SURFACE,
                    markeredgecolor=MUTED, markeredgewidth=1.5, linestyle="none")
            ax.text(before, y + 0.25, f"{before:.1f}%", color=MUTED, fontsize=8, ha="center")
        ax.plot([gold], [y], marker="D", markersize=8, color=ORANGE, markeredgecolor=SURFACE,
                markeredgewidth=1.2, linestyle="none", zorder=4)
        ax.plot([unsup], [y], marker="o", markersize=9, color=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=1.2, linestyle="none", zorder=5)
        ax.text(max(unsup, gold) + 1.5, y - 0.3, f"{unsup:.1f}% vs {gold:.1f}%", color=INK2, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[c] for c in reversed(order)], color=INK)
    ax.set_xlim(-1, 70)
    ax.set_xlabel("Generation commission (%): share of 1,000 sampled sentences the target grammar rejects")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Line2D([], [], marker="o", markersize=8, markerfacecolor=SURFACE, markeredgecolor=MUTED,
                          linestyle="none", label="unsupervised, v2.1 (greedy search)"),
               plt.Line2D([], [], marker="o", markersize=8, color=BLUE, linestyle="none",
                          label="unsupervised, v2.2 (sentences only)"),
               plt.Line2D([], [], marker="D", markersize=7, color=ORANGE, linestyle="none",
                          label="supervised on gold trees")]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0))
    ax.set_ylim(-0.6, len(order) - 0.4)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    titles(fig, "Learning from sentences alone matches learning from gold trees",
           "Generation commission, 320 training sentences, mean of two seeds. Labels: unsupervised vs supervised.",
           top=0.86)
    path_out = out_path("unsupervised_vs_gold.png")
    fig.savefig(path_out, dpi=180)
    plt.close(fig)
    return path_out


# ---------------------------------------------------------------------- #
# Figure 1: the framework at a glance
# ---------------------------------------------------------------------- #
def fig_overview():
    d = graph_header("TB", ranksep="0.42", nodesep="0.5")
    d += f'  exp [label={html("Experience", "a sentence; its analysis too, if one is given")}];\n'
    d += f'  el [label={html("Elements", "every primitive (token) and composite (chunk)", "of the analysis is recorded")}];\n'
    d += (f'  rep_i [color="{BLUE}", fillcolor="{BLUE_TINT}", label='
          f'{html("Representation instance", "how the element behaves:", "surface context + chunk context")}];\n')
    d += (f'  comp_i [color="{ORANGE}", fillcolor="{ORANGE_TINT}", label='
          f'{html("Composition instance", "what the element is made of:", "a token, or (symbol, symbol)")}];\n')
    d += (f'  rep_h [color="{BLUE}", fillcolor="{BLUE_TINT}", label='
          f'{html("Representation hierarchy", "Cobweb concepts of behaviour")}];\n')
    d += (f'  comp_h [color="{ORANGE}", fillcolor="{ORANGE_TINT}", label='
          f'{html("Composition hierarchy", "Cobweb concepts of make-up")}];\n')
    d += (f'  rep_c [color="{BLUE}", fillcolor="{BLUE_TINT}", label='
          f'{html("Cut chosen by description length", "+ symbol merging → SYMBOLS (categories)")}];\n')
    d += (f'  comp_c [color="{ORANGE}", fillcolor="{ORANGE_TINT}", label='
          f'{html("Cut chosen by description length", "→ RULE CLASSES (chunk types)")}];\n')
    d += ('  g [penwidth=1.8, color="#52514e", label=<<b>Factored probabilistic grammar</b>'
          f'<br/><font point-size="10" color="{INK2}">P(A → w) = Σ<sub>c</sub> U[A,c] p<sub>k</sub>[c] E[c,w]</font>'
          f'<br/><font point-size="10" color="{INK2}">P(A → B C) = Σ<sub>c</sub> U[A,c] (1 − p<sub>k</sub>[c]) L[c,B] R[c,C]</font>'
          f'<br/><font point-size="9" color="{MUTED}">a sentence is a sequence of top-level chunks</font>>];\n')
    d += f'  parse [label={html("Parse", "inside-outside posteriors", "→ minimum-risk tree")}];\n'
    d += f'  gen [label={html("Generate", "sample from the", "same grammar")}];\n'
    d += f'  dl [label={html("Description length", "bits for the grammar", "+ bits for the analyses")}];\n'
    d += "  exp -> el; el -> rep_i; el -> comp_i; rep_i -> rep_h; comp_i -> comp_h;\n"
    d += "  rep_h -> rep_c; comp_h -> comp_c; rep_c -> g; comp_c -> g; g -> parse; g -> gen; g -> dl;\n"
    d += (f'  rep_c -> comp_i [style=dashed, color="{BLUE}", constraint=false, '
          'label="  parts are named\\n  by their symbols", fontcolor="#256abf"];\n')
    d += (f'  rep_c -> rep_i [style=dashed, color="{BLUE}", constraint=false, '
          'label="chunk context in symbol terms:\\nre-describe, replay (next round)  ", fontcolor="#256abf"];\n')
    d += (f'  dl -> comp_c [style=dashed, color="{INK2}", constraint=false, '
          'label="  chooses both cuts"];\n')
    d += "  { rank=same; rep_i; comp_i; }\n  { rank=same; rep_h; comp_h; }\n  { rank=same; rep_c; comp_c; }\n"
    d += "  { rank=same; parse; gen; dl; }\n}\n"
    return render_dot(d, "framework_overview.png")


# ---------------------------------------------------------------------- #
# Figure 6: learning by day and by night
# ---------------------------------------------------------------------- #
def fig_day_night():
    d = graph_header("TB", ranksep="0.32", nodesep="0.9", newrank="true")
    d += (f'  subgraph cluster_day {{ label=<<b>Day</b>: observe(sentence)>; labeljust="l"; '
          f'style="rounded,filled"; color="{YELLOW}"; fillcolor="{YELLOW_TINT}"; penwidth=1.5; '
          f'fontsize=13; margin=14;\n')
    d += f'    perceive [label={html("Perceive", "Viterbi analysis under the current grammar:", "the shortest-code analysis, a forest of chunks", "where no larger chunk pays; unknown words get", "the category their context implies")}];\n'
    d += f'    store [label={html("Store", "the sentence with its analysis", "(episodic memory)")}];\n'
    d += "    perceive -> store;\n  }\n"
    d += (f'  subgraph cluster_night {{ label=<<b>Night</b>: sleep()>; labeljust="l"; '
          f'style="rounded,filled"; color="{VIOLET}"; fillcolor="{VIOLET_TINT}"; penwidth=1.5; '
          f'fontsize=13; margin=14;\n')
    d += f'    n1 [label={html("1. Word classes", "merge word types while a class-bigram", "code shrinks; keep the merge path")}];\n'
    d += f'    n2 [label={html("2. Structure", "beam search over chunk and merge moves,", "each scored exactly; start from 12 word-class", "partitions and from the stored analyses;", "keep the shortest code")}];\n'
    d += f'    n3 [label={html("3. Concepts", "replay into the two hierarchies, starting", "from the search\'s categories; MDL cuts", "→ the factored grammar")}];\n'
    d += f'    n4 [label={html("4. Re-analysis", "Viterbi analyses under the new grammar;", "kept only if the total code shrinks")}];\n'
    d += f'    n5 [label={html("5. Rewrite", "the stored analyses in the", "new grammar\'s categories")}];\n'
    d += "    n1 -> n2 -> n3 -> n4 -> n5;\n  }\n"
    d += "  { rank=same; perceive; n1; }\n"
    d += '  store -> n1 [label="time to sleep", constraint=false];\n'
    d += (f'  n5 -> perceive [label="the next day perceives\\nwith the new grammar", '
          f'style=dashed, color="{INK2}", constraint=false];\n')
    d += "}\n"
    return render_dot(d, "day_and_night.png")


FIGURES = {"overview": fig_overview, "element": fig_element, "hierarchies": fig_hierarchies,
           "mdl": fig_mdl_curve, "chart": fig_chart, "day_night": fig_day_night,
           "search": fig_search, "code_commission": fig_code_vs_commission,
           "unsupervised": fig_unsupervised}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        print(FIGURES[name]())
