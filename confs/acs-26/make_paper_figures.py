"""Generate the two illustrative figures referenced from paper.tex by
loading a trained TRELLIS model and reading real attribute--value
distributions out of its content and context hierarchies.

The trained model lives at::

    confs/acs-26/grammar_med/hollow_learn/seed_42/final_ltm_data/

The outputs are written to ``confs/acs-26/paper/graphics/`` so that
the Overleaf-side ``graphics/X.png`` references in main.tex resolve
directly. Run::

    python confs/acs-26/make_paper_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from parse_mh import TRELLIS, PrimitiveParseNode, CompositeParseNode  # noqa: E402

OUT_DIR = os.path.join(HERE, "paper", "graphics")
TRAINED_MODEL = os.path.join(
    HERE, "grammar_med", "hollow_learn", "seed_42", "final_ltm_data"
)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers: short labels for concept and value IDs.
# ---------------------------------------------------------------------------

def _short_concept(name) -> str:
    """Turn a 'CONCEPT-3606598960683166_143329' string into
    'CONCEPT-143329' (last numeric suffix only)."""
    if isinstance(name, str) and name.startswith("CONCEPT-"):
        return "CONCEPT-" + name.split("_")[-1]
    return str(name)


def _value_label(itv, val_id) -> str:
    """Return a short display label for a Cobweb value-id."""
    if val_id is None:
        return "?"
    try:
        v = itv[val_id]
    except Exception:
        return f"id{val_id}"
    return _short_concept(v)


def _node_label(node) -> str:
    """Concept name shown in the title bar of a Cobweb-style node."""
    h = node.concept_hash()
    suffix = str(h)[-6:]
    return f"CONCEPT-{suffix}"


# ---------------------------------------------------------------------------
# Cobweb-style node drawing primitive (shared between both figures).
# ---------------------------------------------------------------------------

def _cobweb_node(ax, cx, cy, w, h, *, title, count, attrs,
                 title_fill="#1f4e79", title_text="white",
                 body_fill="#fbfdff", edge_color=None,
                 highlight=False):
    """Draw a Cobweb-style concept node.

    Parameters
    ----------
    title : header string (concept name).
    count : instance count shown as ``n=...`` (None to omit).
    attrs : list of (attribute_label, [(value_label, prob), ...]).
    """
    title_h = h * 0.20
    body_h = h - title_h
    # Title and body share a single y_top so they are flush.
    y_top = cy + h / 2
    title_bottom = y_top - title_h
    body_bottom = cy - h / 2

    if edge_color is None:
        edge_color = "#b45309" if highlight else "#666666"
    edge_lw = 1.7 if highlight else 1.0

    # Body (drawn first so its edge sits underneath the title bar).
    ax.add_patch(FancyBboxPatch((cx - w / 2, body_bottom),
                                w, body_h,
                                boxstyle="round,pad=0.005,rounding_size=0.014",
                                linewidth=edge_lw, edgecolor=edge_color,
                                facecolor=body_fill))

    # Title bar overdrawn on top of the body's upper edge.
    ax.add_patch(FancyBboxPatch((cx - w / 2, title_bottom),
                                w, title_h,
                                boxstyle="round,pad=0.005,rounding_size=0.014",
                                linewidth=edge_lw, edgecolor=edge_color,
                                facecolor=title_fill))
    ax.text(cx - w * 0.45, title_bottom + title_h / 2, title,
            ha="left", va="center", fontsize=8.6, color=title_text,
            fontweight="bold", family="monospace")
    if count is not None:
        if isinstance(count, float):
            count_str = f"{count:g}"
        else:
            count_str = str(count)
        ax.text(cx + w * 0.45, title_bottom + title_h / 2,
                f"n={count_str}",
                ha="right", va="center", fontsize=7.8, color=title_text,
                family="monospace")

    total_lines = sum(1 + len(values) for _, values in attrs)
    if total_lines == 0:
        total_lines = 1
    # Inner top padding so the first row sits well clear of the title bar
    # bottom edge, plus a small bottom padding for symmetry.
    top_pad = max(body_h * 0.12, 0.20)
    bot_pad = max(body_h * 0.04, 0.06)
    usable_h = body_h - top_pad - bot_pad
    line_h = usable_h / total_lines
    cur_y = title_bottom - top_pad

    for attr_name, values in attrs:
        cur_y -= line_h
        ax.text(cx - w * 0.46, cur_y + line_h * 0.5, f"{attr_name}:",
                ha="left", va="center", fontsize=7.6, family="monospace",
                color="#222", fontweight="bold")
        for val, prob in values:
            cur_y -= line_h
            ax.text(cx - w * 0.40, cur_y + line_h * 0.5, val,
                    ha="left", va="center", fontsize=7.2,
                    family="monospace", color="#444")
            if isinstance(prob, str):
                prob_str = prob
            else:
                prob_str = f"{prob:.2f}"
            ax.text(cx + w * 0.46, cur_y + line_h * 0.5, prob_str,
                    ha="right", va="center", fontsize=7.2,
                    family="monospace", color="#666")


def _link(ax, p1, p2, color="#888888", lw=0.9):
    ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                       color=color, linewidth=lw))


def _callout(ax, x, y, w, h, text, *, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                linewidth=1.2, edgecolor=color,
                                facecolor="#fffcf5", linestyle="--"))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=9.5, color=color, family="serif", style="italic")


# ---------------------------------------------------------------------------
# Extracting attribute-value distributions from a node.
# ---------------------------------------------------------------------------

def _top_av_distribution(node, itv, attr_idx, top_k=3,
                         skip_emptynull=True):
    """Return the top-k (value_label, prob) for a given attribute at a
    Cobweb node, sorted by probability descending."""
    av = node.av_count
    if attr_idx not in av:
        return []
    bucket = av[attr_idx]
    total = sum(bucket.values())
    if total <= 0:
        return []
    items = sorted(bucket.items(), key=lambda kv: -kv[1])
    out = []
    for val_id, c in items:
        label = _value_label(itv, val_id)
        if skip_emptynull and label.upper() == "EMPTYNULL":
            continue
        out.append((label, c / total))
        if len(out) >= top_k:
            break
    return out


def _content_node_attrs(node, itv, top_k=3):
    """Attribute spec for a content-hierarchy node."""
    out = []
    left = _top_av_distribution(node, itv, 0, top_k=top_k)
    right = _top_av_distribution(node, itv, 1, top_k=top_k)
    if left:
        out.append(("content-left", left))
    if right:
        out.append(("content-right", right))
    return out


def _context_node_attrs(node, itv, top_k=3):
    """Attribute spec for a context-hierarchy node, showing the
    immediate-left and immediate-right neighbors."""
    out = []
    a_keys = sorted([k for k in node.av_count.keys() if k >= 0])
    if not a_keys:
        return out
    half = len(a_keys) // 2
    left_attr = a_keys[half - 1]
    right_attr = a_keys[half]
    left = _top_av_distribution(node, itv, left_attr, top_k=top_k)
    right = _top_av_distribution(node, itv, right_attr, top_k=top_k)
    if left:
        out.append(("left-1", left))
    if right:
        out.append(("right+1", right))
    return out


def _select_subtree(root, *, n_top_children=2, n_top_grandkids=2):
    """Return a small subtree rooted at ``root``:
        root, [(child, [grandchildren])] x n_top_children
    sorted by descending count.
    """
    children = sorted(root.children, key=lambda c: -c.count)[:n_top_children]
    out = []
    for c in children:
        gks = sorted(c.children, key=lambda x: -x.count)[:n_top_grandkids]
        out.append((c, gks))
    return out


# ---------------------------------------------------------------------------
# Picking a chunk from a real parse.
# ---------------------------------------------------------------------------

def _pick_np_chunk(fp):
    """Find the smallest NP-like composite (cplx=2 with two primitive
    children) in the parse tree."""
    candidates = []
    for node in fp.nodes:
        if not isinstance(node, CompositeParseNode):
            continue
        if node.complexity != 2:
            continue
        kids = [c for _, c in node.children]
        if all(isinstance(k, PrimitiveParseNode) for k in kids):
            candidates.append(node)
    if not candidates:
        raise RuntimeError("no cplx=2 NP-like chunk found in parse")
    return candidates[0]


# ---------------------------------------------------------------------------
# Figure 1: instances.png
# ---------------------------------------------------------------------------

def make_instances_figure(path: str, trellis) -> None:
    itv = trellis.ltm.id_to_value
    sentence = "the cat saw a dog"
    fp = trellis.parse_sentence(
        sentence, threshold="converge",
        new_vocab=False, learning=False,
    )
    chunk = _pick_np_chunk(fp)
    L = chunk.children[0][1]
    R = chunk.children[1][1]
    left_word = itv[L.word_id]
    right_word = itv[R.word_id]

    # Content/context instances from the chunk (real data).
    ci = chunk.content_instance
    xi = chunk.context_instance

    # --- compose Cobweb-style attribute specs for the instance boxes ---
    def _top_from_inst(dist, top_k=4):
        if not dist:
            return []
        items = sorted(dist.items(), key=lambda kv: -kv[1])
        out = []
        for val_id, weight in items:
            lbl = _value_label(itv, val_id)
            if lbl.upper() == "EMPTYNULL":
                continue
            out.append((lbl, float(weight)))
            if len(out) >= top_k:
                break
        return out

    content_attrs = []
    cl = _top_from_inst(ci.get(0, {}), top_k=3)
    cr = _top_from_inst(ci.get(1, {}), top_k=3)
    if cl:
        content_attrs.append(("content-left", cl))
    if cr:
        content_attrs.append(("content-right", cr))
    # complexity attributes — display as raw labels (e.g. C1)
    for attr_idx, label in [(2, "cplx-left"), (3, "cplx-right")]:
        d = ci.get(attr_idx, {})
        if d:
            val_id = next(iter(d))
            content_attrs.append((label, [(_value_label(itv, val_id), 1.0)]))

    # context instance: show all positive-index slots that have any
    # non-EMPTYNULL value, in slot order.
    context_attrs = []
    pos_keys = sorted([k for k in xi.keys() if isinstance(k, int) and k >= 0])
    half = len(pos_keys) // 2 if pos_keys else 0
    for k in pos_keys:
        d = xi.get(k, {})
        rows = _top_from_inst(d, top_k=2)
        if k < half:
            slot_name = f"left-{half - k}"
        else:
            slot_name = f"right+{k - half + 1}"
        if rows:
            context_attrs.append((slot_name, rows))
        else:
            context_attrs.append((slot_name, [("EMPTYNULL", 0.0)]))

    # --- figure layout ---
    fig, ax = plt.subplots(figsize=(14.5, 6.0))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.0)
    ax.set_axis_off()

    base_y = 5.2
    words = sentence.split()
    word_x = [0.9 + 1.45 * i for i in range(len(words))]
    # identify the chunk span
    try:
        li = words.index(left_word)
        ri = li + 1 if words[li + 1] == right_word else None
    except ValueError:
        li, ri = None, None
    for i, (w, x) in enumerate(zip(words, word_x)):
        is_chunk = (li is not None and i in (li, ri))
        ax.text(x, base_y, w, ha="center", va="center",
                fontsize=16,
                color="#c0392b" if is_chunk else "#222",
                fontweight="bold" if is_chunk else "normal",
                family="serif")

    if li is not None and ri is not None:
        lo = word_x[li] - 0.55
        hi = word_x[ri] + 0.55
        bracket_y = base_y - 0.42
        ax.add_line(Line2D([lo, lo], [bracket_y, base_y - 0.18],
                           color="#c0392b", linewidth=1.6))
        ax.add_line(Line2D([hi, hi], [bracket_y, base_y - 0.18],
                           color="#c0392b", linewidth=1.6))
        ax.add_line(Line2D([lo, hi], [bracket_y, bracket_y],
                           color="#c0392b", linewidth=1.6))
        ax.text((lo + hi) / 2, bracket_y - 0.30,
                "NP chunk",
                ha="center", va="center", fontsize=11, color="#c0392b",
                fontweight="bold", family="serif")
        arrow_y_top = bracket_y - 0.50
    else:
        arrow_y_top = base_y - 0.5

    # arrows to the two instance boxes
    ax.annotate("",
                xy=(4.3, 3.5), xytext=(1.6, arrow_y_top),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    ax.annotate("",
                xy=(9.7, 3.5), xytext=(1.6, arrow_y_top),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))

    # Draw the two instance boxes using the Cobweb-style layout.
    _cobweb_node(ax, cx=4.3 + 2.2, cy=2.0, w=4.4, h=3.0,
                 title="content instance",
                 count=None, attrs=content_attrs,
                 title_fill="#1f4e79",
                 body_fill="#ffffff",
                 edge_color="#1f4e79")

    _cobweb_node(ax, cx=9.7 + 2.2, cy=2.0, w=4.4, h=3.0,
                 title="context instance",
                 count=None, attrs=context_attrs,
                 title_fill="#1f7d4e",
                 body_fill="#ffffff",
                 edge_color="#1f7d4e")

    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 2: hierarchies.png  (real subtrees with CONCEPT- labels)
# ---------------------------------------------------------------------------

def make_hierarchies_figure(path: str, trellis) -> None:
    itv = trellis.ltm.id_to_value

    fig, axes = plt.subplots(1, 2, figsize=(20, 11.0))

    NODE_W, NODE_H = 5.6, 3.4
    LEAF_W, LEAF_H = 4.4, 3.1

    # ====================================================================
    # LEFT: content subtree
    # ====================================================================
    ax = axes[0]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.set_axis_off()
    ax.set_title("Content hierarchy (subtree)",
                 fontsize=15, fontweight="bold", color="#1a365d", pad=10)

    croot = trellis.ltm.content_hierarchy.root
    csub = _select_subtree(croot)

    # Root box.
    _cobweb_node(ax, cx=10.0, cy=14.0, w=NODE_W, h=NODE_H,
                 title=_node_label(croot), count=croot.count,
                 attrs=_content_node_attrs(croot, itv, top_k=3),
                 title_fill="#1f4e79")

    # Two children of root.
    child_cx = [4.5, 15.5]
    for (cnode, gks), cx in zip(csub, child_cx):
        _cobweb_node(ax, cx=cx, cy=9.5, w=NODE_W, h=NODE_H,
                     title=_node_label(cnode), count=cnode.count,
                     attrs=_content_node_attrs(cnode, itv, top_k=3),
                     title_fill="#2b6cb0")
        _link(ax, (10.0, 14.0 - NODE_H / 2),
              (cx, 9.5 + NODE_H / 2))

    # Two grandkids under the first child (highlighted as a shared subtree).
    if csub:
        cnode1, gks1 = csub[0]
        gk_cx = [2.6, 7.4]
        for gk, gcx in zip(gks1, gk_cx):
            _cobweb_node(ax, gcx, 4.5, LEAF_W, LEAF_H,
                         title=_node_label(gk), count=gk.count,
                         attrs=_content_node_attrs(gk, itv, top_k=2),
                         title_fill="#b45309", highlight=True)
            _link(ax, (4.5, 9.5 - NODE_H / 2),
                  (gcx, 4.5 + LEAF_H / 2))

    # One grandkid under the second child as a counterpart.
    if csub and len(csub) > 1:
        cnode2, gks2 = csub[1]
        if gks2:
            gk = gks2[0]
            _cobweb_node(ax, 15.5, 4.5, LEAF_W, LEAF_H,
                         title=_node_label(gk), count=gk.count,
                         attrs=_content_node_attrs(gk, itv, top_k=2),
                         title_fill="#2b6cb0")
            _link(ax, (15.5, 9.5 - NODE_H / 2),
                  (15.5, 4.5 + LEAF_H / 2))

    _callout(ax, x=0.5, y=0.5, w=19.0, h=1.7,
             text="Highlighted leaves of the content tree share a parent and overlap heavily on the\n"
                  "content-left and content-right distributions: they instantiate the same compositional pattern.",
             color="#b45309")

    # ====================================================================
    # RIGHT: context subtree
    # ====================================================================
    ax = axes[1]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.set_axis_off()
    ax.set_title("Context hierarchy (subtree)",
                 fontsize=15, fontweight="bold", color="#1a4d3a", pad=10)

    xroot = trellis.ltm.context_hierarchy.root
    xsub = _select_subtree(xroot)

    _cobweb_node(ax, cx=10.0, cy=14.0, w=NODE_W, h=NODE_H,
                 title=_node_label(xroot), count=xroot.count,
                 attrs=_context_node_attrs(xroot, itv, top_k=3),
                 title_fill="#2f7050")

    for (cnode, gks), cx in zip(xsub, child_cx):
        _cobweb_node(ax, cx=cx, cy=9.5, w=NODE_W, h=NODE_H,
                     title=_node_label(cnode), count=cnode.count,
                     attrs=_context_node_attrs(cnode, itv, top_k=3),
                     title_fill="#3c8062")
        _link(ax, (10.0, 14.0 - NODE_H / 2),
              (cx, 9.5 + NODE_H / 2))

    if xsub:
        cnode1, gks1 = xsub[0]
        gk_cx = [2.6, 7.4]
        for gk, gcx in zip(gks1, gk_cx):
            _cobweb_node(ax, gcx, 4.5, LEAF_W, LEAF_H,
                         title=_node_label(gk), count=gk.count,
                         attrs=_context_node_attrs(gk, itv, top_k=2),
                         title_fill="#b45309", highlight=True)
            _link(ax, (4.5, 9.5 - NODE_H / 2),
                  (gcx, 4.5 + LEAF_H / 2))

    if xsub and len(xsub) > 1:
        cnode2, gks2 = xsub[1]
        if gks2:
            gk = gks2[0]
            _cobweb_node(ax, 15.5, 4.5, LEAF_W, LEAF_H,
                         title=_node_label(gk), count=gk.count,
                         attrs=_context_node_attrs(gk, itv, top_k=2),
                         title_fill="#3c8062")
            _link(ax, (15.5, 9.5 - NODE_H / 2),
                  (15.5, 4.5 + LEAF_H / 2))

    _callout(ax, x=0.5, y=0.5, w=19.0, h=1.7,
             text="Highlighted leaves of the context tree share a parent and overlap heavily on the\n"
                  "left-1 and right+1 distributions: they fill interchangeable distributional roles.",
             color="#b45309")

    fig.suptitle("Side-by-side subtrees of TRELLIS's two hierarchies",
                 fontsize=16, fontweight="bold", color="#222")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 3: generation grammaticality and novelty across all 3 grammars.
# ---------------------------------------------------------------------------

import csv

GRAMMAR_COLORS = {
    "small": "#2b6cb0",
    "med":   "#b45309",
    "large": "#9f1239",
}
GRAMMAR_LABELS = {
    "small": "SMALL  (S→NP VP, VP→V (NP))",
    "med":   "MED  (+AdjP, PP, V NP PP)",
    "large": "LARGE  (+RelClause)",
}


def _load_aggregated(grammar_size: str):
    """Load aggregated.csv for one grammar size. Returns a dict of
    column -> list[float]."""
    path = os.path.join(
        HERE, f"grammar_{grammar_size}",
        "learning_curves", "aggregated.csv",
    )
    cols = None
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for r in reader:
            rows.append(r)
    out = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            v = r[c]
            try:
                out[c].append(float(v))
            except (ValueError, TypeError):
                out[c].append(float("nan"))
    return out


def make_generation_curves_figure(path: str) -> None:
    data = {sz: _load_aggregated(sz) for sz in ("small", "med", "large")}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    panels = [
        ("gen_gram",  "Generation grammaticality", "(a) Grammaticality"),
        ("gen_novel", "Generation novelty",        "(b) Novelty"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for sz in ("small", "med", "large"):
            d = data[sz]
            x = d["n_trained"]
            y = d[f"{key}_mean"]
            std = d[f"{key}_std"]
            lo = [m - s for m, s in zip(y, std)]
            hi = [m + s for m, s in zip(y, std)]
            color = GRAMMAR_COLORS[sz]
            ax.plot(x, y, color=color, linewidth=2.0,
                    label=GRAMMAR_LABELS[sz], marker="o", markersize=4)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.set_xlabel("# training sentences", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Generation grammaticality and novelty across the three grammars",
                 fontsize=13, fontweight="bold", color="#222")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 4: parsing process visualization (step-by-step elaboration).
# ---------------------------------------------------------------------------

def _draw_frontier_strip(ax, y, items, *, x_left=0.5, x_right=15.5,
                        merge_indices=None, highlight_indices=None):
    """Draw a horizontal strip of boxes representing the parse frontier
    at one step. Returns x-centers of each box."""
    n = len(items)
    available = x_right - x_left
    box_w = min(2.4, available / max(n, 1) * 0.85)
    gap = (available - n * box_w) / max(n + 1, 1)
    xs = []
    for i, item in enumerate(items):
        cx = x_left + gap + box_w / 2 + i * (box_w + gap)
        is_highlight = highlight_indices and i in highlight_indices
        fill = "#ffe8d1" if is_highlight else "#eaf2fb"
        edge = "#b45309" if is_highlight else "#2b6cb0"
        ax.add_patch(FancyBboxPatch(
            (cx - box_w / 2, y - 0.45), box_w, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.4, edgecolor=edge, facecolor=fill))
        # Label
        if isinstance(item, tuple):
            tag, body = item
            ax.text(cx, y + 0.18, tag, ha="center", va="center",
                    fontsize=9.0, fontweight="bold", color=edge,
                    family="serif")
            ax.text(cx, y - 0.20, body, ha="center", va="center",
                    fontsize=8.5, color="#333",
                    family="monospace")
        else:
            ax.text(cx, y, item, ha="center", va="center",
                    fontsize=10.5, color="#222",
                    family="serif")
        xs.append(cx)
    return xs, box_w


def make_parsing_process_figure(path: str) -> None:
    """Five-frame vertical sequence showing TRELLIS parsing
    'the cat saw a dog' bottom-up."""
    frames = [
        ("Step 0  initial frontier (primitives)",
         ["the", "cat", "saw", "a", "dog"], None, None),
        ("Step 1  commit (the, cat) → NP_1",
         [("the", ""), ("cat", ""), "saw", "a", "dog"], (0, 1), [0, 1]),
        ("Step 2  commit (a, dog) → NP_2",
         [("NP_1", "the cat"), "saw", ("a", ""), ("dog", "")],
         (2, 3), [2, 3]),
        ("Step 3  commit (saw, NP_2) → VP",
         [("NP_1", "the cat"), ("saw", ""), ("NP_2", "a dog")],
         (1, 2), [1, 2]),
        ("Step 4  commit (NP_1, VP) → S",
         [("NP_1", "the cat"), ("VP", "saw a dog")], (0, 1), [0, 1]),
        ("Final  partonomic parse tree",
         [("S", "the cat saw a dog")], None, [0]),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 11.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, len(frames) * 1.85 + 0.5)
    ax.set_axis_off()

    y_top = len(frames) * 1.85
    for i, (label, items, merge, hl) in enumerate(frames):
        y = y_top - i * 1.85
        ax.text(0.2, y + 0.6, label, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="#222",
                family="serif")
        xs, box_w = _draw_frontier_strip(
            ax, y, items, x_left=0.3, x_right=15.7,
            highlight_indices=hl)
        # Arrow pointing down to the merge composite on the next row.
        if merge is not None and i + 1 < len(frames):
            a, b = merge
            mid_x = (xs[a] + xs[b]) / 2
            ax.annotate("",
                        xy=(mid_x, y - 1.05), xytext=(mid_x, y - 0.45),
                        arrowprops=dict(arrowstyle="->",
                                        color="#b45309", lw=1.6))

    fig.suptitle("Parsing process on \"the cat saw a dog\"  "
                 "(step-by-step elaboration)",
                 fontsize=13.5, fontweight="bold", color="#222")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 5: generation process visualization (step-by-step expansion).
# ---------------------------------------------------------------------------

def make_generation_process_figure(path: str) -> None:
    """Five-frame vertical sequence showing TRELLIS generating a
    sentence top-down by expanding a composite seed."""
    frames = [
        ("Step 0  composite seed (S leaf)",
         [("S", "...")],
         None, [0]),
        ("Step 1  expand S → (NP, VP)",
         [("NP", "..."), ("VP", "...")],
         (0,), [0, 1]),
        ("Step 2  expand NP → (Det, N)",
         [("Det", "..."), ("N", "..."), ("VP", "...")],
         (0,), [0, 1]),
        ("Step 3  Det, N reach primitives",
         ["the", "cat", ("VP", "...")],
         (2,), [2]),
        ("Step 4  expand VP → (V, NP)",
         ["the", "cat", ("V", "..."), ("NP", "...")],
         (3,), [2, 3]),
        ("Final  every branch terminates at a primitive",
         ["the", "cat", "saw", "a", "dog"],
         None, None),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 11.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, len(frames) * 1.85 + 0.5)
    ax.set_axis_off()

    y_top = len(frames) * 1.85
    for i, (label, items, expand, hl) in enumerate(frames):
        y = y_top - i * 1.85
        ax.text(0.2, y + 0.6, label, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="#222",
                family="serif")
        xs, box_w = _draw_frontier_strip(
            ax, y, items, x_left=0.3, x_right=15.7,
            highlight_indices=hl)
        if expand is not None and i + 1 < len(frames):
            # arrow from the expanding composite down to next row
            idx = expand[0]
            mid_x = xs[idx]
            ax.annotate("",
                        xy=(mid_x, y - 1.05), xytext=(mid_x, y - 0.45),
                        arrowprops=dict(arrowstyle="->",
                                        color="#1f7d4e", lw=1.6))

    fig.suptitle("Generation process inverting the parse  "
                 "(step-by-step expansion)",
                 fontsize=13.5, fontweight="bold", color="#222")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------

def main():
    print(f"loading trained TRELLIS from {TRAINED_MODEL}")
    trellis = TRELLIS.load_state(TRAINED_MODEL)
    print(f"  content tree count: {trellis.ltm.content_hierarchy.root.count}")
    print(f"  context tree count: {trellis.ltm.context_hierarchy.root.count}")
    print(f"  vocab size:         {len(trellis.ltm.id_to_value)}")

    make_instances_figure(os.path.join(OUT_DIR, "instances.png"), trellis)
    make_hierarchies_figure(os.path.join(OUT_DIR, "hierarchies.png"), trellis)
    make_generation_curves_figure(
        os.path.join(OUT_DIR, "generation_curves.png"))
    make_parsing_process_figure(
        os.path.join(OUT_DIR, "parsing_process.png"))
    make_generation_process_figure(
        os.path.join(OUT_DIR, "generation_process.png"))


if __name__ == "__main__":
    main()
