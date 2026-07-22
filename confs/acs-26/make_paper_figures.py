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
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
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
                 highlight=False, hidden_attrs=None,
                 font_scale=1.0):
    """Draw a Cobweb-style concept node.

    Parameters
    ----------
    title : header string (concept name).
    count : instance count shown as ``n=...`` (None to omit).
    attrs : list of (attribute_label, [(value_label, prob), ...]).
    hidden_attrs : set of attribute_labels whose row should be drawn
        in a dimmed / italic style with a dashed separator line above,
        indicating that the field is stored as metadata but not used
        by Cobweb's routing.
    font_scale : multiplier for all in-node font sizes. Callers with
        small boxes (e.g., the instances figure) can pass a value < 1
        so the label/value text fits without overflowing rows.
    """
    hidden_attrs = set(hidden_attrs or ())
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
            ha="left", va="center", fontsize=11.5 * font_scale,
            color=title_text,
            fontweight="bold", family="monospace")
    if count is not None:
        if isinstance(count, float):
            count_str = f"{count:g}"
        else:
            count_str = str(count)
        ax.text(cx + w * 0.45, title_bottom + title_h / 2,
                f"n={count_str}",
                ha="right", va="center", fontsize=11.0 * font_scale,
                color=title_text,
                family="monospace")

    total_lines = sum(1 + len(values) for _, values in attrs)
    if total_lines == 0:
        total_lines = 1
    # Inner top padding so the first row sits well clear of the title bar
    # bottom edge, plus a small bottom padding for symmetry.
    top_pad = max(body_h * 0.06, 0.10)
    bot_pad = max(body_h * 0.02, 0.04)
    usable_h = body_h - top_pad - bot_pad
    line_h = usable_h / total_lines
    cur_y = title_bottom - top_pad

    for attr_name, values in attrs:
        is_hidden = attr_name in hidden_attrs
        cur_y -= line_h
        if is_hidden:
            # Draw a thin dashed separator above the hidden row.
            ax.add_line(Line2D(
                [cx - w * 0.46, cx + w * 0.46],
                [cur_y + line_h, cur_y + line_h],
                color="#999", linewidth=0.6, linestyle=":"))
        head_color  = "#888" if is_hidden else "#222"
        head_style  = "italic" if is_hidden else "normal"
        ax.text(cx - w * 0.46, cur_y + line_h * 0.5, f"{attr_name}:",
                ha="left", va="center", fontsize=14.5 * font_scale,
                family="monospace",
                color=head_color, fontweight="bold", style=head_style)
        if is_hidden:
            ax.text(cx + w * 0.46, cur_y + line_h * 0.5,
                    "(hidden)",
                    ha="right", va="center", fontsize=12.5 * font_scale,
                    family="monospace", color="#999", style="italic")
        for val, prob in values:
            cur_y -= line_h
            val_color = "#888" if is_hidden else "#444"
            val_style = "italic" if is_hidden else "normal"
            ax.text(cx - w * 0.42, cur_y + line_h * 0.5, val,
                    ha="left", va="center", fontsize=12.0 * font_scale,
                    family="monospace", color=val_color, style=val_style)
            if isinstance(prob, str):
                prob_str = prob
            elif isinstance(prob, int) or prob >= 1:
                prob_str = f"{int(prob)}"
            else:
                prob_str = f"{prob:.2f}"
            ax.text(cx + w * 0.48, cur_y + line_h * 0.5, prob_str,
                    ha="right", va="center", fontsize=12.0 * font_scale,
                    family="monospace", color="#888" if is_hidden else "#666",
                    style=val_style)


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
    """Show a complete parse tree for "the cat saw a dog", highlight a
    primitive leaf (green) and a sentence-final composite subtree
    (red), and lay out the corresponding instance boxes:

      * primitive ("cat") -- context instance only, on the LEFT.
      * composite ("saw a dog", at the sentence's right edge) --
        content instance + context instance, BOTH on the RIGHT.

    The composite's right-context attributes are EMPTYNULL by
    construction, which is the property the figure makes legible.
    """
    itv = trellis.ltm.id_to_value
    sentence = "the cat saw a dog"
    words = sentence.split()
    fp = trellis.parse_sentence(
        sentence, threshold="converge",
        new_vocab=False, learning=False,
    )

    # ---- find a HIGHER-UP composite (e.g. VP saw + NP_2) whose
    # children mix a primitive and a composite, plus the primitive
    # "cat". Showing this composite makes the recursive structure of
    # the content instance visible: content-left is a primitive ref,
    # content-right is a composite ref.
    cat_node = None
    target_comp = None
    for node in fp.nodes:
        if isinstance(node, PrimitiveParseNode):
            try:
                w = itv[node.word_id]
            except Exception:
                continue
            if w == "cat" and cat_node is None:
                cat_node = node
        elif isinstance(node, CompositeParseNode):
            kids = [c for _, c in node.children]
            if len(kids) != 2: continue
            is_prim = [isinstance(k, PrimitiveParseNode) for k in kids]
            # Want a "mixed" composite: one primitive child + one
            # composite child. Prefer the VP (saw + NP).
            if is_prim.count(True) == 1:
                if target_comp is None:
                    target_comp = node
                else:
                    # Prefer the higher (larger-cplx) composite when
                    # several mixed candidates are available.
                    if (getattr(node, "complexity", 0)
                            > getattr(target_comp, "complexity", 0)):
                        target_comp = node
    # Fallback: smallest NP-like (2 primitives) if no mixed comp found.
    if target_comp is None:
        target_comp = _pick_np_chunk(fp)

    # Build a human-readable yield for the chosen composite.
    def _yield_words(node):
        out = []
        def w(n):
            if isinstance(n, PrimitiveParseNode):
                try: out.append(itv[n.word_id])
                except Exception: out.append("?")
                return
            for _, c in n.children: w(c)
        w(node)
        return out
    target_yield = " ".join(_yield_words(target_comp))

    ci_comp = target_comp.content_instance
    xi_comp = target_comp.context_instance
    xi_prim = cat_node.get_context_instance() if cat_node else {}

    # ---- attribute extraction helpers ----
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

    def _context_attrs_from_inst(xi):
        out = []
        pos_keys = sorted([k for k in xi.keys()
                           if isinstance(k, int) and k >= 0])
        slot_keys = pos_keys[:-1] if len(pos_keys) >= 3 else pos_keys
        half = len(slot_keys) // 2
        for k in slot_keys:
            d = xi.get(k, {})
            rows = _top_from_inst(d, top_k=2)
            if k < half:
                slot_name = f"left-{half - k}"
            else:
                slot_name = f"right+{k - half + 1}"
            if rows:
                out.append((slot_name, rows))
            else:
                out.append((slot_name, [("EMPTYNULL", 0.0)]))
        return out

    # Hand-constructed instances so the figure makes a clean teaching
    # point: 4 slot attributes (left-2, left-1, right+1, right+2) on
    # each context instance, plus a "content-ref" row that points at
    # the element's content-tree identifier. The content-ref row is
    # stored on every instance as metadata but is *hidden* from
    # Cobweb's context-tree routing (see caption).
    def _short_ctx_id(prefix, n):
        return f"{prefix}-{n:06d}"
    SAW_REF        = _short_ctx_id("CTX_CONCEPT", 264738)
    NP_AD_REF      = _short_ctx_id("CTX_CONCEPT", 247665)
    VP_CONTENT_REF = _short_ctx_id("CNT_CONCEPT", 321001)  # VP's content-tree leaf
    VP_CTX_REF     = _short_ctx_id("CTX_CONCEPT", 198754)  # VP's context-tree leaf
    PRIM_CAT_WORD  = "cat"

    comp_content_attrs = [
        ("content-left",  [(SAW_REF,        1)]),
        ("content-right", [(NP_AD_REF,      1)]),
    ]
    # Composite VP "saw a dog" sits at the END of the sentence, so its
    # right context is uniformly EMPTYNULL.
    comp_context_attrs = [
        ("left-2",      [("the",            1.00)]),
        ("left-1",      [("cat",            1.00)]),
        ("right+1",     [("EMPTYNULL",      1.00)]),
        ("right+2",     [("EMPTYNULL",      1.00)]),
        ("content-ref", [(VP_CONTENT_REF,   1)]),
    ]
    # Primitive "cat" -- 4 slot attrs from the surrounding words, plus a
    # content-ref that, for primitives, IS the surface word itself.
    prim_context_attrs = [
        ("left-2",      [("EMPTYNULL",      1.00)]),
        ("left-1",      [("the",            1.00)]),
        ("right+1",     [("saw",            1.00)]),
        ("right+2",     [("a",              1.00)]),
        ("content-ref", [(PRIM_CAT_WORD,    1)]),
    ]
    HIDDEN_ATTRS = {"content-ref"}

    # ---- figure layout ----
    # Three-column condensed layout:
    #   LEFT:   primitive context box (single box)
    #   MIDDLE: parse tree for "the cat saw a dog"
    #   RIGHT:  composite content (top) and composite context (bottom)
    fig, ax = plt.subplots(figsize=(13.0, 4.5))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 4.5)
    ax.set_axis_off()

    # Composite element is highlighted in amber/orange (not red) so it
    # reads as visually distinct from the pink primitive boxes.
    COMP_COLOR = "#d97706"
    PRIM_COLOR = "#1f7d4e"
    CHUNK_BLUE = "#3a5e8a"

    # ----- Tree (centred horizontally between the two box columns) -----
    # Leaf x-positions are spaced so that adjacent primitive boxes
    # (width 0.92) have a small horizontal gap between them.
    leaf_x = {"the": 4.35, "cat": 5.45, "saw": 6.60,
              "a":   7.85, "dog": 8.95}
    np1_x   = (leaf_x["the"] + leaf_x["cat"]) / 2
    np2_x   = (leaf_x["a"]   + leaf_x["dog"]) / 2
    vp_x    = (leaf_x["saw"] + np2_x) / 2
    s_x     = (np1_x + vp_x) / 2

    y_leaf = 0.95
    y_l3   = 1.55
    y_l2   = 2.20
    y_top  = 2.85

    pos_S    = (s_x,           y_top)
    pos_NP1  = (np1_x,         y_l2)
    pos_VP   = (vp_x,          y_l2)
    pos_the  = (leaf_x["the"], y_l3)
    pos_cat  = (leaf_x["cat"], y_l3)
    pos_saw  = (leaf_x["saw"], y_l3)
    pos_NP2  = (np2_x,         y_l3)
    pos_a    = (leaf_x["a"],   y_leaf)
    pos_dog  = (leaf_x["dog"], y_leaf)

    edges = [(pos_S, pos_NP1), (pos_S, pos_VP),
             (pos_NP1, pos_the), (pos_NP1, pos_cat),
             (pos_VP, pos_saw), (pos_VP, pos_NP2),
             (pos_NP2, pos_a), (pos_NP2, pos_dog)]
    for p, c in edges:
        ax.add_line(Line2D([p[0], c[0]],
                           [p[1] - 0.20, c[1] + 0.22],
                           color="#555", linewidth=1.0, zorder=2))

    # Composites (S, NP_1, VP, NP_2).  VP is the highlighted COMPOSITE
    # because it has MIXED children -- "saw" (primitive) + NP_2 (composite)
    # -- which makes the recursive content-instance structure visible.
    _circle_chunk(ax, *pos_S, 0.28, label="S", fontsize=14.5)
    _circle_chunk(ax, *pos_NP1, 0.26, label="NP", fontsize=13.5)
    _circle_chunk(ax, *pos_VP, 0.30,
                  fill="#fef3c7", edge=COMP_COLOR,
                  label="VP", fontsize=14.5)
    _circle_chunk(ax, *pos_NP2, 0.26, label="NP", fontsize=13.5)

    # Primitives.  "cat" is the highlighted PRIMITIVE (green ring).
    def _draw_prim(pos, w, *, highlight=None):
        edge = highlight or "#8a4844"
        ax.add_patch(FancyBboxPatch(
            (pos[0] - 0.46, pos[1] - 0.23), 0.92, 0.46,
            boxstyle="round,pad=0.015,rounding_size=0.04",
            linewidth=2.0 if highlight else 1.4,
            edgecolor=edge,
            facecolor="#eafaf1" if highlight == PRIM_COLOR else "#f3a9a4",
            zorder=3))
        ax.text(pos[0], pos[1], w, ha="center", va="center",
                fontsize=13.5,
                color=PRIM_COLOR if highlight == PRIM_COLOR else "#1a1a1a",
                fontweight="bold" if highlight else "normal",
                family="serif", zorder=3.1)

    _draw_prim(pos_the, "the")
    _draw_prim(pos_cat, "cat", highlight=PRIM_COLOR)
    _draw_prim(pos_saw, "saw")
    _draw_prim(pos_a, "a")
    _draw_prim(pos_dog, "dog")

    # The colour coding (green = primitive, red = composite) is
    # explained in the caption; the previous in-tree text labels have
    # been removed to free the area below the cat node for the
    # down-and-around primitive arrow.

    # ----- 3 instance boxes -----
    # LEFT column: primitive context box (cat).
    # RIGHT column: composite content (top) and composite context
    # (bottom), stacked.  Boxes use a tall, narrow shape so the 5-row
    # attribute lists (4 slots + content-ref) fit cleanly.
    prim_box_cx, prim_box_cy = 1.80, 2.30
    prim_box_w,  prim_box_h  = 3.40, 3.15
    comp_box_w,  comp_box_h  = 3.25, 2.20
    comp_cont_cx, comp_cont_cy = 11.30, 3.35
    comp_ctx_cx,  comp_ctx_cy  = 11.30, 1.10

    _cobweb_node(ax, cx=prim_box_cx, cy=prim_box_cy,
                 w=prim_box_w, h=prim_box_h,
                 title=PRIM_CAT_WORD,
                 count=None, attrs=prim_context_attrs,
                 title_fill=PRIM_COLOR, edge_color=PRIM_COLOR,
                 hidden_attrs=HIDDEN_ATTRS, font_scale=0.58)
    _cobweb_node(ax, cx=comp_cont_cx, cy=comp_cont_cy,
                 w=comp_box_w, h=comp_box_h,
                 title=VP_CONTENT_REF,
                 count=None, attrs=comp_content_attrs,
                 title_fill=CHUNK_BLUE, edge_color=CHUNK_BLUE,
                 hidden_attrs=HIDDEN_ATTRS, font_scale=0.58)
    _cobweb_node(ax, cx=comp_ctx_cx, cy=comp_ctx_cy,
                 w=comp_box_w, h=comp_box_h,
                 title=VP_CTX_REF,
                 count=None, attrs=comp_context_attrs,
                 title_fill=PRIM_COLOR, edge_color=PRIM_COLOR,
                 hidden_attrs=HIDDEN_ATTRS, font_scale=0.58)

    # ---- Arrow helpers ----
    def _path_arrow(points, *, color, lw=1.3):
        """Draw a polyline from a list of (x, y) waypoints with an
        arrowhead on the final segment."""
        from matplotlib.patches import FancyArrowPatch
        for (sx, sy), (tx, ty) in zip(points[:-2], points[1:-1]):
            ax.add_line(Line2D([sx, tx], [sy, ty],
                               color=color, linewidth=lw, zorder=4))
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1],
            arrowstyle="-|>", mutation_scale=12,
            color=color, lw=lw, zorder=4))

    # cat -> primitive context box (LEFT). Down-and-around: exit cat
    # at the BOTTOM, drop below the tree's leaf row, run LEFT under
    # everything, and enter the prim box from below.
    prim_box_bot = prim_box_cy - prim_box_h / 2
    under_y      = 0.20
    _path_arrow(
        [(pos_cat[0], pos_cat[1] - 0.20),
         (pos_cat[0], under_y),
         (prim_box_cx, under_y),
         (prim_box_cx, prim_box_bot)],
        color=PRIM_COLOR,
    )
    # VP -> composite content (TOP RIGHT). Three segments: RIGHT to a
    # midpoint x, UP to the box's y, then RIGHT into the box's left
    # edge midpoint.
    mid_x              = 9.40
    comp_box_left_edge = comp_cont_cx - comp_box_w / 2
    _path_arrow(
        [(pos_VP[0] + 0.30, pos_VP[1]),
         (mid_x,            pos_VP[1]),
         (mid_x,            comp_cont_cy),
         (comp_box_left_edge, comp_cont_cy)],
        color=COMP_COLOR,
    )
    # VP -> composite context (BOTTOM RIGHT). Same shape but DOWN
    # at the midpoint.
    _path_arrow(
        [(pos_VP[0] + 0.30, pos_VP[1]),
         (mid_x,            pos_VP[1]),
         (mid_x,            comp_ctx_cy),
         (comp_box_left_edge, comp_ctx_cy)],
        color=COMP_COLOR,
    )

    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 2: hierarchies.png  (illustrative subtrees with crafted attribute
# distributions; counts at parents sum the children's counts).
# ---------------------------------------------------------------------------

def make_hierarchies_figure(path: str, trellis=None) -> None:
    """Draw two side-by-side Cobweb-style subtrees that illustrate the
    content/context distinction.

    Content hierarchy: composite concepts grouped by their compositional
    pattern. The content-left and content-right slots always reference
    other concepts in the context hierarchy (shown as ``CTX_CONCEPT-...``).

    Context hierarchy: REAL distributions pulled from the trained
    context_hierarchy --- top-2 children of the root, top-2 grandchildren
    of each, four attributes per node (left-2, left-1, right-1, right-2)
    drawn from the actual av_count buckets at attribute slots 1, 2, 3, 4.

    All parent counts equal the sum of their children's counts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 15.0))

    # ----- content hierarchy node sizing -----
    CONT_NODE_W, CONT_NODE_H = 5.4, 3.8
    CONT_LEAF_W, CONT_LEAF_H = 5.0, 3.2
    # ----- context hierarchy needs taller nodes (4 attributes) -----
    # The root has a top-2 + "..." rendering (3 rows per slot), so it
    # gets a slightly taller box than the mid level; the leaves have a
    # single word per slot (1 row), so they get a shorter box. All
    # heights bumped to accommodate the significantly larger row fonts
    # without text overlap.
    CTX_NODE_W,  CTX_NODE_H  = 5.6, 4.5
    CTX_ROOT_H               = 5.0
    CTX_LEAF_W,  CTX_LEAF_H  = 5.0, 3.5

    # ====================================================================
    # Build the context-subtree distributions from one MED-grammar sample
    # sentence by masking out a word at each of four positions. Each leaf
    # represents the context instance of the masked word (single word per
    # slot, count 1); mid-level nodes sum their two leaves, and the root
    # sums all four leaves.
    # ====================================================================
    SAMPLE_SENT = ["the", "big", "cat", "saw", "a", "quick", "dog"]

    def _ctx_instance_for_mask(tokens, p):
        """Single-word-per-slot context instance for the masked word at
        position ``p`` in ``tokens``."""
        def _w(idx):
            return tokens[idx] if 0 <= idx < len(tokens) else "EMPTYNULL"
        return [
            ("left-2",  [(_w(p - 2), 1)]),
            ("left-1",  [(_w(p - 1), 1)]),
            ("right-1", [(_w(p + 1), 1)]),
            ("right-2", [(_w(p + 2), 1)]),
        ]

    def _sum_ctx_attrs(attrs_list):
        """Sum across leaf attribute lists, preserving slot order."""
        from collections import defaultdict
        bucket = defaultdict(lambda: defaultdict(int))
        for attrs in attrs_list:
            for slot, rows in attrs:
                for w, c in rows:
                    bucket[slot][w] += c
        out = []
        for slot in ("left-2", "left-1", "right-1", "right-2"):
            words = sorted(bucket[slot].items(), key=lambda kv: -kv[1])
            out.append((slot, list(words)))
        return out

    # Pedagogical grouping: pair noun-like positions (cat, dog) under
    # one mid, and verb/adj-like positions (saw, quick) under the other,
    # so the mid-level distributions look like emerging POS clusters.
    MASK_POSITIONS_FLAT = [2, 6, 3, 5]   # cat, dog, saw, quick
    LEAF_INSTANCES = [_ctx_instance_for_mask(SAMPLE_SENT, p)
                      for p in MASK_POSITIONS_FLAT]
    LEAF_COUNTS = [1, 1, 1, 1]

    MID_INSTANCES = [
        _sum_ctx_attrs([LEAF_INSTANCES[0], LEAF_INSTANCES[1]]),   # noun-like
        _sum_ctx_attrs([LEAF_INSTANCES[2], LEAF_INSTANCES[3]]),   # non-noun
    ]
    MID_COUNTS    = [2, 2]
    ROOT_INSTANCE = _sum_ctx_attrs(LEAF_INSTANCES)
    ROOT_COUNT    = 4

    def _truncate_with_ellipsis(attrs, top_k=2):
        """Keep only the top-k entries per slot and append a `...` row
        when there were more, indicating truncation in the display."""
        out = []
        for slot, rows in attrs:
            if len(rows) > top_k:
                out.append((slot, list(rows[:top_k]) + [("...", "...")]))
            else:
                out.append((slot, list(rows)))
        return out

    ROOT_INSTANCE_DISPLAY = _truncate_with_ellipsis(ROOT_INSTANCE, top_k=2)

    # Titles for the seven context-tree nodes we draw.
    CTX_ROOT_TITLE  = "CTX_CONCEPT-100000"
    CTX_MID_TITLES  = ["CTX_CONCEPT-110001", "CTX_CONCEPT-120001"]
    CTX_LEAF_FLAT   = ["CTX_CONCEPT-111001",  # mask "cat"
                       "CTX_CONCEPT-111002",  # mask "dog"
                       "CTX_CONCEPT-121001",  # mask "saw"
                       "CTX_CONCEPT-121002"]  # mask "quick"

    # Two visible context-tree leaves referenced from the content tree.
    REF_LEAF_L = CTX_LEAF_FLAT[1]   # noun-position (mask "dog")
    REF_LEAF_R = CTX_LEAF_FLAT[2]   # verb-position (mask "saw")

    # ====================================================================
    # Vertical layout (data coords). Shorter than before to remove
    # empty space; rows are spaced just enough for clean gaps between
    # node boxes.
    # ====================================================================
    YMAX = 15.0
    CONT_ROW = (12.3, 6.8, 2.3)   # root / middle / leaf  (CONT)
    CTX_ROW  = (12.0, 6.5, 2.0)   # root / middle / leaf  (CTX)

    # ====================================================================
    # LEFT: content subtree — all attribute values are CTX_CONCEPT-* IDs
    # ====================================================================
    ax = axes[0]
    ax.set_xlim(0, 21)
    ax.set_ylim(0, YMAX)
    ax.set_axis_off()
    # subtree title removed (info in caption)

    # --- counts (parent = sum of children) ---
    # Illustrative counts sized to keep the arithmetic legible: 1 at every
    # content leaf, 2 at each intermediate, 4 at the root.
    cnt_NP_Det_N         = 1
    cnt_NP_Det_AdjP_N    = 1
    cnt_NP_like          = cnt_NP_Det_N + cnt_NP_Det_AdjP_N      # 2
    cnt_VP_V_NP          = 1
    cnt_VP_V_NP_PP       = 1
    cnt_VP_like          = cnt_VP_V_NP + cnt_VP_V_NP_PP          # 2
    cnt_content_root     = cnt_NP_like + cnt_VP_like             # 4

    # Content-tree CTX_CONCEPT-* references. Visible context-tree leaves
    # use real concept hashes (REF_LEAF_L, REF_LEAF_R, etc.); off-screen
    # ones are synthesized digit IDs to keep the figure self-contained.
    CTX_VISIBLE = CTX_LEAF_FLAT   # [leaf_2.6, leaf_7.4, leaf_13.1, leaf_17.9]
    CTX_OFF1 = "CTX_CONCEPT-140287"
    CTX_OFF2 = "CTX_CONCEPT-145561"
    CTX_OFF3 = "CTX_CONCEPT-561989"

    _cobweb_node(ax, cx=10.0, cy=CONT_ROW[0], w=CONT_NODE_W, h=CONT_NODE_H,
                 title="CNT_CONCEPT-300001", count=cnt_content_root,
                 attrs=[
                     ("content-left",
                      [(CTX_VISIBLE[2], 1),
                       (CTX_OFF1,       1),
                       ("...",          "...")]),
                     ("content-right",
                      [(CTX_VISIBLE[1], 1),
                       (CTX_OFF2,       1),
                       ("...",          "...")]),
                 ],
                 title_fill="#1f4e79")

    np_cx, vp_cx = 4.5, 15.5
    _cobweb_node(ax, cx=np_cx, cy=CONT_ROW[1], w=CONT_NODE_W, h=CONT_NODE_H,
                 title="CNT_CONCEPT-310001", count=cnt_NP_like,
                 attrs=[
                     ("content-left",
                      [(CTX_VISIBLE[2], 1),
                       (CTX_VISIBLE[0], 1)]),
                     ("content-right",
                      [(CTX_VISIBLE[1], 1),
                       (CTX_OFF1,        1)]),
                 ],
                 title_fill="#2b6cb0")
    _link(ax, (10.0, CONT_ROW[0] - CONT_NODE_H / 2),
          (np_cx, CONT_ROW[1] + CONT_NODE_H / 2))

    _cobweb_node(ax, cx=vp_cx, cy=CONT_ROW[1], w=CONT_NODE_W, h=CONT_NODE_H,
                 title="CNT_CONCEPT-320001", count=cnt_VP_like,
                 attrs=[
                     ("content-left",
                      [(CTX_OFF1,        1),
                       (CTX_VISIBLE[2],  1)]),
                     ("content-right",
                      [(CTX_OFF2, 1),
                       (CTX_OFF3, 1)]),
                 ],
                 title_fill="#2b6cb0")
    _link(ax, (10.0, CONT_ROW[0] - CONT_NODE_H / 2),
          (vp_cx, CONT_ROW[1] + CONT_NODE_H / 2))

    _cobweb_node(ax, cx=2.8, cy=CONT_ROW[2], w=CONT_LEAF_W, h=CONT_LEAF_H,
                 title="CNT_CONCEPT-311001", count=cnt_NP_Det_N,
                 attrs=[
                     ("content-left",  [(CTX_VISIBLE[2], 1)]),
                     ("content-right", [(CTX_VISIBLE[1], 1)]),
                 ],
                 title_fill="#2b6cb0")
    _link(ax, (np_cx, CONT_ROW[1] - CONT_NODE_H / 2),
          (2.8, CONT_ROW[2] + CONT_LEAF_H / 2))

    _cobweb_node(ax, cx=7.9, cy=CONT_ROW[2], w=CONT_LEAF_W, h=CONT_LEAF_H,
                 title="CNT_CONCEPT-311002", count=cnt_NP_Det_AdjP_N,
                 attrs=[
                     ("content-left",  [(CTX_VISIBLE[2], 1)]),
                     ("content-right", [(CTX_OFF1,       1)]),
                 ],
                 title_fill="#2b6cb0")
    _link(ax, (np_cx, CONT_ROW[1] - CONT_NODE_H / 2),
          (7.9, CONT_ROW[2] + CONT_LEAF_H / 2))

    _cobweb_node(ax, cx=13.0, cy=CONT_ROW[2], w=CONT_LEAF_W, h=CONT_LEAF_H,
                 title="CNT_CONCEPT-321001", count=cnt_VP_V_NP,
                 attrs=[
                     ("content-left",  [(CTX_OFF1, 1)]),
                     ("content-right", [(CTX_OFF2, 1)]),
                 ],
                 title_fill="#2b6cb0")
    _link(ax, (vp_cx, CONT_ROW[1] - CONT_NODE_H / 2),
          (13.0, CONT_ROW[2] + CONT_LEAF_H / 2))

    _cobweb_node(ax, cx=18.1, cy=CONT_ROW[2], w=CONT_LEAF_W, h=CONT_LEAF_H,
                 title="CNT_CONCEPT-321002", count=cnt_VP_V_NP_PP,
                 attrs=[
                     ("content-left",  [(CTX_OFF1, 1)]),
                     ("content-right", [(CTX_OFF3, 1)]),
                 ],
                 title_fill="#2b6cb0")
    _link(ax, (vp_cx, CONT_ROW[1] - CONT_NODE_H / 2),
          (18.1, CONT_ROW[2] + CONT_LEAF_H / 2))

    # ====================================================================
    # RIGHT: context subtree — REAL distributions from trained model
    # ====================================================================
    ax = axes[1]
    ax.set_xlim(0, 21)
    ax.set_ylim(0, YMAX)
    ax.set_axis_off()
    # subtree title removed (info in caption)

    _cobweb_node(ax, cx=10.0, cy=CTX_ROW[0], w=CTX_NODE_W, h=CTX_ROOT_H,
                 title=CTX_ROOT_TITLE, count=ROOT_COUNT,
                 attrs=ROOT_INSTANCE_DISPLAY,
                 title_fill="#2f7050")

    mid_cxs = [4.5, 15.5]
    for mi, cmid in enumerate(mid_cxs):
        _cobweb_node(ax, cx=cmid, cy=CTX_ROW[1],
                     w=CTX_NODE_W, h=CTX_NODE_H,
                     title=CTX_MID_TITLES[mi], count=MID_COUNTS[mi],
                     attrs=MID_INSTANCES[mi],
                     title_fill="#3c8062")
        _link(ax, (10.0, CTX_ROW[0] - CTX_ROOT_H / 2),
              (cmid, CTX_ROW[1] + CTX_NODE_H / 2))

    # Leaf x positions, in display order matching CTX_LEAF_FLAT.
    leaf_cxs = [2.8, 7.9, 13.0, 18.1]
    leaf_mid_cx = [mid_cxs[0], mid_cxs[0], mid_cxs[1], mid_cxs[1]]
    for i, lx in enumerate(leaf_cxs):
        _cobweb_node(ax, cx=lx, cy=CTX_ROW[2],
                     w=CTX_LEAF_W, h=CTX_LEAF_H,
                     title=CTX_LEAF_FLAT[i], count=LEAF_COUNTS[i],
                     attrs=LEAF_INSTANCES[i],
                     title_fill="#3c8062")
        _link(ax, (leaf_mid_cx[i], CTX_ROW[1] - CTX_NODE_H / 2),
              (lx, CTX_ROW[2] + CTX_LEAF_H / 2))

    # suptitle removed (info in caption)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # ---- cross-hierarchy reference arrows ----------------------------
    # Two arrows show how content-tree CTX_CONCEPT-* references resolve
    # to visible context-tree leaves:
    #   CNT_CONCEPT-311001 (Det + N)  --content-left-->  CTX_CONCEPT-121001 (Det)
    #   CNT_CONCEPT-311001 (Det + N)  --content-right--> CTX_CONCEPT-111002 (N)
    # The arrows are drawn in figure coordinates as U-shaped paths that
    # exit the source's BOTTOM edge, dip below every leaf, run along
    # a horizontal band near the bottom of the figure, and come back UP
    # into the target's BOTTOM edge. They therefore go AROUND every
    # node instead of cutting through any of them.
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch

    CROSS_COLOR = "#9b1c5b"

    def _to_fig(ax, xy):
        """Convert (x, y) in `ax` data coords to figure-fraction coords."""
        return fig.transFigure.inverted().transform(
            ax.transData.transform(xy)
        )

    def _draw_u_arrow(src_ax, src_xy, tgt_ax, tgt_xy, *, dip_y=0.04,
                      color=CROSS_COLOR, lw=1.2):
        s = _to_fig(src_ax, src_xy)
        t = _to_fig(tgt_ax, tgt_xy)
        verts = [
            (s[0], s[1]),         # source (bottom of source leaf)
            (s[0], dip_y),        # straight down to dip band
            (t[0], dip_y),        # across along dip band
            (t[0], t[1]),         # straight up to target (bottom of target leaf)
        ]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
        path = Path(verts, codes)
        arrow = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            lw=lw,
            transform=fig.transFigure,
            zorder=10,
        )
        fig.add_artist(arrow)

    # Source: bottom-center of CNT_CONCEPT-311001 (Det + N) at axes[0].
    src_x = 2.6
    src_y = CONT_ROW[2] - CONT_LEAF_H / 2

    # Both U-arrows dip well below every node so each arrowhead can
    # rise back up cleanly into the bottom of its target context leaf.
    DIP_Y = 0.015
    _draw_u_arrow(axes[0], (src_x, src_y),
                  axes[1], (13.1, CTX_ROW[2] - CTX_LEAF_H / 2),
                  dip_y=DIP_Y)
    _draw_u_arrow(axes[0], (src_x, src_y),
                  axes[1], (7.4, CTX_ROW[2] - CTX_LEAF_H / 2),
                  dip_y=DIP_Y)

    # Vertical dotted divider between the content and context panels.
    from matplotlib.lines import Line2D
    divider = Line2D(
        [0.5, 0.5], [0.04, 0.92],
        transform=fig.transFigure,
        color="#888888", linestyle=":", linewidth=1.2,
        zorder=2,
    )
    fig.add_artist(divider)

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
        # panel title removed (info in caption)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5)

    # suptitle removed (info in caption)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {path}")



# ---------------------------------------------------------------------------
# Figures 6 + 7: parsing / generation INFOGRAPHICS.
#
# Visual style follows the original "Chunkweb: Building Parse Trees"
# cartoon (rounded primitive boxes + circles for chunks + a Cobweb
# hierarchy + a parse tree + numbered explanation text on the right)
# and the multi-panel narrative flow of Fig. 1 in the comp-diffusion
# paper.  Two separate figures: one for parsing, one for generation.
# ---------------------------------------------------------------------------

# ---- shared palette -------------------------------------------------------
PRIM_FILL   = "#f3a9a4"
PRIM_EDGE   = "#8a4844"
CHUNK_FILL  = "#a8c5f0"
CHUNK_EDGE  = "#3a5e8a"
CAND_FILL   = "#b69ecf"
CAND_EDGE   = "#5d3f7d"
HIER_FILL   = "#ffffff"
HIER_EDGE   = "#444444"
HIER_GRAY   = "#cccccc"
LINK_COLOR  = "#555555"

import matplotlib.patches as mpatches


def _primitive_box(ax, cx, cy, w, h, word, *, edge=PRIM_EDGE, fill=PRIM_FILL,
                   fontsize=11):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(cx, cy, word, ha="center", va="center",
            fontsize=fontsize, color="#1a1a1a", family="serif",
            zorder=3.1)


def _circle_chunk(ax, cx, cy, r, *, edge=CHUNK_EDGE, fill=CHUNK_FILL,
                  label=None, fontsize=9):
    ax.add_patch(mpatches.Circle((cx, cy), r, facecolor=fill,
                                 edgecolor=edge, linewidth=1.8,
                                 zorder=3))
    if label:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, color="#1a1a1a",
                family="serif", fontweight="bold", zorder=3.1)


ANCHOR_FILL = "#fef3c7"
ANCHOR_EDGE = "#b45309"


def _mini_hierarchy(ax, cx, cy, *, title, title_color="#222",
                    highlight_path=None, scale=1.0,
                    highlight_basic=None, highlight_leaf=None,
                    highlight_leaves=None, leaf_labels=None,
                    anchor_leaves=None, highlight_basics=None,
                    traversal_arrows=None, shape="deep_left",
                    show_title=True):
    """Draw a 4-level Cobweb-style hierarchy with asymmetric branching.

    Structure:
        root
        / \
       L   R
      /|  |\\
     LL LR RL RR
     /\\
   LLL LLR

    ``highlight_path`` is a list of node-id strings filled grey.
    ``highlight_basic`` (or ``highlight_basics`` list) is the node-id(s)
        ringed with a red dashed circle (the basic-level concept marker).
    ``highlight_leaf`` is the node-id drawn with a purple fill (the
        sampled leaf, used in the generation infographic).
    ``anchor_leaves`` is a list of node-ids drawn with an amber fill
        (the anchor leaf from which the basic-level walk begins).
    ``traversal_arrows`` is a list of (src_id, tgt_id, color, kind)
        tuples drawn as curved arrows on the hierarchy. ``kind`` is
        ``"up"`` for an upward (anchor → basic) arrow or ``"down"`` for
        a downward (basic → sampled-leaf) arrow.
    """
    if highlight_path is None:
        highlight_path = []
    s = scale
    if shape == "deep_right":
        # Mirror-image layout: deep subtree under R_R, with an extra
        # branching at L so the overall silhouette is visibly different
        # from the deep_left shape.
        nodes = {
            "root":   (cx,             cy + 2.05 * s),
            "L":      (cx - 0.95 * s,  cy + 1.15 * s),
            "R":      (cx + 0.95 * s,  cy + 1.15 * s),
            "L_L":    (cx - 1.45 * s,  cy + 0.25 * s),
            "L_M":    (cx - 0.95 * s,  cy + 0.25 * s),
            "L_R":    (cx - 0.45 * s,  cy + 0.25 * s),
            "R_L":    (cx + 0.50 * s,  cy + 0.25 * s),
            "R_R":    (cx + 1.45 * s,  cy + 0.25 * s),
            "R_R_L":  (cx + 1.10 * s,  cy - 0.65 * s),
            "R_R_R":  (cx + 1.65 * s,  cy - 0.65 * s),
        }
        edges = [
            ("root", "L"), ("root", "R"),
            ("L", "L_L"), ("L", "L_M"), ("L", "L_R"),
            ("R", "R_L"), ("R", "R_R"),
            ("R_R", "R_R_L"), ("R_R", "R_R_R"),
        ]
    else:
        nodes = {
            "root":   (cx,             cy + 2.05 * s),
            "L":      (cx - 0.95 * s,  cy + 1.15 * s),
            "R":      (cx + 0.95 * s,  cy + 1.15 * s),
            "L_L":    (cx - 1.45 * s,  cy + 0.25 * s),
            "L_R":    (cx - 0.50 * s,  cy + 0.25 * s),
            "R_L":    (cx + 0.50 * s,  cy + 0.25 * s),
            "R_R":    (cx + 1.45 * s,  cy + 0.25 * s),
            "L_L_L":  (cx - 1.65 * s,  cy - 0.65 * s),
            "L_L_R":  (cx - 1.10 * s,  cy - 0.65 * s),
        }
        edges = [
            ("root", "L"), ("root", "R"),
            ("L", "L_L"), ("L", "L_R"),
            ("R", "R_L"), ("R", "R_R"),
            ("L_L", "L_L_L"), ("L_L", "L_L_R"),
        ]
    for a, b in edges:
        ax.add_line(Line2D([nodes[a][0], nodes[b][0]],
                           [nodes[a][1], nodes[b][1]],
                           color="#444", linewidth=0.9))
    r = 0.18 * s
    purple_set = set(highlight_leaves or [])
    if highlight_leaf is not None:
        purple_set.add(highlight_leaf)
    anchor_set = set(anchor_leaves or [])
    basic_set = set(highlight_basics or [])
    if highlight_basic is not None:
        basic_set.add(highlight_basic)
    for nid, (x, y) in nodes.items():
        if nid in purple_set:
            fill = CAND_FILL
            edge = CAND_EDGE
        elif nid in anchor_set:
            fill = ANCHOR_FILL
            edge = ANCHOR_EDGE
        elif nid in highlight_path:
            fill = HIER_GRAY
            edge = HIER_EDGE
        else:
            fill = HIER_FILL
            edge = HIER_EDGE
        ax.add_patch(mpatches.Circle((x, y), r,
                                     facecolor=fill, edgecolor=edge,
                                     linewidth=1.3, zorder=3))
    for bid in basic_set:
        if bid in nodes:
            ax.add_patch(mpatches.Circle(nodes[bid], r + 0.13,
                                         facecolor="none",
                                         edgecolor="#c0392b",
                                         linewidth=1.6,
                                         linestyle="--", zorder=3.1))
    # Curved traversal arrows (up: anchor → basic, down: basic → sample).
    if traversal_arrows:
        from matplotlib.patches import FancyArrowPatch as _FAP2
        for spec in traversal_arrows:
            src_id, tgt_id, color, kind = spec
            if src_id not in nodes or tgt_id not in nodes:
                continue
            sx, sy = nodes[src_id]
            tx, ty = nodes[tgt_id]
            # offset the source/target so the arrow head doesn't overlap
            # the circle outlines.
            side = -1.0 if kind == "up" else 1.0
            rad = 0.35 * side
            arrow = _FAP2((sx, sy), (tx, ty),
                          connectionstyle=f"arc3,rad={rad}",
                          arrowstyle="-|>", mutation_scale=12,
                          color=color, lw=1.5, zorder=4.5,
                          shrinkA=6, shrinkB=6)
            ax.add_patch(arrow)
    # Optional small labels next to specific nodes. Label entry may be
    # either a plain string (default: below the node) or a tuple
    # ``(text, dx, dy)`` for custom placement.
    if leaf_labels:
        for nid, label in leaf_labels.items():
            if nid not in nodes:
                continue
            x, y = nodes[nid]
            if isinstance(label, tuple):
                text, dx, dy = label
            else:
                text, dx, dy = label, 0.0, -r - 0.20
            # Choose colour: amber for anchor leaves, purple for purple
            # (sampled) leaves, otherwise dark.
            if nid in anchor_set:
                color = ANCHOR_EDGE
            elif nid in purple_set:
                color = CAND_EDGE
            else:
                color = "#1a1a1a"
            ax.text(x + dx, y + dy, text,
                    ha="center", va="top", fontsize=8.0,
                    color=color, fontweight="bold",
                    family="serif")
    if show_title:
        ax.text(cx, cy + 2.55 * s, title,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=title_color, family="serif")
    return nodes


def _panel_box(ax, x, y, w, h, *, number, title, title_color="#1f4e79"):
    """Draw a thin-bordered panel with a black/white numbered title bar.
    ``title_color`` kept for API compatibility but unused; bar is black."""
    # Outer box.
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.04",
        linewidth=1.0, edgecolor="#444", facecolor="#ffffff"))
    # Black title bar.
    title_h = 0.46
    bar_color = "#1a1a1a"
    ax.add_patch(FancyBboxPatch(
        (x + 0.05, y + h - title_h - 0.05), w - 0.10, title_h,
        boxstyle="round,pad=0.005,rounding_size=0.04",
        linewidth=0, facecolor=bar_color))
    # White circle with black number on the left of the title.
    ax.add_patch(mpatches.Circle((x + 0.32, y + h - title_h / 2 - 0.05), 0.17,
                                 facecolor="white", edgecolor=bar_color,
                                 linewidth=1.2))
    ax.text(x + 0.32, y + h - title_h / 2 - 0.05, str(number),
            ha="center", va="center", fontsize=10.5, color="#1a1a1a",
            fontweight="bold", family="serif")
    # White title text.
    ax.text(x + 0.60, y + h - title_h / 2 - 0.05, title,
            ha="left", va="center", fontsize=11,
            fontweight="bold", color="white", family="serif")
    # Content area bounds.
    return (x + 0.08, y + 0.08, w - 0.16, h - title_h - 0.16)


def _panel_arrow(ax, x1, y, x2, *, color="#666"):
    ax.annotate("",
                xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=1.6,
                                mutation_scale=18))


def make_parse_infographic_figure(path: str) -> None:
    # Three-panel banner. All terminals AND unexpanded non-terminals
    # share a single leaf row; chunked composites sit above. No italic
    # subtitle/caption text -- equivalent prose lives in the LaTeX
    # figure caption.
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.5)
    ax.set_axis_off()

    PANEL_Y, PANEL_H = 0.10, 4.30

    # =================================================================
    # Panel 1: candidate generation. All five primitives (the, dog,
    # eats, a, cat) sit on ONE leaf row; NP_1 (committed earlier)
    # sits above its children the/dog at the chunk row. Three candidate
    # chunks at the top span (NP_1, eats), (eats, a), (a, cat).
    # =================================================================
    p1 = _panel_box(ax, x=0.10, y=PANEL_Y, w=4.55, h=PANEL_H, number=1,
                    title="Generate candidates")
    cx0, cy0, cw, ch = p1

    leaf_y    = cy0 + 0.55
    chunk_y   = cy0 + 1.85
    cand_y    = cy0 + 3.10

    prim_w    = 0.50
    leaf_xs   = [cx0 + 0.50 + (cw - 1.00) * i / 4 for i in range(5)]
    primitive_words = ["the", "dog", "eats", "a", "cat"]

    np1_x   = (leaf_xs[0] + leaf_xs[1]) / 2

    # NP_1's edges to the/dog (drawn first so circles sit on top).
    for child_x in (leaf_xs[0], leaf_xs[1]):
        ax.add_line(Line2D([np1_x, child_x],
                           [chunk_y - 0.22, leaf_y + 0.20],
                           color="#666", linewidth=0.9, zorder=2))

    # Leaf row.
    for x, w in zip(leaf_xs, primitive_words):
        _primitive_box(ax, x, leaf_y, prim_w, 0.40, w, fontsize=11.0)
    # NP_1 above the/dog.
    _circle_chunk(ax, np1_x, chunk_y, 0.22, label="NP", fontsize=11.0)

    # Frontier element x positions = [NP_1, eats, a, cat] -- one chunk
    # at chunk_y, three primitives at leaf_y. Candidates fan over the
    # three adjacent pairs.
    frontier_xs = [np1_x, leaf_xs[2], leaf_xs[3], leaf_xs[4]]
    frontier_ys = [chunk_y, leaf_y, leaf_y, leaf_y]
    cand_xs = [(frontier_xs[i] + frontier_xs[i + 1]) / 2
               for i in range(3)]
    cand_r = 0.22
    for i, cx_cand in enumerate(cand_xs):
        for j in (i, i + 1):
            ax.add_line(Line2D([cx_cand, frontier_xs[j]],
                               [cand_y - cand_r, frontier_ys[j] + 0.22],
                               color=CAND_EDGE, linewidth=0.9,
                               linestyle=":"))
    for cx_cand in cand_xs:
        _circle_chunk(ax, cx_cand, cand_y, cand_r,
                      fill=CAND_FILL, edge=CAND_EDGE)

    # =================================================================
    # Panel 2: sort the picked candidate through both hierarchies and
    # show that its joint score clears the recognition threshold.
    # Layout (top -> bottom):
    #   1. Two mini Cobweb-style hierarchies (Content + Context).
    #   2. Worked-example score with a green check-marked threshold box.
    # =================================================================
    p2 = _panel_box(ax, x=4.95, y=PANEL_Y, w=4.95, h=PANEL_H, number=2,
                    title="Score through both hierarchies")
    cx0, cy0, cw, ch = p2
    HIER_SCALE = 0.45
    HIER_CY    = cy0 + 2.10
    cont_nodes = _mini_hierarchy(ax,
                                 cx=cx0 + cw * 0.27,
                                 cy=HIER_CY,
                                 title="Content",
                                 title_color="#2b6cb0",
                                 highlight_path=[
                                     "root", "L", "L_L", "L_L_R"],
                                 scale=HIER_SCALE)
    ctx_nodes  = _mini_hierarchy(ax,
                                 cx=cx0 + cw * 0.73,
                                 cy=HIER_CY,
                                 title="Context",
                                 title_color="#3c8062",
                                 highlight_path=[
                                     "root", "R", "R_L"],
                                 scale=HIER_SCALE)
    SCORE_COLOR    = "#2f8a3e"
    SCORE_FILL     = "#e8f7ed"
    score_cx = cx0 + cw / 2
    score_cy = cy0 + 0.95
    score_w  = cw - 0.35
    score_h  = 0.85
    ax.add_patch(FancyBboxPatch(
        (score_cx - score_w / 2, score_cy - score_h / 2),
        score_w, score_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=SCORE_COLOR,
        facecolor=SCORE_FILL, zorder=3))
    ax.text(score_cx, score_cy + 0.22,
            r"score $=\ \log P_{\mathrm{cnt}}(c) + \log P_{\mathrm{ctx}}(c)"
            r"\ =\ -5.5$",
            ha="center", va="center", fontsize=11.0,
            color="#1a1a1a", family="serif", zorder=4)
    ax.text(score_cx, score_cy - 0.18,
            r"$-5.5\ \geq\ \tau\ =\ -12.0$    "
            r"$\checkmark$ commit",
            ha="center", va="center", fontsize=12.0,
            color=SCORE_COLOR, fontweight="bold",
            family="serif", zorder=4)
    # Two arrows from the (basic-level / leaf) results in each
    # hierarchy down into the score box.
    for src_node, sign in ((cont_nodes["L_L_R"], -1),
                           (ctx_nodes["R_L"],   +1)):
        ax.annotate("",
                    xy=(score_cx + sign * 0.4, score_cy + score_h / 2),
                    xytext=(src_node[0], src_node[1] - 0.18 * HIER_SCALE),
                    arrowprops=dict(arrowstyle="->", color="#6e6e6e",
                                    lw=1.0))

    # =================================================================
    # Panel 3: partial parse tree (forest) after this commit, plus the
    # NEXT round of purple candidate chunks over the resulting frontier
    # [NP_1, eats, NP_2].
    # =================================================================
    p3 = _panel_box(ax, x=10.20, y=PANEL_Y, w=4.70, h=PANEL_H, number=3,
                    title="Commit to partial parse tree")
    cx0, cy0, cw, ch = p3

    # Three vertical rows. ALL primitives (the/dog/eats/a/cat) share
    # the leaf row; NP_1 and NP_2 sit above their respective children;
    # two purple candidates over the new frontier [NP_1, eats, NP_2].
    leaf_y     = cy0 + 0.55
    chunk_y    = cy0 + 1.85
    cand_y     = cy0 + 3.10
    prim_w     = 0.50
    leaf_xs    = [cx0 + 0.50 + (cw - 1.00) * i / 4 for i in range(5)]
    primitive_words = ["the", "dog", "eats", "a", "cat"]

    np1_x  = (leaf_xs[0] + leaf_xs[1]) / 2
    eats_x = leaf_xs[2]
    np2_x  = (leaf_xs[3] + leaf_xs[4]) / 2

    # Edges from NP_1, NP_2 down to their leaf children.
    for parent_x, child_xs in ((np1_x, (leaf_xs[0], leaf_xs[1])),
                               (np2_x, (leaf_xs[3], leaf_xs[4]))):
        for child_x in child_xs:
            ax.add_line(Line2D([parent_x, child_x],
                               [chunk_y - 0.22, leaf_y + 0.20],
                               color="#444", linewidth=1.0,
                               zorder=2))

    # Leaf row: all 5 primitives.
    for x, w in zip(leaf_xs, primitive_words):
        _primitive_box(ax, x, leaf_y, prim_w, 0.40, w, fontsize=11.0)
    # Chunk row: NP_1, NP_2 (blue, committed).
    _circle_chunk(ax, np1_x, chunk_y, 0.22, label="NP", fontsize=11.0)
    _circle_chunk(ax, np2_x, chunk_y, 0.22, label="NP", fontsize=11.0)

    # Candidate row: two purple proposals (NP_1+eats) and (eats+NP_2).
    # Frontier elements at MIXED y levels (NP_1, NP_2 at chunk_y; eats
    # at leaf_y), so candidate attachment lines diverge accordingly.
    frontier_pts = {"np1": (np1_x, chunk_y),
                    "eats": (eats_x, leaf_y),
                    "np2": (np2_x, chunk_y)}
    cand_specs = [(("np1", "eats"), (np1_x + eats_x) / 2),
                  (("eats", "np2"), (eats_x + np2_x) / 2)]
    cand_r = 0.22
    for (lkey, rkey), cx_cand in cand_specs:
        for key in (lkey, rkey):
            tx, ty = frontier_pts[key]
            # Attach to the TOP of frontier element (chunk or primitive).
            if key in ("np1", "np2"):
                top_y = ty + 0.22
            else:
                top_y = ty + 0.22  # primitive box half-height
            ax.add_line(Line2D([cx_cand, tx],
                               [cand_y - cand_r, top_y],
                               color=CAND_EDGE, linewidth=0.9,
                               linestyle=":"))
    for _, cx_cand in cand_specs:
        _circle_chunk(ax, cx_cand, cand_y, cand_r,
                      fill=CAND_FILL, edge=CAND_EDGE)

    # =================================================================
    # Inter-panel arrows.
    # =================================================================
    arrow_y = PANEL_Y + PANEL_H / 2
    _panel_arrow(ax, 4.62, arrow_y, 4.97, color="#444")
    _panel_arrow(ax, 9.87, arrow_y, 10.22, color="#444")

    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {path}")


def make_generation_infographic_figure(path: str) -> None:
    # Four-panel banner. Panels 2 + 3 show the up-then-down resampling
    # walk inside the content and context hierarchies respectively.
    fig, ax = plt.subplots(figsize=(16.5, 4.5))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 4.5)
    ax.set_axis_off()

    PANEL_Y, PANEL_H = 0.05, 4.40

    # =================================================================
    # Panel 1: partial generation tree — VP is the next seed
    # =================================================================
    p1 = _panel_box(ax, x=0.05, y=PANEL_Y, w=3.70, h=PANEL_H,
                    number=1, title="Locate next seed",
                    title_color="#222")
    cx0, cy0, cw, ch = p1
    base_x = cx0 + cw / 2
    # Partial tree: S -> (NP -> the dog) + VP (unexpanded).
    # All leaves of the partial tree -- terminals (the, dog) AND the
    # unexpanded non-terminal VP -- share the SAME leaf row.
    leaf_y = cy0 + 0.95
    s_pos      = (base_x,         cy0 + 2.95)
    np_pos     = (base_x - 1.00,  cy0 + 1.95)
    the_pos    = (base_x - 1.50,  leaf_y)
    dog_pos    = (base_x - 0.55,  leaf_y)
    vp_pos     = (base_x + 1.10,  leaf_y)

    for parent, child, top_dy, bot_dy in [
            (s_pos, np_pos,  0.25, 0.22),
            (s_pos, vp_pos,  0.25, 0.30),
            (np_pos, the_pos, 0.22, 0.20),
            (np_pos, dog_pos, 0.22, 0.20)]:
        ax.add_line(Line2D([parent[0], child[0]],
                           [parent[1] - top_dy, child[1] + bot_dy],
                           color="#444", linewidth=1.0, zorder=2))

    _circle_chunk(ax, *s_pos, 0.27, label="S", fontsize=13)
    _circle_chunk(ax, *np_pos, 0.25, label="NP", fontsize=12)
    _circle_chunk(ax, *vp_pos, 0.30,
                  fill=ANCHOR_FILL, edge=ANCHOR_EDGE,
                  label="VP", fontsize=14)
    _primitive_box(ax, the_pos[0], the_pos[1], 0.60, 0.40,
                   "the", fontsize=11.0)
    _primitive_box(ax, dog_pos[0], dog_pos[1], 0.60, 0.40,
                   "dog", fontsize=11.0)

    ax.text(vp_pos[0], vp_pos[1] - 0.60,
            "next seed",
            ha="center", va="center", fontsize=12,
            color=ANCHOR_EDGE, fontweight="bold", family="serif")
    ax.add_line(Line2D([vp_pos[0], vp_pos[0]],
                       [vp_pos[1] - 0.30, vp_pos[1] - 0.48],
                       color=ANCHOR_EDGE, linewidth=1.0))

    # =================================================================
    # Panel 2: sample new content leaf -- seed → basic level (one node
    # higher than the seed's direct parent) → sampled leaf.
    # =================================================================
    p2 = _panel_box(ax, x=3.90, y=PANEL_Y, w=4.00, h=PANEL_H,
                    number=2,
                    title="Sample new content leaf",
                    title_color="#222")
    cx0, cy0, cw, ch = p2
    SCALE = 0.95
    ax.text(cx0 + cw / 2, cy0 + ch - 0.10,
            "Content", ha="center", va="top",
            fontsize=13.5, fontweight="bold",
            color="#2b6cb0", family="serif")
    cont_nodes = _mini_hierarchy(
        ax,
        cx=cx0 + cw / 2,
        cy=cy0 + 1.35,
        title="",
        scale=SCALE,
        show_title=False,
        highlight_path=["root"],
        highlight_basic="L",
        anchor_leaves=["L_L_L"],
        highlight_leaf="L_R",
        leaf_labels={"L_L_L": ("seed",         -0.10, -0.32),
                     "L_R":   ("sampled leaf",  0.10, -0.32)},
        traversal_arrows=[
            ("L_L_L", "L", ANCHOR_EDGE, "up"),
            ("L", "L_R", CAND_EDGE, "down"),
        ],
    )

    # =================================================================
    # Panel 3: sample new context leaves. For each of the new content
    # leaf's two children, walk seed → basic level → sampled leaf in
    # the context hierarchy. The left sampled leaf turns out to be the
    # primitive ``eats``; the right sampled leaf is a new composite
    # that becomes the next seed.
    # =================================================================
    p3 = _panel_box(ax, x=8.05, y=PANEL_Y, w=4.00, h=PANEL_H,
                    number=3,
                    title="Sample new context leaves",
                    title_color="#222")
    cx0, cy0, cw, ch = p3
    ax.text(cx0 + cw / 2, cy0 + ch - 0.10,
            "Context", ha="center", va="top",
            fontsize=13.5, fontweight="bold",
            color="#3c8062", family="serif")
    ctx_nodes = _mini_hierarchy(
        ax,
        cx=cx0 + cw / 2,
        cy=cy0 + 1.35,
        title="",
        scale=SCALE,
        show_title=False,
        shape="deep_right",
        highlight_path=["root"],
        highlight_basics=["L", "R_R"],
        anchor_leaves=["L_L", "R_R_L"],
        highlight_leaves=["L_R", "R_R_R"],
        leaf_labels={"L_R":   ("\"eats\"",      0.0, -0.32),
                     "R_R_R": ("new composite", -0.22, -0.32)},
        traversal_arrows=[
            ("L_L", "L", ANCHOR_EDGE, "up"),
            ("L", "L_R", CAND_EDGE, "down"),
            ("R_R_L", "R_R", ANCHOR_EDGE, "up"),
            ("R_R", "R_R_R", CAND_EDGE, "down"),
        ],
    )

    # =================================================================
    # Panel 4: expand the seed into the partial generation tree
    # =================================================================
    p3 = _panel_box(ax, x=12.20, y=PANEL_Y, w=4.25, h=PANEL_H,
                    number=4, title="Expand seed",
                    title_color="#222")
    cx0, cy0, cw, ch = p3
    base_x = cx0 + cw / 2

    s_pos      = (base_x,         cy0 + 2.95)
    np_pos     = (base_x - 1.15,  cy0 + 1.95)
    vp_pos     = (base_x + 1.15,  cy0 + 1.95)
    np_l_pos   = (base_x - 1.60,  cy0 + 0.95)
    np_r_pos   = (base_x - 0.70,  cy0 + 0.95)
    vp_l_pos   = (base_x + 0.60,  cy0 + 0.95)
    vp_r_pos   = (base_x + 1.60,  cy0 + 0.95)

    for parent, child in [(s_pos, np_pos), (s_pos, vp_pos),
                          (np_pos, np_l_pos), (np_pos, np_r_pos),
                          (vp_pos, vp_l_pos), (vp_pos, vp_r_pos)]:
        ax.add_line(Line2D([parent[0], child[0]],
                           [parent[1] - 0.25, child[1] + 0.20],
                           color="#444", linewidth=1.0, zorder=2))

    _circle_chunk(ax, *s_pos, 0.27, label="S", fontsize=13)
    _circle_chunk(ax, *np_pos, 0.25, label="NP", fontsize=12)
    # VP is the seed that was just expanded -- draw it in amber (anchor
    # colours) to mark it as the expanded seed.
    _circle_chunk(ax, *vp_pos, 0.27,
                  fill=ANCHOR_FILL, edge=ANCHOR_EDGE,
                  label="VP", fontsize=13)
    _primitive_box(ax, np_l_pos[0], np_l_pos[1], 0.60, 0.40,
                   "the", fontsize=11.0)
    _primitive_box(ax, np_r_pos[0], np_r_pos[1], 0.60, 0.40,
                   "dog", fontsize=11.0)
    _primitive_box(ax, vp_l_pos[0], vp_l_pos[1], 0.60, 0.40,
                   "eats", fontsize=11.0)
    # New composite is the next seed -> purple.
    _circle_chunk(ax, *vp_r_pos, 0.23,
                  fill=CAND_FILL, edge=CAND_EDGE,
                  label="NP", fontsize=11.0)

    ax.text(vp_pos[0] - 0.50, vp_pos[1],
            "expanded",
            ha="right", va="center", fontsize=11.0,
            color=ANCHOR_EDGE, fontweight="bold", family="serif")
    ax.text(vp_r_pos[0], vp_r_pos[1] - 0.40,
            "next seed",
            ha="center", va="center", fontsize=11.0,
            color=CAND_EDGE, fontweight="bold", family="serif")

    # =================================================================
    # Inter-panel arrows (centered vertically on the panel content).
    # =================================================================
    arrow_y = PANEL_Y + PANEL_H / 2
    _panel_arrow(ax, 3.76, arrow_y, 3.92, color="#444")
    _panel_arrow(ax, 7.91, arrow_y, 8.07, color="#444")
    _panel_arrow(ax, 12.06, arrow_y, 12.22, color="#444")

    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {path}")


def make_sample_parses_figure(out_dir: str) -> None:
    """One full-width sample-parse figure per grammar. Each tree is the
    canonical CFG derivation under the grammar's productions (NOT the
    binarised system-internal representation), so the structure is
    directly readable against the grammar in Appendix~A."""

    # Canonical CFG trees for one feature-rich sentence per grammar.
    # Each node is a tuple (label, children, is_primitive). Primitive
    # leaves carry their surface word as label; non-primitives carry
    # the grammar nonterminal symbol (S / NP / VP / PP / RelClause /
    # AdjP). POS preterminals (Det / N / V / P / Adj / RelPro) are
    # collapsed into their child to keep the tree readable.

    def lf(word):   return (word, [], True)
    def nt(sym, *kids): return (sym, list(kids), False)

    EXAMPLES = {
        # SMALL: S -> NP VP, VP -> V NP, NP -> Det N. Strictly binary,
        # no unary syntactic productions.
        "small": (
            "the dog chased a cat",
            nt("S",
                nt("NP", lf("the"), lf("dog")),
                nt("VP",
                    lf("chased"),
                    nt("NP", lf("a"), lf("cat"))))
        ),
        # MED: every syntactic production is strictly binary.
        #   S     -> NP VP
        #   NP    -> Det N | Det AdjP
        #   AdjP  -> Adj N | Adj AdjP       (adjective chaining via AdjP)
        #   VP    -> V NP | V VPobj
        #   VPobj -> NP PP
        #   PP    -> P NP
        # The subject NP uses AdjP to chain two adjectives; bare-noun
        # NPs ("a park", "the cat") skip AdjP and go straight to Det N.
        "med": (
            "a quick big telescope found a park with the cat",
            nt("S",
                nt("NP",
                    lf("a"),
                    nt("AdjP",
                        lf("quick"),
                        nt("AdjP",
                            lf("big"),
                            lf("telescope")))),
                nt("VP",
                    lf("found"),
                    nt("VPobj",
                        nt("NP", lf("a"), lf("park")),
                        nt("PP",
                            lf("with"),
                            nt("NP", lf("the"), lf("cat"))))))
        ),
        # LARGE: every syntactic production is strictly binary.
        #   S         -> NP VP
        #   NP        -> Det N | Det AdjP | Det Nbar
        #   AdjP      -> Adj N | Adj AdjP            (adjective phrase only)
        #   Nbar      -> N RelClause | AdjP RelClause (head + rel clause)
        #   VP        -> V NP | V VPobj | V PP
        #   VPobj     -> NP PP
        #   RelClause -> RelPro VP
        #   PP        -> P NP
        # The subject NP uses ``Det Nbar`` where Nbar is the head noun
        # ``girl`` plus a relative clause; the matrix VP takes the
        # ``V PP`` form.
        "large": (
            "the girl who read a teacher saw with the boy",
            nt("S",
                nt("NP",
                    lf("the"),
                    nt("Nbar",
                        lf("girl"),
                        nt("RelClause",
                            lf("who"),
                            nt("VP",
                                lf("read"),
                                nt("NP", lf("a"), lf("teacher")))))),
                nt("VP",
                    lf("saw"),
                    nt("PP",
                        lf("with"),
                        nt("NP", lf("the"), lf("boy")))))
        ),
    }

    def _pick(grammar):
        return EXAMPLES.get(grammar)

    def _tree_depth(tree):
        _, kids, _ = tree
        if not kids:
            return 1
        return 1 + max(_tree_depth(k) for k in kids)

    def _count_leaves(tree):
        _, kids, _ = tree
        if not kids:
            return 1
        return sum(_count_leaves(k) for k in kids)

    def _assign_positions(tree, depth, leaf_iter, depth_y):
        """Assign x by leaf order, y by depth. ``leaf_iter`` mutates as
        leaves are encountered (left to right). Returns (x, y, label,
        is_primitive) and a list of edges (parent_xy, child_xy)."""
        label, kids, is_prim = tree
        y = depth_y[depth]
        if not kids:
            x = next(leaf_iter)
            return (x, y, label, True), []
        child_recs = [_assign_positions(k, depth + 1, leaf_iter, depth_y)
                      for k in kids]
        cx_list = [c[0][0] for c in child_recs]
        x = sum(cx_list) / len(cx_list)
        edges = []
        for child_rec, child_edges in child_recs:
            edges.extend(child_edges)
            edges.append(((x, y), (child_rec[0], child_rec[1])))
        return (x, y, label, False), edges

    def _node_height(tree):
        _, kids, _ = tree
        if not kids: return 0
        return 1 + max(_node_height(k) for k in kids)

    def _draw_tree(ax, tree, x_left, x_right, y_top, y_bottom):
        """Conventional NLP parse-tree layout:
          * every primitive (leaf) sits on the SAME bottom row;
          * every composite sits at height-from-leaves;
          * a composite's x is the mean of its leaf-descendants' x,
            so it always centres directly above the span it covers.
        """
        N_leaves = _count_leaves(tree)
        H_max    = _node_height(tree)
        if N_leaves == 1:
            leaf_xs = [(x_left + x_right) / 2]
        else:
            leaf_xs = [x_left + (x_right - x_left) * i / (N_leaves - 1)
                       for i in range(N_leaves)]
        # Y rows by height-from-leaves.
        height_y = [y_bottom + (y_top - y_bottom) * (h / max(H_max, 1))
                    for h in range(H_max + 1)]

        positions = []     # (x, y, label, is_primitive)
        edges_acc = []

        leaf_iter = iter(leaf_xs)

        def _go(t):
            label, kids, is_prim = t
            if not kids:
                x = next(leaf_iter)
                y = height_y[0]
                positions.append((x, y, label, True))
                return (x, y, [x])
            cps = [_go(k) for k in kids]
            # Centre this composite over the mean of its CHILDREN'S
            # x positions (not all leaf descendants). This avoids the
            # composite stacking directly above a deeper composite with
            # the same leaf-span centroid.
            child_xs = [cp[0] for cp in cps]
            x = sum(child_xs) / len(child_xs)
            y = height_y[_node_height(t)]
            positions.append((x, y, label, False))
            for cp in cps:
                edges_acc.append(((x, y), (cp[0], cp[1])))
            all_leaves = [lx for _, _, ls in cps for lx in ls]
            return (x, y, all_leaves)

        _go(tree)

        # Edges first (under the nodes). Use the box half-heights
        # (0.20 for both composite and primitive) so each edge meets
        # the bottom of the parent and the top of the child.
        for (px, py), (cx, cy) in edges_acc:
            ax.add_line(Line2D([px, cx], [py - 0.20, cy + 0.20],
                               color="#555", linewidth=0.9, zorder=2))
        # Nodes on top. Composite labels (S, NP, VP, PP, AdjP,
        # RelClause) sit in compact rounded rectangles sized to the
        # label; primitives are pink leaf boxes.
        for (x, y, label, is_prim) in positions:
            if is_prim:
                _primitive_box(ax, x, y, 0.85, 0.40, label,
                               fontsize=10)
            else:
                # Compact: tighter padding + smaller text than before.
                nt_w = 0.30 + 0.13 * max(2, len(label))
                ax.add_patch(FancyBboxPatch(
                    (x - nt_w / 2, y - 0.20), nt_w, 0.40,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    linewidth=1.3, edgecolor="#3a5e8a",
                    facecolor="#dceaf7", zorder=3))
                ax.text(x, y, label, ha="center", va="center",
                        fontsize=10, color="#1a1a1a",
                        family="serif", fontweight="bold",
                        zorder=3.1)

    def _render(grammar):
        spec = _pick(grammar)
        if spec is None:
            print(f"  [{grammar}] no example")
            return
        sentence, tree = spec

        D = _tree_depth(tree)
        fig_w = 14.0
        fig_h = max(2.6, 1.1 + 0.62 * D)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.set_axis_off()

        ax.text(fig_w / 2, fig_h - 0.25,
                f"“{sentence}”",
                ha="center", va="top", fontsize=14,
                color="#1a1a1a", family="serif", fontweight="bold")

        _draw_tree(ax, tree,
                   x_left=0.6, x_right=fig_w - 0.6,
                   y_top=fig_h - 0.85, y_bottom=0.45)

        out_path = os.path.join(out_dir, f"sample_parses_{grammar}.png")
        plt.savefig(out_path, dpi=180,
                    bbox_inches="tight", facecolor="white",
                    pad_inches=0.12)
        plt.close(fig)
        print(f"wrote {out_path}")

    for g in ("small", "med", "large"):
        _render(g)


# ---------------------------------------------------------------------------

def main():
    print(f"loading trained TRELLIS from {TRAINED_MODEL}")
    trellis = TRELLIS.load_state(TRAINED_MODEL)
    print(f"  content tree count: {trellis.ltm.content_hierarchy.root.count}")
    print(f"  context tree count: {trellis.ltm.context_hierarchy.root.count}")
    print(f"  vocab size:         {len(trellis.ltm.id_to_value)}")

    make_instances_figure(os.path.join(OUT_DIR, "instances.png"), trellis)
    make_hierarchies_figure(os.path.join(OUT_DIR, "hierarchies.png"), trellis)
    make_parse_infographic_figure(
        os.path.join(OUT_DIR, "parse_infographic.png"))
    make_generation_infographic_figure(
        os.path.join(OUT_DIR, "generation_infographic.png"))
    make_sample_parses_figure(OUT_DIR)
    # The grammar-experiment + terminal-experiment comparison PNGs are produced
    # by regenerate_viz.py (which reads the saved per-seed CSVs), then
    # copied into ./graphics/ as grids_grammar_experiment.png and
    # grids_terminal_experiment.png. They are not re-rendered here.


if __name__ == "__main__":
    main()
