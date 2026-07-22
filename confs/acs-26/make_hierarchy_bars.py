"""make_hierarchy_bars.py — render the content / context hierarchy bars
shown in Section 5 of the acs-26 paper.

This script TRAINS a fresh TRELLIS model using the EXACT SAME
hyperparameters, primitive-only warm-up, and hollow-corpus replay
pipeline as ``tests/met5/grammar_distillation_test.py``. After
training, it computes per-node POS / chunk-class distributions and
renders bar-decorated trees for the first three levels of both the
content and the context hierarchies. Outputs:

  confs/acs-26/paper/graphics/hierarchy_bars_context.png
  confs/acs-26/paper/graphics/hierarchy_bars_content.png

Run::

    python confs/acs-26/make_hierarchy_bars.py
"""
from __future__ import annotations
import os, sys, glob, json, random
from collections import defaultdict, Counter
from functools import lru_cache as _lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from parse_mh import (
    TRELLIS, PrimitiveParseNode, CompositeParseNode,
    FiniteParseTree,
    _context_weight, _get_or_register_cplx_vid,
)
from util.cfg import (
    TEST_GRAMMAR1, TEST_CORPUS1, generate,
)
try:
    from cobweb import cobweb_set_seed
except Exception:
    def cobweb_set_seed(_):
        pass

# Match the distillation script's palette + classes -----------------------
POS_LIST     = ["Det", "N", "Adj", "V", "P"]
CHUNK_LABELS = ["NP", "AdjP", "PP", "VP", "VPobj", "S"]
ALL_LABELS   = POS_LIST + CHUNK_LABELS + ["OTHER"]
N_LABEL      = len(ALL_LABELS)
_label2id    = {lbl: i for i, lbl in enumerate(ALL_LABELS)}
LABEL_COLOR  = {
    "Det":   "#2ca02c", "N":     "#8c564b", "Adj":   "#1f77b4",
    "V":     "#17becf", "P":     "#7f7f7f",
    "NP":    "#ff7f0e", "AdjP":  "#9467bd",
    "PP":    "#bcbd22",
    "VP":    "#e377c2", "VPobj": "#f7b6d2",
    "S":     "#d62728", "OTHER": "#cccccc",
}

# Build a word -> POS lookup from the MED grammar.
WORD_TO_POS = {}
for pos in POS_LIST:
    for prod in TEST_GRAMMAR1[pos]:
        for w in prod:
            WORD_TO_POS[w] = pos

# Configuration -- mirrors grammar_distillation_test.py -----------------
HOLLOW_CORPUS_DIR   = os.path.join(_ROOT, "data", "cfg_grammar_med")
CONTEXT_LENGTH      = 3
THRESHOLD           = 30
PRIMITIVES_FIRST    = 200
DISTILL_N_SENTENCES = 200
SEED                = 13
MAX_DEPTH           = 3     # root + three descent levels (four rows total)
BL_ALPHA            = 10.0  # eval_alpha passed to get_basic(use_root=True)
OUT_DIR = os.path.join(_HERE, "paper", "graphics")
os.makedirs(OUT_DIR, exist_ok=True)


# ---- Parse helpers (vendored from grammar_distillation_test) -----------
def _walk_composites(node):
    if isinstance(node, PrimitiveParseNode):
        return
    if not getattr(node, "is_global_root", False):
        yield node
    for _, c in getattr(node, "children", []):
        yield from _walk_composites(c)


def _chunk_span(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            out.append(int(n.position_idx)); return
        for _, c in getattr(n, "children", []):
            w(c)
    w(node)
    if not out: return None, None
    return min(out), max(out)


def _chunk_yield(node, itv):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is None or wid < 0 or wid >= len(itv): return
            pos = WORD_TO_POS.get(itv[wid])
            if pos: out.append(pos)
            return
        for _, c in sorted(getattr(n, "children", []),
                           key=lambda x: x[0] if x[0] is not None else 0):
            w(c)
    w(node)
    return out


_OTHER_LOG = []  # diagnostic: capture every OTHER fall-through

def classify_chunk(node, sentence_len, itv, sent_str=""):
    """Classify a composite by the POS sequence of its terminal yield, using
    the new MED grammar's intermediate non-terminals.

    Grammar (see Appendix A):
        S      -> NP VP                     (only at the sentence root)
        NP     -> Det N | Det AdjP
        AdjP   -> Adj N | Adj AdjP
        VP     -> V NP | V VPobj
        VPobj  -> NP PP
        PP     -> P NP

    S exists ONLY as the full-sentence root: any non-full-span chunk that
    happens to yield Det+...+V is an intermediate binary merge of the
    parser, not a constituent the grammar admits.
    """
    pos_seq = _chunk_yield(node, itv)
    if not pos_seq: return None
    if len(pos_seq) == 1: return pos_seq[0]
    s, e = _chunk_span(node)

    if s == 0 and e == sentence_len - 1: return "S"

    starts = pos_seq[0]
    has_v = "V" in pos_seq
    has_p = "P" in pos_seq

    if starts == "Det" and not has_v:
        if has_p:    return "VPobj"
        else:        return "NP"
    if starts == "V": return "VP"
    if starts == "P": return "PP"
    if starts == "Adj" and "N" in pos_seq: return "AdjP"
    if starts == "N": return "AdjP"
    # Diagnostic: record what kinds of chunks fall through.
    tokens = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is None or wid < 0 or wid >= len(itv): return
            tokens.append(itv[wid])
            return
        for _, c in sorted(getattr(n, "children", []),
                           key=lambda x: x[0] if x[0] is not None else 0):
            w(c)
    w(node)
    _OTHER_LOG.append({"pos": "+".join(pos_seq), "tokens": " ".join(tokens),
                        "sent": sent_str, "span_len": e - s + 1})
    return "OTHER"


# ---- Tree layout + bar plotting (vendored) -----------------------------
def _make_layout(root, max_depth):
    all_nodes = [root]; children_of = {0: []}; depth_of = {0: 0}
    hash_to_idx = {str(root.concept_hash()): 0}
    queue = [0]
    while queue:
        idx = queue.pop(0); node = all_nodes[idx]
        if depth_of[idx] < max_depth:
            for c in node.children:
                ci = len(all_nodes); all_nodes.append(c)
                children_of[idx].append(ci); children_of[ci] = []
                depth_of[ci] = depth_of[idx] + 1
                hash_to_idx[str(c.concept_hash())] = ci
                queue.append(ci)
    return all_nodes, children_of, depth_of, hash_to_idx


def compute_basic_level_hashes(root, eval_alpha):
    """Walk all leaves of the FULL tree and collect the hash of each
    leaf's basic-level ancestor under the cobweb-private definition
    (``get_basic(..., eval_alpha=alpha, use_root=True)``). Leaves whose
    basic-level ancestor IS the leaf itself are skipped."""
    leaves = []
    stack = [root]
    while stack:
        n = stack.pop()
        if not n.children:
            leaves.append(n)
        else:
            stack.extend(n.children)
    bl_hashes = set()
    root_hash = str(root.concept_hash())
    for leaf in leaves:
        try:
            bl = leaf.get_basic(1000, 1000, eval_alpha=eval_alpha,
                                use_root=True)
        except Exception:
            continue
        h = str(bl.concept_hash())
        if h == str(leaf.concept_hash()) or h == root_hash:
            continue
        bl_hashes.add(h)
    return bl_hashes


def _layout_x(children_of, max_depth):
    def _leaf_span(idx, depth):
        if depth >= max_depth or not children_of.get(idx): return 1
        return sum(_leaf_span(c, depth + 1) for c in children_of[idx])
    pos = {}
    def _assign(idx, depth, x_left):
        span = _leaf_span(idx, depth)
        pos[idx] = (x_left + span / 2.0, depth)
        if depth < max_depth and children_of.get(idx):
            cur = x_left
            for c in children_of[idx]:
                cs = _leaf_span(c, depth + 1)
                _assign(c, depth + 1, cur); cur += cs
    _assign(0, 0, 0.0)
    return pos, _leaf_span(0, 0)


def _prune_empty(children_of, has_data_fn):
    @_lru_cache(maxsize=None)
    def _alive(idx):
        if has_data_fn(idx): return True
        return any(_alive(c) for c in children_of.get(idx, []))
    new = {}
    for idx in children_of:
        if idx == 0 or _alive(idx):
            new[idx] = [c for c in children_of[idx] if _alive(c)]
    return new


def _draw_external_legend(fig, label_list, color_map, used, legend_title):
    """Horizontal legend strip below the plot, OUTSIDE the axes so it
    never overlaps a bottom-row node."""
    legend_h = [plt.Rectangle((0, 0), 1, 1, color=color_map[lbl], label=lbl)
                for i, lbl in enumerate(label_list) if used[i] > 0]
    fig.legend(handles=legend_h, title=legend_title,
               loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=len(legend_h),
               fontsize=10, title_fontsize=10,
               frameon=True)


def plot_tree_single_bars(children_of, counts, label_list, color_map,
                          title, out_path, max_depth,
                          highlight_idx=None):
    highlight_idx = set() if highlight_idx is None else highlight_idx
    children_of = _prune_empty(
        children_of, lambda i: i in counts and counts[i].sum() > 0)
    pos_map, total_w = _layout_x(children_of, max_depth)
    # Distillation-script compact sizing: thin bars, wide trees.
    bar_w, bar_h, y_gap = 0.7, 0.30, 1.0
    fig, ax = plt.subplots(
        figsize=(max(14, total_w * 0.85), (max_depth + 1) * 1.4))
    ax.set_xlim(0, total_w); ax.set_ylim(-0.45, max_depth * y_gap + 0.55)
    ax.invert_yaxis(); ax.axis("off")  # title removed (info in caption)

    def _has(idx): return idx in counts and counts[idx].sum() > 0
    def _edges(idx, depth):
        if depth >= max_depth or not children_of[idx] or not _has(idx): return
        px, py = pos_map[idx]
        for c in children_of[idx]:
            if not _has(c): continue
            cx, cy = pos_map[c]
            ax.plot([px, cx],
                    [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.8, zorder=0)
            _edges(c, depth + 1)
    _edges(0, 0)
    def _draw(idx, depth):
        if not _has(idx): return
        cnts = counts[idx].astype(float); total = cnts.sum()
        props = cnts / total
        x_c, _ = pos_map[idx]
        x_left = x_c - bar_w / 2; y_top = depth * y_gap - bar_h / 2
        cur = x_left
        for ci, lbl in enumerate(label_list):
            seg = props[ci] * bar_w
            if seg > 0:
                ax.add_patch(plt.Rectangle((cur, y_top), seg, bar_h,
                                           color=color_map[lbl], lw=0))
                cur += seg
        is_bl = idx in highlight_idx
        ax.add_patch(plt.Rectangle(
            (x_left, y_top), bar_w, bar_h, fill=False,
            edgecolor=("red" if is_bl else "black"),
            lw=(2.5 if is_bl else 0.5),
            zorder=(5 if is_bl else 2)))
        ax.text(x_c, depth * y_gap + bar_h / 2 + 0.06,
                f"n={int(total)}", ha="center", va="top", fontsize=7)
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]: _draw(c, depth + 1)
    _draw(0, 0)
    used = np.zeros(len(label_list), dtype=np.int64)
    for c in counts.values(): used += c
    _draw_external_legend(fig, label_list, color_map, used,
                          legend_title="class (red border = basic level)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white"); plt.close()
    print(f"wrote {out_path}")


def plot_tree_pair_bars(children_of, cL, cR, label_list, color_map,
                        title, out_path, max_depth,
                        highlight_idx=None):
    highlight_idx = set() if highlight_idx is None else highlight_idx
    children_of = _prune_empty(
        children_of, lambda i: i in cL and cL[i].sum() > 0)
    pos_map, total_w = _layout_x(children_of, max_depth)
    bar_w, bar_h, gap, y_unit = 0.7, 0.16, 0.05, 1.0
    fig, ax = plt.subplots(
        figsize=(max(14, total_w * 0.85), (max_depth + 1) * 1.6))
    ax.set_xlim(0, total_w); ax.set_ylim(-0.45, max_depth * y_unit + 0.55)
    ax.invert_yaxis(); ax.axis("off")  # title removed (info in caption)

    def _has(idx): return idx in cL and cL[idx].sum() > 0
    def _edges(idx, depth):
        if depth >= max_depth or not children_of[idx] or not _has(idx): return
        px, py = pos_map[idx]
        for c in children_of[idx]:
            if not _has(c): continue
            cx, cy = pos_map[c]
            y_par = py * y_unit + bar_h + gap / 2
            y_chi = cy * y_unit - bar_h - gap / 2
            ax.plot([px, cx], [y_par, y_chi],
                    color="gray", lw=0.8, zorder=0)
            _edges(c, depth + 1)
    _edges(0, 0)
    def _bar(x_left, y_top, props, txt, is_bl):
        cur = x_left
        for i, lbl in enumerate(label_list):
            seg = props[i] * bar_w
            if seg > 0:
                ax.add_patch(plt.Rectangle((cur, y_top), seg, bar_h,
                                           color=color_map[lbl], lw=0))
                cur += seg
        ax.add_patch(plt.Rectangle(
            (x_left, y_top), bar_w, bar_h, fill=False,
            edgecolor=("red" if is_bl else "black"),
            lw=(2.5 if is_bl else 0.5),
            zorder=(5 if is_bl else 2)))
        ax.text(x_left - 0.04, y_top + bar_h / 2, txt,
                ha="right", va="center", fontsize=6)
    def _draw(idx, depth):
        if not _has(idx): return
        cL_a = cL[idx].astype(float); cR_a = cR[idx].astype(float)
        total = cL_a.sum()
        propsL = cL_a / total; propsR = cR_a / total
        is_bl = idx in highlight_idx
        x_c, _ = pos_map[idx]; x_left = x_c - bar_w / 2
        y_top_L = depth * y_unit - bar_h - gap / 2
        y_top_R = depth * y_unit + gap / 2
        _bar(x_left, y_top_L, propsL, "L", is_bl)
        _bar(x_left, y_top_R, propsR, "R", is_bl)
        ax.text(x_c, y_top_L - 0.04, f"n={int(total)}",
                ha="center", va="bottom", fontsize=7)
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]: _draw(c, depth + 1)
    _draw(0, 0)
    used = np.zeros(len(label_list), dtype=np.int64)
    for c in cL.values(): used += c
    for c in cR.values(): used += c
    _draw_external_legend(fig, label_list, color_map, used,
                          legend_title="class (top=L, bottom=R; "
                                       "red border = basic level)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white"); plt.close()
    print(f"wrote {out_path}")


# ---- Counting helpers -------------------------------------------------
def _build_primitive_instances(sentences, n_max=400):
    out = []
    for sent in sentences:
        toks = sent.split()
        for i, w in enumerate(toks):
            if w in WORD_TO_POS:
                out.append((toks, i, w, WORD_TO_POS[w]))
        if len(out) >= n_max: break
    return out


def _build_ctx_inst_for_word(trellis, toks, i):
    ltm = trellis.ltm
    cl  = ltm.context_length
    wt_mode = getattr(ltm, "weighting", "binary")
    emp_wt  = getattr(ltm, "empty_weighting", False)
    wid = ltm.value_to_id.get(toks[i], 0)
    inst = {}
    for j in range(cl):
        s = i - (j + 1)
        if 0 <= s < len(toks):
            cw = ltm.value_to_id.get(toks[s], 0)
            inst[j] = {cw: _context_weight(j, wt_mode), 0: 0}
        else:
            inst[j] = {0: _context_weight(j, wt_mode) if emp_wt else 0}
    for j in range(cl):
        s = i + (j + 1)
        if 0 <= s < len(toks):
            cw = ltm.value_to_id.get(toks[s], 0)
            inst[cl + j] = {cw: _context_weight(j, wt_mode), 0: 0}
        else:
            inst[cl + j] = {0: _context_weight(j, wt_mode) if emp_wt else 0}
    inst[-2] = {_get_or_register_cplx_vid(1, ltm.id_to_value, ltm.value_to_id): 1}
    inst[ltm.content_ref_attr] = {wid: 1}
    return inst


def _build_gold_tree(trellis, sentence, merges):
    """Build a parse tree from gold merges (no parser decisions involved).
    Returns the constructed FiniteParseTree or None on failure."""
    tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
    try:
        tree.build_primitives(sentence, threshold=THRESHOLD)
        for m in merges:
            try: tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        return tree
    except Exception:
        return None


def _collect_ctx_descents(trellis, prim_instances, hollow_test):
    """Walk GOLD parse trees only — never the parser's own eval output.
    Memory contains only canonical chunks because training is supervised;
    visualizing parser-time decisions would re-introduce non-canonical
    composites the parser commits at eval time. Using gold trees ensures
    every composite we visualize is a grammar-licensed constituent."""
    itv = trellis.ltm.id_to_value
    cref_attr = trellis.ltm.content_ref_attr
    pairs = []
    for sent_toks, i, w, pos in prim_instances:
        inst = _build_ctx_inst_for_word(trellis, sent_toks, i)
        pairs.append((inst, _label2id[pos]))
    for h in hollow_test:
        tree = _build_gold_tree(trellis, h["sentence"], h["merges"])
        if tree is None: continue
        sent_len = len(h["sentence"].split())
        for comp in _walk_composites(tree.global_root_node):
            ctx_inst = comp.get_context_instance()
            if not ctx_inst: continue
            ctx_inst = dict(ctx_inst)
            ctx_inst.pop(cref_attr, None)
            cls = classify_chunk(comp, sent_len, itv) or "OTHER"
            pairs.append((ctx_inst,
                          _label2id.get(cls, _label2id["OTHER"])))
    return pairs


def compute_ctx_node_counts(root, ctx_descents, max_depth):
    all_nodes, children_of, _, hash_to_idx = _make_layout(root, max_depth)
    counts = {}
    for inst, label_idx in ctx_descents:
        cur = 0
        for _ in range(max_depth + 1):
            counts.setdefault(cur, np.zeros(N_LABEL, dtype=np.int64))
            counts[cur][label_idx] += 1
            ch = children_of.get(cur, [])
            if not ch: break
            cur = max(ch, key=lambda i: all_nodes[i].log_prob_instance(inst))
    return all_nodes, children_of, counts, hash_to_idx


def compute_cnt_node_counts(trellis, root, hollow_test, max_depth):
    """Walk GOLD parse trees (not parser eval output) so every visualized
    composite is grammar-canonical."""
    itv = trellis.ltm.id_to_value
    all_nodes, children_of, _, hash_to_idx = _make_layout(root, max_depth)
    cnt_L = {}; cnt_R = {}
    for h in hollow_test:
        tree = _build_gold_tree(trellis, h["sentence"], h["merges"])
        if tree is None: continue
        sent_len = len(h["sentence"].split())
        for comp in _walk_composites(tree.global_root_node):
            ci = comp.get_content_instance()
            if not ci: continue
            kids = sorted(comp.children, key=lambda x: x[0])
            if len(kids) != 2: continue
            L_cls = classify_chunk(kids[0][1], sent_len, itv) or "OTHER"
            R_cls = classify_chunk(kids[1][1], sent_len, itv) or "OTHER"
            li = _label2id.get(L_cls, _label2id["OTHER"])
            ri = _label2id.get(R_cls, _label2id["OTHER"])
            cur = 0
            for _ in range(max_depth + 1):
                cnt_L.setdefault(cur, np.zeros(N_LABEL, dtype=np.int64))
                cnt_R.setdefault(cur, np.zeros(N_LABEL, dtype=np.int64))
                cnt_L[cur][li] += 1; cnt_R[cur][ri] += 1
                ch = children_of.get(cur, [])
                if not ch: break
                cur = max(ch, key=lambda i: all_nodes[i].log_prob_instance(ci))
    return all_nodes, children_of, cnt_L, cnt_R, hash_to_idx


# ---- Training (mirrors grammar_distillation_test.py phase 0) ----------
def train_trellis():
    """Phase 0a: 200 primitive-only TEST_GRAMMAR1 sentences with a
    massive threshold so no composites form. Phase 0b: hollow-corpus
    replay using the hand-annotated parses in
    ``data/test_hollow_grammar_1``. Returns the trained TRELLIS."""
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    print("Phase 0a: instantiating TRELLIS")
    trellis = TRELLIS(
        TEST_CORPUS1,
        context_length=CONTEXT_LENGTH,
        threshold=THRESHOLD,
        content_alpha=1e-4,
        context_alpha=1e-4,
        content_bl_alpha=10,
        context_bl_alpha=10,
        bow=False,
        empty_weighting=True,
        chunk_context=False,
        weighting="binary",
        categorization_mode="dfs",
        depth_max_content=1000,
        depth_max_context=1000,
        branch_max_content=1000,
        branch_max_context=1000,
        content_top_k=7,
        content_pool_depth=4,
    )
    print(f"Phase 0a: {PRIMITIVES_FIRST} primitive-only sentences")
    for i in range(PRIMITIVES_FIRST):
        s = generate("S", TEST_GRAMMAR1)
        trellis.parse_sentence(s, threshold=1e9, new_vocab=True,
                               learning=True, debug=False)
        if (i + 1) % 50 == 0: print(f"  primitives [{i+1}/{PRIMITIVES_FIRST}]")

    print(f"Phase 0b: hollow corpus replay from {HOLLOW_CORPUS_DIR}")
    hollow = []
    if os.path.isdir(HOLLOW_CORPUS_DIR):
        for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
            try:
                with open(p) as f: data = json.load(f)
            except Exception: continue
            if "sentence" in data and "merges" in data:
                hollow.append(data)
    if not hollow:
        # No cached corpus available — generate one inline from the
        # current TEST_GRAMMAR_MED productions so the script is
        # self-contained.
        from util.cfg import generate_with_merges
        print(f"  no cached corpus found; generating 300 sentences inline")
        seen = set()
        while len(hollow) < 300:
            text, merges = generate_with_merges(
                "S", TEST_GRAMMAR1, flatten_at_parent=())
            text = text.strip()
            if not text or text in seen: continue
            seen.add(text)
            hollow.append({"sentence": text, "merges": merges})
    print(f"  loaded {len(hollow)} hollow trees")
    random.shuffle(hollow)
    split = int(0.8 * len(hollow))
    train_h = hollow[:split]
    for i, h in enumerate(train_h):
        tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
        tree.build_primitives(h["sentence"], threshold=THRESHOLD)
        for m in h["merges"]:
            try: tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        trellis.ltm.add_parse_tree(tree, shuffle=True, debug=False)
        if (i + 1) % 25 == 0:
            print(f"  hollow [{i+1}/{len(train_h)}]")
    return trellis


# ---- Driver -----------------------------------------------------------
def main():
    trellis = train_trellis()

    print(f"generating {DISTILL_N_SENTENCES} distill hollow trees")
    random.seed(SEED + 1)
    from util.cfg import generate_with_merges
    distill_hollow = []
    seen_dist = set()
    while len(distill_hollow) < DISTILL_N_SENTENCES:
        text, merges = generate_with_merges(
            "S", TEST_GRAMMAR1, flatten_at_parent=())
        text = text.strip()
        if not text or text in seen_dist: continue
        seen_dist.add(text)
        distill_hollow.append({"sentence": text, "merges": merges})
    sentences = [h["sentence"] for h in distill_hollow]

    print("computing per-node distributions ...")
    primtest = _build_primitive_instances(sentences, n_max=400)
    # _collect_ctx_descents walks the GOLD trees (constructed from the
    # hollow trees' merges) instead of the parser's own eval-time parses,
    # so every visualized composite is a grammar-canonical constituent.
    ctx_descents = _collect_ctx_descents(trellis, primtest, distill_hollow)
    _, ctx_children, ctx_counts, ctx_h2i = compute_ctx_node_counts(
        trellis.ltm.context_hierarchy.root,
        ctx_descents,
        max_depth=MAX_DEPTH,
    )
    _, cnt_children, cnt_L, cnt_R, cnt_h2i = compute_cnt_node_counts(
        trellis,
        trellis.ltm.content_hierarchy.root,
        distill_hollow,
        max_depth=MAX_DEPTH,
    )

    print(f"locating basic-level nodes (eval_alpha={BL_ALPHA}) ...")
    ctx_bl_hashes = compute_basic_level_hashes(
        trellis.ltm.context_hierarchy.root, eval_alpha=BL_ALPHA)
    cnt_bl_hashes = compute_basic_level_hashes(
        trellis.ltm.content_hierarchy.root, eval_alpha=BL_ALPHA)
    ctx_bl_idx = {ctx_h2i[h] for h in ctx_bl_hashes if h in ctx_h2i}
    cnt_bl_idx = {cnt_h2i[h] for h in cnt_bl_hashes if h in cnt_h2i}
    print(f"  context BL nodes in view: {len(ctx_bl_idx)}/{len(ctx_bl_hashes)}")
    print(f"  content BL nodes in view: {len(cnt_bl_idx)}/{len(cnt_bl_hashes)}")

    plot_tree_single_bars(
        ctx_children, ctx_counts, ALL_LABELS, LABEL_COLOR,
        title="Context hierarchy --- POS + chunk-class distributions "
              "(first four levels)",
        out_path=os.path.join(OUT_DIR, "hierarchy_bars_context.png"),
        max_depth=MAX_DEPTH,
        highlight_idx=ctx_bl_idx,
    )
    plot_tree_pair_bars(
        cnt_children, cnt_L, cnt_R, ALL_LABELS, LABEL_COLOR,
        title="Content hierarchy --- left/right child class distributions "
              "(first four levels)",
        out_path=os.path.join(OUT_DIR, "hierarchy_bars_content.png"),
        max_depth=MAX_DEPTH,
        highlight_idx=cnt_bl_idx,
    )


if __name__ == "__main__":
    main()
