"""
WEBSTER Bag-Packing / Bag-Unpacking decoding test
=================================================

Trains WEBSTER on the hollow corpus (mirroring ``unittests/hollow_learn_test_mh.py``
exactly so both scripts share the same model), then evaluates both
hierarchies with the same visual idiom as ``tests/basic-level/
grammar_basic_level_test.py`` and ``grammar_chunking_basic_level.py``.

Sections
--------
Phase 1. CONTEXT-tree inspection — primitive POS distributions.
  Every primitive token in the training set greedy-descends to its
  context-tree leaf, then ``get_basic(use_root=True,
  eval_alpha=EVAL_ALPHA)`` lifts it to a basic-level (BL) node. Per-BL
  visualisation: POS histogram, top center words, top context words at
  each offset. Tree-with-POS-bars (red borders on BL nodes). Mean
  empirical-PMI by depth.

Phase 2. CONTENT-tree inspection — chunk class distributions.
  Every supervised hollow chunk greedy-descends through the content
  tree, then ``get_basic(use_root=True)`` lifts it to a BL node. The
  chunk is classified by the **head rule** (so VP shows up for any
  V-containing chunk, including left-binarized ``NP + V`` merges):

      S      — the chunk's span equals the WHOLE sentence
      VP     — chunk contains a V anywhere (predicate head)
      PP     — chunk starts with P, no V
      AdjP   — chunk yield is Adj+ or Adj+ N
      NP     — chunk has Det or N, no V or leading P

  Per-BL visualisation: pair-class joint matrix (L child vs R child),
  top center bigrams, top per-side attr values. Tree-with-L/R-bars
  (red borders on BL nodes). Score-by-depth.

Phase 3. BAG-PACKING / UNPACKING decoding (generalization).
  *Bag-packing* means the encoder turns each chunk into the
  TopK-Pool bag the content tree stores (and turns each primitive into
  its context-tree categorization). *Bag-unpacking* is the reverse:
  given the stored bag, can the tree reproduce the original surface
  tokens?  We run it two ways:

    A) Primitive recovery — for every held-out token we build its
       surrounding-context instance, categorize in the context tree,
       and read the leaf's content-ref distribution to predict the
       central word.  Scores per-POS recovery accuracy.

    B) Chunk recovery — for every held-out phrasal span (NP / VP /
       AdjP / PP from a freshly generated sentence's gold derivation
       tree) we mask the span and call
       ``webster.generate_sentence(masked_sentence=…)``.  Scores
       exact-token, length, first-token-POS, full POS-sequence match.

Outputs (``tests/met5/grammar_decoding_output/``)
------------------------------------------------
  context_tree/
    basic_level_subtrees.png
    cobweb_tree_labels.png
    per_subtree_membership.csv
    method_summary.txt
    score_by_depth.png
  content_tree/
    basic_level_subtrees.png
    content_tree_labels.png
    per_subtree_membership.csv
    method_summary.txt
    score_by_depth.png
  decoding/
    primitive_recovery.csv           — per-POS accuracy
    primitive_recovery.png           — bar chart
    chunk_recovery.csv               — per-case detail
    chunk_summary.csv                — per-phrase aggregate
    chunk_recovery.png               — per-phrase bar chart
"""

import os
import sys
import csv
import glob
import json
import random
import shutil
from collections import Counter, defaultdict
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import (WEBSTER, FiniteParseTree, PrimitiveParseNode,
                      _get_or_register_cplx_vid, _context_weight)
from cobweb.cobweb_discrete import CobwebDiscreteTree

# ── Configuration ─────────────────────────────────────────────────────────────
OUT_DIR             = os.path.join(_HERE, "grammar_decoding_output")
HOLLOW_CORPUS_DIR   = "data/test_hollow_grammar_1"
CONTEXT_LENGTH      = 3
THRESHOLD           = 30
PRIMITIVES_FIRST    = 200
EVAL_ALPHA          = 10.0
PROBE_ALPHA         = 1e-3        # smoothing for the discrete-probe classifier
TREE_DEPTH_FIG      = 3
TOP_WORDS_PER_OFFSET = 3
TOP_CENTER_WORDS    = 6
TOP_CTX_NODES       = 5
SEED                = 13
random.seed(SEED)
np.random.seed(SEED)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
CONTEXT_DIR  = os.path.join(OUT_DIR, "context_tree");  os.makedirs(CONTEXT_DIR, exist_ok=True)
CONTENT_DIR  = os.path.join(OUT_DIR, "content_tree");  os.makedirs(CONTENT_DIR, exist_ok=True)
DECODING_DIR = os.path.join(OUT_DIR, "decoding");       os.makedirs(DECODING_DIR, exist_ok=True)

# ── Word → POS ────────────────────────────────────────────────────────────────
POS_LIST = ["Det", "N", "Adj", "V", "P"]
WORD_TO_POS: dict = {}
for pos in POS_LIST:
    for prod in TEST_GRAMMAR1[pos]:
        for w in prod:
            WORD_TO_POS[w] = pos
print(f"Vocabulary: {len(WORD_TO_POS)} words across {len(POS_LIST)} POS classes")

# ── PHASE 0: TRAIN WEBSTER (mirror hollow_learn_test_mh.py) ──────────────────
print("\n=== PHASE 0: Train WEBSTER (mirroring hollow_learn_test_mh.py) ===")
webster = WEBSTER(
    TEST_CORPUS1,
    context_length=CONTEXT_LENGTH,
    threshold=THRESHOLD,
    content_alpha=1e-3,
    context_alpha=1e-3,
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
    content_top_k=5,
    content_pool_depth=3,
)

print(f"  Phase 0a: {PRIMITIVES_FIRST} primitive-only sentences")
training_sentences = []   # remember for primitive recovery
for i in range(PRIMITIVES_FIRST):
    s = generate("S", TEST_GRAMMAR1)
    training_sentences.append(s)
    webster.parse_sentence(s, threshold=1e9, new_vocab=True,
                           learning=True, debug=False)
    if (i + 1) % 50 == 0:
        print(f"    [{i+1}/{PRIMITIVES_FIRST}]")

print(f"  Phase 0b: hollow corpus replay")
hollow_corpus_all: list = []
for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
    with open(p, "r", encoding="utf-8") as f:
        try:    data = json.load(f)
        except json.JSONDecodeError: continue
    if "sentence" in data and "merges" in data:
        hollow_corpus_all.append(data)
print(f"    Loaded {len(hollow_corpus_all)} hollow trees")

# 80/20 split at the sentence level. Train chunks feed WEBSTER and
# Phase 1/2 inspection; test chunks are held out for Phase 3's
# representation-quality probe.
random.shuffle(hollow_corpus_all)
_split_idx = int(0.8 * len(hollow_corpus_all))
hollow_corpus  = hollow_corpus_all[:_split_idx]   # train (inspected, fitted)
hollow_test    = hollow_corpus_all[_split_idx:]   # held-out (probed)
print(f"    Split: train={len(hollow_corpus)}  test={len(hollow_test)}")

for i, hollow in enumerate(hollow_corpus):
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(hollow["sentence"], threshold=THRESHOLD)
    for m in hollow["merges"]:
        try: tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)
    if (i + 1) % 25 == 0:
        print(f"    [{i+1}/{len(hollow_corpus)}]")

# ── Shared helpers ────────────────────────────────────────────────────────────
def _chunk_yield(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is None or wid < 0 or wid >= len(webster.ltm.id_to_value):
                return
            pos = WORD_TO_POS.get(webster.ltm.id_to_value[wid])
            if pos: out.append(pos)
            return
        for _, c in sorted(getattr(n, "children", []),
                           key=lambda x: x[0] if x[0] is not None else 0):
            w(c)
    w(node)
    return out

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

def _chunk_tokens(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is not None and 0 <= wid < len(webster.ltm.id_to_value):
                out.append((int(n.position_idx), webster.ltm.id_to_value[wid]))
            return
        for _, c in getattr(n, "children", []):
            w(c)
    w(node)
    out.sort()
    return [t for _, t in out]

def _walk_composites(node):
    if isinstance(node, PrimitiveParseNode): return
    if not getattr(node, "is_global_root", False):
        yield node
    for _, c in getattr(node, "children", []):
        yield from _walk_composites(c)

def classify_chunk(node, sentence_len):
    """Head-based chunk classification, S only for the root chunk."""
    pos_seq = _chunk_yield(node)
    if not pos_seq: return None
    if len(pos_seq) == 1: return pos_seq[0]
    s, e = _chunk_span(node)
    if s == 0 and e == sentence_len - 1: return "S"
    if "V" in pos_seq:                                  return "VP"
    if pos_seq[0] == "P":                               return "PP"
    if all(p == "Adj" for p in pos_seq):                return "AdjP"
    if pos_seq[0] == "Adj" and "N" in pos_seq:          return "AdjP"
    if "N" in pos_seq or pos_seq[0] == "Det":           return "NP"
    return "OTHER"

def greedy_descend(root, instance):
    n = root
    while n.children:
        n = max(n.children, key=lambda c: c.log_prob_instance(instance))
    return n

def get_basic_cached(leaf, cache):
    key = id(leaf)
    if key in cache: return cache[key]
    bl = leaf.get_basic(0, 0, debug=False,
                        eval_alpha=EVAL_ALPHA, use_root=True)
    cache[key] = bl
    return bl

# ── Palette ───────────────────────────────────────────────────────────────────
PRIM_LABELS  = POS_LIST                       # Det, N, Adj, V, P
CHUNK_LABELS = ["NP", "AdjP", "PP", "VP", "S"]
ALL_LABELS   = PRIM_LABELS + CHUNK_LABELS + ["OTHER"]
LABEL_COLOR  = {
    "Det":   "#2ca02c",
    "N":     "#8c564b",
    "Adj":   "#1f77b4",
    "V":     "#17becf",
    "P":     "#7f7f7f",
    "NP":    "#ff7f0e",
    "AdjP":  "#9467bd",
    "PP":    "#bcbd22",
    "VP":    "#e377c2",
    "S":     "#d62728",
    "OTHER": "#cccccc",
}
N_PRIM      = len(PRIM_LABELS)
prim2id     = {p: i for i, p in enumerate(PRIM_LABELS)}
id2prim     = {i: p for p, i in prim2id.items()}
pos_colors  = [LABEL_COLOR[p] for p in PRIM_LABELS]

# =============================================================================
# PHASE 1 — CONTEXT TREE INSPECTION
# =============================================================================
# Every primitive token has a context_instance stored in WEBSTER. Greedy
# descend → leaf → get_basic. Per-BL POS histograms, top center & context
# words. Tree-with-POS-bars (red borders on BL nodes). Score by depth.
# =============================================================================
print("\n=== PHASE 1: CONTEXT TREE INSPECTION ===")

ctx_root = webster.ltm.context_hierarchy.root
ctx_offsets = list(range(CONTEXT_LENGTH))  # before slots
ctx_after_offsets = list(range(CONTEXT_LENGTH, 2 * CONTEXT_LENGTH))
ctx_attr_offsets = {j: -(j+1) for j in ctx_offsets}      # 0→-1, 1→-2, 2→-3
ctx_attr_offsets.update({CONTEXT_LENGTH + j: (j+1) for j in ctx_offsets})

def offset_for_attr(attr_id):
    return ctx_attr_offsets.get(attr_id, attr_id)

def _build_ctx_instance(toks, i):
    """Mirror parse_mh.build_primitives' context instance build."""
    ltm = webster.ltm
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

# Build a held-out pool of test primitives from fresh sentences.
print("  Generating fresh test sentences for primitive descent...")
test_primitives = []   # list of (sentence_tokens, position, central_word, POS)
for _ in range(150):
    s = generate("S", TEST_GRAMMAR1)
    toks = s.split()
    for i, w in enumerate(toks):
        if w in WORD_TO_POS:
            test_primitives.append((toks, i, w, WORD_TO_POS[w]))
print(f"  {len(test_primitives)} primitive test instances")

# Descend each, get_basic, accumulate per-BL.
ctx_bl_cache: dict = {}
ctx_bl_members: dict = {}
for sent_toks, i, w, pos in test_primitives:
    inst = _build_ctx_instance(sent_toks, i)
    leaf = greedy_descend(ctx_root, inst)
    bl   = get_basic_cached(leaf, ctx_bl_cache)
    if bl is None: continue
    h = str(bl.concept_hash())
    if h not in ctx_bl_members:
        ctx_bl_members[h] = {
            "node": bl, "depth": bl.depth(),
            "pos_labels": [], "center_words": [],
        }
    ctx_bl_members[h]["pos_labels"].append(prim2id[pos])
    ctx_bl_members[h]["center_words"].append(webster.ltm.value_to_id[w])
print(f"  {len(ctx_bl_members)} unique BL nodes in context tree")

# ── Per-BL viz (POS hist + top center + top context per offset) ─────────────
def _top_context_words(node, k=TOP_WORDS_PER_OFFSET):
    out = {}
    av = node.av_count or {}
    for attr_id in range(2 * CONTEXT_LENGTH):
        val_map = av.get(attr_id, {})
        if not val_map: continue
        items = sorted([(v, c) for v, c in val_map.items() if v != 0],
                       key=lambda kv: -kv[1])[:k]
        if not items: continue
        total = sum(c for _, c in items) or 1
        offset = offset_for_attr(attr_id)
        out[offset] = [(webster.ltm.id_to_value[v]
                          if 0 <= v < len(webster.ltm.id_to_value) else f"<{v}>",
                        c / total)
                       for v, c in items]
    return out

def plot_bl_subtrees_primitive(members, title, out_path):
    sorted_bls = sorted(members.values(),
                        key=lambda m: len(m["pos_labels"]), reverse=True)
    n_rows = len(sorted_bls)
    if n_rows == 0: return
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15, max(2.0, n_rows * 1.6)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.4, 2.5]},
    )
    fig.suptitle(title, fontsize=11)
    for row, m in enumerate(sorted_bls):
        node = m["node"]
        labels = np.array(m["pos_labels"])
        centers = np.array(m["center_words"])
        n_mem = len(m["pos_labels"]); depth = m["depth"]
        cnts = np.bincount(labels, minlength=N_PRIM)
        dom = id2prim[int(cnts.argmax())]

        ax0 = axes[row, 0]
        props = cnts / max(cnts.sum(), 1)
        ax0.bar(np.arange(N_PRIM), props, color=pos_colors,
                edgecolor="white", linewidth=0.4)
        ax0.set_xticks(range(N_PRIM))
        ax0.set_xticklabels(PRIM_LABELS, rotation=45, ha="right", fontsize=6)
        ax0.set_ylim(0, 1.0)
        ax0.tick_params(axis="y", labelsize=5)
        ax0.set_ylabel(
            f"d={depth}\nn={n_mem}\ncnt={int(node.count)}\ndom={dom}",
            fontsize=6, rotation=0, labelpad=28, va="center",
        )
        if row == 0: ax0.set_title("POS histogram", fontsize=8)

        ax1 = axes[row, 1]
        cw_counts = {}
        for c in centers: cw_counts[int(c)] = cw_counts.get(int(c), 0) + 1
        top_cw = sorted(cw_counts.items(), key=lambda kv: -kv[1])[:TOP_CENTER_WORDS]
        if top_cw:
            words = [webster.ltm.id_to_value[w] for w, _ in top_cw]
            counts_ = [c for _, c in top_cw]
            colors_ = [LABEL_COLOR.get(WORD_TO_POS.get(w_str, "OTHER"), "#999")
                       for w_str in words]
            ax1.barh(np.arange(len(words))[::-1], counts_,
                     color=colors_, edgecolor="white", linewidth=0.4)
            ax1.set_yticks(np.arange(len(words))[::-1])
            ax1.set_yticklabels(words, fontsize=6)
            ax1.tick_params(axis="x", labelsize=5)
        if row == 0: ax1.set_title("top center words", fontsize=8)

        ax2 = axes[row, 2]
        ax2.axis("off")
        ctx_top = _top_context_words(node, k=TOP_WORDS_PER_OFFSET)
        offsets = sorted(ctx_top.keys())
        if offsets:
            x_step = 1.0 / max(len(offsets), 1)
            for ci, off in enumerate(offsets):
                cx = (ci + 0.5) * x_step
                ax2.text(cx, 0.95, f"{off:+d}", ha="center", va="top",
                         fontsize=7, fontweight="bold",
                         transform=ax2.transAxes)
                for li, (w, frac) in enumerate(ctx_top[off]):
                    cy = 0.85 - li * 0.20
                    color = LABEL_COLOR.get(WORD_TO_POS.get(w, "OTHER"), "#444")
                    ax2.text(cx, cy, f"{w} ({frac:.2f})",
                             ha="center", va="top", fontsize=6,
                             color=color, transform=ax2.transAxes)
        if row == 0: ax2.set_title("top context word per offset", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

plot_bl_subtrees_primitive(
    ctx_bl_members,
    title=(f"CONTEXT TREE basic-level subtrees "
           f"— get_basic(use_root=True, eval_alpha={EVAL_ALPHA})  "
           f"(N_test={len(test_primitives)}, n_subtrees={len(ctx_bl_members)})"),
    out_path=os.path.join(CONTEXT_DIR, "basic_level_subtrees.png"),
)
print(f"  Context BL subtree fig → {CONTEXT_DIR}/basic_level_subtrees.png")

# CSV.
with open(os.path.join(CONTEXT_DIR, "per_subtree_membership.csv"), "w") as f:
    w_ = csv.writer(f)
    w_.writerow(["subtree_idx", "depth", "node_count", "test_members",
                 "dominant_pos", "pos_distribution"])
    for i, m in enumerate(sorted(ctx_bl_members.values(),
                                 key=lambda m: len(m["pos_labels"]), reverse=True)):
        cnts = np.bincount(np.array(m["pos_labels"]), minlength=N_PRIM)
        dom = id2prim[int(cnts.argmax())]
        dist = "/".join(f"{id2prim[k]}:{int(c)}" for k, c in enumerate(cnts) if c > 0)
        w_.writerow([i, m["depth"], int(m["node"].count),
                     len(m["pos_labels"]), dom, dist])

# Tree-with-POS-bars on context tree, BL nodes highlighted.
def _make_layout(root, max_depth):
    all_nodes = [root]; children_of = {0: []}; depth_of = {0: 0}
    queue = [0]
    while queue:
        idx = queue.pop(0); node = all_nodes[idx]
        if depth_of[idx] < max_depth:
            for c in node.children:
                ci = len(all_nodes)
                all_nodes.append(c)
                children_of[idx].append(ci)
                children_of[ci] = []
                depth_of[ci] = depth_of[idx] + 1
                queue.append(ci)
    return all_nodes, children_of, depth_of

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
                _assign(c, depth + 1, cur)
                cur += cs
    _assign(0, 0, 0.0)
    return pos, _leaf_span(0, 0)

def _prune_empty(children_of, has_data_fn):
    """Remove subtrees that have no data anywhere underneath them.

    A node is kept iff it has data itself OR any descendant does.
    Returns a new ``children_of`` dict; the BFS layout's indexing is
    preserved (we only drop edges, never renumber).
    """
    @lru_cache(maxsize=None)
    def _alive(idx):
        if has_data_fn(idx): return True
        return any(_alive(c) for c in children_of.get(idx, []))
    new = {}
    for idx in children_of:
        if idx == 0 or _alive(idx):
            new[idx] = [c for c in children_of[idx] if _alive(c)]
    return new

def compute_ctx_node_counts(root, test_primitives_data, max_depth):
    """Greedy descent — accumulate POS counts at every node visited."""
    all_nodes, children_of, _ = _make_layout(root, max_depth)
    counts = {}
    for sent_toks, i, w, pos in test_primitives_data:
        inst = _build_ctx_instance(sent_toks, i)
        cur = 0
        for _ in range(max_depth + 1):
            counts.setdefault(cur, np.zeros(N_PRIM, dtype=np.int64))
            counts[cur][prim2id[pos]] += 1
            ch = children_of.get(cur, [])
            if not ch: break
            cur = max(ch, key=lambda i: all_nodes[i].log_prob_instance(inst))
    return all_nodes, children_of, counts

def plot_tree_single_bars(children_of, counts, label_list, color_map,
                          highlight_idx, title, out_path, max_depth,
                          all_nodes=None):
    """Draw a single-bar-per-node tree.  All BFS-discovered Cobweb
    children are kept — none are pruned for empty descent tallies —
    so the layout faithfully reflects Cobweb's actual structure.
    Children with no descended chunks render as a small grey
    "n=cobweb_count" placeholder so the user sees every real branch.
    """
    pos_map, total_w = _layout_x(children_of, max_depth)
    bar_w, bar_h, y_gap = 0.7, 0.35, 1.0
    fig, ax = plt.subplots(
        figsize=(max(14, total_w * 0.9), (max_depth + 1) * 2.2)
    )
    ax.set_xlim(0, total_w); ax.set_ylim(-0.7, max_depth * y_gap + 0.7)
    ax.invert_yaxis(); ax.axis("off")
    ax.set_title(title, fontsize=11)

    def _has(idx):
        return idx in counts and counts[idx].sum() > 0

    def _cobweb_n(idx):
        if all_nodes and 0 <= idx < len(all_nodes):
            return int(getattr(all_nodes[idx], "count", 0))
        return 0

    def _edges(idx, depth):
        if depth >= max_depth or not children_of[idx]: return
        px, py = pos_map[idx]
        for c in children_of[idx]:
            cx, cy = pos_map[c]
            ax.plot([px, cx],
                    [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.7, zorder=0)
            _edges(c, depth + 1)
    _edges(0, 0)

    def _draw(idx, depth):
        x_c, _ = pos_map[idx]
        x_left = x_c - bar_w / 2; y_top = depth * y_gap - bar_h / 2
        if _has(idx):
            cnts = counts[idx].astype(float); total = cnts.sum()
            props = cnts / total
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
                lw=(3.0 if is_bl else 0.4),
                zorder=(5 if is_bl else 2)))
            ax.text(x_c, depth * y_gap + bar_h / 2 + 0.05,
                    f"n={int(total)}", ha="center", va="top", fontsize=5)
        else:
            # Cobweb-real child with no descended chunks.  Show its
            # Cobweb-stored count in a grey placeholder.
            cw_n = _cobweb_n(idx)
            ax.add_patch(plt.Rectangle(
                (x_left, y_top), bar_w, bar_h,
                facecolor="#f0f0f0", edgecolor="gray",
                lw=0.4, zorder=2))
            ax.text(x_c, depth * y_gap, f"cobweb_n={cw_n}",
                    ha="center", va="center", fontsize=5, color="gray",
                    style="italic")
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]: _draw(c, depth + 1)
    _draw(0, 0)

    used = np.zeros(len(label_list), dtype=np.int64)
    for c in counts.values(): used += c
    legend_h = [plt.Rectangle((0, 0), 1, 1, color=color_map[lbl], label=lbl)
                for i, lbl in enumerate(label_list) if used[i] > 0]
    ax.legend(handles=legend_h, title="label", loc="lower right",
              ncol=max(1, len(legend_h) // 4), fontsize=7, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

ctx_layout_nodes, ctx_layout_children, ctx_counts = compute_ctx_node_counts(
    ctx_root, test_primitives, max_depth=TREE_DEPTH_FIG)
ctx_bl_hashes = {str(m["node"].concept_hash()) for m in ctx_bl_members.values()}
ctx_highlight_idx = {i for i, n in enumerate(ctx_layout_nodes)
                     if str(n.concept_hash()) in ctx_bl_hashes}
plot_tree_single_bars(
    ctx_layout_children, ctx_counts, PRIM_LABELS, LABEL_COLOR,
    ctx_highlight_idx,
    title=(f"CONTEXT tree — POS Distributions  "
           f"(red border = BL, eval_alpha={EVAL_ALPHA})"),
    out_path=os.path.join(CONTEXT_DIR, "cobweb_tree_labels.png"),
    max_depth=TREE_DEPTH_FIG,
    all_nodes=ctx_layout_nodes,
)
print(f"  Context tree fig → {CONTEXT_DIR}/cobweb_tree_labels.png")

# Score by depth for context tree.
def _all_nodes(root):
    out = []; stack = [root]
    while stack:
        n = stack.pop(); out.append(n); stack.extend(n.children)
    return out

def _plot_score_by_depth(root, bl_members, out_path, tree_label):
    all_nodes = _all_nodes(root)
    d2s: dict = {}
    for n in all_nodes:
        d = n.depth()
        s = n.expected_pmi(0, 0, eval_alpha=EVAL_ALPHA,
                           uniform_leaf=False, use_root=True)
        d2s.setdefault(d, []).append(s)
    d2n_bl: dict = {}
    for m in bl_members.values():
        d2n_bl[m["depth"]] = d2n_bl.get(m["depth"], 0) + 1
    depths = sorted(d2s.keys())
    means = [np.mean(d2s[d]) for d in depths]
    mins  = [np.min(d2s[d]) for d in depths]
    maxs  = [np.max(d2s[d]) for d in depths]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(depths, mins, maxs, alpha=0.15, color="#1f77b4",
                    label="min–max range")
    ax.plot(depths, means, marker="o", linewidth=2, color="#1f77b4",
            label="mean expected_pmi", zorder=3)
    for d, m in zip(depths, means):
        ax.annotate(f"{m:.3f}", (d, m), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center", color="#1f77b4")
    for d, n in d2n_bl.items():
        ax.axvline(d, color="red", alpha=0.25, linestyle="--", linewidth=1.2)
        ax.text(d, ax.get_ylim()[1], f"BL × {n}",
                color="red", fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Tree depth (root = 0)", fontsize=11)
    ax.set_ylabel(f"Mean expected_pmi (use_root=True, eval_alpha={EVAL_ALPHA})",
                  fontsize=11)
    ax.set_title(f"{tree_label} — mean empirical PMI vs root by depth",
                 fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.set_xticks(depths)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

_plot_score_by_depth(ctx_root, ctx_bl_members,
                     os.path.join(CONTEXT_DIR, "score_by_depth.png"),
                     "CONTEXT tree")
print(f"  Context score-by-depth → {CONTEXT_DIR}/score_by_depth.png")

# Summary.
with open(os.path.join(CONTEXT_DIR, "method_summary.txt"), "w") as f:
    f.write("CONTEXT tree — get_basic(use_root=True, "
            f"eval_alpha={EVAL_ALPHA})\n")
    f.write("=" * 56 + "\n\n")
    f.write(f"  Test primitives: {len(test_primitives)}\n")
    f.write(f"  Unique BL nodes: {len(ctx_bl_members)}\n\n")
    for i, m in enumerate(sorted(ctx_bl_members.values(),
                                 key=lambda m: len(m["pos_labels"]),
                                 reverse=True)):
        cnts = np.bincount(np.array(m["pos_labels"]), minlength=N_PRIM)
        dom = id2prim[int(cnts.argmax())]
        f.write(f"  [{i:>2}] depth={m['depth']:>2}  "
                f"count={int(m['node'].count):>5}  "
                f"members={len(m['pos_labels']):>4}  dom={dom}\n")

# =============================================================================
# PHASE 2 — CONTENT TREE INSPECTION
# =============================================================================
# Every supervised hollow chunk has a content_instance. Greedy descend
# → leaf → get_basic. Per-BL chunk-class histograms + (L,R) child-class
# joint + top center bigrams. Tree-with-L/R-bars (chunk-class palette,
# red borders on BL nodes). Score by depth.
# =============================================================================
print("\n=== PHASE 2: CONTENT TREE INSPECTION ===")

cnt_root = webster.ltm.content_hierarchy.root
N_LABEL  = len(ALL_LABELS)
label2id = {lbl: i for i, lbl in enumerate(ALL_LABELS)}

cnt_bl_cache: dict = {}
cnt_bl_members: dict = {}
chunk_records: list = []
for hollow in hollow_corpus:
    sentence = hollow["sentence"]; sent_toks = sentence.split()
    n_words = len(sent_toks)
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold="converge")
    for m in hollow["merges"]:
        try: tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    for comp in _walk_composites(tree.global_root_node):
        ci = comp.get_content_instance()
        if not ci: continue
        cls = classify_chunk(comp, n_words)
        if cls is None: continue
        # Each child's class (for the (L,R) joint matrix below).
        kids = sorted(getattr(comp, "children", []),
                      key=lambda x: x[0] if x[0] is not None else 0)
        if len(kids) != 2: continue
        l_cls = classify_chunk(kids[0][1], n_words)
        r_cls = classify_chunk(kids[1][1], n_words)
        leaf = greedy_descend(cnt_root, ci)
        bl   = get_basic_cached(leaf, cnt_bl_cache)
        if bl is None: continue
        h = str(bl.concept_hash())
        if h not in cnt_bl_members:
            cnt_bl_members[h] = {
                "node": bl, "depth": bl.depth(),
                "self_cls": [], "L_cls": [], "R_cls": [],
                "tokens_list": [],
            }
        cnt_bl_members[h]["self_cls"].append(cls)
        cnt_bl_members[h]["L_cls"].append(l_cls or "OTHER")
        cnt_bl_members[h]["R_cls"].append(r_cls or "OTHER")
        cnt_bl_members[h]["tokens_list"].append(_chunk_tokens(comp))
        s, e = _chunk_span(comp)
        chunk_records.append({
            "sentence": sentence,
            "span": (s, e), "tokens": sent_toks[s:e+1],
            "pos_yield": _chunk_yield(comp),
            "class": cls, "L_class": l_cls, "R_class": r_cls,
            "leaf_hash": str(leaf.concept_hash()),
            "bl_hash":   h,
            "content_instance": ci,
        })
print(f"  {len(chunk_records)} supervised chunks → {len(cnt_bl_members)} BL nodes")

# Class distribution.
print("  Chunk class distribution (head-based, S=root):")
cls_dist = Counter(r["class"] for r in chunk_records)
for cls in CHUNK_LABELS + ["OTHER"]:
    if cls_dist.get(cls, 0) > 0:
        print(f"    {cls:>5}: {cls_dist[cls]:>4}")

# Per-leaf class breakdown for clustering purity.
leaf_classes: dict = defaultdict(Counter)
for r in chunk_records:
    leaf_classes[r["leaf_hash"]][r["class"]] += 1
class_match = Counter(); class_total = Counter()
for r in chunk_records:
    cls = r["class"]
    dom = leaf_classes[r["leaf_hash"]].most_common(1)[0][0]
    class_total[cls] += 1
    if dom == cls: class_match[cls] += 1
print("  Per-class clustering purity:")
for cls in CHUNK_LABELS:
    t = class_total.get(cls, 0)
    if t == 0: continue
    print(f"    {cls:>5}: {class_match[cls]}/{t} ({100*class_match[cls]/t:.1f}%)")

# ── Per-BL viz (chunk-class joint + top bigrams + per-side attr values) ─────
def _top_per_side_attrs(node, k=TOP_CTX_NODES):
    """For attrs 0/1 (Left/Right TopK pool ids), return the top-k value
    ids and their fractions."""
    out = {0: [], 1: []}
    av = node.av_count or {}
    for attr in (0, 1):
        items = sorted([(v, c) for v, c in (av.get(attr, {}) or {}).items()
                         if v != 0], key=lambda kv: -kv[1])[:k]
        total = sum(c for _, c in items) or 1
        out[attr] = [(int(v), c / total) for v, c in items]
    return out

def plot_bl_subtrees_chunk(members, title, out_path):
    sorted_bls = sorted(members.values(),
                        key=lambda m: len(m["self_cls"]), reverse=True)
    n_rows = len(sorted_bls)
    if n_rows == 0: return
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(16, max(2.4, n_rows * 1.9)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.6, 1.8, 2.2]},
    )
    fig.suptitle(title, fontsize=11)
    for row, m in enumerate(sorted_bls):
        node = m["node"]
        n_mem = len(m["self_cls"]); depth = m["depth"]
        # Joint matrix L_cls × R_cls (over chunk labels + primitive labels)
        labels_lr = ALL_LABELS
        Nl = len(labels_lr)
        joint = np.zeros((Nl, Nl), dtype=np.int32)
        for L, R in zip(m["L_cls"], m["R_cls"]):
            li = label2id.get(L, label2id["OTHER"])
            ri = label2id.get(R, label2id["OTHER"])
            joint[li, ri] += 1
        # Dominant self-class.
        self_cnt = Counter(m["self_cls"])
        dom_self = self_cnt.most_common(1)[0][0]

        # ── Col 0: (L,R) class joint heatmap ──
        ax0 = axes[row, 0]
        used_l = sorted(set(L for L in m["L_cls"]),
                        key=lambda x: label2id.get(x, 99))
        used_r = sorted(set(R for R in m["R_cls"]),
                        key=lambda x: label2id.get(x, 99))
        sub = np.zeros((len(used_l), len(used_r)), dtype=np.int32)
        for L, R in zip(m["L_cls"], m["R_cls"]):
            if L in used_l and R in used_r:
                sub[used_l.index(L), used_r.index(R)] += 1
        ax0.imshow(sub / max(sub.sum(), 1), cmap="Blues",
                   vmin=0, vmax=1, aspect="equal")
        ax0.set_xticks(range(len(used_r)))
        ax0.set_yticks(range(len(used_l)))
        ax0.set_xticklabels(used_r, rotation=45, ha="right", fontsize=6)
        ax0.set_yticklabels(used_l, fontsize=6)
        ax0.set_xlabel("R class", fontsize=6)
        ax0.set_ylabel(
            f"d={depth}\nn={n_mem}\ncnt={int(node.count)}\n"
            f"self={dom_self}\n\nL class",
            fontsize=6, rotation=0, labelpad=46, va="center",
        )
        if row == 0: ax0.set_title("(L,R) class joint", fontsize=8)

        # ── Col 1: top center bigrams ──
        ax1 = axes[row, 1]
        bigrams = Counter()
        for toks in m["tokens_list"]:
            if len(toks) >= 2:
                bigrams[(toks[0], toks[-1])] += 1
        top_big = bigrams.most_common(TOP_CENTER_WORDS)
        if top_big:
            labels_s = [f"{a} … {b}" for (a, b), _ in top_big]
            counts_  = [c for _, c in top_big]
            colors_  = [LABEL_COLOR.get(WORD_TO_POS.get(a, "OTHER"), "#888")
                        for (a, _), _ in top_big]
            ax1.barh(np.arange(len(labels_s))[::-1], counts_,
                     color=colors_, edgecolor="white", linewidth=0.4)
            ax1.set_yticks(np.arange(len(labels_s))[::-1])
            ax1.set_yticklabels(labels_s, fontsize=6)
            ax1.tick_params(axis="x", labelsize=5)
        if row == 0: ax1.set_title("top center (1st…last) bigrams", fontsize=8)

        # ── Col 2: top per-side attr values (TopK-Pool ids) ──
        ax2 = axes[row, 2]; ax2.axis("off")
        attr_top = _top_per_side_attrs(node, k=TOP_CTX_NODES)
        for ci, side in enumerate(("L", "R")):
            cx = (ci + 0.5) / 2.0
            ax2.text(cx, 0.95, side, ha="center", va="top",
                     fontsize=7, fontweight="bold", transform=ax2.transAxes)
            for li, (vid, frac) in enumerate(attr_top[ci]):
                cy = 0.85 - li * 0.16
                ax2.text(cx, cy, f"pool#{vid} ({frac:.2f})",
                         ha="center", va="top", fontsize=6,
                         color="black", transform=ax2.transAxes)
        if row == 0:
            ax2.set_title(f"top per-side attr values (k={TOP_CTX_NODES})",
                          fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

plot_bl_subtrees_chunk(
    cnt_bl_members,
    title=(f"CONTENT TREE basic-level subtrees "
           f"— get_basic(use_root=True, eval_alpha={EVAL_ALPHA})  "
           f"(N_chunks={len(chunk_records)}, n_subtrees={len(cnt_bl_members)})"),
    out_path=os.path.join(CONTENT_DIR, "basic_level_subtrees.png"),
)
print(f"  Content BL subtree fig → {CONTENT_DIR}/basic_level_subtrees.png")

# CSV.
with open(os.path.join(CONTENT_DIR, "per_subtree_membership.csv"), "w") as f:
    w_ = csv.writer(f)
    w_.writerow(["subtree_idx", "depth", "node_count", "test_members",
                 "dominant_self_class", "class_distribution"])
    for i, m in enumerate(sorted(cnt_bl_members.values(),
                                 key=lambda m: len(m["self_cls"]),
                                 reverse=True)):
        cnts = Counter(m["self_cls"])
        dom = cnts.most_common(1)[0][0]
        dist = "/".join(f"{c}:{n}" for c, n in cnts.most_common())
        w_.writerow([i, m["depth"], int(m["node"].count),
                     len(m["self_cls"]), dom, dist])

# Tree-with-L/R-bars on content tree, BL nodes highlighted.
def compute_cnt_node_counts(root, chunk_records, max_depth):
    """Greedy descend each chunk's content instance through the layout;
    accumulate L/R-class counts at every node on the descent path."""
    all_nodes, children_of, _ = _make_layout(root, max_depth)
    cnt_L = {}; cnt_R = {}
    for r in chunk_records:
        ci = r["content_instance"]
        L  = r["L_class"] or "OTHER"
        R  = r["R_class"] or "OTHER"
        li = label2id.get(L, label2id["OTHER"])
        ri = label2id.get(R, label2id["OTHER"])
        cur = 0
        for _ in range(max_depth + 1):
            cnt_L.setdefault(cur, np.zeros(N_LABEL, dtype=np.int64))
            cnt_R.setdefault(cur, np.zeros(N_LABEL, dtype=np.int64))
            cnt_L[cur][li] += 1; cnt_R[cur][ri] += 1
            ch = children_of.get(cur, [])
            if not ch: break
            cur = max(ch, key=lambda i: all_nodes[i].log_prob_instance(ci))
    return all_nodes, children_of, cnt_L, cnt_R

def plot_tree_pair_bars(children_of, cL, cR, label_list, color_map,
                        highlight_idx, title, out_path, max_depth,
                        all_nodes=None):
    """Draw an L/R-bars-per-node tree.  All BFS-discovered Cobweb
    children are kept (no pruning by descent-tally) so the layout
    faithfully reflects Cobweb's structure.  Cobweb-real children
    with no descended chunks render as a small grey placeholder
    showing only the Cobweb-stored count."""
    pos_map, total_w = _layout_x(children_of, max_depth)
    bar_w, bar_h, gap, y_unit = 0.7, 0.18, 0.05, 1.0
    fig, ax = plt.subplots(
        figsize=(max(14, total_w * 0.9), (max_depth + 1) * 2.4))
    ax.set_xlim(0, total_w); ax.set_ylim(-0.7, max_depth * y_unit + 0.7)
    ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=11)

    def _has(idx):
        return idx in cL and cL[idx].sum() > 0

    def _cobweb_n(idx):
        if all_nodes and 0 <= idx < len(all_nodes):
            return int(getattr(all_nodes[idx], "count", 0))
        return 0

    def _edges(idx, depth):
        if depth >= max_depth or not children_of[idx]: return
        px, py = pos_map[idx]
        for c in children_of[idx]:
            cx, cy = pos_map[c]
            y_par = py * y_unit + bar_h + gap / 2
            y_chi = cy * y_unit - bar_h - gap / 2
            ax.plot([px, cx], [y_par, y_chi], color="gray", lw=0.7, zorder=0)
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
            lw=(3.0 if is_bl else 0.4),
            zorder=(5 if is_bl else 2)))
        ax.text(x_left - 0.04, y_top + bar_h / 2, txt,
                ha="right", va="center", fontsize=5)

    def _draw(idx, depth):
        x_c, _ = pos_map[idx]; x_left = x_c - bar_w / 2
        if _has(idx):
            cL_a = cL[idx].astype(float); cR_a = cR[idx].astype(float)
            total = cL_a.sum()
            propsL = cL_a / total; propsR = cR_a / total
            is_bl = idx in highlight_idx
            y_top_L = depth * y_unit - bar_h - gap / 2
            y_top_R = depth * y_unit + gap / 2
            _bar(x_left, y_top_L, propsL, "L", is_bl)
            _bar(x_left, y_top_R, propsR, "R", is_bl)
            ax.text(x_c, y_top_L - 0.04, f"n={int(total)}",
                    ha="center", va="bottom", fontsize=5)
        else:
            # Cobweb-real child with no descended chunks.  Show the
            # Cobweb-stored count as a grey placeholder spanning the
            # L+R-bar height so the user can see the branch exists.
            cw_n = _cobweb_n(idx)
            y_top = depth * y_unit - bar_h - gap / 2
            ax.add_patch(plt.Rectangle(
                (x_left, y_top), bar_w, 2 * bar_h + gap,
                facecolor="#f0f0f0", edgecolor="gray",
                lw=0.4, zorder=2))
            ax.text(x_c, depth * y_unit, f"cobweb_n={cw_n}",
                    ha="center", va="center", fontsize=5, color="gray",
                    style="italic")
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]: _draw(c, depth + 1)
    _draw(0, 0)

    used = np.zeros(len(label_list), dtype=np.int64)
    for c in cL.values(): used += c
    for c in cR.values(): used += c
    legend_h = [plt.Rectangle((0, 0), 1, 1, color=color_map[lbl], label=lbl)
                for i, lbl in enumerate(label_list) if used[i] > 0]
    ax.legend(handles=legend_h,
              title="class (top=L child, bottom=R child; red border=BL)",
              loc="lower right",
              ncol=max(1, len(legend_h) // 4), fontsize=7, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

cnt_layout_nodes, cnt_layout_children, cnt_L, cnt_R = compute_cnt_node_counts(
    cnt_root, chunk_records, max_depth=TREE_DEPTH_FIG)
cnt_bl_hashes = {str(m["node"].concept_hash()) for m in cnt_bl_members.values()}
cnt_highlight_idx = {i for i, n in enumerate(cnt_layout_nodes)
                     if str(n.concept_hash()) in cnt_bl_hashes}
plot_tree_pair_bars(
    cnt_layout_children, cnt_L, cnt_R, ALL_LABELS, LABEL_COLOR,
    cnt_highlight_idx,
    title=(f"CONTENT tree — L/R child class distributions  "
           f"(red border = BL, eval_alpha={EVAL_ALPHA})"),
    out_path=os.path.join(CONTENT_DIR, "content_tree_labels.png"),
    max_depth=TREE_DEPTH_FIG,
    all_nodes=cnt_layout_nodes,
)
print(f"  Content tree fig → {CONTENT_DIR}/content_tree_labels.png")

_plot_score_by_depth(cnt_root, cnt_bl_members,
                     os.path.join(CONTENT_DIR, "score_by_depth.png"),
                     "CONTENT tree")
print(f"  Content score-by-depth → {CONTENT_DIR}/score_by_depth.png")

with open(os.path.join(CONTENT_DIR, "method_summary.txt"), "w") as f:
    f.write(f"CONTENT tree — get_basic(use_root=True, "
            f"eval_alpha={EVAL_ALPHA})\n")
    f.write("=" * 56 + "\n\n")
    f.write(f"  Chunks: {len(chunk_records)}\n")
    f.write(f"  Unique BL nodes: {len(cnt_bl_members)}\n\n")
    f.write("Class distribution:\n")
    for cls, c in cls_dist.most_common():
        f.write(f"  {cls:>5}: {c:>4}\n")
    f.write("\nPer-class clustering purity:\n")
    for cls in CHUNK_LABELS:
        t = class_total.get(cls, 0)
        if t == 0: continue
        f.write(f"  {cls:>5}: {class_match[cls]}/{t} "
                f"({100*class_match[cls]/t:.1f}%)\n")
    f.write(f"\nBL nodes (sorted by membership):\n")
    for i, m in enumerate(sorted(cnt_bl_members.values(),
                                 key=lambda m: len(m["self_cls"]),
                                 reverse=True)):
        cnts = Counter(m["self_cls"])
        dom = cnts.most_common(1)[0][0]
        f.write(f"  [{i:>2}] depth={m['depth']:>2}  "
                f"count={int(m['node'].count):>5}  "
                f"members={len(m['self_cls']):>4}  dom={dom}\n")

# =============================================================================
# PHASE 3 — REPRESENTATION QUALITY (Cobweb-discrete probe on bags)
# =============================================================================
# Treat the bag WEBSTER builds for each item as its learned encoding
# and ask: how DISCERNABLE are these bags by class?
#
# Probe protocol — Cobweb-Discrete classifier, exactly the format the
# bags already use (no flattening, no DictVectorizer, no continuous
# feature treatment): for each training item we ``ifit`` an instance
# whose attrs are the original bag attrs PLUS one extra attribute
# (``_CLASS_ATTR``) that carries the gold class id.  At test time we
# categorize the test bag *without* the class attribute and read the
# landing leaf's class-attr distribution — the majority class there
# is the prediction.  This keeps each attribute discrete (value-set
# per attribute) the way Cobweb is built for, and matches how WEBSTER
# encodes its content / context hierarchies elsewhere.
#
# Bag definitions:
#   * Chunk bag    = ``content_instance``  — attrs 0/1 carry the
#                    TopK-Pool context-tree canonical ids; attr -2
#                    holds the complexity tag.
#   * Primitive bag = ``context_instance`` with the ``content_ref``
#                    attribute stripped — that attribute literally IS
#                    the gold word's vocab id, so leaving it in would
#                    let the probe trivially memorise the answer.
# =============================================================================
print("\n=== PHASE 3: REPRESENTATION QUALITY (Cobweb-discrete probe) ===")
print(f"  Held-out hollow sentences: {len(hollow_test)}")

# Special attr slot for the gold-label attribute on the probe tree.
# Picked far away from existing attrs (chunks use 0/1/-2, primitives
# use 0..2*ctx-1, -2, content_ref).
_CLASS_ATTR = -1000

def _clean_bag(bag):
    """Drop EMPTYNULL (vid 0) sentinels from each attr's value-set so
    they don't dominate the probe's entropy."""
    out = {}
    for a, vm in bag.items():
        cleaned = {v: c for v, c in (vm or {}).items() if v != 0}
        if cleaned: out[a] = cleaned
    return out

def _train_probe(train_bags, train_labels):
    """Fit a CobwebDiscreteTree where each training instance carries
    its bag attrs + one extra class attr."""
    label_ids = {lbl: i + 1 for i, lbl in enumerate(sorted(set(train_labels)))}
    id_labels = {i: lbl for lbl, i in label_ids.items()}
    probe = CobwebDiscreteTree(alpha=PROBE_ALPHA, weight_attr=True)
    for bag, lbl in zip(train_bags, train_labels):
        inst = _clean_bag(bag)
        inst[_CLASS_ATTR] = {label_ids[lbl]: 1}
        probe.ifit(inst)
    return probe, label_ids, id_labels

def _predict_probe(probe, bag, id_labels):
    """Greedy descend through the probe tree using the bag (no class
    attr) and read the leaf's class-attr distribution.  Walks up if
    the leaf carries no class info (shouldn't happen since every
    training item populates every ancestor)."""
    inst = _clean_bag(bag)
    n = probe.root
    while n.children:
        n = max(n.children, key=lambda c: c.log_prob_instance(inst))
    while n is not None:
        dist = (n.av_count or {}).get(_CLASS_ATTR, {})
        winning = [(v, c) for v, c in (dist or {}).items() if v != 0]
        if winning:
            best_id = max(winning, key=lambda kv: kv[1])[0]
            return id_labels.get(best_id)
        n = getattr(n, "parent", None)
    return None

# ── 3a. CHUNK probe ──────────────────────────────────────────────────────────
print("  Collecting chunk bags from train + test hollow sentences...")
train_chunk_bags = []; train_chunk_y = []
for r in chunk_records:               # train fold (already classified in Phase 2)
    train_chunk_bags.append(r["content_instance"])
    train_chunk_y.append(r["class"])

test_chunk_bags = []; test_chunk_y = []; test_chunk_meta = []
for hollow in hollow_test:
    sentence = hollow["sentence"]; sent_toks = sentence.split()
    n_words  = len(sent_toks)
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold="converge")
    for m in hollow["merges"]:
        try: tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    for comp in _walk_composites(tree.global_root_node):
        ci = comp.get_content_instance()
        if not ci: continue
        gold = classify_chunk(comp, n_words)
        if gold is None: continue
        s, e = _chunk_span(comp)
        test_chunk_bags.append(ci)
        test_chunk_y.append(gold)
        test_chunk_meta.append({
            "sentence":  sentence,
            "span":      (s, e),
            "tokens":    " ".join(sent_toks[s:e+1]),
            "pos_yield": " ".join(_chunk_yield(comp)),
        })

print(f"    train bags: {len(train_chunk_bags)}    test bags: {len(test_chunk_bags)}")
print("  Fitting Cobweb-Discrete probe on chunk bags...")
chunk_probe, chunk_label_ids, chunk_id_labels = _train_probe(
    train_chunk_bags, train_chunk_y)
chunk_preds = [_predict_probe(chunk_probe, b, chunk_id_labels) or "UNKNOWN"
               for b in test_chunk_bags]

chunk_test_rows = []
for meta, gold, pred in zip(test_chunk_meta, test_chunk_y, chunk_preds):
    chunk_test_rows.append({
        **meta,
        "gold": gold, "pred": str(pred),
        "ok": str(pred) == gold,
    })

n_chunks  = len(chunk_test_rows)
n_correct = sum(1 for r in chunk_test_rows if r["ok"])
chunk_overall = n_correct / max(n_chunks, 1)
print(f"    Overall chunk accuracy: {n_correct}/{n_chunks} ({100*chunk_overall:.1f}%)")

# Per-class precision / recall / F1.  Classes are the gold class set
# (sklearn might predict any of them, so we iterate over that union).
TEST_CLASSES   = sorted(set(train_chunk_y) | set(test_chunk_y))
chunk_gold_by  = Counter(r["gold"] for r in chunk_test_rows)
chunk_pred_by  = Counter(r["pred"] for r in chunk_test_rows)
chunk_tp_by    = Counter(r["gold"] for r in chunk_test_rows if r["ok"])
chunk_rows_sum = []
print(f"    {'class':<6} {'n':>5} {'TP':>4} {'P':>6} {'R':>6} {'F1':>6}")
for cls in TEST_CLASSES:
    n   = chunk_gold_by.get(cls, 0)
    tp  = chunk_tp_by.get(cls, 0)
    pp  = chunk_pred_by.get(cls, 0)
    prec = tp / pp if pp else 0.0
    rec  = tp / n  if n  else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    if n == 0 and pp == 0: continue
    chunk_rows_sum.append((cls, n, tp, prec, rec, f1))
    print(f"    {cls:<6} {n:>5} {tp:>4} {100*prec:>5.1f}% {100*rec:>5.1f}% {100*f1:>5.1f}%")

# Confusion matrix.
def _confusion(rows, classes, gold_key="gold", pred_key="pred"):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=np.int32)
    for r in rows:
        g, p = r[gold_key], r[pred_key]
        if g in idx and p in idx:
            M[idx[g], idx[p]] += 1
    return M

def _plot_confusion(M, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if M[i, j] == 0: continue
            ax.text(j, i, f"{M[i,j]}\n{100*norm[i,j]:.0f}%",
                    ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black",
                    fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

M_chunk = _confusion(chunk_test_rows, TEST_CLASSES)
_plot_confusion(
    M_chunk, TEST_CLASSES,
    title=(f"Chunk representation classification "
           f"(overall {100*chunk_overall:.1f}%, n={n_chunks})"),
    out_path=os.path.join(DECODING_DIR, "chunk_confusion.png"))

# Per-class accuracy bar chart.
fig, ax = plt.subplots(figsize=(8, 4.5))
cls_with_data = [c for c in CHUNK_LABELS if chunk_gold_by.get(c, 0) > 0]
recalls  = [chunk_tp_by.get(c, 0) / chunk_gold_by[c] for c in cls_with_data]
ax.bar(cls_with_data, recalls,
       color=[LABEL_COLOR[c] for c in cls_with_data])
for i, c in enumerate(cls_with_data):
    n = chunk_gold_by[c]; tp = chunk_tp_by.get(c, 0)
    ax.text(i, recalls[i] + 0.02, f"{tp}/{n}", ha="center", fontsize=9)
ax.axhline(1/len(cls_with_data), color="red", linestyle="--", alpha=0.5,
           label=f"chance ({len(cls_with_data)}-way)")
ax.set_ylim(0, 1.1); ax.set_ylabel("Per-class recall")
ax.set_title(f"Chunk-bag classification recall "
             f"(overall {100*chunk_overall:.1f}%)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DECODING_DIR, "chunk_quality.png"), dpi=120)
plt.close()

with open(os.path.join(DECODING_DIR, "chunk_quality.csv"), "w") as f:
    w_ = csv.writer(f)
    w_.writerow(["sentence", "span_start", "span_end", "tokens",
                 "pos_yield", "gold", "pred", "ok"])
    for r in chunk_test_rows:
        w_.writerow([r["sentence"], r["span"][0], r["span"][1],
                     r["tokens"], r["pos_yield"],
                     r["gold"], r["pred"], int(r["ok"])])

# ── 3b. PRIMITIVE probe ──────────────────────────────────────────────────────
# Same protocol on context_instance bags — but with the content_ref
# attribute (= the gold word's vocab id) stripped, so the probe can't
# trivially memorise the answer from the input.
print(f"\n  Collecting primitive bags from train + test sentences...")
content_ref_attr = webster.ltm.content_ref_attr

def _primitive_bag(toks, i):
    inst = _build_ctx_instance(toks, i)
    inst.pop(content_ref_attr, None)
    return inst

train_prim_bags = []; train_prim_y = []
for s in training_sentences:
    toks = s.split()
    for i, w in enumerate(toks):
        pos = WORD_TO_POS.get(w)
        if pos is None: continue
        train_prim_bags.append(_primitive_bag(toks, i))
        train_prim_y.append(pos)

test_prim_bags = []; test_prim_y = []; test_prim_meta = []
for hollow in hollow_test:
    toks = hollow["sentence"].split()
    for i, w in enumerate(toks):
        gold = WORD_TO_POS.get(w)
        if gold is None: continue
        test_prim_bags.append(_primitive_bag(toks, i))
        test_prim_y.append(gold)
        test_prim_meta.append({
            "sentence": hollow["sentence"],
            "position": i, "word": w,
        })

print(f"    train tokens: {len(train_prim_bags)}    test tokens: {len(test_prim_bags)}")
print("  Fitting Cobweb-Discrete probe on primitive bags...")
prim_probe, prim_label_ids, prim_id_labels = _train_probe(
    train_prim_bags, train_prim_y)
prim_preds = [_predict_probe(prim_probe, b, prim_id_labels) or "UNKNOWN"
              for b in test_prim_bags]

prim_test_rows = []
for meta, gold, pred in zip(test_prim_meta, test_prim_y, prim_preds):
    prim_test_rows.append({
        **meta, "gold": gold, "pred": str(pred),
        "ok": str(pred) == gold,
    })

n_prim  = len(prim_test_rows)
n_pcorr = sum(1 for r in prim_test_rows if r["ok"])
prim_overall = n_pcorr / max(n_prim, 1)
print(f"    Overall primitive accuracy: {n_pcorr}/{n_prim} ({100*prim_overall:.1f}%)")

prim_gold_by = Counter(r["gold"] for r in prim_test_rows)
prim_pred_by = Counter(r["pred"] for r in prim_test_rows)
prim_tp_by   = Counter(r["gold"] for r in prim_test_rows if r["ok"])
prim_rows_sum = []
print(f"    {'POS':<5} {'n':>5} {'TP':>4} {'P':>6} {'R':>6} {'F1':>6}")
for cls in PRIM_LABELS:
    n  = prim_gold_by.get(cls, 0)
    tp = prim_tp_by.get(cls, 0)
    pp = prim_pred_by.get(cls, 0)
    prec = tp / pp if pp else 0.0
    rec  = tp / n  if n  else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    if n == 0 and pp == 0: continue
    prim_rows_sum.append((cls, n, tp, prec, rec, f1))
    print(f"    {cls:<5} {n:>5} {tp:>4} {100*prec:>5.1f}% {100*rec:>5.1f}% {100*f1:>5.1f}%")

M_prim = _confusion(prim_test_rows, PRIM_LABELS)
_plot_confusion(
    M_prim, PRIM_LABELS,
    title=(f"Primitive representation classification "
           f"(overall {100*prim_overall:.1f}%, n={n_prim})"),
    out_path=os.path.join(DECODING_DIR, "primitive_confusion.png"))

fig, ax = plt.subplots(figsize=(8, 4.5))
pos_with_data = [p for p in PRIM_LABELS if prim_gold_by.get(p, 0) > 0]
recalls = [prim_tp_by.get(p, 0) / prim_gold_by[p] for p in pos_with_data]
ax.bar(pos_with_data, recalls,
       color=[LABEL_COLOR[p] for p in pos_with_data])
for i, p in enumerate(pos_with_data):
    n = prim_gold_by[p]; tp = prim_tp_by.get(p, 0)
    ax.text(i, recalls[i] + 0.02, f"{tp}/{n}", ha="center", fontsize=9)
ax.axhline(1/len(pos_with_data), color="red", linestyle="--", alpha=0.5,
           label=f"chance ({len(pos_with_data)}-way)")
ax.set_ylim(0, 1.1); ax.set_ylabel("Per-POS recall")
ax.set_title(f"Primitive-bag classification recall "
             f"(overall {100*prim_overall:.1f}%)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DECODING_DIR, "primitive_quality.png"), dpi=120)
plt.close()

with open(os.path.join(DECODING_DIR, "primitive_quality.csv"), "w") as f:
    w_ = csv.writer(f)
    w_.writerow(["sentence", "position", "word", "gold", "pred", "ok"])
    for r in prim_test_rows:
        w_.writerow([r["sentence"], r["position"], r["word"],
                     r["gold"], r["pred"], int(r["ok"])])

# ── 3d. Precision / Recall / F1 visualizations ──────────────────────────────
# Two-pronged view of the per-class numbers we just printed:
#   1) Grouped bar chart  — P, R, F1 side-by-side per class.
#   2) Annotated heatmap  — class × {P, R, F1} table, RdYlGn-coloured.
# Both rebuild from ``chunk_rows_sum`` and ``prim_rows_sum``, which
# carry (class, n_gold, TP, P, R, F1) tuples computed above using the
# standard multiclass formulas:
#   precision = TP / (TP + FP) = tp_by[c] / pred_by[c]
#   recall    = TP / (TP + FN) = tp_by[c] / gold_by[c]
#   F1        = 2·P·R / (P + R)
def _plot_prf1(rows_sum, title, out_path):
    if not rows_sum: return
    classes = [r[0] for r in rows_sum]
    P  = [r[3] for r in rows_sum]
    R  = [r[4] for r in rows_sum]
    F1 = [r[5] for r in rows_sum]
    n_gold = [r[1] for r in rows_sum]
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.4), 5))
    x = np.arange(len(classes)); w = 0.27
    ax.bar(x - w, P,  w, label="Precision", color="#1f77b4")
    ax.bar(x,     R,  w, label="Recall",    color="#2ca02c")
    ax.bar(x + w, F1, w, label="F1",        color="#d62728")
    for i in range(len(classes)):
        ax.text(x[i] - w, P[i]  + 0.02, f"{100*P[i]:.0f}%",
                ha="center", fontsize=7)
        ax.text(x[i],     R[i]  + 0.02, f"{100*R[i]:.0f}%",
                ha="center", fontsize=7)
        ax.text(x[i] + w, F1[i] + 0.02, f"{100*F1[i]:.0f}%",
                ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, n_gold)],
                       fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

def _plot_prf1_heatmap(rows_sum, title, out_path):
    if not rows_sum: return
    classes = [r[0] for r in rows_sum]
    M = np.array([[r[3], r[4], r[5]] for r in rows_sum])
    fig, ax = plt.subplots(
        figsize=(6, max(2.4, 0.6 * len(classes) + 1.5)))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=10)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{c}  (n={r[1]}, TP={r[2]})"
                        for c, r in zip(classes, rows_sum)], fontsize=9)
    for i in range(len(classes)):
        for j in range(3):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if v > 0.4 else "white",
                    fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

_plot_prf1(
    chunk_rows_sum,
    title=(f"Chunk classification — Precision / Recall / F1 per class  "
           f"(overall {100*chunk_overall:.1f}%, n={n_chunks})"),
    out_path=os.path.join(DECODING_DIR, "chunk_prf1.png"))
_plot_prf1_heatmap(
    chunk_rows_sum,
    title=(f"Chunk PRF1 table  (overall {100*chunk_overall:.1f}%)"),
    out_path=os.path.join(DECODING_DIR, "chunk_prf1_table.png"))

_plot_prf1(
    prim_rows_sum,
    title=(f"Primitive classification — Precision / Recall / F1 per POS  "
           f"(overall {100*prim_overall:.1f}%, n={n_prim})"),
    out_path=os.path.join(DECODING_DIR, "primitive_prf1.png"))
_plot_prf1_heatmap(
    prim_rows_sum,
    title=(f"Primitive PRF1 table  (overall {100*prim_overall:.1f}%)"),
    out_path=os.path.join(DECODING_DIR, "primitive_prf1_table.png"))

# ── 3e. Overall summary CSV ──────────────────────────────────────────────────
with open(os.path.join(DECODING_DIR, "quality_summary.csv"), "w") as f:
    w_ = csv.writer(f)
    w_.writerow(["tree", "class", "n_gold", "n_correct",
                 "precision", "recall", "f1"])
    for cls, n, tp, prec, rec, f1 in prim_rows_sum:
        w_.writerow(["context", cls, n, tp, f"{prec:.4f}",
                     f"{rec:.4f}", f"{f1:.4f}"])
    for cls, n, tp, prec, rec, f1 in chunk_rows_sum:
        w_.writerow(["content", cls, n, tp, f"{prec:.4f}",
                     f"{rec:.4f}", f"{f1:.4f}"])
    w_.writerow(["context", "OVERALL", n_prim,  n_pcorr,
                 "", f"{prim_overall:.4f}", ""])
    w_.writerow(["content", "OVERALL", n_chunks, n_correct,
                 "", f"{chunk_overall:.4f}", ""])

# =============================================================================
# PHASE 4 — HEURISTIC COMPARISON
# =============================================================================
# "If the leaves are pure, why doesn't log-probability recognize chunks
#  correctly?"  This phase audits the answer.
#
# Approach: for every test item, descend WEBSTER's *actual* tree (the
# one parsing will use) to its terminal leaf.  Build a leaf → gold-class
# map from training instances along the same descent (walk up to the
# nearest labelled ancestor when needed) — this is the "leaf-majority"
# classifier.  Compare its accuracy to the Cobweb-Discrete probe from
# Phase 3.
#
# Then, for every test chunk, gather five candidate ranking heuristics
# from the leaf the chunk descends to, and report which ranks chunks
# of the same class together most reliably:
#
#   tree_log_prob          – marginal over the whole tree (current rank)
#   leaf_log_prob          – log p(instance | landing leaf)
#   bl_log_prob            – log p(instance | basic-level node)
#   bl_class_log_prob      – log p(class   | instance) at BL
#   bl_count               – just the BL count (pure frequency)
#
# Each heuristic is judged on its rank-discrimination by gold class
# (Spearman corr to the "this leaf's gold class" indicator).
# =============================================================================
print("\n=== PHASE 4: HEURISTIC COMPARISON ===")

# ── 4a. Direct leaf-majority on WEBSTER's actual tree ───────────────────────
# Build leaf → gold-class map from training data via greedy descent.
def _leaf_majority_map(root, items_with_class):
    leaf_classes = {}  # concept_hash → Counter
    for inst, cls in items_with_class:
        n = greedy_descend(root, inst)
        h = str(n.concept_hash())
        leaf_classes.setdefault(h, Counter())[cls] += 1
    return {h: c.most_common(1)[0][0] for h, c in leaf_classes.items()}

# Walk-up: returns leaf's class, else the nearest ancestor's, else None.
def _predict_leaf_or_ancestor(root, inst, leaf_pred):
    n = greedy_descend(root, inst)
    while n is not None:
        h = str(n.concept_hash())
        if h in leaf_pred:
            return leaf_pred[h]
        n = getattr(n, "parent", None)
    return None

# CHUNKS: build train map + evaluate.
chunk_leaf_pred = _leaf_majority_map(
    cnt_root,
    [(r["content_instance"], r["class"]) for r in chunk_records])
print(f"  Content tree leaves with class labels: {len(chunk_leaf_pred)}")
chunk_lm_correct = 0
for bag, gold in zip(test_chunk_bags, test_chunk_y):
    pred = _predict_leaf_or_ancestor(cnt_root, _clean_bag(bag), chunk_leaf_pred)
    if pred == gold: chunk_lm_correct += 1
chunk_lm_acc = chunk_lm_correct / max(len(test_chunk_y), 1)

# PRIMITIVES: same.
prim_train_items = []
for s in training_sentences:
    toks = s.split()
    for i, w in enumerate(toks):
        pos = WORD_TO_POS.get(w)
        if pos is None: continue
        prim_train_items.append((_primitive_bag(toks, i), pos))
prim_leaf_pred = _leaf_majority_map(ctx_root, prim_train_items)
print(f"  Context tree leaves with POS labels: {len(prim_leaf_pred)}")
prim_lm_correct = 0
for bag, gold in zip(test_prim_bags, test_prim_y):
    pred = _predict_leaf_or_ancestor(ctx_root, _clean_bag(bag), prim_leaf_pred)
    if pred == gold: prim_lm_correct += 1
prim_lm_acc = prim_lm_correct / max(len(test_prim_y), 1)

print(f"\n  --- Probe accuracy comparison (chunks) ---")
print(f"    Cobweb-Discrete probe (Phase 3a)   : "
      f"{n_correct}/{n_chunks} = {100*chunk_overall:5.1f}%")
print(f"    Leaf-majority on WEBSTER tree      : "
      f"{chunk_lm_correct}/{len(test_chunk_y)} = {100*chunk_lm_acc:5.1f}%")
print(f"    Δ = {100*(chunk_lm_acc - chunk_overall):+.1f}pp "
      f"({'WEBSTER tree more discriminative' if chunk_lm_acc > chunk_overall else 'Probe more discriminative'})")

print(f"\n  --- Probe accuracy comparison (primitives) ---")
print(f"    Cobweb-Discrete probe (Phase 3b)   : "
      f"{n_pcorr}/{n_prim} = {100*prim_overall:5.1f}%")
print(f"    Leaf-majority on WEBSTER tree      : "
      f"{prim_lm_correct}/{len(test_prim_y)} = {100*prim_lm_acc:5.1f}%")
print(f"    Δ = {100*(prim_lm_acc - prim_overall):+.1f}pp")

# ── 4b. Ranking-heuristic comparison on the content tree ────────────────────
# For each test chunk we measure five ranking heuristics at the leaf
# the chunk descends to, then ask: does each heuristic let us SELECT
# the right candidate?  Operationally:  among all test chunks that
# descend to the SAME leaf, the heuristic should rank them in a way
# that's consistent with their gold class.
#
# We measure that as the accuracy of "predict via leaf majority then
# tie-break by heuristic" — which is exactly what FiniteParseTree.build
# does (gate by count, rank by tree_log_prob).
from parse_mh import _score_along_path, _categorize
heur_results = []
for bag, gold in zip(test_chunk_bags, test_chunk_y):
    inst = _clean_bag(bag)
    leaf, path_strs, node_path, _ = _categorize(
        inst, webster.ltm.content_hierarchy, mode="dfs")
    sd = _score_along_path(node_path, inst, webster.ltm.content_hierarchy,
                            eval_alpha=getattr(webster.ltm,
                                                "content_bl_alpha", None))
    leaf_lp = leaf.log_prob_instance(inst)
    bl_cnt  = sd.get("basic_level_count", -1)
    bl_lp   = sd.get("basic_level_log_prob", float("-inf"))
    bl_clp  = sd.get("basic_level_class_log_prob", float("-inf"))
    tree_lp = sd.get("tree_log_prob", float("-inf"))
    leaf_pred = chunk_leaf_pred.get(str(leaf.concept_hash()))
    heur_results.append({
        "gold": gold,
        "leaf_pred": leaf_pred,
        "scores": {
            "tree_log_prob":     tree_lp,
            "leaf_log_prob":     leaf_lp,
            "bl_log_prob":       bl_lp,
            "bl_class_log_prob": bl_clp,
            "bl_count":          float(bl_cnt),
        },
    })

# For each heuristic, build a per-(landing-leaf) ranking and report
# how often the top-ranked item's gold class matches the leaf's
# dominant class — i.e. how reliably the heuristic agrees with the
# clustering.  Higher = more reliable as a ranking signal.
heur_names = ["tree_log_prob", "leaf_log_prob", "bl_log_prob",
              "bl_class_log_prob", "bl_count"]
print(f"\n  --- Ranking-heuristic discrimination (content tree) ---")
print(f"    {'heuristic':<22} {'agree-w-leaf-class':>20} {'rank-quality':>14}")

heur_agree = {h: [0, 0] for h in heur_names}   # [hit, total]
# Group test chunks by landing leaf so we can rank within each.
from collections import defaultdict
by_leaf: dict = defaultdict(list)
for bag, gold, hr in zip(test_chunk_bags, test_chunk_y, heur_results):
    if hr["leaf_pred"] is None: continue
    h = str(greedy_descend(cnt_root, _clean_bag(bag)).concept_hash())
    by_leaf[h].append((gold, hr))

# Per-heuristic: across leaves with mixed gold classes, does the
# heuristic rank the leaf-dominant-class items HIGHER than others?
heur_rank_quality = {h: [0, 0] for h in heur_names}   # [correct_rank, total_pairs]
for leaf_hash, items in by_leaf.items():
    if len(items) < 2: continue
    dom = Counter(g for g, _ in items).most_common(1)[0][0]
    # For each ordered pair (i, j), if i's gold==dom and j's gold!=dom,
    # the heuristic should rank i above j.
    for i, (gi, hri) in enumerate(items):
        for j, (gj, hrj) in enumerate(items):
            if i == j: continue
            if gi == dom and gj != dom:
                for h in heur_names:
                    heur_rank_quality[h][1] += 1
                    if hri["scores"][h] > hrj["scores"][h]:
                        heur_rank_quality[h][0] += 1

# Leaf-class agreement: how often does each heuristic's argmax agree
# with the leaf-majority prediction?
for h in heur_names:
    hit = total = 0
    for hr in heur_results:
        if hr["leaf_pred"] is None: continue
        total += 1
        if hr["leaf_pred"] == hr["gold"]:
            hit += 1
    heur_agree[h] = [hit, total]

for h in heur_names:
    hit, tot = heur_agree[h]
    rc, rt   = heur_rank_quality[h]
    agree_pct = 100 * hit / max(tot, 1)
    rank_pct  = 100 * rc / max(rt, 1)
    print(f"    {h:<22} {agree_pct:>18.1f}%   {rank_pct:>12.1f}%")

print(f"\n  Note: 'agree-w-leaf-class' is the SAME across heuristics —")
print(f"  it's the leaf-majority prediction accuracy, independent of "
      f"the ranking score.")
print(f"  'rank-quality' answers: when two test chunks land at the SAME")
print(f"  leaf, does the heuristic rank the dominant-class chunk higher?")
print(f"  Higher = better ranking signal for build()'s tie-break.")

# ── 4c. Heuristic-comparison visualization ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
rqs = [100 * heur_rank_quality[h][0] / max(heur_rank_quality[h][1], 1)
       for h in heur_names]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
ax.bar(range(len(heur_names)), rqs, color=colors)
for i, v in enumerate(rqs):
    ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
ax.axhline(50, color="black", linestyle=":", alpha=0.4,
           label="chance (random rank)")
ax.set_xticks(range(len(heur_names)))
ax.set_xticklabels(heur_names, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 110); ax.set_ylabel("Rank-quality % (pairs ranked right)")
ax.set_title("Ranking-heuristic discrimination — chunk content tree")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DECODING_DIR, "heuristic_comparison.png"), dpi=140)
plt.close()
print(f"  Heuristic comparison plot → {DECODING_DIR}/heuristic_comparison.png")

# ── 4d. Side-by-side accuracy bar (linear probe vs leaf-majority) ──────────
fig, ax = plt.subplots(figsize=(8, 4))
xs = ["Chunk\n(linear probe)", "Chunk\n(leaf majority)",
      "Primitive\n(linear probe)", "Primitive\n(leaf majority)"]
ys = [chunk_overall, chunk_lm_acc, prim_overall, prim_lm_acc]
bar_colors = ["#1f77b4", "#2ca02c", "#1f77b4", "#2ca02c"]
ax.bar(xs, ys, color=bar_colors)
for i, v in enumerate(ys):
    ax.text(i, v + 0.02, f"{100*v:.1f}%", ha="center", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
ax.set_title("Probe accuracy: Cobweb-Discrete probe vs WEBSTER-tree leaf-majority")
plt.tight_layout()
plt.savefig(os.path.join(DECODING_DIR, "probe_vs_leaf_majority.png"), dpi=140)
plt.close()
print(f"  Probe-vs-leaf-majority plot → {DECODING_DIR}/probe_vs_leaf_majority.png")

print(f"\nAll outputs written to {OUT_DIR}/")
print(f"  context_tree/    — primitive POS distributions + BL viz")
print(f"  content_tree/    — chunk L/R-class distributions + BL viz")
print(f"  decoding/        — representation-quality probe on held-out")
print(f"     chunk_quality.csv/.png        + chunk_confusion.png")
print(f"     chunk_prf1.png                + chunk_prf1_table.png")
print(f"     primitive_quality.csv/.png    + primitive_confusion.png")
print(f"     primitive_prf1.png            + primitive_prf1_table.png")
print(f"     quality_summary.csv")
print(f"     heuristic_comparison.png      ← Phase 4 ranking-heuristic test")
print(f"     probe_vs_leaf_majority.png    ← Phase 4 direct-leaf vs probe")
