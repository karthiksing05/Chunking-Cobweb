"""
WEBSTER Threshold + Parsing-Heuristic Analysis (met5)
=====================================================

Train WEBSTER on the hollow corpus (mirroring
``unittests/hollow_learn_test_mh.py``), then ask: which heuristic in
``evaluate_pair`` most reliably distinguishes the **gold** next-merge
from the rest of the parentless pairs at every parse step?

WEBSTER's ``FiniteParseTree.build`` currently uses a two-stage rule:

  Stage 1 (gate) : ``basic_level_count > count_threshold``
  Stage 2 (rank) : argmax content ``tree_log_prob``

The gate and rank are independent design choices — this test logs
EVERY heuristic the pair-evaluator produces so we can see which one
best matches the human's chunking order.

Heuristics logged per candidate pair (from ``evaluate_pair``)
-------------------------------------------------------------
  Content tree side:
    cnt_bl_count        — basic-level count           (current gate)
    cnt_tree_lp         — tree log-prob               (current ranker)
    cnt_bl_lp           — basic-level log-prob
    cnt_bl_class_lp     — basic-level p(class|inst)
    cnt_tree_class_lp   — tree p(class|inst)
  Context tree side (same set with ctx_ prefix).
  Joint:
    sum_tree_lp         — content + context tree-lp
    bl_count_min        — min(content bl_count, context bl_count)

Phases
------
0. Train WEBSTER + 80/20 split of the hollow corpus.
1. HEURISTIC LOGGING. For each TEST hollow sentence, replay the gold
   merge sequence; at each step evaluate ALL parentless pairs and
   record their full heuristic vector + a STRUCTURE-based gold flag.
   The hollow corpus's merge ORDER is one specific traversal of the
   gold parse tree — but two non-interacting chunks (A and B with
   disjoint spans) can be created in either order with the same
   final structure. So `is_gold` for a candidate is "this candidate's
   resulting span ∈ the gold bracket set", not "this candidate is the
   specific human-chosen merge at this step". Multiple gold-valid
   candidates can co-exist at one step.
2. DISCRIMINABILITY. Per-heuristic histograms (gold vs non-gold),
   ROC curves + AUC, Precision-Recall curves + AP. Tells us
   *which heuristic separates the right merge from the wrong ones*.
3. STEP-PICK ACCURACY. At each step, argmax each heuristic over the
   candidate pairs. Does it equal gold? Bar chart per heuristic.
4. THRESHOLD SWEEP on ``cnt_bl_count`` (the current gate): for each
   τ, retain candidates with ``count > τ`` and pick argmax tree_lp.
   Plot accuracy at the admitted steps + coverage.
5. PER-SENTENCE PARSE-TREE VIZ. For ``N_TREE_VIZ`` test sentences,
   render the gold parse tree with each merge node annotated by its
   heuristic values and a ✓/✗ per heuristic indicating whether that
   heuristic would have picked this merge.
6. NEGATIVE TEST (threshold sweep).  Generate random word strings
   (uniform from the lexicon, ungrammatical) and grammatical
   sentences of matched length.  Parse both populations at a sweep
   of count-gate thresholds and plot the resulting chunks-per-sentence
   curves.  Identifies the threshold maximising
   ``gram_norm × (1 − rand_norm)`` as the "best operating point".
7. HEURISTIC ANALYSIS for negative-input rejection.  At the
   primitive-bigram level, evaluate every adjacent pair in both
   populations and ask: which of the 22 heuristics (20 from
   evaluate_pair + bl_counts on both sides) best discriminates
   random-sentence bigrams from grammatical ones?  Reports
   per-heuristic AUC, single-threshold operating point, and the
   top-10 two-heuristic AND-combinations that beat single gates.

Outputs (``tests/met5/grammar_threshold_test_output/``)
-------------------------------------------------------
  candidate_log.csv                — every candidate pair + heuristics
  step_picks.csv                   — per-step heuristic picks
  heuristic_histograms.png         — gold vs non-gold density per heur
  heuristic_roc.png                — ROC curves (AUC in legend)
  heuristic_pr.png                 — Precision-Recall curves
  step_pick_accuracy.png           — argmax-heuristic step accuracy
  threshold_sweep.png              — bl_count gate sweep
  threshold_sweep.csv
  heuristic_summary.csv            — AUC + step-accuracy summary
  parse_trees/{i}_<sentence>.png   — annotated parse tree per sentence
  negative_test_sweep.csv          — count-gate sweep summary
  negative_test_sweep.png          — chunks vs threshold curves
  negative_test_selectivity.png    — selectivity ratio vs threshold
  negative_test_best_thr.csv       — per-sentence detail at best thr
  negative_test_heur_analysis.csv  — per-heuristic AUC + best gate
  negative_test_heur_auc.png       — AUC bar chart, all 22 heuristics
  negative_test_top_heur_hists.png — top-4 heuristic distributions
                                     (random vs grammar)
  negative_test_combo_gates.csv    — best two-heuristic AND combos
  summary.txt                      — overall best-heuristic report
"""

import os
import sys
import csv
import glob
import json
import random
import shutil
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc,
                              precision_recall_curve, average_precision_score)

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import (WEBSTER, FiniteParseTree, PrimitiveParseNode,
                      CompositeParseNode,
                      _get_or_register_cplx_vid, _context_weight)

# ── Configuration ─────────────────────────────────────────────────────────────
OUT_DIR           = os.path.join(_HERE, "grammar_threshold_test_output")
HOLLOW_CORPUS_DIR = "data/test_hollow_grammar_1"
CONTEXT_LENGTH    = 3
THRESHOLD         = 30                # baseline trained threshold
PRIMITIVES_FIRST  = 200
N_TREE_VIZ        = 12                # # of per-sentence trees to render
EVAL_ALPHA        = 10.0              # for tree-inspection get_basic
TREE_DEPTH_FIG    = 3
TOP_WORDS_PER_OFFSET = 3
TOP_CENTER_WORDS  = 6
TOP_CTX_NODES     = 5
SEED              = 13
random.seed(SEED); np.random.seed(SEED)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
TREE_DIR    = os.path.join(OUT_DIR, "parse_trees");  os.makedirs(TREE_DIR,    exist_ok=True)
CONTEXT_DIR = os.path.join(OUT_DIR, "context_tree"); os.makedirs(CONTEXT_DIR, exist_ok=True)
CONTENT_DIR = os.path.join(OUT_DIR, "content_tree"); os.makedirs(CONTENT_DIR, exist_ok=True)

# Word → POS for tree-inspection labelling.
POS_LIST = ["Det", "N", "Adj", "V", "P"]
WORD_TO_POS: dict = {}
for pos in POS_LIST:
    for prod in TEST_GRAMMAR1[pos]:
        for w in prod:
            WORD_TO_POS[w] = pos

# Palette (same as decoding test so visuals are comparable).
PRIM_LABELS  = POS_LIST
CHUNK_LABELS = ["NP", "AdjP", "PP", "VP", "S"]
ALL_LABELS   = PRIM_LABELS + CHUNK_LABELS + ["OTHER"]
LABEL_COLOR  = {
    "Det":   "#2ca02c", "N":     "#8c564b", "Adj":   "#1f77b4",
    "V":     "#17becf", "P":     "#7f7f7f", "NP":    "#ff7f0e",
    "AdjP":  "#9467bd", "PP":    "#bcbd22", "VP":    "#e377c2",
    "S":     "#d62728", "OTHER": "#cccccc",
}
N_PRIM     = len(PRIM_LABELS)
N_LABEL    = len(ALL_LABELS)
prim2id    = {p: i for i, p in enumerate(PRIM_LABELS)}
id2prim    = {i: p for p, i in prim2id.items()}
label2id   = {lbl: i for i, lbl in enumerate(ALL_LABELS)}
pos_colors = [LABEL_COLOR[p] for p in PRIM_LABELS]

# ── Heuristic name set (NO basic-level) ──────────────────────────────────────
# All 8 raw heuristics from evaluate_pair (4 per side) plus a handful
# of natural compositions.  Higher = more evidence for all of them.
# We deliberately exclude every basic-level-derived quantity because
# the basic-level detector is still under inspection.
BASE_HEURS = [
    # raw
    "cnt_tree_lp",
    "cnt_tree_class_lp",
    "cnt_root_lp",
    "cnt_leaf_lp",
    "ctx_tree_lp",
    "ctx_tree_class_lp",
    "ctx_root_lp",
    "ctx_leaf_lp",
    # joint sums
    "sum_tree_lp",         # cnt_tree_lp + ctx_tree_lp
    "sum_class_lp",        # cnt_tree_class_lp + ctx_tree_class_lp
    "sum_leaf_lp",         # cnt_leaf_lp + ctx_leaf_lp
    # how much the path's leaf log-prob beats the root prior
    "cnt_tree_above_root", # cnt_tree_lp - cnt_root_lp
    "ctx_tree_above_root",
    "sum_tree_above_root",
    "cnt_leaf_above_root", # cnt_leaf_lp - cnt_root_lp
    "ctx_leaf_above_root",
    # min/max across content/context tree
    "min_tree_lp",
    "max_tree_lp",
    "min_class_lp",
    "max_class_lp",
]
HEURISTICS = BASE_HEURS   # alias kept so the rest of the script reads same
HEUR_COLORS = plt.cm.tab20(np.linspace(0, 1, len(HEURISTICS)))

# =============================================================================
# PHASE 0 — TRAIN WEBSTER  (mirror unittests/hollow_learn_test_mh.py)
# =============================================================================
print("=== PHASE 0: Train WEBSTER ===")
webster = WEBSTER(
    TEST_CORPUS1,
    context_length=CONTEXT_LENGTH,
    threshold=THRESHOLD,
    content_alpha=1e-6,
    context_alpha=1e-6,
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

print(f"  Phase 0a: {PRIMITIVES_FIRST} primitive-only sentences")
for i in range(PRIMITIVES_FIRST):
    s = generate("S", TEST_GRAMMAR1)
    webster.parse_sentence(s, threshold=1e9, new_vocab=True,
                           learning=True, debug=False)
    if (i + 1) % 50 == 0:
        print(f"    [{i+1}/{PRIMITIVES_FIRST}]")

print(f"  Phase 0b: hollow corpus")
hollow_corpus: list = []
for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
    with open(p) as f:
        try:    data = json.load(f)
        except json.JSONDecodeError: continue
    if "sentence" in data and "merges" in data:
        hollow_corpus.append(data)
print(f"    Loaded {len(hollow_corpus)} hollow trees")

# 80/20 split (train trees go into WEBSTER; test trees feed the heuristic probe).
random.shuffle(hollow_corpus)
_split = int(0.8 * len(hollow_corpus))
train_hollow = hollow_corpus[:_split]
test_hollow  = hollow_corpus[_split:]
print(f"    Split: train={len(train_hollow)}  test={len(test_hollow)}")

for i, hollow in enumerate(train_hollow):
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(hollow["sentence"], threshold=THRESHOLD)
    for m in hollow["merges"]:
        try: tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)
    if (i + 1) % 25 == 0: print(f"    [{i+1}/{len(train_hollow)}]")

# =============================================================================
# PHASE 0b — TREE INSPECTION (sanity-check trees are well-built)
# =============================================================================
# Same visualizations as the met5/grammar_decoding_test.py outputs so
# we can confirm the context and content trees look right BEFORE
# trusting the heuristic analysis in Phase 1+.
# =============================================================================
print("\n=== PHASE 0b: Tree inspection ===")

# ── Shared helpers (mirrors met5/grammar_decoding_test.py) ───────────────────
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

# ── CONTEXT-tree inspection ──────────────────────────────────────────────────
ctx_root = webster.ltm.context_hierarchy.root
ctx_attr_offsets = {j: -(j+1) for j in range(CONTEXT_LENGTH)}
ctx_attr_offsets.update({CONTEXT_LENGTH + j: (j+1) for j in range(CONTEXT_LENGTH)})

def offset_for_attr(attr_id):
    return ctx_attr_offsets.get(attr_id, attr_id)

def _build_ctx_instance(toks, i):
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

print("  Generating fresh test sentences for primitive descent...")
test_primitives = []
for _ in range(150):
    s = generate("S", TEST_GRAMMAR1)
    toks = s.split()
    for i, w in enumerate(toks):
        if w in WORD_TO_POS:
            test_primitives.append((toks, i, w, WORD_TO_POS[w]))
print(f"  {len(test_primitives)} primitive test instances")

ctx_bl_cache: dict = {}
ctx_bl_members: dict = {}
for sent_toks, i, w, pos in test_primitives:
    inst = _build_ctx_instance(sent_toks, i)
    leaf = greedy_descend(ctx_root, inst)
    bl   = get_basic_cached(leaf, ctx_bl_cache)
    if bl is None: continue
    h = str(bl.concept_hash())
    if h not in ctx_bl_members:
        ctx_bl_members[h] = {"node": bl, "depth": bl.depth(),
                              "pos_labels": [], "center_words": []}
    ctx_bl_members[h]["pos_labels"].append(prim2id[pos])
    ctx_bl_members[h]["center_words"].append(webster.ltm.value_to_id[w])
print(f"  {len(ctx_bl_members)} unique BL nodes in context tree")

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
        n_rows, 3, figsize=(15, max(2.0, n_rows * 1.6)),
        squeeze=False, gridspec_kw={"width_ratios": [1.0, 1.4, 2.5]})
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
        ax0.set_ylim(0, 1.0); ax0.tick_params(axis="y", labelsize=5)
        ax0.set_ylabel(
            f"d={depth}\nn={n_mem}\ncnt={int(node.count)}\ndom={dom}",
            fontsize=6, rotation=0, labelpad=28, va="center")
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

        ax2 = axes[row, 2]; ax2.axis("off")
        ctx_top = _top_context_words(node, k=TOP_WORDS_PER_OFFSET)
        offsets = sorted(ctx_top.keys())
        if offsets:
            x_step = 1.0 / max(len(offsets), 1)
            for ci, off in enumerate(offsets):
                cx = (ci + 0.5) * x_step
                ax2.text(cx, 0.95, f"{off:+d}", ha="center", va="top",
                         fontsize=7, fontweight="bold", transform=ax2.transAxes)
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
    title=(f"CONTEXT TREE basic-level subtrees — "
           f"get_basic(use_root=True, eval_alpha={EVAL_ALPHA})  "
           f"(n_subtrees={len(ctx_bl_members)})"),
    out_path=os.path.join(CONTEXT_DIR, "basic_level_subtrees.png"))
print(f"  Context BL subtrees → {CONTEXT_DIR}/basic_level_subtrees.png")

with open(os.path.join(CONTEXT_DIR, "per_subtree_membership.csv"), "w") as f:
    w_ = csv.writer(f)
    w_.writerow(["subtree_idx", "depth", "node_count", "test_members",
                 "dominant_pos", "pos_distribution"])
    for i, m in enumerate(sorted(ctx_bl_members.values(),
                                 key=lambda m: len(m["pos_labels"]),
                                 reverse=True)):
        cnts = np.bincount(np.array(m["pos_labels"]), minlength=N_PRIM)
        dom = id2prim[int(cnts.argmax())]
        dist = "/".join(f"{id2prim[k]}:{int(c)}" for k, c in enumerate(cnts) if c > 0)
        w_.writerow([i, m["depth"], int(m["node"].count),
                     len(m["pos_labels"]), dom, dist])

# ── Tree-with-bars helpers (shared by ctx + cnt) ─────────────────────────────
def _make_layout(root, max_depth):
    all_nodes = [root]; children_of = {0: []}; depth_of = {0: 0}
    queue = [0]
    while queue:
        idx = queue.pop(0); node = all_nodes[idx]
        if depth_of[idx] < max_depth:
            for c in node.children:
                ci = len(all_nodes); all_nodes.append(c)
                children_of[idx].append(ci); children_of[ci] = []
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
                _assign(c, depth + 1, cur); cur += cs
    _assign(0, 0, 0.0)
    return pos, _leaf_span(0, 0)

from functools import lru_cache as _lru_cache
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

def compute_ctx_node_counts(root, test_primitives_data, max_depth):
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
                          highlight_idx, title, out_path, max_depth):
    children_of = _prune_empty(
        children_of, lambda i: i in counts and counts[i].sum() > 0)
    pos_map, total_w = _layout_x(children_of, max_depth)
    bar_w, bar_h, y_gap = 0.7, 0.35, 1.0
    fig, ax = plt.subplots(
        figsize=(max(14, total_w * 0.9), (max_depth + 1) * 2.2))
    ax.set_xlim(0, total_w); ax.set_ylim(-0.7, max_depth * y_gap + 0.7)
    ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=11)

    def _has(idx): return idx in counts and counts[idx].sum() > 0
    def _edges(idx, depth):
        if depth >= max_depth or not children_of[idx] or not _has(idx): return
        px, py = pos_map[idx]
        for c in children_of[idx]:
            if not _has(c): continue
            cx, cy = pos_map[c]
            ax.plot([px, cx],
                    [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.7, zorder=0)
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
            lw=(3.0 if is_bl else 0.4),
            zorder=(5 if is_bl else 2)))
        ax.text(x_c, depth * y_gap + bar_h / 2 + 0.05,
                f"n={int(total)}", ha="center", va="top", fontsize=5)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

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
    max_depth=TREE_DEPTH_FIG)
print(f"  Context tree fig → {CONTEXT_DIR}/cobweb_tree_labels.png")

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
    ax.set_xticks(depths); ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

_plot_score_by_depth(ctx_root, ctx_bl_members,
                     os.path.join(CONTEXT_DIR, "score_by_depth.png"),
                     "CONTEXT tree")
print(f"  Context score-by-depth → {CONTEXT_DIR}/score_by_depth.png")

with open(os.path.join(CONTEXT_DIR, "method_summary.txt"), "w") as f:
    f.write(f"CONTEXT tree — get_basic(use_root=True, eval_alpha={EVAL_ALPHA})\n")
    f.write("=" * 56 + "\n\n")
    f.write(f"  Test primitives: {len(test_primitives)}\n")
    f.write(f"  Unique BL nodes: {len(ctx_bl_members)}\n\n")
    for i, m in enumerate(sorted(ctx_bl_members.values(),
                                 key=lambda m: len(m["pos_labels"]),
                                 reverse=True)):
        cnts = np.bincount(np.array(m["pos_labels"]), minlength=N_PRIM)
        dom = id2prim[int(cnts.argmax())]
        f.write(f"  [{i:>2}] depth={m['depth']:>2}  count={int(m['node'].count):>5}  "
                f"members={len(m['pos_labels']):>4}  dom={dom}\n")

# ── CONTENT-tree inspection ──────────────────────────────────────────────────
cnt_root = webster.ltm.content_hierarchy.root
cnt_bl_cache: dict = {}
cnt_bl_members: dict = {}
chunk_records: list = []
for hollow in train_hollow:
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
            cnt_bl_members[h] = {"node": bl, "depth": bl.depth(),
                                  "self_cls": [], "L_cls": [], "R_cls": [],
                                  "tokens_list": []}
        cnt_bl_members[h]["self_cls"].append(cls)
        cnt_bl_members[h]["L_cls"].append(l_cls or "OTHER")
        cnt_bl_members[h]["R_cls"].append(r_cls or "OTHER")
        cnt_bl_members[h]["tokens_list"].append(_chunk_tokens(comp))
        s_idx, e_idx = _chunk_span(comp)
        chunk_records.append({
            "sentence": sentence,
            "span": (s_idx, e_idx),
            "tokens": sent_toks[s_idx:e_idx+1],
            "pos_yield": _chunk_yield(comp),
            "class": cls, "L_class": l_cls, "R_class": r_cls,
            "leaf_hash": str(leaf.concept_hash()),
            "bl_hash": h,
            "content_instance": ci,
        })
print(f"  {len(chunk_records)} supervised chunks → {len(cnt_bl_members)} BL nodes")

cls_dist = Counter(r["class"] for r in chunk_records)
print(f"  Chunk class distribution (head-based, S=root):")
for cls in CHUNK_LABELS + ["OTHER"]:
    if cls_dist.get(cls, 0) > 0:
        print(f"    {cls:>5}: {cls_dist[cls]:>4}")

leaf_classes: dict = defaultdict(Counter)
for r in chunk_records:
    leaf_classes[r["leaf_hash"]][r["class"]] += 1
class_match = Counter(); class_total = Counter()
for r in chunk_records:
    cls = r["class"]
    dom = leaf_classes[r["leaf_hash"]].most_common(1)[0][0]
    class_total[cls] += 1
    if dom == cls: class_match[cls] += 1
print(f"  Per-class clustering purity:")
for cls in CHUNK_LABELS:
    t = class_total.get(cls, 0)
    if t == 0: continue
    print(f"    {cls:>5}: {class_match[cls]}/{t} ({100*class_match[cls]/t:.1f}%)")

def _top_per_side_attrs(node, k=TOP_CTX_NODES):
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
        n_rows, 3, figsize=(16, max(2.4, n_rows * 1.9)),
        squeeze=False, gridspec_kw={"width_ratios": [1.6, 1.8, 2.2]})
    fig.suptitle(title, fontsize=11)
    for row, m in enumerate(sorted_bls):
        node = m["node"]
        n_mem = len(m["self_cls"]); depth = m["depth"]
        self_cnt = Counter(m["self_cls"])
        dom_self = self_cnt.most_common(1)[0][0]

        ax0 = axes[row, 0]
        used_l = sorted(set(m["L_cls"]), key=lambda x: label2id.get(x, 99))
        used_r = sorted(set(m["R_cls"]), key=lambda x: label2id.get(x, 99))
        sub = np.zeros((len(used_l), len(used_r)), dtype=np.int32)
        for L, R in zip(m["L_cls"], m["R_cls"]):
            if L in used_l and R in used_r:
                sub[used_l.index(L), used_r.index(R)] += 1
        ax0.imshow(sub / max(sub.sum(), 1), cmap="Blues",
                   vmin=0, vmax=1, aspect="equal")
        ax0.set_xticks(range(len(used_r))); ax0.set_yticks(range(len(used_l)))
        ax0.set_xticklabels(used_r, rotation=45, ha="right", fontsize=6)
        ax0.set_yticklabels(used_l, fontsize=6)
        ax0.set_xlabel("R class", fontsize=6)
        ax0.set_ylabel(
            f"d={depth}\nn={n_mem}\ncnt={int(node.count)}\n"
            f"self={dom_self}\n\nL class",
            fontsize=6, rotation=0, labelpad=46, va="center")
        if row == 0: ax0.set_title("(L,R) class joint", fontsize=8)

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
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()

plot_bl_subtrees_chunk(
    cnt_bl_members,
    title=(f"CONTENT TREE basic-level subtrees — "
           f"get_basic(use_root=True, eval_alpha={EVAL_ALPHA})  "
           f"(N_chunks={len(chunk_records)}, n_subtrees={len(cnt_bl_members)})"),
    out_path=os.path.join(CONTENT_DIR, "basic_level_subtrees.png"))
print(f"  Content BL subtrees → {CONTENT_DIR}/basic_level_subtrees.png")

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

def compute_cnt_node_counts(root, chunk_records, max_depth):
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
                        highlight_idx, title, out_path, max_depth):
    children_of = _prune_empty(
        children_of, lambda i: i in cL and cL[i].sum() > 0)
    pos_map, total_w = _layout_x(children_of, max_depth)
    bar_w, bar_h, gap, y_unit = 0.7, 0.18, 0.05, 1.0
    fig, ax = plt.subplots(
        figsize=(max(14, total_w * 0.9), (max_depth + 1) * 2.4))
    ax.set_xlim(0, total_w); ax.set_ylim(-0.7, max_depth * y_unit + 0.7)
    ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=11)

    def _has(idx): return idx in cL and cL[idx].sum() > 0
    def _edges(idx, depth):
        if depth >= max_depth or not children_of[idx] or not _has(idx): return
        px, py = pos_map[idx]
        for c in children_of[idx]:
            if not _has(c): continue
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
                ha="center", va="bottom", fontsize=5)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

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
    max_depth=TREE_DEPTH_FIG)
print(f"  Content tree fig → {CONTENT_DIR}/content_tree_labels.png")

_plot_score_by_depth(cnt_root, cnt_bl_members,
                     os.path.join(CONTENT_DIR, "score_by_depth.png"),
                     "CONTENT tree")
print(f"  Content score-by-depth → {CONTENT_DIR}/score_by_depth.png")

with open(os.path.join(CONTENT_DIR, "method_summary.txt"), "w") as f:
    f.write(f"CONTENT tree — get_basic(use_root=True, eval_alpha={EVAL_ALPHA})\n")
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

# =============================================================================
# PHASE 1 — HEURISTIC LOGGING (structure-based gold flag)
# =============================================================================
# For each test hollow sentence, walk the gold merge sequence.  At
# every step, evaluate EVERY parentless pair (the candidates WEBSTER
# would actually score during automatic parsing) and store its full
# heuristic vector with an ``is_gold`` flag.
print("\n=== PHASE 1: HEURISTIC LOGGING ===")

def _extract_heuristics(res):
    """Pull every non-basic-level heuristic out of evaluate_pair's score
    dicts and add a handful of natural compositions."""
    cnt = res.get("content_score_data", {})
    ctx = res.get("context_score_data", {})
    row = {
        # ── Raw (8) ───────────────────────────────────────────────────
        "cnt_tree_lp":       float(cnt.get("tree_log_prob",       -1e9)),
        "cnt_tree_class_lp": float(cnt.get("tree_class_log_prob", -1e9)),
        "cnt_root_lp":       float(cnt.get("root_log_prob",       -1e9)),
        "cnt_leaf_lp":       float(cnt.get("leaf_log_prob",       -1e9)),
        "ctx_tree_lp":       float(ctx.get("tree_log_prob",       -1e9)),
        "ctx_tree_class_lp": float(ctx.get("tree_class_log_prob", -1e9)),
        "ctx_root_lp":       float(ctx.get("root_log_prob",       -1e9)),
        "ctx_leaf_lp":       float(ctx.get("leaf_log_prob",       -1e9)),
    }
    # ── Joint sums across content + context ──────────────────────────
    row["sum_tree_lp"]   = row["cnt_tree_lp"]       + row["ctx_tree_lp"]
    row["sum_class_lp"]  = row["cnt_tree_class_lp"] + row["ctx_tree_class_lp"]
    row["sum_leaf_lp"]   = row["cnt_leaf_lp"]       + row["ctx_leaf_lp"]
    # ── "Above prior" — how much the leaf/tree path's log-prob exceeds
    # the root marginal.  Captures specificity without any basic-level
    # detector.
    row["cnt_tree_above_root"] = row["cnt_tree_lp"]  - row["cnt_root_lp"]
    row["ctx_tree_above_root"] = row["ctx_tree_lp"]  - row["ctx_root_lp"]
    row["sum_tree_above_root"] = (row["cnt_tree_above_root"]
                                  + row["ctx_tree_above_root"])
    row["cnt_leaf_above_root"] = row["cnt_leaf_lp"]  - row["cnt_root_lp"]
    row["ctx_leaf_above_root"] = row["ctx_leaf_lp"]  - row["ctx_root_lp"]
    # ── Min / max across the two trees (conjunctive vs disjunctive) ──
    row["min_tree_lp"]   = min(row["cnt_tree_lp"],       row["ctx_tree_lp"])
    row["max_tree_lp"]   = max(row["cnt_tree_lp"],       row["ctx_tree_lp"])
    row["min_class_lp"]  = min(row["cnt_tree_class_lp"], row["ctx_tree_class_lp"])
    row["max_class_lp"]  = max(row["cnt_tree_class_lp"], row["ctx_tree_class_lp"])
    return row

# ── Gold-bracket reconstruction ─────────────────────────────────────────────
# The MERGE ORDER in a hollow tree is one specific traversal of the
# parse-tree. Any traversal that produces the same final bracket
# structure is equally correct — chunk A's creation does not interact
# with chunk B's creation when their spans don't overlap. So we
# evaluate by *structure*, not by step order:
#   gold_brackets = { (start, end) for every supervised composite span }
# A candidate (left, right) is GOLD-valid at step t iff its merged
# span is in gold_brackets. Multiple gold-valid candidates can coexist
# at one step (they correspond to non-interacting chunks); any of them
# is an acceptable pick.
def _gold_brackets_from_merges(sentence, merges):
    tokens = sentence.split(); n = len(tokens)
    centers = [float(i) for i in range(n)]
    spans   = [(i, i)      for i in range(n)]
    out = set()
    for m in merges:
        try:
            li = centers.index(m["left"]); ri = centers.index(m["right"])
        except ValueError: return out
        if abs(li - ri) != 1: return out
        a, b = (li, ri) if li < ri else (ri, li)
        new_span = (spans[a][0], spans[b][1])
        out.add(new_span)
        centers[a:b+1] = [(centers[a] + centers[b]) / 2.0]
        spans[a:b+1]   = [new_span]
    return out

candidate_log: list = []   # one row per (sentence, step, candidate)
step_picks:    list = []   # one row per (sentence, step)
test_sentence_records: dict = {}  # sentence → list of per-step records (for viz)

for hollow in test_hollow:
    sentence = hollow["sentence"]
    gold_merges  = hollow["merges"]
    gold_brackets = _gold_brackets_from_merges(sentence, gold_merges)
    sent_steps: list = []
    test_sentence_records[sentence] = sent_steps

    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold="converge")

    # Parallel state alongside `tree`: at each step, ``centers[k]`` is
    # the center position of the k-th parentless node and ``spans[k]``
    # is its (start_token_idx, end_token_idx) inclusive span. Both
    # arrays update as we advance via the gold merges.
    n_tokens = len(sentence.split())
    centers = [float(i) for i in range(n_tokens)]
    spans   = [(i, i)    for i in range(n_tokens)]

    for step_idx, gold in enumerate(gold_merges):
        gold_l = gold["left"]; gold_r = gold["right"]
        pairs  = tree.get_parentless_pairs()

        step_candidates = []
        gold_valid_spans = []
        for p in pairs:
            li = p["left_word_index"]; ri = p["right_word_index"]
            try:
                a = centers.index(li); b = centers.index(ri)
            except ValueError: continue
            if a > b: a, b = b, a
            merged_span = (spans[a][0], spans[b][1])
            is_gold = merged_span in gold_brackets
            try:
                res = tree.evaluate_pair(li, ri, debug=False)
            except Exception:
                continue
            heurs = _extract_heuristics(res)
            row = {
                "sentence": sentence, "step": step_idx,
                "left": li, "right": ri,
                "merged_span": merged_span,
                "is_gold": is_gold,
                **heurs,
            }
            candidate_log.append(row)
            step_candidates.append(row)
            if is_gold:
                gold_valid_spans.append(merged_span)

        # Per-heuristic argmax picks at this step.  ``match_gold``
        # is now "is the argmax's merged span in gold_brackets?" —
        # i.e. ANY structurally-valid merge counts as a hit.
        step_pick = {
            "sentence": sentence, "step": step_idx,
            "gold_left": gold_l, "gold_right": gold_r,
            "gold_valid_spans": gold_valid_spans,
            "n_pairs":  len(step_candidates),
            "n_gold_valid": len(gold_valid_spans),
        }
        for h in HEURISTICS:
            if not step_candidates:
                step_pick[h + "_match_gold"] = None
                step_pick[h + "_picked"]     = None
                continue
            best = max(step_candidates, key=lambda r: r[h])
            step_pick[h + "_match_gold"] = best["is_gold"]
            step_pick[h + "_picked"]     = (best["left"], best["right"])
        step_picks.append(step_pick)
        sent_steps.append({
            "step":       step_idx,
            "candidates": step_candidates,
            "gold":       (gold_l, gold_r),
            "step_pick":  step_pick,
        })

        # Advance using the gold merge AND mirror the same update to
        # the parallel (centers, spans) state.
        try:
            tree.apply_candidate(gold_l, gold_r)
        except Exception:
            break
        try:
            a = centers.index(gold_l); b = centers.index(gold_r)
            if a > b: a, b = b, a
            new_c    = (centers[a] + centers[b]) / 2.0
            new_span = (spans[a][0], spans[b][1])
            centers[a:b+1] = [new_c]
            spans[a:b+1]   = [new_span]
        except ValueError:
            break

print(f"  Logged {len(candidate_log)} candidate pair evaluations "
      f"across {len(step_picks)} merge steps "
      f"from {len(test_hollow)} test sentences")

# Persist the raw log so the user can re-mine it.
with open(os.path.join(OUT_DIR, "candidate_log.csv"), "w") as f:
    w = csv.writer(f)
    cols = ["sentence", "step", "left", "right",
            "span_start", "span_end", "is_gold"] + HEURISTICS
    w.writerow(cols)
    for r in candidate_log:
        s, e = r.get("merged_span", (None, None))
        w.writerow([r["sentence"], r["step"], r["left"], r["right"],
                    s, e, int(r["is_gold"])]
                   + [f"{r[h]:.4f}" for h in HEURISTICS])

with open(os.path.join(OUT_DIR, "step_picks.csv"), "w") as f:
    w = csv.writer(f)
    cols = ["sentence", "step", "gold_left", "gold_right",
            "n_pairs", "n_gold_valid"]
    for h in HEURISTICS:
        cols += [f"{h}_match_gold", f"{h}_picked_left", f"{h}_picked_right"]
    w.writerow(cols)
    for s in step_picks:
        row = [s["sentence"], s["step"], s["gold_left"], s["gold_right"],
               s["n_pairs"], s.get("n_gold_valid", 0)]
        for h in HEURISTICS:
            row.append("" if s.get(h + "_match_gold") is None
                       else int(s[h + "_match_gold"]))
            pick = s.get(h + "_picked")
            row += ["", ""] if pick is None else [pick[0], pick[1]]
        w.writerow(row)

# =============================================================================
# PHASE 2 — PER-HEURISTIC DISCRIMINABILITY
# =============================================================================
print("\n=== PHASE 2: Per-heuristic discriminability (gold vs non-gold) ===")

y_true = np.array([int(r["is_gold"]) for r in candidate_log])
heuristic_auc: dict = {}
heuristic_ap:  dict = {}

# Histograms.
ncol = 4
nrow = (len(HEURISTICS) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
axes = axes.flatten()
for ax, h in zip(axes, HEURISTICS):
    gold_v     = [r[h] for r in candidate_log if r["is_gold"]]
    non_gold_v = [r[h] for r in candidate_log if not r["is_gold"]]
    if not gold_v or not non_gold_v:
        ax.axis("off"); continue
    all_vals = gold_v + non_gold_v
    lo, hi   = float(np.min(all_vals)), float(np.max(all_vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        ax.axis("off"); continue
    bins = np.linspace(lo, hi, 30)
    ax.hist(non_gold_v, bins=bins, alpha=0.5, color="#d62728",
            density=True, label=f"non-gold (n={len(non_gold_v)})")
    ax.hist(gold_v,     bins=bins, alpha=0.5, color="#2ca02c",
            density=True, label=f"gold (n={len(gold_v)})")
    ax.set_title(h, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best")
for ax in axes[len(HEURISTICS):]:
    ax.axis("off")
plt.suptitle("Heuristic distributions — gold vs non-gold candidate pairs",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUT_DIR, "heuristic_histograms.png"), dpi=120)
plt.close()

# ROC curves.
fig, ax = plt.subplots(figsize=(9, 7))
for h, color in zip(HEURISTICS, HEUR_COLORS):
    y_score = np.array([r[h] for r in candidate_log])
    finite_mask = np.isfinite(y_score)
    if finite_mask.sum() == 0 or len(set(y_true[finite_mask])) < 2:
        continue
    fpr, tpr, _ = roc_curve(y_true[finite_mask], y_score[finite_mask])
    a = auc(fpr, tpr)
    heuristic_auc[h] = a
    ax.plot(fpr, tpr, color=color, lw=1.6, label=f"{h} (AUC={a:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC — gold vs non-gold pair, by heuristic")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heuristic_roc.png"), dpi=120)
plt.close()

# Precision-Recall curves.
fig, ax = plt.subplots(figsize=(9, 7))
for h, color in zip(HEURISTICS, HEUR_COLORS):
    y_score = np.array([r[h] for r in candidate_log])
    finite_mask = np.isfinite(y_score)
    if finite_mask.sum() == 0 or len(set(y_true[finite_mask])) < 2:
        continue
    p, r, _ = precision_recall_curve(y_true[finite_mask], y_score[finite_mask])
    ap = average_precision_score(y_true[finite_mask], y_score[finite_mask])
    heuristic_ap[h] = ap
    ax.plot(r, p, color=color, lw=1.6, label=f"{h} (AP={ap:.3f})")
gold_base = y_true.mean()
ax.axhline(gold_base, color="black", linestyle=":", alpha=0.5,
           label=f"gold base rate ({gold_base:.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall — gold vs non-gold pair, by heuristic")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heuristic_pr.png"), dpi=120)
plt.close()

print(f"  Per-heuristic AUC (gold vs non-gold candidate):")
for h, a in sorted(heuristic_auc.items(), key=lambda x: -x[1]):
    print(f"    {h:>22s}: AUC={a:.3f}  AP={heuristic_ap.get(h, float('nan')):.3f}")

# =============================================================================
# PHASE 3 — STEP-PICK ACCURACY
# =============================================================================
# At each step the parser only commits to ONE merge.  For each
# heuristic we check: does its argmax over the candidates equal the
# gold merge for that step?
print("\n=== PHASE 3: Step-pick accuracy (argmax matches gold) ===")
heuristic_step_acc: dict = {}
for h in HEURISTICS:
    key = h + "_match_gold"
    valid = [s for s in step_picks if s.get(key) is not None]
    matches = sum(1 for s in valid if s[key])
    total = len(valid)
    heuristic_step_acc[h] = (matches, total, matches / max(total, 1))

print(f"  {'heuristic':>22s}  {'matches':>8s}  {'total':>6s}  {'acc':>7s}")
for h, (m, t, a) in sorted(heuristic_step_acc.items(),
                            key=lambda kv: -kv[1][2]):
    print(f"  {h:>22s}  {m:>8d}  {t:>6d}  {100*a:>6.1f}%")

# Bar chart.
fig, ax = plt.subplots(figsize=(11, 5))
sorted_items = sorted(heuristic_step_acc.items(), key=lambda kv: -kv[1][2])
names = [h for h, _ in sorted_items]
accs  = [v[2] for _, v in sorted_items]
bars = ax.bar(names, accs, color=HEUR_COLORS[:len(names)])
for i, (n, a) in enumerate(zip(names, accs)):
    ax.text(i, a + 0.01, f"{100*a:.1f}%", ha="center", fontsize=8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0, 1.1); ax.set_ylabel("Step-pick accuracy")
ax.set_title("argmax-heuristic step-pick accuracy "
             "(% of merge steps where the heuristic picks gold)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step_pick_accuracy.png"), dpi=120)
plt.close()

# =============================================================================
# PHASE 4 — PER-HEURISTIC QUANTILE-THRESHOLD SWEEP
# =============================================================================
# For each candidate gate-heuristic, sweep its threshold across quantiles
# of the OBSERVED non-gold distribution (0 = no gate, 0.95 = strict),
# rank by argmax of that same heuristic, and report acc-at-admit /
# coverage / overall acc per quantile.  This is the non-basic-level
# equivalent of the old cnt_bl_count > τ sweep.
print("\n=== PHASE 4: Per-heuristic quantile-threshold sweep ===")
quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
sweep_rows = []

by_step = defaultdict(list)
for r in candidate_log:
    by_step[(r["sentence"], r["step"])].append(r)
total_steps = len(by_step)

# Cache non-gold quantiles per heuristic.
non_gold_quantiles = {}
for h in HEURISTICS:
    vals = [r[h] for r in candidate_log if not r["is_gold"]
            and np.isfinite(r[h])]
    if not vals:
        non_gold_quantiles[h] = {q: float("-inf") for q in quantiles}
        continue
    non_gold_quantiles[h] = {q: float(np.quantile(vals, q)) for q in quantiles}

# Sweep each heuristic.
for h in HEURISTICS:
    for q in quantiles:
        tau = non_gold_quantiles[h][q]
        matches = 0; admit_steps = 0
        for cands in by_step.values():
            admitted = [c for c in cands if c[h] > tau]
            if not admitted: continue
            admit_steps += 1
            best = max(admitted, key=lambda c: c[h])
            if best["is_gold"]: matches += 1
        sweep_rows.append({
            "heur": h, "q": q, "tau": tau,
            "matches": matches, "admit_steps": admit_steps,
            "acc_at_admit": matches / max(admit_steps, 1),
            "coverage":     admit_steps / max(total_steps, 1),
            "overall_acc":  matches / max(total_steps, 1),
        })

# Per-heuristic best overall accuracy.
best_per_heur = {}
for r in sweep_rows:
    h = r["heur"]
    if h not in best_per_heur or r["overall_acc"] > best_per_heur[h]["overall_acc"]:
        best_per_heur[h] = r
print(f"  Best (quantile, overall acc) per gate-heuristic — "
      f"gate and rank are the SAME heuristic:")
print(f"  {'heuristic':>22s}  {'best q':>6s}  {'overall':>8s}  "
      f"{'admit':>7s}  {'cov':>6s}")
for h, r in sorted(best_per_heur.items(),
                    key=lambda kv: -kv[1]["overall_acc"]):
    print(f"  {h:>22s}  {r['q']:>6.2f}  {100*r['overall_acc']:>7.1f}%  "
          f"{100*r['acc_at_admit']:>6.1f}%  {100*r['coverage']:>5.1f}%")

with open(os.path.join(OUT_DIR, "threshold_sweep.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["heuristic", "quantile", "tau",
                "matches", "admit_steps",
                "acc_at_admit", "coverage", "overall_acc"])
    for r in sweep_rows:
        w.writerow([r["heur"], f"{r['q']:.3f}", f"{r['tau']:.4f}",
                    r["matches"], r["admit_steps"],
                    f"{r['acc_at_admit']:.4f}",
                    f"{r['coverage']:.4f}",
                    f"{r['overall_acc']:.4f}"])

# Plot: curves of overall_acc vs quantile, one per heuristic.
fig, ax = plt.subplots(figsize=(12, 6))
for h, color in zip(HEURISTICS, HEUR_COLORS):
    rows_h = [r for r in sweep_rows if r["heur"] == h]
    rows_h.sort(key=lambda r: r["q"])
    ax.plot([r["q"] for r in rows_h],
            [r["overall_acc"] for r in rows_h],
            "o-", color=color, lw=1.3, label=h)
ax.set_xlabel("Non-gold-distribution quantile (τ = q-th percentile of non-gold)")
ax.set_ylabel("Overall step-pick accuracy")
ax.set_title("Per-heuristic quantile-threshold sweep "
             "(gate = rank = same heuristic)")
ax.legend(fontsize=7, loc="best", ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "threshold_sweep.png"), dpi=120,
            bbox_inches="tight")
plt.close()

# Summary table CSV.
with open(os.path.join(OUT_DIR, "heuristic_summary.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["heuristic", "roc_auc", "pr_ap",
                "step_pick_acc", "step_pick_matches", "step_pick_total"])
    for h in HEURISTICS:
        a = heuristic_auc.get(h, float("nan"))
        ap = heuristic_ap.get(h, float("nan"))
        m, t, ac = heuristic_step_acc[h]
        w.writerow([h, f"{a:.4f}", f"{ap:.4f}",
                    f"{ac:.4f}", m, t])

# =============================================================================
# PHASE 4b — GATE × RANK COMBINATORIAL SWEEP
# =============================================================================
# The current policy is "gate by cnt_bl_count > τ, rank by cnt_tree_lp
# argmax".  But Phase 2 says ctx_tree_lp is the best gold-vs-non-gold
# discriminator (AUC 0.80) and Phase 3 says cnt_bl_count is the best
# argmax-picker (51%).  So the optimal policy might use one heuristic
# for the gate and a DIFFERENT one for the rank.
#
# We sweep every (gate_heur, rank_heur) pair and report the best
# overall step-pick accuracy.  For the gate we use a quantile-based
# threshold (top X% of non-gold candidates rejected) so it's
# comparable across heuristics with different units.
print("\n=== PHASE 4b: Gate × Rank combinatorial sweep ===")
GATE_QUANTILES = [0.0, 0.25, 0.5, 0.75]   # 0 = no gate, .75 = strict
gate_rank_rows = []

# Pre-compute per-step candidate lists grouped by (sentence, step).
steps_grouped = defaultdict(list)
for r in candidate_log:
    steps_grouped[(r["sentence"], r["step"])].append(r)
n_total_steps = len(steps_grouped)

for gate_h in HEURISTICS + ["NONE"]:
    # Pre-compute the gate threshold over the non-gold distribution.
    if gate_h == "NONE":
        gate_taus = [float("-inf")]
    else:
        non_gold = [r[gate_h] for r in candidate_log if not r["is_gold"]
                    and np.isfinite(r[gate_h])]
        if not non_gold:
            gate_taus = [float("-inf")]
        else:
            gate_taus = [float(np.quantile(non_gold, q))
                         for q in GATE_QUANTILES]
    for q_idx, gate_tau in enumerate(gate_taus):
        q_label = GATE_QUANTILES[q_idx] if gate_h != "NONE" else "—"
        for rank_h in HEURISTICS:
            matches = 0; admit_steps = 0
            for cands in steps_grouped.values():
                if gate_h == "NONE":
                    admitted = cands
                else:
                    admitted = [c for c in cands if c[gate_h] > gate_tau]
                if not admitted: continue
                admit_steps += 1
                best = max(admitted, key=lambda c: c[rank_h])
                if best["is_gold"]: matches += 1
            overall = matches / max(n_total_steps, 1)
            gate_rank_rows.append({
                "gate_heur":  gate_h,
                "gate_q":     q_label,
                "gate_tau":   gate_tau,
                "rank_heur":  rank_h,
                "matches":    matches,
                "admit_steps":admit_steps,
                "coverage":   admit_steps / max(n_total_steps, 1),
                "acc_at_admit": matches / max(admit_steps, 1),
                "overall_acc":  overall,
            })

# Sort by overall accuracy.
gate_rank_rows.sort(key=lambda r: -r["overall_acc"])
print(f"  Top 15 (gate, rank) combinations by overall step-pick accuracy:")
print(f"  {'gate':>20s} {'q':>5s} {'rank':>20s}  {'overall':>8s}  "
      f"{'admit':>7s}  {'cov':>6s}")
for r in gate_rank_rows[:15]:
    print(f"  {r['gate_heur']:>20s} {str(r['gate_q']):>5s} "
          f"{r['rank_heur']:>20s}  {100*r['overall_acc']:>7.1f}%  "
          f"{100*r['acc_at_admit']:>6.1f}%  {100*r['coverage']:>5.1f}%")

with open(os.path.join(OUT_DIR, "gate_rank_sweep.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["gate_heur", "gate_quantile", "gate_tau",
                "rank_heur", "matches", "admit_steps",
                "coverage", "acc_at_admit", "overall_acc"])
    for r in gate_rank_rows:
        w.writerow([r["gate_heur"], r["gate_q"], f"{r['gate_tau']:.4f}",
                    r["rank_heur"], r["matches"], r["admit_steps"],
                    f"{r['coverage']:.4f}", f"{r['acc_at_admit']:.4f}",
                    f"{r['overall_acc']:.4f}"])

# Heatmap of (gate_heur × rank_heur) overall accuracy at q=0.5
# (median non-gold cutoff) — gives a single-quantile snapshot.
q_target = 0.5
mat_rows = HEURISTICS + ["NONE"]
mat_cols = HEURISTICS
M = np.full((len(mat_rows), len(mat_cols)), np.nan)
for r in gate_rank_rows:
    if r["gate_q"] != q_target and r["gate_heur"] != "NONE": continue
    if r["gate_heur"] == "NONE" and r["gate_q"] != "—": continue
    i = mat_rows.index(r["gate_heur"])
    j = mat_cols.index(r["rank_heur"])
    M[i, j] = r["overall_acc"]

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=0.7, aspect="auto")
ax.set_xticks(range(len(mat_cols)))
ax.set_xticklabels(mat_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(mat_rows)))
ax.set_yticklabels(mat_rows, fontsize=8)
ax.set_xlabel("Rank heuristic (argmax among admitted)")
ax.set_ylabel("Gate heuristic (> 50th-percentile-of-non-gold)")
ax.set_title("Gate × Rank policy — overall step-pick accuracy\n"
             "(gate at median non-gold value; NONE = no gate)")
for i in range(len(mat_rows)):
    for j in range(len(mat_cols)):
        if np.isnan(M[i, j]): continue
        ax.text(j, i, f"{100*M[i,j]:.0f}",
                ha="center", va="center",
                color="black" if M[i, j] > 0.35 else "white",
                fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gate_rank_heatmap.png"), dpi=140,
            bbox_inches="tight")
plt.close()

# =============================================================================
# PHASE 4c — WEIGHTED LINEAR COMBINATIONS (formula exploration)
# =============================================================================
# For each pair of base heuristics (h1, h2), sweep α ∈ {0, 0.1, …, 1}
# and compute score = α * z(h1) + (1 - α) * z(h2), where z(·) is
# z-score standardisation across the candidate log (so the two
# heuristics are on comparable scales before mixing).  For each
# blended score we measure ROC AUC and step-pick accuracy.  This is
# a discrete grid search over the space of linear two-heuristic
# combinations — picks up "mostly h1 but with a small h2 correction"
# style policies the pure single-heuristic and pure (gate, rank)
# sweeps can't see.
print("\n=== PHASE 4c: Weighted linear-combo formula sweep ===")

# z-score-standardise each base heuristic over the candidate log so
# mixing two of them gives meaningful results.
z_vals = {}
for h in HEURISTICS:
    arr = np.array([r[h] for r in candidate_log], dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    mu = np.nanmean(arr); sigma = np.nanstd(arr)
    if not np.isfinite(sigma) or sigma == 0: sigma = 1.0
    z_vals[h] = (arr - mu) / sigma

# Cache per-step candidate row indices so step-pick is cheap.
step_groups: list = []   # list of np.array(int) of indices into candidate_log
step_gold:   list = []   # list of int — index of the gold candidate within group
_step_index = defaultdict(list)
for idx, r in enumerate(candidate_log):
    _step_index[(r["sentence"], r["step"])].append(idx)
for (_sent, _stp), idxs in _step_index.items():
    arr = np.array(idxs, dtype=np.int64)
    step_groups.append(arr)
    gold_idx = [i for i in idxs if candidate_log[i]["is_gold"]]
    step_gold.append(gold_idx[0] if gold_idx else -1)

def _eval_score_vec(score_vec):
    """Given per-candidate scores, compute (AUC, step-pick acc)."""
    y_true = np.array([int(r["is_gold"]) for r in candidate_log])
    # AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, score_vec)
        au = auc(fpr, tpr)
    except Exception:
        au = float("nan")
    # step-pick: argmax per step group
    matches = 0; total = 0
    for arr, gold in zip(step_groups, step_gold):
        if gold < 0: continue
        total += 1
        best = arr[int(np.argmax(score_vec[arr]))]
        if best == gold: matches += 1
    step_acc = matches / max(total, 1)
    return au, step_acc

# Single-heuristic baselines (same as Phase 2/3 but on z-scored values
# — z-score is monotone so AUC is unchanged but useful for parity).
print("  Baseline single-heuristic scores (z-scored):")
single_scores = {}
for h in HEURISTICS:
    au, sp = _eval_score_vec(z_vals[h])
    single_scores[h] = (au, sp)

# Pairwise weighted combos.
import itertools
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
combo_rows = []
print(f"  Sweeping {len(list(itertools.combinations(HEURISTICS, 2)))} "
      f"heuristic pairs × {len(alphas)} α values...")
for h1, h2 in itertools.combinations(HEURISTICS, 2):
    for a in alphas:
        score = a * z_vals[h1] + (1.0 - a) * z_vals[h2]
        au, sp = _eval_score_vec(score)
        combo_rows.append({
            "h1": h1, "h2": h2, "alpha": a,
            "formula": f"{a:.1f}*z({h1}) + {1-a:.1f}*z({h2})",
            "auc": au, "step_acc": sp,
        })

# Single heuristics as rows for ranking.
for h, (au, sp) in single_scores.items():
    combo_rows.append({
        "h1": h, "h2": "—", "alpha": 1.0,
        "formula": f"z({h})",
        "auc": au, "step_acc": sp,
    })

# Sort by step_acc primary, AUC secondary.
combo_rows.sort(key=lambda r: (-r["step_acc"], -r["auc"]))
print(f"  Top 15 formulas by step-pick accuracy:")
print(f"  {'rank':>4s}  {'step':>6s}  {'AUC':>6s}  formula")
for i, r in enumerate(combo_rows[:15], 1):
    print(f"  {i:>4d}  {100*r['step_acc']:>5.1f}%  {r['auc']:>5.3f}  "
          f"{r['formula']}")

# Now sort by AUC.
auc_rank = sorted(combo_rows, key=lambda r: (-r["auc"], -r["step_acc"]))
print(f"  Top 15 formulas by ROC AUC:")
print(f"  {'rank':>4s}  {'AUC':>6s}  {'step':>6s}  formula")
for i, r in enumerate(auc_rank[:15], 1):
    print(f"  {i:>4d}  {r['auc']:>5.3f}  {100*r['step_acc']:>5.1f}%  "
          f"{r['formula']}")

# Persist.
with open(os.path.join(OUT_DIR, "weighted_combo_sweep.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["h1", "h2", "alpha", "formula", "auc", "step_acc"])
    for r in combo_rows:
        w.writerow([r["h1"], r["h2"], f"{r['alpha']:.2f}",
                    r["formula"], f"{r['auc']:.4f}",
                    f"{r['step_acc']:.4f}"])

# Plot: best alpha per (h1, h2) pair — heatmap of best step_acc.
best_pair_step = {}
for r in combo_rows:
    if r["h2"] == "—": continue
    key = (r["h1"], r["h2"])
    if key not in best_pair_step or r["step_acc"] > best_pair_step[key]["step_acc"]:
        best_pair_step[key] = r

M = np.full((len(HEURISTICS), len(HEURISTICS)), np.nan)
A = np.full((len(HEURISTICS), len(HEURISTICS)), np.nan)
for (h1, h2), r in best_pair_step.items():
    i = HEURISTICS.index(h1); j = HEURISTICS.index(h2)
    M[i, j] = r["step_acc"]
    M[j, i] = r["step_acc"]
    A[i, j] = r["alpha"]
    A[j, i] = 1.0 - r["alpha"]
# Diagonal = single-heuristic step-pick.
for i, h in enumerate(HEURISTICS):
    M[i, i] = single_scores[h][1]

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(M, cmap="RdYlGn", vmin=0.2, vmax=0.7, aspect="equal")
ax.set_xticks(range(len(HEURISTICS)))
ax.set_yticks(range(len(HEURISTICS)))
ax.set_xticklabels(HEURISTICS, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(HEURISTICS, fontsize=7)
ax.set_title("Best step-pick accuracy per weighted pair α·z(h1)+(1-α)·z(h2)\n"
             "(diagonal = single heuristic baseline)")
for i in range(len(HEURISTICS)):
    for j in range(len(HEURISTICS)):
        if np.isnan(M[i, j]): continue
        ax.text(j, i, f"{100*M[i,j]:.0f}", ha="center", va="center",
                color="black" if M[i, j] > 0.4 else "white", fontsize=6)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
             label="best step-pick accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "weighted_combo_heatmap.png"), dpi=140,
            bbox_inches="tight")
plt.close()

# =============================================================================
# PHASE 5 — PER-SENTENCE PARSE TREE VIZ WITH HEURISTIC ANNOTATIONS
# =============================================================================
# For up to N_TREE_VIZ test sentences, render a parse tree where every
# internal node carries the heuristic values that were observed when
# that merge step was being evaluated.  We mark whether each heuristic
# would have picked this exact merge as gold (✓) or not (✗).
print(f"\n=== PHASE 5: Per-sentence parse tree viz ({N_TREE_VIZ} sentences) ===")

# Walk gold merges symbolically to reconstruct the parse tree.
def _gold_tree(sentence, merges):
    tokens = sentence.split(); n = len(tokens)
    leaf_nodes = [{"label": tokens[i], "span": (i, i),
                    "children": [], "is_primitive": True,
                    "center": float(i)} for i in range(n)]
    centers = [float(i) for i in range(n)]
    nodes   = list(leaf_nodes)
    merge_nodes = []
    for step_idx, m in enumerate(merges):
        try:
            li = centers.index(m["left"])
            ri = centers.index(m["right"])
        except ValueError:
            break
        if abs(li - ri) != 1: break
        a, b = (li, ri) if li < ri else (ri, li)
        parent = {
            "label":   f"step{step_idx}",
            "span":    (nodes[a]["span"][0], nodes[b]["span"][1]),
            "children":[nodes[a], nodes[b]],
            "is_primitive": False,
            "step":    step_idx,
            "merge":   (m["left"], m["right"]),
            "center":  (centers[a] + centers[b]) / 2.0,
        }
        merge_nodes.append(parent)
        nodes[a:b+1]   = [parent]
        centers[a:b+1] = [parent["center"]]
    return nodes[0] if nodes else None, merge_nodes

def _draw_tree(root, sentence, step_records, out_path):
    """Layout the gold parse tree with merge nodes annotated."""
    if root is None: return
    # Index step_records by (left, right) center positions so we can
    # look up each merge's heuristic data.
    by_pos = {(s["candidates"][0]["sentence"], r["step"]): r["step_pick"]
              for r in step_records
              for s in [r]}
    # by-step lookup
    by_step = {r["step"]: r for r in step_records}

    # Layout: x = mean leaf position, y = depth from root.
    def depth(n):
        if n["is_primitive"]: return 0
        return 1 + max(depth(c) for c in n["children"])
    H = depth(root)

    def leaves(n):
        if n["is_primitive"]: return [n]
        out = []
        for c in n["children"]:
            out.extend(leaves(c))
        return out
    all_leaves = leaves(root)
    leaf_x = {id(l): i for i, l in enumerate(all_leaves)}

    def x_of(n):
        if n["is_primitive"]:
            return leaf_x[id(n)]
        return float(np.mean([x_of(c) for c in n["children"]]))
    def y_of(n):
        if n["is_primitive"]: return H
        return H - depth(n)

    n_leaves = len(all_leaves)
    fig_w = max(14, n_leaves * 2.0)
    fig_h = max(6, (H + 1) * 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-1, n_leaves)
    ax.set_ylim(-0.5, H + 0.8)
    ax.invert_yaxis(); ax.axis("off")
    ax.set_title(f"Parse tree — \"{sentence}\"\n"
                 "Annotated with heuristic values per merge "
                 "(✓ = this heuristic would pick this merge)",
                 fontsize=11)

    # Draw edges.
    def _draw_edges(n):
        if n["is_primitive"]: return
        px, py = x_of(n), y_of(n)
        for c in n["children"]:
            cx, cy = x_of(c), y_of(c)
            ax.plot([px, cx], [py + 0.2, cy - 0.2],
                    color="gray", lw=0.8, zorder=0)
            _draw_edges(c)
    _draw_edges(root)

    # Draw primitives.
    for i, l in enumerate(all_leaves):
        ax.text(i, H + 0.25, l["label"], ha="center", va="top",
                fontsize=10, fontweight="bold")
        ax.add_patch(plt.Rectangle((i - 0.35, H - 0.1), 0.7, 0.2,
                                    facecolor="#cccccc", edgecolor="black"))

    # Draw internal merge nodes with annotations.
    def _draw_merges(n):
        if n["is_primitive"]: return
        nx, ny = x_of(n), y_of(n)
        step_idx = n["step"]
        sp = by_step.get(step_idx, {})
        pick = sp.get("step_pick", {})
        n_pairs = pick.get("n_pairs", "?")
        # Find this merge's heuristic values from candidates.
        gold_cand = next(
            (c for c in sp.get("candidates", [])
             if c["left"] == n["merge"][0] and c["right"] == n["merge"][1]),
            None)
        # Layout the annotation box.
        text_lines = [
            f"step {step_idx}   ({n_pairs} candidates)",
        ]
        if gold_cand:
            for h in HEURISTICS:
                ok = pick.get(h + "_match_gold", False)
                mark = "✓" if ok else "✗"
                text_lines.append(f"{mark} {h:<18s} = {gold_cand[h]:.3f}")
        box_w = 0.95; n_lines = len(text_lines)
        line_h = 0.045
        box_h  = n_lines * line_h + 0.06
        box_top = ny - box_h
        # bg
        ax.add_patch(plt.Rectangle(
            (nx - box_w/2, box_top), box_w, box_h,
            facecolor="#ffffff", edgecolor="black", lw=0.6, zorder=3))
        # header
        ax.text(nx, box_top + 0.03, text_lines[0],
                ha="center", va="top", fontsize=7, fontweight="bold",
                zorder=4)
        for li, line in enumerate(text_lines[1:]):
            mark, rest = line[0], line[2:]
            color = "#2ca02c" if mark == "✓" else "#d62728"
            ax.text(nx, box_top + 0.05 + (li + 1) * line_h, line,
                    ha="center", va="top", fontsize=6,
                    color=color, family="monospace", zorder=4)
        for c in n["children"]:
            _draw_merges(c)
    _draw_merges(root)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()

# Render a sample of test sentences.
viz_pool = list(test_hollow)
random.shuffle(viz_pool)
viz_sample = viz_pool[:N_TREE_VIZ]
for i, hollow in enumerate(viz_sample):
    sentence = hollow["sentence"]
    sent_steps = test_sentence_records.get(sentence, [])
    if not sent_steps: continue
    root, _ = _gold_tree(sentence, hollow["merges"])
    safe_name = sentence.replace(" ", "_")[:60]
    out_path = os.path.join(TREE_DIR, f"{i:02d}_{safe_name}.png")
    _draw_tree(root, sentence, sent_steps, out_path)
    print(f"  [{i+1}/{len(viz_sample)}] → {os.path.basename(out_path)}")

# =============================================================================
# PHASE 6 — NEGATIVE TEST: random word strings should NOT parse
# =============================================================================
# A real recognition system has to be SELECTIVE — high chunk-count for
# grammatical sentences, low chunk-count for random word strings.
#
# A single THRESHOLD value can't tell us whether the parser is
# selective or not — at low thresholds everything merges, at high
# thresholds nothing does.  What we need is the **threshold sweep**:
# vary the gate value and find the regime where:
#
#     grammar_chunks ≈ n_words - 1    (full tree on valid inputs)
#     random_chunks   ≈ 0              (random inputs rejected)
#
# The ratio  grammar_mean / random_mean  is the SELECTIVITY ratio.
# A working gate produces a clear peak in selectivity around some
# threshold range; the curves below identify it.
print(f"\n=== PHASE 6: NEGATIVE TEST (threshold sweep) ===")

N_NEG_RANDOM   = 50         # random ungrammatical strings
N_NEG_GRAMMAR  = 50         # grammatical comparison set
NEG_LEN_LO     = 4
NEG_LEN_HI     = 8

# Sweep grid spans the realistic range of basic_level_count values
# WEBSTER's count-gate sees: 0 (everything passes) up to 500+
# (essentially nothing passes).  Picked to cover the typical hollow-
# corpus count distribution.
NEG_THRESHOLDS = [-1, 0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

def _chunks_formed(parse_tree):
    """Count composite (non-primitive, non-root) nodes in a parse tree."""
    out = 0
    def w(n):
        nonlocal out
        if isinstance(n, PrimitiveParseNode): return
        if not getattr(n, "is_global_root", False): out += 1
        for _, c in getattr(n, "children", []): w(c)
    w(parse_tree.global_root_node)
    return out

# Random ungrammatical strings — uniform draws from the lexicon.
neg_random_sents = []
for _ in range(N_NEG_RANDOM):
    n = random.randint(NEG_LEN_LO, NEG_LEN_HI)
    neg_random_sents.append(" ".join(random.choice(TEST_CORPUS1) for _ in range(n)))

# Grammatical comparison set — same length distribution.
neg_grammar_sents = []
attempts = 0
while len(neg_grammar_sents) < N_NEG_GRAMMAR and attempts < N_NEG_GRAMMAR * 50:
    attempts += 1
    s = generate("S", TEST_GRAMMAR1)
    n = len(s.split())
    if NEG_LEN_LO <= n <= NEG_LEN_HI:
        neg_grammar_sents.append(s)
print(f"  Random sentences (~ungrammatical): {len(neg_random_sents)}")
print(f"  Grammatical sentences:             {len(neg_grammar_sents)}")

def _parse_and_count(sentences, thr):
    out_chunks, out_n_words = [], []
    for s in sentences:
        try:
            pt = webster.parse_sentence(s, threshold=thr,
                                         new_vocab=False, learning=False,
                                         debug=False)
            out_chunks.append(_chunks_formed(pt))
            out_n_words.append(len(s.split()))
        except Exception:
            continue
    return out_chunks, out_n_words

# Sweep — parse both populations at each threshold.
sweep_rows = []
print(f"\n  {'threshold':>9}  {'rand μ':>7}  {'gram μ':>7}  "
      f"{'rand/(n-1)':>11}  {'gram/(n-1)':>11}  {'select':>7}  "
      f"{'%rand=0':>8}")
for thr in NEG_THRESHOLDS:
    rand_c, rand_n = _parse_and_count(neg_random_sents,  thr)
    gram_c, gram_n = _parse_and_count(neg_grammar_sents, thr)
    if not rand_c or not gram_c: continue
    rand_mean = float(np.mean(rand_c))
    gram_mean = float(np.mean(gram_c))
    rand_norm = float(np.mean([c / max(n - 1, 1) for c, n in zip(rand_c, rand_n)]))
    gram_norm = float(np.mean([c / max(n - 1, 1) for c, n in zip(gram_c, gram_n)]))
    selectivity = gram_mean / max(rand_mean, 1e-9)
    zero_rand   = float(np.mean([1.0 if c == 0 else 0.0 for c in rand_c]))
    sweep_rows.append({
        "threshold":  thr,
        "rand_mean":  rand_mean, "gram_mean":  gram_mean,
        "rand_norm":  rand_norm, "gram_norm":  gram_norm,
        "selectivity": selectivity, "zero_rand": zero_rand,
        "rand_chunks": rand_c, "gram_chunks": gram_c,
        "rand_n":      rand_n,  "gram_n":      gram_n,
    })
    print(f"  {thr:>9}  {rand_mean:>7.2f}  {gram_mean:>7.2f}  "
          f"{rand_norm:>10.3f}   {gram_norm:>10.3f}   "
          f"{selectivity:>6.2f}×  {100*zero_rand:>6.1f}%")

# Pick the "best operating point" — the threshold that maximises
# (gram_norm * (1 - rand_norm)) — high grammar coverage, low random
# coverage.  Anything else is a single-axis greedy pick.
def _selectivity_score(r):
    return r["gram_norm"] * (1.0 - r["rand_norm"])

best = max(sweep_rows, key=_selectivity_score) if sweep_rows else None
if best:
    print(f"\n  Best operating point:")
    print(f"    threshold = {best['threshold']}")
    print(f"    grammar coverage  = {100*best['gram_norm']:.1f}%  of full binary tree")
    print(f"    random coverage   = {100*best['rand_norm']:.1f}%  of full binary tree")
    print(f"    selectivity ratio = {best['selectivity']:.1f}×")
    print(f"    {100*best['zero_rand']:.0f}% of random sentences produce zero chunks")

# Persist CSV.
with open(os.path.join(OUT_DIR, "negative_test_sweep.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "rand_mean", "gram_mean", "rand_norm",
                "gram_norm", "selectivity", "frac_random_zero"])
    for r in sweep_rows:
        w.writerow([r["threshold"], f"{r['rand_mean']:.3f}",
                    f"{r['gram_mean']:.3f}", f"{r['rand_norm']:.4f}",
                    f"{r['gram_norm']:.4f}", f"{r['selectivity']:.4f}",
                    f"{r['zero_rand']:.4f}"])

# Plot 1: mean chunks per sentence vs threshold for both populations.
fig, ax = plt.subplots(figsize=(9, 4.5))
xs    = [r["threshold"] for r in sweep_rows]
yrand = [r["rand_norm"] for r in sweep_rows]
ygram = [r["gram_norm"] for r in sweep_rows]
ax.plot(xs, ygram, "o-", color="#2ca02c", label="grammar", linewidth=2)
ax.plot(xs, yrand, "o-", color="#d62728", label="random",  linewidth=2)
ax.fill_between(xs, yrand, ygram, where=[g > r for g, r in zip(ygram, yrand)],
                color="#2ca02c", alpha=0.15, label="selectivity margin")
if best:
    ax.axvline(best["threshold"], color="black", linestyle="--", alpha=0.5,
               label=f"best @ thr={best['threshold']}")
ax.set_xlabel("count-gate threshold")
ax.set_ylabel("mean chunks / (n_words - 1)")
ax.set_ylim(0, 1.05)
ax.set_title("Negative test — chunks formed vs threshold\n"
             "(grammar should stay high, random should drop to 0)")
ax.legend(loc="lower left"); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "negative_test_sweep.png"), dpi=140,
            bbox_inches="tight")
plt.close()

# Plot 2: selectivity ratio vs threshold.
fig, ax = plt.subplots(figsize=(9, 4.5))
sels = [r["selectivity"] for r in sweep_rows]
ax.plot(xs, sels, "o-", color="#1f77b4", linewidth=2)
ax.axhline(1.0, color="black", linestyle=":", alpha=0.4,
           label="parity (no selectivity)")
if best:
    ax.axvline(best["threshold"], color="black", linestyle="--", alpha=0.5,
               label=f"best @ thr={best['threshold']}: "
                     f"{best['selectivity']:.1f}×")
ax.set_xlabel("count-gate threshold")
ax.set_ylabel("grammar / random  (selectivity ratio)")
ax.set_title("Recognition selectivity vs threshold")
ax.legend(loc="best"); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "negative_test_selectivity.png"), dpi=140,
            bbox_inches="tight")
plt.close()

# Persist the per-sentence breakdown at the BEST threshold for auditing.
if best:
    with open(os.path.join(OUT_DIR, "negative_test_best_thr.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["kind", "sentence", "n_words", "n_chunks",
                    f"frac_full_at_thr_{best['threshold']}"])
        for s, c, n in zip(neg_random_sents,
                            best["rand_chunks"], best["rand_n"]):
            w.writerow(["random", s, n, c, c / max(n - 1, 1)])
        for s, c, n in zip(neg_grammar_sents,
                            best["gram_chunks"], best["gram_n"]):
            w.writerow(["grammar", s, n, c, c / max(n - 1, 1)])

print(f"\n  negative_test_sweep.csv         — per-threshold metrics")
print(f"  negative_test_sweep.png         — chunks vs threshold (both pops)")
print(f"  negative_test_selectivity.png   — selectivity ratio vs threshold")
print(f"  negative_test_best_thr.csv      — per-sentence detail at best thr")

# =============================================================================
# PHASE 7 — HEURISTIC ANALYSIS FOR NEGATIVE-INPUT REJECTION
# =============================================================================
# Phase 6 sweeps a SINGLE gate (basic_level_count) and shows it has at
# best ~2× selectivity.  But there are 20 heuristics WEBSTER computes
# per candidate — maybe one of them separates random-sentence
# candidates from grammatical-sentence candidates more cleanly.
#
# Protocol — primitive-bigram level (no merges, so both populations
# are evaluated at the same parse depth):
#   1. Build primitives for each random / grammatical sentence.
#   2. For every adjacent bigram, run ``evaluate_pair`` and record
#      the full heuristic vector + a kind ∈ {random, grammar}.
#   3. Per heuristic: ROC AUC (grammar = positive class), best
#      single-threshold operating point that maximises
#      ``P(score > t | grammar) · (1 − P(score > t | random))``.
#   4. Best two-heuristic AND combination (gate1 > t1 AND gate2 > t2)
#      reported alongside the single-heuristic table.
#
# Goal: identify the heuristic (and threshold) that ACTUALLY rejects
# random inputs while preserving grammatical bigrams — better than
# the 2× ceiling the count gate alone gives.
print(f"\n=== PHASE 7: HEURISTIC ANALYSIS FOR NEGATIVE-INPUT REJECTION ===")

def _bigram_heurs(sentence):
    """Evaluate every adjacent primitive-bigram in ``sentence`` and
    return its full heuristic vector."""
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold="converge")
    out = []
    for p in tree.get_parentless_pairs():
        try:
            res = tree.evaluate_pair(p["left_word_index"],
                                      p["right_word_index"], debug=False)
        except Exception:
            continue
        out.append(_extract_heuristics(res))
    return out

p7_random  = []
p7_grammar = []
for s in neg_random_sents:
    p7_random.extend(_bigram_heurs(s))
for s in neg_grammar_sents:
    p7_grammar.extend(_bigram_heurs(s))
print(f"  Bigrams logged — random: {len(p7_random)}   grammar: {len(p7_grammar)}")

# Per-heuristic ROC-AUC + best-threshold operating point.
from sklearn.metrics import roc_auc_score

# Include basic_level_count as a candidate gate too — Phase 6 swept it
# in isolation; Phase 7 puts it on equal footing with the log-prob
# heuristics.  Add it to a temporary expanded list just for analysis.
P7_HEUR_NAMES = list(HEURISTICS) + ["cnt_bl_count", "ctx_bl_count"]
# Backfill bl_count columns since _extract_heuristics didn't include them.
def _add_bl(rows, side):
    key_in  = "cnt_bl_count" if side == "cnt" else "ctx_bl_count"
    src_key = "basic_level_count"
    score_key = ("content_score_data" if side == "cnt"
                 else "context_score_data")
    # Already on the heur dicts? skip.
    # We re-evaluate; instead just compute from existing keys —
    # _extract_heuristics didn't store bl_count, so re-run.
    pass

# Re-eval to also collect bl_count for both sides.
p7_random_full  = []
p7_grammar_full = []
def _bigram_heurs_full(sentence):
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold="converge")
    rows = []
    for p in tree.get_parentless_pairs():
        try:
            res = tree.evaluate_pair(p["left_word_index"],
                                      p["right_word_index"], debug=False)
        except Exception:
            continue
        h = _extract_heuristics(res)
        h["cnt_bl_count"] = float(res.get("content_score_data", {})
                                       .get("basic_level_count", -1))
        h["ctx_bl_count"] = float(res.get("context_score_data", {})
                                       .get("basic_level_count", -1))
        rows.append(h)
    return rows
for s in neg_random_sents:  p7_random_full.extend(_bigram_heurs_full(s))
for s in neg_grammar_sents: p7_grammar_full.extend(_bigram_heurs_full(s))

all_rows = ([(r, "random")  for r in p7_random_full] +
            [(r, "grammar") for r in p7_grammar_full])
y_true = np.array([1 if k == "grammar" else 0 for _, k in all_rows])

heur_analysis = []
for h in P7_HEUR_NAMES:
    g_vals = np.array([r[h] for r in p7_grammar_full])
    r_vals = np.array([r[h] for r in p7_random_full])
    all_vals = np.concatenate([r_vals, g_vals])
    all_kind = np.concatenate([np.zeros_like(r_vals), np.ones_like(g_vals)])
    try:    auc = float(roc_auc_score(all_kind, all_vals))
    except Exception:
        auc = 0.5
    # Best single-threshold operating point.
    cands = sorted(set(np.concatenate([g_vals, r_vals]).tolist()))
    best_t = best_score = best_g = best_r = None
    for t in cands:
        g_pass = float((g_vals > t).mean())
        r_pass = float((r_vals > t).mean())
        score = g_pass * (1.0 - r_pass)
        if best_score is None or score > best_score:
            best_score, best_t = score, t
            best_g, best_r = g_pass, r_pass
    heur_analysis.append({
        "heuristic": h, "auc": auc,
        "best_t":  best_t, "best_score": best_score,
        "g_pass":  best_g, "r_pass": best_r,
        "g_mean":  float(g_vals.mean()),
        "r_mean":  float(r_vals.mean()),
        "g_std":   float(g_vals.std()),
        "r_std":   float(r_vals.std()),
    })

# Sort by AUC (best separator first).
heur_analysis.sort(key=lambda d: d["auc"], reverse=True)

print(f"\n  Per-heuristic random-vs-grammar discriminability:")
print(f"  {'heuristic':<22}  {'AUC':>5}  "
      f"{'best_t':>10}  {'g_pass':>8}  {'r_pass':>8}  "
      f"{'selectivity':>11}  {'gain':>7}")
for d in heur_analysis:
    sel = d["g_pass"] / max(1 - d["r_pass"], 1e-9)  # not used as primary metric
    gain = d["g_pass"] - d["r_pass"]
    print(f"  {d['heuristic']:<22}  "
          f"{d['auc']:>5.3f}  {d['best_t']:>10.4g}  "
          f"{100*d['g_pass']:>7.1f}%  {100*d['r_pass']:>7.1f}%  "
          f"{(d['g_pass']/max(d['r_pass'], 1e-9)):>10.2f}×  "
          f"{100*gain:>6.1f}pp")

# Persist CSV.
with open(os.path.join(OUT_DIR, "negative_test_heur_analysis.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["heuristic", "auc", "best_threshold",
                "grammar_pass_rate", "random_pass_rate",
                "selectivity_ratio", "discriminative_gain",
                "grammar_mean", "grammar_std",
                "random_mean", "random_std"])
    for d in heur_analysis:
        w.writerow([d["heuristic"], f"{d['auc']:.4f}",
                    f"{d['best_t']:.4g}",
                    f"{d['g_pass']:.4f}", f"{d['r_pass']:.4f}",
                    f"{d['g_pass']/max(d['r_pass'], 1e-9):.4f}",
                    f"{d['g_pass']-d['r_pass']:.4f}",
                    f"{d['g_mean']:.4f}", f"{d['g_std']:.4f}",
                    f"{d['r_mean']:.4f}", f"{d['r_std']:.4f}"])

# Plot: AUC bar chart for all heuristics.
fig, ax = plt.subplots(figsize=(10, max(4, len(heur_analysis) * 0.35)))
xs = [d["heuristic"] for d in heur_analysis]
aucs = [d["auc"] for d in heur_analysis]
colors = ["#2ca02c" if a > 0.6 else "#d62728" if a < 0.4
          else "#888888" for a in aucs]
ax.barh(range(len(xs))[::-1], aucs, color=colors)
ax.set_yticks(range(len(xs))[::-1]); ax.set_yticklabels(xs, fontsize=8)
ax.axvline(0.5, color="black", linestyle=":", alpha=0.5,
           label="chance (AUC=0.5)")
ax.set_xlim(0, 1.0); ax.set_xlabel("ROC AUC (grammar = positive)")
ax.set_title("Heuristic discriminability — random vs grammar bigrams")
for i, a in enumerate(aucs):
    ax.text(a + 0.01, len(xs) - 1 - i, f"{a:.2f}", va="center", fontsize=7)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "negative_test_heur_auc.png"),
            dpi=140, bbox_inches="tight")
plt.close()

# Plot: distributions for the top-4 most discriminative heuristics.
top_h = [d["heuristic"] for d in heur_analysis[:4]]
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, h in zip(axes.flatten(), top_h):
    g = np.array([r[h] for r in p7_grammar_full])
    r = np.array([r[h] for r in p7_random_full])
    bins = np.linspace(min(g.min(), r.min()),
                       max(g.max(), r.max()), 30)
    ax.hist(r, bins=bins, color="#d62728", alpha=0.55,
            label=f"random  (μ={r.mean():.2f})", edgecolor="white")
    ax.hist(g, bins=bins, color="#2ca02c", alpha=0.55,
            label=f"grammar (μ={g.mean():.2f})", edgecolor="white")
    info = next(d for d in heur_analysis if d["heuristic"] == h)
    ax.axvline(info["best_t"], color="black", linestyle="--", alpha=0.6,
               label=f"best t={info['best_t']:.2f}")
    ax.set_title(f"{h}   (AUC={info['auc']:.3f},  "
                 f"g_pass={100*info['g_pass']:.0f}%, "
                 f"r_pass={100*info['r_pass']:.0f}%)", fontsize=10)
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "negative_test_top_heur_hists.png"),
            dpi=140, bbox_inches="tight")
plt.close()

# ── Two-heuristic AND-combination search ─────────────────────────────────
# Optionally combine TWO gates: admit only if both gate1 > t1 AND
# gate2 > t2.  Useful if no single heuristic is enough.  Brute-force
# search over the top-K heuristics × a small threshold grid each.
TOP_K_FOR_COMBO = 5
GRID_PER_HEUR   = 7   # quantile grid over the bigram-population values

top_for_combo = [d["heuristic"] for d in heur_analysis[:TOP_K_FOR_COMBO]]
combo_results = []
for i, h1 in enumerate(top_for_combo):
    for h2 in top_for_combo[i+1:]:
        g1 = np.array([r[h1] for r in p7_grammar_full])
        r1 = np.array([r[h1] for r in p7_random_full])
        g2 = np.array([r[h2] for r in p7_grammar_full])
        r2 = np.array([r[h2] for r in p7_random_full])
        q1_grid = np.quantile(np.concatenate([g1, r1]),
                              np.linspace(0, 1, GRID_PER_HEUR + 2)[1:-1])
        q2_grid = np.quantile(np.concatenate([g2, r2]),
                              np.linspace(0, 1, GRID_PER_HEUR + 2)[1:-1])
        for t1 in q1_grid:
            for t2 in q2_grid:
                g_mask = (g1 > t1) & (g2 > t2)
                r_mask = (r1 > t1) & (r2 > t2)
                g_pass = float(g_mask.mean())
                r_pass = float(r_mask.mean())
                score  = g_pass * (1.0 - r_pass)
                combo_results.append({
                    "h1": h1, "t1": float(t1),
                    "h2": h2, "t2": float(t2),
                    "g_pass": g_pass, "r_pass": r_pass, "score": score,
                })

combo_results.sort(key=lambda d: d["score"], reverse=True)
print(f"\n  Top 10 two-heuristic AND-combinations:")
print(f"  {'gate 1':<22} {'t1':>9}  {'gate 2':<22} {'t2':>9}  "
      f"{'g_pass':>7}  {'r_pass':>7}  {'score':>6}")
for d in combo_results[:10]:
    print(f"  {d['h1']:<22} {d['t1']:>9.3g}  "
          f"{d['h2']:<22} {d['t2']:>9.3g}  "
          f"{100*d['g_pass']:>6.1f}%  {100*d['r_pass']:>6.1f}%  "
          f"{d['score']:>6.3f}")

with open(os.path.join(OUT_DIR, "negative_test_combo_gates.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["heur_1", "t1", "heur_2", "t2",
                "g_pass", "r_pass", "score"])
    for d in combo_results[:50]:
        w.writerow([d["h1"], f"{d['t1']:.4g}",
                    d["h2"], f"{d['t2']:.4g}",
                    f"{d['g_pass']:.4f}", f"{d['r_pass']:.4f}",
                    f"{d['score']:.4f}"])

print(f"\n  negative_test_heur_analysis.csv  — single-heuristic AUC+threshold")
print(f"  negative_test_heur_auc.png       — AUC bar chart per heuristic")
print(f"  negative_test_top_heur_hists.png — top-4 heuristic distributions")
print(f"  negative_test_combo_gates.csv    — top two-heuristic AND combos")

# =============================================================================
# Final summary
# =============================================================================
print("\n=== SUMMARY ===")
best_auc_h    = max(heuristic_auc.items(),       key=lambda kv: kv[1])
best_step     = max(heuristic_step_acc.items(),  key=lambda kv: kv[1][2])
best_quantile = max(best_per_heur.items(),       key=lambda kv: kv[1]["overall_acc"])
best_combo    = max(combo_rows,                  key=lambda r: r["step_acc"])
best_combo_auc = max(combo_rows,                 key=lambda r: r["auc"])

print(f"  Best heuristic by gold-pair AUC: {best_auc_h[0]} (AUC={best_auc_h[1]:.3f})")
print(f"  Best heuristic by step-pick acc:  {best_step[0]} "
      f"({100*best_step[1][2]:.1f}%)")
print(f"  Best single-heuristic gate (Phase 4): "
      f"{best_quantile[0]} @ q={best_quantile[1]['q']} "
      f"(overall acc {100*best_quantile[1]['overall_acc']:.1f}%)")
print(f"  Best weighted-combo formula (Phase 4c): "
      f"{best_combo['formula']}  "
      f"(step acc {100*best_combo['step_acc']:.1f}%, AUC {best_combo['auc']:.3f})")
if best:
    print(f"  Negative-test best operating point (Phase 6): "
          f"threshold={best['threshold']}  "
          f"grammar={100*best['gram_norm']:.0f}%  "
          f"random={100*best['rand_norm']:.0f}%  "
          f"selectivity={best['selectivity']:.1f}×")
print(f"\nOutputs in {OUT_DIR}/")
print(f"  candidate_log.csv             — every candidate pair + heuristics")
print(f"  step_picks.csv                — per-step heuristic picks")
print(f"  heuristic_histograms.png      — gold vs non-gold per heuristic")
print(f"  heuristic_roc.png             — ROC curves (AUC in legend)")
print(f"  heuristic_pr.png              — Precision-Recall curves")
print(f"  step_pick_accuracy.png        — argmax-heuristic step accuracy")
print(f"  threshold_sweep.csv/.png      — per-heuristic quantile sweep")
print(f"  gate_rank_sweep.csv           — (gate × rank) full sweep")
print(f"  gate_rank_heatmap.png         — (gate × rank) heatmap @ q=0.5")
print(f"  weighted_combo_sweep.csv      — α-weighted pair combinations")
print(f"  weighted_combo_heatmap.png    — best step-pick per pair")
print(f"  heuristic_summary.csv         — AUC + step-accuracy summary")
print(f"  parse_trees/*.png             — annotated parse trees "
      f"({len(viz_sample)} sentences)")
print(f"  context_tree/ + content_tree/ — hierarchy inspection visuals")

with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write("WEBSTER Threshold + Parsing-Heuristic Analysis\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Trained WEBSTER on {len(train_hollow)} hollow trees.\n")
    f.write(f"Probed on {len(test_hollow)} held-out hollow trees.\n")
    f.write(f"Logged {len(candidate_log)} candidate pair evaluations "
            f"across {len(step_picks)} merge steps.\n")
    f.write(f"Heuristic set: NO basic-level — only tree_log_prob, "
            f"tree_class_log_prob, root_log_prob, leaf_log_prob and "
            f"their derived combinations.\n\n")
    f.write("ROC AUC (gold vs non-gold candidate pair):\n")
    for h, a in sorted(heuristic_auc.items(), key=lambda x: -x[1]):
        f.write(f"  {h:>22s}: AUC={a:.4f}  AP={heuristic_ap.get(h, 0):.4f}\n")
    f.write("\nStep-pick accuracy (argmax-heuristic == gold):\n")
    for h, (m, t, a) in sorted(heuristic_step_acc.items(),
                                key=lambda kv: -kv[1][2]):
        f.write(f"  {h:>22s}: {m:>4}/{t:<4} ({100*a:.1f}%)\n")
    f.write("\nBest single-heuristic gate τ (Phase 4 quantile sweep):\n")
    for h, r in sorted(best_per_heur.items(),
                       key=lambda kv: -kv[1]["overall_acc"]):
        f.write(f"  {h:>22s}: q={r['q']}  overall={100*r['overall_acc']:.1f}%\n")
    f.write(f"\nBest heuristic by AUC:           {best_auc_h[0]} "
            f"(AUC={best_auc_h[1]:.4f})\n")
    f.write(f"Best heuristic by step acc:      {best_step[0]} "
            f"({100*best_step[1][2]:.1f}%)\n")
    f.write(f"Best single-heuristic gate:      "
            f"{best_quantile[0]} @ q={best_quantile[1]['q']} "
            f"({100*best_quantile[1]['overall_acc']:.1f}%)\n")
    f.write(f"Best weighted-combo formula:     "
            f"{best_combo['formula']}\n"
            f"  step acc = {100*best_combo['step_acc']:.1f}%  "
            f"AUC = {best_combo['auc']:.4f}\n")
