"""
Grammar Distillation Test (met5)
================================

Methodology 5.2 — "Grammar Distillation" (MULTIHIERARCHY.md):

After WEBSTER is trained, treat a FRONTIER of nodes from the content
and context cobweb trees as the non-terminal alphabet of a learned
CFG. Each frontier node is a "category" — NP, VP, S, …  Parsing then
labels each candidate chunk with its best-matching frontier node and
commits to whichever candidate matches its label most strongly.
Generation inverts: sample a frontier node, then unpack via the
frontier-matched children of its pool bags.

Frontier strategies tested
--------------------------
  fixed_d1, fixed_d2, fixed_d3 — all nodes at exactly that depth
                                 (root = depth 0).
  basic_level                  — unique get_basic() ancestors collected
                                 across all leaves.

Scoring modes tested
--------------------
  ITERATE     — for each candidate's content+context bags, score against
                EVERY frontier node and take the max log_prob_instance.
                "Exhaustive label scan".
  CATEGORIZE  — DFS from the cobweb root, stop at the first frontier
                node encountered; that node's log_prob_instance is the
                label-score. "Hierarchical label routing".

Cross product: 4 frontiers × 2 modes = 8 parse configurations, plus a
BASELINE row for the current ``build()`` (climbing-ancestor) strategy.

Generation
----------
Custom frontier-based generator: sample a sentence-shaped frontier
node, walk down to its most-supported leaf descendant, unpack via
``webster.generate_sentence(start_content_leaf=…)``.

Phase 5 — UNSUPERVISED grammar distillation
-------------------------------------------
For the two most-informative frontiers (``fixed_d3`` and
``basic_level``) we distill a CFG **without any gold supervision**:

  1. **NT naming = pure cluster identity.** Each frontier node gets
     an opaque ``NT_<i>`` label (ordered by training count). For
     readability, the rules table shows the *top representative
     tokens* (words appearing under that NT in WEBSTER's auto-parses)
     and adds a ``{ gold_class }`` sanity tag — but the gold tag is
     never consumed by the grammar itself.

  2. **Rules from auto-parses.** WEBSTER parses 200 fresh sentences
     from the test grammar (no hollow gold, no merge replay). Each
     composite contributes one ``(parent_NT, left_NT, right_NT)``
     observation, with left/right being either an NT label or a
     terminal word for primitive children.

  3. **Greedy reconstruction.** All observations are flattened to
     ``(parent, (left, right), count)``, sorted by count, and added
     to the distilled grammar one rule at a time. We track the
     cumulative fraction of all observed composites covered by the
     first K rules → the "coverage curve" (a compact grammar climbs
     fast).

Renderers:
  * ``rules_{frontier}.png``   — CFG rules table with NT labels,
                                  representative tokens, gold sanity
                                  tag, count, log p(RHS | LHS).
  * ``coverage_{frontier}.png`` — greedy reconstruction curve with
                                  50% / 80% / 95% milestones marked.
  * ``node_link_{frontier}.png`` — directed graph of NT → child NT
                                   edges weighted by count.
  * ``derivations/derivation_parse_{frontier}_{i}.png`` — parse
    trees annotated with the same unsupervised labels + log-probs.
  * ``derivations/derivation_gen_{frontier}_{i}.png`` — generation
    unpacking trees with the same labels.

Outputs (under ``grammar_distillation_test_output/``)
-----------------------------------------------------
  parse_eval.csv                       — per-config step-pick + P/R/F1
  gen_eval.csv                         — per-frontier generation rates
  frontier_summary.csv                 — frontier sizes / depth dists
  parse_comparison.png                 — bar chart of F1 / step-pick
  generation_comparison.png            — bar chart of generation rates
  rules_{frontier}.csv + .png          — distilled CFG productions
                                          (greedy order in CSV)
  coverage_{frontier}.png              — greedy reconstruction curve
  node_link_{frontier}.png             — NT-connection graph
  derivation_parse_{frontier}_{i}.png  — parse trees with NT labels
  derivation_gen_{frontier}_{i}.png    — generation unpacking trees
"""

import os
import sys
import csv
import glob
import json
import math
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
import matplotlib.patches as mpatches

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import (WEBSTER, FiniteParseTree, PrimitiveParseNode,
                      CompositeParseNode)
from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed

# ── Configuration ─────────────────────────────────────────────────────────────
OUT_DIR           = os.path.join(_HERE, "grammar_distillation_test_output")
HOLLOW_CORPUS_DIR = "data/test_hollow_grammar_1"
CONTEXT_LENGTH    = 3
THRESHOLD         = 30
PRIMITIVES_FIRST  = 200
EVAL_ALPHA        = 10.0
SEED              = 13
N_GEN             = 30           # sentences to generate per frontier strategy
N_DERIV_VIZ       = 5            # parse + gen derivation trees rendered

# Match hyperparameters in hollow_learn_test_mh.py (alpha=1e-4 picked via
# grammar_param_sweep_test.py — better EM and step-pick than 1e-6).
random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
DERIV_DIR = os.path.join(OUT_DIR, "derivations")
os.makedirs(DERIV_DIR, exist_ok=True)

# Word → POS for chunk classification.
POS_LIST = ["Det", "N", "Adj", "V", "P"]
WORD_TO_POS = {}
for pos in POS_LIST:
    for prod in TEST_GRAMMAR1[pos]:
        for w in prod:
            WORD_TO_POS[w] = pos
CHUNK_LABELS = ["NP", "AdjP", "PP", "VP", "S"]
ALL_LABELS   = POS_LIST + CHUNK_LABELS + ["OTHER"]
# Per-non-terminal colors so visualizations are consistent.
LABEL_COLOR  = {
    "Det":   "#2ca02c", "N":     "#8c564b", "Adj":   "#1f77b4",
    "V":     "#17becf", "P":     "#7f7f7f", "NP":    "#ff7f0e",
    "AdjP":  "#9467bd", "PP":    "#bcbd22", "VP":    "#e377c2",
    "S":     "#d62728", "OTHER": "#cccccc",
}

# =============================================================================
# PHASE 0 — Train WEBSTER (mirror grammar_threshold_test.py)
# =============================================================================
print("=== PHASE 0: Train WEBSTER ===")
webster = WEBSTER(
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

print(f"  Phase 0a: {PRIMITIVES_FIRST} primitive-only sentences")
for i in range(PRIMITIVES_FIRST):
    s = generate("S", TEST_GRAMMAR1)
    webster.parse_sentence(s, threshold=1e9, new_vocab=True,
                           learning=True, debug=False)
    if (i + 1) % 50 == 0:
        print(f"    [{i+1}/{PRIMITIVES_FIRST}]")

print(f"  Phase 0b: hollow corpus replay")
hollow_corpus: list = []
for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
    with open(p) as f:
        try:    data = json.load(f)
        except json.JSONDecodeError: continue
    if "sentence" in data and "merges" in data:
        hollow_corpus.append(data)
print(f"    Loaded {len(hollow_corpus)} hollow trees")

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
    if (i + 1) % 25 == 0:
        print(f"    [{i+1}/{len(train_hollow)}]")

cnt_root = webster.ltm.content_hierarchy.root
ctx_root = webster.ltm.context_hierarchy.root


# =============================================================================
# Shared helpers
# =============================================================================
def _walk(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        for c in n.children:
            stack.append(c)

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

def _chunk_tokens(node):
    """Return the literal tokens in left-to-right order."""
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

def bracket_set(parse_tree):
    out = set()
    for comp in _walk_composites(parse_tree.global_root_node):
        s, e = _chunk_span(comp)
        if s is not None and e is not None and s != e:
            out.add((s, e))
    return out

def _categorize_to_frontier(bag, root, frontier_hashes):
    """DFS from root, stop at first node whose concept_hash is in
    frontier_hashes. Returns (node, found_bool)."""
    node = root
    if str(node.concept_hash()) in frontier_hashes:
        return node, True
    while node.children:
        node = max(node.children, key=lambda c: c.log_prob_instance(bag))
        if str(node.concept_hash()) in frontier_hashes:
            return node, True
    return node, False


# =============================================================================
# PHASE 1 — Extract frontiers
# =============================================================================
print("\n=== PHASE 1: Extract frontiers ===")

def fixed_depth_frontier(root, target_depth):
    frontier = []
    def walk(node, d):
        if d == target_depth:
            frontier.append(node); return
        if d < target_depth and node.children:
            for c in node.children:
                walk(c, d + 1)
        elif d < target_depth and not node.children:
            frontier.append(node)
    walk(root, 0)
    return frontier

def basic_level_frontier(root, eval_alpha=EVAL_ALPHA):
    seen = {}
    for node in _walk(root):
        if node.children: continue
        bl = node.get_basic(0, 0, debug=False,
                            eval_alpha=eval_alpha, use_root=True)
        seen[str(bl.concept_hash())] = bl
    return list(seen.values())

def frontier_stats(name, frontier, all_nodes):
    if not frontier: return {"name": name, "size": 0}
    depths = [n.depth() for n in frontier]
    counts = [int(n.count) for n in frontier]
    n_total = len(all_nodes)
    return {
        "name":           name,
        "size":           len(frontier),
        "tree_size":      n_total,
        "frontier_pct":   100 * len(frontier) / max(n_total, 1),
        "depth_min":      min(depths),
        "depth_median":   int(np.median(depths)),
        "depth_max":      max(depths),
        "count_min":      min(counts),
        "count_median":   int(np.median(counts)),
        "count_max":      max(counts),
        "count_sum":      sum(counts),
    }

CONTENT_NODES = list(_walk(cnt_root))
CONTEXT_NODES = list(_walk(ctx_root))
print(f"  Content tree: {len(CONTENT_NODES)} nodes total")
print(f"  Context tree: {len(CONTEXT_NODES)} nodes total")

FRONTIERS_CONTENT = {
    "fixed_d1":     fixed_depth_frontier(cnt_root, 1),
    "fixed_d2":     fixed_depth_frontier(cnt_root, 2),
    "fixed_d3":     fixed_depth_frontier(cnt_root, 3),
    "basic_level":  basic_level_frontier(cnt_root),
}
FRONTIERS_CONTEXT = {
    "fixed_d1":     fixed_depth_frontier(ctx_root, 1),
    "fixed_d2":     fixed_depth_frontier(ctx_root, 2),
    "fixed_d3":     fixed_depth_frontier(ctx_root, 3),
    "basic_level":  basic_level_frontier(ctx_root),
}

frontier_rows = []
print(f"\n  {'tree':<8} {'frontier':<14} {'size':>5} {'%':>5}  "
      f"{'depth (min/med/max)':>22}  {'count (min/med/max)':>22}")
for tree_name, frontiers in [("content", FRONTIERS_CONTENT),
                              ("context", FRONTIERS_CONTEXT)]:
    all_nodes = CONTENT_NODES if tree_name == "content" else CONTEXT_NODES
    for name, frontier in frontiers.items():
        st = frontier_stats(name, frontier, all_nodes)
        st["tree"] = tree_name
        frontier_rows.append(st)
        if st["size"]:
            print(f"  {tree_name:<8} {name:<14} {st['size']:>5} "
                  f"{st['frontier_pct']:>4.1f}%  "
                  f"{st['depth_min']:>2}/{st['depth_median']:>2}/{st['depth_max']:<3}"
                  f"               "
                  f"{st['count_min']:>5}/{st['count_median']:>5}/{st['count_max']:<5}")

with open(os.path.join(OUT_DIR, "frontier_summary.csv"), "w") as f:
    w = csv.DictWriter(f, fieldnames=[
        "tree", "name", "size", "tree_size", "frontier_pct",
        "depth_min", "depth_median", "depth_max",
        "count_min", "count_median", "count_max", "count_sum"])
    w.writeheader()
    for r in frontier_rows:
        if r.get("size", 0):
            w.writerow(r)


# =============================================================================
# PHASE 2 — Frontier-based parsing (parse + step-pick per (frontier × mode))
# =============================================================================
print("\n=== PHASE 2: Frontier-based parsing ===")

def score_candidate(content_bag, context_bag,
                    frontier_content, frontier_context,
                    frontier_content_hashes, frontier_context_hashes,
                    mode):
    """Combined frontier-match score (content + context)."""
    if mode == "iterate":
        cnt_score = max((f.log_prob_instance(content_bag)
                         for f in frontier_content),
                        default=-float("inf"))
        ctx_score = max((f.log_prob_instance(context_bag)
                         for f in frontier_context),
                        default=-float("inf"))
        cnt_node, ctx_node = None, None
    else:  # "categorize"
        cnt_node, _ = _categorize_to_frontier(
            content_bag, cnt_root, frontier_content_hashes)
        ctx_node, _ = _categorize_to_frontier(
            context_bag, ctx_root, frontier_context_hashes)
        cnt_score = cnt_node.log_prob_instance(content_bag)
        ctx_score = ctx_node.log_prob_instance(context_bag)
    return cnt_score + ctx_score, cnt_score, ctx_score, cnt_node, ctx_node

def best_frontier_label(bag, frontier_nodes):
    """Return (best_node, best_logp) — for iterate mode + viz labeling."""
    if not frontier_nodes: return None, -float("inf")
    best_node, best_lp = None, -float("inf")
    for f in frontier_nodes:
        lp = f.log_prob_instance(bag)
        if lp > best_lp:
            best_lp, best_node = lp, f
    return best_node, best_lp

def parse_with_frontier(sentence, frontier_content, frontier_context,
                        mode, record_labels=False):
    """End-to-end parse. If record_labels=True, every composite node
    gets a ._frontier_label and ._frontier_logp attached for viz."""
    frontier_content_hashes = {str(n.concept_hash()) for n in frontier_content}
    frontier_context_hashes = {str(n.concept_hash()) for n in frontier_context}

    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold="converge")

    while True:
        pairs = tree.get_parentless_pairs()
        if len(pairs) <= 0: break
        best_score, best_pair, best_label_node, best_label_lp = (
            -float("inf"), None, None, -float("inf"))
        for p in pairs:
            try:
                res = tree.evaluate_pair(
                    p["left_word_index"], p["right_word_index"])
            except Exception:
                continue
            score, _, _, cnt_node, _ = score_candidate(
                res["content_inst"], res["context_inst"],
                frontier_content, frontier_context,
                frontier_content_hashes, frontier_context_hashes,
                mode)
            if score > best_score:
                best_score, best_pair = score, p
                if record_labels:
                    if cnt_node is None:
                        # iterate mode — find argmax frontier for labeling
                        cnt_node, lp = best_frontier_label(
                            res["content_inst"], frontier_content)
                        best_label_node, best_label_lp = cnt_node, lp
                    else:
                        best_label_node = cnt_node
                        best_label_lp = cnt_node.log_prob_instance(res["content_inst"])
        if best_pair is None or best_score <= -float("inf"):
            break
        try:
            result = tree.apply_candidate(
                best_pair["left_word_index"],
                best_pair["right_word_index"])
        except Exception:
            break
        if record_labels and best_label_node is not None:
            # Find the just-added composite and stamp the label.
            added_title = result.get("added_node", {}).get("title")
            for n in tree.nodes:
                if getattr(n, "title", None) == added_title:
                    n._frontier_label = best_label_node
                    n._frontier_logp  = best_label_lp
                    break
        if len(tree.global_root_node.children) <= 1: break
    return tree

def step_pick_with_frontier(test_hollow, frontier_content, frontier_context,
                             mode):
    frontier_content_hashes = {str(n.concept_hash()) for n in frontier_content}
    frontier_context_hashes = {str(n.concept_hash()) for n in frontier_context}
    n_correct = n_total = 0
    for hollow in test_hollow:
        sentence = hollow["sentence"]
        gold_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        gold_tree.build_primitives(sentence, threshold="converge")
        for m in hollow["merges"]:
            try: gold_tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        gold = bracket_set(gold_tree)
        if not gold: continue

        step_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        step_tree.build_primitives(sentence, threshold="converge")
        for m in hollow["merges"]:
            pairs = step_tree.get_parentless_pairs()
            if not pairs: break
            best_score, best_span = -float("inf"), None
            for p in pairs:
                try:
                    res = step_tree.evaluate_pair(
                        p["left_word_index"], p["right_word_index"])
                except Exception:
                    continue
                score, _, _, _, _ = score_candidate(
                    res["content_inst"], res["context_inst"],
                    frontier_content, frontier_context,
                    frontier_content_hashes, frontier_context_hashes,
                    mode)
                left_node  = step_tree._find_root_child_by_index(p["left_word_index"])
                right_node = step_tree._find_root_child_by_index(p["right_word_index"])
                if left_node is None or right_node is None: continue
                ls, _  = _chunk_span(left_node)
                _, re_ = _chunk_span(right_node)
                if score > best_score:
                    best_score, best_span = score, (int(ls), int(re_))
            n_total += 1
            if best_span is not None and best_span in gold:
                n_correct += 1
            try:
                step_tree.apply_candidate(m["left"], m["right"])
            except Exception:
                break
    return n_correct, n_total

parse_eval_rows = []
print(f"\n  {'config':<24} {'step-pick':>10}  {'P':>6} {'R':>6} {'F1':>6}  "
      f"{'exact':>6}")

# ── BASELINE: current build() (climbing-ancestor) ───────────────────────────
print("\n  (baseline: current build() — climbing-ancestor gate)")
bl_tp = bl_fp = bl_fn = bl_exact = bl_total = 0
bl_sp_hit = bl_sp_tot = 0
for hollow in test_hollow:
    sentence = hollow["sentence"]
    gold_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    gold_tree.build_primitives(sentence, threshold="converge")
    for m in hollow["merges"]:
        try: gold_tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    gold = bracket_set(gold_tree)
    if not gold: continue
    pred_tree = webster.parse_sentence(sentence, threshold=THRESHOLD,
                                        new_vocab=False, learning=False)
    pred = bracket_set(pred_tree)
    bl_tp += len(gold & pred); bl_fp += len(pred - gold); bl_fn += len(gold - pred)
    bl_total += 1
    if gold == pred and len(gold) > 0: bl_exact += 1
    step_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    step_tree.build_primitives(sentence, threshold="converge")
    for m in hollow["merges"]:
        pairs = step_tree.get_parentless_pairs()
        if not pairs: break
        best_score, best_span = -float("inf"), None
        for p in pairs:
            try:
                res = step_tree.evaluate_pair(
                    p["left_word_index"], p["right_word_index"],
                    climb_count_threshold=THRESHOLD)
            except Exception:
                continue
            csd = res.get("content_score_data", {})
            if csd.get("climb_hit_root", True):
                continue
            score = csd.get("root_log_prob", -float("inf"))
            left_node  = step_tree._find_root_child_by_index(p["left_word_index"])
            right_node = step_tree._find_root_child_by_index(p["right_word_index"])
            if left_node is None or right_node is None: continue
            ls, _  = _chunk_span(left_node)
            _, re_ = _chunk_span(right_node)
            if score > best_score:
                best_score, best_span = score, (int(ls), int(re_))
        bl_sp_tot += 1
        if best_span is not None and best_span in gold:
            bl_sp_hit += 1
        try:
            step_tree.apply_candidate(m["left"], m["right"])
        except Exception:
            break
bl_prec = bl_tp / max(bl_tp + bl_fp, 1)
bl_rec  = bl_tp / max(bl_tp + bl_fn, 1)
bl_f1   = 2 * bl_prec * bl_rec / max(bl_prec + bl_rec, 1e-12)
bl_exact_pct = bl_exact / max(bl_total, 1)
bl_sp_acc = bl_sp_hit / max(bl_sp_tot, 1)
parse_eval_rows.append({
    "config":      "BASELINE_climb",
    "frontier":    "climb_ancestor",
    "mode":        "build()",
    "step_pick":   bl_sp_acc,
    "precision":   bl_prec,
    "recall":      bl_rec,
    "f1":          bl_f1,
    "exact_match": bl_exact_pct,
    "frontier_size_content": -1,
    "frontier_size_context": -1,
})
print(f"  {'BASELINE_climb':<24} {100*bl_sp_acc:>9.1f}%  "
      f"{100*bl_prec:>5.1f}% {100*bl_rec:>5.1f}% {100*bl_f1:>5.1f}%  "
      f"{100*bl_exact_pct:>5.1f}%")

for f_name in ["fixed_d1", "fixed_d2", "fixed_d3", "basic_level"]:
    f_cnt = FRONTIERS_CONTENT[f_name]
    f_ctx = FRONTIERS_CONTEXT[f_name]
    if not f_cnt or not f_ctx:
        continue
    for mode in ["iterate", "categorize"]:
        cfg_name = f"{f_name}_{mode}"
        sp_hit, sp_tot = step_pick_with_frontier(test_hollow, f_cnt, f_ctx, mode)
        sp_acc = sp_hit / max(sp_tot, 1)
        tp = fp = fn = exact = total = 0
        for hollow in test_hollow:
            sentence = hollow["sentence"]
            gold_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
            gold_tree.build_primitives(sentence, threshold="converge")
            for m in hollow["merges"]:
                try: gold_tree.apply_candidate(m["left"], m["right"])
                except Exception: pass
            gold = bracket_set(gold_tree)
            pred_tree = parse_with_frontier(sentence, f_cnt, f_ctx, mode)
            pred = bracket_set(pred_tree)
            tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
            total += 1
            if gold == pred and len(gold) > 0:
                exact += 1
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-12)
        exact_pct = exact / max(total, 1)
        parse_eval_rows.append({
            "config":      cfg_name,
            "frontier":    f_name,
            "mode":        mode,
            "step_pick":   sp_acc,
            "precision":   prec,
            "recall":      rec,
            "f1":          f1,
            "exact_match": exact_pct,
            "frontier_size_content": len(f_cnt),
            "frontier_size_context": len(f_ctx),
        })
        print(f"  {cfg_name:<24} {100*sp_acc:>9.1f}%  "
              f"{100*prec:>5.1f}% {100*rec:>5.1f}% {100*f1:>5.1f}%  "
              f"{100*exact_pct:>5.1f}%")

with open(os.path.join(OUT_DIR, "parse_eval.csv"), "w") as f:
    w = csv.DictWriter(f, fieldnames=[
        "config", "frontier", "mode",
        "step_pick", "precision", "recall", "f1", "exact_match",
        "frontier_size_content", "frontier_size_context"])
    w.writeheader()
    for r in parse_eval_rows:
        w.writerow(r)


# =============================================================================
# PHASE 3 — Frontier-based generation
# =============================================================================
print("\n=== PHASE 3: Frontier-based generation ===")

# CYK
def _cyk_top_cell(tokens):
    n = len(tokens)
    if n == 0: return set()
    term_lhs = defaultdict(set)
    for lhs, prods in TEST_GRAMMAR1.items():
        for prod in prods:
            if len(prod) == 1 and prod[0] in WORD_TO_POS or (
                    len(prod) == 1 and prod[0] not in TEST_GRAMMAR1):
                term_lhs[prod[0]].add(lhs)
    chart = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for i, tok in enumerate(tokens):
        chart[i][i + 1] = set(term_lhs.get(tok, set()))
        changed = True
        while changed:
            changed = False
            for lhs, prods in TEST_GRAMMAR1.items():
                for prod in prods:
                    if len(prod) == 1 and prod[0] in chart[i][i + 1]:
                        if lhs not in chart[i][i + 1]:
                            chart[i][i + 1].add(lhs); changed = True
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                left, right = chart[i][k], chart[k][j]
                if not left or not right: continue
                for lhs, prods in TEST_GRAMMAR1.items():
                    for prod in prods:
                        if len(prod) == 2 and prod[0] in left and prod[1] in right:
                            chart[i][j].add(lhs)
                        elif len(prod) == 3 and k > i + 1:
                            for k2 in range(i + 1, k):
                                mid_set = chart[k2][k]
                                left2   = chart[i][k2]
                                if (prod[0] in left2 and
                                        prod[1] in mid_set and
                                        prod[2] in right):
                                    chart[i][j].add(lhs)
            changed = True
            while changed:
                changed = False
                for lhs, prods in TEST_GRAMMAR1.items():
                    for prod in prods:
                        if len(prod) == 1 and prod[0] in chart[i][j]:
                            if lhs not in chart[i][j]:
                                chart[i][j].add(lhs); changed = True
    return chart[0][n]

def _grammar_recognize(tokens, start="S"):
    return start in _cyk_top_cell(tokens)

def _is_valid_constituent(tokens):
    return len(_cyk_top_cell(tokens)) > 0

def _representative_leaf(frontier_node):
    node = frontier_node
    while node.children:
        node = max(node.children, key=lambda c: c.count)
    return node

def generate_from_frontier(frontier_content, return_leaf=False):
    median_count = int(np.median([n.count for n in frontier_content])) \
                   if frontier_content else 0
    candidates = [n for n in frontier_content
                  if n.depth() >= 2 or n.count >= median_count]
    if not candidates:
        candidates = list(frontier_content)
    if not candidates:
        return ("<no candidates>", None) if return_leaf else "<no candidates>"
    weights = [max(1, n.count) * (1 + n.depth()) for n in candidates]
    chosen = random.choices(candidates,
                            weights=[max(w, 1e-12) for w in weights],
                            k=1)[0]
    leaf = _representative_leaf(chosen)
    try:
        text, parse = webster.generate_sentence(
            start_content_leaf=leaf, debug=False)
    except Exception as e:
        text, parse = f"<gen failed: {e}>", None
    if return_leaf:
        return text, parse, chosen, leaf
    return text

print(f"  Generating {N_GEN} sentences per frontier...")
gen_eval_rows = []
gen_samples_per_frontier: dict = {}
for f_name in ["fixed_d1", "fixed_d2", "fixed_d3", "basic_level"]:
    f_cnt = FRONTIERS_CONTENT[f_name]
    if not f_cnt: continue
    sentences = []
    lex_ok = gram_ok = const_ok = 0
    for _ in range(N_GEN):
        s = generate_from_frontier(f_cnt)
        sentences.append(s)
        toks = s.split()
        if toks and all(t in WORD_TO_POS for t in toks):
            lex_ok += 1
            if _grammar_recognize(toks):
                gram_ok += 1
            if _is_valid_constituent(toks):
                const_ok += 1
    gen_eval_rows.append({
        "frontier":           f_name,
        "n_gen":              N_GEN,
        "lex_ok_rate":        lex_ok   / max(N_GEN, 1),
        "constituent_rate":   const_ok / max(N_GEN, 1),
        "gram_ok_rate":       gram_ok  / max(N_GEN, 1),
    })
    gen_samples_per_frontier[f_name] = sentences
    print(f"  {f_name:<14} : in-lex {100*lex_ok/N_GEN:>5.1f}% | "
          f"constituent {100*const_ok/N_GEN:>5.1f}% | "
          f"full sentence {100*gram_ok/N_GEN:>5.1f}%")

with open(os.path.join(OUT_DIR, "gen_eval.csv"), "w") as f:
    w = csv.DictWriter(f, fieldnames=["frontier", "n_gen",
                                       "lex_ok_rate",
                                       "constituent_rate",
                                       "gram_ok_rate"])
    w.writeheader()
    for r in gen_eval_rows:
        w.writerow(r)

with open(os.path.join(OUT_DIR, "gen_samples.txt"), "w") as f:
    for f_name, samples in gen_samples_per_frontier.items():
        f.write(f"=== {f_name} ===\n")
        for i, s in enumerate(samples):
            f.write(f"  [{i+1:>2}] {s}\n")
        f.write("\n")


# =============================================================================
# PHASE 4 — Per-config comparison plots
# =============================================================================
print("\n=== PHASE 4: Comparison plots ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cfgs   = [r["config"] for r in parse_eval_rows]
sp_acc = [r["step_pick"] for r in parse_eval_rows]
f1     = [r["f1"]        for r in parse_eval_rows]
exact  = [r["exact_match"] for r in parse_eval_rows]
prec   = [r["precision"] for r in parse_eval_rows]
rec    = [r["recall"]    for r in parse_eval_rows]
is_baseline = [r["config"] == "BASELINE_climb" for r in parse_eval_rows]

x = np.arange(len(cfgs))
w = 0.35
ax = axes[0]
ax.bar(x - w/2, sp_acc, w, label="step-pick",
       color=["#7f7f7f" if b else "#1f77b4" for b in is_baseline],
       edgecolor="black", linewidth=0.5)
ax.bar(x + w/2, exact, w, label="exact-match",
       color=["#3f3f3f" if b else "#9467bd" for b in is_baseline],
       edgecolor="black", linewidth=0.5)
for i, (sa, ex) in enumerate(zip(sp_acc, exact)):
    ax.text(x[i] - w/2, sa + 0.02, f"{100*sa:.0f}", ha="center", fontsize=7)
    ax.text(x[i] + w/2, ex + 0.02, f"{100*ex:.0f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy")
ax.set_title("Step-pick vs Exact-match parses (grey = build() baseline)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ww = 0.27
ax.bar(x - ww, prec, ww, label="Precision", color="#1f77b4")
ax.bar(x,      rec,  ww, label="Recall",    color="#2ca02c")
ax.bar(x + ww, f1,   ww, label="F1",        color="#d62728")
for i in range(len(cfgs)):
    ax.text(x[i] - ww, prec[i] + 0.02, f"{100*prec[i]:.0f}", ha="center", fontsize=7)
    ax.text(x[i],      rec[i]  + 0.02, f"{100*rec[i]:.0f}",  ha="center", fontsize=7)
    ax.text(x[i] + ww, f1[i]   + 0.02, f"{100*f1[i]:.0f}",   ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
ax.set_title("Bracket P / R / F1 (per frontier × mode)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

plt.suptitle("Grammar Distillation — Parsing performance per frontier × scoring mode",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, "parse_comparison.png"), dpi=140, bbox_inches="tight")
plt.close()
print(f"  Parse comparison → {OUT_DIR}/parse_comparison.png")

if gen_eval_rows:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fnames      = [r["frontier"] for r in gen_eval_rows]
    lex_rates   = [r["lex_ok_rate"]      for r in gen_eval_rows]
    const_rates = [r["constituent_rate"] for r in gen_eval_rows]
    gram_rates  = [r["gram_ok_rate"]     for r in gen_eval_rows]
    x = np.arange(len(fnames)); w = 0.27
    ax.bar(x - w, lex_rates,   w, label="in-lexicon",
           color="#17becf", edgecolor="black", linewidth=0.5)
    ax.bar(x,     const_rates, w, label="valid constituent (any NT)",
           color="#bcbd22", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, gram_rates,  w, label="full sentence (S)",
           color="#9467bd", edgecolor="black", linewidth=0.5)
    for i in range(len(fnames)):
        ax.text(x[i] - w, lex_rates[i]   + 0.02,
                f"{100*lex_rates[i]:.0f}%",   ha="center", fontsize=7)
        ax.text(x[i],     const_rates[i] + 0.02,
                f"{100*const_rates[i]:.0f}%", ha="center", fontsize=7)
        ax.text(x[i] + w, gram_rates[i]  + 0.02,
                f"{100*gram_rates[i]:.0f}%",  ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(fnames, fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Rate")
    ax.set_title(f"Generation quality per frontier  (n={N_GEN} per frontier)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "generation_comparison.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Generation comparison → {OUT_DIR}/generation_comparison.png")


# =============================================================================
# PHASE 5 — Distilled grammar (UNSUPERVISED)
# =============================================================================
# Goal: reconstruct a CFG from the trained cobweb tree alone, WITHOUT
# using any gold labels (no head-based chunk classes, no gold merges).
#
# Pipeline:
#   1. Name each frontier non-terminal by a pure CLUSTER IDENTITY
#      (NT_0, NT_1, …) and annotate it with its representative
#      terminal tokens — the words that most commonly appear under
#      this NT in WEBSTER's own auto-parses of the training corpus.
#   2. Extract production rules from WEBSTER's AUTO-PARSES of the
#      training sentences (no gold merges, no class labels). Each
#      composite contributes one (parent_NT, left_NT, right_NT)
#      observation; left/right may also be terminal words for
#      primitive children.
#   3. Greedy reconstruction: sort all observed productions by count
#      and add them to the distilled grammar one at a time. Track
#      what fraction of all observed composites is covered by the
#      first K rules. This gives a "grammar growth" curve that
#      directly visualizes how compact the learned CFG is.
#   4. (Sanity check, NOT used in the grammar itself) compare each
#      NT's dominant gold chunk class to confirm the unsupervised
#      cluster identity tracks a real linguistic category.
#
# Renderers: rules table + node-link graph + per-sentence derivation
# trees for parse and generation, all using the unsupervised labels.
# =============================================================================
print("\n=== PHASE 5: Distilled grammar (UNSUPERVISED) ===")

# ── Generate the corpus we'll use for distillation. Pure auto-parses,
# no hollow gold. Sentences are sampled fresh from the grammar — this
# is the *only* source of structure for the distillation step.
DISTILL_N_SENTENCES = 200
distill_sentences = [generate("S", TEST_GRAMMAR1)
                     for _ in range(DISTILL_N_SENTENCES)]
print(f"  Distillation corpus: {len(distill_sentences)} auto-parsed sentences")


def label_frontier_unsupervised(frontier_nodes, sentences):
    """Name each frontier node by a pure cluster identity (NT_0, NT_1,
    …) and gather the representative tokens that appear under it in
    WEBSTER's own auto-parses. No gold class labels involved.

    Returns ``(labels, tokens_per_nt, gold_classes_per_nt)`` where
    ``labels[concept_hash] = "NT_<i>"``, ``tokens_per_nt[label]`` is a
    ``Counter`` of terminal tokens that fall under the NT during
    auto-parsing, and ``gold_classes_per_nt[label]`` is a sanity
    Counter of gold head-based chunk classes (informational only,
    not part of the grammar)."""
    # Order frontier nodes deterministically by count desc, then hash
    # so the NT_<i> indices are stable across runs.
    ordered = sorted(frontier_nodes,
                     key=lambda n: (-int(n.count), str(n.concept_hash())))
    labels = {}
    for i, n in enumerate(ordered):
        labels[str(n.concept_hash())] = f"NT_{i}"

    tokens_per_nt = defaultdict(Counter)
    gold_per_nt   = defaultdict(Counter)
    for sent in sentences:
        try:
            parse = webster.parse_sentence(
                sent, threshold=THRESHOLD, new_vocab=False,
                learning=False, debug=False)
        except Exception:
            continue
        sent_len = len(sent.split())
        for comp in _walk_composites(parse.global_root_node):
            ci = comp.get_content_instance()
            if not ci: continue
            best, _ = best_frontier_label(ci, frontier_nodes)
            if best is None: continue
            lab = labels[str(best.concept_hash())]
            tokens_per_nt[lab].update(_chunk_tokens(comp))
            cls = classify_chunk(comp, sent_len)
            if cls is not None:
                gold_per_nt[lab][cls] += 1
    return labels, tokens_per_nt, gold_per_nt


def extract_rules_unsupervised(frontier_nodes, frontier_labels, sentences):
    """Extract production rules from WEBSTER's auto-parses (NO gold
    merges). For each composite C produced by build():
        parent_label = NT label of C.content_instance
        left_label   = NT label of C.left_child, OR "'<word>'" if primitive
        right_label  = same for right child
    Aggregate (parent, (left, right)) counts and return as production
    rules + an NT-frequency Counter."""
    rules = defaultdict(Counter)
    nt_count = Counter()

    def label_node(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is not None and 0 <= wid < len(webster.ltm.id_to_value):
                return f"'{webster.ltm.id_to_value[wid]}'"
            return "?"
        ci = n.get_content_instance()
        if not ci: return "?"
        best, _ = best_frontier_label(ci, frontier_nodes)
        if best is None: return "?"
        return frontier_labels.get(str(best.concept_hash()), "?")

    for sent in sentences:
        try:
            parse = webster.parse_sentence(
                sent, threshold=THRESHOLD, new_vocab=False,
                learning=False, debug=False)
        except Exception:
            continue
        for comp in _walk_composites(parse.global_root_node):
            parent = label_node(comp)
            children = sorted(comp.children, key=lambda x: x[0])
            if len(children) != 2: continue
            l = label_node(children[0][1])
            r = label_node(children[1][1])
            nt_count[parent] += 1
            rules[parent][(l, r)] += 1
    return rules, nt_count


def greedy_reconstruct(rules, nt_count):
    """Build the distilled grammar greedily by adding production rules
    in count-descending order. Returns ``ordered_rules`` (list of
    (parent, (l, r), count)) and ``coverage`` (a parallel list of
    cumulative fraction-of-composites-explained values)."""
    # Flatten to (parent, (l, r), count), sort by count desc.
    flat = []
    for parent, rhs_counter in rules.items():
        for rhs, c in rhs_counter.items():
            flat.append((parent, rhs, c))
    flat.sort(key=lambda t: -t[2])
    total = sum(c for *_, c in flat) or 1
    cumulative = 0
    coverage = []
    ordered = []
    for triple in flat:
        cumulative += triple[2]
        ordered.append(triple)
        coverage.append(cumulative / total)
    return ordered, coverage


def label_with_tokens(label, tokens_per_nt, gold_per_nt, top_k=3):
    """Return a display string that combines the unsupervised cluster
    id with its top-K representative tokens, plus a sanity tag for the
    dominant gold class (in brackets, for human readability only)."""
    toks = tokens_per_nt.get(label, Counter()).most_common(top_k)
    top_tok_str = "/".join(t for t, _ in toks) if toks else "?"
    gold = gold_per_nt.get(label, Counter()).most_common(1)
    gold_tag = f"  [{gold[0][0]}]" if gold else ""
    return f"{label}[{top_tok_str}]{gold_tag}"

def plot_rules(rules, nt_count, tokens_per_nt, gold_per_nt,
               frontier_name, out_path, top_k=4):
    """Render the distilled grammar as CFG-style production rules.
    Each NT is shown with its top-K representative tokens (from
    auto-parses) AND the dominant gold class in brackets (sanity
    check only — not used in the rule extraction)."""
    nts = sorted(rules.keys(), key=lambda n: -nt_count.get(n, 0))
    nts = [n for n in nts if nt_count.get(n, 0) >= 3 or len(rules[n]) >= 1]
    if not nts:
        return
    fig_h = max(2.5, 0.55 * sum(min(top_k, len(rules[n])) + 1.5 for n in nts))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.99,
            f"Distilled CFG (UNSUPERVISED) — {frontier_name} frontier",
            ha="center", va="top", fontsize=15, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.96,
            f"{len(nts)} non-terminals · auto-parses of "
            f"{DISTILL_N_SENTENCES} sentences · "
            f"productions ranked by count (log p = log p(RHS | LHS))",
            ha="center", va="top", fontsize=10, color="#555",
            transform=ax.transAxes)
    ax.text(0.5, 0.937,
            "Cluster identity → NT_<i> · representative tokens in [..] "
            "· gold-class sanity tag in {..} (not in grammar)",
            ha="center", va="top", fontsize=9, color="#777",
            style="italic", transform=ax.transAxes)

    y = 0.90
    line_h = 0.022
    for nt in nts:
        # Color: derive from the dominant gold class IF we have one
        # (purely for visual grouping — the grammar itself doesn't use it).
        gold = gold_per_nt.get(nt, Counter()).most_common(1)
        cls = gold[0][0] if gold else "OTHER"
        col = LABEL_COLOR.get(cls, "#666")

        # Representative tokens column.
        toks = tokens_per_nt.get(nt, Counter()).most_common(3)
        tok_str = "/".join(t for t, _ in toks) if toks else "?"
        gold_str = f"{{ {cls} }}" if gold else "{ ? }"

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - line_h * 0.9), 0.06, line_h * 1.1,
            boxstyle="round,pad=0.005", linewidth=0.6,
            facecolor=col, alpha=0.7, edgecolor="black",
            transform=ax.transAxes))
        ax.text(0.05, y - line_h * 0.35, nt,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if cls in
                ["S","Det","P","N","VP","V"] else "black",
                transform=ax.transAxes)
        ax.text(0.085, y - line_h * 0.35,
                f" [{tok_str}]  {gold_str}   "
                f"(seen as parent: {nt_count.get(nt, 0)}×)",
                ha="left", va="center", fontsize=9, color="#444",
                transform=ax.transAxes)
        y -= line_h

        prods = sorted(rules[nt].items(), key=lambda kv: -kv[1])[:top_k]
        total = sum(rules[nt].values()) or 1
        for (l, r), c in prods:
            log_p = math.log(c / total) if c > 0 else float("-inf")
            ax.text(0.15, y - line_h * 0.35,
                    f"{nt}  →  ({l},  {r})",
                    ha="left", va="center", fontsize=9,
                    fontfamily="monospace", transform=ax.transAxes)
            ax.text(0.78, y - line_h * 0.35,
                    f"count={c:>4}    log p = {log_p:>6.2f}",
                    ha="left", va="center", fontsize=9, color="#888",
                    fontfamily="monospace", transform=ax.transAxes)
            y -= line_h
        y -= 0.005
        if y < 0.02:
            break

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_coverage_curve(ordered_rules, coverage, frontier_name, out_path):
    """The greedy reconstruction curve: x = # rules added (in
    count-descending order), y = cumulative fraction of all
    auto-parse composites explained. A steep curve means a few
    high-count rules cover most of the data — i.e. the distilled
    grammar is compact."""
    if not ordered_rules:
        return
    ks = list(range(1, len(ordered_rules) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, coverage, color="#1f77b4", linewidth=2,
            marker="o", markersize=3)
    ax.fill_between(ks, 0, coverage, alpha=0.15, color="#1f77b4")
    # Annotate 50%, 80%, 95% milestones.
    for milestone in [0.5, 0.8, 0.95]:
        for k, cov in zip(ks, coverage):
            if cov >= milestone:
                ax.axhline(milestone, color="#888", linestyle=":",
                            linewidth=0.7, alpha=0.6)
                ax.plot(k, cov, marker="o", markersize=8,
                        color="#d62728", zorder=5)
                ax.annotate(f"{int(milestone*100)}% @ {k} rules",
                            xy=(k, cov), xytext=(k + 1, cov - 0.04),
                            fontsize=9, color="#d62728",
                            arrowprops=dict(arrowstyle="-",
                                             color="#d62728", alpha=0.5))
                break
    ax.set_xlabel("# rules included (greedy, count-desc)")
    ax.set_ylabel("Fraction of auto-parse composites covered")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(ks) + 1)
    ax.set_title(
        f"Greedy grammar reconstruction — {frontier_name} frontier  "
        f"({len(ordered_rules)} unique productions, "
        f"{sum(c for *_, c in ordered_rules)} observations)")
    ax.grid(axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

def plot_node_link(rules, nt_count, frontier_name, out_path):
    """Directed graph: edges parent → (left, right) child non-terminals,
    weighted by co-occurrence count.  Terminals (words) are shown as
    smaller leaf nodes with thinner edges."""
    # Collect nodes (only non-terminals for layout; terminals are
    # rendered as small text leaves attached to their producing NT).
    edges = defaultdict(int)   # (parent_label, child_label) → count
    is_nt = lambda lab: not (lab.startswith("'") and lab.endswith("'"))
    for parent, child_counter in rules.items():
        for (l, r), c in child_counter.items():
            edges[(parent, l)] += c
            edges[(parent, r)] += c

    nts = sorted(set([p for p, _ in edges.keys()] +
                     [c for _, c in edges.keys() if is_nt(c)]),
                 key=lambda n: -nt_count.get(n, 0))
    if not nts:
        return

    # Simple radial layout: place NTs around a circle.
    n_nt = len(nts)
    angles = np.linspace(0, 2 * np.pi, n_nt, endpoint=False)
    coords = {nt: (np.cos(a), np.sin(a)) for nt, a in zip(nts, angles)}

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    ax.set_title(
        f"Distilled grammar — node-link graph ({frontier_name} frontier)\n"
        f"edges: parent → child non-terminal, thickness = count",
        fontsize=12, pad=10)

    # Edges (NT → NT only; word terminals listed as small text labels)
    max_w = max(edges.values()) if edges else 1
    for (src, dst), w in edges.items():
        if src not in coords or dst not in coords: continue
        if src == dst: continue
        x0, y0 = coords[src]; x1, y1 = coords[dst]
        # Slight curvature to distinguish src/dst.
        thickness = 0.5 + 4 * (w / max_w)
        alpha = 0.4 + 0.5 * (w / max_w)
        ax.annotate(
            "", xy=(x1 * 0.93, y1 * 0.93),
            xytext=(x0 * 0.93, y0 * 0.93),
            arrowprops=dict(arrowstyle="->",
                            color="#444",
                            lw=thickness, alpha=alpha,
                            connectionstyle="arc3,rad=0.12"))

    # NT chips
    for nt, (x, y) in coords.items():
        cls = nt.split("_")[1] if nt.startswith("NT_") else "OTHER"
        col = LABEL_COLOR.get(cls, "#888")
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.13, y - 0.04), 0.26, 0.08,
            boxstyle="round,pad=0.01",
            facecolor=col, alpha=0.85,
            edgecolor="black", linewidth=1))
        ax.text(x, y, nt, ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color="white" if cls in ["S","Det","P","N","VP","V"]
                      else "black")
        ax.text(x, y - 0.075, f"n={nt_count.get(nt, 0)}",
                ha="center", va="center", fontsize=7, color="#444")

    # Terminal words: collect distinct terminal RHSs and list them in
    # a side panel (terminals are often many words — we don't want
    # them cluttering the graph).
    term_set = set()
    for parent, child_counter in rules.items():
        for (l, r), _ in child_counter.items():
            if not is_nt(l): term_set.add(l)
            if not is_nt(r): term_set.add(r)
    if term_set:
        ax.text(1.4, 1.3, "Terminals (words) seen:",
                ha="right", va="top", fontsize=9, fontweight="bold")
        terms = sorted(term_set)
        line = ", ".join(terms[:12])
        if len(terms) > 12:
            line += f",  …  (+{len(terms)-12} more)"
        ax.text(1.4, 1.22, line, ha="right", va="top",
                fontsize=8, color="#555")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


# ── Tree-drawing helpers (used for parse + generation derivations) ──────────
def _draw_tree(root, label_fn, child_fn, out_path, title,
               nt_color_fn=None, lp_fn=None):
    """Generic tree-plotter. root → start node; child_fn(node) returns
    list of children; label_fn(node) returns the display string;
    nt_color_fn(node) optional → color string; lp_fn(node) optional →
    log-prob to display as a secondary line under each node."""
    # Compute layout via DFS.
    positions = {}            # node → (x, y)
    next_x = [0.0]
    def layout(n, depth):
        children = child_fn(n)
        if not children:
            positions[id(n)] = (next_x[0], -depth, n)
            next_x[0] += 1.0
            return
        for c in children:
            layout(c, depth + 1)
        # x = midpoint of children
        cxs = [positions[id(c)][0] for c in children]
        positions[id(n)] = (sum(cxs) / len(cxs), -depth, n)
    layout(root, 0)

    if not positions:
        return
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    fig_w = max(8, 1.1 * (max(xs) - min(xs) + 2))
    fig_h = max(4, 1.0 * (max(ys) - min(ys) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    # Edges
    def draw_edges(n):
        x0, y0, _ = positions[id(n)]
        for c in child_fn(n):
            x1, y1, _ = positions[id(c)]
            ax.plot([x0, x1], [y0 - 0.18, y1 + 0.18],
                    color="#888", linewidth=1.0, zorder=0)
            draw_edges(c)
    draw_edges(root)

    # Nodes
    for nid, (x, y, n) in positions.items():
        lab = label_fn(n)
        is_leaf = not child_fn(n)
        col = (nt_color_fn(n) if nt_color_fn else None) or (
            "#fff4d6" if is_leaf else "#cfe7ff")
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.45, y - 0.18), 0.9, 0.36,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor="black", linewidth=0.7))
        ax.text(x, y + 0.04, lab,
                ha="center", va="center", fontsize=9,
                fontweight="bold" if not is_leaf else "normal")
        if lp_fn is not None and not is_leaf:
            lp_val = lp_fn(n)
            if lp_val is not None and math.isfinite(lp_val):
                ax.text(x, y - 0.09, f"log p = {lp_val:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="#444", fontfamily="monospace")

    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _label_color_from_gold(nt_label, gold_per_nt):
    """Color a non-terminal box by its dominant gold class if known
    (sanity coloring only — doesn't affect grammar semantics)."""
    gold = gold_per_nt.get(nt_label, Counter()).most_common(1)
    cls = gold[0][0] if gold else "OTHER"
    return LABEL_COLOR.get(cls, "#cfe7ff")


def plot_parse_derivation(parse_tree, frontier_labels, frontier_nodes,
                          tokens_per_nt, gold_per_nt,
                          frontier_name, sentence, out_path):
    """Render the parse tree, annotating composites with their
    unsupervised NT label (+ representative tokens) + log-prob, and
    primitives with their token. Box colors come from the gold
    sanity tag, NOT from the grammar itself."""
    root = parse_tree.global_root_node
    top_children = [c[1] for c in root.children]
    if not top_children:
        return
    fake_root = top_children[0] if len(top_children) == 1 else root

    def child_fn(n):
        if isinstance(n, PrimitiveParseNode): return []
        return [c[1] for c in n.children]

    def _nt_for(n):
        if isinstance(n, PrimitiveParseNode): return None
        nt = getattr(n, "_frontier_label", None)
        if nt is None:
            ci = n.get_content_instance()
            if ci:
                nt, _ = best_frontier_label(ci, frontier_nodes)
        return nt

    def label_fn(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is not None and 0 <= wid < len(webster.ltm.id_to_value):
                return f'"{webster.ltm.id_to_value[wid]}"'
            return "?"
        if getattr(n, "is_global_root", False):
            return "ROOT"
        nt = _nt_for(n)
        if nt is None: return "?"
        lab = frontier_labels.get(str(nt.concept_hash()), "?")
        toks = tokens_per_nt.get(lab, Counter()).most_common(2)
        tok_str = "/".join(t for t, _ in toks) if toks else ""
        return f"{lab}\n[{tok_str}]" if tok_str else lab

    def lp_fn(n):
        if isinstance(n, PrimitiveParseNode): return None
        return getattr(n, "_frontier_logp", None)

    def color_fn(n):
        if isinstance(n, PrimitiveParseNode):
            return "#fff4d6"
        nt = _nt_for(n)
        if nt is None: return "#cfe7ff"
        lab = frontier_labels.get(str(nt.concept_hash()), "?")
        return _label_color_from_gold(lab, gold_per_nt)

    _draw_tree(fake_root, label_fn, child_fn, out_path,
               title=(f"Parse derivation (UNSUPERVISED labels) — "
                      f"{frontier_name} frontier\n"
                      f"\"{sentence}\""),
               nt_color_fn=color_fn, lp_fn=lp_fn)


def plot_generation_derivation(parse_tree, frontier_labels, frontier_nodes,
                                 tokens_per_nt, gold_per_nt,
                                 frontier_name, seed_label, seed_node,
                                 gen_text, out_path):
    """Same as parse derivation but titled 'Generation unpacking'.
    The root composite from webster.generate_sentence() has no
    content_instance, so we override the root's label with the
    seed_label that drove this generation."""
    root = parse_tree.global_root_node
    top_children = [c[1] for c in root.children]
    if not top_children: return
    fake_root = top_children[0] if len(top_children) == 1 else root
    fake_root_id = id(fake_root)

    def child_fn(n):
        if isinstance(n, PrimitiveParseNode): return []
        return [c[1] for c in n.children]

    def _nt_for(n):
        if isinstance(n, PrimitiveParseNode): return None
        if id(n) == fake_root_id: return seed_node
        ci = n.get_content_instance()
        if ci:
            nt, _ = best_frontier_label(ci, frontier_nodes)
            return nt
        return None

    def label_fn(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is not None and 0 <= wid < len(webster.ltm.id_to_value):
                return f'"{webster.ltm.id_to_value[wid]}"'
            return "?"
        nt = _nt_for(n)
        if nt is None: return "?"
        lab = frontier_labels.get(str(nt.concept_hash()), "?")
        toks = tokens_per_nt.get(lab, Counter()).most_common(2)
        tok_str = "/".join(t for t, _ in toks) if toks else ""
        return f"{lab}\n[{tok_str}]" if tok_str else lab

    def lp_fn(n):
        if isinstance(n, PrimitiveParseNode): return None
        if id(n) == fake_root_id: return None
        ci = n.get_content_instance()
        if ci:
            _, lp = best_frontier_label(ci, frontier_nodes)
            return lp
        return None

    def color_fn(n):
        if isinstance(n, PrimitiveParseNode):
            return "#fff4d6"
        nt = _nt_for(n)
        if nt is None: return "#cfe7ff"
        lab = frontier_labels.get(str(nt.concept_hash()), "?")
        return _label_color_from_gold(lab, gold_per_nt)

    seed_info = ""
    if seed_node is not None:
        seed_info = (f"    (seed count = {int(seed_node.count)}, "
                     f"depth = {seed_node.depth()})")
    _draw_tree(fake_root, label_fn, child_fn, out_path,
               title=(f"Generation unpacking (UNSUPERVISED labels) — "
                      f"{frontier_name} frontier\n"
                      f"seed = {seed_label}{seed_info}    →    "
                      f"\"{gen_text}\""),
               nt_color_fn=color_fn, lp_fn=lp_fn)


# Build everything for the two most informative frontiers.
for f_name in ["fixed_d3", "basic_level"]:
    f_cnt = FRONTIERS_CONTENT[f_name]
    f_ctx = FRONTIERS_CONTEXT[f_name]
    if not f_cnt: continue
    print(f"\n  Distilling for frontier = {f_name} "
          f"(|content|={len(f_cnt)})")

    # 1. Unsupervised NT naming (NT_0, NT_1, …) + representative tokens.
    labels, tokens_per_nt, gold_per_nt = label_frontier_unsupervised(
        f_cnt, distill_sentences)
    print(f"    Non-terminals (unsupervised cluster id · top tokens · "
          f"gold sanity tag):")
    for n in sorted(f_cnt, key=lambda n: -int(n.count)):
        h = str(n.concept_hash())
        lab = labels.get(h, "?")
        toks = tokens_per_nt.get(lab, Counter()).most_common(3)
        tok_str = "/".join(t for t, _ in toks) if toks else "—"
        gold = gold_per_nt.get(lab, Counter()).most_common(1)
        gold_str = (f"{{ {gold[0][0]}: {gold[0][1]}/"
                    f"{sum(gold_per_nt[lab].values())} }}"
                    if gold else "{ — }")
        print(f"      {lab:<6} [{tok_str:<22}]  {gold_str:<22}  "
              f"cluster_count={int(n.count):>4}  depth={n.depth()}")

    # 2. Extract production rules from WEBSTER's auto-parses (no gold).
    rules, nt_count = extract_rules_unsupervised(
        f_cnt, labels, distill_sentences)
    n_total_rules = sum(len(v) for v in rules.values())
    n_observations = sum(c for cnts in rules.values()
                          for c in cnts.values())
    print(f"    Distilled grammar: {n_total_rules} unique productions "
          f"across {len(rules)} non-terminals "
          f"({n_observations} composite observations)")

    # 3. Greedy reconstruction in count-descending order.
    ordered_rules, coverage = greedy_reconstruct(rules, nt_count)
    if coverage:
        for milestone in [0.5, 0.8, 0.95]:
            for k, cov in enumerate(coverage, 1):
                if cov >= milestone:
                    print(f"    Greedy coverage: "
                          f"{int(milestone*100)}% reached at rule #{k}")
                    break
        print(f"    Top-5 productions (greedy order):")
        for parent, (l, r), c in ordered_rules[:5]:
            print(f"      [{c:>3}]  {parent}  →  ({l},  {r})")

    # 4. Persist rules CSV (sorted greedily).
    with open(os.path.join(OUT_DIR, f"rules_{f_name}.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["rank", "lhs", "left", "right", "count", "log_p",
                    "cumulative_coverage"])
        for i, ((parent, (l, r), c), cov) in enumerate(
                zip(ordered_rules, coverage)):
            total = sum(rules[parent].values()) or 1
            w.writerow([i + 1, parent, l, r, c,
                        f"{math.log(c/total):.4f}" if c else "",
                        f"{cov:.4f}"])

    # 5. Plot rules table + node-link + coverage curve.
    plot_rules(rules, nt_count, tokens_per_nt, gold_per_nt, f_name,
               os.path.join(OUT_DIR, f"rules_{f_name}.png"), top_k=4)
    plot_node_link(rules, nt_count, f_name,
                   os.path.join(OUT_DIR, f"node_link_{f_name}.png"))
    plot_coverage_curve(ordered_rules, coverage, f_name,
                        os.path.join(OUT_DIR, f"coverage_{f_name}.png"))
    print(f"    Rules table     → rules_{f_name}.png")
    print(f"    Node-link       → node_link_{f_name}.png")
    print(f"    Coverage curve  → coverage_{f_name}.png")

    # 6. Sample parse derivations on test-fold sentences (still
    # auto-parsed; we just *show* the labels to the user).
    sample_sents = [h["sentence"] for h in test_hollow[:N_DERIV_VIZ]]
    for i, sent in enumerate(sample_sents):
        pt = parse_with_frontier(sent, f_cnt, f_ctx,
                                 mode="iterate", record_labels=True)
        out = os.path.join(DERIV_DIR,
                           f"derivation_parse_{f_name}_{i}.png")
        plot_parse_derivation(pt, labels, f_cnt, tokens_per_nt,
                              gold_per_nt, f_name, sent, out)
    print(f"    Parse derivations → derivations/derivation_parse_{f_name}_*.png "
          f"(n={len(sample_sents)})")

    # 7. Sample generation derivations.
    for i in range(N_DERIV_VIZ):
        try:
            text, parse, seed_node, _ = generate_from_frontier(
                f_cnt, return_leaf=True)
        except Exception:
            continue
        if parse is None: continue
        seed_label = labels.get(str(seed_node.concept_hash()), "?")
        out = os.path.join(DERIV_DIR,
                           f"derivation_gen_{f_name}_{i}.png")
        plot_generation_derivation(parse, labels, f_cnt, tokens_per_nt,
                                   gold_per_nt, f_name,
                                   seed_label, seed_node, text, out)
    print(f"    Gen derivations   → derivations/derivation_gen_{f_name}_*.png "
          f"(n={N_DERIV_VIZ})")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
best_parse = max(parse_eval_rows, key=lambda r: r["f1"]) if parse_eval_rows else None
if best_parse:
    print(f"  Best parsing config: {best_parse['config']}  "
          f"F1={100*best_parse['f1']:.1f}%  "
          f"step-pick={100*best_parse['step_pick']:.1f}%  "
          f"exact-match={100*best_parse['exact_match']:.1f}%")
best_gen = max(gen_eval_rows, key=lambda r: r["constituent_rate"]) \
           if gen_eval_rows else None
if best_gen:
    print(f"  Best generation:     {best_gen['frontier']}  "
          f"constituent={100*best_gen['constituent_rate']:.1f}%  "
          f"in-lexicon={100*best_gen['lex_ok_rate']:.1f}%")

print(f"\nArtefacts in {OUT_DIR}/:")
print("  frontier_summary.csv, parse_eval.csv, gen_eval.csv, gen_samples.txt,")
print("  rules_{fixed_d3,basic_level}.csv + .png   (distilled UNSUPERVISED CFG),")
print("  coverage_{fixed_d3,basic_level}.png       (greedy reconstruction curve),")
print("  node_link_{fixed_d3,basic_level}.png,")
print("  derivations/derivation_{parse,gen}_*.png,")
print("  parse_comparison.png, generation_comparison.png")
