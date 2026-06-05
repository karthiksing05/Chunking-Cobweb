"""
Grammar Distillation Test (met5) — fixed_d3 vanilla vs redistributed
====================================================================

After TRELLIS training, the FIXED_D3 frontier with categorize-mode
scoring is the winner among all the distillation strategies we tried
(F1 ≈ 83%, step-pick ≈ 90% — beats the climbing-ancestor baseline).
This test isolates that strategy and compares two variants:

  * VANILLA   — train, freeze, extract the d3 frontier as-is.
  * REDIST    — train, then call ``tree.redistribute(N)`` on the
                CONTENT cobweb tree. Cobweb redistributes "misplaced"
                nodes (those whose current parent is not their best
                fit) by removing them and re-inserting them via normal
                Cobweb-CU descent. The d3 frontier is then re-extracted
                from the now-cleaner content tree. (The context tree is
                left untouched — the TopK-Pool encoder caches its
                depth-d node ids as the stable substrate for content
                bags, and moving context-tree nodes around invalidates
                stored chunk representations in ways that aren't
                recoverable by sync_remap alone.)

Both variants run through the same pipeline:

  * Parse evaluation (categorize-mode scoring):
      - end-to-end bracket P / R / F1
      - exact-match parses
      - step-pick accuracy on gold-replay
  * Grammar distillation (unsupervised):
      - NT_<i> labels + representative tokens
      - production rules from auto-parses
      - greedy reconstruction with coverage milestones
  * Visualizations:
      - cobweb tree-with-bars (POS / chunk-class distributions per
        node, frontier highlighted)
      - rules table
      - node-link graph
      - per-sentence parse + generation derivations
      - coverage curve

Outputs (under ``grammar_distillation_test_output/``):

  parse_comparison.png            — F1/step-pick/exact-match bar chart
  vanilla/<artefacts>             — every distillation artefact for the
                                    pre-redistribution d3 frontier
  redist/<artefacts>              — same for the redistributed d3 frontier
"""

import os
import sys
import csv
import glob
import json
import math
import random
import shutil
import contextlib
from collections import Counter, defaultdict
from functools import lru_cache as _lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

# Cobweb's C++ code prints PMI debug info from inside get_basic() and
# the redistribute progress lines. They drown out our Python logs and
# bloat the output to gigabytes. Silence stdout (fd 1) at the C level
# for the duration of the run, and route Python progress prints to
# stderr instead.
@contextlib.contextmanager
def silence_stdout_fd():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)

def log(*args, **kwargs):
    """Status print that always reaches the user (via stderr)."""
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import (TRELLIS, FiniteParseTree, PrimitiveParseNode,
                      CompositeParseNode)
from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed

# ── Configuration ─────────────────────────────────────────────────────────────
OUT_DIR             = os.path.join(_HERE, "grammar_distillation_test_output")
HOLLOW_CORPUS_DIR   = "data/test_hollow_grammar_1"
CONTEXT_LENGTH      = 3
THRESHOLD           = 30
PRIMITIVES_FIRST    = 200
EVAL_ALPHA          = 10.0
SEED                = 13
N_GEN               = 30
N_DERIV_VIZ         = 5
DISTILL_N_SENTENCES = 200
REDIST_N            = 500       # cobweb redistribute steps per tree
                                # (auto-stops if no progress; 500 is a
                                # safe upper bound that usually finishes
                                # in a fraction of that)

random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
VANILLA_DIR  = os.path.join(OUT_DIR, "vanilla")
REDIST_DIR   = os.path.join(OUT_DIR, "redist")
BFS_PURE_DIR = os.path.join(OUT_DIR, "bfs_pure")
for d in [VANILLA_DIR, REDIST_DIR, BFS_PURE_DIR]:
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "derivations"), exist_ok=True)

# BFS-pure frontier parameters.
PURITY_THRESHOLD = 0.85   # min dominant-class share to consider a node 'pure'
PURITY_MAX_DEPTH = 4      # hard cap on BFS descent
PURITY_MIN_COUNT = 5      # min observations to evaluate purity

POS_LIST = ["Det", "N", "Adj", "V", "P"]
WORD_TO_POS = {}
for pos in POS_LIST:
    for prod in TEST_GRAMMAR1[pos]:
        for w in prod:
            WORD_TO_POS[w] = pos
CHUNK_LABELS = ["NP", "AdjP", "PP", "VP", "S"]
ALL_LABELS   = POS_LIST + CHUNK_LABELS + ["OTHER"]
LABEL_COLOR  = {
    "Det":   "#2ca02c", "N":     "#8c564b", "Adj":   "#1f77b4",
    "V":     "#17becf", "P":     "#7f7f7f", "NP":    "#ff7f0e",
    "AdjP":  "#9467bd", "PP":    "#bcbd22", "VP":    "#e377c2",
    "S":     "#d62728", "OTHER": "#cccccc",
}

PRIM_LABELS = POS_LIST
N_PRIM      = len(PRIM_LABELS)
_prim2id    = {p: i for i, p in enumerate(PRIM_LABELS)}
N_LABEL     = len(ALL_LABELS)
_label2id   = {lbl: i for i, lbl in enumerate(ALL_LABELS)}
TREE_DEPTH_FIG = 4


# =============================================================================
# PHASE 0 — Train TRELLIS
# =============================================================================
log("=== PHASE 0: Train TRELLIS ===")
# Cobweb's internal debug prints (PMI lines, redistribute progress)
# stream to stdout (fd 1) and bury our logs. Silence stdout at the
# file-descriptor level for the rest of the run; our progress goes
# through log() → stderr.
_silencer = silence_stdout_fd()
_silencer.__enter__()

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

log(f"  Phase 0a: {PRIMITIVES_FIRST} primitive-only sentences")
for i in range(PRIMITIVES_FIRST):
    s = generate("S", TEST_GRAMMAR1)
    trellis.parse_sentence(s, threshold=1e9, new_vocab=True,
                           learning=True, debug=False)
    if (i + 1) % 50 == 0: log(f"    [{i+1}/{PRIMITIVES_FIRST}]")

log(f"  Phase 0b: hollow corpus replay")
hollow_corpus: list = []
for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
    with open(p) as f:
        try: data = json.load(f)
        except json.JSONDecodeError: continue
    if "sentence" in data and "merges" in data:
        hollow_corpus.append(data)
log(f"    Loaded {len(hollow_corpus)} hollow trees")
random.shuffle(hollow_corpus)
_split = int(0.8 * len(hollow_corpus))
train_hollow = hollow_corpus[:_split]
test_hollow  = hollow_corpus[_split:]
log(f"    Split: train={len(train_hollow)}  test={len(test_hollow)}")
for i, hollow in enumerate(train_hollow):
    tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(hollow["sentence"], threshold=THRESHOLD)
    for m in hollow["merges"]:
        try: tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    trellis.ltm.add_parse_tree(tree, shuffle=True, debug=False)
    if (i + 1) % 25 == 0: log(f"    [{i+1}/{len(train_hollow)}]")

# Distillation corpus used in Phase 5 etc. — fresh auto-parsed sentences.
distill_sentences = [generate("S", TEST_GRAMMAR1)
                     for _ in range(DISTILL_N_SENTENCES)]
log(f"  Distillation corpus: {len(distill_sentences)} sentences")


# =============================================================================
# Shared helpers (only the bits we still need with d3+categorize)
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
            if wid is None or wid < 0 or wid >= len(trellis.ltm.id_to_value):
                return
            pos = WORD_TO_POS.get(trellis.ltm.id_to_value[wid])
            if pos: out.append(pos)
            return
        for _, c in sorted(getattr(n, "children", []),
                           key=lambda x: x[0] if x[0] is not None else 0):
            w(c)
    w(node)
    return out

def _chunk_tokens(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is not None and 0 <= wid < len(trellis.ltm.id_to_value):
                out.append((int(n.position_idx), trellis.ltm.id_to_value[wid]))
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

def fixed_depth_frontier(root, target_depth):
    """All nodes at exactly target_depth. Leaves shallower than that
    are also included so every leaf path is covered."""
    seen = {}
    def walk(node, d):
        if (d == target_depth) or (not node.children):
            seen[str(node.concept_hash())] = node
            return
        for c in node.children:
            walk(c, d + 1)
    walk(root, 0)
    return list(seen.values())


def collect_node_chunk_class_counts(root, sentences):
    """For every node visited while categorizing chunks from
    auto-parses, tally the gold head-based chunk class of each
    descending chunk. Used by ``bfs_pure_frontier`` to decide
    whether a node's cluster is 'pure'."""
    counts = defaultdict(Counter)
    for sent in sentences:
        try:
            parse = trellis.parse_sentence(
                sent, threshold=THRESHOLD, new_vocab=False,
                learning=False, debug=False)
        except Exception:
            continue
        sent_len = len(sent.split())
        for comp in _walk_composites(parse.global_root_node):
            ci = comp.get_content_instance()
            if not ci: continue
            cls = classify_chunk(comp, sent_len) or "OTHER"
            node = root
            counts[str(node.concept_hash())][cls] += 1
            while node.children:
                node = max(node.children,
                            key=lambda c: c.log_prob_instance(ci))
                counts[str(node.concept_hash())][cls] += 1
    return counts


def bfs_pure_frontier(root, node_class_counts,
                     purity_threshold=0.85, max_depth=4,
                     min_count=5):
    """BFS down from root, emitting a 'frontier' node and stopping
    descent whenever a node is sufficiently CLASS-PURE according to
    the chunks that descend to it from auto-parses. Hard caps at
    ``max_depth`` so deeply-fragmented branches don't run away.

    Stopping criteria (any one):
      - depth >= max_depth      → emit (force cap)
      - node has no children    → emit (real leaf)
      - dominant class share at this node >= purity_threshold
                                → emit (cluster is class-pure)
      - count below ``min_count`` (too few observations to trust)
                                → emit (climb-up safety)

    Otherwise descend into all children. This naturally:
      * Merges sibling sub-trees when the cobweb decided to split a
        cluster the chunk-class distribution doesn't care about
        (stop high, treat the merge as 'free').
      * Keeps depth-3 (or even depth-4) splits where the chunk class
        actually differs between branches.

    The class labels used here are gold/diagnostic (same head-based
    classification we already use for sanity-tagging). Treat the
    resulting frontier as an UPPER BOUND on how compact an
    unsupervised distillation could get."""
    frontier_set = {}
    def visit(node):
        h = str(node.concept_hash())
        d = node.depth()
        c = node_class_counts.get(h, Counter())
        total = sum(c.values())
        # Force-emit if we've hit a leaf or the depth cap.
        if d >= max_depth or not node.children:
            frontier_set[h] = node; return
        # Sparse coverage — climb out: this branch isn't well-supported.
        if total < min_count:
            frontier_set[h] = node; return
        # Pure cluster — no benefit to splitting further.
        dom_share = c.most_common(1)[0][1] / total
        if dom_share >= purity_threshold:
            frontier_set[h] = node; return
        # Mixed cluster, well below depth cap → descend.
        for child in node.children:
            visit(child)
    visit(root)
    return list(frontier_set.values())

def _categorize_to_frontier(bag, root, frontier_hashes):
    """DFS from root, stop at first node whose hash is in
    frontier_hashes. Returns (node, found_bool)."""
    node = root
    if str(node.concept_hash()) in frontier_hashes:
        return node, True
    while node.children:
        node = max(node.children, key=lambda c: c.log_prob_instance(bag))
        if str(node.concept_hash()) in frontier_hashes:
            return node, True
    return node, False

def best_frontier_label(bag, frontier_nodes):
    if not frontier_nodes: return None, -float("inf")
    best_node, best_lp = None, -float("inf")
    for f in frontier_nodes:
        lp = f.log_prob_instance(bag)
        if lp > best_lp:
            best_lp, best_node = lp, f
    return best_node, best_lp


# =============================================================================
# Parsing + step-pick using fixed_d3 + categorize mode
# =============================================================================
def score_candidate(content_bag, context_bag,
                    frontier_content, frontier_context,
                    f_cnt_hashes, f_ctx_hashes, cnt_root, ctx_root):
    """categorize-mode scoring only: DFS to first frontier node in
    each tree, add their log_prob_instance scores."""
    cnt_node, _ = _categorize_to_frontier(content_bag, cnt_root, f_cnt_hashes)
    ctx_node, _ = _categorize_to_frontier(context_bag, ctx_root, f_ctx_hashes)
    return (cnt_node.log_prob_instance(content_bag)
            + ctx_node.log_prob_instance(context_bag),
            cnt_node, ctx_node)

def parse_with_frontier(sentence, f_cnt, f_ctx, cnt_root, ctx_root,
                         record_labels=False):
    f_cnt_hashes = {str(n.concept_hash()) for n in f_cnt}
    f_ctx_hashes = {str(n.concept_hash()) for n in f_ctx}
    tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold="converge")

    while True:
        pairs = tree.get_parentless_pairs()
        if len(pairs) <= 0: break
        best_score = -float("inf"); best_pair = None
        best_cnt_node = None; best_lp = -float("inf")
        for p in pairs:
            try:
                res = tree.evaluate_pair(
                    p["left_word_index"], p["right_word_index"])
            except Exception:
                continue
            sc, cnt_node, _ = score_candidate(
                res["content_inst"], res["context_inst"],
                f_cnt, f_ctx, f_cnt_hashes, f_ctx_hashes,
                cnt_root, ctx_root)
            if sc > best_score:
                best_score = sc; best_pair = p
                if record_labels:
                    best_cnt_node = cnt_node
                    best_lp = cnt_node.log_prob_instance(res["content_inst"])
        if best_pair is None or best_score <= -float("inf"): break
        try:
            result = tree.apply_candidate(
                best_pair["left_word_index"],
                best_pair["right_word_index"])
        except Exception:
            break
        if record_labels and best_cnt_node is not None:
            added_title = result.get("added_node", {}).get("title")
            for n in tree.nodes:
                if getattr(n, "title", None) == added_title:
                    n._frontier_label = best_cnt_node
                    n._frontier_logp  = best_lp
                    break
        if len(tree.global_root_node.children) <= 1: break
    return tree

def step_pick(test_hollow, f_cnt, f_ctx, cnt_root, ctx_root):
    f_cnt_hashes = {str(n.concept_hash()) for n in f_cnt}
    f_ctx_hashes = {str(n.concept_hash()) for n in f_ctx}
    n_correct = n_total = 0
    for hollow in test_hollow:
        sentence = hollow["sentence"]
        gold_tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
        gold_tree.build_primitives(sentence, threshold="converge")
        for m in hollow["merges"]:
            try: gold_tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        gold = bracket_set(gold_tree)
        if not gold: continue
        step_tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
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
                sc, _, _ = score_candidate(
                    res["content_inst"], res["context_inst"],
                    f_cnt, f_ctx, f_cnt_hashes, f_ctx_hashes,
                    cnt_root, ctx_root)
                left_node  = step_tree._find_root_child_by_index(p["left_word_index"])
                right_node = step_tree._find_root_child_by_index(p["right_word_index"])
                if left_node is None or right_node is None: continue
                ls, _  = _chunk_span(left_node)
                _, re_ = _chunk_span(right_node)
                if sc > best_score:
                    best_score, best_span = sc, (int(ls), int(re_))
            n_total += 1
            if best_span is not None and best_span in gold:
                n_correct += 1
            try: step_tree.apply_candidate(m["left"], m["right"])
            except Exception: break
    return n_correct, n_total


# =============================================================================
# Generation
# =============================================================================
def _grammar_recognize_top(tokens, want_constituent=False):
    n = len(tokens)
    if n == 0: return set()
    term_lhs = defaultdict(set)
    for lhs, prods in TEST_GRAMMAR1.items():
        for prod in prods:
            if len(prod) == 1 and (prod[0] in WORD_TO_POS
                                    or prod[0] not in TEST_GRAMMAR1):
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

def _representative_leaf(node):
    while node.children:
        node = max(node.children, key=lambda c: c.count)
    return node

def generate_from_frontier(frontier_content, return_leaf=False):
    median_count = (int(np.median([n.count for n in frontier_content]))
                    if frontier_content else 0)
    candidates = [n for n in frontier_content
                  if n.depth() >= 2 or n.count >= median_count] \
                 or list(frontier_content)
    if not candidates:
        return ("<no candidates>", None, None, None) if return_leaf \
               else "<no candidates>"
    weights = [max(1, n.count) * (1 + n.depth()) for n in candidates]
    chosen = random.choices(candidates,
                            weights=[max(w, 1e-12) for w in weights],
                            k=1)[0]
    leaf = _representative_leaf(chosen)
    try:
        text, parse = trellis.generate_sentence(
            start_content_leaf=leaf, debug=False)
    except Exception as e:
        text, parse = f"<gen failed: {e}>", None
    if return_leaf:
        return text, parse, chosen, leaf
    return text


# =============================================================================
# Tree-with-bars helpers (recomputed per-variant because redistribute
# changes the tree structure)
# =============================================================================
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

def _build_primitive_instances(sentences, n_max=400):
    out = []
    for sent in sentences:
        toks = sent.split()
        for i, w in enumerate(toks):
            if w in WORD_TO_POS:
                out.append((toks, i, w, WORD_TO_POS[w]))
        if len(out) >= n_max: break
    return out

def _build_ctx_inst_for_word(toks, i):
    from parse_mh import _context_weight, _get_or_register_cplx_vid
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

def _collect_ctx_descents(prim_instances, sentences):
    """Build the full list of (context_instance, label_idx) descents we
    want to render in the context-tree bars:

      * Primitives:  context_instance built from sliding-window context,
                     labelled by gold POS (Det, N, Adj, V, P).
      * Composites:  context_instance carried on the composite (set by
                     apply_candidate during TRELLIS's auto-parse),
                     labelled by gold head-based chunk class (NP, AdjP,
                     PP, VP, S). The content-ref attribute is stripped
                     so we descend the pure-context signal (matching
                     how add_parse_tree fit the composite during
                     training)."""
    cref_attr = trellis.ltm.content_ref_attr
    pairs = []
    # Primitives — use POS as the label.
    for sent_toks, i, w, pos in prim_instances:
        inst = _build_ctx_inst_for_word(sent_toks, i)
        pairs.append((inst, _label2id[pos]))
    # Composites — descend through every chunk that TRELLIS's parser
    # produces on the same distillation corpus. Labels come from
    # head-based chunk classification.
    for sent in sentences:
        try:
            parse = trellis.parse_sentence(
                sent, threshold=THRESHOLD, new_vocab=False,
                learning=False, debug=False)
        except Exception:
            continue
        sent_len = len(sent.split())
        for comp in _walk_composites(parse.global_root_node):
            ctx_inst = comp.get_context_instance()
            if not ctx_inst: continue
            ctx_inst = dict(ctx_inst)
            ctx_inst.pop(cref_attr, None)   # strip content-ref
            cls = classify_chunk(comp, sent_len) or "OTHER"
            pairs.append((ctx_inst,
                          _label2id.get(cls, _label2id["OTHER"])))
    return pairs

def compute_ctx_node_counts(root, ctx_descents, max_depth):
    """Descend (instance, label_idx) pairs through the context tree
    and tally label-idx at every visited node. ``ctx_descents`` is the
    output of ``_collect_ctx_descents`` — a mix of primitive and
    composite descents, both contributing to the same per-node
    distribution. The resulting counts span ALL_LABELS (POS classes +
    chunk classes), and the plot uses the same colors as the content
    tree so the two views are directly comparable."""
    all_nodes, children_of, _ = _make_layout(root, max_depth)
    counts = {}
    for inst, label_idx in ctx_descents:
        cur = 0
        for _ in range(max_depth + 1):
            counts.setdefault(cur, np.zeros(N_LABEL, dtype=np.int64))
            counts[cur][label_idx] += 1
            ch = children_of.get(cur, [])
            if not ch: break
            cur = max(ch, key=lambda i: all_nodes[i].log_prob_instance(inst))
    return all_nodes, children_of, counts

def compute_cnt_node_counts(root, sentences, max_depth):
    all_nodes, children_of, _ = _make_layout(root, max_depth)
    cnt_L = {}; cnt_R = {}
    for sent in sentences:
        try:
            parse = trellis.parse_sentence(
                sent, threshold=THRESHOLD, new_vocab=False,
                learning=False, debug=False)
        except Exception:
            continue
        sent_len = len(sent.split())
        for comp in _walk_composites(parse.global_root_node):
            ci = comp.get_content_instance()
            if not ci: continue
            kids = sorted(comp.children, key=lambda x: x[0])
            if len(kids) != 2: continue
            L_cls = classify_chunk(kids[0][1], sent_len) or "OTHER"
            R_cls = classify_chunk(kids[1][1], sent_len) or "OTHER"
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
    return all_nodes, children_of, cnt_L, cnt_R

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
              title="class (top=L, bottom=R; red border=frontier)",
              loc="lower right",
              ncol=max(1, len(legend_h) // 4), fontsize=7, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()


# =============================================================================
# Grammar distillation (unsupervised NT naming + rules + greedy
# reconstruction + coverage curve)
# =============================================================================
def label_frontier_unsupervised(frontier_nodes, sentences):
    ordered = sorted(frontier_nodes,
                     key=lambda n: (-int(n.count), str(n.concept_hash())))
    labels = {str(n.concept_hash()): f"NT_{i}"
              for i, n in enumerate(ordered)}
    tokens_per_nt = defaultdict(Counter)
    gold_per_nt   = defaultdict(Counter)
    for sent in sentences:
        try:
            parse = trellis.parse_sentence(
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
    rules = defaultdict(Counter)
    nt_count = Counter()
    def label_node(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is not None and 0 <= wid < len(trellis.ltm.id_to_value):
                return f"'{trellis.ltm.id_to_value[wid]}'"
            return "?"
        ci = n.get_content_instance()
        if not ci: return "?"
        best, _ = best_frontier_label(ci, frontier_nodes)
        if best is None: return "?"
        return frontier_labels.get(str(best.concept_hash()), "?")
    for sent in sentences:
        try:
            parse = trellis.parse_sentence(
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

def greedy_reconstruct(rules):
    flat = []
    for parent, rhs_counter in rules.items():
        for rhs, c in rhs_counter.items():
            flat.append((parent, rhs, c))
    flat.sort(key=lambda t: -t[2])
    total = sum(c for *_, c in flat) or 1
    coverage, ordered, cumulative = [], [], 0
    for triple in flat:
        cumulative += triple[2]
        ordered.append(triple)
        coverage.append(cumulative / total)
    return ordered, coverage

def plot_rules(rules, nt_count, tokens_per_nt, gold_per_nt,
               variant_name, out_path, top_k=4):
    nts = [n for n in sorted(rules.keys(),
                              key=lambda n: -nt_count.get(n, 0))
           if nt_count.get(n, 0) >= 3 or len(rules[n]) >= 1]
    if not nts: return
    fig_h = max(2.5, 0.55 * sum(min(top_k, len(rules[n])) + 1.5 for n in nts))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.99,
            f"Distilled CFG (UNSUPERVISED) — {variant_name}",
            ha="center", va="top", fontsize=15, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.96,
            f"{len(nts)} non-terminals · auto-parses of "
            f"{DISTILL_N_SENTENCES} sentences · "
            f"productions ranked by count (log p = log p(RHS | LHS))",
            ha="center", va="top", fontsize=10, color="#555",
            transform=ax.transAxes)
    y = 0.93; line_h = 0.022
    for nt in nts:
        gold = gold_per_nt.get(nt, Counter()).most_common(1)
        cls = gold[0][0] if gold else "OTHER"
        col = LABEL_COLOR.get(cls, "#666")
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
                f" [{tok_str}]  {gold_str}   (n={nt_count.get(nt, 0)})",
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
        if y < 0.02: break
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()

def plot_node_link(rules, nt_count, variant_name, out_path):
    edges = defaultdict(int)
    is_nt = lambda lab: not (lab.startswith("'") and lab.endswith("'"))
    for parent, child_counter in rules.items():
        for (l, r), c in child_counter.items():
            edges[(parent, l)] += c
            edges[(parent, r)] += c
    nts = sorted(set([p for p, _ in edges.keys()] +
                     [c for _, c in edges.keys() if is_nt(c)]),
                 key=lambda n: -nt_count.get(n, 0))
    if not nts: return
    n_nt = len(nts)
    angles = np.linspace(0, 2 * np.pi, n_nt, endpoint=False)
    coords = {nt: (np.cos(a), np.sin(a)) for nt, a in zip(nts, angles)}
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    ax.set_title(
        f"Distilled grammar — node-link graph ({variant_name})\n"
        f"edges: parent → child non-terminal, thickness = count",
        fontsize=12, pad=10)
    max_w = max(edges.values()) if edges else 1
    for (src, dst), w in edges.items():
        if src not in coords or dst not in coords: continue
        if src == dst: continue
        x0, y0 = coords[src]; x1, y1 = coords[dst]
        thickness = 0.5 + 4 * (w / max_w)
        alpha = 0.4 + 0.5 * (w / max_w)
        ax.annotate("", xy=(x1 * 0.93, y1 * 0.93),
                    xytext=(x0 * 0.93, y0 * 0.93),
                    arrowprops=dict(arrowstyle="->",
                                    color="#444",
                                    lw=thickness, alpha=alpha,
                                    connectionstyle="arc3,rad=0.12"))
    for nt, (x, y) in coords.items():
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.13, y - 0.04), 0.26, 0.08,
            boxstyle="round,pad=0.01",
            facecolor="#cfe7ff", alpha=0.85,
            edgecolor="black", linewidth=1))
        ax.text(x, y, nt, ha="center", va="center",
                fontsize=9.5, fontweight="bold")
        ax.text(x, y - 0.075, f"n={nt_count.get(nt, 0)}",
                ha="center", va="center", fontsize=7, color="#444")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()

def plot_coverage_curve(ordered_rules, coverage, variant_name, out_path):
    if not ordered_rules: return
    ks = list(range(1, len(ordered_rules) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, coverage, color="#1f77b4", linewidth=2, marker="o", markersize=3)
    ax.fill_between(ks, 0, coverage, alpha=0.15, color="#1f77b4")
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
    ax.set_ylim(0, 1.05); ax.set_xlim(0, max(ks) + 1)
    ax.set_title(
        f"Greedy grammar reconstruction — {variant_name}  "
        f"({len(ordered_rules)} unique productions, "
        f"{sum(c for *_, c in ordered_rules)} observations)")
    ax.grid(axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()


# =============================================================================
# Per-NT rule breakdown viz
# =============================================================================
def plot_nt_breakdown(rules, nt_count, tokens_per_nt, gold_per_nt,
                      variant_name, out_path, top_k=5):
    """For every NT (sorted by total count desc), draw a horizontal
    stacked bar showing how its production mass is distributed across
    rules: top-1 contribution, top-2, …, top-K, then a "tail" segment
    for everything past rank K. Reveals which NTs are "compact"
    (one or two dominant productions) vs "spread" (long tail of one-off
    rules — usually noise or undersplit categories)."""
    nts = sorted(rules.keys(), key=lambda n: -nt_count.get(n, 0))
    nts = [n for n in nts if nt_count.get(n, 0) >= 1]
    if not nts: return

    fig, ax = plt.subplots(figsize=(14, max(4, 0.4 * len(nts))))
    y_positions = np.arange(len(nts))
    bar_height = 0.7

    # Sequential blue palette for top-K, gray for tail.
    rank_colors = ["#1a5d8c", "#3a7eaa", "#5fa0c2",
                   "#88c0d5", "#b3ddea"]
    tail_color  = "#dddddd"
    legend_labels = [f"top-{i+1}" for i in range(top_k)] + [f"tail (≥{top_k+1})"]
    legend_handles = [mpatches.Patch(facecolor=c, edgecolor="black",
                                      linewidth=0.4, label=l)
                      for c, l in zip(rank_colors + [tail_color],
                                       legend_labels)]

    max_total = 0
    for yi, nt in enumerate(nts):
        sorted_rules = sorted(rules[nt].items(), key=lambda kv: -kv[1])
        cumulative = 0
        for ki in range(min(top_k, len(sorted_rules))):
            c = sorted_rules[ki][1]
            ax.barh(yi, c, left=cumulative, height=bar_height,
                    color=rank_colors[ki], edgecolor="black", linewidth=0.4)
            cumulative += c
        tail = sum(c for _, c in sorted_rules[top_k:])
        if tail > 0:
            ax.barh(yi, tail, left=cumulative, height=bar_height,
                    color=tail_color, edgecolor="black", linewidth=0.4)
            cumulative += tail
        max_total = max(max_total, cumulative)

        total   = nt_count[nt]
        n_rules = len(rules[nt])
        # Top-1 mass share (concentration indicator).
        top1_share = sorted_rules[0][1] / max(total, 1) if sorted_rules else 0
        topK_share = sum(c for _, c in sorted_rules[:top_k]) / max(total, 1)
        ax.text(cumulative + max_total * 0.005, yi,
                f"  {n_rules:>3} rules · n={total:>3} · "
                f"top1={100*top1_share:>3.0f}% · "
                f"top{top_k}={100*topK_share:>3.0f}%",
                va="center", fontsize=8, fontfamily="monospace")

    # Y-labels: NT id + top tokens + dominant gold class chip.
    y_labels = []
    label_colors = []
    for nt in nts:
        gold = gold_per_nt.get(nt, Counter()).most_common(1)
        cls = gold[0][0] if gold else "?"
        toks = tokens_per_nt.get(nt, Counter()).most_common(2)
        tok_str = "/".join(t for t, _ in toks) if toks else "—"
        y_labels.append(f"{nt}  [{tok_str}]  {{{cls}}}")
        label_colors.append(LABEL_COLOR.get(cls, "#444"))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    for tick, col in zip(ax.get_yticklabels(), label_colors):
        tick.set_color(col)
    ax.invert_yaxis()

    ax.set_xlabel("Observation count (sum across all of this NT's rules)")
    ax.set_title(
        f"Per-NT rule breakdown — {variant_name}\n"
        f"Wide top-1 bar = concentrated grammar; long gray tail = "
        f"fragmented / noisy category")
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8, title="rule rank within NT")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()


# =============================================================================
# Mergeable-NT heuristic
# =============================================================================
# Combines three signals to propose pairs of non-terminals that could
# be merged into one without losing grammatical fidelity:
#
#   (1) Cobweb LCA depth — frontier NTs sit at depth 3. Two NTs whose
#       lowest common ancestor in the cobweb tree is at depth 2 are
#       SIBLINGS (one node above the frontier) — the cobweb's last
#       split between them. Depth-1 LCA = cousins. Depth-0 LCA = root
#       (unrelated). Higher LCA depth ⇒ closer relatives ⇒ better
#       merge candidates.
#
#   (2) Token Jaccard — fraction of representative tokens shared in
#       each NT's top-5 most-frequent surface words. High Jaccard
#       ⇒ same kind of chunk content.
#
#   (3) Production overlap — number of (left, right) productions that
#       appear in BOTH NTs. These are the deduplications we'd save
#       by merging: rules currently double-counted across the pair.
#
# All three signals correlate with "these NTs should be one cluster
# — the cobweb just happened to over-split them at depth 3".
# =============================================================================
def _node_path_to_root(node):
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = getattr(cur, "parent", None)
    return path

def _lca_depth(node_a, node_b):
    """Lowest-common-ancestor depth in the cobweb content tree."""
    path_a = _node_path_to_root(node_a)
    a_set = {str(n.concept_hash()): n for n in path_a}
    cur = node_b
    while cur is not None:
        h = str(cur.concept_hash())
        if h in a_set:
            return cur.depth()
        cur = getattr(cur, "parent", None)
    return 0   # root

def propose_merges(frontier_nodes, rules, nt_count, tokens_per_nt,
                   labels, gold_per_nt, top_k=5):
    """Score every unordered pair of frontier NTs. Returns a list of
    candidate dicts sorted best-first."""
    nt_list = list(frontier_nodes)
    candidates = []
    for i, n1 in enumerate(nt_list):
        for j, n2 in enumerate(nt_list):
            if j <= i: continue
            lab1 = labels[str(n1.concept_hash())]
            lab2 = labels[str(n2.concept_hash())]
            lca_d = _lca_depth(n1, n2)

            # Top-token Jaccard
            t1 = set(t for t, _ in tokens_per_nt.get(lab1, Counter()).most_common(top_k))
            t2 = set(t for t, _ in tokens_per_nt.get(lab2, Counter()).most_common(top_k))
            jaccard = (len(t1 & t2) / len(t1 | t2)) if (t1 or t2) else 0.0

            # Production overlap (number of identical (l, r) rules)
            r1 = rules.get(lab1, Counter())
            r2 = rules.get(lab2, Counter())
            shared = set(r1.keys()) & set(r2.keys())
            n_overlap = len(shared)
            mass_overlap = sum(min(r1[k], r2[k]) for k in shared)
            merged_rules_total = len(set(r1.keys()) | set(r2.keys()))
            savings = (len(r1) + len(r2)) - merged_rules_total

            # Gold-class agreement (sanity tag, not used in scoring)
            g1 = gold_per_nt.get(lab1, Counter()).most_common(1)
            g2 = gold_per_nt.get(lab2, Counter()).most_common(1)
            gold_match = bool(g1 and g2 and g1[0][0] == g2[0][0])

            # Combined score: weight LCA depth most (structural signal),
            # then Jaccard (semantic similarity), then rule savings.
            score = (3 * lca_d
                     + 4 * jaccard
                     + 0.05 * savings)
            candidates.append({
                "a":           lab1,
                "b":           lab2,
                "lca_depth":   lca_d,
                "jaccard":     jaccard,
                "shared_rules":n_overlap,
                "shared_mass": mass_overlap,
                "rule_savings":savings,
                "merged_total":merged_rules_total,
                "n1":          nt_count.get(lab1, 0),
                "n2":          nt_count.get(lab2, 0),
                "gold_match":  gold_match,
                "score":       score,
            })
    candidates.sort(key=lambda c: -c["score"])
    return candidates

def plot_merge_candidates(candidates, variant_name, out_path, top_k=15):
    """Horizontal bar chart of the top-K merge candidates. Each row
    shows the pair, three signals, and the gold-class sanity flag."""
    top = candidates[:top_k]
    if not top: return
    fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * len(top))))
    y = np.arange(len(top))
    # Use rule_savings as the bar length (concrete payoff of merging).
    savings = [c["rule_savings"] for c in top]
    colors = ["#2ca02c" if c["gold_match"] else "#7f7f7f" for c in top]
    ax.barh(y, savings, color=colors, edgecolor="black", linewidth=0.4)
    for i, c in enumerate(top):
        ax.text(savings[i] + 0.2, i,
                f"  LCA depth={c['lca_depth']}  "
                f"Jaccard={c['jaccard']:.2f}  "
                f"shared rules={c['shared_rules']}  "
                f"({'✓' if c['gold_match'] else '✗'} gold-match)",
                va="center", fontsize=8, fontfamily="monospace")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c['a']} + {c['b']}" for c in top], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("# rules collapsed if these two NTs were merged")
    ax.set_title(
        f"Top {len(top)} mergeable NT pairs — {variant_name}\n"
        f"Bar length = rule savings · LCA depth = how close they are in "
        f"the cobweb tree · Jaccard = top-token overlap · "
        f"green = gold-class matches")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()


def _draw_tree_generic(root_node, label_fn, child_fn, color_fn, lp_fn,
                       title, out_path):
    positions = {}; next_x = [0.0]
    def layout(n, depth):
        children = child_fn(n)
        if not children:
            positions[id(n)] = (next_x[0], -depth, n)
            next_x[0] += 1.0
            return
        for c in children:
            layout(c, depth + 1)
        cxs = [positions[id(c)][0] for c in children]
        positions[id(n)] = (sum(cxs) / len(cxs), -depth, n)
    layout(root_node, 0)
    if not positions: return
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    fig_w = max(8, 1.1 * (max(xs) - min(xs) + 2))
    fig_h = max(4, 1.0 * (max(ys) - min(ys) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off"); ax.set_title(title, fontsize=11)
    def draw_edges(n):
        x0, y0, _ = positions[id(n)]
        for c in child_fn(n):
            x1, y1, _ = positions[id(c)]
            ax.plot([x0, x1], [y0 - 0.18, y1 + 0.18],
                    color="#888", linewidth=1.0, zorder=0)
            draw_edges(c)
    draw_edges(root_node)
    for nid, (x, y, n) in positions.items():
        lab = label_fn(n)
        is_leaf = not child_fn(n)
        col = color_fn(n) or ("#fff4d6" if is_leaf else "#cfe7ff")
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.45, y - 0.18), 0.9, 0.36,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor="black", linewidth=0.7))
        ax.text(x, y + 0.04, lab, ha="center", va="center", fontsize=9,
                fontweight="bold" if not is_leaf else "normal")
        if not is_leaf and lp_fn is not None:
            lp_val = lp_fn(n)
            if lp_val is not None and math.isfinite(lp_val):
                ax.text(x, y - 0.09, f"log p = {lp_val:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="#444", fontfamily="monospace")
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()

def _label_color_from_gold(nt_label, gold_per_nt):
    gold = gold_per_nt.get(nt_label, Counter()).most_common(1)
    cls = gold[0][0] if gold else "OTHER"
    return LABEL_COLOR.get(cls, "#cfe7ff")

def plot_parse_derivation(parse_tree, frontier_labels, frontier_nodes,
                           tokens_per_nt, gold_per_nt, variant_name,
                           sentence, out_path):
    root = parse_tree.global_root_node
    top_children = [c[1] for c in root.children]
    if not top_children: return
    fake_root = top_children[0] if len(top_children) == 1 else root
    def child_fn(n):
        if isinstance(n, PrimitiveParseNode): return []
        return [c[1] for c in n.children]
    def _nt_for(n):
        if isinstance(n, PrimitiveParseNode): return None
        nt = getattr(n, "_frontier_label", None)
        if nt is None:
            ci = n.get_content_instance()
            if ci: nt, _ = best_frontier_label(ci, frontier_nodes)
        return nt
    def label_fn(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is not None and 0 <= wid < len(trellis.ltm.id_to_value):
                return f'"{trellis.ltm.id_to_value[wid]}"'
            return "?"
        if getattr(n, "is_global_root", False): return "ROOT"
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
        if isinstance(n, PrimitiveParseNode): return "#fff4d6"
        nt = _nt_for(n)
        if nt is None: return "#cfe7ff"
        lab = frontier_labels.get(str(nt.concept_hash()), "?")
        return _label_color_from_gold(lab, gold_per_nt)
    _draw_tree_generic(
        fake_root, label_fn, child_fn, color_fn, lp_fn,
        title=(f"Parse derivation ({variant_name})\n"
               f"\"{sentence}\""),
        out_path=out_path)

def plot_generation_derivation(parse_tree, frontier_labels, frontier_nodes,
                                 tokens_per_nt, gold_per_nt, variant_name,
                                 seed_label, seed_node, gen_text, out_path):
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
            if wid is not None and 0 <= wid < len(trellis.ltm.id_to_value):
                return f'"{trellis.ltm.id_to_value[wid]}"'
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
        if isinstance(n, PrimitiveParseNode): return "#fff4d6"
        nt = _nt_for(n)
        if nt is None: return "#cfe7ff"
        lab = frontier_labels.get(str(nt.concept_hash()), "?")
        return _label_color_from_gold(lab, gold_per_nt)
    seed_info = ""
    if seed_node is not None:
        seed_info = (f"  (seed count = {int(seed_node.count)}, "
                     f"depth = {seed_node.depth()})")
    _draw_tree_generic(
        fake_root, label_fn, child_fn, color_fn, lp_fn,
        title=(f"Generation unpacking ({variant_name})\n"
               f"seed = {seed_label}{seed_info}  →  \"{gen_text}\""),
        out_path=out_path)


# =============================================================================
# Run one variant end-to-end (parse eval + distillation + viz)
# =============================================================================
def run_variant(variant_name, out_dir, cnt_root, ctx_root,
                f_cnt=None, f_ctx=None, frontier_kind="fixed_d3"):
    """Run a full variant: parse-eval, distillation, viz. If
    ``f_cnt``/``f_ctx`` are not supplied, defaults to fixed_d3."""
    log(f"\n=== Running variant: {frontier_kind} ({variant_name}) ===")
    if f_cnt is None:
        f_cnt = fixed_depth_frontier(cnt_root, 3)
    if f_ctx is None:
        f_ctx = fixed_depth_frontier(ctx_root, 3)
    log(f"  Frontier sizes — content={len(f_cnt)}  context={len(f_ctx)}")

    # ── Parse evaluation: F1, P, R, exact-match, step-pick ──
    log(f"  Evaluating parser on {len(test_hollow)} held-out sentences…")
    tp = fp = fn = exact = total = 0
    for hollow in test_hollow:
        sentence = hollow["sentence"]
        gold_tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
        gold_tree.build_primitives(sentence, threshold="converge")
        for m in hollow["merges"]:
            try: gold_tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        gold = bracket_set(gold_tree)
        pred_tree = parse_with_frontier(
            sentence, f_cnt, f_ctx, cnt_root, ctx_root)
        pred = bracket_set(pred_tree)
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
        total += 1
        if gold == pred and len(gold) > 0: exact += 1
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    exact_pct = exact / max(total, 1)

    sp_hit, sp_tot = step_pick(test_hollow, f_cnt, f_ctx, cnt_root, ctx_root)
    sp_acc = sp_hit / max(sp_tot, 1)

    log(f"  Bracket P={100*prec:.1f}%  R={100*rec:.1f}%  F1={100*f1:.1f}%  "
          f"Exact-match={100*exact_pct:.1f}%  Step-pick={100*sp_acc:.1f}%")

    # ── Grammar distillation ──
    labels, tokens_per_nt, gold_per_nt = label_frontier_unsupervised(
        f_cnt, distill_sentences)
    rules, nt_count = extract_rules_unsupervised(
        f_cnt, labels, distill_sentences)
    ordered, coverage = greedy_reconstruct(rules)
    log(f"  Distilled grammar: {sum(len(v) for v in rules.values())} rules "
          f"across {len(rules)} NTs ({len(ordered)} observations)")
    if coverage:
        for milestone in [0.5, 0.8, 0.95]:
            for k, cov in enumerate(coverage, 1):
                if cov >= milestone:
                    log(f"    Greedy coverage: {int(milestone*100)}% "
                          f"reached at rule #{k}")
                    break

    plot_rules(rules, nt_count, tokens_per_nt, gold_per_nt, variant_name,
               os.path.join(out_dir, "rules.png"))
    plot_node_link(rules, nt_count, variant_name,
                   os.path.join(out_dir, "node_link.png"))
    plot_coverage_curve(ordered, coverage, variant_name,
                        os.path.join(out_dir, "coverage.png"))

    # ── Per-NT rule breakdown (how concentrated is each NT's grammar?) ──
    plot_nt_breakdown(rules, nt_count, tokens_per_nt, gold_per_nt,
                      variant_name,
                      os.path.join(out_dir, "nt_breakdown.png"))

    # ── Mergeable-NT heuristic (cobweb sibling + token overlap + rule overlap) ──
    merges = propose_merges(f_cnt, rules, nt_count, tokens_per_nt,
                            labels, gold_per_nt)
    plot_merge_candidates(merges, variant_name,
                          os.path.join(out_dir, "merge_candidates.png"))
    with open(os.path.join(out_dir, "merge_candidates.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "a", "b", "lca_depth", "jaccard", "shared_rules",
            "shared_mass", "rule_savings", "merged_total",
            "n1", "n2", "gold_match", "score"])
        w.writeheader()
        for m in merges:
            row = dict(m)
            row["jaccard"] = f"{m['jaccard']:.4f}"
            row["score"]   = f"{m['score']:.3f}"
            w.writerow(row)
    if merges:
        log(f"  Mergeable NT pairs found: {len(merges)}; "
            f"top candidate: {merges[0]['a']} + {merges[0]['b']}  "
            f"(LCA depth={merges[0]['lca_depth']}, "
            f"Jaccard={merges[0]['jaccard']:.2f}, "
            f"would collapse {merges[0]['rule_savings']} rules)")

    # ── Cobweb tree-with-bars (frontier highlighted) ──
    primtest = _build_primitive_instances(distill_sentences, n_max=400)
    ctx_descents = _collect_ctx_descents(primtest, distill_sentences)
    ctx_nodes_layout, ctx_children, ctx_counts = compute_ctx_node_counts(
        ctx_root, ctx_descents, max_depth=TREE_DEPTH_FIG)
    cnt_nodes_layout, cnt_children, cnt_L, cnt_R = compute_cnt_node_counts(
        cnt_root, distill_sentences, max_depth=TREE_DEPTH_FIG)
    ctx_hashes = {str(n.concept_hash()) for n in f_ctx}
    cnt_hashes = {str(n.concept_hash()) for n in f_cnt}
    ctx_hi = {i for i, n in enumerate(ctx_nodes_layout)
              if str(n.concept_hash()) in ctx_hashes}
    cnt_hi = {i for i, n in enumerate(cnt_nodes_layout)
              if str(n.concept_hash()) in cnt_hashes}
    plot_tree_single_bars(
        ctx_children, ctx_counts, ALL_LABELS, LABEL_COLOR, ctx_hi,
        title=(f"CONTEXT tree — POS + chunk-class distributions  "
               f"(primitives + composites descended through context tree; "
               f"red border = {variant_name}, "
               f"|frontier|={len(f_ctx)})"),
        out_path=os.path.join(out_dir, "tree_context.png"),
        max_depth=TREE_DEPTH_FIG)
    plot_tree_pair_bars(
        cnt_children, cnt_L, cnt_R, ALL_LABELS, LABEL_COLOR, cnt_hi,
        title=(f"CONTENT tree — L/R child class distributions  "
               f"(red border = {variant_name}, "
               f"|frontier|={len(f_cnt)})"),
        out_path=os.path.join(out_dir, "tree_content.png"),
        max_depth=TREE_DEPTH_FIG)

    # ── Per-sentence parse derivations + generation derivations ──
    sample_sents = [h["sentence"] for h in test_hollow[:N_DERIV_VIZ]]
    for i, sent in enumerate(sample_sents):
        pt = parse_with_frontier(sent, f_cnt, f_ctx, cnt_root, ctx_root,
                                  record_labels=True)
        plot_parse_derivation(
            pt, labels, f_cnt, tokens_per_nt, gold_per_nt, variant_name, sent,
            os.path.join(out_dir, f"derivations/parse_{i}.png"))
    n_lex = n_const = n_gram = 0
    for i in range(N_GEN):
        try:
            text, parse, seed_node, _ = generate_from_frontier(
                f_cnt, return_leaf=True)
        except Exception:
            continue
        toks = text.split() if text else []
        if toks and all(t in WORD_TO_POS for t in toks):
            n_lex += 1
            top = _grammar_recognize_top(toks)
            if top: n_const += 1
            if "S" in top: n_gram += 1
        if i < N_DERIV_VIZ and parse is not None:
            seed_label = labels.get(str(seed_node.concept_hash()), "?")
            plot_generation_derivation(
                parse, labels, f_cnt, tokens_per_nt, gold_per_nt, variant_name,
                seed_label, seed_node, text,
                os.path.join(out_dir, f"derivations/gen_{i}.png"))
    log(f"  Generation: in-lex {100*n_lex/N_GEN:.0f}%  "
          f"constituent {100*n_const/N_GEN:.0f}%  "
          f"full sentence {100*n_gram/N_GEN:.0f}%")

    # ── Save the rules CSV ──
    with open(os.path.join(out_dir, "rules.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["rank", "lhs", "left", "right", "count", "log_p",
                    "cum_coverage"])
        for i, ((parent, (l, r), c), cov) in enumerate(zip(ordered, coverage)):
            total_p = sum(rules[parent].values()) or 1
            w.writerow([i + 1, parent, l, r, c,
                        f"{math.log(c/total_p):.4f}" if c else "",
                        f"{cov:.4f}"])

    return {
        "variant":         variant_name,
        "frontier_content_size": len(f_cnt),
        "frontier_context_size": len(f_ctx),
        "precision":       prec,
        "recall":          rec,
        "f1":              f1,
        "exact_match":     exact_pct,
        "step_pick":       sp_acc,
        "n_rules":         sum(len(v) for v in rules.values()),
        "n_nts":           len(rules),
        "coverage_50":     next((k for k, cov in enumerate(coverage, 1)
                                  if cov >= 0.5), None),
        "coverage_80":     next((k for k, cov in enumerate(coverage, 1)
                                  if cov >= 0.8), None),
        "coverage_95":     next((k for k, cov in enumerate(coverage, 1)
                                  if cov >= 0.95), None),
        "gen_in_lex":      n_lex / N_GEN,
        "gen_constituent": n_const / N_GEN,
        "gen_sentence":    n_gram / N_GEN,
    }


# =============================================================================
# PHASE 1 — Vanilla d3
# =============================================================================
log("\n" + "=" * 70)
log("PHASE 1 — VANILLA fixed_d3")
log("=" * 70)
log(f"  Content tree: {sum(1 for _ in _walk(trellis.ltm.content_hierarchy.root))} nodes")
log(f"  Context tree: {sum(1 for _ in _walk(trellis.ltm.context_hierarchy.root))} nodes")

vanilla_results = run_variant(
    "fixed_d3 vanilla", VANILLA_DIR,
    trellis.ltm.content_hierarchy.root,
    trellis.ltm.context_hierarchy.root,
    frontier_kind="fixed_d3")


# =============================================================================
# PHASE 2 — Redistribute both cobweb trees
# =============================================================================
log("\n" + "=" * 70)
log(f"PHASE 2 — REDISTRIBUTE (n={REDIST_N} per tree)")
log("=" * 70)
log(f"  Calling tree.redistribute({REDIST_N}) on CONTENT tree…")
trellis.ltm.content_hierarchy.redistribute(REDIST_N)
# Skip redistributing the CONTEXT tree: the TopK-Pool encoder caches
# the context tree's depth-d node list as stable int ids, and moving
# context-tree nodes invalidates those ids in ways the encoder's
# sync_remap can't recover from cleanly (results in segfaults on
# subsequent log_prob_instance calls because stored chunk bags
# reference orphaned context nodes). Redistributing the content tree
# alone is the safe-and-meaningful variant — it re-clusters the chunk
# representations without touching the substrate the encoder relies on.
log(f"  Content tree (post-redist): "
      f"{sum(1 for _ in _walk(trellis.ltm.content_hierarchy.root))} nodes")
log(f"  Context tree (untouched):  "
      f"{sum(1 for _ in _walk(trellis.ltm.context_hierarchy.root))} nodes")


# =============================================================================
# PHASE 3 — Redistributed d3
# =============================================================================
log("\n" + "=" * 70)
log("PHASE 3 — REDISTRIBUTED fixed_d3")
log("=" * 70)
redist_results = run_variant(
    "fixed_d3 redist", REDIST_DIR,
    trellis.ltm.content_hierarchy.root,
    trellis.ltm.context_hierarchy.root,
    frontier_kind="fixed_d3")


# =============================================================================
# PHASE 3.5 — BFS-pure frontier (variable depth, post-redist)
# =============================================================================
# Selective sibling-merging: instead of a flat depth-3 cut, BFS down
# from root and stop early at any node whose chunk-class distribution
# is already concentrated above PURITY_THRESHOLD. The chunks that the
# cobweb tree wants to over-split into separate branches collapse
# back into a single high-up NT; branches where the chunk class
# actually differs keep their finer-grained splits. Hard-capped at
# depth ``PURITY_MAX_DEPTH``.
# =============================================================================
log("\n" + "=" * 70)
log("PHASE 3.5 — BFS-PURE FRONTIER  "
    f"(purity≥{PURITY_THRESHOLD}, depth≤{PURITY_MAX_DEPTH})")
log("=" * 70)
log("  Tallying chunk-class distributions at every node visited "
    "during auto-parses…")
_cnt_class_counts = collect_node_chunk_class_counts(
    trellis.ltm.content_hierarchy.root, distill_sentences)
_ctx_class_counts = collect_node_chunk_class_counts(
    trellis.ltm.context_hierarchy.root, distill_sentences)
bfs_cnt = bfs_pure_frontier(trellis.ltm.content_hierarchy.root,
                             _cnt_class_counts,
                             purity_threshold=PURITY_THRESHOLD,
                             max_depth=PURITY_MAX_DEPTH,
                             min_count=PURITY_MIN_COUNT)
bfs_ctx = bfs_pure_frontier(trellis.ltm.context_hierarchy.root,
                             _ctx_class_counts,
                             purity_threshold=PURITY_THRESHOLD,
                             max_depth=PURITY_MAX_DEPTH,
                             min_count=PURITY_MIN_COUNT)
log(f"  BFS-pure content frontier: {len(bfs_cnt)} NTs  "
    f"(depths: {sorted({n.depth() for n in bfs_cnt})})")
log(f"  BFS-pure context frontier: {len(bfs_ctx)} NTs  "
    f"(depths: {sorted({n.depth() for n in bfs_ctx})})")

bfs_pure_results = run_variant(
    "bfs_pure redist", BFS_PURE_DIR,
    trellis.ltm.content_hierarchy.root,
    trellis.ltm.context_hierarchy.root,
    f_cnt=bfs_cnt, f_ctx=bfs_ctx,
    frontier_kind=f"bfs_pure (≥{PURITY_THRESHOLD}, ≤{PURITY_MAX_DEPTH})")


# =============================================================================
# PHASE 4 — Side-by-side comparison
# =============================================================================
log("\n" + "=" * 70)
log("PHASE 4 — VANILLA vs REDISTRIBUTED vs BFS-PURE COMPARISON")
log("=" * 70)

results_table = [vanilla_results, redist_results, bfs_pure_results]

# Persist results CSV.
with open(os.path.join(OUT_DIR, "parse_eval.csv"), "w") as f:
    w = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
    w.writeheader()
    for r in results_table: w.writerow(r)

# Print summary table.
log(f"\n  {'metric':<24} {'vanilla':>12} {'redist':>12} {'Δ':>10}")
def _fmt(v, pct=True): return (f"{100*v:.1f}%" if pct else f"{v}") if v is not None else "—"
log(f"\n  {'metric':<24} {'vanilla':>11} {'redist':>11} {'bfs_pure':>11}")
for metric, label, pct in [
    ("frontier_content_size", "Frontier (content)",  False),
    ("frontier_context_size", "Frontier (context)",  False),
    ("precision",             "Bracket Precision",   True),
    ("recall",                "Bracket Recall",      True),
    ("f1",                    "Bracket F1",          True),
    ("exact_match",           "Exact-match parses",  True),
    ("step_pick",             "Step-pick accuracy",  True),
    ("n_nts",                 "# non-terminals",     False),
    ("n_rules",               "# productions",       False),
    ("coverage_50",           "Rules for 50% cov.",  False),
    ("coverage_80",           "Rules for 80% cov.",  False),
    ("coverage_95",           "Rules for 95% cov.",  False),
    ("gen_in_lex",            "Gen in-lexicon",      True),
    ("gen_constituent",       "Gen constituent",     True),
    ("gen_sentence",          "Gen full sentence",   True),
]:
    v = vanilla_results.get(metric)
    r = redist_results.get(metric)
    b = bfs_pure_results.get(metric)
    log(f"  {label:<24} {_fmt(v, pct):>11} {_fmt(r, pct):>11} "
        f"{_fmt(b, pct):>11}")

# Comparison plot (3-way).
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
VARIANT_COLOR = {
    "vanilla":     "#7f7f7f",
    "redist":      "#d62728",
    "bfs_pure":    "#2ca02c",
}
all_results = {"vanilla":  vanilla_results,
               "redist":   redist_results,
               "bfs_pure": bfs_pure_results}
W = 0.27

def _grouped_bar(ax, keys, labels, ylabel, title, pct=False, label_fmt=None):
    label_fmt = label_fmt or (lambda v: f"{100*v:.0f}" if pct else str(v))
    x = np.arange(len(keys))
    for i, name in enumerate(["vanilla", "redist", "bfs_pure"]):
        vals = [all_results[name].get(k) or 0 for k in keys]
        ax.bar(x + (i - 1) * W, vals, W, label=name,
               color=VARIANT_COLOR[name], edgecolor="black", linewidth=0.4)
        for xi, v in zip(x, vals):
            offset = 0.02 if pct else max(vals) * 0.02
            ax.text(xi + (i - 1) * W, v + offset, label_fmt(v),
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    if pct: ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

_grouped_bar(axes[0],
             keys=["precision", "recall", "f1", "exact_match", "step_pick"],
             labels=["Precision", "Recall", "F1", "Exact", "Step-pick"],
             ylabel="Rate",
             title="(A) Parse metrics",
             pct=True)

_grouped_bar(axes[1],
             keys=["frontier_content_size", "n_nts", "n_rules"],
             labels=["|Frontier|", "# NTs in grammar", "# productions"],
             ylabel="Count",
             title="(B) Grammar size (lower = more compact)")

_grouped_bar(axes[2],
             keys=["coverage_50", "coverage_80", "coverage_95"],
             labels=["50% cov.", "80% cov.", "95% cov."],
             ylabel="# rules needed for milestone",
             title="(C) Greedy coverage (lower = more compact)")

plt.suptitle(
    f"Grammar distillation — fixed_d3 vs fixed_d3 + redistribute "
    f"vs BFS-pure frontier  "
    f"(redist N={REDIST_N}; purity≥{PURITY_THRESHOLD}, depth≤{PURITY_MAX_DEPTH})",
    fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, "parse_comparison.png"),
            dpi=140, bbox_inches="tight")
plt.close()
log(f"\n  Comparison plot → parse_comparison.png")
log(f"\nArtefacts in {OUT_DIR}/:")
log("  parse_comparison.png · parse_eval.csv")
log("  vanilla/  — fixed_d3 on the as-trained tree")
log("  redist/   — fixed_d3 on the redistributed tree")
log("  bfs_pure/ — BFS-pure frontier on the redistributed tree")
log("  each variant subdir contains: rules.png, node_link.png,")
log("    coverage.png, nt_breakdown.png, merge_candidates.png,")
log("    tree_{context,content}.png, derivations/*.png,")
log("    rules.csv, merge_candidates.csv")
