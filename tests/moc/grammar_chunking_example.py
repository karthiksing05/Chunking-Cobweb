"""
Grammar Chunking — content/context two-tree experiment

Inherits the corpus and context-encoding setup from grammar_example.py.
For each adjacent bigram (w_i, w_{i+1}) in TEST_GRAMMAR3 sentences we
encode w_i (Left) and w_{i+1} (Right) via a shared context tree
(CobwebDiscreteTree).  That left/right encoding is then fed into a
per-variant content tree, mirroring TRELLIS's content-vs-context
hierarchy idea (see src/parse_mh.py around line 2700).

Variants (each builds its own content tree):

  Baseline (path-based, single truncated leaf path):
    PathBag-Cnt1        — top-1 leaf path → ancestor bag (count=1)
                          two attrs: 0=Left, 1=Right         → CobwebDiscreteTree
    PathBag-DepthAttrs  — top-1 leaf path → per-(side,depth) attrs (count=1)
                          attrs 0..PATH_DEPTH=Left, PATH_DEPTH+1..= Right
                          → CobwebDiscreteTree
    PathBag-LogProb     — top-1 leaf path → ancestor bag with
                          counts = exp(log_prob - max_log_prob_in_path)
                          (positive, monotone with log-prob, best=1.0)
                          → CobwebDiscreteTree

  Cobweb-BFS dense:
    BFS-Cont            — Cobweb-BFS-128 dense vec for L and R
                          → 256D dense → CobwebContinuousTree

  Cobweb Sparse (TopK):
    TopK-Cont           — Cobweb-TopK-10 (per-side sparsified scaled vec)
                          → 256D sparse-dense → CobwebContinuousTree
    TopK-Disc-Cnt1      — TopK selected node ids → bag (count=1)
                          two attrs (Left / Right)             → CobwebDiscreteTree
    TopK-Disc-LogProb   — TopK selected node ids → bag
                          counts = exp(log_prob - max)         → CobwebDiscreteTree

  Incremental BFS-bag (priority-queue BFS via bfs_top_k_leaves):
    TopK-Pool-Cache     — incremental analog of TopK-Disc-Cnt1. Every
                          context-tree node at depth d is scored
                          directly via log_prob_instance and the top-K
                          are emitted as a set with count=1 per id.
                          Ids are stable concept_hash-derived ints; if
                          a pool node leaves depth d (merge above /
                          split-at-depth / split-above) the encoder
                          pushes a {old_id → current_canonical_id}
                          entry through set_value_remap so old
                          av_count entries keep accumulating against
                          the right canonical value. Encoded by
                          TopKPoolEncoder.
    BFSBag-BLRemap      — bfs_top_k_leaves(K, max_nodes) → each leaf is
                          hard-remapped to *its own* basic-level node via
                          leaf.get_basic(use_root=True,
                          eval_alpha=BL_EVAL_ALPHA), cached by
                          BasicLevelCache. Plain CobwebDiscreteTree (no
                          ref_tree). Captures variable-coarseness
                          categorisation when no single REMAP_DEPTH fits.

Evaluation mirrors grammar_example.py: extract BFS-DZ_CONTENT nodes from
each content tree, compute per-node log_prob for each pair, then linear
probe + KNN against the (left_POS, right_POS) flattened class label.
"""

import os
import sys
import csv
import math
import random
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP
from sklearn.manifold import TSNE

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from util.cfg import TEST_GRAMMAR3, generate
from cobweb.cobweb_discrete   import CobwebDiscreteTree
from cobweb.cobweb_continuous import CobwebContinuousTree
from cobweb.leaf_remap        import (
    TopKPoolEncoder, BasicLevelCache, bfsbag_blremap_instance,
)

OUT_DIR  = os.path.join(_HERE, "grammar_chunking_output")
ARR_DIR  = os.path.join(OUT_DIR, "arrays")
TREE_DIR = os.path.join(OUT_DIR, "tree_viz")
for _d in (OUT_DIR, ARR_DIR, TREE_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_SENTENCES = 1000
WINDOW      = 3       # context half-window  (offsets ±1, ±2, ±3, exclude self)
DZ_CONTEXT  = 128     # context tree BFS / depth target
DZ_CONTENT  = 128     # content tree BFS-feature target for downstream eval
TOP_K       = 7      # per-instance TopK sparsification on context tree
TOPK_DEPTH  = 4       # fixed depth in context tree to pull the TopK pool from
                      # (clamped to deepest available if tree is shallower)
PATH_DEPTH  = 6       # truncation depth for path-based baselines
BFS_K         = 7     # bag width for BFSBag-* variants
BFS_MAX_NODES = 128   # BFS expansion budget for BFSBag-* (matches DZ_CONTEXT)
REMAP_DEPTH   = 4     # ancestor depth for the BFSBag-Remap remap step
BL_EVAL_ALPHA = 10.0  # eval_alpha for BFSBag-BLRemap's get_basic(use_root=True)
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Per-tree alphas (verbose for tuning) ──────────────────────────────────────
ALPHA_CONTEXT             = 1e-3
ALPHA_PATHBAG_CNT1        = 1e-3
ALPHA_PATHBAG_DEPTH       = 1e-3
ALPHA_PATHBAG_LOGPROB     = 1e-3
ALPHA_BFS_CONT            = 1e-3
ALPHA_TOPK_CONT           = 1e-3
ALPHA_TOPK_DISC_CNT1      = 1e-3
ALPHA_TOPK_DISC_LOGPROB   = 1e-3
ALPHA_BFSBAG_REMAP        = 1e-3
ALPHA_BFSBAG_BLREMAP      = 1e-3

# ── Generate corpus ───────────────────────────────────────────────────────────
print(f"Generating {N_SENTENCES} sentences from TEST_GRAMMAR3 …")
sentences = []
for _ in range(N_SENTENCES):
    sent = [w for w in generate("S", TEST_GRAMMAR3).split() if w]
    if len(sent) >= 2:                      # need ≥2 tokens to form a pair
        sentences.append(sent)

all_tokens = [w for sent in sentences for w in sent]
vocab      = sorted(set(all_tokens))
word2id    = {w: i for i, w in enumerate(vocab)}
id2word    = {i: w for w, i in word2id.items()}
V          = len(vocab)
print(f"  Sentences: {len(sentences)} | Vocab: {V} | Tokens: {len(all_tokens)}")

# Word → POS via grammar terminal rules
_TERMINAL_POS = ["Det", "N", "Adj", "RelPro", "V", "P"]
word2pos = {}
for _pos_tag in _TERMINAL_POS:
    if _pos_tag in TEST_GRAMMAR3:
        for _prod in TEST_GRAMMAR3[_pos_tag]:
            if len(_prod) == 1 and _prod[0] not in word2pos:
                word2pos[_prod[0]] = _pos_tag
for _w in vocab:
    if _w not in word2pos:
        word2pos[_w] = "Unk"
pos_tags = sorted(set(word2pos.values()))
pos2id   = {p: i for i, p in enumerate(pos_tags)}
id2pos   = {i: p for p, i in pos2id.items()}
N_POS    = len(pos_tags)
print(f"  POS tags ({N_POS}): {pos_tags}")


def pair_label(left_w, right_w):
    """Flatten (left_pos, right_pos) → single int id."""
    return pos2id[word2pos[left_w]] * N_POS + pos2id[word2pos[right_w]]


def split_pair_label(lbl):
    """Inverse of pair_label."""
    return lbl // N_POS, lbl % N_POS


# ── Context-window encoding ───────────────────────────────────────────────────
CONTEXT_OFFSETS = [p for p in range(-WINDOW, WINDOW + 1) if p != 0]
pos2attr        = {p: i for i, p in enumerate(CONTEXT_OFFSETS)}


def make_context_instance(sent, pos):
    inst = {}
    for offset in CONTEXT_OFFSETS:
        ctx = pos + offset
        if 0 <= ctx < len(sent):
            inst[pos2attr[offset]] = {word2id[sent[ctx]]: 1.0}
    return inst


# ── Sentence-level 80/20 split (avoid context leakage between train/test pairs)
rng         = np.random.default_rng(SEED)
sent_perm   = rng.permutation(len(sentences))
split_s     = int(0.8 * len(sentences))
train_sents = [sentences[i] for i in sent_perm[:split_s]]
test_sents  = [sentences[i] for i in sent_perm[split_s:]]
print(f"  Sentence split: train={len(train_sents)} test={len(test_sents)}")

# Context-tree training data: every train-sentence token
context_train_tokens = [make_context_instance(s, p)
                        for s in train_sents for p in range(len(s))]
print(f"  Context tree training tokens: {len(context_train_tokens)}")


def build_pair_set(sents):
    L, R, labels = [], [], []
    for s in sents:
        for i in range(len(s) - 1):
            L.append(make_context_instance(s, i))
            R.append(make_context_instance(s, i + 1))
            labels.append(pair_label(s[i], s[i + 1]))
    return L, R, np.array(labels, dtype=np.int32)


train_L, train_R, y_train = build_pair_set(train_sents)
test_L,  test_R,  y_test  = build_pair_set(test_sents)
print(f"  Pairs:  train={len(train_L)}  test={len(test_L)}")
print(f"  Distinct pair-label classes (train): {len(set(y_train.tolist()))}")

np.save(os.path.join(ARR_DIR, "y_train.npy"), y_train)
np.save(os.path.join(ARR_DIR, "y_test.npy"),  y_test)

# ── Train shared context tree ─────────────────────────────────────────────────
print(f"\nBuilding Context tree (CobwebDiscreteTree, alpha={ALPHA_CONTEXT}) …")
context_tree = CobwebDiscreteTree(alpha=ALPHA_CONTEXT, weight_attr=True)
for i, inst in enumerate(context_train_tokens):
    context_tree.ifit(inst)
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(context_train_tokens)} inserted")
print("  Context tree built.")


# ── Generic node-collection helpers ───────────────────────────────────────────
def collect_by_depth_nodes(root):
    by_depth, queue = {}, [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        by_depth.setdefault(d, []).append(node)
        for c in node.children:
            queue.append((c, d + 1))
    return by_depth


def bfs_first_n_nodes(root, n):
    nodes, queue = [], [root]
    while queue and len(nodes) < n:
        node = queue.pop(0)
        for c in node.children:
            if len(nodes) >= n:
                break
            nodes.append(c)
            queue.append(c)
    return nodes


def collect_path_tree_with_chains(root, max_depth):
    """BFS up to max_depth.  Returns:
         all_nodes, node_to_idx, leaves, leaf_path_chain
       where leaf_path_chain[id(leaf)] = [(depth, node), ...] ordered root→leaf."""
    all_nodes, node_to_idx, leaves, chain = [], {}, [], {}
    queue = [(root, 0, [])]
    while queue:
        node, depth, parent_chain = queue.pop(0)
        node_to_idx[id(node)] = len(all_nodes)
        all_nodes.append(node)
        my_chain = parent_chain + [(depth, node)]
        if depth >= max_depth or not node.children:
            leaves.append(node)
            chain[id(node)] = my_chain
        else:
            for c in node.children:
                queue.append((c, depth + 1, my_chain))
    return all_nodes, node_to_idx, leaves, chain


# ── Extract context-tree node sets ────────────────────────────────────────────
print("\nExtracting context-tree node sets …")
ctx_bfs_nodes = bfs_first_n_nodes(context_tree.root, DZ_CONTEXT)
if not ctx_bfs_nodes:
    ctx_bfs_nodes = [context_tree.root]
print(f"  BFS-{DZ_CONTEXT}: {len(ctx_bfs_nodes)} nodes")

by_depth     = collect_by_depth_nodes(context_tree.root)
depth_counts = {d: len(v) for d, v in by_depth.items()}
print(f"  Nodes per depth: {dict(sorted(depth_counts.items()))}")

topk_depth      = min(TOPK_DEPTH, max(depth_counts))
topk_pool_nodes = by_depth[topk_depth]
if topk_depth != TOPK_DEPTH:
    print(f"  WARN: requested TOPK_DEPTH={TOPK_DEPTH} not available; "
          f"clamped to deepest depth {topk_depth}")
print(f"  TopK pool: depth {topk_depth} ({len(topk_pool_nodes)} nodes)  k={TOP_K}")

path_all_nodes, path_node_to_idx, path_leaves, path_leaf_chain = \
    collect_path_tree_with_chains(context_tree.root, PATH_DEPTH)
print(f"  Path tree: {len(path_all_nodes)} nodes, {len(path_leaves)} leaves "
      f"(depth ≤ {PATH_DEPTH})")


# ── Discrete encoding helper: log P(instance | node) per BFS / pool node ──────
def encode_logpost_disc(instances, nodes):
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, inst in enumerate(instances):
            out[i, j] = node.log_prob_instance(inst)
    return out


# ── BFS dense context vectors (used by BFS-Cont) ──────────────────────────────
print("\nEncoding BFS context vectors …")
_bfs_train_L = encode_logpost_disc(train_L, ctx_bfs_nodes)
_bfs_train_R = encode_logpost_disc(train_R, ctx_bfs_nodes)
_bfs_test_L  = encode_logpost_disc(test_L,  ctx_bfs_nodes)
_bfs_test_R  = encode_logpost_disc(test_R,  ctx_bfs_nodes)

# Fit scaler on left+right train rows pooled (same node columns, same statistics)
_scaler_bfs   = StandardScaler()
_scaler_bfs.fit(np.vstack([_bfs_train_L, _bfs_train_R]))
ctx_bfs_train_L = _scaler_bfs.transform(_bfs_train_L)
ctx_bfs_train_R = _scaler_bfs.transform(_bfs_train_R)
ctx_bfs_test_L  = _scaler_bfs.transform(_bfs_test_L)
ctx_bfs_test_R  = _scaler_bfs.transform(_bfs_test_R)

bfs_cont_train = np.hstack([ctx_bfs_train_L, ctx_bfs_train_R]).astype(np.float32)
bfs_cont_test  = np.hstack([ctx_bfs_test_L,  ctx_bfs_test_R ]).astype(np.float32)
print(f"  BFS-Cont input dim: {bfs_cont_train.shape[1]}  "
      f"(train={bfs_cont_train.shape[0]}, test={bfs_cont_test.shape[0]})")


# ── TopK sparsified context vectors (used by TopK-Cont) ───────────────────────
print("\nEncoding TopK pool context vectors …")
_pool_train_L = encode_logpost_disc(train_L, topk_pool_nodes)
_pool_train_R = encode_logpost_disc(train_R, topk_pool_nodes)
_pool_test_L  = encode_logpost_disc(test_L,  topk_pool_nodes)
_pool_test_R  = encode_logpost_disc(test_R,  topk_pool_nodes)

_scaler_pool = StandardScaler()
_scaler_pool.fit(np.vstack([_pool_train_L, _pool_train_R]))
ctx_pool_train_L = _scaler_pool.transform(_pool_train_L)
ctx_pool_train_R = _scaler_pool.transform(_pool_train_R)
ctx_pool_test_L  = _scaler_pool.transform(_pool_test_L)
ctx_pool_test_R  = _scaler_pool.transform(_pool_test_R)


def topk_sparsify(Z, k):
    out = np.zeros_like(Z)
    k   = min(k, Z.shape[1])
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    rows    = np.arange(Z.shape[0])[:, None]
    out[rows, top_idx] = Z[rows, top_idx]
    return out


def topk_indices(Z_raw, k):
    """Return per-row indices of the top-k entries (selected on the RAW
    log-prob ranking — equivalent to ranking on the scaled values since
    StandardScaler shifts/scales each column identically across rows)."""
    k = min(k, Z_raw.shape[1])
    return np.argpartition(Z_raw, -k, axis=1)[:, -k:]


topk_cont_train = np.hstack([
    topk_sparsify(ctx_pool_train_L, TOP_K),
    topk_sparsify(ctx_pool_train_R, TOP_K),
]).astype(np.float32)
topk_cont_test  = np.hstack([
    topk_sparsify(ctx_pool_test_L,  TOP_K),
    topk_sparsify(ctx_pool_test_R,  TOP_K),
]).astype(np.float32)
print(f"  TopK-Cont input dim: {topk_cont_train.shape[1]}  "
      f"(per-side k={TOP_K} non-zeros)")

# Per-row TopK indices into pool (used by TopK-Disc-* discrete bags)
topk_idx_train_L = topk_indices(_pool_train_L, TOP_K)
topk_idx_train_R = topk_indices(_pool_train_R, TOP_K)
topk_idx_test_L  = topk_indices(_pool_test_L,  TOP_K)
topk_idx_test_R  = topk_indices(_pool_test_R,  TOP_K)


# ── Path-based encodings (single top-1 leaf) ──────────────────────────────────
def best_leaf_chain(inst, leaves, leaf_chain):
    """Return [(depth, node), …] root→leaf for the leaf with highest log_prob."""
    best_lp, best_leaf = -math.inf, leaves[0]
    for lf in leaves:
        lp = lf.log_prob_instance(inst)
        if lp > best_lp:
            best_lp, best_leaf = lp, lf
    return leaf_chain[id(best_leaf)]


print("\nBuilding path-based context encodings (top-1 leaf) …")


def encode_path_variants(insts):
    """Return parallel lists of chains and per-node log-probs along that chain."""
    chains, lp_lists = [], []
    for inst in insts:
        chain = best_leaf_chain(inst, path_leaves, path_leaf_chain)
        lps   = [n.log_prob_instance(inst) for _, n in chain]
        chains.append(chain)
        lp_lists.append(lps)
    return chains, lp_lists


train_chains_L, train_lps_L = encode_path_variants(train_L)
train_chains_R, train_lps_R = encode_path_variants(train_R)
test_chains_L,  test_lps_L  = encode_path_variants(test_L)
test_chains_R,  test_lps_R  = encode_path_variants(test_R)
print(f"  Path encodings built: {len(train_chains_L)} train, {len(test_chains_L)} test")


# Convert chains into discrete-instance dicts ----------------------------------
def pathbag_cnt1_inst(chain_L, chain_R):
    """{0:{nid:1}, 1:{nid:1}} — bag of ancestor ids, count=1."""
    left  = {path_node_to_idx[id(n)]: 1.0 for _, n in chain_L}
    right = {path_node_to_idx[id(n)]: 1.0 for _, n in chain_R}
    return {0: left, 1: right}


def pathbag_depth_inst(chain_L, chain_R):
    """attrs 0..PATH_DEPTH → left depth d  ;  PATH_DEPTH+1..2*PATH_DEPTH+1 → right.
       Only one node per (side, depth)."""
    out = {}
    for d, n in chain_L:
        out[d] = {path_node_to_idx[id(n)]: 1.0}
    for d, n in chain_R:
        out[PATH_DEPTH + 1 + d] = {path_node_to_idx[id(n)]: 1.0}
    return out


def pathbag_logprob_inst(chain_L, chain_R, lps_L, lps_R):
    """{0:{nid:exp(lp - max_lp)}, 1:{nid:exp(lp - max_lp)}} — relative-likelihood
       counts.  Best ancestor on each side gets 1.0; others ∈ (0, 1].
       Keeps log-prob ranking, stays positive (Cobweb counts must be ≥0)."""
    def _bag(chain, lps):
        if not lps:
            return {}
        m = max(lps)
        return {path_node_to_idx[id(n)]: math.exp(lp - m)
                for (_, n), lp in zip(chain, lps)}
    return {0: _bag(chain_L, lps_L), 1: _bag(chain_R, lps_R)}


# Convert TopK indices into discrete-instance dicts ----------------------------
def topkdisc_cnt1_inst(idx_L, idx_R):
    return {0: {int(j): 1.0 for j in idx_L},
            1: {int(j): 1.0 for j in idx_R}}


def topkdisc_logprob_inst(idx_L, idx_R, raw_L, raw_R):
    """exp(lp - max) over the K selected pool nodes (positive, monotone)."""
    sub_L = raw_L[idx_L]; m_L = sub_L.max() if len(sub_L) else 0.0
    sub_R = raw_R[idx_R]; m_R = sub_R.max() if len(sub_R) else 0.0
    return {0: {int(j): math.exp(lp - m_L) for j, lp in zip(idx_L, sub_L)},
            1: {int(j): math.exp(lp - m_R) for j, lp in zip(idx_R, sub_R)}}


# Build the seven instance lists ------------------------------------------------
print("Building per-variant pair instances …")

def _disc_list_from_path(chains_L, chains_R, lps_L, lps_R, kind):
    out = []
    if kind == "cnt1":
        for cL, cR in zip(chains_L, chains_R):
            out.append(pathbag_cnt1_inst(cL, cR))
    elif kind == "depth":
        for cL, cR in zip(chains_L, chains_R):
            out.append(pathbag_depth_inst(cL, cR))
    elif kind == "logprob":
        for cL, cR, lL, lR in zip(chains_L, chains_R, lps_L, lps_R):
            out.append(pathbag_logprob_inst(cL, cR, lL, lR))
    return out


def _disc_list_from_topk(idx_L, idx_R, raw_L, raw_R, kind):
    out = []
    if kind == "cnt1":
        for iL, iR in zip(idx_L, idx_R):
            out.append(topkdisc_cnt1_inst(iL, iR))
    elif kind == "logprob":
        for iL, iR, rL, rR in zip(idx_L, idx_R, raw_L, raw_R):
            out.append(topkdisc_logprob_inst(iL, iR, rL, rR))
    return out


pathbag_cnt1_train    = _disc_list_from_path(train_chains_L, train_chains_R,
                                             train_lps_L, train_lps_R, "cnt1")
pathbag_cnt1_test     = _disc_list_from_path(test_chains_L,  test_chains_R,
                                             test_lps_L,  test_lps_R,  "cnt1")
pathbag_depth_train   = _disc_list_from_path(train_chains_L, train_chains_R,
                                             train_lps_L, train_lps_R, "depth")
pathbag_depth_test    = _disc_list_from_path(test_chains_L,  test_chains_R,
                                             test_lps_L,  test_lps_R,  "depth")
pathbag_logprob_train = _disc_list_from_path(train_chains_L, train_chains_R,
                                             train_lps_L, train_lps_R, "logprob")
pathbag_logprob_test  = _disc_list_from_path(test_chains_L,  test_chains_R,
                                             test_lps_L,  test_lps_R,  "logprob")

topk_disc_cnt1_train    = _disc_list_from_topk(topk_idx_train_L, topk_idx_train_R,
                                               _pool_train_L, _pool_train_R, "cnt1")
topk_disc_cnt1_test     = _disc_list_from_topk(topk_idx_test_L,  topk_idx_test_R,
                                               _pool_test_L,  _pool_test_R,  "cnt1")
topk_disc_logprob_train = _disc_list_from_topk(topk_idx_train_L, topk_idx_train_R,
                                               _pool_train_L, _pool_train_R, "logprob")
topk_disc_logprob_test  = _disc_list_from_topk(topk_idx_test_L,  topk_idx_test_R,
                                               _pool_test_L,  _pool_test_R,  "logprob")


# ── BFSBag-* encodings (BFS K leaves per side from the context tree) ──────────
# Uses CobwebDiscreteTree.bfs_top_k_leaves(inst, k, max_nodes) — a
# priority-queue BFS that returns up to K leaves under the query, bounded by
# max_nodes pops (same style as tree.log_prob).
#
print("\nBuilding BFSBag-* encodings (K=%d, max_nodes=%d) …" % (BFS_K, BFS_MAX_NODES))

# Variant: TopK-Pool-Cache (incremental analog of TopK-Disc-Cnt1)
# Cached pool of depth-d context-tree nodes; for each query, score every
# pool node via log_prob_instance and emit a set of the top-K stable
# concept-hash ids. The encoder maintains a {pool_id → canonical_id}
# remap in lockstep that survives merges / splits / deletions in the
# context tree (see TopKPoolEncoder for details).
_pool_encoder_moc       = TopKPoolEncoder(
    context_tree=context_tree, depth=REMAP_DEPTH, k=BFS_K,
)
_pool_content_tree_moc  = CobwebDiscreteTree(
    alpha=ALPHA_BFSBAG_REMAP, weight_attr=True,
)
print(f"  TopK-Pool-Cache : pool size = {_pool_encoder_moc.pool_size} "
      f"depth-{REMAP_DEPTH} nodes, K={BFS_K}")

def _topk_pool_inst(inst_L, inst_R):
    # Defer the set_value_remap push (content_tree=None) — we sync once
    # after both train and test bags are built.
    return {0: _pool_encoder_moc.bag_for(inst_L, None),
            1: _pool_encoder_moc.bag_for(inst_R, None)}

bfsbag_remap_train = [_topk_pool_inst(L, R) for L, R in zip(train_L, train_R)]
bfsbag_remap_test  = [_topk_pool_inst(L, R) for L, R in zip(test_L,  test_R)]
_pool_encoder_moc.sync_remap(_pool_content_tree_moc)
print(f"  TopK-Pool-Cache : value_vocab={len(_pool_encoder_moc.value_vocab)} pool ids, "
      f"remap_dict={len(_pool_encoder_moc.value_remap_dict)} (non-identity), "
      f"use_value_remap={_pool_content_tree_moc.use_value_remap}")
print(f"  TopK-Pool-Cache last refresh stats: {_pool_encoder_moc.last_refresh_stats}  "
      f"(moved/rescued/orphaned)")

# Variant: BFSBag-BLRemap (remap each BFS-leaf to its basic-level ancestor)
_bl_cache_moc        = BasicLevelCache(context_tree, eval_alpha=BL_EVAL_ALPHA)
_blremap_hash_to_id  : dict = {}
def _blremap_id_moc(h: str) -> int:
    lid = _blremap_hash_to_id.get(h)
    if lid is None:
        lid = len(_blremap_hash_to_id) + 1
        _blremap_hash_to_id[h] = lid
    return lid

def _bfsbag_blremap_inst(inst_L, inst_R):
    bag_L = bfsbag_blremap_instance(context_tree, inst_L,
                                    BFS_K, BFS_MAX_NODES, _bl_cache_moc)
    bag_R = bfsbag_blremap_instance(context_tree, inst_R,
                                    BFS_K, BFS_MAX_NODES, _bl_cache_moc)
    return {0: {_blremap_id_moc(h): float(c) for h, c in bag_L.items()},
            1: {_blremap_id_moc(h): float(c) for h, c in bag_R.items()}}

print("  Building BFSBag-BLRemap instances "
      "(calls leaf.get_basic for every new BFS-leaf — slowest setup) …")
_t = time.time()
bfsbag_blremap_train = [_bfsbag_blremap_inst(L, R) for L, R in zip(train_L, train_R)]
bfsbag_blremap_test  = [_bfsbag_blremap_inst(L, R) for L, R in zip(test_L,  test_R)]
print(f"  BFSBag-BLRemap : bl_vocab={len(_blremap_hash_to_id)} "
      f"bl_cache_size={len(_bl_cache_moc)}  built in {time.time() - _t:.1f}s")

print("  Per-variant pair instances built.")


# ── Train each content tree ───────────────────────────────────────────────────
def train_disc_content(name, alpha, train_insts):
    print(f"\n[{name}] Training CobwebDiscreteTree (alpha={alpha}, "
          f"n_train={len(train_insts)}) …")
    tree = CobwebDiscreteTree(alpha=alpha, weight_attr=True)
    for i, inst in enumerate(train_insts):
        tree.ifit(inst)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(train_insts)} inserted")
    print(f"[{name}] Tree built.")
    return tree


def train_cont_content(name, alpha, train_vecs):
    print(f"\n[{name}] Training CobwebContinuousTree (alpha={alpha}, "
          f"size={train_vecs.shape[1]}, n_train={train_vecs.shape[0]}) …")
    tree = CobwebContinuousTree(size=train_vecs.shape[1],
                                covar_from=1,
                                num_labels=0,
                                alpha=alpha)
    empty_lbl = np.zeros(0, dtype=np.float32)
    for i, x in enumerate(train_vecs):
        tree.ifit(x, empty_lbl)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{train_vecs.shape[0]} inserted")
    print(f"[{name}] Tree built.")
    return tree


tree_pathbag_cnt1     = train_disc_content("PathBag-Cnt1",
                                           ALPHA_PATHBAG_CNT1, pathbag_cnt1_train)
tree_pathbag_depth    = train_disc_content("PathBag-DepthAttrs",
                                           ALPHA_PATHBAG_DEPTH, pathbag_depth_train)
tree_pathbag_logprob  = train_disc_content("PathBag-LogProb",
                                           ALPHA_PATHBAG_LOGPROB, pathbag_logprob_train)
tree_bfs_cont         = train_cont_content("BFS-Cont",
                                           ALPHA_BFS_CONT, bfs_cont_train)
tree_topk_cont        = train_cont_content("TopK-Cont",
                                           ALPHA_TOPK_CONT, topk_cont_train)
tree_topk_disc_cnt1   = train_disc_content("TopK-Disc-Cnt1",
                                           ALPHA_TOPK_DISC_CNT1, topk_disc_cnt1_train)
tree_topk_disc_logprob = train_disc_content("TopK-Disc-LogProb",
                                            ALPHA_TOPK_DISC_LOGPROB,
                                            topk_disc_logprob_train)

# TopK-Pool-Cache: pre-created above as ``_pool_content_tree_moc`` so the
# encoder's set_value_remap pushes land on the actual tree object.
# Train in place — same fit loop as the other disc variants.
tree_bfsbag_remap = _pool_content_tree_moc
print(f"\n[TopK-Pool-Cache] Training pre-created CobwebDiscreteTree "
      f"(alpha={ALPHA_BFSBAG_REMAP}, n_train={len(bfsbag_remap_train)}) …")
for _i, _inst in enumerate(bfsbag_remap_train):
    tree_bfsbag_remap.ifit(_inst)
    if (_i + 1) % 2000 == 0:
        print(f"  {_i + 1}/{len(bfsbag_remap_train)} inserted")
print(f"[TopK-Pool-Cache] Tree built. use_value_remap="
      f"{tree_bfsbag_remap.use_value_remap}")

# BFSBag-BLRemap: plain CobwebDiscreteTree (no ref_tree).  Attribute values
# are integer ids for *basic-level* targets (one per leaf, found via
# leaf.get_basic(use_root=True, eval_alpha=BL_EVAL_ALPHA)).
tree_bfsbag_blremap = train_disc_content("BFSBag-BLRemap",
                                         ALPHA_BFSBAG_BLREMAP, bfsbag_blremap_train)


# ── Extract BFS-DZ_CONTENT features per content tree ──────────────────────────
_empty_lbl = np.zeros(0, dtype=np.float32)


def encode_disc(instances, nodes):
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, inst in enumerate(instances):
            out[i, j] = node.log_prob_instance(inst)
    return out


def encode_cont(vecs, nodes):
    out = np.empty((vecs.shape[0], len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i in range(vecs.shape[0]):
            out[i, j] = node.log_prob(vecs[i], _empty_lbl)
    return out


def bfs_or_root(tree, k):
    nodes = bfs_first_n_nodes(tree.root, k)
    return nodes if nodes else [tree.root]


print("\nEncoding pair → content-tree BFS features …")


def _encode_disc_variant(name, tree, train_insts, test_insts):
    t0 = time.time()
    nodes = bfs_or_root(tree, DZ_CONTENT)
    print(f"  [{name}] BFS-{DZ_CONTENT}: {len(nodes)} nodes", flush=True)
    print(f"  [{name}]   encoding train ({len(train_insts)} pairs) …", flush=True)
    t1 = time.time()
    Z_tr_raw = encode_disc(train_insts, nodes)
    print(f"  [{name}]   train encoded in {time.time() - t1:.1f}s", flush=True)
    print(f"  [{name}]   encoding test  ({len(test_insts)} pairs) …", flush=True)
    t1 = time.time()
    Z_te_raw = encode_disc(test_insts,  nodes)
    print(f"  [{name}]   test  encoded in {time.time() - t1:.1f}s", flush=True)
    sc = StandardScaler().fit(Z_tr_raw)
    print(f"  [{name}]   total {time.time() - t0:.1f}s", flush=True)
    return sc.transform(Z_tr_raw), sc.transform(Z_te_raw), nodes


def _encode_cont_variant(name, tree, train_vecs, test_vecs):
    t0 = time.time()
    nodes = bfs_or_root(tree, DZ_CONTENT)
    print(f"  [{name}] BFS-{DZ_CONTENT}: {len(nodes)} nodes", flush=True)
    print(f"  [{name}]   encoding train ({train_vecs.shape[0]} pairs) …", flush=True)
    t1 = time.time()
    Z_tr_raw = encode_cont(train_vecs, nodes)
    print(f"  [{name}]   train encoded in {time.time() - t1:.1f}s", flush=True)
    print(f"  [{name}]   encoding test  ({test_vecs.shape[0]} pairs) …", flush=True)
    t1 = time.time()
    Z_te_raw = encode_cont(test_vecs,  nodes)
    print(f"  [{name}]   test  encoded in {time.time() - t1:.1f}s", flush=True)
    sc = StandardScaler().fit(Z_tr_raw)
    print(f"  [{name}]   total {time.time() - t0:.1f}s", flush=True)
    return sc.transform(Z_tr_raw), sc.transform(Z_te_raw), nodes


Z_pbcnt1_tr,  Z_pbcnt1_te,  nodes_pbcnt1 = _encode_disc_variant(
    "PathBag-Cnt1",     tree_pathbag_cnt1,    pathbag_cnt1_train,    pathbag_cnt1_test)
Z_pbdep_tr,   Z_pbdep_te,   nodes_pbdep   = _encode_disc_variant(
    "PathBag-Depth",    tree_pathbag_depth,   pathbag_depth_train,   pathbag_depth_test)
Z_pblp_tr,    Z_pblp_te,    nodes_pblp    = _encode_disc_variant(
    "PathBag-LogProb",  tree_pathbag_logprob, pathbag_logprob_train, pathbag_logprob_test)
Z_bfsc_tr,    Z_bfsc_te,    nodes_bfsc    = _encode_cont_variant(
    "BFS-Cont",         tree_bfs_cont,        bfs_cont_train,        bfs_cont_test)
Z_tkc_tr,     Z_tkc_te,     nodes_tkc     = _encode_cont_variant(
    "TopK-Cont",        tree_topk_cont,       topk_cont_train,       topk_cont_test)
Z_tkdc_tr,    Z_tkdc_te,    nodes_tkdc    = _encode_disc_variant(
    "TopK-Disc-Cnt1",   tree_topk_disc_cnt1,  topk_disc_cnt1_train,  topk_disc_cnt1_test)
Z_tkdlp_tr,   Z_tkdlp_te,   nodes_tkdlp   = _encode_disc_variant(
    "TopK-Disc-LogProb",tree_topk_disc_logprob,topk_disc_logprob_train,
    topk_disc_logprob_test)
Z_bbrm_tr,    Z_bbrm_te,    nodes_bbrm    = _encode_disc_variant(
    "TopK-Pool-Cache",  tree_bfsbag_remap,    bfsbag_remap_train,    bfsbag_remap_test)
Z_bbbl_tr,    Z_bbbl_te,    nodes_bbbl    = _encode_disc_variant(
    "BFSBag-BLRemap",   tree_bfsbag_blremap,  bfsbag_blremap_train,  bfsbag_blremap_test)

# Save all feature arrays ------------------------------------------------------
for name, Ztr, Zte in [
    ("pbcnt1",  Z_pbcnt1_tr,  Z_pbcnt1_te),
    ("pbdep",   Z_pbdep_tr,   Z_pbdep_te),
    ("pblp",    Z_pblp_tr,    Z_pblp_te),
    ("bfsc",    Z_bfsc_tr,    Z_bfsc_te),
    ("tkc",     Z_tkc_tr,     Z_tkc_te),
    ("tkdc",    Z_tkdc_tr,    Z_tkdc_te),
    ("tkdlp",   Z_tkdlp_tr,   Z_tkdlp_te),
    ("bbrm",    Z_bbrm_tr,    Z_bbrm_te),
    ("bbbl",    Z_bbbl_tr,    Z_bbbl_te),
]:
    np.save(os.path.join(ARR_DIR, f"Z_{name}_train.npy"), Ztr)
    np.save(os.path.join(ARR_DIR, f"Z_{name}_test.npy"),  Zte)
print("Arrays saved.")


# ── Evaluation ────────────────────────────────────────────────────────────────
KNN_KS = [1, 3, 5, 10, 20, 50]
CLASSES = sorted(set(y_train.tolist()) | set(y_test.tolist()))
N_CLS   = len(CLASSES)
print(f"\nDistinct pair-label classes (train ∪ test): {N_CLS}")


def linear_probe_per_class(Z_tr, y_tr, Z_te, y_te):
    lin = LinearSVC(max_iter=4000)
    lin.fit(Z_tr, y_tr)
    overall = lin.score(Z_te, y_te)
    per_cls = np.array([
        lin.score(Z_te[y_te == c], y_te[y_te == c]) if (y_te == c).sum() > 0 else 0.0
        for c in CLASSES
    ])
    return overall, per_cls


def knn_accuracy_vs_k(Z_tr, y_tr, Z_te, y_te, ks=KNN_KS):
    return [KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y_tr).score(Z_te, y_te)
            for k in ks]


def _repr_stats(Z):
    nz = (Z != 0)
    return nz.sum(axis=1).mean(), (~nz.any(axis=0)).mean() * 100


def softmax_entropy(Z):
    shifted = Z - Z.max(axis=1, keepdims=True)
    expZ    = np.exp(shifted)
    p       = expZ / expZ.sum(axis=1, keepdims=True)
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)


print("\nEvaluating …")
# Variant labels show tree-type and the input-side constants that actually
# distinguish each variant.  (Output BFS feature dim is always DZ_CONTENT={DZ}
# across all variants, so we omit it here.)
results = {}
for short, lbl, Ztr, Zte in [
    ("pbcnt1",  f"PathBag-Cnt1 (disc, path_d={PATH_DEPTH})",
                                                                Z_pbcnt1_tr, Z_pbcnt1_te),
    ("pbdep",   f"PathBag-DepthAttrs (disc, path_d={PATH_DEPTH})",
                                                                Z_pbdep_tr,  Z_pbdep_te),
    ("pblp",    f"PathBag-LogProb (disc, path_d={PATH_DEPTH})",
                                                                Z_pblp_tr,   Z_pblp_te),
    ("bfsc",    f"BFS-Cont (cont, in={bfs_cont_train.shape[1]})",
                                                                Z_bfsc_tr,   Z_bfsc_te),
    ("tkc",     f"TopK-Cont (cont, in={topk_cont_train.shape[1]}, "
                f"k={TOP_K}, depth={topk_depth})",              Z_tkc_tr,    Z_tkc_te),
    ("tkdc",    f"TopK-Disc-Cnt1 (disc, k={TOP_K}, depth={topk_depth})",
                                                                Z_tkdc_tr,   Z_tkdc_te),
    ("tkdlp",   f"TopK-Disc-LogProb (disc, k={TOP_K}, depth={topk_depth})",
                                                                Z_tkdlp_tr,  Z_tkdlp_te),
    ("bbrm",    f"TopK-Pool-Cache (disc, K={BFS_K}, depth={REMAP_DEPTH}, "
                f"stable_concept_hash ids)",                    Z_bbrm_tr,   Z_bbrm_te),
    ("bbbl",    f"BFSBag-BLRemap (disc, K={BFS_K}, max_nodes={BFS_MAX_NODES}, "
                f"target=basic-level, bl_eval_α={BL_EVAL_ALPHA})",
                                                                Z_bbbl_tr,   Z_bbbl_te),
]:
    print(f"  evaluating {lbl} …")
    lin_overall, lin_per = linear_probe_per_class(Ztr, y_train, Zte, y_test)
    knn_accs             = knn_accuracy_vs_k(Ztr, y_train, Zte, y_test)
    avg_l0, dead_pct     = _repr_stats(Ztr)
    avg_ent              = softmax_entropy(Ztr).mean()
    results[short] = dict(label=lbl, lin_overall=lin_overall, lin_per=lin_per,
                          knn_accs=knn_accs, Ztr=Ztr, Zte=Zte,
                          avg_l0=avg_l0, dead_pct=dead_pct, avg_ent=avg_ent)


print(f"\n  {'Method':<55} {'Lin.Probe':>10} {'KNN@5':>7} {'Avg L0':>8} {'Dead%':>7}")
print(f"  {'-' * 95}")
_knn5_idx     = KNN_KS.index(5)
_summary_rows = []
for short, r in results.items():
    knn5 = r["knn_accs"][_knn5_idx] * 100
    print(f"  {r['label']:<55} {r['lin_overall']*100:>9.1f}% "
          f"{knn5:>6.1f}% {r['avg_l0']:>8.1f} {r['dead_pct']:>6.1f}%  "
          f"ent={r['avg_ent']:.3f}")
    _summary_rows.append({
        "method":         r["label"],
        "lin_probe_pct":  round(r["lin_overall"] * 100, 2),
        "knn5_pct":       round(knn5, 2),
        "avg_l0":         round(float(r["avg_l0"]), 2),
        "dead_pct":       round(float(r["dead_pct"]), 2),
        "avg_entropy":    round(float(r["avg_ent"]), 4),
    })

_csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(_csv_path, "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=["method", "lin_probe_pct", "knn5_pct",
                                         "avg_l0", "dead_pct", "avg_entropy"])
    _w.writeheader()
    _w.writerows(_summary_rows)
print(f"  Summary saved → {_csv_path}")


# ── Tree visualisation: stacked left/right POS bars at each node ──────────────
CMAP_POS    = plt.get_cmap("tab10") if N_POS <= 10 else plt.get_cmap("tab20")
pos_colors  = [CMAP_POS(i / max(N_POS - 1, 1)) for i in range(N_POS)]


# Note: cobweb_{discrete,continuous}.children returns fresh Python wrappers each
# call (id() is NOT stable across iterations), so we materialise a static layout
# once per tree and key everything off integer indices into the captured node
# list (which keeps the wrappers alive and gives stable identity).

def make_layout(root, max_depth):
    """Single BFS up to max_depth.  Returns:
         all_nodes      list[node]                — refs stay alive, idx is identity
         children_of    {idx: [child_idx, …]}
         depth_of       {idx: depth}"""
    all_nodes   = [root]
    children_of = {0: []}
    depth_of    = {0: 0}
    queue       = [0]
    while queue:
        idx  = queue.pop(0)
        node = all_nodes[idx]
        if depth_of[idx] < max_depth:
            for c in node.children:
                cidx = len(all_nodes)
                all_nodes.append(c)
                children_of[idx].append(cidx)
                children_of[cidx] = []
                depth_of[cidx]    = depth_of[idx] + 1
                queue.append(cidx)
    return all_nodes, children_of, depth_of


def descend_layout_disc(all_nodes, children_of, inst, max_depth):
    visited, cur = [0], 0
    for _ in range(max_depth):
        ch = children_of[cur]
        if not ch:
            break
        best = max(ch, key=lambda i: all_nodes[i].log_prob_instance(inst))
        visited.append(best)
        cur = best
    return visited


def descend_layout_cont(all_nodes, children_of, vec, max_depth):
    visited, cur = [0], 0
    for _ in range(max_depth):
        ch = children_of[cur]
        if not ch:
            break
        best = max(ch, key=lambda i: all_nodes[i].log_prob(vec, _empty_lbl))
        visited.append(best)
        cur = best
    return visited


def compute_pair_label_counts_idx(all_nodes, children_of,
                                   train_data, y_tr, descend_fn, max_depth=3):
    cL, cR = {}, {}
    for x, lbl in zip(train_data, y_tr):
        l_pos, r_pos = split_pair_label(int(lbl))
        for idx in descend_fn(all_nodes, children_of, x, max_depth):
            if idx not in cL:
                cL[idx] = np.zeros(N_POS, dtype=np.int32)
                cR[idx] = np.zeros(N_POS, dtype=np.int32)
            cL[idx][l_pos] += 1
            cR[idx][r_pos] += 1
    return cL, cR


def plot_pair_tree_idx(children_of, depth_of, counts_L, counts_R,
                        title, out_path, max_depth=3):
    def leaf_span(idx, depth):
        if depth >= max_depth or not children_of[idx]:
            return 1
        return sum(leaf_span(c, depth + 1) for c in children_of[idx])

    pos = {}
    def assign_pos(idx, depth, x_left):
        span     = leaf_span(idx, depth)
        x_centre = x_left + span / 2.0
        pos[idx] = (x_centre, depth)
        if depth < max_depth and children_of[idx]:
            cur = x_left
            for c in children_of[idx]:
                cs = leaf_span(c, depth + 1)
                assign_pos(c, depth + 1, cur)
                cur += cs

    assign_pos(0, 0, 0.0)
    total_w = leaf_span(0, 0)

    bar_w, bar_h, gap, y_unit = 0.7, 0.18, 0.05, 1.0
    fig, ax = plt.subplots(figsize=(max(14, total_w * 0.9),
                                    (max_depth + 1) * 2.4))
    ax.set_xlim(0, total_w)
    ax.set_ylim(-0.7, max_depth * y_unit + 0.7)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    def draw_edges(idx, depth):
        if depth >= max_depth or not children_of[idx]:
            return
        px, py = pos[idx]
        for c in children_of[idx]:
            cx, cy = pos[c]
            y_par = py * y_unit + bar_h + gap / 2
            y_chi = cy * y_unit - bar_h - gap / 2
            ax.plot([px, cx], [y_par, y_chi], color="gray", lw=0.8, zorder=0)
            draw_edges(c, depth + 1)
    draw_edges(0, 0)

    def _draw_bar(x_left, y_top, props, label_text):
        cur = x_left
        for p_idx in range(N_POS):
            seg_w = props[p_idx] * bar_w
            if seg_w > 0:
                ax.add_patch(plt.Rectangle((cur, y_top), seg_w, bar_h,
                                            color=pos_colors[p_idx], lw=0))
                cur += seg_w
        ax.add_patch(plt.Rectangle((x_left, y_top), bar_w, bar_h,
                                    fill=False, edgecolor="black", lw=0.5))
        ax.text(x_left - 0.05, y_top + bar_h / 2, label_text,
                ha="right", va="center", fontsize=5)

    def draw_node(idx, depth):
        if idx not in counts_L:
            return
        cntL = counts_L[idx].astype(float)
        cntR = counts_R[idx].astype(float)
        tot  = cntL.sum()
        if tot == 0:
            return
        propsL = cntL / tot
        propsR = cntR / tot
        x_c, _ = pos[idx]
        x_left = x_c - bar_w / 2
        y_top_L = depth * y_unit - bar_h - gap / 2
        y_top_R = depth * y_unit + gap / 2
        _draw_bar(x_left, y_top_L, propsL, "L")
        _draw_bar(x_left, y_top_R, propsR, "R")
        ax.text(x_c, depth * y_unit - bar_h - gap / 2 - 0.04,
                f"n={int(tot)}", ha="center", va="bottom", fontsize=5)
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]:
                draw_node(c, depth + 1)
    draw_node(0, 0)

    legend_h = [plt.Rectangle((0, 0), 1, 1, color=pos_colors[i], label=id2pos[i])
                for i in range(N_POS)]
    ax.legend(handles=legend_h, title="POS (Left top bar / Right bottom bar)",
              loc="lower right", ncol=max(1, N_POS // 4),
              fontsize=6, title_fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


print("\nRendering content-tree visualizations …")
for short, tree, train_data, descend_fn in [
    ("pathbag_cnt1",      tree_pathbag_cnt1,     pathbag_cnt1_train,     descend_layout_disc),
    ("pathbag_depth",     tree_pathbag_depth,    pathbag_depth_train,    descend_layout_disc),
    ("pathbag_logprob",   tree_pathbag_logprob,  pathbag_logprob_train,  descend_layout_disc),
    ("bfs_cont",          tree_bfs_cont,         bfs_cont_train,         descend_layout_cont),
    ("topk_cont",         tree_topk_cont,        topk_cont_train,        descend_layout_cont),
    ("topk_disc_cnt1",    tree_topk_disc_cnt1,   topk_disc_cnt1_train,   descend_layout_disc),
    ("topk_disc_logprob", tree_topk_disc_logprob,topk_disc_logprob_train,descend_layout_disc),
    ("topk_pool_cache",   tree_bfsbag_remap,     bfsbag_remap_train,     descend_layout_disc),
    ("bfsbag_blremap",    tree_bfsbag_blremap,   bfsbag_blremap_train,   descend_layout_disc),
]:
    all_nodes, children_of, depth_of = make_layout(tree.root, max_depth=3)
    cL, cR = compute_pair_label_counts_idx(all_nodes, children_of,
                                            train_data, y_train,
                                            descend_fn, max_depth=3)
    plot_pair_tree_idx(children_of, depth_of, cL, cR,
                        title=f"{short} — Pair (Left/Right POS) distributions (depths 0–3)",
                        out_path=os.path.join(TREE_DIR, f"{short}.png"),
                        max_depth=3)
    print(f"  {short} → {os.path.join(TREE_DIR, short + '.png')}")


# ── Aggregate visualizations ──────────────────────────────────────────────────
METHODS = [
    (Z_pbcnt1_tr,  Z_pbcnt1_te,  results["pbcnt1"]["lin_per"],  results["pbcnt1"]["knn_accs"],
     results["pbcnt1"]["label"], "o-", "#6acc65"),
    (Z_pbdep_tr,   Z_pbdep_te,   results["pbdep"]["lin_per"],   results["pbdep"]["knn_accs"],
     results["pbdep"]["label"],  "s-", "#2a7d3a"),
    (Z_pblp_tr,    Z_pblp_te,    results["pblp"]["lin_per"],    results["pblp"]["knn_accs"],
     results["pblp"]["label"],   "v-", "#a8e3a0"),
    (Z_bfsc_tr,    Z_bfsc_te,    results["bfsc"]["lin_per"],    results["bfsc"]["knn_accs"],
     results["bfsc"]["label"],   "D-", "#d65f5f"),
    (Z_tkc_tr,     Z_tkc_te,     results["tkc"]["lin_per"],     results["tkc"]["knn_accs"],
     results["tkc"]["label"],    "P-", "#956cb4"),
    (Z_tkdc_tr,    Z_tkdc_te,    results["tkdc"]["lin_per"],    results["tkdc"]["knn_accs"],
     results["tkdc"]["label"],   "X-", "#17becf"),
    (Z_tkdlp_tr,   Z_tkdlp_te,   results["tkdlp"]["lin_per"],   results["tkdlp"]["knn_accs"],
     results["tkdlp"]["label"],  "p-", "#8c564b"),
    (Z_bbrm_tr,    Z_bbrm_te,    results["bbrm"]["lin_per"],    results["bbrm"]["knn_accs"],
     results["bbrm"]["label"],   "*-", "#e377c2"),
    (Z_bbbl_tr,    Z_bbbl_te,    results["bbbl"]["lin_per"],    results["bbbl"]["knn_accs"],
     results["bbbl"]["label"],   "1-", "#bcbd22"),
]
n_meth = len(METHODS)

# Pair-class color palette for scatter plots
CMAP_PAIR = plt.get_cmap("tab20") if N_CLS <= 20 else plt.get_cmap("hsv")


def _pair_color(i):
    return CMAP_PAIR(i / max(N_CLS - 1, 1))


def _pair_label_str(c):
    l, r = split_pair_label(c)
    return f"{id2pos[l]}-{id2pos[r]}"


# Subsample legend: only the most frequent classes
class_freq  = np.array([(y_train == c).sum() for c in CLASSES])
_top_legend = np.argsort(-class_freq)[:min(15, N_CLS)]
_leg_handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=_pair_color(i),
               markersize=6, label=_pair_label_str(CLASSES[i]))
    for i in _top_legend
]


# 1a. UMAP scatter
print("\nComputing UMAP projections …")
projs_umap = []
for Z, _, _, _, _, _, _ in METHODS:
    projs_umap.append(UMAP(n_components=2, random_state=SEED).fit_transform(Z))

fig, axes = plt.subplots(1, n_meth, figsize=(n_meth * 5, 5))
fig.suptitle("UMAP Projections — Grammar Chunking (TEST_GRAMMAR3)",
             fontsize=11, y=1.01)
for ax, Z2, (_, _, _, _, lbl, _, _) in zip(axes, projs_umap, METHODS):
    for ci, c in enumerate(CLASSES):
        mask = y_train == c
        if mask.any():
            ax.scatter(Z2[mask, 0], Z2[mask, 1], color=_pair_color(ci),
                       alpha=0.45, s=6)
    ax.set_title(lbl, fontsize=8)
    ax.set_xlabel("Dim 1", fontsize=7)
    ax.set_ylabel("Dim 2", fontsize=7)
    ax.tick_params(labelsize=6)
fig.legend(handles=_leg_handles, title="POS-pair (top freq)",
           loc="center right", bbox_to_anchor=(1.0, 0.5),
           ncol=1, fontsize=6, title_fontsize=7, frameon=True)
plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig(os.path.join(OUT_DIR, "scatter_umap.png"), dpi=120, bbox_inches="tight")
plt.close()
print("UMAP scatter saved.")

# 1b. t-SNE scatter
print("Computing t-SNE projections …")
projs_tsne = []
for Z, _, _, _, _, _, _ in METHODS:
    projs_tsne.append(TSNE(n_components=2, random_state=SEED, n_jobs=-1)
                      .fit_transform(Z))

fig, axes = plt.subplots(1, n_meth, figsize=(n_meth * 5, 5))
fig.suptitle("t-SNE Projections — Grammar Chunking (TEST_GRAMMAR3)",
             fontsize=11, y=1.01)
for ax, Z2, (_, _, _, _, lbl, _, _) in zip(axes, projs_tsne, METHODS):
    for ci, c in enumerate(CLASSES):
        mask = y_train == c
        if mask.any():
            ax.scatter(Z2[mask, 0], Z2[mask, 1], color=_pair_color(ci),
                       alpha=0.45, s=6)
    ax.set_title(lbl, fontsize=8)
    ax.set_xlabel("Dim 1", fontsize=7)
    ax.set_ylabel("Dim 2", fontsize=7)
    ax.tick_params(labelsize=6)
fig.legend(handles=_leg_handles, title="POS-pair (top freq)",
           loc="center right", bbox_to_anchor=(1.0, 0.5),
           ncol=1, fontsize=6, title_fontsize=7, frameon=True)
plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig(os.path.join(OUT_DIR, "scatter_tsne.png"), dpi=120, bbox_inches="tight")
plt.close()
print("t-SNE scatter saved.")

# 2. Linear probe — per-pair-class test accuracy
w_bar   = 0.8 / n_meth
x_bar   = np.arange(N_CLS)
offsets = [(i - (n_meth - 1) / 2) * w_bar for i in range(n_meth)]
fig, ax = plt.subplots(figsize=(max(16, N_CLS * 0.5), 5))
for (_, _, per, _, lbl, _, color), offset in zip(METHODS, offsets):
    ax.bar(x_bar + offset, per * 100, w_bar, label=lbl, color=color, alpha=0.85)
ax.set_xticks(x_bar)
ax.set_xticklabels([_pair_label_str(c) for c in CLASSES],
                   rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Test Accuracy %")
ax.set_title("Linear Probe — Per-(L,R)-POS Test Accuracy  "
             "(Grammar Chunking / TEST_GRAMMAR3)")
ax.set_ylim(0, 115)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "linear_probe_per_class.png"), dpi=120)
plt.close()

# 3. KNN accuracy vs k
fig, ax = plt.subplots(figsize=(7, 5))
for _, _, _, knn_accs, lbl, marker, color in METHODS:
    ax.plot(KNN_KS, [a * 100 for a in knn_accs], marker, label=lbl, color=color)
ax.set_xlabel("k (number of neighbours)")
ax.set_ylabel("Test Accuracy %")
ax.set_title("KNN Test Accuracy vs k  (Grammar Chunking / TEST_GRAMMAR3)")
ax.set_xticks(KNN_KS)
ax.set_ylim(0, 105)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "knn_vs_k.png"), dpi=120)
plt.close()

# 4. Dimension activity histograms + softmax entropy
_n_methods = n_meth
_n_cols    = 3
_n_rows    = (_n_methods + _n_cols - 1) // _n_cols
fig, axes  = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 3.8))
axes       = axes.flatten()
for idx, (Z_tr, _, _, _, lbl, _, color) in enumerate(METHODS):
    ax        = axes[idx]
    fire_freq = (Z_tr != 0).mean(axis=0)
    avg_ent   = softmax_entropy(Z_tr).mean()
    ax.hist(fire_freq, bins=60, color=color, alpha=0.82,
            edgecolor="white", linewidth=0.3)
    ax.axvline(fire_freq.mean(), color="black", linewidth=1.0, linestyle="--",
               label=f"mean={fire_freq.mean():.3f}")
    ax.set_title(f"{lbl}\nAvg softmax entropy: {avg_ent:.3f} nats", fontsize=8)
    ax.set_xlabel("Dimension fire frequency", fontsize=7)
    ax.set_ylabel("# dimensions", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
for idx in range(_n_methods, len(axes)):
    axes[idx].set_visible(False)
fig.suptitle("Per-dimension Activity Frequency  —  Grammar Chunking", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dim_activity_hist.png"), dpi=120, bbox_inches="tight")
plt.close()

# 5. Per-node token-count bar chart
fig, axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 6, _n_rows * 3.8))
axes      = axes.flatten()
for idx, (Z_tr, _, _, _, lbl, _, color) in enumerate(METHODS):
    ax          = axes[idx]
    node_counts = (Z_tr != 0).sum(axis=0)
    n_active    = (node_counts > 0).sum()
    n_dead      = (node_counts == 0).sum()
    ax.bar(np.arange(len(node_counts)), node_counts, color=color,
           alpha=0.82, width=1.0, linewidth=0)
    ax.axhline(node_counts[node_counts > 0].mean() if n_active > 0 else 0,
               color="black", linewidth=1.0, linestyle="--",
               label=(f"mean (active)={node_counts[node_counts > 0].mean():.1f}"
                      if n_active > 0 else "mean=0"))
    ax.set_title(f"{lbl}\nactive: {n_active} | dead: {n_dead} | "
                 f"total: {Z_tr.shape[1]}", fontsize=8)
    ax.set_xlabel("node index (BFS order)", fontsize=7)
    ax.set_ylabel("# training pairs node is active for", fontsize=7)
    ax.set_xlim(0, len(node_counts))
    ax.set_ylim(0, len(Z_tr) * 1.05)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
for idx in range(_n_methods, len(axes)):
    axes[idx].set_visible(False)
fig.suptitle("Per-node Pair Activity  —  Grammar Chunking", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "node_activity_hist.png"), dpi=120, bbox_inches="tight")
plt.close()

# 6. Competitive-node count histograms
def _competitive_node_fig(methods, threshold, fname_suffix):
    pct_str   = f"{int(threshold * 100)}%"
    _cn_cols  = min(3, len(methods))
    _cn_rows  = (len(methods) + _cn_cols - 1) // _cn_cols
    fig, axes = plt.subplots(_cn_rows, _cn_cols,
                             figsize=(_cn_cols * 5, _cn_rows * 3.8))
    axes      = np.array(axes).flatten()
    for idx, (Z_tr, lbl, color) in enumerate(methods):
        ax       = axes[idx]
        Z_f      = Z_tr.astype(np.float64)
        best     = Z_f.max(axis=1, keepdims=True)
        rel_prob = np.exp(np.clip(Z_f - best, -500, 0))
        counts   = (rel_prob >= threshold).sum(axis=1)
        if len(counts) == 0:
            max_count = 1
        else:
            hist_vals, _ = np.histogram(
                counts, bins=np.arange(0.5, counts.max() + 1.5, 1.0))
            nz_bins   = np.where(hist_vals > 0)[0]
            max_count = int(nz_bins[-1]) + 1 if len(nz_bins) else 1
        bins = np.arange(0.5, max_count + 1.5, 1.0)
        ax.hist(counts, bins=bins, color=color, alpha=0.82,
                edgecolor="white", linewidth=0.4)
        ax.axvline(counts.mean(), color="black", linewidth=1.2, linestyle="--",
                   label=f"mean={counts.mean():.2f}")
        ax.set_title(f"{lbl}\nmedian={np.median(counts):.0f}  |  "
                     f"mean={counts.mean():.2f}", fontsize=8)
        ax.set_xlabel(f"# nodes with relative prob ≥ {pct_str} of best",
                      fontsize=7)
        ax.set_ylabel("# training pairs", fontsize=7)
        _ticks = np.unique(np.round(np.linspace(1, max_count,
                                                  min(11, max_count))).astype(int))
        ax.set_xticks(_ticks)
        ax.set_xlim(0.5, max_count + 0.5)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
    for idx in range(len(methods), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(f"Competitive-node count per pair  —  Grammar Chunking  "
                 f"(rel-prob ≥ {pct_str})", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"competitive_node_hist_{fname_suffix}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Competitive-node histogram ({pct_str}) saved → {out_path}")


_methods_for_cn = [(Z_tr, lbl, color)
                    for Z_tr, _, _, _, lbl, _, color in METHODS]
_competitive_node_fig(_methods_for_cn, 0.90, "90pct")
_competitive_node_fig(_methods_for_cn, 0.75, "75pct")
_competitive_node_fig(_methods_for_cn, 0.50, "50pct")

print(f"\nAll outputs written to: {OUT_DIR}")
