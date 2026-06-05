"""
Grammar Chunking basic-level — five content-tree variants side by side.
=======================================================================

Builds one shared context tree on TEST_GRAMMAR3 tokens, then trains
*four* content trees and runs the new ``get_basic(use_root=True,
eval_alpha=EVAL_ALPHA)`` pipeline on each so the resulting basic-level
nodes can be compared head-to-head.

Variants:
  A) ``TopK-Disc-Cnt1`` (bag-of-context-concepts at fixed depth)
       Attrs 0/1 = top-K context-tree pool-node ids at depth TOPK_DEPTH
       per side. Plain ``CobwebDiscreteTree``. NOT incremental: the pool
       is a fixed-depth slice; ifit-driven structural change at depth ≤
       TOPK_DEPTH invalidates the pool numbering.

  B) ``TRELLIS`` (single leaf-pointer + ref_tree, matches src/parse_mh.py:651)
       Attrs 0/1 = single integer leaf id per side. Content tree wires
       ``ref_tree=context_tree`` + ``set_ref_attr(0/1)`` so
       ``log_prob_instance`` does LCA-similarity soft matching.
       Incremental: ``register_ref_val`` re-points ids when leaves split.

  C) ``TopK-Pool-Cache`` (incremental analog of TopK-Disc-Cnt1)
       Attrs 0/1 = set of TOP_K stable int ids — the top-scoring depth-
       ``POOL_DEPTH`` context-tree nodes under each side, scored
       directly via ``log_prob_instance``. Same emission semantics as
       variant A, but identifiers are minted from each pool node's
       ``concept_hash`` so they survive context-tree restructuring.
       When a pool node moves out of the depth band (merged above or
       split) the encoder pushes a ``{old_id → current_canonical_id}``
       entry through ``content_tree.set_value_remap`` so old
       ``av_count`` entries keep accumulating against the right
       canonical value. Encoded by ``TopKPoolEncoder``
       (``cobweb-private/src/cobweb/leaf_remap.py``).

  D) ``BFSBag-RefTree`` (BFS K-leaves directly as ref-attr values)
       Attrs 0/1 = bag of K leaf-pointer ids found by BFS, fed into a
       content tree with ``ref_tree=context_tree`` + ``set_ref_attr(0/1)``.
       Soft-matching via LCA similarity. Like TRELLIS but K-wide instead
       of single-pointer. Incremental: same as TRELLIS plus the
       BFS-vs-greedy difference.

  E) ``BFSBag-BLRemap`` (BFS K-leaves → remap to each leaf's basic-level)
       Same BFS step as C and D, but each leaf is hard-remapped to *its
       own* basic-level node via
       ``leaf.get_basic(use_root=True, eval_alpha=EVAL_ALPHA)`` cached by
       ``BasicLevelCache`` (generation-validated). Plain
       ``CobwebDiscreteTree`` (no ref_tree). Useful when the natural
       coarse category isn't a fixed depth but varies across the tree.

Outputs (under tests/basic-level/grammar_chunking_basic_level_output/):
  - topk_disc_cnt1/...    ↘ five files per variant:
  - trellis/...             basic_level_subtrees.png, content_tree_labels.png,
  - topk_pool_cache/...     per_subtree_membership.csv, method_summary.txt,
  - bfsbag_reftree/...      score_by_depth.png
  - bfsbag_blremap/...
"""

import os
import sys
import csv
import random
from collections import Counter

import numpy as np
import matplotlib
# Leave the default backend in place so the interactive α-slider can
# pop up (cf. ``corter_gluck_hierarchies.py``). To run the test
# headless (CI / no display) set ``MPLBACKEND=Agg`` in the environment
# before invocation — ``plt.savefig`` keeps working on any backend.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from util.cfg import TEST_GRAMMAR3, generate
from cobweb.cobweb_discrete import CobwebDiscreteTree
from cobweb.leaf_remap import (
    TopKPoolEncoder, BasicLevelCache, bfsbag_blremap_instance,
)

OUT_DIR = os.path.join(_HERE, "grammar_chunking_basic_level_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
N_SENTENCES   = 1000
WINDOW        = 3
TOP_K         = 5            # bag width for variant A
TOPK_DEPTH    = 4            # fixed pool depth for variant A
BFS_K         = 5            # bag width for variant D (BFSBag-RefTree) + E
BFS_MAX_NODES = 128          # BFS expansion budget (variants D, E)
POOL_DEPTH    = 4            # pool depth for variant C (TopK-Pool-Cache)
ALPHA_CONTEXT = 1e-3
ALPHA_CONTENT = 1e-3
EVAL_ALPHA    = 10.0     # for get_basic / expected_pmi (use_root=True)
SEED          = 42
TREE_DEPTH_FOR_LABEL_FIG = 3
TOP_CENTER_BIGRAMS = 6
TOP_CTX_NODES      = 5
random.seed(SEED); np.random.seed(SEED)

# ── Corpus ───────────────────────────────────────────────────────────────────
print(f"Generating {N_SENTENCES} sentences from TEST_GRAMMAR3 …")
sentences = []
for _ in range(N_SENTENCES):
    sent = [w for w in generate("S", TEST_GRAMMAR3).split() if w]
    if len(sent) >= 2:
        sentences.append(sent)

all_tokens = [w for sent in sentences for w in sent]
vocab      = sorted(set(all_tokens))
word2id    = {w: i for i, w in enumerate(vocab)}
id2word    = {i: w for w, i in word2id.items()}
V          = len(vocab)
print(f"  Sentences: {len(sentences)} | Vocab: {V} | Tokens: {len(all_tokens)}")

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
    return pos2id[word2pos[left_w]] * N_POS + pos2id[word2pos[right_w]]


def split_pair_label(lbl):
    return lbl // N_POS, lbl % N_POS


# ── Context-window encoding ──────────────────────────────────────────────────
CONTEXT_OFFSETS = [p for p in range(-WINDOW, WINDOW + 1) if p != 0]
pos2attr        = {p: i for i, p in enumerate(CONTEXT_OFFSETS)}


def make_context_instance(sent, pos):
    inst = {}
    for offset in CONTEXT_OFFSETS:
        ctx = pos + offset
        if 0 <= ctx < len(sent):
            inst[pos2attr[offset]] = {word2id[sent[ctx]]: 1.0}
    return inst


rng         = np.random.default_rng(SEED)
sent_perm   = rng.permutation(len(sentences))
split_s     = int(0.8 * len(sentences))
train_sents = [sentences[i] for i in sent_perm[:split_s]]
test_sents  = [sentences[i] for i in sent_perm[split_s:]]
print(f"  Sentence split: train={len(train_sents)} test={len(test_sents)}")

context_train_tokens = [make_context_instance(s, p)
                        for s in train_sents for p in range(len(s))]
print(f"  Context tree training tokens: {len(context_train_tokens)}")


def build_pair_set(sents):
    L, R, labels, words_L, words_R = [], [], [], [], []
    for s in sents:
        for i in range(len(s) - 1):
            L.append(make_context_instance(s, i))
            R.append(make_context_instance(s, i + 1))
            labels.append(pair_label(s[i], s[i + 1]))
            words_L.append(word2id[s[i]])
            words_R.append(word2id[s[i + 1]])
    return (L, R,
            np.array(labels,  dtype=np.int32),
            np.array(words_L, dtype=np.int32),
            np.array(words_R, dtype=np.int32))


train_L, train_R, y_train, train_wL, train_wR = build_pair_set(train_sents)
test_L,  test_R,  y_test,  test_wL,  test_wR  = build_pair_set(test_sents)
print(f"  Pairs:  train={len(train_L)}  test={len(test_L)}")


# ── Shared Context tree ──────────────────────────────────────────────────────
print(f"\nBuilding Context tree (CobwebDiscreteTree, alpha={ALPHA_CONTEXT}) …")
context_tree = CobwebDiscreteTree(alpha=ALPHA_CONTEXT, weight_attr=True)
for i, inst in enumerate(context_train_tokens):
    context_tree.ifit(inst)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(context_train_tokens)} inserted")
print("  Context tree built.")


def collect_by_depth_nodes(root):
    by_depth, queue = {}, [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        by_depth.setdefault(d, []).append(node)
        for c in node.children:
            queue.append((c, d + 1))
    return by_depth


by_depth     = collect_by_depth_nodes(context_tree.root)
depth_counts = {d: len(v) for d, v in by_depth.items()}
print(f"  Nodes per depth: {dict(sorted(depth_counts.items()))}")
topk_depth      = min(TOPK_DEPTH, max(depth_counts))
topk_pool_nodes = by_depth[topk_depth]
if topk_depth != TOPK_DEPTH:
    print(f"  WARN: requested TOPK_DEPTH={TOPK_DEPTH} clamped to {topk_depth}")
print(f"  TopK pool: depth {topk_depth} ({len(topk_pool_nodes)} nodes)  k={TOP_K}")


def greedy_descend(root, instance):
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.log_prob_instance(instance))
    return node


# ============================================================================
# Variant A — TopK-Disc-Cnt1 (bag of context concepts per side)
# ============================================================================

def encode_logpost_disc(instances, nodes):
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, inst in enumerate(instances):
            out[i, j] = node.log_prob_instance(inst)
    return out


def topk_indices(Z_raw, k):
    k = min(k, Z_raw.shape[1])
    return np.argpartition(Z_raw, -k, axis=1)[:, -k:]


print("\n[A] Encoding TopK pool log-probs …")
_pool_train_L = encode_logpost_disc(train_L, topk_pool_nodes)
_pool_train_R = encode_logpost_disc(train_R, topk_pool_nodes)
_pool_test_L  = encode_logpost_disc(test_L,  topk_pool_nodes)
_pool_test_R  = encode_logpost_disc(test_R,  topk_pool_nodes)

topk_idx_train_L = topk_indices(_pool_train_L, TOP_K)
topk_idx_train_R = topk_indices(_pool_train_R, TOP_K)
topk_idx_test_L  = topk_indices(_pool_test_L,  TOP_K)
topk_idx_test_R  = topk_indices(_pool_test_R,  TOP_K)


def topkdisc_cnt1_inst(idx_L, idx_R):
    return {0: {int(j): 1.0 for j in idx_L},
            1: {int(j): 1.0 for j in idx_R}}


pair_train_topk = [topkdisc_cnt1_inst(iL, iR)
                   for iL, iR in zip(topk_idx_train_L, topk_idx_train_R)]
pair_test_topk  = [topkdisc_cnt1_inst(iL, iR)
                   for iL, iR in zip(topk_idx_test_L,  topk_idx_test_R)]
print(f"  TopK-Disc-Cnt1 instances: train={len(pair_train_topk)}  test={len(pair_test_topk)}")

print(f"[A] Building TopK content tree (CobwebDiscreteTree, alpha={ALPHA_CONTENT}) …")
content_tree_topk = CobwebDiscreteTree(alpha=ALPHA_CONTENT, weight_attr=True)
for i, inst in enumerate(pair_train_topk):
    content_tree_topk.ifit(inst)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(pair_train_topk)} inserted")
print("[A] Content tree built.")


# ============================================================================
# Variant B — TRELLIS leaf-pointer + ref_tree (mirrors parse_mh.py 2710–2716)
# ============================================================================

print("\n[B] Walking context tree leaves and assigning label-path ids …")
ctx_leaves = []
stack = [context_tree.root]
while stack:
    n = stack.pop()
    if not n.children:
        ctx_leaves.append(n)
    else:
        stack.extend(n.children)
hash_to_leaf_id = {}
id_to_leaf      = {}
for leaf in ctx_leaves:
    h = leaf.concept_hash()
    if h not in hash_to_leaf_id:
        lid = len(hash_to_leaf_id) + 1   # reserve 0 as "missing"
        hash_to_leaf_id[h] = lid
        id_to_leaf[lid]    = leaf
print(f"  Context leaves: {len(ctx_leaves)} ({len(hash_to_leaf_id)} unique by concept_hash)")


def trellis_inst(inst_L, inst_R):
    leaf_L = greedy_descend(context_tree.root, inst_L)
    leaf_R = greedy_descend(context_tree.root, inst_R)
    l_id = hash_to_leaf_id[leaf_L.concept_hash()]
    r_id = hash_to_leaf_id[leaf_R.concept_hash()]
    return {0: {l_id: 1.0}, 1: {r_id: 1.0}}


print("[B] Building TRELLIS content instances (single leaf-pointer per side) …")
pair_train_trellis = [trellis_inst(L, R) for L, R in zip(train_L, train_R)]
pair_test_trellis  = [trellis_inst(L, R) for L, R in zip(test_L,  test_R)]

print(f"[B] Building TRELLIS content tree "
      f"(ref_tree=context_tree, ref_attrs=[0,1], alpha={ALPHA_CONTENT}) …")
content_tree_trellis = CobwebDiscreteTree(
    alpha=ALPHA_CONTENT,
    weight_attr=False,            # matches TRELLIS (parse_mh.py:2713)
    ref_tree=context_tree,
)
content_tree_trellis.set_ref_attr(0)
content_tree_trellis.set_ref_attr(1)
# Register every leaf id ↔ context-tree node before fitting so LCA-similarity
# soft matching is available from the first ifit.
for lid, leaf in id_to_leaf.items():
    content_tree_trellis.register_ref_val(lid, leaf)
for i, inst in enumerate(pair_train_trellis):
    content_tree_trellis.ifit(inst)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(pair_train_trellis)} inserted")
print("[B] TRELLIS content tree built.")


# ============================================================================
# Variant C — TopK-Pool-Cache:
#   Same emission as TopK-Disc-Cnt1 (set of K depth-d node ids scored
#   directly via log_prob_instance) but with **stable** concept_hash-
#   derived ids. ``TopKPoolEncoder`` caches the pool list by
#   ``context_tree.structure_generation`` and pushes a
#   ``{old_id → current_canonical_id}`` entry through
#   ``set_value_remap`` whenever a pool node moves out of its depth
#   band (merge above, split above, or split-at-depth removing the
#   node). This is the incremental counterpart of variant A.
# ============================================================================

print(f"\n[C] Setting up TopKPoolEncoder (POOL_DEPTH={POOL_DEPTH}, K={TOP_K}) …")
encoder_C = TopKPoolEncoder(
    context_tree=context_tree,
    depth=POOL_DEPTH,
    k=TOP_K,
)
print(f"  Pool size: {encoder_C.pool_size} depth-{POOL_DEPTH} nodes")

print(f"[C] Building TopK-Pool-Cache content tree "
      f"(plain CobwebDiscreteTree, alpha={ALPHA_CONTENT}) …")
content_tree_pool = CobwebDiscreteTree(alpha=ALPHA_CONTENT, weight_attr=True)


def topk_pool_inst(inst_L, inst_R):
    # Defer set_value_remap push during the encoding loop; sync once
    # after both train and test bags are built.
    return {0: encoder_C.bag_for(inst_L, None),
            1: encoder_C.bag_for(inst_R, None)}


print("[C] Building TopK-Pool-Cache content instances …")
pair_train_pool = [topk_pool_inst(L, R) for L, R in zip(train_L, train_R)]
pair_test_pool  = [topk_pool_inst(L, R) for L, R in zip(test_L,  test_R)]
encoder_C.sync_remap(content_tree_pool)
print(f"  Encoder value_vocab: {len(encoder_C.value_vocab)} pool ids; "
      f"remap_dict entries (non-identity): {len(encoder_C.value_remap_dict)}")
print(f"  Last refresh stats: {encoder_C.last_refresh_stats}  "
      f"(moved=pool nodes pushed/pulled to a different depth; "
      f"rescued=deleted nodes recovered via best-leaf-under-parent; "
      f"orphaned=cascading deletions left identity-mapped)")

for i, inst in enumerate(pair_train_pool):
    content_tree_pool.ifit(inst)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(pair_train_pool)} inserted")
print(f"[C] TopK-Pool-Cache content tree built. "
      f"use_value_remap={content_tree_pool.use_value_remap}")


# ============================================================================
# Variant D — BFSBag-RefTree: BFS K leaves used directly as ref-attr values.
# ============================================================================

# Per-variant value vocab: leaf concept_hash → int id, plus the actual
# context-tree leaf node so we can register_ref_val before any ifit.
leaf_hash_to_id_D: dict = {}
leaf_id_to_node_D: dict = {}

def _leaf_id_D(h: str, leaf_node) -> int:
    lid = leaf_hash_to_id_D.get(h)
    if lid is None:
        lid = len(leaf_hash_to_id_D) + 1
        leaf_hash_to_id_D[h] = lid
        leaf_id_to_node_D[lid] = leaf_node
    return lid


def bfsbag_reftree_inst(inst_L, inst_R):
    bag = {0: {}, 1: {}}
    for side_idx, side_inst in ((0, inst_L), (1, inst_R)):
        for leaf, _score in context_tree.bfs_top_k_leaves(
                side_inst, BFS_K, BFS_MAX_NODES):
            lid = _leaf_id_D(leaf.concept_hash(), leaf)
            bag[side_idx][lid] = bag[side_idx].get(lid, 0.0) + 1.0
    return bag


print("\n[D] Building BFSBag-RefTree content instances …")
pair_train_reftree = [bfsbag_reftree_inst(L, R) for L, R in zip(train_L, train_R)]
pair_test_reftree  = [bfsbag_reftree_inst(L, R) for L, R in zip(test_L,  test_R)]
print(f"  Distinct leaves seen via BFS: {len(leaf_hash_to_id_D)}")

print(f"[D] Building BFSBag-RefTree content tree "
      f"(ref_tree=context_tree, ref_attrs=[0,1], alpha={ALPHA_CONTENT}) …")
content_tree_reftree = CobwebDiscreteTree(
    alpha=ALPHA_CONTENT,
    weight_attr=False,
    ref_tree=context_tree,
)
content_tree_reftree.set_ref_attr(0)
content_tree_reftree.set_ref_attr(1)
for lid, leaf in leaf_id_to_node_D.items():
    content_tree_reftree.register_ref_val(lid, leaf)
for i, inst in enumerate(pair_train_reftree):
    content_tree_reftree.ifit(inst)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(pair_train_reftree)} inserted")
print("[D] BFSBag-RefTree content tree built.")


# ============================================================================
# Variant E — BFSBag-BLRemap: BFS K leaves → remap each to its basic-level
# node (via leaf.get_basic(use_root=True, eval_alpha=EVAL_ALPHA), cached).
# Bag of basic-level concept_hashes (multiset). No ref_tree.
# ============================================================================

print(f"\n[E] Setting up BasicLevelCache (eval_alpha={EVAL_ALPHA}) …")
bl_cache_E = BasicLevelCache(context_tree, eval_alpha=EVAL_ALPHA)

# Per-variant value vocab: basic-level concept_hash → int id (≥1).
bl_hash_to_id_E: dict = {}
def _bl_id(h: str) -> int:
    lid = bl_hash_to_id_E.get(h)
    if lid is None:
        lid = len(bl_hash_to_id_E) + 1
        bl_hash_to_id_E[h] = lid
    return lid


def bfsbag_blremap_inst(inst_L, inst_R):
    bag_L_h = bfsbag_blremap_instance(context_tree, inst_L,
                                      BFS_K, BFS_MAX_NODES, bl_cache_E)
    bag_R_h = bfsbag_blremap_instance(context_tree, inst_R,
                                      BFS_K, BFS_MAX_NODES, bl_cache_E)
    return {0: {_bl_id(h): float(c) for h, c in bag_L_h.items()},
            1: {_bl_id(h): float(c) for h, c in bag_R_h.items()}}


print("[E] Building BFSBag-BLRemap content instances "
      "(this calls leaf.get_basic on every encountered leaf — slowest setup) …")
pair_train_blremap = [bfsbag_blremap_inst(L, R) for L, R in zip(train_L, train_R)]
pair_test_blremap  = [bfsbag_blremap_inst(L, R) for L, R in zip(test_L,  test_R)]
print(f"  Basic-level target vocab: {len(bl_hash_to_id_E)} distinct BL nodes")
print(f"  BL cache size: {len(bl_cache_E)}  "
      f"(gen={context_tree.structure_generation})")

print(f"[E] Building BFSBag-BLRemap content tree "
      f"(plain CobwebDiscreteTree, alpha={ALPHA_CONTENT}) …")
content_tree_blremap = CobwebDiscreteTree(alpha=ALPHA_CONTENT, weight_attr=True)
for i, inst in enumerate(pair_train_blremap):
    content_tree_blremap.ifit(inst)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(pair_train_blremap)} inserted")
print("[E] BFSBag-BLRemap content tree built.")


# ============================================================================
# BL pipeline + viz (shared between the five variants)
# ============================================================================

CMAP_POS    = plt.get_cmap("tab10") if N_POS <= 10 else plt.get_cmap("tab20")
pos_colors  = [CMAP_POS(i / max(N_POS - 1, 1)) for i in range(N_POS)]


def top_context_attrs(node, k=TOP_CTX_NODES):
    out = {0: [], 1: []}
    for attr in (0, 1):
        if attr not in node.av_count:
            continue
        items = sorted(node.av_count[attr].items(), key=lambda kv: -kv[1])[:k]
        total = sum(node.av_count[attr].values()) or 1
        out[attr] = [(int(val_id), c / total) for val_id, c in items]
    return out


def plot_subtrees(members, title, out_path, attr_label_fn):
    sorted_bls = sorted(members.values(),
                        key=lambda m: len(m["indices"]), reverse=True)
    n_rows = len(sorted_bls)
    if n_rows == 0:
        return
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(16, max(2.4, n_rows * 1.9)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.4, 1.6, 2.2]},
    )
    fig.suptitle(title, fontsize=11)

    for row, m in enumerate(sorted_bls):
        node    = m["node"]
        labels  = np.array(m["labels"])
        wL_arr  = np.array(m["wL"])
        wR_arr  = np.array(m["wR"])
        n_mem   = len(m["indices"])
        depth   = m["depth"]

        joint = np.zeros((N_POS, N_POS), dtype=np.int32)
        for lbl in labels:
            lp, rp = split_pair_label(int(lbl))
            joint[lp, rp] += 1
        flat_top = joint.flatten().argmax()
        dom_l, dom_r = flat_top // N_POS, flat_top % N_POS
        dom_pair = f"{id2pos[dom_l]}-{id2pos[dom_r]}"

        ax0 = axes[row, 0]
        ax0.imshow(joint / max(joint.sum(), 1), cmap="Blues",
                   vmin=0, vmax=1, aspect="equal")
        ax0.set_xticks(range(N_POS))
        ax0.set_yticks(range(N_POS))
        ax0.set_xticklabels([id2pos[i] for i in range(N_POS)],
                            rotation=45, ha="right", fontsize=6)
        ax0.set_yticklabels([id2pos[i] for i in range(N_POS)], fontsize=6)
        ax0.set_xlabel("R POS", fontsize=6)
        ax0.set_ylabel(
            f"d={depth}\nn={n_mem}\ncnt={int(node.count)}\ndom={dom_pair}\n\nL POS",
            fontsize=6, rotation=0, labelpad=42, va="center",
        )
        if row == 0:
            ax0.set_title("(L,R) POS joint", fontsize=8)

        ax1 = axes[row, 1]
        bigram_counts = Counter(zip(wL_arr.tolist(), wR_arr.tolist()))
        top_big = bigram_counts.most_common(TOP_CENTER_BIGRAMS)
        if top_big:
            labels_str = [f"{id2word[wl]} {id2word[wr]}" for (wl, wr), _ in top_big]
            counts_    = [c for _, c in top_big]
            colors_    = [pos_colors[pos2id[word2pos[id2word[wl]]]]
                          for (wl, _), _ in top_big]
            ax1.barh(np.arange(len(labels_str))[::-1], counts_,
                     color=colors_, edgecolor="white", linewidth=0.4)
            ax1.set_yticks(np.arange(len(labels_str))[::-1])
            ax1.set_yticklabels(labels_str, fontsize=6)
            ax1.tick_params(axis="x", labelsize=5)
        if row == 0:
            ax1.set_title("top center bigrams", fontsize=8)

        ax2 = axes[row, 2]
        ax2.axis("off")
        ctx_top = top_context_attrs(node, k=TOP_CTX_NODES)
        for ci, side in enumerate(("L", "R")):
            cx = (ci + 0.5) / 2.0
            ax2.text(cx, 0.95, side, ha="center", va="top",
                     fontsize=7, fontweight="bold", transform=ax2.transAxes)
            for li, (val_id, frac) in enumerate(ctx_top[ci]):
                cy = 0.85 - li * 0.16
                ax2.text(cx, cy, f"{attr_label_fn(val_id)} ({frac:.2f})",
                         ha="center", va="top", fontsize=6,
                         color="black", transform=ax2.transAxes)
        if row == 0:
            ax2.set_title(f"top per-side attr values (k={TOP_CTX_NODES})",
                          fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def make_layout(root, max_depth):
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


def compute_pair_label_counts_idx(all_nodes, children_of,
                                  train_data, y_tr, max_depth):
    cL, cR = {}, {}
    for x, lbl in zip(train_data, y_tr):
        l_pos, r_pos = split_pair_label(int(lbl))
        for idx in descend_layout_disc(all_nodes, children_of, x, max_depth):
            if idx not in cL:
                cL[idx] = np.zeros(N_POS, dtype=np.int32)
                cR[idx] = np.zeros(N_POS, dtype=np.int32)
            cL[idx][l_pos] += 1
            cR[idx][r_pos] += 1
    return cL, cR


def plot_pair_tree_idx(children_of, depth_of, counts_L, counts_R,
                        title, out_path, max_depth, highlight_idx=None):
    highlight_idx = highlight_idx or set()
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
    ax.invert_yaxis(); ax.axis("off")
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

    def _draw_bar(x_left, y_top, props, label_text, is_bl):
        cur = x_left
        for p_idx in range(N_POS):
            seg_w = props[p_idx] * bar_w
            if seg_w > 0:
                ax.add_patch(plt.Rectangle((cur, y_top), seg_w, bar_h,
                                            color=pos_colors[p_idx], lw=0))
                cur += seg_w
        ax.add_patch(plt.Rectangle(
            (x_left, y_top), bar_w, bar_h, fill=False,
            edgecolor=("red" if is_bl else "black"),
            lw=(3.0 if is_bl else 0.5),
            zorder=(5 if is_bl else 2),
        ))
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
        is_bl  = idx in highlight_idx
        x_c, _ = pos[idx]
        x_left = x_c - bar_w / 2
        y_top_L = depth * y_unit - bar_h - gap / 2
        y_top_R = depth * y_unit + gap / 2
        _draw_bar(x_left, y_top_L, propsL, "L", is_bl)
        _draw_bar(x_left, y_top_R, propsR, "R", is_bl)
        ax.text(x_c, depth * y_unit - bar_h - gap / 2 - 0.04,
                f"n={int(tot)}", ha="center", va="bottom", fontsize=5)
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]:
                draw_node(c, depth + 1)
    draw_node(0, 0)

    legend_h = [plt.Rectangle((0, 0), 1, 1, color=pos_colors[i], label=id2pos[i])
                for i in range(N_POS)]
    ax.legend(handles=legend_h,
              title="POS (top bar=L, bottom=R; red border=BL)",
              loc="lower right", ncol=max(1, N_POS // 4),
              fontsize=6, title_fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_bl_pipeline(content_tree, pair_train, pair_test,
                     y_test_arr, test_wL_arr, test_wR_arr,
                     out_subdir, variant_label, attr_label_fn):
    os.makedirs(out_subdir, exist_ok=True)
    print(f"\n=== Running BL pipeline for variant: {variant_label} ===")

    # Per-leaf get_basic with use_root=True
    _cache = {}
    def get_bl(leaf):
        key = id(leaf)
        if key in _cache:
            return _cache[key]
        bl = leaf.get_basic(0, 0, debug=False,
                            eval_alpha=EVAL_ALPHA, use_root=True)
        _cache[key] = bl
        return bl

    bl_members = {}
    for i, inst in enumerate(pair_test):
        leaf = greedy_descend(content_tree.root, inst)
        bl   = get_bl(leaf)
        if bl is None:
            continue
        nid = id(bl)
        if nid not in bl_members:
            bl_members[nid] = {
                "node":     bl, "depth": bl.depth(),
                "indices":  [], "wL": [], "wR": [], "labels": [],
            }
        bl_members[nid]["indices"].append(i)
        bl_members[nid]["wL"].append(int(test_wL_arr[i]))
        bl_members[nid]["wR"].append(int(test_wR_arr[i]))
        bl_members[nid]["labels"].append(int(y_test_arr[i]))

    print(f"  {len(bl_members)} unique BL nodes covering {len(pair_test)} test pairs")

    # 1) per-BL subtree view
    plot_subtrees(
        bl_members,
        title=(f"{variant_label} — Basic-level subtrees, "
               f"get_basic(use_root=True, eval_alpha={EVAL_ALPHA})  "
               f"(N_test={len(pair_test)}, n_subtrees={len(bl_members)})"),
        out_path=os.path.join(out_subdir, "basic_level_subtrees.png"),
        attr_label_fn=attr_label_fn,
    )
    print(f"  Subtrees → {os.path.join(out_subdir, 'basic_level_subtrees.png')}")

    # 2) CSV
    csv_path = os.path.join(out_subdir, "per_subtree_membership.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtree_idx", "depth", "node_count", "test_members",
                    "dominant_pair", "top_pair_dist"])
        for i, m in enumerate(sorted(bl_members.values(),
                                      key=lambda m: len(m["indices"]),
                                      reverse=True)):
            labels = np.array(m["labels"])
            joint  = np.bincount(labels, minlength=N_POS * N_POS)
            top5 = np.argsort(-joint)[:5]
            dl, dr = split_pair_label(int(top5[0]))
            dist_str = "/".join(
                f"{id2pos[split_pair_label(int(p))[0]]}-"
                f"{id2pos[split_pair_label(int(p))[1]]}:{int(joint[p])}"
                for p in top5 if joint[p] > 0
            )
            w.writerow([i, m["depth"], int(m["node"].count),
                        len(m["indices"]),
                        f"{id2pos[dl]}-{id2pos[dr]}", dist_str])
    print(f"  CSV → {csv_path}")

    # 3) content tree visualisation
    layout_nodes, layout_children, layout_depth = make_layout(
        content_tree.root, max_depth=TREE_DEPTH_FOR_LABEL_FIG,
    )
    cL_map, cR_map = compute_pair_label_counts_idx(
        layout_nodes, layout_children, pair_train, y_train,
        max_depth=TREE_DEPTH_FOR_LABEL_FIG,
    )
    bl_node_hashes = {m["node"].concept_hash() for m in bl_members.values()}
    highlight_idx = {idx for idx, n in enumerate(layout_nodes)
                     if n.concept_hash() in bl_node_hashes}
    tree_fig_path = os.path.join(out_subdir, "content_tree_labels.png")
    plot_pair_tree_idx(
        layout_children, layout_depth, cL_map, cR_map,
        title=(f"{variant_label} content tree — Pair POS distributions  "
               f"(red border = BL, eval_alpha={EVAL_ALPHA})"),
        out_path=tree_fig_path,
        max_depth=TREE_DEPTH_FOR_LABEL_FIG,
        highlight_idx=highlight_idx,
    )
    print(f"  Tree fig → {tree_fig_path}")

    # 4) score by depth
    all_nodes_full = []
    stack = [content_tree.root]
    while stack:
        n = stack.pop()
        all_nodes_full.append(n)
        stack.extend(n.children)
    depth_to_scores = {}
    for n in all_nodes_full:
        d = n.depth()
        score = n.expected_pmi(0, 0, eval_alpha=EVAL_ALPHA,
                               uniform_leaf=False, use_root=True)
        depth_to_scores.setdefault(d, []).append(score)
    depth_to_n_bl = {}
    for m in bl_members.values():
        d = m["depth"]
        depth_to_n_bl[d] = depth_to_n_bl.get(d, 0) + 1

    depths_sorted = sorted(depth_to_scores.keys())
    means = [np.mean(depth_to_scores[d]) for d in depths_sorted]
    mins  = [np.min(depth_to_scores[d])  for d in depths_sorted]
    maxs  = [np.max(depth_to_scores[d])  for d in depths_sorted]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(depths_sorted, mins, maxs, alpha=0.15, color="#1f77b4",
                    label="min–max range")
    ax.plot(depths_sorted, means, marker="o", linewidth=2, color="#1f77b4",
            label="mean expected_pmi", zorder=3)
    for d, m in zip(depths_sorted, means):
        ax.annotate(f"{m:.3f}", (d, m),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, ha="center", color="#1f77b4")
    for d, n in depth_to_n_bl.items():
        ax.axvline(d, color="red", alpha=0.25, linestyle="--", linewidth=1.2,
                   zorder=1)
        ax.text(d, ax.get_ylim()[1], f"BL × {n}",
                color="red", fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Tree depth (root = 0)", fontsize=11)
    ax.set_ylabel(f"Mean expected_pmi (use_root=True, eval_alpha={EVAL_ALPHA})",
                  fontsize=11)
    ax.set_title(f"{variant_label} — mean empirical PMI vs root by depth  "
                 "(red dashed = depth contains a BL)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.set_xticks(depths_sorted)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    depth_plot_path = os.path.join(out_subdir, "score_by_depth.png")
    plt.savefig(depth_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Score-by-depth → {depth_plot_path}")

    # 5) summary
    summary_path = os.path.join(out_subdir, "method_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 64 + "\n")
        f.write(f" {variant_label}\n")
        f.write(f" Method: leaf.get_basic(use_root=True, eval_alpha={EVAL_ALPHA})\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Train pairs: {len(pair_train)}\n")
        f.write(f"Test  pairs: {len(pair_test)}\n")
        f.write(f"Content tree nodes: {len(all_nodes_full)}\n\n")
        f.write(f"{len(bl_members)} unique basic-level nodes:\n")
        for i, m in enumerate(sorted(bl_members.values(),
                                      key=lambda m: len(m["indices"]),
                                      reverse=True)):
            labels = np.array(m["labels"])
            joint  = np.bincount(labels, minlength=N_POS * N_POS)
            top1   = int(joint.argmax())
            dl, dr = split_pair_label(top1)
            f.write(f"  [{i:>2}] depth={m['depth']:>2}  "
                    f"count={int(m['node'].count):>6}  "
                    f"members={len(m['indices']):>4}  "
                    f"dom={id2pos[dl]}-{id2pos[dr]}\n")
    print(f"  Summary → {summary_path}")

    # Return ``all_nodes_full`` itself (not just its length) so the
    # cross-variant interactive α-slider at the bottom of the script
    # can re-call ``expected_pmi`` per node at the slider's current α.
    return bl_members, all_nodes_full


# ── Run both variants ────────────────────────────────────────────────────────

bl_topk, nodes_topk = run_bl_pipeline(
    content_tree_topk,  pair_train_topk,  pair_test_topk,
    y_test, test_wL, test_wR,
    out_subdir=os.path.join(OUT_DIR, "topk_disc_cnt1"),
    variant_label="TopK-Disc-Cnt1",
    attr_label_fn=lambda v: f"ctx#{v}",
)

bl_trellis, nodes_trellis = run_bl_pipeline(
    content_tree_trellis, pair_train_trellis, pair_test_trellis,
    y_test, test_wL, test_wR,
    out_subdir=os.path.join(OUT_DIR, "trellis"),
    variant_label="TRELLIS (leaf-pointer + ref_tree)",
    attr_label_fn=lambda v: f"leaf#{v}",
)

bl_pool, nodes_pool = run_bl_pipeline(
    content_tree_pool, pair_train_pool, pair_test_pool,
    y_test, test_wL, test_wR,
    out_subdir=os.path.join(OUT_DIR, "topk_pool_cache"),
    variant_label=f"TopK-Pool-Cache (K={TOP_K}, depth={POOL_DEPTH})",
    attr_label_fn=lambda v: f"pool#{v}",
)

bl_reftree, nodes_reftree = run_bl_pipeline(
    content_tree_reftree, pair_train_reftree, pair_test_reftree,
    y_test, test_wL, test_wR,
    out_subdir=os.path.join(OUT_DIR, "bfsbag_reftree"),
    variant_label=f"BFSBag-RefTree (K={BFS_K}, ref_tree=context)",
    attr_label_fn=lambda v: f"leaf#{v}",
)

bl_blremap, nodes_blremap = run_bl_pipeline(
    content_tree_blremap, pair_train_blremap, pair_test_blremap,
    y_test, test_wL, test_wR,
    out_subdir=os.path.join(OUT_DIR, "bfsbag_blremap"),
    variant_label=f"BFSBag-BLRemap (K={BFS_K}, target=basic-level)",
    attr_label_fn=lambda v: f"bl#{v}",
)


# ── Cross-variant comparison line ────────────────────────────────────────────
print("\n" + "=" * 64)
print(" Cross-variant comparison")
print("=" * 64)
print(f"  TopK-Disc-Cnt1   : {len(bl_topk):>4} BLs  /  {len(nodes_topk):>6} tree nodes")
print(f"  TRELLIS          : {len(bl_trellis):>4} BLs  /  {len(nodes_trellis):>6} tree nodes")
print(f"  TopK-Pool-Cache  : {len(bl_pool):>4} BLs  /  {len(nodes_pool):>6} tree nodes")
print(f"  BFSBag-RefTree   : {len(bl_reftree):>4} BLs  /  {len(nodes_reftree):>6} tree nodes")
print(f"  BFSBag-BLRemap   : {len(bl_blremap):>4} BLs  /  {len(nodes_blremap):>6} tree nodes")
print(f"\nOutputs in {OUT_DIR}/"
      "{topk_disc_cnt1, trellis, topk_pool_cache, bfsbag_reftree, bfsbag_blremap}/")


# ── Cross-variant interactive α-slider ──────────────────────────────────────
# Mirrors corter_gluck_hierarchies.py: overlay every variant's
# score-by-depth curve on the same axes and sweep
# ``log_10(eval_alpha)`` to watch all five curves move together.
# The per-variant static ``score_by_depth.png`` saved inside each
# variant's subfolder uses the fixed ``EVAL_ALPHA``; this slider is
# the exploratory companion.
#
# BL vertical markers are intentionally **not** drawn here because each
# variant has its own BL set at different depths — five overlaid
# vertical-line clouds would render the panel unreadable. The slider
# is about comparing the depth-score *curves* across variants.

print("\nOpening interactive α-slider (cross-variant overlay)…")

_variant_specs = [
    ("TopK-Disc-Cnt1",                                 nodes_topk,    "#1f77b4"),
    ("TRELLIS (leaf-pointer + ref_tree)",              nodes_trellis, "#ff7f0e"),
    (f"TopK-Pool-Cache (K={TOP_K}, depth={POOL_DEPTH})",
                                                       nodes_pool,    "#2ca02c"),
    (f"BFSBag-RefTree (K={BFS_K})",                    nodes_reftree, "#d62728"),
    (f"BFSBag-BLRemap (K={BFS_K})",                    nodes_blremap, "#9467bd"),
]

fig_sl = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.32)
ax_plot   = fig_sl.add_subplot(gs[0])
ax_slider = fig_sl.add_subplot(gs[1])

slider = Slider(
    ax=ax_slider,
    label="log₁₀(eval_alpha)",
    valmin=-3, valmax=5,
    valinit=float(np.log10(EVAL_ALPHA)),
    valstep=0.05,
)


def _redraw_variants(eval_a):
    ax_plot.clear()
    for label, nodes, color in _variant_specs:
        d2s: dict = {}
        for n in nodes:
            d = n.depth()
            score = n.expected_pmi(0, 0, eval_alpha=eval_a,
                                   uniform_leaf=False, use_root=True)
            d2s.setdefault(d, []).append(score)
        if not d2s:
            continue
        depths_sorted = sorted(d2s.keys())
        means = [float(np.mean(d2s[d])) for d in depths_sorted]
        ax_plot.plot(depths_sorted, means, marker="o", linewidth=2,
                      color=color, label=label, zorder=3)
        for d, m in zip(depths_sorted, means):
            ax_plot.annotate(f"{m:.2f}", (d, m),
                              textcoords="offset points", xytext=(0, 6),
                              fontsize=6, ha="center", color=color)
    ax_plot.set_xlabel("Tree depth (root = 0)", fontsize=11)
    ax_plot.set_ylabel(f"Mean expected_pmi  (eval_alpha = {eval_a:.4g})",
                        fontsize=11)
    ax_plot.set_title(
        "Interactive α sweep — score_by_depth across variants",
        fontsize=12)
    ax_plot.axhline(0, color="black", linewidth=0.8, linestyle=":",
                     alpha=0.4)
    ax_plot.grid(axis="y", alpha=0.25)
    ax_plot.legend(loc="best", fontsize=8)
    fig_sl.canvas.draw_idle()


slider.on_changed(lambda _val: _redraw_variants(10 ** slider.val))
_redraw_variants(10 ** slider.valinit)
plt.show()
