"""
Grammar Representation Learning  (TEST_GRAMMAR3 × 1000 sentences)

For each word token in a CFG-generated corpus we build a context-window
instance (neighbouring words at offsets ±1, ±2), then train a Cobweb
Discrete tree on those instances.  The resulting node-log-prob features
are evaluated as word-type representations via linear probing and KNN —
mirroring the structure of mnist_example.py but without any autoencoder
variants.

Cobweb variants:
  Cobweb-BFS       — first DZ BFS-order nodes
  Cobweb-Depth     — all nodes at the deepest level with < DZ nodes
  Cobweb-TopK      — pool at first level with >= DZ nodes, per-sample top-k
  Cobweb-Depth-TopK — BFS nodes + per-sample top-k sparsification
  Cobweb-Path      — path-information encoding (top N_PATHS leaf paths)
"""

import os
import sys
import csv
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP
from sklearn.manifold import TSNE

# Make src/ importable both via pytest (pythonpath=src in pytest.ini)
# and when the file is run directly.
_HERE     = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_HERE, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from util.cfg import TEST_GRAMMAR3, generate

from cobweb.cobweb_discrete import CobwebDiscreteTree

HERE    = _HERE
OUT_DIR = os.path.join(HERE, "grammar_output")
ARR_DIR = os.path.join(OUT_DIR, "arrays")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARR_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_SENTENCES = 1000
WINDOW      = 3      # context half-window  (offsets ±1, ±2, excluding self)
DZ          = 128     # target BFS / depth node count
TOP_K       = 10      # per-instance top-K sparsification
PATH_DEPTH  = 6      # max depth for path-info prefix
N_PATHS     = 4      # top-N leaf paths to trace per instance
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)

# ── Generate corpus ───────────────────────────────────────────────────────────
print(f"Generating {N_SENTENCES} sentences from TEST_GRAMMAR3 …")
sentences = []
for _ in range(N_SENTENCES):
    sent = [w for w in generate("S", TEST_GRAMMAR3).split() if w]
    if sent:
        sentences.append(sent)

all_tokens = [w for sent in sentences for w in sent]
vocab      = sorted(set(all_tokens))
word2id    = {w: i for i, w in enumerate(vocab)}
id2word    = {i: w for w, i in word2id.items()}
V          = len(vocab)
print(f"  Vocab size: {V}  |  Total tokens: {len(all_tokens)}")
print(f"  Vocab: {vocab}")

# Build word → POS mapping from grammar terminal rules
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
print(f"  POS tags ({len(pos_tags)}): {pos_tags}")
print(f"  word→POS: { {w: word2pos[w] for w in vocab} }")

# ── Build context instances ───────────────────────────────────────────────────
# INSTANCE_TYPE = {int attr_id: {int val_id: float}} where
#   attr_id = position offset index  (0 → offset -2, 1 → -1, 2 → +1, 3 → +2)
#   val_id  = word2id of the word at that context position
CONTEXT_OFFSETS = [p for p in range(-WINDOW, WINDOW + 1) if p != 0]
pos2attr        = {p: i for i, p in enumerate(CONTEXT_OFFSETS)}


def make_context_instance(sentence, pos):
    """Return a cobweb INSTANCE_TYPE dict for word at `pos` in `sentence`."""
    instance = {}
    for offset in CONTEXT_OFFSETS:
        ctx = pos + offset
        if 0 <= ctx < len(sentence):
            attr = pos2attr[offset]
            val  = word2id[sentence[ctx]]
            instance[attr] = {val: 1.0}
    return instance


instances_raw = []
labels_all    = []

for sent in sentences:
    for pos, word in enumerate(sent):
        instances_raw.append(make_context_instance(sent, pos))
        labels_all.append(pos2id[word2pos[word]])

labels_all = np.array(labels_all, dtype=np.int32)
print(f"  Total instances (tokens): {len(instances_raw)}")

# 80 / 20 train-test split
rng       = np.random.default_rng(SEED)
idx       = rng.permutation(len(instances_raw))
split     = int(0.8 * len(instances_raw))
train_idx = idx[:split]
test_idx  = idx[split:]

instances_train = [instances_raw[i] for i in train_idx]
instances_test  = [instances_raw[i] for i in test_idx]
y               = labels_all[train_idx]
y_test          = labels_all[test_idx]
print(f"  Train: {len(instances_train)}  |  Test: {len(instances_test)}")

np.save(os.path.join(ARR_DIR, "y_train.npy"), y)
np.save(os.path.join(ARR_DIR, "y_test.npy"),  y_test)

# ── Build Cobweb Discrete Tree ────────────────────────────────────────────────
print("Building Cobweb Discrete tree …")
cobweb_tree = CobwebDiscreteTree(
    alpha=1e-3, 
    weight_attr=True
)
for i, inst in enumerate(instances_train):
    cobweb_tree.ifit(inst)
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(instances_train)} inserted")
print("  Tree built.")

# ── Node collection helpers ───────────────────────────────────────────────────
def collect_by_depth_nodes(root):
    """Return {depth: [node objects]} for every node in the tree."""
    by_depth = {}
    queue    = [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        by_depth.setdefault(d, []).append(node)
        for child in node.children:
            queue.append((child, d + 1))
    return by_depth


def bfs_first_n_nodes(root, n):
    """Collect the first n non-root nodes via BFS."""
    nodes, queue = [], [root]
    while queue and len(nodes) < n:
        node = queue.pop(0)
        for child in node.children:
            if len(nodes) >= n:
                break
            nodes.append(child)
            queue.append(child)
    return nodes


print("Extracting BFS nodes …")
bfs_nodes = bfs_first_n_nodes(cobweb_tree.root, DZ)
if not bfs_nodes:
    bfs_nodes = [cobweb_tree.root]   # degenerate tree
print(f"  {len(bfs_nodes)} BFS nodes collected")

print("Extracting depth nodes …")
by_depth_nodes = collect_by_depth_nodes(cobweb_tree.root)
depth_counts   = {d: len(v) for d, v in by_depth_nodes.items()}
print(f"  Nodes per depth: {dict(sorted(depth_counts.items()))}")

best_depth = 0
for d in sorted(depth_counts.keys()):
    if depth_counts[d] >= DZ:
        break
    best_depth = d
depth_nodes = by_depth_nodes[best_depth]
n_depth     = len(depth_nodes)
print(f"  Using depth {best_depth} ({n_depth} nodes, target < {DZ})")

depths_with_enough = [d for d in sorted(depth_counts.keys()) if depth_counts[d] >= DZ]
if depths_with_enough:
    topk_depth      = depths_with_enough[0]
    topk_pool_nodes = by_depth_nodes[topk_depth]
else:
    topk_depth      = max(depth_counts.keys())
    topk_pool_nodes = by_depth_nodes[topk_depth]
n_topk_pool = len(topk_pool_nodes)
print(f"  Top-K pool: depth {topk_depth} ({n_topk_pool} nodes)")

# ── Encoding: log P(instance | node) ─────────────────────────────────────────
def encode_logpost(instances, nodes):
    """Return (n_samples, n_nodes) matrix of log_prob_instance per node."""
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, inst in enumerate(instances):
            out[i, j] = node.log_prob_instance(inst)
    return out


print("Encoding train set (BFS) …")
_scaler_bfs    = StandardScaler()
Z_cob_bfs      = _scaler_bfs.fit_transform(encode_logpost(instances_train, bfs_nodes))
print("Encoding test set (BFS) …")
Z_cob_bfs_test = _scaler_bfs.transform(encode_logpost(instances_test, bfs_nodes))

print("Encoding train set (Depth) …")
_scaler_dep    = StandardScaler()
Z_cob_dep      = _scaler_dep.fit_transform(encode_logpost(instances_train, depth_nodes))
print("Encoding test set (Depth) …")
Z_cob_dep_test = _scaler_dep.transform(encode_logpost(instances_test, depth_nodes))


def topk_sparsify(Z, k):
    """Zero out all but the k largest values per row."""
    out = np.zeros_like(Z)
    k   = min(k, Z.shape[1])
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    rows    = np.arange(Z.shape[0])[:, None]
    out[rows, top_idx] = Z[rows, top_idx]
    return out


def topk_binarize(Z, k):
    """Set the k largest values per row to 1.0, rest to 0.0."""
    out = np.zeros_like(Z)
    k   = min(k, Z.shape[1])
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    rows    = np.arange(Z.shape[0])[:, None]
    out[rows, top_idx] = 1.0
    return out


print(f"  Top-K pool size: {n_topk_pool} nodes at depth {topk_depth}")
print("Encoding Top-K pool (train) …")
_scaler_topk    = StandardScaler()
Z_topk_pool     = _scaler_topk.fit_transform(encode_logpost(instances_train, topk_pool_nodes))
Z_cob_topk      = topk_sparsify(Z_topk_pool, TOP_K)
print("Encoding Top-K pool (test) …")
Z_topk_pool_test = _scaler_topk.transform(encode_logpost(instances_test, topk_pool_nodes))
Z_cob_topk_test  = topk_sparsify(Z_topk_pool_test, TOP_K)

Z_cob_bfs_topk      = topk_sparsify(Z_cob_bfs,      TOP_K)
Z_cob_bfs_topk_test = topk_sparsify(Z_cob_bfs_test, TOP_K)
print(f"  Applied per-instance top-{TOP_K} sparsification")

# TopK-Binary / Depth-TopK-Binary: same selection but active nodes fixed to 1.0
Z_cob_topk_bin          = topk_binarize(Z_topk_pool,      TOP_K)
Z_cob_topk_bin_test     = topk_binarize(Z_topk_pool_test, TOP_K)
Z_cob_bfs_topk_bin      = topk_binarize(Z_cob_bfs,        TOP_K)
Z_cob_bfs_topk_bin_test = topk_binarize(Z_cob_bfs_test,   TOP_K)
print(f"  Applied per-instance top-{TOP_K} binarisation (TopK-Bin, Depth-TopK-Bin)")

# ── Path-information encoding ─────────────────────────────────────────────────
def collect_path_tree_nodes(root, max_depth):
    """BFS-collect all nodes at depths 0..max_depth with ancestor tracking."""
    all_nodes, node_to_idx, leaves, ancestor_ids = [], {}, [], {}
    queue = [(root, 0, frozenset())]
    while queue:
        node, depth, parent_ancs = queue.pop(0)
        idx_n = len(all_nodes)
        all_nodes.append(node)
        node_to_idx[id(node)] = idx_n
        my_ancs = parent_ancs | {id(node)}
        ancestor_ids[id(node)] = my_ancs
        if depth >= max_depth or not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                queue.append((child, depth + 1, my_ancs))
    return all_nodes, node_to_idx, leaves, ancestor_ids


def path_sparsify(Z_raw, Z_scaled, node_to_idx, leaves, ancestor_ids, n_paths):
    """Retain Z_scaled only for nodes on the top-n_paths leaf paths."""
    leaf_col = np.array([node_to_idx[id(lf)] for lf in leaves])
    n_paths  = min(n_paths, len(leaf_col))
    out      = np.zeros_like(Z_scaled)
    for i in range(Z_raw.shape[0]):
        scores    = Z_raw[i, leaf_col]
        top_local = np.argpartition(scores, -n_paths)[-n_paths:]
        path_ids  = set()
        for li in top_local:
            path_ids |= ancestor_ids[id(leaves[li])]
        for nid in path_ids:
            j         = node_to_idx[nid]
            out[i, j] = Z_scaled[i, j]
    return out


print(f"Collecting path-tree nodes (depth ≤ {PATH_DEPTH}) …")
path_all_nodes, path_node_to_idx, path_leaves, path_ancestor_ids = \
    collect_path_tree_nodes(cobweb_tree.root, PATH_DEPTH)
n_path_dim = len(path_all_nodes)
print(f"  {n_path_dim} total nodes, {len(path_leaves)} leaves")

print("Encoding path-tree (train) …")
Z_path_raw      = encode_logpost(instances_train, path_all_nodes)
print("Encoding path-tree (test) …")
Z_path_raw_test = encode_logpost(instances_test,  path_all_nodes)

_scaler_path    = StandardScaler()
Z_path_sc       = _scaler_path.fit_transform(Z_path_raw)
Z_path_sc_test  = _scaler_path.transform(Z_path_raw_test)

Z_cob_path      = path_sparsify(Z_path_raw,      Z_path_sc,      path_node_to_idx,
                                 path_leaves, path_ancestor_ids, N_PATHS)
Z_cob_path_test = path_sparsify(Z_path_raw_test, Z_path_sc_test, path_node_to_idx,
                                 path_leaves, path_ancestor_ids, N_PATHS)

# Save arrays
np.save(os.path.join(ARR_DIR, "Z_cob_bfs_train.npy"),      Z_cob_bfs)
np.save(os.path.join(ARR_DIR, "Z_cob_bfs_test.npy"),       Z_cob_bfs_test)
np.save(os.path.join(ARR_DIR, "Z_cob_dep_train.npy"),      Z_cob_dep)
np.save(os.path.join(ARR_DIR, "Z_cob_dep_test.npy"),       Z_cob_dep_test)
np.save(os.path.join(ARR_DIR, "Z_cob_topk_train.npy"),     Z_cob_topk)
np.save(os.path.join(ARR_DIR, "Z_cob_topk_test.npy"),      Z_cob_topk_test)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopk_train.npy"),  Z_cob_bfs_topk)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopk_test.npy"),   Z_cob_bfs_topk_test)
np.save(os.path.join(ARR_DIR, "Z_cob_topkbin_train.npy"),  Z_cob_topk_bin)
np.save(os.path.join(ARR_DIR, "Z_cob_topkbin_test.npy"),   Z_cob_topk_bin_test)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopkbin_train.npy"), Z_cob_bfs_topk_bin)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopkbin_test.npy"),  Z_cob_bfs_topk_bin_test)
np.save(os.path.join(ARR_DIR, "Z_cob_path_train.npy"),     Z_cob_path)
np.save(os.path.join(ARR_DIR, "Z_cob_path_test.npy"),      Z_cob_path_test)
print("Arrays saved.")

# ── Tree visualisation (word-type label distributions) ────────────────────────
CLASSES   = sorted(set(y.tolist()))
N_CLASSES = len(CLASSES)
CMAP      = plt.get_cmap("tab20") if N_CLASSES > 10 else plt.get_cmap("tab10")
word_colors = [CMAP(i / max(N_CLASSES - 1, 1)) for i in range(N_CLASSES)]
class_to_ci = {c: i for i, c in enumerate(CLASSES)}   # pos_id → color index


def compute_node_label_counts_disc(root, instances_tr, y_tr, max_depth=3):
    """Greedy-descend each training token; accumulate per-word counts at every
    ancestor node along the descent path."""
    n_classes = int(y_tr.max()) + 1
    counts    = {}
    node_obj  = {}

    def _ensure(node):
        nid = id(node)
        if nid not in counts:
            counts[nid]   = np.zeros(n_classes, dtype=np.int32)
            node_obj[nid] = node

    for inst, label in zip(instances_tr, y_tr):
        node = root
        for depth in range(max_depth + 1):
            _ensure(node)
            counts[id(node)][int(label)] += 1
            if not node.children or depth == max_depth:
                break
            node = max(node.children, key=lambda c: c.log_prob_instance(inst))

    return counts, node_obj


print("Computing node label distributions (greedy descent) …")
label_counts_map, node_obj_map = compute_node_label_counts_disc(
    cobweb_tree.root, instances_train, y, max_depth=3)


def plot_tree_word_labels(root, label_counts_map, out_path=None, max_depth=3):
    def leaf_span(node, depth, md):
        if depth >= md or not node.children:
            return 1
        return sum(leaf_span(c, depth + 1, md) for c in node.children)

    pos = {}

    def assign_pos(node, depth, x_left):
        span     = leaf_span(node, depth, max_depth)
        x_centre = x_left + span / 2.0
        pos[id(node)] = (x_centre, depth)
        if depth < max_depth and node.children:
            cursor = x_left
            for child in node.children:
                cs = leaf_span(child, depth + 1, max_depth)
                assign_pos(child, depth + 1, cursor)
                cursor += cs

    assign_pos(root, 0, 0.0)
    total_width = leaf_span(root, 0, max_depth)

    bar_w, bar_h, y_gap = 0.7, 0.35, 1.0
    fig, ax = plt.subplots(figsize=(max(14, total_width * 0.9), (max_depth + 1) * 2.2))
    ax.set_xlim(0, total_width)
    ax.set_ylim(-0.7, max_depth * y_gap + 0.7)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"Cobweb Discrete Tree — POS Distributions (depths 0–{max_depth})", fontsize=11)

    def draw_edges(node, depth):
        if depth >= max_depth or not node.children:
            return
        px, py = pos[id(node)]
        for child in node.children:
            cx, cy = pos[id(child)]
            ax.plot([px, cx], [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.8, zorder=0)
            draw_edges(child, depth + 1)

    draw_edges(root, 0)

    def draw_node(node, depth):
        nid = id(node)
        if nid not in label_counts_map:
            return
        cnts  = label_counts_map[nid].astype(float)
        total = cnts.sum()
        if total == 0:
            return
        props  = cnts / total
        x_c, _ = pos[nid]
        x_left  = x_c - bar_w / 2
        y_top   = depth * y_gap - bar_h / 2
        cursor  = x_left
        for wid in CLASSES:
            seg_w = props[wid] * bar_w
            if seg_w > 0:
                rect = plt.Rectangle((cursor, y_top), seg_w, bar_h,
                                     color=word_colors[class_to_ci[wid]], lw=0)
                ax.add_patch(rect)
                cursor += seg_w
        ax.add_patch(plt.Rectangle((x_left, y_top), bar_w, bar_h,
                                   fill=False, edgecolor="black", lw=0.5))
        ax.text(x_c, depth * y_gap + bar_h / 2 + 0.05,
                f"n={int(total)}", ha="center", va="top", fontsize=5)
        if depth < max_depth and node.children:
            for child in node.children:
                draw_node(child, depth + 1)

    draw_node(root, 0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=word_colors[class_to_ci[c]],
                       label=id2pos[c])
        for c in CLASSES
    ]
    ax.legend(handles=legend_handles, title="POS", loc="lower right",
              ncol=max(1, N_CLASSES // 4), fontsize=6, title_fontsize=7)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


plot_tree_word_labels(cobweb_tree.root, label_counts_map,
                      out_path=os.path.join(OUT_DIR, "cobweb_tree_labels.png"))
print("Tree label visualisation saved.")

# ── Evaluation ────────────────────────────────────────────────────────────────
KNN_KS = [1, 3, 5, 10, 20, 50]


def linear_probe_per_class(Z_tr, y_tr, Z_te, y_te):
    lin     = LinearSVC(max_iter=4000)
    lin.fit(Z_tr, y_tr)
    overall = lin.score(Z_te, y_te)
    per_class = np.array([
        lin.score(Z_te[y_te == c], y_te[y_te == c]) if (y_te == c).sum() > 0 else 0.0
        for c in CLASSES
    ])
    return overall, per_class


def knn_accuracy_vs_k(Z_tr, y_tr, Z_te, y_te, ks=KNN_KS):
    return [KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y_tr).score(Z_te, y_te)
            for k in ks]


def _repr_stats(Z):
    nz = (Z != 0)
    return nz.sum(axis=1).mean(), (~nz.any(axis=0)).mean() * 100


def softmax_entropy(Z):
    shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z   = np.exp(shifted)
    p       = exp_Z / exp_Z.sum(axis=1, keepdims=True)
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)


print("\nEvaluating …")
cob_bfs_lin_overall,      cob_bfs_lin_per      = linear_probe_per_class(Z_cob_bfs,      y, Z_cob_bfs_test,      y_test)
cob_dep_lin_overall,      cob_dep_lin_per      = linear_probe_per_class(Z_cob_dep,      y, Z_cob_dep_test,      y_test)
cob_topk_lin_overall,         cob_topk_lin_per         = linear_probe_per_class(Z_cob_topk,         y, Z_cob_topk_test,         y_test)
cob_bfs_topk_lin_overall,     cob_bfs_topk_lin_per     = linear_probe_per_class(Z_cob_bfs_topk,     y, Z_cob_bfs_topk_test,     y_test)
cob_topk_bin_lin_overall,     cob_topk_bin_lin_per     = linear_probe_per_class(Z_cob_topk_bin,     y, Z_cob_topk_bin_test,     y_test)
cob_bfs_topk_bin_lin_overall, cob_bfs_topk_bin_lin_per = linear_probe_per_class(Z_cob_bfs_topk_bin, y, Z_cob_bfs_topk_bin_test, y_test)
cob_path_lin_overall,     cob_path_lin_per     = linear_probe_per_class(Z_cob_path,     y, Z_cob_path_test,     y_test)

cob_bfs_knn_accs      = knn_accuracy_vs_k(Z_cob_bfs,      y, Z_cob_bfs_test,      y_test)
cob_dep_knn_accs      = knn_accuracy_vs_k(Z_cob_dep,      y, Z_cob_dep_test,      y_test)
cob_topk_knn_accs         = knn_accuracy_vs_k(Z_cob_topk,         y, Z_cob_topk_test,         y_test)
cob_bfs_topk_knn_accs     = knn_accuracy_vs_k(Z_cob_bfs_topk,     y, Z_cob_bfs_topk_test,     y_test)
cob_topk_bin_knn_accs     = knn_accuracy_vs_k(Z_cob_topk_bin,     y, Z_cob_topk_bin_test,     y_test)
cob_bfs_topk_bin_knn_accs = knn_accuracy_vs_k(Z_cob_bfs_topk_bin, y, Z_cob_bfs_topk_bin_test, y_test)
cob_path_knn_accs     = knn_accuracy_vs_k(Z_cob_path,     y, Z_cob_path_test,     y_test)

print(f"\n  {'Method':<60} {'Lin.Probe':>10} {'KNN@5':>7} {'Avg L0':>8} {'Dead%':>7}")
print(f"  {'-'*95}")
_knn5_idx     = KNN_KS.index(5)
_summary_rows = []
for name, overall, Z_tr, knn_accs in [
    (f"Cobweb-BFS ({len(bfs_nodes)}d)",                                  cob_bfs_lin_overall,      Z_cob_bfs,      cob_bfs_knn_accs),
    (f"Cobweb-Depth (depth={best_depth},dim={n_depth})",                 cob_dep_lin_overall,      Z_cob_dep,      cob_dep_knn_accs),
    (f"Cobweb-TopK (depth={topk_depth},dim={n_topk_pool},k={TOP_K})",     cob_topk_lin_overall,         Z_cob_topk,         cob_topk_knn_accs),
    (f"Cobweb-Depth-TopK ({len(bfs_nodes)}d, k={TOP_K})",                cob_bfs_topk_lin_overall,     Z_cob_bfs_topk,     cob_bfs_topk_knn_accs),
    (f"Cobweb-TopK-Bin (depth={topk_depth},dim={n_topk_pool},k={TOP_K})", cob_topk_bin_lin_overall,     Z_cob_topk_bin,     cob_topk_bin_knn_accs),
    (f"Cobweb-Depth-TopK-Bin ({len(bfs_nodes)}d, k={TOP_K})",             cob_bfs_topk_bin_lin_overall, Z_cob_bfs_topk_bin, cob_bfs_topk_bin_knn_accs),
    (f"Cobweb-Path (d={PATH_DEPTH},n={N_PATHS},dim={n_path_dim})",        cob_path_lin_overall,         Z_cob_path,         cob_path_knn_accs),
]:
    avg_l0, dead_pct = _repr_stats(Z_tr)
    avg_ent = softmax_entropy(Z_tr).mean()
    knn5    = knn_accs[_knn5_idx] * 100
    print(f"  {name:<60} {overall*100:>9.1f}% {knn5:>6.1f}% {avg_l0:>8.1f} {dead_pct:>6.1f}%  ent={avg_ent:.3f}")
    _summary_rows.append({
        "method":        name,
        "lin_probe_pct": round(overall * 100, 2),
        "knn5_pct":      round(knn5, 2),
        "avg_l0":        round(float(avg_l0), 2),
        "dead_pct":      round(float(dead_pct), 2),
        "avg_entropy":   round(float(avg_ent), 4),
    })

_csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(_csv_path, "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=["method", "lin_probe_pct", "knn5_pct",
                                         "avg_l0", "dead_pct", "avg_entropy"])
    _w.writeheader()
    _w.writerows(_summary_rows)
print(f"  Summary saved → {_csv_path}")

# ── Visualisation ─────────────────────────────────────────────────────────────
METHODS = [
    (Z_cob_bfs,      Z_cob_bfs_test,      cob_bfs_lin_per,      cob_bfs_knn_accs,
     f"Cobweb-BFS ({len(bfs_nodes)}d)",                               "^-", "#6acc65"),
    (Z_cob_dep,      Z_cob_dep_test,      cob_dep_lin_per,      cob_dep_knn_accs,
     f"Cobweb-Depth (depth={best_depth},dim={n_depth})",              "D-", "#d65f5f"),
    (Z_cob_topk,         Z_cob_topk_test,         cob_topk_lin_per,         cob_topk_knn_accs,
     f"Cobweb-TopK (depth={topk_depth},dim={n_topk_pool},k={TOP_K})",     "P-", "#956cb4"),
    (Z_cob_bfs_topk,     Z_cob_bfs_topk_test,     cob_bfs_topk_lin_per,     cob_bfs_topk_knn_accs,
     f"Cobweb-Depth-TopK ({len(bfs_nodes)}d, k={TOP_K})",                "X-", "#17becf"),
    (Z_cob_topk_bin,     Z_cob_topk_bin_test,     cob_topk_bin_lin_per,     cob_topk_bin_knn_accs,
     f"Cobweb-TopK-Bin (depth={topk_depth},dim={n_topk_pool},k={TOP_K})", "8-", "#c39bd3"),
    (Z_cob_bfs_topk_bin, Z_cob_bfs_topk_bin_test, cob_bfs_topk_bin_lin_per, cob_bfs_topk_bin_knn_accs,
     f"Cobweb-Depth-TopK-Bin ({len(bfs_nodes)}d, k={TOP_K})",            ">-", "#76d7c4"),
    (Z_cob_path,         Z_cob_path_test,         cob_path_lin_per,         cob_path_knn_accs,
     f"Cobweb-Path (d={PATH_DEPTH},n={N_PATHS},dim={n_path_dim})",        "p-", "#8c564b"),
]
n_meth = len(METHODS)

# Legend handles (shared across scatter plots)
_leg_handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CMAP(i / max(N_CLASSES - 1, 1)),
               markersize=6, label=id2pos[CLASSES[i]])
    for i in range(N_CLASSES)
]

# 1a. UMAP scatter
print("Computing UMAP projections …")
_umap      = UMAP(n_components=2, random_state=SEED)
projs_umap = [_umap.fit_transform(Z) for Z, _, _, _, _, _, _ in METHODS]

fig, axes = plt.subplots(1, n_meth, figsize=(n_meth * 5, 5))
fig.suptitle("UMAP Projections  —  Grammar (TEST_GRAMMAR3)", fontsize=11, y=1.01)
for ax, Z2, (_, _, _, _, lbl, _, _) in zip(axes, projs_umap, METHODS):
    for ci, wid in enumerate(CLASSES):
        mask = y == wid
        ax.scatter(Z2[mask, 0], Z2[mask, 1],
                   color=CMAP(ci / max(N_CLASSES - 1, 1)), alpha=0.45, s=6)
    ax.set_title(lbl, fontsize=8)
    ax.set_xlabel("Dim 1", fontsize=7)
    ax.set_ylabel("Dim 2", fontsize=7)
    ax.tick_params(labelsize=6)
fig.legend(handles=_leg_handles, title="POS", loc="center right",
           bbox_to_anchor=(1.0, 0.5), ncol=2, fontsize=6, title_fontsize=7,
           frameon=True)
plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig(os.path.join(OUT_DIR, "scatter_umap.png"), dpi=120, bbox_inches="tight")
plt.close()
print("UMAP scatter saved.")

# 1b. t-SNE scatter
print("Computing t-SNE projections …")
_tsne      = TSNE(n_components=2, random_state=SEED, n_jobs=-1)
projs_tsne = [_tsne.fit_transform(Z) for Z, _, _, _, _, _, _ in METHODS]

fig, axes = plt.subplots(1, n_meth, figsize=(n_meth * 5, 5))
fig.suptitle("t-SNE Projections  —  Grammar (TEST_GRAMMAR3)", fontsize=11, y=1.01)
for ax, Z2, (_, _, _, _, lbl, _, _) in zip(axes, projs_tsne, METHODS):
    for ci, wid in enumerate(CLASSES):
        mask = y == wid
        ax.scatter(Z2[mask, 0], Z2[mask, 1],
                   color=CMAP(ci / max(N_CLASSES - 1, 1)), alpha=0.45, s=6)
    ax.set_title(lbl, fontsize=8)
    ax.set_xlabel("Dim 1", fontsize=7)
    ax.set_ylabel("Dim 2", fontsize=7)
    ax.tick_params(labelsize=6)
fig.legend(handles=_leg_handles, title="POS", loc="center right",
           bbox_to_anchor=(1.0, 0.5), ncol=2, fontsize=6, title_fontsize=7,
           frameon=True)
plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig(os.path.join(OUT_DIR, "scatter_tsne.png"), dpi=120, bbox_inches="tight")
plt.close()
print("t-SNE scatter saved.")

# 2. Linear probe — per-word test accuracy
w_bar   = 0.8 / n_meth
x_bar   = np.arange(N_CLASSES)
offsets = [(i - (n_meth - 1) / 2) * w_bar for i in range(n_meth)]
fig, ax = plt.subplots(figsize=(max(16, N_CLASSES * 1.0), 5))
for (_, _, per, _, lbl, _, color), offset in zip(METHODS, offsets):
    ax.bar(x_bar + offset, per * 100, w_bar, label=lbl, color=color, alpha=0.85)
ax.set_xticks(x_bar)
ax.set_xticklabels([id2pos[c] for c in CLASSES], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Test Accuracy %")
ax.set_title("Linear Probe — Per-POS Test Accuracy  (Grammar / TEST_GRAMMAR3)")
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
ax.set_title("KNN Test Accuracy vs k  (Grammar / TEST_GRAMMAR3)")
ax.set_xticks(KNN_KS)
ax.set_ylim(0, 105)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "knn_vs_k.png"), dpi=120)
plt.close()

# 4. Dimension activity histograms + softmax entropy
_n_methods = len(METHODS)
_n_cols    = 3
_n_rows    = (_n_methods + _n_cols - 1) // _n_cols
fig, axes  = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 3.8))
axes       = axes.flatten()

for idx, (Z_tr, _, _, _, lbl, _, color) in enumerate(METHODS):
    ax        = axes[idx]
    fire_freq = (Z_tr != 0).mean(axis=0)
    avg_ent   = softmax_entropy(Z_tr).mean()
    ax.hist(fire_freq, bins=60, color=color, alpha=0.82, edgecolor="white", linewidth=0.3)
    ax.axvline(fire_freq.mean(), color="black", linewidth=1.0, linestyle="--",
               label=f"mean={fire_freq.mean():.3f}")
    ax.set_title(f"{lbl}\nAvg softmax entropy: {avg_ent:.3f} nats", fontsize=8)
    ax.set_xlabel("Dimension fire frequency (fraction of samples)", fontsize=7)
    ax.set_ylabel("# dimensions", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

for idx in range(_n_methods, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    "Per-dimension Activity Frequency  —  Grammar (TEST_GRAMMAR3)\n"
    "(bar height = # features that fire at that fraction of tokens;\n"
    " subtitle entropy = average H[softmax(z)] per token)",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dim_activity_hist.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"Dim-activity histogram saved → {os.path.join(OUT_DIR, 'dim_activity_hist.png')}")

# 5. Per-node token-count bar chart
fig, axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 6, _n_rows * 3.8))
axes      = axes.flatten()

for idx, (Z_tr, _, _, _, lbl, _, color) in enumerate(METHODS):
    ax          = axes[idx]
    node_counts = (Z_tr != 0).sum(axis=0)
    n_active    = (node_counts > 0).sum()
    n_dead      = (node_counts == 0).sum()
    ax.bar(np.arange(len(node_counts)), node_counts,
           color=color, alpha=0.82, width=1.0, linewidth=0)
    ax.axhline(
        node_counts[node_counts > 0].mean() if n_active > 0 else 0,
        color="black", linewidth=1.0, linestyle="--",
        label=(f"mean (active)={node_counts[node_counts > 0].mean():.1f}"
               if n_active > 0 else "mean=0"),
    )
    ax.set_title(
        f"{lbl}\nactive: {n_active}  |  dead: {n_dead}  |  total: {Z_tr.shape[1]}",
        fontsize=8,
    )
    ax.set_xlabel("node index (native / BFS order)", fontsize=7)
    ax.set_ylabel("# training tokens node is active for", fontsize=7)
    ax.set_xlim(0, len(node_counts))
    ax.set_ylim(0, len(Z_tr) * 1.05)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

for idx in range(_n_methods, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    "Per-node Token Activity  —  Grammar (TEST_GRAMMAR3)\n"
    "(nodes in native/BFS order; dashed line = mean over active nodes)",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "node_activity_hist.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"Node-activity plot saved → {os.path.join(OUT_DIR, 'node_activity_hist.png')}")

# 6. Competitive-node count histograms (Cobweb variants only)
def _competitive_node_fig(cob_methods, threshold, fname_suffix, dataset="Grammar"):
    """Histogram of per-token competitive-node count at a given probability threshold."""
    pct_str   = f"{int(threshold * 100)}%"
    _cn_cols  = min(3, len(cob_methods))
    _cn_rows  = (len(cob_methods) + _cn_cols - 1) // _cn_cols
    fig, axes = plt.subplots(_cn_rows, _cn_cols, figsize=(_cn_cols * 5, _cn_rows * 3.8))
    axes      = np.array(axes).flatten()

    for idx, (Z_tr, lbl, color) in enumerate(cob_methods):
        ax       = axes[idx]
        Z_f      = Z_tr.astype(np.float64)
        best     = Z_f.max(axis=1, keepdims=True)
        rel_prob = np.exp(np.clip(Z_f - best, -500, 0))
        counts   = (rel_prob >= threshold).sum(axis=1)

        if len(counts) == 0:
            max_count = 1
        else:
            hist_vals, _ = np.histogram(counts,
                                        bins=np.arange(0.5, counts.max() + 1.5, 1.0))
            nonzero_bins = np.where(hist_vals > 0)[0]
            max_count    = int(nonzero_bins[-1]) + 1 if len(nonzero_bins) else 1

        bins = np.arange(0.5, max_count + 1.5, 1.0)
        ax.hist(counts, bins=bins, color=color, alpha=0.82,
                edgecolor="white", linewidth=0.4)
        ax.axvline(counts.mean(), color="black", linewidth=1.2, linestyle="--",
                   label=f"mean={counts.mean():.2f}")
        ax.set_title(
            f"{lbl}\nmedian={np.median(counts):.0f}  |  mean={counts.mean():.2f}",
            fontsize=8,
        )
        ax.set_xlabel(
            f"# nodes with relative probability ≥ {pct_str} of best node",
            fontsize=7,
        )
        ax.set_ylabel("# training tokens", fontsize=7)
        _tick_positions = np.unique(
            np.round(np.linspace(1, max_count, min(11, max_count))).astype(int)
        )
        ax.set_xticks(_tick_positions)
        ax.set_xlim(0.5, max_count + 0.5)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    for idx in range(len(cob_methods), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Competitive-node count per token  —  {dataset}  (Cobweb variants)\n"
        f"X = # nodes whose probability is ≥ {pct_str} of the best node's probability\n"
        f"[exp(z_j − z_best) ≥ {threshold}]   Y = # training tokens with that count",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"competitive_node_hist_{fname_suffix}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Competitive-node histogram ({pct_str}) saved → {out_path}")


_cob_methods = [(Z_tr, lbl, color)
                for Z_tr, _, _, _, lbl, _, color in METHODS
                if lbl.startswith("Cobweb")]

_competitive_node_fig(_cob_methods, threshold=0.90, fname_suffix="90pct")
_competitive_node_fig(_cob_methods, threshold=0.75, fname_suffix="75pct")
_competitive_node_fig(_cob_methods, threshold=0.50, fname_suffix="50pct")

print(f"\nAll outputs written to: {OUT_DIR}")
