"""
Grammar-chunking — default Cobweb vs CU Cobweb on the two-tree content/context
setup, for the two variants:

  TopK-Cont       (context: discrete, content: continuous)
  TopK-Disc-Cnt1  (context: discrete, content: discrete)

For each variant we build a full default-stack and a full CU-stack
(BOTH context and content trees are the same family) from the IDENTICAL
sentence stream.  This yields 4 (variant × family) configurations.

Outputs (tests/cu_cobweb/grammar_chunking_output/):
  - tree_context_default.png / tree_context_cu.png
  - tree_content_topk_cont_default.png / _cu.png
  - tree_content_topk_disc_cnt1_default.png / _cu.png
  - scatter_umap.png / scatter_tsne.png        4 panels per scatter plot
  - linear_probe_per_class.png                 per-(L,R)-POS bar chart, 4 methods
  - knn_vs_k.png                                4-line KNN curve
  - dim_activity_hist.png                       per-dim fire-freq, 4 panels
  - competitive_node_hist_*pct.png              4 panels
  - summary.csv
"""

import os
import sys
import csv
import math
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

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from util.cfg import TEST_GRAMMAR3, generate

from cobweb.cobweb_discrete       import CobwebDiscreteTree
from cobweb.cobweb_continuous     import CobwebContinuousTree
from cobweb.cu_cobweb_discrete    import CUCobwebDiscreteTree
from cobweb.cu_cobweb_continuous  import CUCobwebContinuousTree

OUT_DIR  = os.path.join(_HERE, "grammar_chunking_output")
ARR_DIR  = os.path.join(OUT_DIR, "arrays")
TREE_DIR = os.path.join(OUT_DIR, "tree_viz")
for _d in (OUT_DIR, ARR_DIR, TREE_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Constants (match tests/moc/grammar_chunking_example.py defaults) ─────────
N_SENTENCES = 1000
WINDOW      = 3
DZ_CONTEXT  = 128
DZ_CONTENT  = 128
TOP_K       = 5
TOPK_DEPTH  = 4
SEED        = 42
ALPHA       = 1e-3
WEIGHT_ATTR = True
SCALING     = 0.5   # CU continuous
COVAR_FROM  = 1     # default continuous
KNN_KS      = [1, 3, 5, 10, 20, 50]
TREE_VIZ_DEPTH = 3

random.seed(SEED)
np.random.seed(SEED)

# ── Corpus ───────────────────────────────────────────────────────────────────
print(f"Generating {N_SENTENCES} sentences …")
sentences = []
for _ in range(N_SENTENCES):
    sent = [w for w in generate("S", TEST_GRAMMAR3).split() if w]
    if len(sent) >= 2:
        sentences.append(sent)
vocab   = sorted({w for s in sentences for w in s})
word2id = {w: i for i, w in enumerate(vocab)}
V       = len(vocab)
print(f"  Sentences: {len(sentences)} | Vocab: {V}")

_TERMINAL_POS = ["Det", "N", "Adj", "RelPro", "V", "P"]
word2pos = {}
for _pos in _TERMINAL_POS:
    if _pos in TEST_GRAMMAR3:
        for _prod in TEST_GRAMMAR3[_pos]:
            if len(_prod) == 1 and _prod[0] not in word2pos:
                word2pos[_prod[0]] = _pos
for _w in vocab:
    word2pos.setdefault(_w, "Unk")
pos_tags = sorted(set(word2pos.values()))
pos2id   = {p: i for i, p in enumerate(pos_tags)}
id2pos   = {i: p for p, i in pos2id.items()}
N_POS    = len(pos_tags)


def pair_label(l, r):
    return pos2id[word2pos[l]] * N_POS + pos2id[word2pos[r]]


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
print(f"  Context-tree training tokens: {len(context_train_tokens)}")


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
print(f"  Pairs: train={len(train_L)}  test={len(test_L)}")

np.save(os.path.join(ARR_DIR, "y_train.npy"), y_train)
np.save(os.path.join(ARR_DIR, "y_test.npy"),  y_test)


# ── Build both context trees from the IDENTICAL stream ───────────────────────
print(f"\nBuilding default context tree (CobwebDiscreteTree, alpha={ALPHA}) …")
ctx_default = CobwebDiscreteTree(alpha=ALPHA, weight_attr=WEIGHT_ATTR)
for i, inst in enumerate(context_train_tokens):
    ctx_default.ifit(inst)
    if (i + 1) % 1000 == 0:
        print(f"  default ctx: {i+1}/{len(context_train_tokens)}")

print(f"\nBuilding CU context tree (CUCobwebDiscreteTree, alpha={ALPHA}) …")
ctx_cu = CUCobwebDiscreteTree(alpha=ALPHA, weight_attr=WEIGHT_ATTR)
for i, inst in enumerate(context_train_tokens):
    ctx_cu.ifit(inst)
    if (i + 1) % 1000 == 0:
        print(f"  CU ctx: {i+1}/{len(context_train_tokens)}")


# ── Helpers ──────────────────────────────────────────────────────────────────
def collect_by_depth(root):
    by_depth, queue = {}, [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        by_depth.setdefault(d, []).append(node)
        for c in node.children:
            queue.append((c, d + 1))
    return by_depth


def bfs_first_n(root, n):
    nodes, queue = [], [root]
    while queue and len(nodes) < n:
        node = queue.pop(0)
        for c in node.children:
            if len(nodes) >= n:
                break
            nodes.append(c)
            queue.append(c)
    return nodes


def bfs_or_root(tree, k):
    n = bfs_first_n(tree.root, k)
    return n if n else [tree.root]


# Discrete encoding (default == CU API)
def encode_disc_nodes(instances, nodes):
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, inst in enumerate(instances):
            out[i, j] = node.log_prob_instance(inst)
    return out


# Continuous encoding (default needs labels arg, CU does not)
_empty_lbl = np.zeros(0, dtype=np.float32)


def encode_cont_default(vecs, nodes):
    out = np.empty((vecs.shape[0], len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i in range(vecs.shape[0]):
            out[i, j] = node.log_prob(vecs[i], _empty_lbl)
    return out


def encode_cont_cu(vecs, nodes):
    out = np.empty((vecs.shape[0], len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i in range(vecs.shape[0]):
            out[i, j] = node.log_prob(vecs[i])
    return out


def topk_sparsify(Z, k):
    out = np.zeros_like(Z)
    k   = min(k, Z.shape[1])
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    rows    = np.arange(Z.shape[0])[:, None]
    out[rows, top_idx] = Z[rows, top_idx]
    return out


def topk_indices(Z_raw, k):
    k = min(k, Z_raw.shape[1])
    return np.argpartition(Z_raw, -k, axis=1)[:, -k:]


# ── Extract context-tree TopK pool ───────────────────────────────────────────
def topk_pool_for(ctx_tree, label):
    by_depth = collect_by_depth(ctx_tree.root)
    topk_d   = min(TOPK_DEPTH, max(by_depth))
    pool     = by_depth[topk_d]
    if topk_d != TOPK_DEPTH:
        print(f"  [{label}] requested depth {TOPK_DEPTH} unavailable; clamped to {topk_d}")
    print(f"  [{label}] TopK pool: depth {topk_d}  ({len(pool)} nodes)  k={TOP_K}")
    return pool, topk_d


print("\nExtracting TopK pools …")
ctx_default_pool, depth_def = topk_pool_for(ctx_default, "default")
ctx_cu_pool,      depth_cu  = topk_pool_for(ctx_cu,      "CU")


# ── Build per-(family) inputs to the content trees ───────────────────────────
def build_topk_cont_input(ctx_tree_pool, name):
    """Encode L,R against the pool; per-side StandardScaler + TopK sparsify;
    concatenate to get a 2*dim_pool dense (but k-sparse per side) vector."""
    print(f"  [{name}] encoding pool features …")
    raw_tr_L = encode_disc_nodes(train_L, ctx_tree_pool)
    raw_tr_R = encode_disc_nodes(train_R, ctx_tree_pool)
    raw_te_L = encode_disc_nodes(test_L,  ctx_tree_pool)
    raw_te_R = encode_disc_nodes(test_R,  ctx_tree_pool)
    sc       = StandardScaler().fit(np.vstack([raw_tr_L, raw_tr_R]))
    tr_L     = sc.transform(raw_tr_L); tr_R = sc.transform(raw_tr_R)
    te_L     = sc.transform(raw_te_L); te_R = sc.transform(raw_te_R)
    cont_tr  = np.hstack([topk_sparsify(tr_L, TOP_K),
                          topk_sparsify(tr_R, TOP_K)]).astype(np.float32)
    cont_te  = np.hstack([topk_sparsify(te_L, TOP_K),
                          topk_sparsify(te_R, TOP_K)]).astype(np.float32)
    # raw L/R also used by topk-disc-cnt1 below
    return cont_tr, cont_te, (raw_tr_L, raw_tr_R, raw_te_L, raw_te_R)


def build_topk_disc_cnt1_input(raw_LR, name):
    """Per-row TopK indices into the pool → {0:{idx:1}, 1:{idx:1}} discrete inst."""
    raw_tr_L, raw_tr_R, raw_te_L, raw_te_R = raw_LR
    idx_tr_L = topk_indices(raw_tr_L, TOP_K)
    idx_tr_R = topk_indices(raw_tr_R, TOP_K)
    idx_te_L = topk_indices(raw_te_L, TOP_K)
    idx_te_R = topk_indices(raw_te_R, TOP_K)

    def _inst(idx_L, idx_R):
        return [{0: {int(j): 1.0 for j in iL},
                 1: {int(j): 1.0 for j in iR}}
                for iL, iR in zip(idx_L, idx_R)]

    return _inst(idx_tr_L, idx_tr_R), _inst(idx_te_L, idx_te_R)


print("\nBuilding TopK-Cont inputs …")
cont_def_tr, cont_def_te, raw_LR_def = build_topk_cont_input(ctx_default_pool, "default")
cont_cu_tr,  cont_cu_te,  raw_LR_cu  = build_topk_cont_input(ctx_cu_pool,      "CU")

print("Building TopK-Disc-Cnt1 inputs …")
disc_def_tr, disc_def_te = build_topk_disc_cnt1_input(raw_LR_def, "default")
disc_cu_tr,  disc_cu_te  = build_topk_disc_cnt1_input(raw_LR_cu,  "CU")


# ── Train content trees ──────────────────────────────────────────────────────
def train_disc_tree(cls, name, train_insts):
    print(f"\n[{name}] Training {cls.__name__} (alpha={ALPHA}, n={len(train_insts)}) …")
    tree = cls(alpha=ALPHA, weight_attr=WEIGHT_ATTR)
    for i, inst in enumerate(train_insts):
        tree.ifit(inst)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(train_insts)}")
    return tree


def train_default_cont_tree(name, train_vecs):
    print(f"\n[{name}] Training CobwebContinuousTree "
          f"(covar_from={COVAR_FROM}, size={train_vecs.shape[1]}, n={train_vecs.shape[0]}) …")
    tree = CobwebContinuousTree(size=train_vecs.shape[1],
                                 covar_from=COVAR_FROM,
                                 num_labels=0,
                                 alpha=ALPHA)
    for i, x in enumerate(train_vecs):
        tree.ifit(x, _empty_lbl)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{train_vecs.shape[0]}")
    return tree


def train_cu_cont_tree(name, train_vecs):
    print(f"\n[{name}] Training CUCobwebContinuousTree "
          f"(scaling={SCALING}, size={train_vecs.shape[1]}, n={train_vecs.shape[0]}) …")
    tree = CUCobwebContinuousTree(size=train_vecs.shape[1],
                                   num_labels=0,
                                   scaling=SCALING)
    for i, x in enumerate(train_vecs):
        tree.ifit(x, _empty_lbl)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{train_vecs.shape[0]}")
    return tree


tree_tkc_def    = train_default_cont_tree("TopK-Cont default", cont_def_tr)
tree_tkc_cu     = train_cu_cont_tree     ("TopK-Cont CU",      cont_cu_tr)
tree_tkdc1_def  = train_disc_tree(CobwebDiscreteTree,    "TopK-Disc-Cnt1 default", disc_def_tr)
tree_tkdc1_cu   = train_disc_tree(CUCobwebDiscreteTree,  "TopK-Disc-Cnt1 CU",      disc_cu_tr)


# ── Encode pair → BFS-DZ_CONTENT features per content tree ───────────────────
def encode_disc_variant(tree, train_insts, test_insts, name):
    nodes = bfs_or_root(tree, DZ_CONTENT)
    print(f"  [{name}] content BFS-{DZ_CONTENT}: {len(nodes)} nodes")
    Z_tr_raw = encode_disc_nodes(train_insts, nodes)
    Z_te_raw = encode_disc_nodes(test_insts,  nodes)
    sc       = StandardScaler().fit(Z_tr_raw)
    return sc.transform(Z_tr_raw), sc.transform(Z_te_raw), nodes


def encode_cont_variant_default(tree, train_vecs, test_vecs, name):
    nodes = bfs_or_root(tree, DZ_CONTENT)
    print(f"  [{name}] content BFS-{DZ_CONTENT}: {len(nodes)} nodes")
    Z_tr_raw = encode_cont_default(train_vecs, nodes)
    Z_te_raw = encode_cont_default(test_vecs,  nodes)
    sc       = StandardScaler().fit(Z_tr_raw)
    return sc.transform(Z_tr_raw), sc.transform(Z_te_raw), nodes


def encode_cont_variant_cu(tree, train_vecs, test_vecs, name):
    nodes = bfs_or_root(tree, DZ_CONTENT)
    print(f"  [{name}] content BFS-{DZ_CONTENT}: {len(nodes)} nodes")
    Z_tr_raw = encode_cont_cu(train_vecs, nodes)
    Z_te_raw = encode_cont_cu(test_vecs,  nodes)
    sc       = StandardScaler().fit(Z_tr_raw)
    return sc.transform(Z_tr_raw), sc.transform(Z_te_raw), nodes


print("\nEncoding pair features …")
Z_tkc_def_tr,   Z_tkc_def_te,   nodes_tkc_def   = encode_cont_variant_default(
    tree_tkc_def,  cont_def_tr, cont_def_te, "TopK-Cont default")
Z_tkc_cu_tr,    Z_tkc_cu_te,    nodes_tkc_cu    = encode_cont_variant_cu(
    tree_tkc_cu,   cont_cu_tr,  cont_cu_te,  "TopK-Cont CU")
Z_tkdc1_def_tr, Z_tkdc1_def_te, nodes_tkdc1_def = encode_disc_variant(
    tree_tkdc1_def, disc_def_tr, disc_def_te, "TopK-Disc-Cnt1 default")
Z_tkdc1_cu_tr,  Z_tkdc1_cu_te,  nodes_tkdc1_cu  = encode_disc_variant(
    tree_tkdc1_cu,  disc_cu_tr,  disc_cu_te,  "TopK-Disc-Cnt1 CU")

for nm, Ztr, Zte in [
    ("tkc_def",   Z_tkc_def_tr,   Z_tkc_def_te),
    ("tkc_cu",    Z_tkc_cu_tr,    Z_tkc_cu_te),
    ("tkdc1_def", Z_tkdc1_def_tr, Z_tkdc1_def_te),
    ("tkdc1_cu",  Z_tkdc1_cu_tr,  Z_tkdc1_cu_te),
]:
    np.save(os.path.join(ARR_DIR, f"Z_{nm}_train.npy"), Ztr)
    np.save(os.path.join(ARR_DIR, f"Z_{nm}_test.npy"),  Zte)


# ── Tree visualisation (top levels) ──────────────────────────────────────────
CMAP_POS   = plt.get_cmap("tab10") if N_POS <= 10 else plt.get_cmap("tab20")
pos_colors = [CMAP_POS(i / max(N_POS - 1, 1)) for i in range(N_POS)]


def make_static_layout(root, max_depth):
    """One BFS that materialises wrappers so id() stays stable for layout."""
    all_nodes   = [root]
    children_of = {0: []}
    depth_of    = {0: 0}
    queue       = [0]
    while queue:
        idx  = queue.pop(0)
        node = all_nodes[idx]
        if depth_of[idx] < max_depth:
            for c in node.children:
                ci = len(all_nodes)
                all_nodes.append(c)
                children_of[idx].append(ci)
                children_of[ci] = []
                depth_of[ci]    = depth_of[idx] + 1
                queue.append(ci)
    return all_nodes, children_of, depth_of


def descend_idx_disc(all_nodes, children_of, inst, max_depth):
    visited, cur = [0], 0
    for _ in range(max_depth):
        ch = children_of[cur]
        if not ch:
            break
        best = max(ch, key=lambda i: all_nodes[i].log_prob_instance(inst))
        visited.append(best)
        cur = best
    return visited


def descend_idx_cont_default(all_nodes, children_of, vec, max_depth):
    visited, cur = [0], 0
    for _ in range(max_depth):
        ch = children_of[cur]
        if not ch:
            break
        best = max(ch, key=lambda i: all_nodes[i].log_prob(vec, _empty_lbl))
        visited.append(best)
        cur = best
    return visited


def descend_idx_cont_cu(all_nodes, children_of, vec, max_depth):
    visited, cur = [0], 0
    for _ in range(max_depth):
        ch = children_of[cur]
        if not ch:
            break
        best = max(ch, key=lambda i: all_nodes[i].log_prob(vec))
        visited.append(best)
        cur = best
    return visited


def count_single_label_at_nodes(all_nodes, children_of, instances, y_arr,
                                 descend_fn, max_depth, n_classes):
    counts = {}
    for x, lbl in zip(instances, y_arr):
        for idx in descend_fn(all_nodes, children_of, x, max_depth):
            if idx not in counts:
                counts[idx] = np.zeros(n_classes, dtype=np.int32)
            counts[idx][int(lbl)] += 1
    return counts


def count_pair_label_at_nodes(all_nodes, children_of, instances, y_arr,
                               descend_fn, max_depth):
    cL, cR = {}, {}
    for x, lbl in zip(instances, y_arr):
        l_pos, r_pos = split_pair_label(int(lbl))
        for idx in descend_fn(all_nodes, children_of, x, max_depth):
            if idx not in cL:
                cL[idx] = np.zeros(N_POS, dtype=np.int32)
                cR[idx] = np.zeros(N_POS, dtype=np.int32)
            cL[idx][l_pos] += 1
            cR[idx][r_pos] += 1
    return cL, cR


def _leaf_span(children_of, idx, depth, max_depth):
    if depth >= max_depth or not children_of[idx]:
        return 1
    return sum(_leaf_span(children_of, c, depth + 1, max_depth) for c in children_of[idx])


def _layout_x(children_of, max_depth):
    pos = {}
    def _assign(idx, depth, x_left):
        span = _leaf_span(children_of, idx, depth, max_depth)
        pos[idx] = (x_left + span / 2.0, depth)
        if depth < max_depth and children_of[idx]:
            cur = x_left
            for c in children_of[idx]:
                cs = _leaf_span(children_of, c, depth + 1, max_depth)
                _assign(c, depth + 1, cur)
                cur += cs
    _assign(0, 0, 0.0)
    return pos, _leaf_span(children_of, 0, 0, max_depth)


def plot_single_label_tree_idx(children_of, counts, title, out_path, max_depth):
    pos, total_w = _layout_x(children_of, max_depth)
    bar_w, bar_h, y_gap = 0.7, 0.35, 1.0
    fig, ax = plt.subplots(figsize=(max(14, total_w * 0.55), (max_depth + 1) * 2.2))
    ax.set_xlim(0, total_w); ax.set_ylim(-0.7, max_depth * y_gap + 0.7)
    ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=12)

    def _edges(idx, depth):
        if depth >= max_depth or not children_of[idx]:
            return
        px, py = pos[idx]
        for c in children_of[idx]:
            cx, cy = pos[c]
            ax.plot([px, cx], [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.7, zorder=0)
            _edges(c, depth + 1)
    _edges(0, 0)

    def _draw(idx, depth):
        if idx not in counts:
            return
        cnts  = counts[idx].astype(float); total = cnts.sum()
        if total == 0:
            return
        props  = cnts / total
        x_c, _ = pos[idx]
        x_left = x_c - bar_w / 2
        y_top  = depth * y_gap - bar_h / 2
        cur = x_left
        for p_i in range(N_POS):
            seg_w = props[p_i] * bar_w
            if seg_w > 0:
                ax.add_patch(plt.Rectangle((cur, y_top), seg_w, bar_h,
                                            color=pos_colors[p_i], lw=0))
                cur += seg_w
        ax.add_patch(plt.Rectangle((x_left, y_top), bar_w, bar_h,
                                    fill=False, edgecolor="black", lw=0.4))
        ax.text(x_c, depth * y_gap + bar_h / 2 + 0.05,
                f"n={int(total)}", ha="center", va="top", fontsize=4)
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]:
                _draw(c, depth + 1)
    _draw(0, 0)

    legend_h = [plt.Rectangle((0, 0), 1, 1, color=pos_colors[i], label=id2pos[i])
                for i in range(N_POS)]
    ax.legend(handles=legend_h, title="POS", loc="lower right",
              ncol=max(1, N_POS // 4), fontsize=7, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_pair_label_tree_idx(children_of, counts_L, counts_R, title, out_path, max_depth):
    pos, total_w = _layout_x(children_of, max_depth)
    bar_w, bar_h, gap, y_unit = 0.7, 0.18, 0.05, 1.0
    fig, ax = plt.subplots(figsize=(max(14, total_w * 0.55), (max_depth + 1) * 2.4))
    ax.set_xlim(0, total_w); ax.set_ylim(-0.7, max_depth * y_unit + 0.7)
    ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=12)

    def _edges(idx, depth):
        if depth >= max_depth or not children_of[idx]:
            return
        px, py = pos[idx]
        for c in children_of[idx]:
            cx, cy = pos[c]
            y_par = py * y_unit + bar_h + gap / 2
            y_chi = cy * y_unit - bar_h - gap / 2
            ax.plot([px, cx], [y_par, y_chi], color="gray", lw=0.7, zorder=0)
            _edges(c, depth + 1)
    _edges(0, 0)

    def _bar(x_left, y_top, props, txt):
        cur = x_left
        for p_i in range(N_POS):
            seg_w = props[p_i] * bar_w
            if seg_w > 0:
                ax.add_patch(plt.Rectangle((cur, y_top), seg_w, bar_h,
                                            color=pos_colors[p_i], lw=0))
                cur += seg_w
        ax.add_patch(plt.Rectangle((x_left, y_top), bar_w, bar_h,
                                    fill=False, edgecolor="black", lw=0.4))
        ax.text(x_left - 0.04, y_top + bar_h / 2, txt,
                ha="right", va="center", fontsize=5)

    def _draw(idx, depth):
        if idx not in counts_L:
            return
        cL = counts_L[idx].astype(float)
        cR = counts_R[idx].astype(float)
        tot = cL.sum()
        if tot == 0:
            return
        propsL = cL / tot; propsR = cR / tot
        x_c, _ = pos[idx]; x_left = x_c - bar_w / 2
        y_top_L = depth * y_unit - bar_h - gap / 2
        y_top_R = depth * y_unit + gap / 2
        _bar(x_left, y_top_L, propsL, "L")
        _bar(x_left, y_top_R, propsR, "R")
        ax.text(x_c, y_top_L - 0.04, f"n={int(tot)}", ha="center", va="bottom", fontsize=5)
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]:
                _draw(c, depth + 1)
    _draw(0, 0)

    legend_h = [plt.Rectangle((0, 0), 1, 1, color=pos_colors[i], label=id2pos[i])
                for i in range(N_POS)]
    ax.legend(handles=legend_h, title="POS (top=L, bottom=R)",
              loc="lower right", ncol=max(1, N_POS // 4),
              fontsize=7, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


# Context trees: greedy-descend on tokens → POS bars
print("\nRendering context-tree visualisations …")
ctx_train_lbls = np.array([pos2id[word2pos[w]] for s in train_sents for w in s],
                          dtype=np.int32)
for label, tree, out_name in [
    ("default context tree", ctx_default, "tree_context_default.png"),
    ("CU context tree",      ctx_cu,      "tree_context_cu.png"),
]:
    nodes, c_of, _ = make_static_layout(tree.root, TREE_VIZ_DEPTH)
    counts = count_single_label_at_nodes(nodes, c_of, context_train_tokens,
                                          ctx_train_lbls, descend_idx_disc,
                                          TREE_VIZ_DEPTH, N_POS)
    plot_single_label_tree_idx(
        c_of, counts,
        f"{label} — top {TREE_VIZ_DEPTH+1} levels  (POS distribution)",
        os.path.join(TREE_DIR, out_name), TREE_VIZ_DEPTH)
    print(f"  {out_name}")

# Content trees: greedy-descend on pair instances → L/R POS bars
print("Rendering content-tree visualisations …")
for label, tree, train_data, descend_fn, out_name in [
    ("TopK-Cont default", tree_tkc_def, cont_def_tr,
     descend_idx_cont_default, "tree_content_topk_cont_default.png"),
    ("TopK-Cont CU",      tree_tkc_cu,  cont_cu_tr,
     descend_idx_cont_cu,      "tree_content_topk_cont_cu.png"),
    ("TopK-Disc-Cnt1 default", tree_tkdc1_def, disc_def_tr,
     descend_idx_disc,             "tree_content_topk_disc_cnt1_default.png"),
    ("TopK-Disc-Cnt1 CU",      tree_tkdc1_cu,  disc_cu_tr,
     descend_idx_disc,             "tree_content_topk_disc_cnt1_cu.png"),
]:
    nodes, c_of, _ = make_static_layout(tree.root, TREE_VIZ_DEPTH)
    cL, cR = count_pair_label_at_nodes(nodes, c_of, train_data, y_train,
                                        descend_fn, TREE_VIZ_DEPTH)
    plot_pair_label_tree_idx(
        c_of, cL, cR,
        f"{label} content tree — top {TREE_VIZ_DEPTH+1} levels  (L/R POS)",
        os.path.join(TREE_DIR, out_name), TREE_VIZ_DEPTH)
    print(f"  {out_name}")


# ── Evaluation ───────────────────────────────────────────────────────────────
CLASSES = sorted(set(y_train.tolist()) | set(y_test.tolist()))
N_CLS   = len(CLASSES)


def linear_probe_per_class(Z_tr, y_tr, Z_te, y_te):
    lin = LinearSVC(max_iter=4000)
    lin.fit(Z_tr, y_tr)
    overall = lin.score(Z_te, y_te)
    per_cls = np.array([
        lin.score(Z_te[y_te == c], y_te[y_te == c]) if (y_te == c).sum() > 0 else 0.0
        for c in CLASSES])
    return overall, per_cls


def knn_accuracy_vs_k(Z_tr, y_tr, Z_te, y_te, ks=KNN_KS):
    return [KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y_tr).score(Z_te, y_te) for k in ks]


def _repr_stats(Z):
    nz = (Z != 0)
    return nz.sum(axis=1).mean(), (~nz.any(axis=0)).mean() * 100


def softmax_entropy(Z):
    shifted = Z - Z.max(axis=1, keepdims=True)
    expZ    = np.exp(shifted)
    p       = expZ / expZ.sum(axis=1, keepdims=True)
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)


print("\nEvaluating …")
results = {}
for short, lbl, Ztr, Zte in [
    ("tkc_def",   f"TopK-Cont default   (in={cont_def_tr.shape[1]}, k={TOP_K})",
     Z_tkc_def_tr,   Z_tkc_def_te),
    ("tkc_cu",    f"TopK-Cont CU        (in={cont_cu_tr.shape[1]}, k={TOP_K}, scaling={SCALING})",
     Z_tkc_cu_tr,    Z_tkc_cu_te),
    ("tkdc1_def", f"TopK-Disc-Cnt1 default (k={TOP_K})",
     Z_tkdc1_def_tr, Z_tkdc1_def_te),
    ("tkdc1_cu",  f"TopK-Disc-Cnt1 CU      (k={TOP_K})",
     Z_tkdc1_cu_tr,  Z_tkdc1_cu_te),
]:
    print(f"  evaluating {lbl} …")
    lin_overall, lin_per = linear_probe_per_class(Ztr, y_train, Zte, y_test)
    knn_accs             = knn_accuracy_vs_k(Ztr, y_train, Zte, y_test)
    avg_l0, dead_pct     = _repr_stats(Ztr)
    avg_ent              = softmax_entropy(Ztr).mean()
    results[short] = dict(label=lbl, lin_overall=lin_overall, lin_per=lin_per,
                          knn_accs=knn_accs, Ztr=Ztr, Zte=Zte,
                          avg_l0=avg_l0, dead_pct=dead_pct, avg_ent=avg_ent)


print(f"\n  {'Method':<60} {'Lin.Probe':>10} {'KNN@5':>7} {'Avg L0':>8} {'Dead%':>7}")
print(f"  {'-'*100}")
_knn5 = KNN_KS.index(5)
_rows = []
for short, r in results.items():
    knn5 = r["knn_accs"][_knn5] * 100
    print(f"  {r['label']:<60} {r['lin_overall']*100:>9.1f}% "
          f"{knn5:>6.1f}% {r['avg_l0']:>8.1f} {r['dead_pct']:>6.1f}%  ent={r['avg_ent']:.3f}")
    _rows.append({"method": r["label"], "lin_probe_pct": round(r["lin_overall"] * 100, 2),
                  "knn5_pct": round(knn5, 2), "avg_l0": round(float(r["avg_l0"]), 2),
                  "dead_pct": round(float(r["dead_pct"]), 2),
                  "avg_entropy": round(float(r["avg_ent"]), 4)})

_csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(_csv_path, "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=["method", "lin_probe_pct", "knn5_pct",
                                         "avg_l0", "dead_pct", "avg_entropy"])
    _w.writeheader()
    _w.writerows(_rows)
print(f"  Summary saved → {_csv_path}")


# ── Combined plots ───────────────────────────────────────────────────────────
METHODS = [
    (Z_tkc_def_tr,   Z_tkc_def_te,   results["tkc_def"]["lin_per"],
     results["tkc_def"]["knn_accs"],   results["tkc_def"]["label"],   "o-", "#4878d0"),
    (Z_tkc_cu_tr,    Z_tkc_cu_te,    results["tkc_cu"]["lin_per"],
     results["tkc_cu"]["knn_accs"],    results["tkc_cu"]["label"],    "s-", "#d65f5f"),
    (Z_tkdc1_def_tr, Z_tkdc1_def_te, results["tkdc1_def"]["lin_per"],
     results["tkdc1_def"]["knn_accs"], results["tkdc1_def"]["label"], "v-", "#6acc65"),
    (Z_tkdc1_cu_tr,  Z_tkdc1_cu_te,  results["tkdc1_cu"]["lin_per"],
     results["tkdc1_cu"]["knn_accs"],  results["tkdc1_cu"]["label"],  "P-", "#956cb4"),
]
n_meth = len(METHODS)

CMAP_PAIR = plt.get_cmap("tab20") if N_CLS <= 20 else plt.get_cmap("hsv")
def _pair_color(i):
    return CMAP_PAIR(i / max(N_CLS - 1, 1))


def _pair_label_str(c):
    l, r = split_pair_label(c)
    return f"{id2pos[l]}-{id2pos[r]}"


class_freq  = np.array([(y_train == c).sum() for c in CLASSES])
_top_legend = np.argsort(-class_freq)[:min(15, N_CLS)]
_leg = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=_pair_color(i),
                    markersize=6, label=_pair_label_str(CLASSES[i]))
        for i in _top_legend]

print("\nComputing UMAP projections …")
projs_umap = [UMAP(n_components=2, random_state=SEED).fit_transform(Z)
              for Z, _, _, _, _, _, _ in METHODS]
print("Computing t-SNE projections …")
projs_tsne = [TSNE(n_components=2, random_state=SEED, n_jobs=-1).fit_transform(Z)
              for Z, _, _, _, _, _, _ in METHODS]


def _scatter(projs, suptitle, out_path):
    fig, axes = plt.subplots(1, n_meth, figsize=(n_meth * 5, 5))
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    for ax, Z2, (_, _, _, _, lbl, _, _) in zip(axes, projs, METHODS):
        for ci, c in enumerate(CLASSES):
            mask = y_train == c
            if mask.any():
                ax.scatter(Z2[mask, 0], Z2[mask, 1], color=_pair_color(ci),
                           alpha=0.45, s=6)
        ax.set_title(lbl, fontsize=8)
        ax.set_xlabel("Dim 1", fontsize=7); ax.set_ylabel("Dim 2", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.legend(handles=_leg, title="POS-pair (top freq)",
               loc="center right", bbox_to_anchor=(1.0, 0.5),
               ncol=1, fontsize=6, title_fontsize=7, frameon=True)
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


_scatter(projs_umap, "UMAP — Grammar Chunking (default vs CU)",
         os.path.join(OUT_DIR, "scatter_umap.png"))
_scatter(projs_tsne, "t-SNE — Grammar Chunking (default vs CU)",
         os.path.join(OUT_DIR, "scatter_tsne.png"))

# Per-class lin probe
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
ax.set_title("Linear Probe — Per-(L,R)-POS Test Accuracy  (Grammar Chunking, default vs CU)")
ax.set_ylim(0, 115); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "linear_probe_per_class.png"), dpi=120)
plt.close()

# KNN vs k
fig, ax = plt.subplots(figsize=(7, 5))
for _, _, _, knn_accs, lbl, marker, color in METHODS:
    ax.plot(KNN_KS, [a * 100 for a in knn_accs], marker, label=lbl, color=color)
ax.set_xlabel("k"); ax.set_ylabel("Test Accuracy %")
ax.set_title("KNN Test Accuracy vs k  (Grammar Chunking, default vs CU)")
ax.set_xticks(KNN_KS); ax.set_ylim(0, 105); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "knn_vs_k.png"), dpi=120)
plt.close()

# Dim-activity (4-panel)
_n_cols = 2; _n_rows = 2
fig, axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 3.8))
axes = axes.flatten()
for idx, (Z_tr, _, _, _, lbl, _, color) in enumerate(METHODS):
    ax        = axes[idx]
    fire_freq = (Z_tr != 0).mean(axis=0)
    avg_ent   = softmax_entropy(Z_tr).mean()
    ax.hist(fire_freq, bins=60, color=color, alpha=0.82, edgecolor="white", linewidth=0.3)
    ax.axvline(fire_freq.mean(), color="black", linewidth=1.0, linestyle="--",
               label=f"mean={fire_freq.mean():.3f}")
    ax.set_title(f"{lbl}\nAvg softmax entropy: {avg_ent:.3f} nats", fontsize=8)
    ax.set_xlabel("Dimension fire frequency", fontsize=7)
    ax.set_ylabel("# dimensions", fontsize=7)
    ax.legend(fontsize=6); ax.tick_params(labelsize=6)
fig.suptitle("Per-dimension Activity Frequency  —  Grammar Chunking", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dim_activity_hist.png"), dpi=120, bbox_inches="tight")
plt.close()

# Competitive-node histograms
def _competitive_node_fig(threshold, fname):
    pct_str = f"{int(threshold * 100)}%"
    fig, axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 3.8))
    axes = axes.flatten()
    for idx, (Z_tr, _, _, _, lbl, _, color) in enumerate(METHODS):
        ax       = axes[idx]
        Z_f      = Z_tr.astype(np.float64)
        best     = Z_f.max(axis=1, keepdims=True)
        rel_prob = np.exp(np.clip(Z_f - best, -500, 0))
        counts   = (rel_prob >= threshold).sum(axis=1)
        max_c    = max(int(counts.max()), 1)
        bins     = np.arange(0.5, max_c + 1.5, 1.0)
        ax.hist(counts, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.axvline(counts.mean(), color="black", linewidth=1.2, linestyle="--",
                   label=f"mean={counts.mean():.2f}")
        ax.set_title(f"{lbl}\nmedian={np.median(counts):.0f} | mean={counts.mean():.2f}",
                     fontsize=8)
        ax.set_xlabel(f"# nodes with rel-prob ≥ {pct_str} of best", fontsize=7)
        ax.set_ylabel("# pairs", fontsize=7)
        ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    fig.suptitle(f"Competitive-node count per pair  (rel-prob ≥ {pct_str})", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches="tight")
    plt.close()


_competitive_node_fig(0.90, "competitive_node_hist_90pct.png")
_competitive_node_fig(0.75, "competitive_node_hist_75pct.png")
_competitive_node_fig(0.50, "competitive_node_hist_50pct.png")

print(f"\nAll outputs written to: {OUT_DIR}")
