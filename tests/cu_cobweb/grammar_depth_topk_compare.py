"""
Grammar — Cobweb-Depth-TopK: default (info-CU) vs CU (Fisher ECG)

Mirrors tests/moc/grammar_example.py's Cobweb-Depth-TopK setup, but builds
TWO CobwebDiscreteTree variants from the IDENTICAL context-instance stream:

  default: cobweb.cobweb_discrete.CobwebDiscreteTree
  CU     : cobweb.cu_cobweb_discrete.CUCobwebDiscreteTree

Both share alpha=1e-3 and weight_attr=True so the difference is purely the
partition-utility heuristic (Laplace-smoothed info CU vs Fisher ECG).

Outputs (tests/cu_cobweb/grammar_output/):
  - tree_default_top4.png / tree_cu_top4.png   POS distributions at top 4 levels
  - bfs_subtree_default.png / bfs_subtree_cu.png  the BFS-DZ feature-unit subtree
  - scatter_umap.png / scatter_tsne.png          embedding visualisations
  - linear_probe_per_class.png                   per-POS bar chart
  - knn_vs_k.png                                  KNN curves
  - dim_activity_hist.png                         per-dim fire-freq histograms
  - competitive_node_hist_*pct.png                competitive-node count histograms
  - summary.csv
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

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from util.cfg import TEST_GRAMMAR3, generate

from cobweb.cobweb_discrete    import CobwebDiscreteTree
from cobweb.cu_cobweb_discrete import CUCobwebDiscreteTree

OUT_DIR = os.path.join(_HERE, "grammar_output")
ARR_DIR = os.path.join(OUT_DIR, "arrays")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARR_DIR, exist_ok=True)

# ── Constants (match tests/moc/grammar_example.py) ───────────────────────────
N_SENTENCES = 1000
WINDOW      = 3
DZ          = 128
TOP_K       = 10
SEED        = 42
KNN_KS      = [1, 3, 5, 10, 20, 50]
ALPHA       = 1e-3
WEIGHT_ATTR = True
TREE_VIZ_DEPTH    = 4
BFS_SUBTREE_NODES = 64

random.seed(SEED)
np.random.seed(SEED)

# ── Generate corpus + label tokens by POS ────────────────────────────────────
print(f"Generating {N_SENTENCES} sentences from TEST_GRAMMAR3 …")
sentences = []
for _ in range(N_SENTENCES):
    sent = [w for w in generate("S", TEST_GRAMMAR3).split() if w]
    if sent:
        sentences.append(sent)

all_tokens = [w for sent in sentences for w in sent]
vocab      = sorted(set(all_tokens))
word2id    = {w: i for i, w in enumerate(vocab)}
print(f"  Vocab: {len(vocab)}  |  Tokens: {len(all_tokens)}")

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
print(f"  POS tags ({N_POS}): {pos_tags}")

CONTEXT_OFFSETS = [p for p in range(-WINDOW, WINDOW + 1) if p != 0]
pos2attr        = {p: i for i, p in enumerate(CONTEXT_OFFSETS)}


def make_context_instance(sent, pos):
    inst = {}
    for offset in CONTEXT_OFFSETS:
        ctx = pos + offset
        if 0 <= ctx < len(sent):
            inst[pos2attr[offset]] = {word2id[sent[ctx]]: 1.0}
    return inst


instances_raw, labels_all = [], []
for sent in sentences:
    for p, w in enumerate(sent):
        instances_raw.append(make_context_instance(sent, p))
        labels_all.append(pos2id[word2pos[w]])
labels_all = np.array(labels_all, dtype=np.int32)

rng       = np.random.default_rng(SEED)
idx       = rng.permutation(len(instances_raw))
split     = int(0.8 * len(instances_raw))
train_idx = idx[:split]
test_idx  = idx[split:]

instances_train = [instances_raw[i] for i in train_idx]
instances_test  = [instances_raw[i] for i in test_idx]
y               = labels_all[train_idx]
y_test          = labels_all[test_idx]
print(f"  Train: {len(instances_train)}  Test: {len(instances_test)}")

np.save(os.path.join(ARR_DIR, "y_train.npy"), y)
np.save(os.path.join(ARR_DIR, "y_test.npy"),  y_test)

# ── Build BOTH trees from the IDENTICAL stream ───────────────────────────────
print(f"\nBuilding default CobwebDiscreteTree (alpha={ALPHA}) …")
tree_default = CobwebDiscreteTree(alpha=ALPHA, weight_attr=WEIGHT_ATTR)
for i, inst in enumerate(instances_train):
    tree_default.ifit(inst)
    if (i + 1) % 1000 == 0:
        print(f"  default: {i+1}/{len(instances_train)}")
print("  default tree built.")

print(f"\nBuilding CUCobwebDiscreteTree (alpha={ALPHA}) …")
tree_cu = CUCobwebDiscreteTree(alpha=ALPHA, weight_attr=WEIGHT_ATTR)
for i, inst in enumerate(instances_train):
    tree_cu.ifit(inst)
    if (i + 1) % 1000 == 0:
        print(f"  CU: {i+1}/{len(instances_train)}")
print("  CU tree built.")


# ── Tree helpers ─────────────────────────────────────────────────────────────
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


# ── Encoding ─────────────────────────────────────────────────────────────────
# Both default and CU discrete share node.log_prob_instance(inst) API.
def encode_logpost(instances, nodes):
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, inst in enumerate(instances):
            out[i, j] = node.log_prob_instance(inst)
    return out


def topk_sparsify(Z, k):
    out = np.zeros_like(Z)
    k   = min(k, Z.shape[1])
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    rows    = np.arange(Z.shape[0])[:, None]
    out[rows, top_idx] = Z[rows, top_idx]
    return out


def build_depth_topk(tree, label):
    bfs_nodes = bfs_first_n(tree.root, DZ)
    if not bfs_nodes:
        bfs_nodes = [tree.root]
    print(f"  [{label}] BFS-{DZ}: {len(bfs_nodes)} nodes")
    raw_tr = encode_logpost(instances_train, bfs_nodes)
    raw_te = encode_logpost(instances_test,  bfs_nodes)
    sc     = StandardScaler().fit(raw_tr)
    Z_tr   = topk_sparsify(sc.transform(raw_tr), TOP_K)
    Z_te   = topk_sparsify(sc.transform(raw_te), TOP_K)
    return Z_tr, Z_te, bfs_nodes


print("\nEncoding Depth-TopK features …")
Z_def_tr, Z_def_te, bfs_default = build_depth_topk(tree_default, "default")
Z_cu_tr,  Z_cu_te,  bfs_cu      = build_depth_topk(tree_cu,      "CU")

np.save(os.path.join(ARR_DIR, "Z_default_train.npy"), Z_def_tr)
np.save(os.path.join(ARR_DIR, "Z_default_test.npy"),  Z_def_te)
np.save(os.path.join(ARR_DIR, "Z_cu_train.npy"),      Z_cu_tr)
np.save(os.path.join(ARR_DIR, "Z_cu_test.npy"),       Z_cu_te)


# ── Tree visualisation: top-N levels with POS bars ───────────────────────────
# IMPORTANT: cobweb's node.children returns fresh Python wrappers each call —
# id() is NOT stable across iterations.  We materialise wrappers once into a
# list (which keeps them alive) and key everything off integer indices.

CMAP_POS   = plt.get_cmap("tab10") if N_POS <= 10 else plt.get_cmap("tab20")
pos_colors = [CMAP_POS(i / max(N_POS - 1, 1)) for i in range(N_POS)]


def make_static_layout(root, max_depth):
    """One BFS that materialises wrappers; returns (all_nodes, children_of, depth_of)."""
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


def bfs_static_layout(root, n_nodes):
    """Like make_static_layout but bounded by n_nodes non-root nodes."""
    all_nodes   = [root]
    children_of = {0: []}
    depth_of    = {0: 0}
    queue       = [0]
    while queue and len(all_nodes) - 1 < n_nodes:
        idx  = queue.pop(0)
        node = all_nodes[idx]
        for c in node.children:
            if len(all_nodes) - 1 >= n_nodes:
                break
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


def count_labels_idx(all_nodes, children_of, instances, y_lbl, max_depth):
    counts = {}
    for inst, label in zip(instances, y_lbl):
        for idx in descend_idx_disc(all_nodes, children_of, inst, max_depth):
            if idx not in counts:
                counts[idx] = np.zeros(N_POS, dtype=np.int32)
            counts[idx][int(label)] += 1
    return counts


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


def plot_tree_top_levels_idx(children_of, counts, max_depth, title, out_path):
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


def plot_bfs_subtree_idx(all_nodes, children_of, depth_of, title, out_path):
    """Render the materialised BFS subtree as a node-link diagram."""
    if len(all_nodes) <= 1:
        return
    layers = {}
    for idx, d in depth_of.items():
        layers.setdefault(d, []).append(idx)
    max_d = max(layers)

    pos = {}
    for d in sorted(layers):
        for i, idx in enumerate(layers[d]):
            pos[idx] = ((i + 0.5) / len(layers[d]), d)

    parent = {}   # child_idx -> parent_idx
    for par_idx, ch in children_of.items():
        for c in ch:
            parent[c] = par_idx

    counts_list = [all_nodes[i].count for i in range(len(all_nodes))]
    cmax = max(counts_list); cmin = max(1, min(counts_list))

    fig, ax = plt.subplots(figsize=(max(12, len(all_nodes) * 0.18), (max_d + 1) * 1.5))
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, max_d + 0.5)
    ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=11)

    for c_idx, p_idx in parent.items():
        px, py = pos[p_idx]
        cx, cy = pos[c_idx]
        ax.plot([px, cx], [py, cy], color="gray", lw=0.5, zorder=0)
    for idx, (x, yv) in pos.items():
        cnt = all_nodes[idx].count
        sz  = 30 + 80 * (np.log1p(cnt) - np.log1p(cmin)) / \
                       max(1.0, np.log1p(cmax) - np.log1p(cmin))
        ax.scatter([x], [yv], s=sz, color="#4878d0", edgecolor="black", lw=0.4, zorder=2)
    for d in sorted(layers):
        ax.text(-0.01, d, f"d={d}", ha="right", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


print("\nComputing greedy label distributions …")
nodes_def, c_of_def, _ = make_static_layout(tree_default.root, TREE_VIZ_DEPTH)
nodes_cu,  c_of_cu,  _ = make_static_layout(tree_cu.root,      TREE_VIZ_DEPTH)
counts_def = count_labels_idx(nodes_def, c_of_def, instances_train, y, TREE_VIZ_DEPTH)
counts_cu  = count_labels_idx(nodes_cu,  c_of_cu,  instances_train, y, TREE_VIZ_DEPTH)

plot_tree_top_levels_idx(c_of_def, counts_def, TREE_VIZ_DEPTH,
                          f"Default CobwebDiscreteTree — top {TREE_VIZ_DEPTH+1} levels  "
                          f"(alpha={ALPHA})",
                          os.path.join(OUT_DIR, "tree_default_top4.png"))
plot_tree_top_levels_idx(c_of_cu, counts_cu, TREE_VIZ_DEPTH,
                          f"CU CobwebDiscreteTree — top {TREE_VIZ_DEPTH+1} levels  "
                          f"(alpha={ALPHA})",
                          os.path.join(OUT_DIR, "tree_cu_top4.png"))


# ── BFS-DZ subtree viz (feature units of Depth-TopK) ─────────────────────────
sub_nodes_def, sub_c_def, sub_d_def = bfs_static_layout(tree_default.root, BFS_SUBTREE_NODES)
sub_nodes_cu,  sub_c_cu,  sub_d_cu  = bfs_static_layout(tree_cu.root,      BFS_SUBTREE_NODES)

plot_bfs_subtree_idx(sub_nodes_def, sub_c_def, sub_d_def,
                      f"Default tree — BFS-{BFS_SUBTREE_NODES} subtree (Depth-TopK feature units)",
                      os.path.join(OUT_DIR, "bfs_subtree_default.png"))
plot_bfs_subtree_idx(sub_nodes_cu, sub_c_cu, sub_d_cu,
                      f"CU tree — BFS-{BFS_SUBTREE_NODES} subtree (Depth-TopK feature units)",
                      os.path.join(OUT_DIR, "bfs_subtree_cu.png"))


# ── Evaluation ───────────────────────────────────────────────────────────────
CLASSES = sorted(set(y.tolist()))


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
def_lin, def_per = linear_probe_per_class(Z_def_tr, y, Z_def_te, y_test)
cu_lin,  cu_per  = linear_probe_per_class(Z_cu_tr,  y, Z_cu_te,  y_test)
def_knn = knn_accuracy_vs_k(Z_def_tr, y, Z_def_te, y_test)
cu_knn  = knn_accuracy_vs_k(Z_cu_tr,  y, Z_cu_te,  y_test)

print(f"\n  {'Method':<50} {'Lin.Probe':>10} {'KNN@5':>7} {'Avg L0':>8} {'Dead%':>7}")
print(f"  {'-'*90}")
_knn5 = KNN_KS.index(5)
_rows = []
for label, overall, Z_tr, knn_accs in [
    (f"Default Cobweb-Depth-TopK ({len(bfs_default)}d, k={TOP_K})",
     def_lin, Z_def_tr, def_knn),
    (f"CU      Cobweb-Depth-TopK ({len(bfs_cu)}d, k={TOP_K})",
     cu_lin,  Z_cu_tr,  cu_knn),
]:
    avg_l0, dead_pct = _repr_stats(Z_tr)
    avg_ent          = softmax_entropy(Z_tr).mean()
    knn5             = knn_accs[_knn5] * 100
    print(f"  {label:<50} {overall*100:>9.1f}% {knn5:>6.1f}% {avg_l0:>8.1f} {dead_pct:>6.1f}%  ent={avg_ent:.3f}")
    _rows.append({"method": label, "lin_probe_pct": round(overall * 100, 2),
                  "knn5_pct": round(knn5, 2), "avg_l0": round(float(avg_l0), 2),
                  "dead_pct": round(float(dead_pct), 2),
                  "avg_entropy": round(float(avg_ent), 4)})

_csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(_csv_path, "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=["method", "lin_probe_pct", "knn5_pct",
                                         "avg_l0", "dead_pct", "avg_entropy"])
    _w.writeheader()
    _w.writerows(_rows)
print(f"  Summary saved → {_csv_path}")


# ── Combined plots ───────────────────────────────────────────────────────────
METHODS = [
    (Z_def_tr, Z_def_te, def_per, def_knn,
     f"Default ({len(bfs_default)}d, k={TOP_K})", "o-", "#4878d0"),
    (Z_cu_tr,  Z_cu_te,  cu_per,  cu_knn,
     f"CU ({len(bfs_cu)}d, k={TOP_K})",            "s-", "#d65f5f"),
]

_leg = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=pos_colors[i], markersize=7, label=id2pos[i])
        for i in range(N_POS)]

print("Computing UMAP …")
projs_umap = [UMAP(n_components=2, random_state=SEED).fit_transform(Z)
              for Z, _, _, _, _, _, _ in METHODS]
print("Computing t-SNE …")
projs_tsne = [TSNE(n_components=2, random_state=SEED, n_jobs=-1).fit_transform(Z)
              for Z, _, _, _, _, _, _ in METHODS]


def _scatter(projs, suptitle, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    for ax, Z2, (_, _, _, _, lbl, _, _) in zip(axes, projs, METHODS):
        for ci, c in enumerate(CLASSES):
            mask = y == c
            ax.scatter(Z2[mask, 0], Z2[mask, 1], color=pos_colors[ci],
                       alpha=0.5, s=6)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    fig.legend(handles=_leg, title="POS", loc="center right",
               bbox_to_anchor=(1.0, 0.5), frameon=True)
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


_scatter(projs_umap, "UMAP — Cobweb-Depth-TopK (default vs CU)",
         os.path.join(OUT_DIR, "scatter_umap.png"))
_scatter(projs_tsne, "t-SNE — Cobweb-Depth-TopK (default vs CU)",
         os.path.join(OUT_DIR, "scatter_tsne.png"))

# Per-class lin probe
w_bar = 0.38
x_bar = np.arange(len(CLASSES))
fig, ax = plt.subplots(figsize=(max(10, N_POS * 1.5), 5))
for (_, _, per, _, lbl, _, color), offset in zip(METHODS, [-w_bar/2, w_bar/2]):
    ax.bar(x_bar + offset, per * 100, w_bar, label=lbl, color=color, alpha=0.85)
ax.set_xticks(x_bar)
ax.set_xticklabels([id2pos[c] for c in CLASSES], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Test Accuracy %")
ax.set_title("Linear Probe — Per-POS Test Accuracy  (default vs CU)")
ax.set_ylim(0, 115); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "linear_probe_per_class.png"), dpi=120)
plt.close()

# KNN vs k
fig, ax = plt.subplots(figsize=(7, 5))
for _, _, _, knn_accs, lbl, marker, color in METHODS:
    ax.plot(KNN_KS, [a * 100 for a in knn_accs], marker, label=lbl, color=color)
ax.set_xlabel("k (number of neighbours)"); ax.set_ylabel("Test Accuracy %")
ax.set_title("KNN Test Accuracy vs k (default vs CU)")
ax.set_xticks(KNN_KS); ax.set_ylim(0, 105); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "knn_vs_k.png"), dpi=120)
plt.close()

# Dim-activity
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (Z_tr, _, _, _, lbl, _, color) in zip(axes, METHODS):
    fire_freq = (Z_tr != 0).mean(axis=0)
    avg_ent   = softmax_entropy(Z_tr).mean()
    ax.hist(fire_freq, bins=60, color=color, alpha=0.82, edgecolor="white", linewidth=0.3)
    ax.axvline(fire_freq.mean(), color="black", linewidth=1.0, linestyle="--",
               label=f"mean={fire_freq.mean():.3f}")
    ax.set_title(f"{lbl}\nAvg softmax entropy: {avg_ent:.3f} nats", fontsize=9)
    ax.set_xlabel("Dimension fire frequency", fontsize=8)
    ax.set_ylabel("# dimensions", fontsize=8)
    ax.legend(fontsize=7)
fig.suptitle("Per-dimension Activity Frequency  —  Grammar  (default vs CU)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dim_activity_hist.png"), dpi=120, bbox_inches="tight")
plt.close()

# Competitive-node histograms
def _competitive_node_fig(threshold, fname):
    pct_str = f"{int(threshold * 100)}%"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (Z_tr, _, _, _, lbl, _, color) in zip(axes, METHODS):
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
                     fontsize=9)
        ax.set_xlabel(f"# nodes with rel-prob ≥ {pct_str} of best", fontsize=8)
        ax.set_ylabel("# tokens", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(f"Competitive-node count per token — rel-prob ≥ {pct_str}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches="tight")
    plt.close()


_competitive_node_fig(0.90, "competitive_node_hist_90pct.png")
_competitive_node_fig(0.75, "competitive_node_hist_75pct.png")
_competitive_node_fig(0.50, "competitive_node_hist_50pct.png")

print(f"\nAll outputs written to: {OUT_DIR}")
