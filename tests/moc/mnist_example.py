"""
MNIST Representation Learning
Learn representations via PCA and Autoencoder, then evaluate with
linear probing, KNN, and visualize results.

What I have learned so far:
- covar_from = 2 provides good linear probing scores but worse KNN
- prior_var REALLY matters (like it REALLY REALLY MATTERS)
- sparsity benefits us slightly!! log-prob signals of some prototypes is like noise
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP
from cobweb.cobweb_continuous import CobwebContinuousTree

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "mnist_output")
DATA_DIR = os.path.join(OUT_DIR, "data")
ARR_DIR  = os.path.join(OUT_DIR, "arrays")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARR_DIR,  exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

def to_numpy(dataset, n):
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    imgs, lbls = next(iter(loader))
    return imgs.view(n, -1).numpy(), lbls.numpy()

X,      y      = to_numpy(trainset, 10_000)
X_test, y_test = to_numpy(testset,   2_000)

# ── PCA ───────────────────────────────────────────────────────────────────────

DZ    = 64   # representation dimensionality
TOP_K =  32   # fixed top-k per-instance sparsification
TOP_PCT = 0.25 # fraction of activations to keep (top-pct variant)
pca = PCA(n_components=DZ)
Z_pca      = pca.fit_transform(X)
Z_pca_test = pca.transform(X_test)

# ── Autoencoder ───────────────────────────────────────────────────────────────

class AE(nn.Module):
    def __init__(self, dz=64, input_dim=784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),       nn.ReLU(),
            nn.Linear(256, dz),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dz, 256),       nn.ReLU(),
            nn.Linear(256, 512),      nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return F.mse_loss(self.decoder(self.encoder(x)), x)


def train_ae(model, X_train, epochs=20, batch_size=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    model.train()
    for ep in range(epochs):
        total = sum(
            (opt.zero_grad(), loss := model(batch), loss.backward(), opt.step(), loss.item())[-1]
            for (batch,) in loader
        )
        print(f"  epoch {ep+1:02d}/{epochs}  loss={total/len(loader):.4f}")
    model.eval()
    return model


print("Training AE …")
ae = train_ae(AE(dz=DZ), X, epochs=20)

with torch.no_grad():
    Z_ae      = ae.encoder(torch.tensor(X,      dtype=torch.float32)).numpy()
    Z_ae_test = ae.encoder(torch.tensor(X_test, dtype=torch.float32)).numpy()

# ── Save data arrays ──────────────────────────────────────────────────────────

np.save(os.path.join(ARR_DIR, "X_train.npy"),      X)
np.save(os.path.join(ARR_DIR, "y_train.npy"),      y)
np.save(os.path.join(ARR_DIR, "X_test.npy"),       X_test)
np.save(os.path.join(ARR_DIR, "y_test.npy"),       y_test)
np.save(os.path.join(ARR_DIR, "Z_pca_train.npy"),  Z_pca)
np.save(os.path.join(ARR_DIR, "Z_pca_test.npy"),   Z_pca_test)
np.save(os.path.join(ARR_DIR, "Z_ae_train.npy"),   Z_ae)
np.save(os.path.join(ARR_DIR, "Z_ae_test.npy"),    Z_ae_test)
torch.save(ae.state_dict(), os.path.join(ARR_DIR, "ae_weights.pt"))
print("Data arrays saved.")

# ── Cobweb: raw-input setup ───────────────────────────────────────────────────

X_cob = X.astype(np.float32, copy=False)
X_cob_test = X_test.astype(np.float32, copy=False)

# ── Cobweb: build tree ────────────────────────────────────────────────────────

print("Building Cobweb tree …")
cobweb_tree = CobwebContinuousTree(
    size=X_cob.shape[1],
    covar_from=2,
    prior_var=1 / 784,
    num_labels=0
)
_empty_label = np.zeros(0, dtype=np.float32)
for i, x in enumerate(X_cob):
    cobweb_tree.ifit(x, _empty_label)
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(X_cob)} inserted")
print("  Tree built.")

# ── Cobweb: centroid extraction ───────────────────────────────────────────────

def collect_by_depth_nodes(root):
    """Return {depth: [node objects]} for every node in the tree."""
    by_depth = {}
    queue = [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        by_depth.setdefault(d, []).append(node)
        for child in node.children:
            queue.append((child, d + 1))
    return by_depth

def bfs_first_n_nodes(root, n):
    """Collect the first n node objects via BFS (excluding root)."""
    nodes, queue = [], [root]
    while queue and len(nodes) < n:
        node = queue.pop(0)
        for child in node.children:
            if len(nodes) > n:
                break
            nodes.append(child)
            queue.append(child)
    return nodes

print("Extracting BFS nodes …")
bfs_nodes = bfs_first_n_nodes(cobweb_tree.root, DZ)

print("Extracting static-depth nodes …")
by_depth_nodes = collect_by_depth_nodes(cobweb_tree.root)
depth_counts = {d: len(v) for d, v in by_depth_nodes.items()}
print(f"  Nodes per depth: {dict(sorted(depth_counts.items()))}")
# Deepest level that still has strictly fewer nodes than DZ
best_depth = 0
for d in sorted(depth_counts.keys()):
    if depth_counts[d] >= DZ:
        break
    best_depth = d
print(f"  Using depth {best_depth} ({depth_counts[best_depth]} nodes, target <{DZ})")
depth_nodes = by_depth_nodes[best_depth]
n_depth = len(depth_nodes)

# Top-K: first depth with >= DZ nodes, then keep top DZ by mean log-prob
topk_depth = next(
    d for d in sorted(depth_counts.keys()) if depth_counts[d] >= DZ
)
print(f"  Top-K pool: depth {topk_depth} ({depth_counts[topk_depth]} nodes)")
topk_pool_nodes = by_depth_nodes[topk_depth]

# Encode: log P(node | instance) for each centroid node
_empty = np.zeros(0, dtype=np.float32)

def encode_logpost(instances, nodes):
    """Return (n_samples, n_nodes) matrix of log P(node | x) = log P(x | node) + log P(node).

    Uses log_prob_class_given_instance, which adds the node prior log(count/root_count)
    to the joint log-density.  Values are still large negatives; call StandardScaler
    on the result before passing to classifiers.
    """
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, x in enumerate(instances):
            out[i, j] = node.log_prob_class_given_instance(x, _empty)
    return out

print("Encoding train set (BFS) …")
_scaler_bfs = StandardScaler()
Z_cob_bfs = _scaler_bfs.fit_transform(encode_logpost(X_cob, bfs_nodes))
print("Encoding test set (BFS) …")
Z_cob_bfs_test = _scaler_bfs.transform(encode_logpost(X_cob_test, bfs_nodes))
print("Encoding train set (Depth) …")
_scaler_dep = StandardScaler()
Z_cob_dep = _scaler_dep.fit_transform(encode_logpost(X_cob, depth_nodes))
print("Encoding test set (Depth) …")
Z_cob_dep_test = _scaler_dep.transform(encode_logpost(X_cob_test, depth_nodes))

# Top-K: encode all pool nodes, then per-instance zero out all but TOP_K highest
def topk_sparsify(Z, k):
    """Zero out all but the k largest values in each row."""
    out = np.zeros_like(Z)
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]   # (n, k) indices of top-k per row
    rows = np.arange(Z.shape[0])[:, None]
    out[rows, top_idx] = Z[rows, top_idx]
    return out

def pct_sparsify(Z, pct):
    """Per-instance: keep only the top `pct` fraction of activations by magnitude."""
    k_per_row = max(1, int(round(Z.shape[1] * pct)))
    return topk_sparsify(Z, k_per_row)

n_topk_pool = len(topk_pool_nodes)
print(f"  Top-K pool size: {n_topk_pool} nodes at depth {topk_depth}")
print("Encoding Top-K pool (train) …")
_scaler_topk = StandardScaler()
Z_topk_pool      = _scaler_topk.fit_transform(encode_logpost(X_cob, topk_pool_nodes))
Z_cob_topk       = topk_sparsify(Z_topk_pool, TOP_K)
print(f"  Applied per-instance top-{TOP_K} sparsification (dim={n_topk_pool})")
print("Encoding Top-K pool (test) …")
Z_topk_pool_test = _scaler_topk.transform(encode_logpost(X_cob_test, topk_pool_nodes))
Z_cob_topk_test  = topk_sparsify(Z_topk_pool_test, TOP_K)

print(f"  Applying per-instance top-{int(TOP_PCT*100)}% sparsification (dim={n_topk_pool}) …")
Z_cob_pct      = pct_sparsify(Z_topk_pool, TOP_PCT)
Z_cob_pct_test = pct_sparsify(Z_topk_pool_test, TOP_PCT)

# SelectTopK / SelectTopPct: choose the most-variable columns globally (by training
# variance before scaling) and keep only those — dense K-dim vectors, no zeros.
sel_topk_cols = np.argsort(_scaler_topk.var_)[-TOP_K:]
Z_cob_sel_topk      = Z_topk_pool[:, sel_topk_cols]
Z_cob_sel_topk_test = Z_topk_pool_test[:, sel_topk_cols]
print(f"  SelectTopK: selected {TOP_K} most-variable nodes (dim={TOP_K})")

_k_sel_pct = max(1, int(round(n_topk_pool * TOP_PCT)))
sel_pct_cols = np.argsort(_scaler_topk.var_)[-_k_sel_pct:]
Z_cob_sel_pct      = Z_topk_pool[:, sel_pct_cols]
Z_cob_sel_pct_test = Z_topk_pool_test[:, sel_pct_cols]
print(f"  SelectTopPct: selected {_k_sel_pct} most-variable nodes ({int(TOP_PCT*100)}% of {n_topk_pool}, dim={_k_sel_pct})")

np.save(os.path.join(ARR_DIR, "Z_cob_bfs_train.npy"),  Z_cob_bfs)
np.save(os.path.join(ARR_DIR, "Z_cob_bfs_test.npy"),   Z_cob_bfs_test)
np.save(os.path.join(ARR_DIR, "Z_cob_dep_train.npy"),  Z_cob_dep)
np.save(os.path.join(ARR_DIR, "Z_cob_dep_test.npy"),   Z_cob_dep_test)
np.save(os.path.join(ARR_DIR, "Z_cob_topk_train.npy"), Z_cob_topk)
np.save(os.path.join(ARR_DIR, "Z_cob_topk_test.npy"),  Z_cob_topk_test)
np.save(os.path.join(ARR_DIR, "Z_cob_pct_train.npy"),      Z_cob_pct)
np.save(os.path.join(ARR_DIR, "Z_cob_pct_test.npy"),       Z_cob_pct_test)
np.save(os.path.join(ARR_DIR, "Z_cob_sel_topk_train.npy"), Z_cob_sel_topk)
np.save(os.path.join(ARR_DIR, "Z_cob_sel_topk_test.npy"),  Z_cob_sel_topk_test)
np.save(os.path.join(ARR_DIR, "Z_cob_sel_pct_train.npy"),  Z_cob_sel_pct)
np.save(os.path.join(ARR_DIR, "Z_cob_sel_pct_test.npy"),   Z_cob_sel_pct_test)
print("Cobweb data saved.")

# ── Cobweb: tree visualisation (top 4 depths, label distributions) ────────────

def compute_node_label_counts(root, X_instances, y_labels, max_depth=3):
    """
    Greedy-descend each training sample through the tree up to max_depth,
    accumulating per-class counts at every ancestor node along the path.
    Returns: {id(node): np.ndarray shape (10,)}
    """
    n_classes = int(y_labels.max()) + 1
    counts = {}           # id(node) -> int array of shape (n_classes,)
    node_obj = {}         # id(node) -> node object (for drawing)

    def _ensure(node):
        nid = id(node)
        if nid not in counts:
            counts[nid] = np.zeros(n_classes, dtype=np.int32)
            node_obj[nid] = node
        return nid

    for x, label in zip(X_instances, y_labels):
        node = root
        for depth in range(max_depth + 1):
            _ensure(node)
            counts[id(node)][int(label)] += 1
            if not node.children or depth == max_depth:
                break
            # greedy: pick child with highest log_prob
            best_child = max(node.children,
                             key=lambda c: c.log_prob(x, _empty))
            node = best_child

    return counts, node_obj

print("Computing node label distributions (greedy descent) …")
label_counts_map, node_obj_map = compute_node_label_counts(
    cobweb_tree.root, X_cob, y, max_depth=3)

def plot_cobweb_tree_labels(root, label_counts_map, node_obj_map,
                            max_depth=3, out_path=None):
    """
    Render the Cobweb tree (top max_depth+1 levels) as a PNG.
    Each node is drawn as a stacked horizontal bar showing digit proportions.
    """
    tab10 = plt.get_cmap("tab10")
    digit_colors = [tab10(i) for i in range(10)]

    # ── 1. BFS to collect nodes at each depth with positional layout ──────────
    # layout: assign x-positions so subtrees don't overlap
    # We first compute the "leaf span" of each node (number of depth-max leaves
    # in its subtree), then use that to space things horizontally.

    def leaf_span(node, depth, max_depth):
        if depth >= max_depth or not node.children:
            return 1
        return sum(leaf_span(c, depth + 1, max_depth) for c in node.children)

    # Assign x positions via DFS
    pos = {}   # id(node) -> (x_centre, depth)

    def assign_pos(node, depth, x_left):
        span = leaf_span(node, depth, max_depth)
        x_centre = x_left + span / 2.0
        pos[id(node)] = (x_centre, depth)
        if depth < max_depth and node.children:
            cursor = x_left
            for child in node.children:
                child_span = leaf_span(child, depth + 1, max_depth)
                assign_pos(child, depth + 1, cursor)
                cursor += child_span
        return span

    assign_pos(root, 0, 0.0)
    total_width = leaf_span(root, 0, max_depth)

    # ── 2. Draw ───────────────────────────────────────────────────────────────
    bar_w   = 0.7          # fraction of slot width used by bar
    bar_h   = 0.35         # height of each bar in data coords
    y_gap   = 1.0          # vertical distance between depth levels

    fig_w  = max(14, total_width * 0.9)
    fig_h  = (max_depth + 1) * 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_width)
    ax.set_ylim(-0.7, max_depth * y_gap + 0.7)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"Cobweb Tree — Label Distributions (depths 0–{max_depth})",
                 fontsize=13)

    # draw edges first
    def draw_edges(node, depth):
        if depth >= max_depth or not node.children:
            return
        px, py = pos[id(node)]
        for child in node.children:
            cx, cy = pos[id(child)]
            ax.plot([px, cx],
                    [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.8, zorder=0)
            draw_edges(child, depth + 1)

    draw_edges(root, 0)

    # draw node bars
    def draw_node(node, depth):
        nid = id(node)
        if nid not in label_counts_map:
            return
        counts = label_counts_map[nid].astype(float)
        total  = counts.sum()
        if total == 0:
            return
        props = counts / total

        x_c, _ = pos[nid]
        x_left = x_c - bar_w / 2
        y_top  = depth * y_gap - bar_h / 2

        cursor = x_left
        for digit in range(10):
            seg_w = props[digit] * bar_w
            if seg_w > 0:
                rect = plt.Rectangle((cursor, y_top), seg_w, bar_h,
                                     color=digit_colors[digit], lw=0)
                ax.add_patch(rect)
                cursor += seg_w

        # border
        ax.add_patch(plt.Rectangle(
            (x_left, y_top), bar_w, bar_h,
            fill=False, edgecolor="black", lw=0.5))

        # dominant label text
        dom = int(counts.argmax())
        ax.text(x_c, depth * y_gap + bar_h / 2 + 0.05,
                f"n={int(total)}", ha="center", va="top", fontsize=5)

        if depth < max_depth and node.children:
            for child in node.children:
                draw_node(child, depth + 1)

    draw_node(root, 0)

    # legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=digit_colors[d], label=str(d))
        for d in range(10)
    ]
    ax.legend(handles=legend_handles, title="digit", loc="lower right",
              ncol=5, fontsize=7, title_fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

plot_cobweb_tree_labels(
    cobweb_tree.root, label_counts_map, node_obj_map,
    max_depth=3,
    out_path=os.path.join(OUT_DIR, "cobweb_tree_labels.png"),
)

# ── Evaluation ────────────────────────────────────────────────────────────────

CLASSES = list(range(10))
CMAP    = plt.get_cmap("tab10")
KNN_KS  = [1, 3, 5, 10, 20, 50]

def linear_probe_per_class(Z_tr, y_tr, Z_te, y_te):
    lin = LinearSVC(max_iter=2000)
    lin.fit(Z_tr, y_tr)
    overall   = lin.score(Z_te, y_te)
    per_class = np.array([lin.score(Z_te[y_te == c], y_te[y_te == c]) for c in CLASSES])
    return overall, per_class

def knn_accuracy_vs_k(Z_tr, y_tr, Z_te, y_te, ks=KNN_KS):
    return [KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y_tr).score(Z_te, y_te) for k in ks]

print("\nEvaluating …")
pca_lin_overall,     pca_lin_per     = linear_probe_per_class(Z_pca,      y, Z_pca_test,      y_test)
ae_lin_overall,      ae_lin_per      = linear_probe_per_class(Z_ae,       y, Z_ae_test,       y_test)
cob_bfs_lin_overall,  cob_bfs_lin_per  = linear_probe_per_class(Z_cob_bfs,  y, Z_cob_bfs_test,  y_test)
cob_dep_lin_overall,  cob_dep_lin_per  = linear_probe_per_class(Z_cob_dep,  y, Z_cob_dep_test,  y_test)
cob_topk_lin_overall,     cob_topk_lin_per     = linear_probe_per_class(Z_cob_topk,     y, Z_cob_topk_test,     y_test)
cob_pct_lin_overall,      cob_pct_lin_per      = linear_probe_per_class(Z_cob_pct,      y, Z_cob_pct_test,      y_test)
cob_sel_topk_lin_overall, cob_sel_topk_lin_per = linear_probe_per_class(Z_cob_sel_topk, y, Z_cob_sel_topk_test, y_test)
cob_sel_pct_lin_overall,  cob_sel_pct_lin_per  = linear_probe_per_class(Z_cob_sel_pct,  y, Z_cob_sel_pct_test,  y_test)

pca_knn_accs          = knn_accuracy_vs_k(Z_pca,          y, Z_pca_test,          y_test)
ae_knn_accs           = knn_accuracy_vs_k(Z_ae,           y, Z_ae_test,           y_test)
cob_bfs_knn_accs      = knn_accuracy_vs_k(Z_cob_bfs,      y, Z_cob_bfs_test,      y_test)
cob_dep_knn_accs      = knn_accuracy_vs_k(Z_cob_dep,      y, Z_cob_dep_test,      y_test)
cob_topk_knn_accs     = knn_accuracy_vs_k(Z_cob_topk,     y, Z_cob_topk_test,     y_test)
cob_pct_knn_accs      = knn_accuracy_vs_k(Z_cob_pct,      y, Z_cob_pct_test,      y_test)
cob_sel_topk_knn_accs = knn_accuracy_vs_k(Z_cob_sel_topk, y, Z_cob_sel_topk_test, y_test)
cob_sel_pct_knn_accs  = knn_accuracy_vs_k(Z_cob_sel_pct,  y, Z_cob_sel_pct_test,  y_test)

_k_kept = max(1, int(round(n_topk_pool * TOP_PCT)))
for name, overall in [(f"PCA ({DZ}d)",                                                          pca_lin_overall),
                      (f"AE  ({DZ}d)",                                                          ae_lin_overall),
                      (f"Cobweb-BFS  ({DZ}d)",                                                 cob_bfs_lin_overall),
                      (f"Cobweb-Depth (d={best_depth},{n_depth}d)",                             cob_dep_lin_overall),
                      (f"Cobweb-TopK     (depth={topk_depth},dim={n_topk_pool},k={TOP_K})",     cob_topk_lin_overall),
                      (f"Cobweb-Pct      (depth={topk_depth},dim={n_topk_pool},k={_k_kept})",   cob_pct_lin_overall),
                      (f"Cobweb-SelTopK  (depth={topk_depth},dim={n_topk_pool},k={TOP_K})",     cob_sel_topk_lin_overall),
                      (f"Cobweb-SelPct   (depth={topk_depth},dim={n_topk_pool},k={_k_sel_pct})", cob_sel_pct_lin_overall)]:
    print(f"  {name:<50} linear probe: {overall*100:.1f}%")

# ── Visualisation ─────────────────────────────────────────────────────────────

from matplotlib.patches import Patch

METHODS = [
    (Z_pca,           Z_pca_test,           pca_lin_per,          pca_knn_accs,          f"PCA ({DZ}d)",                                                          "o-", "#4878d0"),
    (Z_ae,            Z_ae_test,            ae_lin_per,           ae_knn_accs,           f"AE  ({DZ}d)",                                                          "s-", "#ee854a"),
    (Z_cob_bfs,       Z_cob_bfs_test,       cob_bfs_lin_per,      cob_bfs_knn_accs,      f"Cobweb-BFS ({DZ}d)",                                                  "^-", "#6acc65"),
    (Z_cob_dep,       Z_cob_dep_test,       cob_dep_lin_per,      cob_dep_knn_accs,      f"Cobweb-Depth (d={best_depth},{n_depth}d)",                            "D-", "#d65f5f"),
    (Z_cob_topk,      Z_cob_topk_test,      cob_topk_lin_per,     cob_topk_knn_accs,     f"Cobweb-TopK    (depth={topk_depth},dim={n_topk_pool},k={TOP_K})",      "P-", "#956cb4"),
    (Z_cob_pct,       Z_cob_pct_test,       cob_pct_lin_per,      cob_pct_knn_accs,      f"Cobweb-Pct     (depth={topk_depth},dim={n_topk_pool},k={_k_kept})",    "X-", "#e47298"),
    (Z_cob_sel_topk,  Z_cob_sel_topk_test,  cob_sel_topk_lin_per, cob_sel_topk_knn_accs, f"Cobweb-SelTopK (depth={topk_depth},dim={n_topk_pool},k={TOP_K})",      "*-", "#8c613c"),
    (Z_cob_sel_pct,   Z_cob_sel_pct_test,   cob_sel_pct_lin_per,  cob_sel_pct_knn_accs,  f"Cobweb-SelPct  (depth={topk_depth},dim={n_topk_pool},k={_k_sel_pct})", "h-", "#b47cc7"),
]

# 1. 2-D scatter plots (6 panels)
print("Computing UMAP projections for scatter plots …")
_umap = UMAP(n_components=2, random_state=42)
Z_pca2       = _umap.fit_transform(Z_pca)
Z_ae2        = _umap.fit_transform(Z_ae)
Z_bfs2       = _umap.fit_transform(Z_cob_bfs)
Z_dep2       = _umap.fit_transform(Z_cob_dep)
Z_topk2      = _umap.fit_transform(Z_cob_topk)
Z_pct2       = _umap.fit_transform(Z_cob_pct)
Z_sel_topk2  = _umap.fit_transform(Z_cob_sel_topk)
Z_sel_pct2   = _umap.fit_transform(Z_cob_sel_pct)

scatter_data = [
    (Z_pca2,      "PCA → UMAP 2D"),
    (Z_ae2,       "AE → UMAP 2D"),
    (Z_bfs2,      "Cobweb-BFS → UMAP 2D"),
    (Z_dep2,      "Cobweb-Depth → UMAP 2D"),
    (Z_topk2,     "Cobweb-TopK → UMAP 2D"),
    (Z_pct2,      "Cobweb-Pct → UMAP 2D"),
    (Z_sel_topk2, "Cobweb-SelTopK → UMAP 2D"),
    (Z_sel_pct2,  "Cobweb-SelPct → UMAP 2D"),
]
fig, axes = plt.subplots(1, 8, figsize=(42, 5))
for ax, (Z, title) in zip(axes, scatter_data):
    for c in CLASSES:
        mask = y == c
        ax.scatter(Z[mask, 0], Z[mask, 1], color=CMAP(c), alpha=0.5, s=3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CMAP(c),
                      markersize=7, label=str(c)) for c in CLASSES]
fig.legend(handles=handles, title="digit", loc="center right",
           bbox_to_anchor=(1.0, 0.5), frameon=True)
plt.tight_layout(rect=[0, 0, 0.96, 1])
plt.savefig(os.path.join(OUT_DIR, "scatter_2d.png"), dpi=120, bbox_inches="tight")
plt.close()

# 2. Linear probe — per-class test accuracy (6 methods × 10 digits)
n_methods = len(METHODS)
w = 0.8 / n_methods
x = np.arange(len(CLASSES))
offsets = [(i - (n_methods - 1) / 2) * w for i in range(n_methods)]
fig, ax = plt.subplots(figsize=(20, 5))
for (_, _, per, _, lbl, _, color), offset in zip(METHODS, offsets):
    ax.bar(x + offset, per * 100, w, label=lbl, color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"digit {c}" for c in CLASSES])
ax.set_ylabel("Test Accuracy %")
ax.set_title("Linear Probe — Per-class Test Accuracy")
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "linear_probe_per_class.png"), dpi=120)
plt.close()

# 3. KNN — test accuracy vs k (4 lines)
fig, ax = plt.subplots(figsize=(7, 5))
for _, _, _, knn_accs, lbl, marker, color in METHODS:
    ax.plot(KNN_KS, [a * 100 for a in knn_accs], marker, label=lbl, color=color)
ax.set_xlabel("k (number of neighbours)")
ax.set_ylabel("Test Accuracy %")
ax.set_title("KNN Test Accuracy vs k")
ax.set_xticks(KNN_KS)
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "knn_vs_k.png"), dpi=120)
plt.close()

# 4. Reconstruction gallery (10 samples, PCA and AE only)
with torch.no_grad():
    X_rec = ae.decoder(torch.tensor(Z_ae[:10], dtype=torch.float32)).numpy()
X_rec_pca = pca.inverse_transform(Z_pca[:10])

fig, axes = plt.subplots(3, 10, figsize=(15, 5))
fig.suptitle("Original vs PCA vs AE reconstructions (first 10 samples)")
for i in range(10):
    axes[0, i].imshow(X[i].reshape(28, 28), cmap="gray")
    axes[1, i].imshow(X_rec_pca[i].reshape(28, 28), cmap="gray")
    axes[2, i].imshow(X_rec[i].reshape(28, 28), cmap="gray")
    for row in range(3):
        axes[row, i].axis("off")
axes[0, 0].set_title("Original", fontsize=8)
axes[1, 0].set_title("PCA rec",  fontsize=8)
axes[2, 0].set_title("AE rec",   fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reconstructions.png"), dpi=120)
plt.close()
