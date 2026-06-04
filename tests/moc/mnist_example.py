"""
MNIST Representation Learning
Learn representations via PCA and Autoencoder, then evaluate with
linear probing, KNN, and visualize results.

What I have learned so far:
- covar_from = 2 provides good linear probing scores but worse KNN
- prior_var REALLY matters (like it REALLY REALLY MATTERS) - BEST TO LEAVE IT AS DEFAULT THO!
- sparsity benefits us slightly!! log-prob signals of some prototypes is like noise
- 
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
from sklearn.manifold import TSNE
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

DZ      = 256    # representation dimensionality
TOP_K   =  16    # fixed top-k per-instance sparsification
AUXK    =  16    # AuxK: dead-neuron top-k for aux reconstruction loss
AUXK_W  = 1/32   # weight on AuxK loss term
L1_LAM     = 3e-4   # L1 penalty coefficient for L1-SAE
PATH_DEPTH = 6      # tree depth prefix for path-information encoding
N_PATHS    = 4      # number of top-scoring leaf paths to trace per instance

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

# ── Sparse Autoencoders ───────────────────────────────────────────────────────

class L1SAE(nn.Module):
    """SAE with ReLU encoder and L1 sparsity penalty."""
    def __init__(self, dz=DZ, input_dim=784):
        super().__init__()
        self.dz = dz
        self.encoder = nn.Sequential(nn.Linear(input_dim, dz), nn.ReLU())
        self.decoder = nn.Linear(dz, input_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encode(x)
        return h, self.decoder(h)


class TopKSAE(nn.Module):
    """SAE whose encoder keeps only the top-k activations per sample.
    Supports AuxK: an auxiliary loss that routes dead-neuron activations
    through the decoder toward the residual error, reviving dead neurons.
    """
    def __init__(self, dz=DZ, input_dim=784, k=TOP_K):
        super().__init__()
        self.dz = dz
        self.k  = k
        self._enc = nn.Linear(input_dim, dz)
        self.decoder = nn.Linear(dz, input_dim)
        # fire-count buffer for dead-neuron tracking (not a parameter)
        self.register_buffer("fire_counts", torch.zeros(dz))

    def encode(self, x):
        pre = F.relu(self._enc(x))
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=1)
        h = torch.zeros_like(pre)
        h.scatter_(1, topk_idx, topk_vals)
        return h

    def encode_pre(self, x):
        """Return pre-sparsification ReLU activations."""
        return F.relu(self._enc(x))

    def forward(self, x):
        h = self.encode(x)
        return h, self.decoder(h)


def train_l1sae(model, X_train, lam=L1_LAM, epochs=20, batch_size=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    model.train()
    for ep in range(epochs):
        total_rec, total_l1 = 0.0, 0.0
        for (batch,) in loader:
            opt.zero_grad()
            h, x_hat = model(batch)
            rec = F.mse_loss(x_hat, batch)
            l1  = lam * h.abs().mean()
            (rec + l1).backward()
            opt.step()
            total_rec += rec.item()
            total_l1  += l1.item()
        n = len(loader)
        print(f"  epoch {ep+1:02d}/{epochs}  rec={total_rec/n:.4f}  l1={total_l1/n:.5f}")
    model.eval()
    return model


def train_topksae(model, X_train, auxk=AUXK, auxk_w=AUXK_W, epochs=20, batch_size=256):
    """Train TopKSAE with AuxK loss to revive dead neurons.

    AuxK: after normal TopK forward, compute residual = x - x_hat (no grad).
    Among the neurons that have fired fewer times than a threshold (dead), take
    the top-AUXK by their pre-activation magnitude, reconstruct the residual
    from those, and add a weighted MSE loss.  This pushes dead neurons toward
    useful directions without corrupting the primary TopK representation.
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    model.train()
    dead_threshold = len(X_train) // batch_size  # ~1 epoch of batches
    for ep in range(epochs):
        total_rec, total_aux = 0.0, 0.0
        model.fire_counts.zero_()
        for (batch,) in loader:
            opt.zero_grad()
            # ── primary TopK forward ──
            pre = model.encode_pre(batch)                # (B, dz)
            topk_vals, topk_idx = torch.topk(pre, model.k, dim=1)
            h = torch.zeros_like(pre)
            h.scatter_(1, topk_idx, topk_vals)
            x_hat = model.decoder(h)
            rec = F.mse_loss(x_hat, batch)

            # update fire counts
            with torch.no_grad():
                model.fire_counts += (h > 0).float().sum(dim=0)

            # ── AuxK loss on dead neurons ──
            dead_mask = model.fire_counts < dead_threshold  # (dz,) bool
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and auxk > 0:
                k_aux = min(auxk, n_dead)
                # activations of dead neurons only, others zeroed
                pre_dead = pre * dead_mask.float()          # (B, dz)
                auxk_vals, auxk_idx = torch.topk(pre_dead, k_aux, dim=1)
                h_aux = torch.zeros_like(pre)
                h_aux.scatter_(1, auxk_idx, auxk_vals)
                residual = (batch - x_hat).detach()         # target for aux
                x_aux = model.decoder(h_aux)
                aux = auxk_w * F.mse_loss(x_aux, residual + x_hat.detach())
            else:
                aux = torch.tensor(0.0)

            (rec + aux).backward()
            opt.step()
            total_rec += rec.item()
            total_aux += aux.item()
        n = len(loader)
        n_dead_final = (model.fire_counts < dead_threshold).sum().item()
        print(f"  epoch {ep+1:02d}/{epochs}  rec={total_rec/n:.4f}  aux={total_aux/n:.5f}  dead={n_dead_final}/{model.dz}")
    model.eval()
    return model


print("Training L1-SAE …")
l1sae = train_l1sae(L1SAE(dz=DZ), X, epochs=20)

print("Training TopK-SAE …")
topksae = train_topksae(TopKSAE(dz=DZ, k=TOP_K), X, epochs=20)

with torch.no_grad():
    _Xt  = torch.tensor(X,      dtype=torch.float32)
    _Xtt = torch.tensor(X_test, dtype=torch.float32)
    Z_l1sae        = l1sae.encode(_Xt).numpy()
    Z_l1sae_test   = l1sae.encode(_Xtt).numpy()
    Z_topksae      = topksae.encode(_Xt).numpy()
    Z_topksae_test = topksae.encode(_Xtt).numpy()

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
np.save(os.path.join(ARR_DIR, "Z_l1sae_train.npy"),   Z_l1sae)
np.save(os.path.join(ARR_DIR, "Z_l1sae_test.npy"),    Z_l1sae_test)
np.save(os.path.join(ARR_DIR, "Z_topksae_train.npy"), Z_topksae)
np.save(os.path.join(ARR_DIR, "Z_topksae_test.npy"),  Z_topksae_test)
torch.save(l1sae.state_dict(),   os.path.join(ARR_DIR, "l1sae_weights.pt"))
torch.save(topksae.state_dict(), os.path.join(ARR_DIR, "topksae_weights.pt"))
print("Data arrays saved.")

# ── Cobweb: raw-input setup ───────────────────────────────────────────────────

X_cob = X.astype(np.float32, copy=False)
X_cob_test = X_test.astype(np.float32, copy=False)

# ── Cobweb: build tree ────────────────────────────────────────────────────────

print("Building Cobweb tree …")
cobweb_tree = CobwebContinuousTree(
    size=X_cob.shape[1],
    covar_from=1,
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
            out[i, j] = node.log_prob(x, _empty)
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

def topk_binarize(Z, k):
    """Set the k largest values per row to 1.0, rest to 0.0."""
    out = np.zeros_like(Z)
    k   = min(k, Z.shape[1])
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    rows    = np.arange(Z.shape[0])[:, None]
    out[rows, top_idx] = 1.0
    return out

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

# Depth-TopK (BFS-TopK): take the DZ BFS nodes and sparsify per-instance to TOP_K
Z_cob_bfs_topk      = topk_sparsify(Z_cob_bfs,      TOP_K)
Z_cob_bfs_topk_test = topk_sparsify(Z_cob_bfs_test, TOP_K)
print(f"  Applied per-instance top-{TOP_K} sparsification to BFS nodes (Depth-TopK, dim={DZ})")

# TopK-Binary / Depth-TopK-Binary: same selection but active nodes fixed to 1.0
Z_cob_topk_bin          = topk_binarize(Z_topk_pool,      TOP_K)
Z_cob_topk_bin_test     = topk_binarize(Z_topk_pool_test, TOP_K)
Z_cob_bfs_topk_bin      = topk_binarize(Z_cob_bfs,        TOP_K)
Z_cob_bfs_topk_bin_test = topk_binarize(Z_cob_bfs_test,   TOP_K)
print(f"  Applied per-instance top-{TOP_K} binarisation (TopK-Binary, Depth-TopK-Binary)")

np.save(os.path.join(ARR_DIR, "Z_cob_bfs_train.npy"),      Z_cob_bfs)
np.save(os.path.join(ARR_DIR, "Z_cob_bfs_test.npy"),       Z_cob_bfs_test)
np.save(os.path.join(ARR_DIR, "Z_cob_dep_train.npy"),      Z_cob_dep)
np.save(os.path.join(ARR_DIR, "Z_cob_dep_test.npy"),       Z_cob_dep_test)
np.save(os.path.join(ARR_DIR, "Z_cob_topk_train.npy"),     Z_cob_topk)
np.save(os.path.join(ARR_DIR, "Z_cob_topk_test.npy"),      Z_cob_topk_test)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopk_train.npy"),     Z_cob_bfs_topk)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopk_test.npy"),      Z_cob_bfs_topk_test)
np.save(os.path.join(ARR_DIR, "Z_cob_topkbin_train.npy"),     Z_cob_topk_bin)
np.save(os.path.join(ARR_DIR, "Z_cob_topkbin_test.npy"),      Z_cob_topk_bin_test)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopkbin_train.npy"),  Z_cob_bfs_topk_bin)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopkbin_test.npy"),   Z_cob_bfs_topk_bin_test)
print("Cobweb data saved.")

# ── Cobweb: path-information encoding ────────────────────────────────────────
# For each instance find the N_PATHS leaves at depth PATH_DEPTH (or shallowest
# terminal node when the tree is thinner) with the highest log P(x | leaf).
# Retain standardised log-probs for every ancestor node on those paths;
# zero out every other node in the same tree prefix.

def collect_path_tree_nodes(root, max_depth):
    """BFS-collect all nodes at depths 0..max_depth with ancestor tracking.
    Returns:
      all_nodes    : list of nodes in BFS order
      node_to_idx  : {id(node): index in all_nodes}
      leaves       : nodes at depth max_depth (or terminal if tree is shallower)
      ancestor_ids : {id(node): frozenset of ids on the path root→node (incl. self)}
    """
    all_nodes, node_to_idx, leaves, ancestor_ids = [], {}, [], {}
    queue = [(root, 0, frozenset())]
    while queue:
        node, depth, parent_ancs = queue.pop(0)
        idx = len(all_nodes)
        all_nodes.append(node)
        node_to_idx[id(node)] = idx
        my_ancs = parent_ancs | {id(node)}
        ancestor_ids[id(node)] = my_ancs
        if depth >= max_depth or not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                queue.append((child, depth + 1, my_ancs))
    return all_nodes, node_to_idx, leaves, ancestor_ids


def path_sparsify(Z_raw, Z_scaled, node_to_idx, leaves, ancestor_ids, n_paths):
    """Use raw log_probs to pick the n_paths top leaves per sample, then return
    Z_scaled values for every node on those ancestor paths, zeros elsewhere."""
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


print(f"Collecting path-tree nodes (depth <= {PATH_DEPTH}) ...")
path_all_nodes, path_node_to_idx, path_leaves, path_ancestor_ids = \
    collect_path_tree_nodes(cobweb_tree.root, PATH_DEPTH)
n_path_dim = len(path_all_nodes)
print(f"  {n_path_dim} total nodes, {len(path_leaves)} leaves")

print("Encoding path-tree train ...")
Z_path_raw      = encode_logpost(X_cob,      path_all_nodes)
print("Encoding path-tree test ...")
Z_path_raw_test = encode_logpost(X_cob_test, path_all_nodes)

_scaler_path    = StandardScaler()
Z_path_sc       = _scaler_path.fit_transform(Z_path_raw)
Z_path_sc_test  = _scaler_path.transform(Z_path_raw_test)

Z_cob_path      = path_sparsify(Z_path_raw,      Z_path_sc,      path_node_to_idx, path_leaves, path_ancestor_ids, N_PATHS)
Z_cob_path_test = path_sparsify(Z_path_raw_test, Z_path_sc_test, path_node_to_idx, path_leaves, path_ancestor_ids, N_PATHS)

np.save(os.path.join(ARR_DIR, "Z_cob_path_train.npy"), Z_cob_path)
np.save(os.path.join(ARR_DIR, "Z_cob_path_test.npy"),  Z_cob_path_test)
print("Path-info data saved.")

# ── Cobweb: Product-of-Experts compositional encoding ─────────────────────────
# After Wang, Gupta, Zhu & MacLellan, "Test-Time Compositional Generalization in
# Diffusion Models via Concept Discovery" (2026).  That work repurposes a
# pretrained diffusion model's time-indexed score field as a *hierarchy of
# density modes*: local modes of the noisy marginals p_t(x_t) act as reusable
# concept prototypes whose abstraction level is set by the noise level t.  Given a
# query it (i) discovers candidate prototypes by mode-ascent, summarised as local
# Gaussians q_j(x)=N(x; m_j, Σ_j); (ii) greedily selects a few with a submodular
# coverage objective; (iii) composes their Gaussians into a Product-of-Experts.
#
# Cobweb already *is* that hierarchy of Gaussian density modes — no mode-ascent or
# Tweedie pullback required.  Every node n is a diagonal-Gaussian expert
#     q_n(x) = N(x; m_n, Σ_n),  m_n = n.mean,  σ²_{n,r} = n.sum_sq_r/n.count + prior_var
# — exactly the model Cobweb itself scores with (verified to match node.log_prob).
# Shallow nodes are coarse concept prototypes, deep nodes fine ones: the direct
# analog of diffusion modes coarsening as the noise level rises.
#
# Encoding a query x  (paper Eqs. 8-10):
#     ℓ_{n,r}(x) = log N(x_r; m_{n,r}, σ²_{n,r})           per-dim expert log-lik
#     F(S)       = Σ_r max_{n∈S} ℓ_{n,r}(x)                submodular coverage (Eq. 8)
#     grow S: best singleton, then max marginal gain ∆_n   until |S| = K  (Eq. 9)
#     w_n(r)     = softmax_{n∈S}( ℓ_{n,r}(x) / τ )          per-dim PoE weights (Eq. 10)
#
# CODING — each selected prototype's *marginal gain*.  We seed the greedy
# coverage with the tree root (the most general "mode" — the trivial concept that
# covers everything), so S_0 = {root}.  The value stored for the prototype chosen
# at each step is its marginal gain over the current explanation,
#     ∆_n = F(S_{i-1} ∪ {n}) − F(S_{i-1}) = Σ_r max( ℓ_{n,r} − cur_max_r , 0 ) ≥ 0,
# which is exactly the quantity the greedy step maximises (Eq. 9).  Seeding with
# the root gives every prototype a principled, all-positive, instance-comparable
# gain (no arbitrary floor).  The code is sparse over the prototype pool: entry n =
# ∆_n for n∈S, 0 otherwise.  Unlike Cobweb-TopK (independent top-k log-probs, which
# keeps redundant parent/child nodes on one path), the submodular objective forces
# the K prototypes to cover *complementary* dimensions — a genuine compositional
# part decomposition.  The PoE mean μ_T (Eq. 7) is also reconstructed (gallery), and
# the per-instance selected concepts are rendered in poe_concepts.png.

POE_DEPTHS    = (1, 2, 3, 4, 5, 6)   # tree levels donating concept prototypes
POE_MIN_COUNT = 20                    # min support for a trustworthy Gaussian expert
POE_CAP       = DZ                    # prototype-pool size (matches the other methods)
POE_K         = TOP_K                 # number of prototypes composed per instance
POE_TAU       = 1.0                   # PoE softmax temperature (Eq. 10)
PRIOR_VAR     = 0.05854983152         # CobwebContinuousTree default prior_var

print("Building Cobweb-PoE prototype pool …")
_poe_candidates = []
for d in POE_DEPTHS:
    for node in by_depth_nodes.get(d, []):
        if node.count >= POE_MIN_COUNT:
            _poe_candidates.append((d, node))
# keep the best-supported POE_CAP prototypes across the pooled depths
_poe_candidates.sort(key=lambda dn: dn[1].count, reverse=True)
_poe_candidates = _poe_candidates[:POE_CAP]
poe_depths = [d for d, _ in _poe_candidates]
poe_nodes  = [n for _, n in _poe_candidates]
n_poe      = len(poe_nodes)
_poe_depth_hist = {d: poe_depths.count(d) for d in sorted(set(poe_depths))}
print(f"  {n_poe} prototypes; per-depth counts {_poe_depth_hist}")

# Diagonal-Gaussian parameters for every prototype (P, d)
proto_mean = np.stack([np.asarray(n.mean,   dtype=np.float32) for n in poe_nodes])
proto_var  = (np.stack([np.asarray(n.sum_sq, dtype=np.float32) / np.float32(n.count)
                        for n in poe_nodes]) + np.float32(PRIOR_VAR))
proto_logc = (-0.5 * np.log(2.0 * np.pi * proto_var)).astype(np.float32)   # per-dim log-norm const
proto_iv   = (0.5 / proto_var).astype(np.float32)                          # ½·precision
POE_K_EFF  = int(min(POE_K, n_poe))

# Root node = the most general "mode": the empty-set baseline S_0 = {root} that
# seeds the greedy coverage, so every prototype gets an all-positive marginal gain.
_root_mean = np.asarray(cobweb_tree.root.mean,   dtype=np.float32)
_root_var  = (np.asarray(cobweb_tree.root.sum_sq, dtype=np.float32)
              / np.float32(cobweb_tree.root.count) + np.float32(PRIOR_VAR))
root_logc  = (-0.5 * np.log(2.0 * np.pi * _root_var)).astype(np.float32)
root_iv    = (0.5 / _root_var).astype(np.float32)


def encode_poe(instances, k=POE_K_EFF, tau=POE_TAU, return_mu=False, batch=256):
    """Greedy submodular prototype selection + Product-of-Experts composition.

    Coding = each selected prototype's *marginal gain* ∆_n (Eq. 9), the coverage it
    adds over the current explanation, with the greedy coverage seeded by the tree
    root (S_0 = {root}) so all gains are ≥ 0 and comparable across instances.

    Returns (code, mu_out, sel, gain):
      code  (n, n_poe)  sparse — entry = ∆_n for the K selected prototypes, else 0
      mu_out(n, d)      PoE mean μ_T (Eq. 7) if return_mu else None
      sel   (n, k)      pool indices of the selected prototypes, in selection order
      gain  (n, k)      their marginal gains ∆_n, aligned with sel
    """
    Xf = instances.astype(np.float32, copy=False)
    n, d = Xf.shape
    code     = np.zeros((n, n_poe), dtype=np.float64)
    sel_all  = np.empty((n, k), dtype=np.int64)
    gain_all = np.empty((n, k), dtype=np.float64)
    mu_out   = np.zeros((n, d), dtype=np.float32) if return_mu else None
    for s in range(0, n, batch):
        xb = Xf[s:s + batch]                                          # (B, d)
        B  = xb.shape[0]
        # per-dim expert log-likelihood  L[b, j, r] = logc[j,r] - ½·iv[j,r]·(x-m)²
        diff = xb[:, None, :] - proto_mean[None, :, :]                # (B, P, d)
        L = proto_logc[None, :, :] - proto_iv[None, :, :] * diff * diff
        # seed coverage with the root expert (S_0 = {root}): ℓ_root per dimension
        rdiff   = xb - _root_mean[None, :]
        cur_max = (root_logc[None, :] - root_iv[None, :] * rdiff * rdiff).copy()  # (B, d)
        chosen  = np.zeros((B, n_poe), dtype=bool)
        sel     = np.empty((B, k), dtype=np.int64)
        gains   = np.empty((B, k), dtype=np.float64)
        br      = np.arange(B)
        # ── greedy: at each step add the prototype with the largest marginal gain ──
        for step in range(k):
            gain = np.maximum(L - cur_max[:, None, :], 0.0).sum(axis=2)   # (B, P) ∆_n ≥ 0
            gain[chosen] = -np.inf
            j = gain.argmax(axis=1)
            sel[:, step]   = j
            gains[:, step] = gain[br, j]                              # store the marginal gain
            chosen[br, j]  = True
            cur_max = np.maximum(cur_max, L[br, j, :])                # extend coverage
        # write marginal gains into the sparse code
        code[np.repeat(br, k) + s, sel.reshape(-1)] = gains.reshape(-1)
        sel_all[s:s + B]  = sel
        gain_all[s:s + B] = gains
        if return_mu:
            # ── per-dim PoE weights over the selected set: softmax(ℓ/τ) (Eq. 10) ──
            Lsel = L[br[:, None], sel, :]                             # (B, k, d)
            w = Lsel / tau
            w -= w.max(axis=1, keepdims=True)
            np.exp(w, out=w)
            w /= w.sum(axis=1, keepdims=True)
            iv_sel, mean_sel = proto_iv[sel], proto_mean[sel]         # (B, k, d) each
            num = (w * iv_sel * mean_sel).sum(axis=1)                 # PoE: Σ w·Σ⁻¹·m
            den = (w * iv_sel).sum(axis=1)                            # PoE: Σ w·Σ⁻¹
            mu_out[s:s + B] = num / den
    return code, mu_out, sel_all, gain_all


print(f"Encoding Cobweb-PoE train (K={POE_K_EFF}, τ={POE_TAU}, coding=marginal-gain) …")
Z_cob_poe, Z_cob_poe_mu, poe_sel, poe_gain = encode_poe(X_cob, return_mu=True)
print("Encoding Cobweb-PoE test …")
Z_cob_poe_test, _, _, _ = encode_poe(X_cob_test)

np.save(os.path.join(ARR_DIR, "Z_cob_poe_train.npy"), Z_cob_poe)
np.save(os.path.join(ARR_DIR, "Z_cob_poe_test.npy"),  Z_cob_poe_test)
print("Cobweb-PoE data saved.")

# ── Cobweb-PoE: visualise the selected concepts (not the full code) ────────────
# For one example per digit, render the K' top concept prototypes that compose it
# (each as its Gaussian mean-image m_n, the literal "concept"), ordered by marginal
# gain, alongside the original and the PoE reconstruction μ_T.  Titles show each
# concept's tree depth (abstraction level) and its marginal gain ∆_n.

POE_VIZ_CONCEPTS = 6                                   # top concepts shown per instance
_poe_depths_arr  = np.asarray(poe_depths)
print("Rendering Cobweb-PoE selected-concept gallery …")
_viz_idx = [int(np.where(y == c)[0][0]) for c in range(10)]   # first sample of each digit
_n_cols  = 2 + POE_VIZ_CONCEPTS
fig, axes = plt.subplots(len(_viz_idx), _n_cols,
                         figsize=(_n_cols * 1.5, len(_viz_idx) * 1.6))
fig.suptitle(
    "Cobweb-PoE — concepts composing each digit "
    f"(top {POE_VIZ_CONCEPTS} of K={POE_K_EFF} by marginal gain)\n"
    "col 1 = input · col 2 = PoE μ_T · cols 3+ = selected concept prototypes (mean-image)",
    fontsize=10,
)
for row, si in enumerate(_viz_idx):
    # order this instance's selected concepts by marginal gain (descending)
    order = np.argsort(-poe_gain[si])
    axes[row, 0].imshow(X[si].reshape(28, 28), cmap="gray")
    axes[row, 0].set_ylabel(f"digit {y[si]}", fontsize=8)
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    if row == 0:
        axes[row, 0].set_title("input", fontsize=8)
    axes[row, 1].imshow(Z_cob_poe_mu[si].reshape(28, 28), cmap="gray")
    axes[row, 1].axis("off")
    if row == 0:
        axes[row, 1].set_title("PoE μ_T", fontsize=8)
    for c in range(POE_VIZ_CONCEPTS):
        ax = axes[row, 2 + c]
        ax.axis("off")
        if c < len(order):
            pj    = int(poe_sel[si, order[c]])
            gj    = poe_gain[si, order[c]]
            depth = int(_poe_depths_arr[pj])
            ax.imshow(proto_mean[pj].reshape(28, 28), cmap="magma")
            ax.set_title(f"d{depth}·Δ{gj:.0f}", fontsize=7)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, "poe_concepts.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Selected-concept gallery saved → {os.path.join(OUT_DIR, 'poe_concepts.png')}")

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

def _repr_stats(Z):
    """(avg_l0, dead_pct): mean non-zero features per sample; % of features always zero."""
    nz = (Z != 0)
    return nz.sum(axis=1).mean(), (~nz.any(axis=0)).mean() * 100

def softmax_entropy(Z):
    """Per-sample entropy (nats) after softmax normalisation over dimensions.
    Uses the signed activations directly; large negative values get near-zero
    probability, so the softmax naturally suppresses dead/inactive features.
    Returns an array of shape (n_samples,)."""
    shifted = Z - Z.max(axis=1, keepdims=True)   # numerical stability
    exp_Z   = np.exp(shifted)
    p       = exp_Z / exp_Z.sum(axis=1, keepdims=True)
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)

print("\nEvaluating …")
pca_lin_overall,      pca_lin_per      = linear_probe_per_class(Z_pca,      y, Z_pca_test,      y_test)
ae_lin_overall,       ae_lin_per       = linear_probe_per_class(Z_ae,       y, Z_ae_test,       y_test)
l1sae_lin_overall,    l1sae_lin_per    = linear_probe_per_class(Z_l1sae,    y, Z_l1sae_test,    y_test)
topksae_lin_overall,  topksae_lin_per  = linear_probe_per_class(Z_topksae,  y, Z_topksae_test,  y_test)
cob_bfs_lin_overall,  cob_bfs_lin_per  = linear_probe_per_class(Z_cob_bfs,  y, Z_cob_bfs_test,  y_test)
cob_dep_lin_overall,  cob_dep_lin_per  = linear_probe_per_class(Z_cob_dep,  y, Z_cob_dep_test,  y_test)
cob_topk_lin_overall,         cob_topk_lin_per         = linear_probe_per_class(Z_cob_topk,         y, Z_cob_topk_test,         y_test)
cob_bfs_topk_lin_overall,     cob_bfs_topk_lin_per     = linear_probe_per_class(Z_cob_bfs_topk,     y, Z_cob_bfs_topk_test,     y_test)
cob_topk_bin_lin_overall,     cob_topk_bin_lin_per     = linear_probe_per_class(Z_cob_topk_bin,     y, Z_cob_topk_bin_test,     y_test)
cob_bfs_topk_bin_lin_overall, cob_bfs_topk_bin_lin_per = linear_probe_per_class(Z_cob_bfs_topk_bin, y, Z_cob_bfs_topk_bin_test, y_test)
cob_path_lin_overall,         cob_path_lin_per          = linear_probe_per_class(Z_cob_path,         y, Z_cob_path_test,         y_test)
cob_poe_lin_overall,          cob_poe_lin_per           = linear_probe_per_class(Z_cob_poe,          y, Z_cob_poe_test,          y_test)

pca_knn_accs      = knn_accuracy_vs_k(Z_pca,      y, Z_pca_test,      y_test)
ae_knn_accs       = knn_accuracy_vs_k(Z_ae,       y, Z_ae_test,       y_test)
l1sae_knn_accs    = knn_accuracy_vs_k(Z_l1sae,    y, Z_l1sae_test,    y_test)
topksae_knn_accs  = knn_accuracy_vs_k(Z_topksae,  y, Z_topksae_test,  y_test)
cob_bfs_knn_accs  = knn_accuracy_vs_k(Z_cob_bfs,  y, Z_cob_bfs_test,  y_test)
cob_dep_knn_accs  = knn_accuracy_vs_k(Z_cob_dep,  y, Z_cob_dep_test,  y_test)
cob_topk_knn_accs         = knn_accuracy_vs_k(Z_cob_topk,         y, Z_cob_topk_test,         y_test)
cob_bfs_topk_knn_accs     = knn_accuracy_vs_k(Z_cob_bfs_topk,     y, Z_cob_bfs_topk_test,     y_test)
cob_topk_bin_knn_accs     = knn_accuracy_vs_k(Z_cob_topk_bin,     y, Z_cob_topk_bin_test,     y_test)
cob_bfs_topk_bin_knn_accs = knn_accuracy_vs_k(Z_cob_bfs_topk_bin, y, Z_cob_bfs_topk_bin_test, y_test)
cob_path_knn_accs         = knn_accuracy_vs_k(Z_cob_path,         y, Z_cob_path_test,         y_test)
cob_poe_knn_accs          = knn_accuracy_vs_k(Z_cob_poe,          y, Z_cob_poe_test,          y_test)

print(f"\n  {'Method':<54} {'Lin.Probe':>10} {'KNN@5':>7} {'Avg L0':>8} {'Dead%':>7}")
print(f"  {'-'*90}")
_knn5_idx = KNN_KS.index(5)
_summary_rows = []
for name, overall, Z_tr, knn_accs in [
    (f"PCA ({DZ}d)",                                              pca_lin_overall,      Z_pca,      pca_knn_accs),
    (f"AE  ({DZ}d)",                                              ae_lin_overall,       Z_ae,       ae_knn_accs),
    (f"L1-SAE ({DZ}d, λ={L1_LAM})",                              l1sae_lin_overall,    Z_l1sae,    l1sae_knn_accs),
    (f"TopK-SAE ({DZ}d, k={TOP_K})",                              topksae_lin_overall,  Z_topksae,  topksae_knn_accs),
    (f"Cobweb-BFS ({DZ}d)",                                       cob_bfs_lin_overall,  Z_cob_bfs,  cob_bfs_knn_accs),
    (f"Cobweb-Depth (depth={best_depth},dim={n_depth})",          cob_dep_lin_overall,  Z_cob_dep,  cob_dep_knn_accs),
    (f"Cobweb-TopK (depth={topk_depth},dim={n_topk_pool},k={TOP_K})",     cob_topk_lin_overall,         Z_cob_topk,         cob_topk_knn_accs),
    (f"Cobweb-Depth-TopK ({DZ}d, k={TOP_K})",                             cob_bfs_topk_lin_overall,     Z_cob_bfs_topk,     cob_bfs_topk_knn_accs),
    (f"Cobweb-TopK-Bin (depth={topk_depth},dim={n_topk_pool},k={TOP_K})", cob_topk_bin_lin_overall,     Z_cob_topk_bin,     cob_topk_bin_knn_accs),
    (f"Cobweb-Depth-TopK-Bin ({DZ}d, k={TOP_K})",                         cob_bfs_topk_bin_lin_overall, Z_cob_bfs_topk_bin, cob_bfs_topk_bin_knn_accs),
    (f"Cobweb-Path (depth={PATH_DEPTH}, n={N_PATHS}, dim={n_path_dim})",  cob_path_lin_overall,         Z_cob_path,         cob_path_knn_accs),
    (f"Cobweb-PoE (dim={n_poe}, K={POE_K_EFF}, marginal-gain)",           cob_poe_lin_overall,          Z_cob_poe,          cob_poe_knn_accs),
]:
    avg_l0, dead_pct = _repr_stats(Z_tr)
    avg_ent = softmax_entropy(Z_tr).mean()
    knn5 = knn_accs[_knn5_idx] * 100
    print(f"  {name:<54} {overall*100:>9.1f}% {knn5:>6.1f}% {avg_l0:>8.1f} {dead_pct:>6.1f}%  ent={avg_ent:.3f}")
    _summary_rows.append({
        "method":        name,
        "lin_probe_pct": round(overall * 100, 2),
        "knn5_pct":      round(knn5, 2),
        "avg_l0":        round(float(avg_l0), 2),
        "dead_pct":      round(float(dead_pct), 2),
        "avg_entropy":   round(float(avg_ent), 4),
    })

import csv
_csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(_csv_path, "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=["method", "lin_probe_pct", "knn5_pct", "avg_l0", "dead_pct", "avg_entropy"])
    _w.writeheader()
    _w.writerows(_summary_rows)
print(f"  Summary saved → {_csv_path}")

# ── Visualisation ─────────────────────────────────────────────────────────────

from matplotlib.patches import Patch

METHODS = [
    (Z_pca,      Z_pca_test,      pca_lin_per,     pca_knn_accs,     f"PCA ({DZ}d)",                                                    "o-", "#4878d0"),
    (Z_ae,       Z_ae_test,       ae_lin_per,      ae_knn_accs,      f"AE  ({DZ}d)",                                                    "s-", "#ee854a"),
    (Z_l1sae,    Z_l1sae_test,    l1sae_lin_per,   l1sae_knn_accs,   f"L1-SAE ({DZ}d, λ={L1_LAM})",                                    "v-", "#ff7f0e"),
    (Z_topksae,  Z_topksae_test,  topksae_lin_per, topksae_knn_accs, f"TopK-SAE ({DZ}d, k={TOP_K})",                                   "H-", "#bcbd22"),
    (Z_cob_bfs,  Z_cob_bfs_test,  cob_bfs_lin_per, cob_bfs_knn_accs, f"Cobweb-BFS ({DZ}d)",                                            "^-", "#6acc65"),
    (Z_cob_dep,  Z_cob_dep_test,  cob_dep_lin_per, cob_dep_knn_accs, f"Cobweb-Depth (depth={best_depth},dim={n_depth})",               "D-", "#d65f5f"),
    (Z_cob_topk,         Z_cob_topk_test,         cob_topk_lin_per,         cob_topk_knn_accs,         f"Cobweb-TopK (depth={topk_depth},dim={n_topk_pool},k={TOP_K})",     "P-", "#956cb4"),
    (Z_cob_bfs_topk,     Z_cob_bfs_topk_test,     cob_bfs_topk_lin_per,     cob_bfs_topk_knn_accs,     f"Cobweb-Depth-TopK ({DZ}d, k={TOP_K})",                             "X-", "#17becf"),
    (Z_cob_topk_bin,     Z_cob_topk_bin_test,     cob_topk_bin_lin_per,     cob_topk_bin_knn_accs,     f"Cobweb-TopK-Bin (depth={topk_depth},dim={n_topk_pool},k={TOP_K})", "8-", "#c39bd3"),
    (Z_cob_bfs_topk_bin, Z_cob_bfs_topk_bin_test, cob_bfs_topk_bin_lin_per, cob_bfs_topk_bin_knn_accs, f"Cobweb-Depth-TopK-Bin ({DZ}d, k={TOP_K})",                         ">-", "#76d7c4"),
    (Z_cob_path,         Z_cob_path_test,          cob_path_lin_per,         cob_path_knn_accs,         f"Cobweb-Path (d={PATH_DEPTH},n={N_PATHS},dim={n_path_dim})",         "p-", "#8c564b"),
    (Z_cob_poe,          Z_cob_poe_test,           cob_poe_lin_per,          cob_poe_knn_accs,          f"Cobweb-PoE (dim={n_poe},K={POE_K_EFF},Δgain)",                     "*-", "#e377c2"),
]

# 1a. UMAP scatter plots
print("Computing UMAP projections for scatter plots …")
_umap = UMAP(n_components=2, random_state=42)
Z_pca2     = _umap.fit_transform(Z_pca)
Z_ae2      = _umap.fit_transform(Z_ae)
Z_l1sae2   = _umap.fit_transform(Z_l1sae)
Z_topksae2 = _umap.fit_transform(Z_topksae)
Z_bfs2     = _umap.fit_transform(Z_cob_bfs)
Z_dep2     = _umap.fit_transform(Z_cob_dep)
Z_topk2      = _umap.fit_transform(Z_cob_topk)
Z_bfstopk2   = _umap.fit_transform(Z_cob_bfs_topk)
Z_path2        = _umap.fit_transform(Z_cob_path)
Z_topkbin2     = _umap.fit_transform(Z_cob_topk_bin)
Z_bfstopkbin2  = _umap.fit_transform(Z_cob_bfs_topk_bin)
Z_poe2         = _umap.fit_transform(Z_cob_poe)

scatter_data_umap = [
    (Z_pca2,        "PCA → UMAP 2D"),
    (Z_ae2,         "AE → UMAP 2D"),
    (Z_l1sae2,      "L1-SAE → UMAP 2D"),
    (Z_topksae2,    "TopK-SAE → UMAP 2D"),
    (Z_bfs2,        "Cobweb-BFS → UMAP 2D"),
    (Z_dep2,        "Cobweb-Depth → UMAP 2D"),
    (Z_topk2,       "Cobweb-TopK → UMAP 2D"),
    (Z_bfstopk2,    "Cobweb-Depth-TopK → UMAP 2D"),
    (Z_topkbin2,    "Cobweb-TopK-Bin → UMAP 2D"),
    (Z_bfstopkbin2, "Cobweb-Depth-TopK-Bin → UMAP 2D"),
    (Z_path2,       f"Cobweb-Path (d={PATH_DEPTH},n={N_PATHS}) → UMAP 2D"),
    (Z_poe2,        f"Cobweb-PoE (K={POE_K_EFF}) → UMAP 2D"),
]
fig, axes = plt.subplots(1, len(scatter_data_umap), figsize=(len(scatter_data_umap) * 5.2, 5))
fig.suptitle("UMAP Projections", fontsize=12, y=1.01)
for ax, (Z, title) in zip(axes, scatter_data_umap):
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
plt.savefig(os.path.join(OUT_DIR, "scatter_umap.png"), dpi=120, bbox_inches="tight")
plt.close()

# 1b. t-SNE scatter plots
print("Computing t-SNE projections for scatter plots …")
_tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
Z_pca2t     = _tsne.fit_transform(Z_pca)
Z_ae2t      = _tsne.fit_transform(Z_ae)
Z_l1sae2t   = _tsne.fit_transform(Z_l1sae)
Z_topksae2t = _tsne.fit_transform(Z_topksae)
Z_bfs2t     = _tsne.fit_transform(Z_cob_bfs)
Z_dep2t     = _tsne.fit_transform(Z_cob_dep)
Z_topk2t      = _tsne.fit_transform(Z_cob_topk)
Z_bfstopk2t   = _tsne.fit_transform(Z_cob_bfs_topk)
Z_path2t        = _tsne.fit_transform(Z_cob_path)
Z_topkbin2t     = _tsne.fit_transform(Z_cob_topk_bin)
Z_bfstopkbin2t  = _tsne.fit_transform(Z_cob_bfs_topk_bin)
Z_poe2t         = _tsne.fit_transform(Z_cob_poe)

scatter_data_tsne = [
    (Z_pca2t,        "PCA → t-SNE 2D"),
    (Z_ae2t,         "AE → t-SNE 2D"),
    (Z_l1sae2t,      "L1-SAE → t-SNE 2D"),
    (Z_topksae2t,    "TopK-SAE → t-SNE 2D"),
    (Z_bfs2t,        "Cobweb-BFS → t-SNE 2D"),
    (Z_dep2t,        "Cobweb-Depth → t-SNE 2D"),
    (Z_topk2t,       "Cobweb-TopK → t-SNE 2D"),
    (Z_bfstopk2t,    "Cobweb-Depth-TopK → t-SNE 2D"),
    (Z_topkbin2t,    "Cobweb-TopK-Bin → t-SNE 2D"),
    (Z_bfstopkbin2t, "Cobweb-Depth-TopK-Bin → t-SNE 2D"),
    (Z_path2t,       f"Cobweb-Path (d={PATH_DEPTH},n={N_PATHS}) → t-SNE 2D"),
    (Z_poe2t,        f"Cobweb-PoE (K={POE_K_EFF}) → t-SNE 2D"),
]
fig, axes = plt.subplots(1, len(scatter_data_tsne), figsize=(len(scatter_data_tsne) * 5.2, 5))
fig.suptitle("t-SNE Projections", fontsize=12, y=1.01)
for ax, (Z, title) in zip(axes, scatter_data_tsne):
    for c in CLASSES:
        mask = y == c
        ax.scatter(Z[mask, 0], Z[mask, 1], color=CMAP(c), alpha=0.5, s=3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
fig.legend(handles=handles, title="digit", loc="center right",
           bbox_to_anchor=(1.0, 0.5), frameon=True)
plt.tight_layout(rect=[0, 0, 0.96, 1])
plt.savefig(os.path.join(OUT_DIR, "scatter_tsne.png"), dpi=120, bbox_inches="tight")
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

# 4. Reconstruction gallery (10 samples: Original, PCA, AE, L1-SAE, TopK-SAE)
with torch.no_grad():
    _t10 = torch.tensor(X[:10], dtype=torch.float32)
    X_rec_ae     = ae.decoder(torch.tensor(Z_ae[:10],     dtype=torch.float32)).numpy()
    X_rec_l1sae  = l1sae.decoder(l1sae.encode(_t10)).numpy()
    X_rec_topksae = topksae.decoder(topksae.encode(_t10)).numpy()
X_rec_pca = pca.inverse_transform(Z_pca[:10])

_rec_rows = [
    (X[:10],          "Original"),
    (X_rec_pca,       "PCA rec"),
    (X_rec_ae,        "AE rec"),
    (X_rec_l1sae,     "L1-SAE rec"),
    (X_rec_topksae,   "TopK-SAE rec"),
    (Z_cob_poe_mu[:10], "Cobweb-PoE μ_T"),
]
fig, axes = plt.subplots(len(_rec_rows), 10, figsize=(15, len(_rec_rows) * 1.7))
fig.suptitle("Reconstruction gallery (first 10 samples)")
for row_idx, (imgs, label) in enumerate(_rec_rows):
    for i in range(10):
        axes[row_idx, i].imshow(imgs[i].reshape(28, 28), cmap="gray")
        axes[row_idx, i].axis("off")
    axes[row_idx, 0].set_title(label, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reconstructions.png"), dpi=120)
plt.close()

# 5. Dimension activity histograms + softmax entropy
# Each subplot: histogram of per-dimension fire frequencies (fraction of samples
# where that feature is non-zero).  Subtitle shows average softmax entropy,
# which summarises how spread/concentrated activations are per sample.
_n_methods = len(METHODS)
_n_cols    = 3
_n_rows    = (_n_methods + _n_cols - 1) // _n_cols
fig, axes  = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 5, _n_rows * 3.8))
axes       = axes.flatten()

for idx, (Z_tr, _, _, _, lbl, _, color) in enumerate(METHODS):
    ax = axes[idx]
    fire_freq = (Z_tr != 0).mean(axis=0)          # fraction of samples firing each dim
    avg_ent   = softmax_entropy(Z_tr).mean()       # average per-sample softmax entropy
    ax.hist(fire_freq, bins=60, color=color, alpha=0.82, edgecolor="white", linewidth=0.3)
    ax.axvline(fire_freq.mean(), color="black", linewidth=1.0, linestyle="--", label=f"mean={fire_freq.mean():.3f}")
    ax.set_title(f"{lbl}\nAvg softmax entropy: {avg_ent:.3f} nats", fontsize=8)
    ax.set_xlabel("Dimension fire frequency (fraction of samples)", fontsize=7)
    ax.set_ylabel("# dimensions", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

for idx in range(_n_methods, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    "Per-dimension Activity Frequency  —  MNIST\n"
    "(bar height = # features that fire at that fraction of samples;\n"
    " subtitle entropy = average H[softmax(z)] per sample)",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dim_activity_hist.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"Dim-activity histogram saved → {os.path.join(OUT_DIR, 'dim_activity_hist.png')}")

# 6. Per-node image-count histograms
# For each method: x = node index (sorted by count descending for readability),
# y = number of training images that node was active for.
fig, axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 6, _n_rows * 3.8))
axes      = axes.flatten()

for idx, (Z_tr, _, _, _, lbl, _, color) in enumerate(METHODS):
    ax          = axes[idx]
    node_counts = (Z_tr != 0).sum(axis=0)          # (n_dims,) — images per node, native order
    n_active    = (node_counts > 0).sum()
    n_dead      = (node_counts == 0).sum()
    ax.bar(np.arange(len(node_counts)), node_counts,
           color=color, alpha=0.82, width=1.0, linewidth=0)
    ax.axhline(node_counts[node_counts > 0].mean() if n_active > 0 else 0,
               color="black", linewidth=1.0, linestyle="--",
               label=f"mean (active)={node_counts[node_counts > 0].mean():.1f}" if n_active > 0 else "mean=0")
    ax.set_title(
        f"{lbl}\n"
        f"active: {n_active}  |  dead: {n_dead}  |  total: {Z_tr.shape[1]}",
        fontsize=8,
    )
    ax.set_xlabel("node index (native / BFS order)", fontsize=7)
    ax.set_ylabel("# training images node is active for", fontsize=7)
    ax.set_xlim(0, len(node_counts))
    ax.set_ylim(0, len(Z_tr) * 1.05)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

for idx in range(_n_methods, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    "Per-node Image Activity  —  MNIST\n"
    "(nodes in native/BFS order; dashed line = mean over active nodes)",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "node_activity_hist.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"Node-activity plot saved → {os.path.join(OUT_DIR, 'node_activity_hist.png')}")

# 7. Per-sample competitive-node count histograms  (Cobweb variants only)
# For each training sample: find the best node score (max), then count how many
# nodes are "competitive" — i.e., have relative probability >= threshold.
# Uses probability-space comparison: exp(z_j - z_best) >= threshold
# (works correctly for negative log_probs and sparse/standardised values alike).

def _competitive_node_fig(cob_methods, threshold, fname_suffix, dataset="MNIST"):
    """Plot competitive-node count histograms for one threshold value."""
    pct_str    = f"{int(threshold * 100)}%"
    _cn_cols   = min(3, len(cob_methods))
    _cn_rows   = (len(cob_methods) + _cn_cols - 1) // _cn_cols
    fig, axes  = plt.subplots(_cn_rows, _cn_cols,
                              figsize=(_cn_cols * 5, _cn_rows * 3.8))
    axes       = np.array(axes).flatten()

    for idx, (Z_tr, lbl, color) in enumerate(cob_methods):
        ax       = axes[idx]
        Z_f      = Z_tr.astype(np.float64)
        best     = Z_f.max(axis=1, keepdims=True)        # (N, 1)
        rel_prob = np.exp(np.clip(Z_f - best, -500, 0))  # in (0, 1], best node = 1
        counts   = (rel_prob >= threshold).sum(axis=1)   # (N,) integer counts

        # Trim x-axis to the last bin that actually has samples (ignore trailing zeros).
        if len(counts) == 0:
            max_count = 1
        else:
            # highest count with at least 1 sample
            hist_vals, _ = np.histogram(counts, bins=np.arange(0.5, counts.max() + 1.5, 1.0))
            nonzero_bins  = np.where(hist_vals > 0)[0]
            max_count     = int(nonzero_bins[-1]) + 1 if len(nonzero_bins) else 1

        bins = np.arange(0.5, max_count + 1.5, 1.0)
        ax.hist(counts, bins=bins, color=color, alpha=0.82,
                edgecolor="white", linewidth=0.4)
        ax.axvline(counts.mean(), color="black", linewidth=1.2, linestyle="--",
                   label=f"mean={counts.mean():.2f}")
        ax.set_title(
            f"{lbl}\nmedian={np.median(counts):.0f}  |  mean={counts.mean():.2f}",
            fontsize=8,
        )
        ax.set_xlabel(f"# nodes with relative probability ≥ {pct_str} of best node",
                      fontsize=7)
        ax.set_ylabel("# training samples", fontsize=7)
        # Place ~11 ticks at 0%, 10%, 20%, …, 100% of the x-range.
        _tick_positions = np.unique(np.round(
            np.linspace(1, max_count, min(11, max_count))
        ).astype(int))
        ax.set_xticks(_tick_positions)
        ax.set_xlim(0.5, max_count + 0.5)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    for idx in range(len(cob_methods), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Competitive-node count per sample  —  {dataset}  (Cobweb variants)\n"
        f"X = # nodes whose probability is ≥ {pct_str} of the best node's probability\n"
        f"[exp(z_j − z_best) ≥ {threshold}]   Y = # training samples with that count",
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
