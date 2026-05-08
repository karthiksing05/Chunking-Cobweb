"""
CIFAR-10 Representation Learning
Pipeline:
  Raw CIFAR (3×32×32) ──► ConvAE encoder ──► Z_conv (256-d)
                               │
          ┌────────────────────┼───────────────────────────────┐
          │                    │                               │
        PCA               SAEs on Z_conv                  Cobweb on Z_conv
     (256-d)     Linear-AE / L1-SAE / TopK-SAE+AuxK     BFS/Depth/TopK/Depth-TopK
                        (1024-d bottleneck)

Reconstruction path back to pixels:
  Z_sae ──► SAE decoder ──► Z_conv ──► ConvAE decoder ──► pixel
"""

import os
import csv
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
OUT_DIR  = os.path.join(HERE, "cifar_output")
DATA_DIR = os.path.join(OUT_DIR, "data")
ARR_DIR  = os.path.join(OUT_DIR, "arrays")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARR_DIR,  exist_ok=True)

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# ── Data ──────────────────────────────────────────────────────────────────────

transform = transforms.ToTensor()   # → [0,1] float32, shape (3,32,32)
trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,  download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

N_TRAIN = 10_000
N_TEST  =  2_000

def to_numpy(dataset, n):
    """Return (imgs_chw: float32 array, labels: int array) for the first n samples."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    imgs, lbls = next(iter(loader))
    return imgs.numpy(), lbls.numpy()

X_img,      y      = to_numpy(trainset, N_TRAIN)   # (N, 3, 32, 32)
X_img_test, y_test = to_numpy(testset,  N_TEST)

print(f"Data loaded. train={X_img.shape}  test={X_img_test.shape}")

# ── Config ────────────────────────────────────────────────────────────────────

DZ_CONV = 256    # ConvAE bottleneck dimensionality
DZ_SAE  = 1024   # SAE overcomplete bottleneck (4× expansion of ConvAE latents)
TOP_K   =  32    # top-k sparsification for TopK-SAE
AUXK    =  32    # AuxK dead-neuron revival count
AUXK_W  = 1/32   # weight on AuxK loss term
L1_LAM     = 3e-4   # L1 penalty coefficient for L1-SAE
PATH_DEPTH = 6      # tree depth prefix for path-information encoding
N_PATHS    = 4      # number of top-scoring leaf paths to trace per instance

# ── Convolutional Autoencoder ─────────────────────────────────────────────────

class ConvAE(nn.Module):
    """
    Encoder:
      Conv(3→32, 3×3, pad=1) → BN → ReLU                       [32×32]
      Conv(32→64, 3×3, pad=1, stride=2) → BN → ReLU             [16×16]
      Conv(64→128, 3×3, pad=1, stride=2) → BN → ReLU            [ 8× 8]
      Flatten → Linear(8192 → DZ_CONV)

    Decoder (symmetric):
      Linear(DZ_CONV → 8192) → Unflatten(128,8,8)
      ConvTranspose(128→64, 4×4, stride=2, pad=1) → BN → ReLU  [16×16]
      ConvTranspose(64→32,  4×4, stride=2, pad=1) → BN → ReLU  [32×32]
      Conv(32→3, 3×3, pad=1) → Sigmoid
    """
    def __init__(self, dz=DZ_CONV):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.enc_fc  = nn.Linear(128 * 8 * 8, dz)
        self.dec_fc  = nn.Linear(dz, 128 * 8 * 8)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x).flatten(1)
        return self.enc_fc(h)

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 8, 8)
        return self.dec_conv(h)

    def forward(self, x):
        return F.mse_loss(self.decode(self.encode(x)), x)


def train_convae(model, X_imgs, epochs=30, batch_size=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_imgs, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    model.train()
    for ep in range(epochs):
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = model(batch)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {ep+1:02d}/{epochs}  loss={total/len(loader):.4f}")
    model.eval()
    return model


print("Training ConvAE …")
conv_ae = train_convae(ConvAE(dz=DZ_CONV), X_img, epochs=30)

with torch.no_grad():
    Z_conv      = conv_ae.encode(torch.tensor(X_img,      dtype=torch.float32)).numpy()
    Z_conv_test = conv_ae.encode(torch.tensor(X_img_test, dtype=torch.float32)).numpy()

torch.save(conv_ae.state_dict(), os.path.join(ARR_DIR, "convae_weights.pt"))
np.save(os.path.join(ARR_DIR, "Z_conv_train.npy"), Z_conv)
np.save(os.path.join(ARR_DIR, "Z_conv_test.npy"),  Z_conv_test)
print(f"  ConvAE latents: train={Z_conv.shape}  test={Z_conv_test.shape}")

# ── PCA on ConvAE latents ─────────────────────────────────────────────────────

print("Computing PCA on ConvAE latents …")
pca = PCA(n_components=DZ_CONV)
Z_pca      = pca.fit_transform(Z_conv)
Z_pca_test = pca.transform(Z_conv_test)

# ── Linear-AE / SAEs — all operate on ConvAE latents ─────────────────────────
#
#  LinearAE: Linear(DZ_CONV → DZ_SAE), no activation, MSE only.
#            Represents the "blown-up bottleneck" baseline with no sparsity.
#
#  L1SAE:    Linear(DZ_CONV → DZ_SAE) + ReLU + L1 penalty → Linear(DZ_SAE → DZ_CONV)
#
#  TopKSAE:  Linear(DZ_CONV → DZ_SAE) + TopK + AuxK       → Linear(DZ_SAE → DZ_CONV)
#
#  All three reconstruct back to DZ_CONV (ConvAE latent space).
#  Reconstructions to pixel space chain: SAE decoder → ConvAE decoder.

class LinearAE(nn.Module):
    """Overcomplete linear bottleneck, no sparsity — the unregularised baseline."""
    def __init__(self, in_dim=DZ_CONV, dz=DZ_SAE):
        super().__init__()
        self.dz = dz
        self.encoder = nn.Linear(in_dim, dz)
        self.decoder = nn.Linear(dz, in_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encode(x)
        return h, self.decoder(h)


class L1SAE(nn.Module):
    """SAE with ReLU encoder and L1 sparsity penalty."""
    def __init__(self, in_dim=DZ_CONV, dz=DZ_SAE):
        super().__init__()
        self.dz = dz
        self.encoder = nn.Sequential(nn.Linear(in_dim, dz), nn.ReLU())
        self.decoder = nn.Linear(dz, in_dim)

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
    def __init__(self, in_dim=DZ_CONV, dz=DZ_SAE, k=TOP_K):
        super().__init__()
        self.dz = dz
        self.k  = k
        self._enc = nn.Linear(in_dim, dz)
        self.decoder = nn.Linear(dz, in_dim)
        self.register_buffer("fire_counts", torch.zeros(dz))

    def encode(self, x):
        pre = F.relu(self._enc(x))
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=1)
        h = torch.zeros_like(pre)
        h.scatter_(1, topk_idx, topk_vals)
        return h

    def encode_pre(self, x):
        return F.relu(self._enc(x))

    def forward(self, x):
        h = self.encode(x)
        return h, self.decoder(h)


def train_linear_ae(model, Z_train, epochs=20, batch_size=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Z_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    model.train()
    for ep in range(epochs):
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            _, x_hat = model(batch)
            loss = F.mse_loss(x_hat, batch)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {ep+1:02d}/{epochs}  loss={total/len(loader):.4f}")
    model.eval()
    return model


def train_l1sae(model, Z_train, lam=L1_LAM, epochs=20, batch_size=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Z_train, dtype=torch.float32)),
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


def train_topksae(model, Z_train, auxk=AUXK, auxk_w=AUXK_W, epochs=20, batch_size=256):
    """Train TopKSAE with AuxK loss to revive dead neurons."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Z_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    model.train()
    dead_threshold = len(Z_train) // batch_size
    for ep in range(epochs):
        total_rec, total_aux = 0.0, 0.0
        model.fire_counts.zero_()
        for (batch,) in loader:
            opt.zero_grad()
            pre = model.encode_pre(batch)
            topk_vals, topk_idx = torch.topk(pre, model.k, dim=1)
            h = torch.zeros_like(pre)
            h.scatter_(1, topk_idx, topk_vals)
            x_hat = model.decoder(h)
            rec = F.mse_loss(x_hat, batch)
            with torch.no_grad():
                model.fire_counts += (h > 0).float().sum(dim=0)
            dead_mask = model.fire_counts < dead_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and auxk > 0:
                k_aux = min(auxk, n_dead)
                pre_dead = pre * dead_mask.float()
                auxk_vals, auxk_idx = torch.topk(pre_dead, k_aux, dim=1)
                h_aux = torch.zeros_like(pre)
                h_aux.scatter_(1, auxk_idx, auxk_vals)
                residual = (batch - x_hat).detach()
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


print("Training Linear-AE on ConvAE latents …")
lin_ae = train_linear_ae(LinearAE(), Z_conv, epochs=20)

print("Training L1-SAE on ConvAE latents …")
l1sae = train_l1sae(L1SAE(), Z_conv, epochs=20)

print("Training TopK-SAE on ConvAE latents …")
topksae = train_topksae(TopKSAE(k=TOP_K), Z_conv, epochs=20)

with torch.no_grad():
    _Zt  = torch.tensor(Z_conv,      dtype=torch.float32)
    _Ztt = torch.tensor(Z_conv_test, dtype=torch.float32)
    Z_linAE        = lin_ae.encode(_Zt).numpy()
    Z_linAE_test   = lin_ae.encode(_Ztt).numpy()
    Z_l1sae        = l1sae.encode(_Zt).numpy()
    Z_l1sae_test   = l1sae.encode(_Ztt).numpy()
    Z_topksae      = topksae.encode(_Zt).numpy()
    Z_topksae_test = topksae.encode(_Ztt).numpy()

# ── Save data arrays ──────────────────────────────────────────────────────────

np.save(os.path.join(ARR_DIR, "y_train.npy"),          y)
np.save(os.path.join(ARR_DIR, "y_test.npy"),           y_test)
np.save(os.path.join(ARR_DIR, "Z_pca_train.npy"),      Z_pca)
np.save(os.path.join(ARR_DIR, "Z_pca_test.npy"),       Z_pca_test)
np.save(os.path.join(ARR_DIR, "Z_linAE_train.npy"),    Z_linAE)
np.save(os.path.join(ARR_DIR, "Z_linAE_test.npy"),     Z_linAE_test)
np.save(os.path.join(ARR_DIR, "Z_l1sae_train.npy"),    Z_l1sae)
np.save(os.path.join(ARR_DIR, "Z_l1sae_test.npy"),     Z_l1sae_test)
np.save(os.path.join(ARR_DIR, "Z_topksae_train.npy"),  Z_topksae)
np.save(os.path.join(ARR_DIR, "Z_topksae_test.npy"),   Z_topksae_test)
torch.save(lin_ae.state_dict(),  os.path.join(ARR_DIR, "linAE_weights.pt"))
torch.save(l1sae.state_dict(),   os.path.join(ARR_DIR, "l1sae_weights.pt"))
torch.save(topksae.state_dict(), os.path.join(ARR_DIR, "topksae_weights.pt"))
print("Data arrays saved.")

# ── Cobweb: build tree on ConvAE latents ──────────────────────────────────────
#
# Standardise the ConvAE latents before feeding to Cobweb so that each
# dimension has zero mean / unit variance (the Gaussian model in the tree
# is sensitive to feature scale).

_cob_scaler      = StandardScaler()
Z_cob_input      = _cob_scaler.fit_transform(Z_conv).astype(np.float32)
Z_cob_input_test = _cob_scaler.transform(Z_conv_test).astype(np.float32)

print("Building Cobweb tree on ConvAE latents …")
cobweb_tree = CobwebContinuousTree(
    size=DZ_CONV,
    covar_from=1,
    num_labels=0,
)
_empty_label = np.zeros(0, dtype=np.float32)
for i, z in enumerate(Z_cob_input):
    cobweb_tree.ifit(z, _empty_label)
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(Z_cob_input)} inserted")
print("  Tree built.")

# ── Cobweb: node extraction helpers ──────────────────────────────────────────

def collect_by_depth_nodes(root):
    by_depth = {}
    queue = [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        by_depth.setdefault(d, []).append(node)
        for child in node.children:
            queue.append((child, d + 1))
    return by_depth


def bfs_first_n_nodes(root, n):
    nodes, queue = [], [root]
    while queue and len(nodes) < n:
        node = queue.pop(0)
        for child in node.children:
            if len(nodes) >= n:
                break
            nodes.append(child)
            queue.append(child)
    return nodes


DZ = DZ_CONV   # target dimensionality for Cobweb BFS / depth encodings

print("Extracting BFS nodes …")
bfs_nodes = bfs_first_n_nodes(cobweb_tree.root, DZ)

print("Extracting static-depth nodes …")
by_depth_nodes = collect_by_depth_nodes(cobweb_tree.root)
depth_counts = {d: len(v) for d, v in by_depth_nodes.items()}
print(f"  Nodes per depth: {dict(sorted(depth_counts.items()))}")
best_depth = 0
for d in sorted(depth_counts.keys()):
    if depth_counts[d] >= DZ:
        break
    best_depth = d
print(f"  Using depth {best_depth} ({depth_counts[best_depth]} nodes, target <{DZ})")
depth_nodes = by_depth_nodes[best_depth]
n_depth = len(depth_nodes)

topk_depth = next(d for d in sorted(depth_counts.keys()) if depth_counts[d] >= DZ)
print(f"  Top-K pool: depth {topk_depth} ({depth_counts[topk_depth]} nodes)")
topk_pool_nodes = by_depth_nodes[topk_depth]
n_topk_pool = len(topk_pool_nodes)

# ── Cobweb: encoding ──────────────────────────────────────────────────────────

_empty = np.zeros(0, dtype=np.float32)


def encode_logpost(instances, nodes):
    """Return (n_samples, n_nodes) matrix of log P(x | node) for each node."""
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, x in enumerate(instances):
            out[i, j] = node.log_prob(x, _empty)
    return out


def topk_sparsify(Z, k):
    """Zero out all but the k largest values in each row."""
    out = np.zeros_like(Z)
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    rows = np.arange(Z.shape[0])[:, None]
    out[rows, top_idx] = Z[rows, top_idx]
    return out


print("Encoding train set (BFS) …")
_scaler_bfs = StandardScaler()
Z_cob_bfs      = _scaler_bfs.fit_transform(encode_logpost(Z_cob_input, bfs_nodes))
print("Encoding test set (BFS) …")
Z_cob_bfs_test = _scaler_bfs.transform(encode_logpost(Z_cob_input_test, bfs_nodes))

print("Encoding train set (Depth) …")
_scaler_dep = StandardScaler()
Z_cob_dep      = _scaler_dep.fit_transform(encode_logpost(Z_cob_input, depth_nodes))
print("Encoding test set (Depth) …")
Z_cob_dep_test = _scaler_dep.transform(encode_logpost(Z_cob_input_test, depth_nodes))

print(f"  Top-K pool size: {n_topk_pool} nodes at depth {topk_depth}")
print("Encoding Top-K pool (train) …")
_scaler_topk = StandardScaler()
Z_topk_pool      = _scaler_topk.fit_transform(encode_logpost(Z_cob_input, topk_pool_nodes))
Z_cob_topk       = topk_sparsify(Z_topk_pool, TOP_K)
print("Encoding Top-K pool (test) …")
Z_topk_pool_test = _scaler_topk.transform(encode_logpost(Z_cob_input_test, topk_pool_nodes))
Z_cob_topk_test  = topk_sparsify(Z_topk_pool_test, TOP_K)

Z_cob_bfs_topk      = topk_sparsify(Z_cob_bfs,      TOP_K)
Z_cob_bfs_topk_test = topk_sparsify(Z_cob_bfs_test, TOP_K)
print(f"  Applied per-instance top-{TOP_K} sparsification to BFS nodes (Depth-TopK, dim={DZ})")

np.save(os.path.join(ARR_DIR, "Z_cob_bfs_train.npy"),     Z_cob_bfs)
np.save(os.path.join(ARR_DIR, "Z_cob_bfs_test.npy"),      Z_cob_bfs_test)
np.save(os.path.join(ARR_DIR, "Z_cob_dep_train.npy"),     Z_cob_dep)
np.save(os.path.join(ARR_DIR, "Z_cob_dep_test.npy"),      Z_cob_dep_test)
np.save(os.path.join(ARR_DIR, "Z_cob_topk_train.npy"),    Z_cob_topk)
np.save(os.path.join(ARR_DIR, "Z_cob_topk_test.npy"),     Z_cob_topk_test)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopk_train.npy"), Z_cob_bfs_topk)
np.save(os.path.join(ARR_DIR, "Z_cob_bfstopk_test.npy"),  Z_cob_bfs_topk_test)
print("Cobweb data saved.")

# ── Cobweb: path-information encoding ────────────────────────────────────────
# Collect all nodes at depths 0…PATH_DEPTH.  For each instance find the
# N_PATHS leaves with the highest log P(x|leaf), union their ancestor chains,
# and retain standardised log-probs for on-path nodes; zero everything else.

def collect_path_tree_nodes(root, max_depth):
    """BFS-collect all nodes at depths 0..max_depth with ancestor tracking.
    Returns:
      all_nodes    : list of nodes in BFS order
      node_to_idx  : {id(node): index in all_nodes}
      leaves       : nodes at depth max_depth (or terminal if shallower)
      ancestor_ids : {id(node): frozenset of ids on the path root→node}
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
    Z_scaled values for every on-path ancestor node, zeros elsewhere."""
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
Z_path_raw      = encode_logpost(Z_cob_input,      path_all_nodes)
print("Encoding path-tree test ...")
Z_path_raw_test = encode_logpost(Z_cob_input_test, path_all_nodes)

_scaler_path   = StandardScaler()
Z_path_sc      = _scaler_path.fit_transform(Z_path_raw)
Z_path_sc_test = _scaler_path.transform(Z_path_raw_test)

Z_cob_path      = path_sparsify(Z_path_raw,      Z_path_sc,      path_node_to_idx, path_leaves, path_ancestor_ids, N_PATHS)
Z_cob_path_test = path_sparsify(Z_path_raw_test, Z_path_sc_test, path_node_to_idx, path_leaves, path_ancestor_ids, N_PATHS)

np.save(os.path.join(ARR_DIR, "Z_cob_path_train.npy"), Z_cob_path)
np.save(os.path.join(ARR_DIR, "Z_cob_path_test.npy"),  Z_cob_path_test)
print("Path-info data saved.")

# ── Cobweb: tree visualisation ────────────────────────────────────────────────

def compute_node_label_counts(root, Z_instances, y_labels, max_depth=3):
    n_classes = len(CIFAR_CLASSES)
    counts, node_obj = {}, {}

    def _ensure(node):
        nid = id(node)
        if nid not in counts:
            counts[nid] = np.zeros(n_classes, dtype=np.int32)
            node_obj[nid] = node
        return nid

    for z, label in zip(Z_instances, y_labels):
        node = root
        for depth in range(max_depth + 1):
            _ensure(node)
            counts[id(node)][int(label)] += 1
            if not node.children or depth == max_depth:
                break
            best_child = max(node.children, key=lambda c: c.log_prob(z, _empty))
            node = best_child

    return counts, node_obj


print("Computing node label distributions …")
label_counts_map, node_obj_map = compute_node_label_counts(
    cobweb_tree.root, Z_cob_input, y, max_depth=3)


def plot_cobweb_tree_labels(root, label_counts_map, max_depth=3, out_path=None):
    cmap10     = plt.get_cmap("tab10")
    cls_colors = [cmap10(i) for i in range(len(CIFAR_CLASSES))]

    def leaf_span(node, depth, max_depth):
        if depth >= max_depth or not node.children:
            return 1
        return sum(leaf_span(c, depth + 1, max_depth) for c in node.children)

    pos = {}

    def assign_pos(node, depth, x_left):
        span = leaf_span(node, depth, max_depth)
        pos[id(node)] = (x_left + span / 2.0, depth)
        if depth < max_depth and node.children:
            cursor = x_left
            for child in node.children:
                child_span = leaf_span(child, depth + 1, max_depth)
                assign_pos(child, depth + 1, cursor)
                cursor += child_span
        return span

    assign_pos(root, 0, 0.0)
    total_width = leaf_span(root, 0, max_depth)

    bar_w, bar_h, y_gap = 0.7, 0.35, 1.0
    fig, ax = plt.subplots(figsize=(max(14, total_width * 0.9), (max_depth + 1) * 2.2))
    ax.set_xlim(0, total_width)
    ax.set_ylim(-0.7, max_depth * y_gap + 0.7)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        f"Cobweb Tree — CIFAR-10 Label Distributions (depths 0–{max_depth})\n"
        "(trained on ConvAE latents)", fontsize=13)

    def draw_edges(node, depth):
        if depth >= max_depth or not node.children:
            return
        px, _ = pos[id(node)]
        for child in node.children:
            cx, _ = pos[id(child)]
            ax.plot([px, cx],
                    [depth * y_gap + bar_h / 2, (depth + 1) * y_gap - bar_h / 2],
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
        x_left = x_c - bar_w / 2
        y_top  = depth * y_gap - bar_h / 2
        cursor = x_left
        for ci in range(len(CIFAR_CLASSES)):
            seg_w = props[ci] * bar_w
            if seg_w > 0:
                ax.add_patch(plt.Rectangle((cursor, y_top), seg_w, bar_h,
                                           color=cls_colors[ci], lw=0))
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
        plt.Rectangle((0, 0), 1, 1, color=cls_colors[i], label=CIFAR_CLASSES[i])
        for i in range(len(CIFAR_CLASSES))
    ]
    ax.legend(handles=legend_handles, title="class", loc="lower right",
              ncol=5, fontsize=7, title_fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

plot_cobweb_tree_labels(
    cobweb_tree.root, label_counts_map, max_depth=3,
    out_path=os.path.join(OUT_DIR, "cobweb_tree_labels.png"),
)

# ── Evaluation ────────────────────────────────────────────────────────────────

CLASSES = list(range(len(CIFAR_CLASSES)))
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


print("\nEvaluating …")
conv_lin_overall,         conv_lin_per         = linear_probe_per_class(Z_conv,         y, Z_conv_test,         y_test)
pca_lin_overall,          pca_lin_per          = linear_probe_per_class(Z_pca,          y, Z_pca_test,          y_test)
linAE_lin_overall,        linAE_lin_per        = linear_probe_per_class(Z_linAE,        y, Z_linAE_test,        y_test)
l1sae_lin_overall,        l1sae_lin_per        = linear_probe_per_class(Z_l1sae,        y, Z_l1sae_test,        y_test)
topksae_lin_overall,      topksae_lin_per      = linear_probe_per_class(Z_topksae,      y, Z_topksae_test,      y_test)
cob_bfs_lin_overall,      cob_bfs_lin_per      = linear_probe_per_class(Z_cob_bfs,      y, Z_cob_bfs_test,      y_test)
cob_dep_lin_overall,      cob_dep_lin_per      = linear_probe_per_class(Z_cob_dep,      y, Z_cob_dep_test,      y_test)
cob_topk_lin_overall,     cob_topk_lin_per     = linear_probe_per_class(Z_cob_topk,     y, Z_cob_topk_test,     y_test)
cob_bfs_topk_lin_overall, cob_bfs_topk_lin_per = linear_probe_per_class(Z_cob_bfs_topk, y, Z_cob_bfs_topk_test, y_test)
cob_path_lin_overall,     cob_path_lin_per      = linear_probe_per_class(Z_cob_path,     y, Z_cob_path_test,     y_test)

conv_knn_accs         = knn_accuracy_vs_k(Z_conv,         y, Z_conv_test,         y_test)
pca_knn_accs          = knn_accuracy_vs_k(Z_pca,          y, Z_pca_test,          y_test)
linAE_knn_accs        = knn_accuracy_vs_k(Z_linAE,        y, Z_linAE_test,        y_test)
l1sae_knn_accs        = knn_accuracy_vs_k(Z_l1sae,        y, Z_l1sae_test,        y_test)
topksae_knn_accs      = knn_accuracy_vs_k(Z_topksae,      y, Z_topksae_test,      y_test)
cob_bfs_knn_accs      = knn_accuracy_vs_k(Z_cob_bfs,      y, Z_cob_bfs_test,      y_test)
cob_dep_knn_accs      = knn_accuracy_vs_k(Z_cob_dep,      y, Z_cob_dep_test,      y_test)
cob_topk_knn_accs     = knn_accuracy_vs_k(Z_cob_topk,     y, Z_cob_topk_test,     y_test)
cob_bfs_topk_knn_accs = knn_accuracy_vs_k(Z_cob_bfs_topk, y, Z_cob_bfs_topk_test, y_test)
cob_path_knn_accs     = knn_accuracy_vs_k(Z_cob_path,     y, Z_cob_path_test,     y_test)

print(f"\n  {'Method':<64} {'Lin.Probe':>10} {'KNN@5':>7} {'Avg L0':>8} {'Dead%':>7}")
print(f"  {'-'*100}")
_knn5_idx = KNN_KS.index(5)
_summary_rows = []
for name, overall, Z_tr, knn_accs in [
    (f"ConvAE ({DZ_CONV}d)",                                                   conv_lin_overall,         Z_conv,         conv_knn_accs),
    (f"PCA on ConvAE latents ({DZ_CONV}d)",                                    pca_lin_overall,          Z_pca,          pca_knn_accs),
    (f"Linear-AE ({DZ_CONV}→{DZ_SAE}→{DZ_CONV}, no sparsity)",               linAE_lin_overall,        Z_linAE,        linAE_knn_accs),
    (f"L1-SAE ({DZ_SAE}d, λ={L1_LAM})",                                       l1sae_lin_overall,        Z_l1sae,        l1sae_knn_accs),
    (f"TopK-SAE ({DZ_SAE}d, k={TOP_K}+AuxK)",                                 topksae_lin_overall,      Z_topksae,      topksae_knn_accs),
    (f"Cobweb-BFS ({DZ}d, on ConvAE latents)",                                cob_bfs_lin_overall,      Z_cob_bfs,      cob_bfs_knn_accs),
    (f"Cobweb-Depth (depth={best_depth}, dim={n_depth})",                      cob_dep_lin_overall,      Z_cob_dep,      cob_dep_knn_accs),
    (f"Cobweb-TopK (depth={topk_depth}, dim={n_topk_pool}, k={TOP_K})",       cob_topk_lin_overall,     Z_cob_topk,     cob_topk_knn_accs),
    (f"Cobweb-Depth-TopK ({DZ}d, k={TOP_K})",                                 cob_bfs_topk_lin_overall, Z_cob_bfs_topk, cob_bfs_topk_knn_accs),
    (f"Cobweb-Path (depth={PATH_DEPTH}, n={N_PATHS}, dim={n_path_dim})",       cob_path_lin_overall,     Z_cob_path,     cob_path_knn_accs),
]:
    avg_l0, dead_pct = _repr_stats(Z_tr)
    knn5 = knn_accs[_knn5_idx] * 100
    print(f"  {name:<64} {overall*100:>9.1f}% {knn5:>6.1f}% {avg_l0:>8.1f} {dead_pct:>6.1f}%")
    _summary_rows.append({
        "method":        name,
        "lin_probe_pct": round(overall * 100, 2),
        "knn5_pct":      round(knn5, 2),
        "avg_l0":        round(float(avg_l0), 2),
        "dead_pct":      round(float(dead_pct), 2),
    })

_csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(_csv_path, "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=["method", "lin_probe_pct", "knn5_pct", "avg_l0", "dead_pct"])
    _w.writeheader()
    _w.writerows(_summary_rows)
print(f"  Summary saved → {_csv_path}")

# ── Visualisation ─────────────────────────────────────────────────────────────

METHODS = [
    (Z_conv,         Z_conv_test,         conv_lin_per,         conv_knn_accs,         f"ConvAE ({DZ_CONV}d)",                                              "o-", "#4878d0"),
    (Z_pca,          Z_pca_test,          pca_lin_per,          pca_knn_accs,          f"PCA on ConvAE ({DZ_CONV}d)",                                       "s-", "#ee854a"),
    (Z_linAE,        Z_linAE_test,        linAE_lin_per,        linAE_knn_accs,        f"Linear-AE ({DZ_SAE}d)",                                            "P-", "#2ca02c"),
    (Z_l1sae,        Z_l1sae_test,        l1sae_lin_per,        l1sae_knn_accs,        f"L1-SAE ({DZ_SAE}d, λ={L1_LAM})",                                  "v-", "#ff7f0e"),
    (Z_topksae,      Z_topksae_test,      topksae_lin_per,      topksae_knn_accs,      f"TopK-SAE ({DZ_SAE}d, k={TOP_K})",                                 "H-", "#bcbd22"),
    (Z_cob_bfs,      Z_cob_bfs_test,      cob_bfs_lin_per,      cob_bfs_knn_accs,      f"Cobweb-BFS ({DZ}d)",                                               "^-", "#6acc65"),
    (Z_cob_dep,      Z_cob_dep_test,      cob_dep_lin_per,      cob_dep_knn_accs,      f"Cobweb-Depth (d={best_depth}, dim={n_depth})",                     "D-", "#d65f5f"),
    (Z_cob_topk,     Z_cob_topk_test,     cob_topk_lin_per,     cob_topk_knn_accs,     f"Cobweb-TopK (d={topk_depth}, n={n_topk_pool}, k={TOP_K})",        "X-", "#956cb4"),
    (Z_cob_bfs_topk, Z_cob_bfs_topk_test, cob_bfs_topk_lin_per, cob_bfs_topk_knn_accs, f"Cobweb-Depth-TopK ({DZ}d, k={TOP_K})",                            "*-", "#17becf"),
    (Z_cob_path,     Z_cob_path_test,      cob_path_lin_per,     cob_path_knn_accs,     f"Cobweb-Path (d={PATH_DEPTH},n={N_PATHS},dim={n_path_dim})",          "p-", "#8c564b"),
]

# 1a. UMAP scatter plots
print("Computing UMAP projections …")
_umap = UMAP(n_components=2, random_state=42)
scatter_2d_umap = [_umap.fit_transform(m[0]) for m in METHODS]

n_panels = len(METHODS)
fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5.5, 5))
fig.suptitle("UMAP Projections — CIFAR-10 (ConvAE backbone)", fontsize=12, y=1.01)
for ax, Z2, m in zip(axes, scatter_2d_umap, METHODS):
    for c in CLASSES:
        mask = y == c
        ax.scatter(Z2[mask, 0], Z2[mask, 1], color=CMAP(c), alpha=0.5, s=3)
    ax.set_title(m[4], fontsize=8)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CMAP(c),
                      markersize=7, label=CIFAR_CLASSES[c]) for c in CLASSES]
fig.legend(handles=handles, title="class", loc="center right",
           bbox_to_anchor=(1.0, 0.5), frameon=True)
plt.tight_layout(rect=[0, 0, 0.97, 1])
plt.savefig(os.path.join(OUT_DIR, "scatter_umap.png"), dpi=120, bbox_inches="tight")
plt.close()

# 1b. t-SNE scatter plots
print("Computing t-SNE projections …")
_tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
scatter_2d_tsne = [_tsne.fit_transform(m[0]) for m in METHODS]

fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5.5, 5))
fig.suptitle("t-SNE Projections — CIFAR-10 (ConvAE backbone)", fontsize=12, y=1.01)
for ax, Z2, m in zip(axes, scatter_2d_tsne, METHODS):
    for c in CLASSES:
        mask = y == c
        ax.scatter(Z2[mask, 0], Z2[mask, 1], color=CMAP(c), alpha=0.5, s=3)
    ax.set_title(m[4], fontsize=8)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
fig.legend(handles=handles, title="class", loc="center right",
           bbox_to_anchor=(1.0, 0.5), frameon=True)
plt.tight_layout(rect=[0, 0, 0.97, 1])
plt.savefig(os.path.join(OUT_DIR, "scatter_tsne.png"), dpi=120, bbox_inches="tight")
plt.close()

# 2. Linear probe — per-class bar chart
n_methods = len(METHODS)
w = 0.8 / n_methods
x = np.arange(len(CLASSES))
offsets = [(i - (n_methods - 1) / 2) * w for i in range(n_methods)]
fig, ax = plt.subplots(figsize=(24, 5))
for (_, _, per, _, lbl, _, color), offset in zip(METHODS, offsets):
    ax.bar(x + offset, per * 100, w, label=lbl, color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(CIFAR_CLASSES, rotation=30, ha="right")
ax.set_ylabel("Test Accuracy %")
ax.set_title("Linear Probe — Per-class Test Accuracy (CIFAR-10, ConvAE backbone)")
ax.set_ylim(0, 105)
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
ax.set_title("KNN Test Accuracy vs k  —  CIFAR-10 (ConvAE backbone)")
ax.set_xticks(KNN_KS)
ax.set_ylim(0, 105)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "knn_vs_k.png"), dpi=120)
plt.close()

# 4. Reconstruction gallery
# Rows: Original | ConvAE | PCA→ConvAE | Linear-AE→ConvAE | L1-SAE→ConvAE | TopK-SAE→ConvAE
# All learned reconstructions are chained through the ConvAE decoder back to pixel space.

with torch.no_grad():
    _t10_z = torch.tensor(Z_conv[:10], dtype=torch.float32)

    # ConvAE reconstruction (direct decode from latent)
    X_rec_convae = conv_ae.decode(_t10_z).numpy()          # (10, 3, 32, 32)

    # PCA: inverse-project back to ConvAE latent space, then decode
    _pca_z    = torch.tensor(pca.inverse_transform(Z_pca[:10]), dtype=torch.float32)
    X_rec_pca = conv_ae.decode(_pca_z).numpy()

    # Linear-AE: SAE decoder → ConvAE latent → pixels
    _, _linAE_z  = lin_ae(_t10_z)
    X_rec_linAE  = conv_ae.decode(_linAE_z).numpy()

    # L1-SAE: SAE decoder → ConvAE latent → pixels
    _, _l1_z    = l1sae(_t10_z)
    X_rec_l1sae = conv_ae.decode(_l1_z).numpy()

    # TopK-SAE: SAE decoder → ConvAE latent → pixels
    _, _topk_z     = topksae(_t10_z)
    X_rec_topksae  = conv_ae.decode(_topk_z).numpy()


def to_img(chw):
    """(3, 32, 32) → (32, 32, 3) clipped to [0, 1] for imshow."""
    return np.clip(chw.transpose(1, 2, 0), 0, 1)


_rec_rows = [
    (X_img[:10],    "Original"),
    (X_rec_convae,  "ConvAE rec"),
    (X_rec_pca,     "PCA→ConvAE"),
    (X_rec_linAE,   "Linear-AE→ConvAE"),
    (X_rec_l1sae,   "L1-SAE→ConvAE"),
    (X_rec_topksae, "TopK-SAE→ConvAE"),
]
fig, axes = plt.subplots(len(_rec_rows), 10, figsize=(15, len(_rec_rows) * 1.7))
fig.suptitle("Reconstruction gallery — CIFAR-10 (first 10 samples)")
for row_idx, (imgs, label) in enumerate(_rec_rows):
    for i in range(10):
        axes[row_idx, i].imshow(to_img(imgs[i]))
        axes[row_idx, i].axis("off")
    axes[row_idx, 0].set_title(label, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reconstructions.png"), dpi=120)
plt.close()

print(f"\nAll outputs saved to {OUT_DIR}")
