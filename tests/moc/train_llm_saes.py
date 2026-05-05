"""
train_llm_saes.py
==================
Train four SAE variants on GPT-2 last-layer residual-stream activations:

  1. L1-SAE     – classic ReLU encoder + L1 sparsity (Bricken et al. 2023)
  2. TopK-SAE   – hard TopK sparsification  (Gao et al. 2024)
  3. JumpReLU   – learnable per-neuron threshold with STE (Rajamanoharan et al. 2024)
  4. CobwebTopK – CobwebContinuousTree node log-posteriors, top-k sparsified

All models encode d_model=768 → d_sae=3072 (4× expansion).
Trained models / arrays are saved to tests/moc/gpt_acts_output/models/.

Usage:
    python tests/moc/train_llm_saes.py
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from cobweb.cobweb_continuous import CobwebContinuousTree

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(HERE, "gpt_acts_output")
ACT_DIR   = os.path.join(OUT_DIR, "acts")
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

D_MODEL   = 768          # GPT-2 small residual-stream dimension
D_SAE     = D_MODEL * 4  # 3072 — standard 4× expansion
TOP_K     = 32           # TopK and CobwebTopK sparsity
AUXK      = 32           # AuxK dead-neuron revival budget
AUXK_W    = 1 / 32       # AuxK loss weight
L1_LAM    = 3e-4         # L1 sparsity coefficient
EPOCHS    = 10
BATCH     = 512
LR        = 2e-4
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEED      = 42
COB_TRAIN = 50_000       # samples used to build Cobweb tree (first N of train)
COB_NODES = D_SAE        # BFS nodes used as "latents" for Cobweb

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load activations ──────────────────────────────────────────────────────────

print("Loading activations …")
X_train = np.load(os.path.join(ACT_DIR, "acts_train.npy"))   # (N, 768)
X_test  = np.load(os.path.join(ACT_DIR, "acts_test.npy"))

print(f"  train: {X_train.shape}   test: {X_test.shape}")

# Global whitening (zero-mean, unit-variance per feature)
_scaler = StandardScaler()
X_train = _scaler.fit_transform(X_train).astype(np.float32)
X_test  = _scaler.transform(X_test).astype(np.float32)
np.save(os.path.join(MODEL_DIR, "input_scaler_mean.npy"), _scaler.mean_.astype(np.float32))
np.save(os.path.join(MODEL_DIR, "input_scaler_std.npy"),  _scaler.scale_.astype(np.float32))

# ── Shared dataloader helper ──────────────────────────────────────────────────

def make_loader(X, batch_size=BATCH, shuffle=True):
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  L1-SAE
# ─────────────────────────────────────────────────────────────────────────────

class L1SAE(nn.Module):
    """Sparse Autoencoder with ReLU encoder and L1 sparsity penalty."""
    def __init__(self, d_in=D_MODEL, d_sae=D_SAE):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d_sae), nn.ReLU())
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        # tie decoder columns to unit norm (common practice)
        self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encode(x)
        return h, self.decoder(h)


def train_l1sae(model, X, lam=L1_LAM, epochs=EPOCHS, device=DEVICE):
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(X)
    for ep in range(epochs):
        model.train()
        total_rec = total_l1 = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            h, x_hat = model(batch)
            rec = F.mse_loss(x_hat, batch)
            l1  = lam * h.abs().mean()
            (rec + l1).backward()
            # re-normalise decoder columns after each step
            with torch.no_grad():
                model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0)
            opt.step()
            total_rec += rec.item()
            total_l1  += l1.item()
        n = len(loader)
        print(f"  [L1-SAE] ep {ep+1:02d}/{epochs}  rec={total_rec/n:.5f}  l1={total_l1/n:.6f}")
    return model.eval().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TopK-SAE  (Gao et al. 2024)  + AuxK dead-neuron revival
# ─────────────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """Hard TopK encoder + decoder.  Fire-count buffer for AuxK tracking."""
    def __init__(self, d_in=D_MODEL, d_sae=D_SAE, k=TOP_K):
        super().__init__()
        self.k = k
        self._enc = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.register_buffer("fire_counts", torch.zeros(d_sae))

    def _normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode_pre(self, x):
        return F.relu(self._enc(x))

    def encode(self, x):
        pre = self.encode_pre(x)
        vals, idx = torch.topk(pre, self.k, dim=1)
        h = torch.zeros_like(pre)
        h.scatter_(1, idx, vals)
        return h

    def forward(self, x):
        h = self.encode(x)
        return h, self.decoder(h)


def train_topksae(model, X, auxk=AUXK, auxk_w=AUXK_W, epochs=EPOCHS, device=DEVICE):
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(X)
    dead_threshold = len(X) // BATCH  # ~1 epoch of batches
    for ep in range(epochs):
        model.train()
        model.fire_counts.zero_()
        total_rec = total_aux = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            pre = model.encode_pre(batch)
            vals, idx = torch.topk(pre, model.k, dim=1)
            h = torch.zeros_like(pre)
            h.scatter_(1, idx, vals)
            x_hat = model.decoder(h)
            rec = F.mse_loss(x_hat, batch)
            with torch.no_grad():
                model.fire_counts += (h > 0).float().sum(dim=0)
            # AuxK
            dead_mask = model.fire_counts < dead_threshold
            n_dead    = dead_mask.sum().item()
            if n_dead > 0 and auxk > 0:
                k_aux = min(auxk, n_dead)
                pre_dead = pre * dead_mask.float()
                av, ai = torch.topk(pre_dead, k_aux, dim=1)
                h_aux = torch.zeros_like(pre)
                h_aux.scatter_(1, ai, av)
                residual = (batch - x_hat).detach()
                aux = auxk_w * F.mse_loss(model.decoder(h_aux), residual + x_hat.detach())
            else:
                aux = torch.tensor(0.0, device=device)
            (rec + aux).backward()
            with torch.no_grad():
                model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0)
            opt.step()
            total_rec += rec.item()
            total_aux += aux.item()
        n = len(loader)
        n_dead_ep = (model.fire_counts < dead_threshold).sum().item()
        print(f"  [TopK-SAE] ep {ep+1:02d}/{epochs}  rec={total_rec/n:.5f}  "
              f"aux={total_aux/n:.6f}  dead={n_dead_ep}/{model._enc.out_features}")
    return model.eval().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  JumpReLU-SAE  (Rajamanoharan et al. 2024)
# ─────────────────────────────────────────────────────────────────────────────

class JumpReLUSAE(nn.Module):
    """
    JumpReLU SAE with a learnable per-neuron threshold θ.
    Forward:  h_i = pre_i  if pre_i > θ_i  else  0
    Because the Heaviside step is not differentiable, we use a
    straight-through estimator (STE): ∂L/∂θ is passed through as-is.
    
    Loss = MSE(x_hat, x)  +  λ * ||h||_0
    The L0 term is approximated differentiably via a bandwidth-limited
    rectangle function: ∂step/∂θ ≈ rect(pre - θ, bandwidth) / bandwidth.
    """
    def __init__(self, d_in=D_MODEL, d_sae=D_SAE, bandwidth=0.001, l0_target=TOP_K):
        super().__init__()
        self.bandwidth  = bandwidth
        self.l0_target  = l0_target
        self._enc       = nn.Linear(d_in, d_sae)
        self.decoder    = nn.Linear(d_sae, d_in, bias=False)
        # log-threshold (initialised so exp(log_theta)≈0.1 — activations are ~N(0,1))
        self.log_theta  = nn.Parameter(torch.full((d_sae,), -2.3))   # exp(-2.3)≈0.1

    def _normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        pre   = F.relu(self._enc(x))          # (B, d_sae)
        theta = torch.exp(self.log_theta)      # (d_sae,) always positive
        # straight-through: forward uses hard step, backward approximates it
        hard  = (pre > theta).float()
        # STE gradient for theta:
        #   d(step)/d(theta) ≈ -rect((pre - theta)/bandwidth) / bandwidth
        ste   = hard - (
            (torch.abs(pre - theta) < self.bandwidth / 2).float()
        ).detach() + (
            (torch.abs(pre - theta) < self.bandwidth / 2).float()
        )
        h = pre * ste                          # zeroed where pre ≤ theta
        return h

    def forward(self, x):
        h = self.encode(x)
        return h, self.decoder(h)


def train_jumprelu(model, X, l0_lam=1e-4, epochs=EPOCHS, device=DEVICE):
    """
    l0_lam: penalty weight on the L0 sparsity (expected active features).
    The L0 is approximated differentiably; we scale it to encourage the
    model to reach roughly l0_target active features per token.
    """
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(X)
    for ep in range(epochs):
        model.train()
        total_rec = total_l0 = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            h, x_hat = model(batch)
            rec = F.mse_loss(x_hat, batch)
            # differentiable L0: count expected non-zeros using the STE approx
            theta = torch.exp(model.log_theta)
            pre   = F.relu(model._enc(batch))
            # approx indicator: sigmoid with sharp bandwidth
            approx_l0 = torch.sigmoid((pre - theta) / model.bandwidth).sum(dim=1).mean()
            l0_penalty = l0_lam * torch.abs(approx_l0 - model.l0_target)
            (rec + l0_penalty).backward()
            with torch.no_grad():
                model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0)
            opt.step()
            total_rec += rec.item()
            total_l0  += approx_l0.item()
        n = len(loader)
        print(f"  [JumpReLU] ep {ep+1:02d}/{epochs}  rec={total_rec/n:.5f}  "
              f"avg_l0≈{total_l0/n:.1f}  target={model.l0_target}")
    return model.eval().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CobwebTopK
#     Build a CobwebContinuousTree on the whitened activations, extract
#     D_SAE BFS nodes, encode each token as log P(node | x), then sparsify
#     to TOP_K non-zeros per row.
# ─────────────────────────────────────────────────────────────────────────────

def bfs_first_n_nodes(root, n):
    """Return first n node objects via BFS (excluding root)."""
    nodes, queue = [], [root]
    while queue and len(nodes) < n:
        node = queue.pop(0)
        for child in node.children:
            if len(nodes) >= n:
                break
            nodes.append(child)
            queue.append(child)
    return nodes


def encode_cobweb_logpost(instances, nodes):
    """(N, len(nodes)) matrix of log P(node | x), float64."""
    _empty = np.zeros(0, dtype=np.float32)
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, x in enumerate(instances):
            out[i, j] = node.log_prob(x, _empty)
    return out


def topk_sparsify(Z, k):
    """Zero all but the k largest entries per row."""
    out = np.zeros_like(Z)
    idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    rows = np.arange(Z.shape[0])[:, None]
    out[rows, idx] = Z[rows, idx]
    return out


def build_and_encode_cobweb(X_train_cob, X_train_full, X_test, k=TOP_K, n_nodes=COB_NODES):
    print(f"  Building CobwebContinuousTree on {len(X_train_cob):,} samples …")
    tree = CobwebContinuousTree(size=D_MODEL, covar_from=1, num_labels=0)
    _empty = np.zeros(0, dtype=np.float32)
    for i, x in enumerate(X_train_cob):
        tree.ifit(x, _empty)
        if (i + 1) % 10_000 == 0:
            print(f"    {i+1}/{len(X_train_cob)} inserted")
    print(f"  Extracting {n_nodes} BFS nodes …")
    nodes = bfs_first_n_nodes(tree.root, n_nodes)
    actual_n = len(nodes)
    print(f"  Encoding train ({len(X_train_full):,}) …")
    _sc = StandardScaler()
    Z_tr  = _sc.fit_transform(encode_cobweb_logpost(X_train_full, nodes))
    Z_tr  = topk_sparsify(Z_tr, k).astype(np.float32)
    print(f"  Encoding test ({len(X_test):,}) …")
    Z_te  = _sc.transform(encode_cobweb_logpost(X_test, nodes))
    Z_te  = topk_sparsify(Z_te, k).astype(np.float32)
    return Z_tr, Z_te, tree, nodes, actual_n


# ─────────────────────────────────────────────────────────────────────────────
# Train everything
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Training on {DEVICE}. D_MODEL={D_MODEL}, D_SAE={D_SAE}, TOP_K={TOP_K}")
print(f"{'='*60}\n")

print("─── L1-SAE ────────────────────────────────────────────────")
l1sae = train_l1sae(L1SAE(), X_train)
torch.save(l1sae.state_dict(), os.path.join(MODEL_DIR, "l1sae.pt"))
with torch.no_grad():
    _Xt  = torch.tensor(X_train, dtype=torch.float32)
    _Xte = torch.tensor(X_test,  dtype=torch.float32)
    Z_l1sae_train = l1sae.encode(_Xt).numpy()
    Z_l1sae_test  = l1sae.encode(_Xte).numpy()
np.save(os.path.join(MODEL_DIR, "Z_l1sae_train.npy"), Z_l1sae_train)
np.save(os.path.join(MODEL_DIR, "Z_l1sae_test.npy"),  Z_l1sae_test)
print(f"  Saved L1-SAE  →  latent shape: {Z_l1sae_train.shape}")

print("\n─── TopK-SAE ──────────────────────────────────────────────")
topksae = train_topksae(TopKSAE(), X_train)
torch.save(topksae.state_dict(), os.path.join(MODEL_DIR, "topksae.pt"))
with torch.no_grad():
    Z_topksae_train = topksae.encode(_Xt).numpy()
    Z_topksae_test  = topksae.encode(_Xte).numpy()
np.save(os.path.join(MODEL_DIR, "Z_topksae_train.npy"), Z_topksae_train)
np.save(os.path.join(MODEL_DIR, "Z_topksae_test.npy"),  Z_topksae_test)
print(f"  Saved TopK-SAE  →  latent shape: {Z_topksae_train.shape}")

print("\n─── JumpReLU-SAE ──────────────────────────────────────────")
jumprelu = train_jumprelu(JumpReLUSAE(), X_train)
torch.save(jumprelu.state_dict(), os.path.join(MODEL_DIR, "jumprelu.pt"))
with torch.no_grad():
    Z_jumprelu_train = jumprelu.encode(_Xt).numpy()
    Z_jumprelu_test  = jumprelu.encode(_Xte).numpy()
np.save(os.path.join(MODEL_DIR, "Z_jumprelu_train.npy"), Z_jumprelu_train)
np.save(os.path.join(MODEL_DIR, "Z_jumprelu_test.npy"),  Z_jumprelu_test)
print(f"  Saved JumpReLU  →  latent shape: {Z_jumprelu_train.shape}")

print("\n─── Cobweb-TopK ───────────────────────────────────────────")
Z_cobweb_train, Z_cobweb_test, cob_tree, cob_nodes, actual_cob_n = build_and_encode_cobweb(
    X_train[:COB_TRAIN], X_train, X_test
)
np.save(os.path.join(MODEL_DIR, "Z_cobweb_train.npy"), Z_cobweb_train)
np.save(os.path.join(MODEL_DIR, "Z_cobweb_test.npy"),  Z_cobweb_test)
print(f"  Saved Cobweb-TopK  →  latent shape: {Z_cobweb_train.shape}  (nodes={actual_cob_n})")

# ── Save metadata ─────────────────────────────────────────────────────────────

meta = {
    "d_model":    D_MODEL,
    "d_sae":      D_SAE,
    "top_k":      TOP_K,
    "auxk":       AUXK,
    "auxk_w":     AUXK_W,
    "l1_lam":     L1_LAM,
    "epochs":     EPOCHS,
    "batch":      BATCH,
    "lr":         LR,
    "cob_train":  COB_TRAIN,
    "cob_nodes":  actual_cob_n,
    "n_train":    int(len(X_train)),
    "n_test":     int(len(X_test)),
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nAll models saved to {MODEL_DIR}")
