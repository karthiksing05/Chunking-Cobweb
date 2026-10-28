"""Chess state compression, stage B (docs/CHESS_CS.md): representation learners
and one value head for every representation. Runs in a torch environment on the
file chess_compress_a.py wrote.

Representation learners, each trained on the same ``n_rep`` unlabelled positions
(768 piece-square indicators in, never the values):
  ae          MLP autoencoder, 768 -> 256 -> d -> 256 -> 768 (binary cross-entropy)
  vqvae       the same encoder into ``m`` slots, each quantised to one of ``K``
              codes (straight-through, commitment 0.25); a position is m code
              indices, m log2 K bits
  contrastive SimCLR-style encoder, 768 -> 256 -> d; two views of a position
              drop 25% of its pieces each (NT-Xent, temperature 0.2)
Plus raw (768 indicators) and trellis (concept indicators + unrecognised pieces,
from stage A).

Value head: ridge regression on the representation (lambda by 5-fold CV on the
training labels), the same for every representation; and a small MLP head
(one hidden layer of 128) as a check. Reported on the held-out labelled
positions: R^2, Spearman, and the representation's bits per position.

Bits: raw = pieces x log2 768; trellis from stage A; ae and contrastive = d x 8
(each dimension quantised to 8 bits); vqvae = m x log2 K.

Usage: chess_compress_b.py <stage-A npz> <tag> [d m K epochs]
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

path, tag = sys.argv[1], sys.argv[2]
d, m, K, epochs = [int(x) for x in sys.argv[3:7]] if len(sys.argv) > 6 else [32, 8, 64, 60]
torch.manual_seed(0)
np.random.seed(0)
z = np.load(path)
boards = torch.tensor(z["boards"])
n_rep, n_train = int(z["n_rep"]), int(z["n_train"])
values = z["values"].astype(np.float64)
rep, lab = boards[:n_rep], boards[n_rep:]
y_tr, y_te = values[:n_train], values[n_train:]


def mlp(sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


def batches(n, bs=128):
    idx = torch.randperm(n)
    return [idx[i:i + bs] for i in range(0, n, bs)]


def train_ae():
    enc, dec = mlp([768, 256, d]), mlp([d, 256, 768])
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3)
    for _ in range(epochs):
        for b in batches(n_rep):
            x = rep[b]
            loss = F.binary_cross_entropy_with_logits(dec(enc(x)), x)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return enc(lab).numpy(), d * 8.0, float(loss)


def train_vq():
    enc, dec = mlp([768, 256, m * 16]), mlp([m * 16, 256, 768])
    book = nn.Parameter(torch.randn(K, 16) * 0.1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [book], 1e-3)

    def quant(h):
        h = h.view(-1, m, 16)
        dist = (h.unsqueeze(2) - book.view(1, 1, K, 16)).pow(2).sum(-1)
        idx = dist.argmin(-1)
        q = book[idx]
        return h + (q - h).detach(), q, h, idx
    for _ in range(epochs):
        for b in batches(n_rep):
            x = rep[b]
            st, q, h, _ = quant(enc(x))
            loss = F.binary_cross_entropy_with_logits(dec(st.reshape(len(b), -1)), x) + \
                F.mse_loss(q, h.detach()) + 0.25 * F.mse_loss(h, q.detach())
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, _, _, idx = quant(enc(lab))
        onehot = F.one_hot(idx, K).float().reshape(len(lab), -1).numpy()
        used = len(set(idx.flatten().tolist()))
    return onehot, m * math.log2(K), float(loss), used


def train_contrastive():
    enc = mlp([768, 256, d])
    opt = torch.optim.Adam(enc.parameters(), 1e-3)

    def view(x):
        return x * (torch.rand_like(x) > 0.25).float()
    for _ in range(epochs):
        for b in batches(n_rep, 256):
            x = rep[b]
            z1, z2 = F.normalize(enc(view(x)), dim=1), F.normalize(enc(view(x)), dim=1)
            zz = torch.cat([z1, z2])
            sim = zz @ zz.T / 0.2
            sim.fill_diagonal_(-1e9)
            n = len(b)
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
            loss = F.cross_entropy(sim, target)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return enc(lab).numpy(), d * 8.0, float(loss)


def ridge(X):
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0) + 1e-6
    X = (X - mu) / sd
    Xtr, Xte = X[:n_train], X[n_train:]
    folds = np.array_split(np.random.RandomState(0).permutation(n_train), 5)
    best = None
    for lam in (0.1, 1, 10, 100, 1000, 10000):
        err = 0.0
        for fo in folds:
            tr = np.setdiff1d(np.arange(n_train), fo)
            w = solve(Xtr[tr], y_tr[tr], lam)
            err += ((Xtr[fo] @ w[:-1] + w[-1] - y_tr[fo]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, lam)
    w = solve(Xtr, y_tr, best[1])
    return Xte @ w[:-1] + w[-1], best[1]


def solve(X, y, lam):
    A = np.hstack([X, np.ones((len(X), 1))])
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def mlp_head(X):
    X = torch.tensor(np.asarray(X, dtype=np.float32))
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0) + 1e-6
    X = (X - mu) / sd
    net = mlp([X.shape[1], 128, 1])
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(200):
        for b in batches(n_train):
            loss = F.mse_loss(net(X[:n_train][b]).squeeze(1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(X[n_train:]).squeeze(1).numpy().astype(np.float64)


def scores(pred):
    r2 = 1 - ((pred - y_te) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
    rk = lambda v: np.argsort(np.argsort(v))
    sp = np.corrcoef(rk(pred), rk(y_te))[0, 1]
    return round(float(r2), 3), round(float(sp), 3)


reps = {"raw": (lab.numpy(), float(lab[n_train:].sum(1).mean()) * math.log2(768), {}),
        "trellis": (z["trellis"], float(z["trellis_bits"][n_train:].mean()), {"units": float(z["trellis_units"][n_train:].mean())})}
reps["material"] = (lab.numpy().reshape(len(lab), 12, 64).sum(-1), 12 * math.log2(9), {})   # piece counts: a reference, not a state code
X, b, l = train_ae()
reps["ae"] = (X, b, {"loss": round(l, 4)})
X, b, l, used = train_vq()
reps["vqvae"] = (X, b, {"loss": round(l, 4), "codes used": used})
X, b, l = train_contrastive()
reps["contrastive"] = (X, b, {"loss": round(l, 4)})
rows = []
for name, (X, bits, extra) in reps.items():
    pred, lam = ridge(X)
    r2, sp = scores(pred)
    r2m, spm = scores(mlp_head(X))
    row = {"tag": tag, "representation": name, "dims": int(np.asarray(X).shape[1]), "bits per position": round(bits, 1),
           "ridge R2": r2, "ridge spearman": sp, "lambda": lam, "mlp R2": r2m, "mlp spearman": spm, **extra}
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(os.path.splitext(path)[0] + f"_{tag}_b.json", "w"), indent=1)
