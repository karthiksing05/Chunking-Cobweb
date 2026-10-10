"""Neural storage baselines for chess positions (Track 2; torch side).

Reads data/chess/memory/boards.npz (written by run_chess_memory.py --export:
the seed-13 permutation of run_chess.py, so the first N boards are the
training positions and the next 500 the held-out ones; 'random' holds each
position's random counterpart). Does not import trellis2.

Two model families, each swept over size:
  ar  an autoregressive transformer over the 64 squares in the read's order
      (a1 b1 ... h8), 13-way per square, learned positional embeddings.
  ae  an autoencoder whose k-dim latent is quantized to 5 levels per dimension
      (finite scalar quantization, straight-through); a factorized learned
      prior codes the latent and the decoder gives 64 independent 13-way
      distributions. Lossless code of a board = -log2 prior(z) - log2 p(board|z),
      z the encoder's (deterministic) latent. The receiver needs only the
      decoder and the prior, so those are the parameters paid for; the
      encoder (sender side) is reported but not charged.

Per (family, kind, size):
  prequential  train on growing prefixes BOUNDS (early stopping on the last
               10% of the prefix), code each next block with the model
               trained on what came before; the first block (16 boards) at
               64 log2 13 bits each. The code length of the first N boards
               needs no parameter cost (Blier and Ollivier 2018). The prefix
               models trained on exactly N boards also give held-out bits.
  two-part     train on all N boards (fixed step budget, no early stopping:
               the objective is the training code), quantize the transmitted
               parameters to b bits per weight (per-tensor symmetric uniform,
               plus a 32-bit scale per tensor), and report the quantized
               model's training NLL + parameter bits, for b = 8 and 16.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "chess_memory")
BOARDS = os.path.join(ROOT, "data", "chess", "memory", "boards.npz")
BOUNDS = [16, 32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
NS = (1000, 4000, 16000, 32000)
TEST = 500
LN2 = math.log(2)
UNIFORM = 64 * math.log2(13)

# name: (d_model, layers, heads)
AR_SIZES = {"ar1": (4, 1, 1), "ar2": (8, 1, 1), "ar3": (16, 2, 2), "ar4": (32, 2, 2),
            "ar5": (64, 3, 4), "ar6": (128, 4, 4)}
# name: (decoder hidden, latent dims k, encoder hidden)
AE_SIZES = {"ae1": (2, 8, 128), "ae2": (8, 16, 128), "ae3": (32, 24, 256), "ae4": (128, 32, 512),
            "ae5": (512, 48, 1024), "ae6": (1024, 64, 1024)}


class AR(nn.Module):
    def __init__(self, d, L, h):
        super().__init__()
        self.tok = nn.Embedding(14, d)          # 13 values + a start token
        self.pos = nn.Parameter(torch.randn(64, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 13)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(64), persistent=False)

    def bits(self, x):
        """Bits of each board (B,) under the model."""
        inp = torch.cat([torch.full_like(x[:, :1], 13), x[:, :-1]], 1)
        h = self.blocks(self.tok(inp) + self.pos, mask=self.mask, is_causal=True)
        logits = self.head(self.ln(h))
        return F.cross_entropy(logits.transpose(1, 2), x, reduction="none").sum(1) / LN2

    def transmitted(self):
        return [n for n, _ in self.named_parameters()]


class AE(nn.Module):
    LEVELS = 5

    def __init__(self, h, k, he):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Linear(64 * 13, he), nn.GELU(), nn.Linear(he, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, 64 * 13))
        self.prior = nn.Parameter(torch.zeros(k, self.LEVELS))

    def latent(self, x):
        z = (self.LEVELS // 2) * torch.tanh(self.enc(F.one_hot(x, 13).float().flatten(1)))
        zq = z + (torch.round(z) - z).detach()
        return zq, torch.round(z).long() + self.LEVELS // 2

    def parts(self, x):
        """Latent bits under the prior and decoder bits, per board. The rate's
        value is exact (the hard level); its gradient reaches the encoder
        through a soft assignment of z to the levels (straight-through), so
        training minimizes the whole lossless code."""
        h = (self.LEVELS // 2) * torch.tanh(self.enc(F.one_hot(x, 13).float().flatten(1)))
        zq = h + (torch.round(h) - h).detach()
        levels = torch.arange(self.LEVELS, device=x.device, dtype=h.dtype) - self.LEVELS // 2
        soft = torch.softmax(-(h.unsqueeze(-1) - levels) ** 2 / 0.5, -1)
        hard = F.one_hot(torch.round(h).long() + self.LEVELS // 2, self.LEVELS).to(h.dtype)
        assign = hard + soft - soft.detach()
        rate = -(assign * torch.log_softmax(self.prior, -1)).sum((1, 2)) / LN2
        logits = self.dec(zq).view(-1, 64, 13)
        nll = F.cross_entropy(logits.transpose(1, 2), x, reduction="none").sum(1) / LN2
        return rate, nll

    def bits(self, x):
        rate, nll = self.parts(x)
        return rate + nll

    def transmitted(self):
        return [n for n, _ in self.named_parameters() if not n.startswith("enc.")]


def build(name):
    if name in AR_SIZES:
        return AR(*AR_SIZES[name])
    return AE(*AE_SIZES[name])


def n_params(model, names=None):
    names = set(model.transmitted() if names is None else names)
    return sum(p.numel() for n, p in model.named_parameters() if n in names)


@torch.no_grad()
def total_bits(model, X, bs=2048):
    model.eval()
    out = sum(float(model.bits(X[i:i + bs]).double().sum()) for i in range(0, len(X), bs))
    return out


@torch.no_grad()
def ae_parts(model, X):
    model.eval()
    rate, nll = model.parts(X)
    return float(rate.double().mean()), float(nll.double().mean())


def quantized(model, b):
    """A copy whose transmitted parameters are rounded to b bits (per-tensor
    symmetric uniform grid) and the parameter bits that costs."""
    q = copy.deepcopy(model)
    keep = set(q.transmitted())
    bits = 0
    with torch.no_grad():
        for n, p in q.named_parameters():
            if n not in keep:
                continue
            s = p.abs().max() / (2 ** (b - 1) - 1)
            if s > 0:
                p.copy_(torch.round(p / s) * s)
            bits += b * p.numel() + 32
    return q, bits


def train(model, X, Xval=None, steps=5000, lr=3e-3, bs=256, patience=8, seed=0):
    """Fixed budget with cosine decay (Xval None), or early stopping on Xval
    (constant rate after warm-up, best state restored)."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    bs = min(bs, len(X))
    warm = min(100, steps // 10)
    every = max(50, math.ceil(len(X) / bs))
    best, best_state, bad = float("inf"), None, 0
    step = 0
    for step in range(steps):
        frac = step / steps
        scale = min(1.0, (step + 1) / warm) if Xval is not None else \
            min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * frac))
        for grp in opt.param_groups:
            grp["lr"] = lr * scale
        model.train()
        idx = torch.randint(len(X), (bs,), generator=g, device=X.device)
        loss = model.bits(X[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if Xval is not None and (step + 1) % every == 0:
            v = total_bits(model, Xval) / len(Xval)
            if v < best - 1e-4:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if Xval is not None and best_state is not None:
        model.load_state_dict(best_state)
    return step + 1


def init_from(model, X):
    """AE: start the decoder's output bias at each square's smoothed marginal
    (log frequencies of the training boards), so that a latent carrying
    nothing already gives the per-square model."""
    if isinstance(model, AE):
        c = torch.zeros(64, 13, device=X.device)
        c.scatter_add_(1, X.t(), torch.ones_like(X.t(), dtype=c.dtype))
        with torch.no_grad():
            model.dec[-1].bias.copy_(torch.log((c + 0.5) / (c.sum(1, keepdim=True) + 6.5)).reshape(-1))
    return model


def lr_for(model):
    return 3e-3 if n_params(model, [n for n, _ in model.named_parameters()]) < 50_000 else 1e-3


def run(name, kind, steps, out, dev, seed=0):
    X = torch.from_numpy(np.load(BOARDS)[kind].astype(np.int64)).to(dev)
    probe = build(name)
    res = {"model": name, "family": name[:2], "kind": kind, "config": (AR_SIZES | AE_SIZES)[name],
           "params (transmitted)": n_params(probe), "params (all)": sum(p.numel() for p in probe.parameters()),
           "steps": steps, "prequential": {}, "two-part": {}}
    print(name, kind, res["params (transmitted)"], "transmitted params", flush=True)

    # Prequential: blocks [BOUNDS[j], BOUNDS[j+1]) coded by the model trained on X[:BOUNDS[j]].
    t0 = time.time()
    block_bits = {f"0-{BOUNDS[0]}": BOUNDS[0] * UNIFORM}
    held = {}
    for j, P in enumerate(BOUNDS):
        torch.manual_seed(seed + j)
        nval = max(1, P // 10)
        m = init_from(build(name).to(dev), X[:P - nval])
        used = train(m, X[:P - nval], X[P - nval:P], steps=steps, lr=lr_for(m), seed=seed + j)
        entry = {"steps used": used}
        if j + 1 < len(BOUNDS):
            blk = X[P:BOUNDS[j + 1]]
            entry["block"] = f"{P}-{BOUNDS[j + 1]}"
            entry["block bits/pos"] = total_bits(m, blk) / len(blk)
            block_bits[entry["block"]] = entry["block bits/pos"] * len(blk)
        if P in NS:
            entry["held-out bits/pos"] = total_bits(m, X[P:P + TEST]) / TEST
            if name.startswith("ae"):
                entry["held-out latent, decoder bits/pos"] = ae_parts(m, X[P:P + TEST])
            held[P] = entry["held-out bits/pos"]
        res["prequential"][P] = entry
        print(f"  preq prefix {P}: {entry}", flush=True)
    cum, pre = 0.0, {}
    for key, b in block_bits.items():
        cum += b
        end = int(key.split("-")[1])
        if end in NS:
            pre[end] = cum / end
    res["prequential bits/pos"] = pre
    res["held-out bits/pos (early-stopped prefix model)"] = held
    res["prequential seconds"] = time.time() - t0

    # Two-part: fit all N, quantize.
    for N in NS:
        t0 = time.time()
        torch.manual_seed(seed + 100)
        m = init_from(build(name).to(dev), X[:N])
        train(m, X[:N], None, steps=steps, lr=lr_for(m), seed=seed + 100)
        Xt = X[N:N + TEST]
        e = {"float train bits/pos": total_bits(m, X[:N]) / N, "float held-out bits/pos": total_bits(m, Xt) / TEST}
        for b in (8, 16):
            q, pb = quantized(m, b)
            tb = total_bits(q, X[:N])
            e[f"b={b}"] = {"param bits": pb, "train data bits/pos": tb / N, "total bits/pos": (tb + pb) / N,
                           "held-out bits/pos": total_bits(q, Xt) / TEST}
        if name.startswith("ae"):
            e["train latent, decoder bits/pos (float)"] = ae_parts(m, X[:N])
        e["seconds"] = time.time() - t0
        res["two-part"][N] = e
        print(f"  two-part N={N}: {json.dumps(e)}", flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["ar", "ae"], required=True)
    ap.add_argument("--kind", choices=["real", "random"], required=True)
    ap.add_argument("--sizes", default="", help="comma-free list, e.g. ar1+ar2 (default: all)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--boards", default=BOARDS, help="npz with 'real' and 'random' (permuted) board arrays")
    ap.add_argument("--max-n", type=int, default=max(NS), help="largest N (the prefix bounds stop there)")
    args = ap.parse_args()
    global BOARDS, BOUNDS, NS
    BOARDS = args.boards
    BOUNDS = [b for b in BOUNDS if b <= args.max_n]
    NS = tuple(n for n in NS if n <= args.max_n)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sizes = args.sizes.split("+") if args.sizes else list(AR_SIZES if args.family == "ar" else AE_SIZES)
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in sizes:
        path = os.path.join(args.out, f"nets_{name}_{args.kind}.json")
        if os.path.exists(path):
            print("done already:", path)
            continue
        run(name, args.kind, args.steps, path, dev)


if __name__ == "__main__":
    main()
