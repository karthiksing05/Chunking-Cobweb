"""Track 3, the neural reference: a small GRU language model over IDS tokens.

Reads ``results/characters_memory/data.json`` (written by
``run_characters_memory.py prepare``), and for each corpus (real, shuffled,
slot-shuffled) and hidden size trains a one-layer GRU on 90% of the learned
characters, early-stopped on the other 10%. Reported: held-out bits per
character (with the end symbol), and a two-part total: the training
characters' code under the weights quantized to 8 bits (per-tensor min-max)
plus 8 bits per parameter and 64 bits per tensor for its range. With the
hidden size chosen on the real characters' validation split, the same
completion tasks as the TRELLIS jobs: masked component, closed-set cue and
open cue (100 draws ranked by frequency).

Runs in the ``chunks`` env (torch):
    ~/flash/miniconda3/envs/chunks/bin/python experiments/v2/run_characters_memory_neural.py
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "characters_memory")
CONDS = ["real", "shuffled", "slot_shuffled"]
OPERATORS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
ARITY = {o: 2 for o in OPERATORS}
ARITY["⿲"] = ARITY["⿳"] = 3
LN2 = math.log(2)


def parse_ok(tokens) -> bool:
    pos = 0

    def node():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError
        t = tokens[pos]
        pos += 1
        if t in OPERATORS:
            for _ in range(ARITY[t]):
                node()
    try:
        node()
    except (ValueError, RecursionError):
        return False
    return pos == len(tokens)


class LM(nn.Module):
    def __init__(self, V, h):
        super().__init__()
        self.emb = nn.Embedding(V, h)
        self.rnn = nn.GRU(h, h, batch_first=True)
        self.out = nn.Linear(h, V)

    def forward(self, x):
        y, _ = self.rnn(self.emb(x))
        return self.out(y)


class Vocab:
    def __init__(self, train):
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted({t for s in train for t in s})
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, s):
        return [1] + [self.stoi.get(t, 3) for t in s] + [2]


def batches(seqs, vocab, bs, shuffle, rng=None):
    idx = list(range(len(seqs)))
    if shuffle:
        rng.shuffle(idx)
    for k in range(0, len(idx), bs):
        enc = [vocab.encode(seqs[i]) for i in idx[k:k + bs]]
        L = max(len(e) for e in enc)
        x = torch.zeros(len(enc), L, dtype=torch.long)
        for i, e in enumerate(enc):
            x[i, :len(e)] = torch.tensor(e)
        yield x


@torch.no_grad()
def nll_each(model, seqs, vocab, bs=512):
    """ln P of every sequence (nats), in order."""
    model.eval()
    out = []
    for x in batches(seqs, vocab, bs, False):
        logits = model(x[:, :-1])
        lp = torch.log_softmax(logits, -1).gather(-1, x[:, 1:, None])[..., 0]
        mask = (x[:, 1:] != 0).float()
        out += (lp * mask).sum(1).tolist()
    return out


def train_lm(train, val, vocab, h, seed, max_epochs=300, patience=15):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LM(len(vocab.itos), h)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    for ep in range(max_epochs):
        model.train()
        for x in batches(train, vocab, 64, True, rng):
            logits = model(x[:, :-1])
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = -sum(nll_each(model, val, vocab))
        if v < best - 1e-6:
            best, best_state, bad, epochs = v, copy.deepcopy(model.state_dict()), 0, ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best / LN2 / len(val), epochs


def quantized(model):
    q = copy.deepcopy(model)
    n_tensors = 0
    with torch.no_grad():
        for p in q.parameters():
            lo, hi = p.min(), p.max()
            step = (hi - lo) / 255 if hi > lo else torch.tensor(1.0)
            p.copy_(torch.round((p - lo) / step) * step + lo)
            n_tensors += 1
    return q, n_tensors


def entropy_coded(model, bits: int):
    """Weights quantized to ``bits`` bits (per-tensor min-max), and the length
    of their adaptive (KT, add-1/2) code: the Dirichlet-multinomial marginal
    of each tensor's level counts, plus 64 bits per tensor for its range."""
    q = copy.deepcopy(model)
    L = 2 ** bits
    total = 0.0
    with torch.no_grad():
        for p in q.parameters():
            lo, hi = p.min(), p.max()
            step = (hi - lo) / (L - 1) if hi > lo else torch.tensor(1.0)
            idx = torch.round((p - lo) / step)
            p.copy_(idx * step + lo)
            counts = np.bincount(idx.flatten().long().numpy(), minlength=L)
            n = counts.sum()
            nats = (math.lgamma(n + L / 2) - math.lgamma(L / 2)
                    - sum(math.lgamma(c + 0.5) - math.lgamma(0.5) for c in counts if c > 0))
            total += nats / LN2 + 64
    return q, total


def rank_stats(hits, sizes) -> dict:
    """As ``run_characters_memory.rank_stats``."""
    multi = [(h, n) for h, n in zip(hits, sizes) if n >= 2]
    pct = [((h[2] - 1) / (n - 1)) if h[2] > 0 else 1.0 for h, n in multi]
    return {"queries with >= 2 candidates": len(multi),
            "top1 (>= 2 candidates)": float(np.mean([h[0] for h, _ in multi])),
            "chance top1 (>= 2 candidates)": float(np.mean([1 / n for _, n in multi])),
            "mean percentile rank (>= 2 candidates; chance 0.5)": float(np.mean(pct)),
            "MRR": float(np.mean([1 / h[2] if h[2] > 0 else 0.0 for h in hits])),
            "ranks": [int(h[2]) for h in hits], "sizes": [int(n) for n in sizes]}


def topk(ranked, truth):
    r = ranked.index(truth) + 1 if truth in ranked else 0
    return r == 1, 0 < r <= 5, r


@torch.no_grad()
def prefix_lp(model, vocab, prompts):
    """ln P(the sequence begins with the prompt): BOS, then the prompt's tokens, no end."""
    model.eval()
    out = []
    for p in prompts:
        x = torch.tensor([vocab.encode(p)[:-1]])
        lp = torch.log_softmax(model(x[:, :-1]), -1).gather(-1, x[:, 1:, None])[..., 0]
        out.append(float(lp.sum()))
    return out


@torch.no_grad()
def sample(model, vocab, prompt, n, rng, max_len=20):
    model.eval()
    out = []
    for _ in range(n):
        seq = list(prompt)
        x = torch.tensor([vocab.encode(prompt)[:-1]])
        logits, hstate = None, None
        y, hstate = model.rnn(model.emb(x))
        logits = model.out(y[:, -1])
        while len(seq) < max_len:
            p = torch.softmax(logits[0], -1).numpy().astype(float)
            p[0] = p[1] = p[3] = 0.0
            p /= p.sum()
            t = int(rng.choice(len(p), p=p))
            if t == 2:
                break
            seq.append(vocab.itos[t])
            y, hstate = model.rnn(model.emb(torch.tensor([[t]])), hstate)
            logits = model.out(y[:, -1])
        out.append(tuple(seq))
    return out


def completion(model, vocab, data, draws, seed):
    corpus = data["corpus"]
    res = {}
    # Masked component.
    voc = data["mask_vocab"]
    hits = []
    for q in data["mask"]:
        cands = []
        for v in voc:
            t = list(q["tokens"])
            t[q["position"]] = v
            cands.append(t)
        lp = nll_each(model, cands, vocab)
        ranked = [tuple(cands[i]) for i in np.argsort(-np.array(lp), kind="stable")]
        hits.append(topk(ranked, tuple(q["tokens"])))
    res["masked"] = {"top1": float(np.mean([h[0] for h in hits])), "top5": float(np.mean([h[1] for h in hits])),
                     "queries": len(hits), **rank_stats(hits, [len(voc)] * len(hits))}
    # Closed-set cue.
    scores = nll_each(model, [c["tokens"] for c in corpus], vocab)
    prompts = defaultdict(list)
    for i, c in enumerate(corpus):
        t = c["tokens"]
        if t[0] not in OPERATORS:
            continue
        # operator + first part: the shortest complete structure after the operator
        for k in range(2, len(t) + 1):
            if parse_ok(t[1:k]):
                prompts[tuple(t[:k])].append(i)
                break
    res["cue closed"] = {}
    for which in ("all", "unseen"):
        hs, sizes = [], []
        for q in data["cue"]:
            ids = [i for i in prompts[tuple(q["prompt"])] if which == "all" or not corpus[i]["trained"]]
            ranked = [tuple(corpus[i]["tokens"]) for i in sorted(ids, key=lambda i: -scores[i])]
            hs.append(topk(ranked, tuple(q["truth"])))
            sizes.append(len(ids))
        res["cue closed"][which] = {"top1": float(np.mean([h[0] for h in hs])), "top5": float(np.mean([h[1] for h in hs])),
                                    "mean candidates": float(np.mean(sizes)), "queries": len(hs),
                                    **rank_stats(hs, sizes)}
    # Cue bits: -log2 P(character | its cue).
    full = nll_each(model, [q["truth"] for q in data["cue"]], vocab)
    pre = prefix_lp(model, vocab, [q["prompt"] for q in data["cue"]])
    bits = [-(a - b) / LN2 for a, b in zip(full, pre)]
    res["cue bits"] = {"mean bits": float(np.mean(bits)), "median bits": float(np.median(bits)), "queries": len(bits),
                       "per query": [float(x) for x in bits]}
    # Open cue.
    real_forms = {tuple(c["tokens"]) for c in corpus}
    learned = {tuple(c["tokens"]) for c in corpus if c["trained"]}
    rng = np.random.default_rng(seed)
    hs, wf, real, real_new = [], [], [], []
    for q in data["cue"]:
        s = sample(model, vocab, q["prompt"], draws, rng)
        wf.append(np.mean([parse_ok(x) for x in s]))
        real.append(np.mean([x in real_forms for x in s]))
        real_new.append(np.mean([x in real_forms and x not in learned for x in s]))
        ranked = [x for x, _ in Counter(s).most_common()]
        hs.append(topk(ranked, tuple(q["truth"])))
    res["cue open"] = {"top1": float(np.mean([h[0] for h in hs])), "top5": float(np.mean([h[1] for h in hs])),
                       "truth drawn at all": float(np.mean([h[2] > 0 for h in hs])),
                       "draws that are real characters": float(np.mean(real)),
                       "draws that are real unlearned characters": float(np.mean(real_new)),
                       "well formed draws": float(np.mean(wf)), "queries": len(hs), "draws": draws}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--hidden", default="8,16,32,64,128")
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--suffix", default="", help="written to neural<suffix>.json")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    with open(os.path.join(args.out, "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    seed = data["seed"]
    res = {"models": {}, "note": "train on 90% of the learned characters, early-stopped on 10%"}
    kept = {}
    for h in [int(x) for x in args.hidden.split(",")]:
        res["models"][str(h)] = {}
        for cond in CONDS:
            t0 = time.time()
            tr, te = data["conds"][cond]["train"], data["conds"][cond]["test"]
            n_val = len(tr) // 10
            vocab = Vocab(tr)
            model, val_bits, epochs = train_lm(tr[n_val:], tr[:n_val], vocab, h, seed)
            n_params = sum(p.numel() for p in model.parameters())
            qm, n_tensors = quantized(model)
            r = {"parameters": n_params, "epochs": epochs, "validation bits per character": val_bits,
                 "held-out bits per character": -sum(nll_each(model, te, vocab)) / LN2 / len(te),
                 "held-out bits per character (8-bit)": -sum(nll_each(qm, te, vocab)) / LN2 / len(te),
                 "training bits per character (8-bit)": -sum(nll_each(qm, tr, vocab)) / LN2 / len(tr),
                 "parameter bits per character (8-bit)": (8 * n_params + 64 * n_tensors) / len(tr),
                 "seconds": time.time() - t0}
            r["total bits per character (8-bit)"] = r["training bits per character (8-bit)"] + r["parameter bits per character (8-bit)"]
            ec = {}
            for b in (2, 3, 4, 5, 6, 8):
                qb, pbits = entropy_coded(model, b)
                train_b = -sum(nll_each(qb, tr, vocab)) / LN2 / len(tr)
                ec[str(b)] = {"training bits per character": train_b, "parameter bits per character": pbits / len(tr),
                              "total bits per character": train_b + pbits / len(tr),
                              "held-out bits per character": -sum(nll_each(qb, te, vocab)) / LN2 / len(te)}
            best_b = min(ec, key=lambda b: ec[b]["total bits per character"])
            r["entropy-coded weights"] = ec
            r["best entropy-coded total bits per character"] = ec[best_b]["total bits per character"]
            r["best entropy-coded bit depth"] = int(best_b)
            res["models"][str(h)][cond] = r
            if cond == "real":
                kept[h] = (val_bits, model, vocab)
            print(h, cond, json.dumps(r), flush=True)
    best_h = min(kept, key=lambda h: kept[h][0])
    _, model, vocab = kept[best_h]
    t0 = time.time()
    res["completion"] = completion(model, vocab, data, args.draws, seed)
    res["completion"]["hidden"] = best_h
    res["completion"]["seconds"] = time.time() - t0
    print("completion done", flush=True)
    with open(os.path.join(args.out, f"neural{args.suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
