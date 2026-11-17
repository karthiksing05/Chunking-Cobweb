"""Word-level causal LM reference baselines for t3_story.py (STORY_BASE arms {"lm": "gpt"|"lstm"}).

Trained only on the n training sentences, size picked by held-out per-token loss (words + </s>) on
sentences from stories the stream never reached; AdamW with early stopping; ancestral sampling (default
temperature 1; train_lm(temp=) rescales the logits, r184) from <s> to </s>.
"""
import copy
import itertools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2
MAXLEN = 10   # <s> + up to 9 tokens


class GPT(nn.Module):
    def __init__(self, V, layers, d, drop):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(MAXLEN + 2, d)
        L = nn.TransformerEncoderLayer(d, max(1, d // 32), 4 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(L, layers, enable_nested_tensor=False)
        self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, V)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return self.out(self.ln(self.enc(h, mask=mask)))


class LSTM(nn.Module):
    def __init__(self, V, layers, d, drop):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.rnn = nn.LSTM(d, d, layers, batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.drop, self.out = nn.Dropout(drop), nn.Linear(d, V)

    def forward(self, x):
        return self.out(self.drop(self.rnn(self.drop(self.emb(x)))[0]))


def _batch(sents, w2i, dev):
    T = max(len(s) for s in sents) + 2
    x = torch.full((len(sents), T), PAD, dtype=torch.long)
    for i, s in enumerate(sents):
        t = [BOS] + [w2i[w] for w in s] + [EOS]
        x[i, :len(t)] = torch.tensor(t)
    return x.to(dev)


def _loss(model, x, reduce=True):
    lg = model(x[:, :-1])
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), ignore_index=PAD, reduction="sum" if reduce else "none")


def _fit(kind, V, layers, d, drop, tr, ho, w2i, dev, seed, max_ep=80, patience=6, bs=64):
    torch.manual_seed(seed)
    model = (GPT if kind == "gpt" else LSTM)(V, layers, d, drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3 if kind == "gpt" else 3e-3, weight_decay=0.01)
    xho = _batch(ho, w2i, dev)
    ntok_ho = int((xho[:, 1:] != PAD).sum())
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, ep_best = 1e9, None, 0, 0
    for ep in range(max_ep):
        model.train()
        perm = torch.randperm(len(tr), generator=g).tolist()
        for i in range(0, len(perm), bs):
            x = _batch([tr[j] for j in perm[i:i + bs]], w2i, dev)
            loss = _loss(model, x) / int((x[:, 1:] != PAD).sum())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            h = float(_loss(model, xho)) / ntok_ho
        if h < best - 1e-4:
            best, best_state, bad, ep_best = h, copy.deepcopy(model.state_dict()), 0, ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, ep_best


def train_lm(kind, train_sents, held_sents, vocab, seed, log=print, temp=1.0):
    """Grid search by held-out loss; returns (sampler, info)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    words = sorted(vocab)
    w2i = {w: i + 3 for i, w in enumerate(words)}
    i2w = {i: w for w, i in w2i.items()}
    V = len(words) + 3
    grid = (list(itertools.product([1, 2, 4], [64, 128, 256], [0.1, 0.3])) if kind == "gpt"
            else list(itertools.product([1, 2], [128, 256], [0.1, 0.3])))
    t0, best = time.time(), None
    results = []
    for layers, d, drop in grid:
        model, h, ep = _fit(kind, V, layers, d, drop, train_sents, held_sents, w2i, dev, seed)
        results.append({"layers": layers, "d": d, "drop": drop, "heldout": round(h, 4), "epochs": ep})
        if best is None or h < best[1]:
            best = (model, h, {"layers": layers, "d": d, "drop": drop, "epochs": ep})
    model, h, cfg = best
    info = {"lm": kind, "chosen": cfg, "heldout per-token loss": round(h, 4), "heldout sentences": len(held_sents),
            "grid": results, "train s": round(time.time() - t0), "device": dev}
    gen = torch.Generator(device=dev).manual_seed(seed + 99)

    @torch.no_grad()
    def sample(m):
        """m ancestral samples at temperature 1, <s> to </s> (or 9 tokens), as word lists (None if cut off)."""
        x = torch.full((m, 1), BOS, dtype=torch.long, device=dev)
        done = torch.zeros(m, dtype=torch.bool, device=dev)
        for _ in range(9):
            p = F.softmax(model(x)[:, -1] / temp, -1)
            p[:, :2] = 0.0   # never PAD or BOS
            nxt = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=gen)
            nxt[done] = PAD
            x = torch.cat([x, nxt], 1)
            done |= nxt[:, 0] == EOS
            if done.all():
                break
        out = []
        for row in x[:, 1:].tolist():
            s = []
            for t in row:
                if t == EOS:
                    out.append(s)
                    break
                s.append(i2w[t])
            else:
                out.append(None)   # no sentence end within 9 words (the ngram arm rejects these too)
        return out
    return sample, info
