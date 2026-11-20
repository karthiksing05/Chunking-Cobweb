"""Round 225 (reference codes): held-out code length in bits per sentence for the baselines, trained on the same n sentences as TRELLIS.
  uniform      (words + </s>) x log2 251
  bpe_k<k>     BPE vocabulary + unigram code (the chess convention, chess_compress_bpe.py): word-level BPE merges the most frequent adjacent
               item pair (count >= 2) of the training sentences, repeatedly; a sentence is rewritten by the merges in order; items are
               entropy-coded by their usage counts in the rewritten training sentences (Laplace +1 over the 250 words + k chunk items + a stop
               item; one stop per sentence). k = 0 is the plain word unigram. `bpe` = the k with the lowest dev bits. Dictionary cost
               (sum over merges of words x log2 250) is stored beside (model cost, reported separately, not in the code length).
  wb2/wb3/wb5  Witten-Bell interpolated word k-grams (r215_lms.WB), the repo's existing baseline
  kn2..kn5     interpolated Kneser-Ney k-grams (absolute discount D = n1 / (n1 + 2 n2) per order from the order's counts of counts, continuation
               counts below the top order, raw counts for histories that start at <s>, unigram = continuation counts + 1 over 251)
  ngram        the n-gram (any of the above) with the lowest dev bits
  lstm, gpt    the r215 / r212 grid and early stopping (diag/t3_lm.py _fit), trained here; test sentences scored with PAD/BOS logits masked
               (renormalised over the 250 words + </s>)
Usage (repo root, compute node, env chunks, GPU):  r225_lms.py <seed> <n>
"""
import itertools
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r225_common as D  # noqa: E402
from r215_lms import WB  # noqa: E402

L2 = math.log(2.0)
KGRID = (0, 10, 25, 50, 100, 200, 400, 800, 1600)


class KN:
    def __init__(self, sents, k, V):
        self.k, self.V = k, V
        pad = [["<s>"] * (k - 1) + list(s) + ["</s>"] for s in sents]
        raw = {m: defaultdict(Counter) for m in range(1, k + 1)}      # raw[m][history of m-1 words][w]
        for t in pad:
            for i in range(k - 1, len(t)):
                for m in range(1, k + 1):
                    raw[m][tuple(t[i - m + 1:i])][t[i]] += 1
        self.c = {k: raw[k]}
        for m in range(k - 1, 0, -1):
            up = self.c[m + 1]
            S = defaultdict(set)
            for h, cw in up.items():
                for w in cw:
                    S[(h[1:], w)].add(h[0])
            low = defaultdict(Counter)
            for (h, w), vs in S.items():
                low[h][w] = len(vs)
            for h, cw in raw[m].items():                               # histories that start at the sentence start: raw counts
                if h and h[0] == "<s>":
                    low[h] = Counter(cw)
            self.c[m] = low
        self.D, self.tot = {}, {}
        for m in range(1, k + 1):
            cc = Counter(v for d in self.c[m].values() for v in d.values())
            n1, n2 = cc.get(1, 0), cc.get(2, 0)
            self.D[m] = n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.75
            self.tot[m] = {h: sum(d.values()) for h, d in self.c[m].items()}

    def p(self, h, w):
        d1 = self.c[1][()]
        pr = (d1.get(w, 0) + 1.0) / (self.tot[1][()] + self.V)
        for m in range(2, self.k + 1):
            hh = tuple(h[len(h) - (m - 1):])
            d = self.c[m].get(hh)
            if not d:
                continue
            tot, typ = self.tot[m][hh], len(d)
            pr = max(d.get(w, 0) - self.D[m], 0.0) / tot + self.D[m] * typ / tot * pr
        return pr

    def bits(self, s):
        t = ["<s>"] * (self.k - 1) + list(s) + ["</s>"]
        return -sum(math.log2(self.p(t[i - self.k + 1:i], t[i])) for i in range(self.k - 1, len(t)))


def wb_bits(m, s):
    return -m.logp(s) / L2


def merge_seq(sq, a, b, new):
    out, i = [], 0
    while i < len(sq):
        if i + 1 < len(sq) and sq[i] == a and sq[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(sq[i])
            i += 1
    return out


def bpe(train, dev, test, V0):
    """bits per sentence of dev and test at every k in KGRID (items coded by a Laplace unigram), and the dictionary cost."""
    st = [[(w,) for w in s] for s in train]
    sd = [[(w,) for w in s] for s in dev]
    se = [[(w,) for w in s] for s in test]
    out, dict_bits, nm = {}, 0.0, 0

    def snap(k):
        cnt = Counter(it for sq in st for it in sq)
        N = sum(cnt.values()) + len(st)                                # + one stop per sentence
        A = V0 + k + 1
        lp = lambda it: math.log2((cnt.get(it, 0) + 1.0) / (N + A))
        stop = math.log2((len(st) + 1.0) / (N + A))
        f = lambda seqs: [-(sum(lp(it) for it in sq) + stop) for sq in seqs]
        out[k] = {"dev": f(sd), "test": f(se), "dict bits": dict_bits, "merges": nm}
    snap(0) if 0 in KGRID else None
    for t in range(1, max(KGRID) + 1):
        cnt = Counter()
        for sq in st:
            for a, b in zip(sq, sq[1:]):
                cnt[(a, b)] += 1
        if not cnt:
            break
        (a, b), c = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
        if c < 2:
            break
        new = a + b
        st = [merge_seq(sq, a, b, new) for sq in st]
        sd = [merge_seq(sq, a, b, new) for sq in sd]
        se = [merge_seq(sq, a, b, new) for sq in se]
        dict_bits += len(new) * math.log2(V0)
        nm = t
        if t in KGRID:
            snap(t)
    return out


def main(seed, n):
    import torch
    import t3_lm as T
    train, dev, test, vocab = D.data(seed, n)
    V0 = len(vocab)
    res = {"seed": seed, "n": n, "V": V0, "test_words": [len(s) for s in test], "dev_words": [len(s) for s in dev], "models": {}, "info": {}}
    M = res["models"]
    uni = lambda ss: [(len(s) + 1) * math.log2(V0 + 1) for s in ss]
    M["uniform"] = {"dev": uni(dev), "test": uni(test)}
    for k, v in bpe(train, dev, test, V0).items():
        M["bpe_k%d" % k] = {"dev": v["dev"], "test": v["test"]}
        res["info"]["bpe_k%d" % k] = {"dict bits": v["dict bits"], "merges": v["merges"]}
    for k in (2, 3, 5):
        m = WB(train, k, V0 + 1)
        M["wb%d" % k] = {"dev": [wb_bits(m, s) for s in dev], "test": [wb_bits(m, s) for s in test]}
    for k in (2, 3, 4, 5):
        m = KN(train, k, V0 + 1)
        M["kn%d" % k] = {"dev": [m.bits(s) for s in dev], "test": [m.bits(s) for s in test]}
        res["info"]["kn%d" % k] = {"D": {str(a): round(b, 3) for a, b in m.D.items()}}
    print(json.dumps({"seed": seed, "n": n, "n-grams and BPE done": True, "bpe merges": res["info"]["bpe_k1600"]["merges"] if "bpe_k1600" in res["info"] else None}), flush=True)
    dv = "cuda" if torch.cuda.is_available() else "cpu"
    words = sorted(vocab)
    w2i = {w: i + 3 for i, w in enumerate(words)}
    Vn = len(words) + 3

    def lm_bits(model, sents):
        out = []
        with torch.no_grad():
            for i in range(0, len(sents), 256):
                x = T._batch(sents[i:i + 256], w2i, dv)
                lg = model(x[:, :-1]).float()
                lg[:, :, T.PAD] = -1e9
                lg[:, :, T.BOS] = -1e9
                tgt = x[:, 1:]
                g = torch.log_softmax(lg, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).masked_fill(tgt == T.PAD, 0.0)
                out += (-g.sum(1) / L2).tolist()
        return out
    for kind in ("lstm", "gpt"):
        grid = (list(itertools.product([1, 2, 4], [64, 128, 256], [0.1, 0.3])) if kind == "gpt"
                else list(itertools.product([1, 2], [128, 256], [0.1, 0.3])))
        best, t0 = None, time.time()
        for layers, d, drop in grid:
            model, h, ep = T._fit(kind, Vn, layers, d, drop, train, dev, w2i, dv, seed)
            if best is None or h < best[1]:
                best = (model, h, {"layers": layers, "d": d, "drop": drop, "epochs": ep})
        model, h, cfg = best
        M[kind] = {"dev": lm_bits(model, dev), "test": lm_bits(model, test)}
        res["info"][kind] = {"chosen": cfg, "dev loss per token (nats, unmasked)": round(h, 4), "params": sum(p.numel() for p in model.parameters()), "train s": round(time.time() - t0)}
        print(json.dumps({"seed": seed, "n": n, "lm": kind, **res["info"][kind]}), flush=True)
        del model, best
    os.makedirs(D.RAW, exist_ok=True)
    json.dump(res, open(os.path.join(D.RAW, "lms_s%d_n%d.json" % (seed, n)), "w", encoding="utf-8"))
    print("wrote", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
