"""Round 215: baselines on the same sentences. Word-level n-gram (Witten-Bell interpolation, orders 2, 3, 5), LSTM and GPT (diag/t3_lm.py's
grid and early stopping, models kept) trained on the same n sentences as the learner; score = sentence log-prob (words + </s>), raw and per word.
Also a random score (the right-branching / random-parser floor: a parser that ignores the content cannot discriminate) and -length.
Usage (repo root, compute node, env chunks, GPU):  r215_lms.py <seed> <n>
"""
import itertools
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402


class WB:
    """Witten-Bell interpolated k-gram over words + </s>."""
    def __init__(self, sents, k, V):
        self.k, self.V = k, V
        self.c = [defaultdict(Counter) for _ in range(k)]           # c[m][history of m words][w]
        for s in sents:
            t = ["<s>"] * (k - 1) + list(s) + ["</s>"]
            for i in range(k - 1, len(t)):
                for m in range(k):
                    self.c[m][tuple(t[i - m:i])][t[i]] += 1

    def p(self, h, w):
        pr = 1.0 / self.V
        for m in range(self.k):
            hh = tuple(h[len(h) - m:]) if m else ()
            d = self.c[m].get(hh)
            if not d:
                continue
            tot, typ = sum(d.values()), len(d)
            pr = (d.get(w, 0) + typ * pr) / (tot + typ)
        return pr

    def logp(self, s):
        t = ["<s>"] * (self.k - 1) + list(s) + ["</s>"]
        return sum(math.log(self.p(t[i - self.k + 1:i], t[i])) for i in range(self.k - 1, len(t)))


def main(seed, n):
    import torch
    import t3_lm as T
    train, held, _, vocab = C.stream(seed, n)
    sents = [s for s, _ in train]
    S = C.load_sets(seed)
    allsent = []
    for name, ss in C.all_sentences(S).items():
        for line in ss:
            if line not in allsent:
                if all(w in vocab for w in line.split()):
                    allsent.append(line)
    V = len(vocab) + 1
    scores = {}                                      # model -> {sentence: log-prob}
    for k in (2, 3, 5):
        m = WB(sents, k, V)
        scores["ng%d" % k] = {l: m.logp(l.split()) for l in allsent}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    words = sorted(vocab)
    w2i = {w: i + 3 for i, w in enumerate(words)}
    Vn = len(words) + 3
    info = {}
    for kind in ("lstm", "gpt"):
        grid = (list(itertools.product([1, 2, 4], [64, 128, 256], [0.1, 0.3])) if kind == "gpt"
                else list(itertools.product([1, 2], [128, 256], [0.1, 0.3])))
        best, t0 = None, time.time()
        for layers, d, drop in grid:
            model, h, ep = T._fit(kind, Vn, layers, d, drop, sents, held, w2i, dev, seed)
            if best is None or h < best[1]:
                best = (model, h, {"layers": layers, "d": d, "drop": drop, "epochs": ep})
        model, h, cfg = best
        info[kind] = {"chosen": cfg, "heldout per-token loss": round(h, 4), "train s": round(time.time() - t0)}
        print(json.dumps({"seed": seed, "n": n, "lm": kind, **info[kind]}), flush=True)
        out = {}
        with torch.no_grad():
            for i in range(0, len(allsent), 128):
                chunk = allsent[i:i + 128]
                x = T._batch([l.split() for l in chunk], w2i, dev)
                lg = model(x[:, :-1])
                ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), ignore_index=T.PAD, reduction="none")
                lp = -ce.view(x.shape[0], -1).sum(1)
                for l, v in zip(chunk, lp.tolist()):
                    out[l] = v
        scores[kind] = out
    rng = random.Random(seed * 1000 + n)
    scores["random"] = {l: rng.random() for l in allsent}
    scores["neglen"] = {l: -float(len(l.split())) for l in allsent}
    os.makedirs(C.RAW, exist_ok=True)
    json.dump({"seed": seed, "n": n, "info": info, "scores": scores}, open(os.path.join(C.RAW, "lms_s%d_n%d.json" % (seed, n)), "w", encoding="utf-8"))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
