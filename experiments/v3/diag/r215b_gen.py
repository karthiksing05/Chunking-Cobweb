"""Round 215b: does TRELLIS's own generation probability discriminate where the parse score does not?

Retrains the r215 learner (same recipe, window, seed, stream), then scores every r215 control sentence under the causal generator's distribution
(the one experiments/v3/diag/t3_story.py `gen` samples from for the arm {"causal":1,"units":"words","pair":"gate","edge":"exact"} at temp 1; code copied
from its `_fills`, units "words", read None, no escape / both / esc, so only the pair gate differs between the two variants).

Per prefix (window = the learner's window, padded with BOS): masses a_ = L._gap_fills(q) (word w placed here) and e_ = L._gap_fills({q, r1: EOS})
(word w placed here and the sentence ends right after it); outcomes (w, done=True) weigh e_, (w, done=False) weigh max(a_ - e_, 0); the sampler draws one
outcome in proportion to its weight, so the sentence's probability is prod_i P(w_i, done_i = [i == last]) / Z_i. That is the sum over positions of
log P(w_i | prefix) plus log P(</s> | prefix, last word) (the last step's joint = P(w_k) * P(end | w_k)).
Stored raw per step: p = weight / Z (0 if the outcome has no mass), the rank of the true outcome (1 + outcomes with a larger weight), the support
size and Z-fallback flag. The floor eps is applied in r215b_analyse.py (p -> (p + eps) / (1 + eps K), K = 2 |vocab| outcomes), so any eps is free.
Variants: "gate" (pair gate as in generation, q = .5 median of stored pairs) and "nogate".
parse_norm: logZ minus the log mean of logZ-weights (exp) over K random permutations of the sentence's own words (Monte Carlo of the
log partition over the bag's permutations, up to the constant log n!), per word.
Usage (repo root, compute node, env trellis):  r215b_gen.py <seed> <n>
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402
from trellis2.trellis3 import Trellis3, BOS, EOS  # noqa: E402
from trellis2.trellis3_r204 import Trellis3R204  # noqa: E402

KPERM = 16
RAW = C.RAW


class Gen:
    def __init__(self, L, gate, raw):
        self.L, self.gate, self.cache, self.raw = L, gate, {}, raw
        self.W = max(getattr(L, "window", 2), 2)
        if gate:
            self.w_, self.ix_, self.M_ = L.type_licence(near=None)
            self.st_ = set(getattr(L, "_pair_counts", {}))
            vals = sorted(float(self.M_[self.ix_[a], self.ix_[b]]) for a, b in self.st_ if a in self.ix_ and b in self.ix_)
            self.med_ = vals[min(len(vals) - 1, int(len(vals) * 0.5))] if vals else 0.0

    def ok(self, l1, w):
        if not self.gate or l1 == BOS or (l1, w) in self.st_:
            return True
        return l1 in self.ix_ and w in self.ix_ and float(self.M_[self.ix_[l1], self.ix_[w]]) >= self.med_ > 0

    def fills(self, l2, l1, far):
        k = (far, l2, l1)
        if k not in self.cache:
            L = self.L
            q = {"l2": l2, "l1": l1, **{"l%d" % (3 + i): w for i, w in enumerate(far)}}
            if k not in self.raw:
                a0, e0 = L._gap_fills(q), L._gap_fills({**q, "r1": EOS})
                self.raw[k] = {v[1]: (m0, e0.get(v, 0.0)) for v, m0 in a0.items() if v[0] == "w"}   # reduced: word -> (mass, mass ending here)
                L.__dict__.pop("_gap_cache", None)      # n=4000 runs out of memory with it (r215b first attempt)
            a_ = self.raw[k]
            d = {}
            for w_, (m_, me) in a_.items():
                if not self.ok(l1, w_):
                    continue
                d[(w_, True)] = me
                d[(w_, False)] = max(m_ - me, 0.0)
            self.cache[k] = d
        return self.cache[k]

    def score(self, words):
        """per step (p, rank, support, fallback) of the true outcome (w_i, i == last)."""
        out, sofar = [], []
        for i, w in enumerate(words):
            t_ = [BOS] * max(5, getattr(self.L, "window", 2)) + sofar
            d = self.fills(t_[-2], t_[-1], tuple(t_[-3 - j] for j in range(self.W - 2)))
            Z = sum(d.values())
            x = (w, i == len(words) - 1)
            if Z <= 0:
                out.append((None, None, 0, 1))
            else:
                wx = d.get(x, 0.0)
                out.append((wx / Z, 1 + sum(1 for v in d.values() if v > wx), sum(1 for v in d.values() if v > 0), 0))
            sofar.append(w)
        return out


def main(seed, n, stage="all"):
    train, _, _, vocab = C.stream(seed, n)
    cfg = dict(C.CFG, window=C.WINDOW[n])
    L = Trellis3(seed=seed, **cfg)
    t0 = time.time()
    for s, fol in train:
        L.learn(s, follows=fol)
    learn_s = time.time() - t0
    print(json.dumps({"seed": seed, "n": n, "window": C.WINDOW[n], "learn s": round(learn_s), "records": len(L.records)}), flush=True)
    L._quiet = True
    S = C.load_sets(seed)
    lines = []
    for ss in C.all_sentences(S).values():
        for l in ss:
            if l not in lines:
                lines.append(l)
    # determinism check against the r215 parse of the same seed / n (a retrained learner must give the same logZ)
    old = json.load(open(os.path.join(RAW, "trellis_s%d_n%d.json" % (seed, n)), encoding="utf-8"))["res"]
    chk = [abs(L.parse(l.split())["logZ"] - old[l]["logZ"]) for l in lines[:15]]
    print(json.dumps({"logZ reproduces r215 (max abs diff over 15)": max(chk)}), flush=True)
    if stage == "pn":
        return pn(L, lines, seed, n)
    L.__class__ = Trellis3R204             # as t3_story.py's arm "edge": "exact"
    L.edge_exact = True
    t1 = time.time()
    res = {l: {"len": len(l.split())} for l in lines}
    raw = {}
    for variant, gate in (("gate", True), ("nogate", False)):
        G = Gen(L, gate, raw)
        for l in lines:
            res[l][variant] = G.score(l.split())
        print(json.dumps({"variant": variant, "scored": len(lines), "s": round(time.time() - t1), "fill contexts": len(G.cache)}), flush=True)
    if stage == "gen":     # n=4000: parse_norm in a separate run (memory)
        os.makedirs(RAW, exist_ok=True)
        json.dump({"seed": seed, "n": n, "window": C.WINDOW[n], "learn_s": learn_s, "V": len(vocab), "res": res}, open(os.path.join(RAW, "gen_s%d_n%d.json" % (seed, n)), "w", encoding="utf-8"))
        return
    rng = random.Random(seed * 100 + n)
    t2 = time.time()
    for l in lines:
        w = l.split()
        lz = []
        for _ in range(KPERM):
            p = w[:]
            rng.shuffle(p)
            lz.append(L.parse(p)["logZ"])
        m = max(lz)
        res[l]["perm_lse"] = m + math.log(sum(math.exp(x - m) for x in lz) / len(lz))
        res[l]["logZ"] = L.parse(w)["logZ"]
    print(json.dumps({"parse_norm s": round(time.time() - t2), "K": KPERM}), flush=True)
    os.makedirs(RAW, exist_ok=True)
    json.dump({"seed": seed, "n": n, "window": C.WINDOW[n], "learn_s": learn_s, "V": len(vocab), "res": res},
              open(os.path.join(RAW, "gen_s%d_n%d.json" % (seed, n)), "w", encoding="utf-8"))


def pn(L, lines, seed, n):
    rng = random.Random(seed * 100 + n)
    out, t2 = {}, time.time()
    for l in lines:
        w = l.split()
        lz = []
        for _ in range(KPERM):
            p = w[:]
            rng.shuffle(p)
            lz.append(L.parse(p)["logZ"])
        m = max(lz)
        out[l] = m + math.log(sum(math.exp(x - m) for x in lz) / len(lz))
    print(json.dumps({"parse_norm s": round(time.time() - t2), "K": KPERM}), flush=True)
    json.dump({"seed": seed, "n": n, "K": KPERM, "perm_lse": out}, open(os.path.join(RAW, "pn_s%d_n%d.json" % (seed, n)), "w", encoding="utf-8"))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "all")
