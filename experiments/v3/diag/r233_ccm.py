"""Round 233: constituent-context model (Klein & Manning 2002) over word tokens as an unsupervised-parsing baseline.
usage: r233_ccm.py <seed> [n list]   |   r233_ccm.py --selftest
Binary bracketings, P(B) uniform; every span (i,j), j>i, of the sentence emits (yield, context=(prev word,next word)) from the
constituent distributions if it is in B and from the distituent distributions otherwise. Dividing out the all-distituent
configuration gives the ratio form: sum_B prod_{span in B} Pc(y)Pc(c)/(Pd(y)Pd(c)); inside-outside over binary trees gives span posteriors.
Init = split-uniform (Klein & Manning: the first M step uses span posteriors of the prior that picks each split uniformly,
top down; fixed 2026-11-26 after review, the earlier version used all ratios 1, i.e. uniform over bracketings, and fell into an
anti-gold optimum). 20 EM iterations. Add-K smoothing, K=0.01, over
all training yield (context) types plus one shared UNK type (MAP with Dirichlet alpha=1+K, so EM maximises loglik + prior). Decode = Viterbi."""
import json
import math
import os
import random
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r233_goldparse as GP  # noqa: E402
import r230_grammar as G  # noqa: E402

G2 = os.environ.get("R233_GRAMMAR") == "g2"
SUF = "_g2" if G2 else ""
if G2:
    import r234_grammar2 as G  # noqa: E402,F811

K = 0.01
ITERS = 20


class Data:
    def __init__(self, train):
        self.yid, self.cid = {None: 0}, {None: 0}
        self.sents = [self.encode(s, True) for s in train]

    def encode(self, s, grow=False):
        w = ["<s>"] + list(s) + ["</s>"]
        n = len(s)
        out = []
        for i in range(n):
            for j in range(i + 1, n + 1):
                y, c = tuple(s[i:j]), (w[i], w[j + 1])
                if grow:
                    self.yid.setdefault(y, len(self.yid))
                    self.cid.setdefault(c, len(self.cid))
                out.append((i, j, self.yid.get(y, 0), self.cid.get(c, 0)))
        return n, out


def normalise(cnt):
    t = sum(cnt) + K * len(cnt)
    return [math.log((x + K) / t) for x in cnt]


class CCM:
    def __init__(self, data):
        self.d = data
        ny, nc = len(data.yid), len(data.cid)
        self.ny, self.nc = ny, nc

    def mstep(self, cy, dy, cc, dc):
        self.lcy, self.ldy, self.lcc, self.ldc = normalise(cy), normalise(dy), normalise(cc), normalise(dc)

    def objective_prior(self):
        return K * sum(sum(v) for v in (self.lcy, self.ldy, self.lcc, self.ldc))

    def ratios(self, n, spans):
        r = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i, j, y, c in spans:
            r[i][j] = math.exp(self.lcy[y] - self.ldy[y] + self.lcc[c] - self.ldc[c]) if self.lcy else 1.0
        return r

    def inside(self, n, r):
        A = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i in range(n):
            A[i][i + 1] = r[i][i + 1]
        for L in range(2, n + 1):
            for i in range(n - L + 1):
                j = i + L
                A[i][j] = r[i][j] * sum(A[i][k] * A[k][j] for k in range(i + 1, j))
        return A

    def outside(self, n, r, A):
        B = [[0.0] * (n + 1) for _ in range(n + 1)]
        B[0][n] = 1.0
        for L in range(n, 1, -1):
            for i in range(n - L + 1):
                j = i + L
                f = B[i][j] * r[i][j]
                for k in range(i + 1, j):
                    B[i][k] += f * A[k][j]
                    B[k][j] += f * A[i][k]
        return B

    def epoch(self, uniform):
        d = self.d
        cy, dy, cc, dc = [0.0] * self.ny, [0.0] * self.ny, [0.0] * self.nc, [0.0] * self.nc
        ll = 0.0
        for n, spans in d.sents:
            r = [[1.0] * (n + 1) for _ in range(n + 1)] if uniform else self.ratios(n, spans)
            A = self.inside(n, r)
            Z = A[0][n]
            B = self.outside(n, r, A)
            for i, j, y, c in spans:
                p = A[i][j] * B[i][j] / Z
                cy[y] += p
                dy[y] += 1 - p
                cc[c] += p
                dc[c] += 1 - p
            if not uniform:
                cat = math.comb(2 * (n - 1), n - 1) // n
                ll += math.log(Z) - math.log(cat) + sum(self.ldy[y] + self.ldc[c] for _, _, y, c in spans)
        return (cy, dy, cc, dc), ll

    def split_stats(self):
        cy, dy, cc, dc = [0.0] * self.ny, [0.0] * self.ny, [0.0] * self.nc, [0.0] * self.nc
        for n, spans in self.d.sents:
            P = [[0.0] * (n + 1) for _ in range(n + 1)]
            P[0][n] = 1.0
            for L in range(n, 1, -1):
                for i in range(n - L + 1):
                    j = i + L
                    p = P[i][j]
                    if p:
                        for k in range(i + 1, j):
                            P[i][k] += p / (L - 1)
                            P[k][j] += p / (L - 1)
            for i, j, y, c in spans:
                p = P[i][j]
                cy[y] += p
                dy[y] += 1 - p
                cc[c] += p
                dc[c] += 1 - p
        return cy, dy, cc, dc

    def fit(self, iters=ITERS):
        self.mstep(*self.split_stats())
        log = []
        for it in range(iters + 1):
            stats, ll = self.epoch(False)
            log.append({"loglik": ll, "map_objective": ll + self.objective_prior()})
            if it < iters:
                self.mstep(*stats)
        return log

    def mbr(self, s):
        """Max-expected-recall tree: maximise the sum of span posteriors P(constituent)."""
        n, spans = self.d.encode(s)
        r = self.ratios(n, spans)
        A = self.inside(n, r)
        B = self.outside(n, r, A)
        Z = A[0][n]
        M = [[0.0] * (n + 1) for _ in range(n + 1)]
        bp = [[0] * (n + 1) for _ in range(n + 1)]
        for L in range(2, n + 1):
            for i in range(n - L + 1):
                j = i + L
                best = max(range(i + 1, j), key=lambda k: M[i][k] + M[k][j])
                bp[i][j] = best
                M[i][j] = A[i][j] * B[i][j] / Z + M[i][best] + M[best][j]
        return self._walk(bp, n)

    @staticmethod
    def _walk(bp, n):
        out = set()

        def walk(i, j):
            if j - i < 2:
                return
            out.add((i, j))
            k = bp[i][j]
            walk(i, k)
            walk(k, j)
        walk(0, n)
        return {(i, j) for (i, j) in out if 2 <= j - i < n}

    def viterbi(self, s, score=False):
        n, spans = self.d.encode(s)
        lr = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i, j, y, c in spans:
            lr[i][j] = self.lcy[y] - self.ldy[y] + self.lcc[c] - self.ldc[c]
        V = [[0.0] * (n + 1) for _ in range(n + 1)]
        bp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n):
            V[i][i + 1] = lr[i][i + 1]
        for L in range(2, n + 1):
            for i in range(n - L + 1):
                j = i + L
                best = max(range(i + 1, j), key=lambda k: V[i][k] + V[k][j])
                bp[i][j] = best
                V[i][j] = lr[i][j] + V[i][best] + V[best][j]
        if score:
            return V[0][n]
        return self._walk(bp, n)


def all_trees(i, j):
    if j - i == 1:
        return [frozenset([(i, j)])]
    out = []
    for k in range(i + 1, j):
        for a in all_trees(i, k):
            for b in all_trees(k, j):
                out.append(a | b | {(i, j)})
    return out


def selftest():
    rng = random.Random(0)
    vocab = "a b c d e".split()
    train = [[rng.choice(vocab) for _ in range(rng.choice([3, 4, 4, 5]))] for _ in range(40)]
    m = CCM(Data(train))
    log = m.fit(ITERS)
    # inside total vs brute force on a 4-word sentence under the trained (non-trivial) parameters
    for s in (["a", "b", "c", "d"], ["e", "a", "a", "b"]):
        n, spans = m.d.encode(s)
        r = m.ratios(n, spans)
        Z = m.inside(n, r)[0][n]
        bf = sum(math.prod(r[i][j] for i, j in t) for t in all_trees(0, n))
        assert len(all_trees(0, n)) == 5
        assert abs(Z - bf) < 1e-9 * bf, (Z, bf)
        print("selftest inside total", Z, "brute force", bf)
        # posteriors from outside match brute force
        B = m.outside(n, r, m.inside(n, r))
        A = m.inside(n, r)
        for i in range(n):
            for j in range(i + 1, n + 1):
                pb = sum(math.prod(r[a][b] for a, b in t) for t in all_trees(0, n) if (i, j) in t) / bf
                assert abs(A[i][j] * B[i][j] / Z - pb) < 1e-9, (i, j)
    print("selftest span posteriors match brute force")
    for it, e in enumerate(log):
        print(f"iter {it} loglik {e['loglik']:.6f} map_objective {e['map_objective']:.6f}")
    ll = [e["loglik"] for e in log]
    ob = [e["map_objective"] for e in log]
    assert all(b >= a - 1e-9 for a, b in zip(ob, ob[1:])), "MAP objective decreased"
    print("MAP objective non-decreasing: OK; raw loglik non-decreasing:", all(b >= a - 1e-9 for a, b in zip(ll, ll[1:])))
    print("SELFTEST OK")


def tagify(s):
    return ["_".join(str(x) for x in G.syncat(w)) for w in s]


def diagnose(m, held, golds):
    from collections import Counter
    cnt = Counter()
    nrb = nlb = 0
    for s in held:
        t = m.viterbi(s)
        nrb += t == GP.rb(len(s))
        nlb += t == GP.lb(len(s))
        for (i, j) in t:
            cnt[(" ".join(s[i:j]), " ".join(G.TAG[w] for w in s[i:j]))] += 1
    print("top 15 predicted spans (words | gold tags | count)")
    for (w, tg), c in cnt.most_common(15):
        print(f"  {w} | {tg} | {c}")
    print(f"share equal right-branching {nrb / len(held):.3f}  left-branching {nlb / len(held):.3f}")
    # Viterbi vs brute force on 5 short held-out sentences
    short = [s for s in held if len(s) <= 7][:5]
    for s in short:
        n, spans = m.d.encode(s)
        r = m.ratios(n, spans)
        trees = all_trees(0, n)
        sc = lambda t: sum(math.log(r[i][j]) for i, j in t)
        bt = max(trees, key=sc)
        vt = m.viterbi(s)
        vfull = {(i, j) for (i, j) in bt if 2 <= j - i < n}
        print("len", n, "viterbi score", round(m.viterbi(s, True), 6), "brute-force best", round(sc(bt), 6), "same tree", vt == vfull)
    print("gold spans of a held-out example", sorted(golds[0]), "pred", sorted(m.viterbi(held[0])), " ".join(held[0]))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tags, mbr, diag = "--tags" in sys.argv, "--mbr" in sys.argv, "--diag" in sys.argv
    trans = "--transductive" in sys.argv                     # also train on the (unlabelled) test sentences, as Klein & Manning did
    iters = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--iters=")), ITERS))  # 0 = split-init counts only
    seed = int(args[0])
    ns = [int(x) for x in args[1].split(",")] if len(args) > 1 else [125, 250, 500, 1000, 2000, 4000]
    C = G.load_corpus(seed, max(ns))
    stream = [s for s, _ in C["train"]]
    held = GP.held_unseen(C, max(ns))
    golds = [GP.gold(s) for s in held]
    if tags:
        stream = [tagify(s) for s in stream]
        held = [tagify(s) for s in held]
    res = {"seed": seed, "K": K, "iters": iters, "transductive": trans, "tags": tags, "decode": "mbr" if mbr else "viterbi", "ccm": {}, "em_log_n1000": None}
    for n in ns:
        m = CCM(Data(stream[:n] + (held if trans else [])))
        log = m.fit(iters)
        if diag:
            diagnose(m, GP.held_unseen(C, max(ns)), golds)
            return
        trees = [(m.mbr if mbr else m.viterbi)(s) for s in held]
        res["ccm"][n] = GP.score(trees, golds)
        if n == 1000:
            res["em_log_n1000"] = log
        print(json.dumps({"seed": seed, "n": n, **res["ccm"][n], "final_loglik": log[-1]["loglik"]}), flush=True)
    os.makedirs("experiments/v3/results/story_gen/r233/unseen" + ("_g2" if G2 else ""), exist_ok=True)
    json.dump(res, open(f"experiments/v3/results/story_gen/r233/unseen{SUF}/ccm{'_tags' if tags else ''}{'_mbr' if mbr else ''}{'_trans' if trans else ''}{'' if iters == ITERS else f'_it{iters}'}_s{seed}.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
