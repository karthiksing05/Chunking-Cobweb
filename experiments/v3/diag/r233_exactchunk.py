"""Round 233: a chunk memory without concepts, as a parsing baseline (controlled grammar).
usage: r233_exactchunk.py <seed> [n list]
Stores every contiguous word sequence (length >= 2) of the first n training sentences with its count. A held-out sentence is
parsed by CKY into the binary tree that maximises the summed score of its spans (length >= 2, below the whole sentence):
  count : log(1 + count(exact sequence))
  pmi   : log2 [ c(l r) N / (c(l) c(r)) ] for the span's best split into l, r, with c = exact sequence counts (+0.5 smoothing)
Ties go to the smallest split (as TRELLIS's MBR). Same unseen held-out set and gold as r233_goldparse."""
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r230_grammar as G  # noqa: E402
import r233_goldparse as GP  # noqa: E402


def counts(train):
    c = Counter()
    for s in train:
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                c[tuple(s[i:j])] += 1
    return c


def parse(s, span_score):
    n = len(s)
    best, arg = {}, {}
    for i in range(n):
        best[(i, i + 1)] = 0.0
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L
            own = span_score(s, i, j) if L < n else 0.0
            b, bk = None, None
            for k in range(i + 1, j):
                v = best[(i, k)] + best[(k, j)]
                if b is None or v > b + 1e-12:
                    b, bk = v, k
            best[(i, j)], arg[(i, j)] = b + own, bk
    out = set()

    def walk(i, j):
        if j - i < 2:
            return
        if j - i < n:
            out.add((i, j))
        k = arg[(i, j)]
        walk(i, k)
        walk(k, j)
    walk(0, n)
    return out


def main():
    seed = int(sys.argv[1])
    ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [125, 250, 500, 1000, 2000, 4000]
    C = G.load_corpus(seed, max(ns))
    stream = [s for s, _ in C["train"]]
    held = GP.held_unseen(C, max(ns))
    golds = [GP.gold(s) for s in held]
    res = {"seed": seed, "held": len(held), "count": {}, "pmi": {}}
    for n in ns:
        c = counts(stream[:n])
        N = sum(v for k, v in c.items() if len(k) == 1)

        def sc_count(s, i, j):
            return math.log(1 + c[tuple(s[i:j])])

        def sc_pmi(s, i, j):
            y = tuple(s[i:j])
            return max(math.log2((c[y] + 0.5) * N / ((c[y[:k]] + 0.5) * (c[y[k:]] + 0.5))) for k in range(1, len(y)))
        for name, f in (("count", sc_count), ("pmi", sc_pmi)):
            res[name][n] = GP.score([parse(s, f) for s in held], golds)
            print(json.dumps({"seed": seed, "n": n, "score": name, **res[name][n]}), flush=True)
    os.makedirs("experiments/v3/results/story_gen/r233/unseen", exist_ok=True)
    json.dump(res, open(f"experiments/v3/results/story_gen/r233/unseen/exactchunk_s{seed}.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
