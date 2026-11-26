"""Round 233: TRELLIS's unsupervised parses against the exact gold trees of the controlled grammar.
usage: r233_goldparse.py <seed> [n list, default 125,250,500,1000,2000,4000]
Learns on the first n sentences (cfg_w5; nested stream), then parses 500 held-out sentences that never occur in the 4,000-sentence
training stream (learning off; held_unseen) and compares the chosen
(MBR) tree with gold constituents: subject NP, object NP, PP (Prep NP), VP (when longer than the verb), and NP-internal [Adj N]
is NOT gold (the grammar's NP is flat), so we report:
  recall   = share of gold constituents (length >= 2, below the sentence) that the tree brackets
  crossing = share of the tree's brackets (length >= 2, below the sentence) that cross a gold constituent
  exact    = share of sentences with every gold constituent bracketed and no crossing bracket
  micro_P / micro_F1 = bracket precision and F1 pooled over sentences (the flat gold NP makes [Adj N] a false positive)
for TRELLIS and for right-branching, left-branching and random binary trees on the same sentences."""
import json
import os
import random
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r230_grammar as G  # noqa: E402
from trellis2.trellis3_r223 import Trellis3R223  # noqa: E402
from trellis2.trellis3_r225 import Trellis3R225  # noqa: E402

# env R233_ARM: default (Trellis3R223, as before) | full | nochunk | noconcept | neither  (Trellis3R225 removal switches, as r225)
ARMS = {"full": ({}, {}), "nochunk": ({"store": "none"}, {}), "noconcept": ({}, {"concepts_off": True, "concepts_off_parse": True}),
        "neither": ({"store": "none"}, {"concepts_off": True, "concepts_off_parse": True})}


def gold(s):
    """Gold constituents (i, j) with 2 <= j - i < len(s)."""
    L = G.LEX
    out = set()
    i = 1 + (L[s[1]][0] == "adj")
    subj_end = i + 1
    out.add((0, subj_end))
    v = subj_end
    if v + 1 < len(s):
        out.add((v, len(s)))                       # VP
        k = v + 1
        if s[k] in G.PREPS:
            out.add((k, len(s)))                   # PP
            k += 1
        out.add((k, len(s)))                       # object NP
    return {(a, b) for (a, b) in out if 2 <= b - a < len(s)}


def tree_spans(chosen, n):
    return {(i, j) for (i, k, j) in chosen if 2 <= j - i < n}


def rb(n):
    return {(i, n) for i in range(1, n - 1)}


def lb(n):
    return {(0, j) for j in range(2, n)}


def rnd(n, rng):
    out = set()

    def split(i, j):
        if j - i < 2:
            return
        if j - i < n:
            out.add((i, j))
        k = rng.randint(i + 1, j - 1)
        split(i, k)
        split(k, j)
    split(0, n)
    return out


def cross(a, b):
    (i, j), (k, l) = a, b
    return (i < k < j < l) or (k < i < l < j)


def held_unseen(C, n_max=4000, k=500):
    """Held-out sentences that never occur in the training stream (first n_max sentences), first k of them. The grammar's held
    set overlaps training (about a fifth at 4,000 sentences, almost all three-word sentences), so every r233 parse score uses this."""
    seen = {tuple(s) for s, _ in C["train"][:n_max]}
    return [s for s in C["held"] if tuple(s) not in seen][:k]


def score(trees, golds):
    rec = [len(t & g) / len(g) for t, g in zip(trees, golds) if g]
    tp = sum(len(t & g) for t, g in zip(trees, golds))
    nt, ng = sum(len(t) for t in trees), sum(len(g) for g in golds)
    P, R = tp / max(nt, 1), tp / max(ng, 1)
    crs = [sum(any(cross(x, y) for y in g) for x in t) / len(t) for t, g in zip(trees, golds) if t]
    ex = [g <= t and not any(cross(x, y) for x in t for y in g) for t, g in zip(trees, golds)]
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    return {"recall": m(rec), "crossing": m(crs), "exact": m(ex), "micro_P": round(P, 3), "micro_F1": round(2 * P * R / max(P + R, 1e-9), 3)}


def main():
    seed = int(sys.argv[1])
    ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [125, 250, 500, 1000, 2000, 4000]
    cfg = json.load(open("experiments/v3/results/story_gen/r214p01/cfg_w5.json"))
    C = G.load_corpus(seed, max(ns))
    stream = [s for s, _ in C["train"]]
    held = held_unseen(C, max(ns))
    golds = [gold(s) for s in held]
    rng = random.Random(seed)
    base = {"right-branching": score([rb(len(s)) for s in held], golds),
            "left-branching": score([lb(len(s)) for s in held], golds),
            "random": score([rnd(len(s), rng) for s in held], golds)}
    print(json.dumps({"seed": seed, "baselines": base}), flush=True)
    arm = os.environ.get("R233_ARM", "default")
    if arm == "default":
        L = Trellis3R223(seed=seed, **cfg)
    elif arm.startswith("minbits:"):
        from trellis2.trellis3_r233 import Trellis3R233
        L = Trellis3R233(seed=seed, parse_min_bits=float(arm.split(":")[1]), **cfg)
    else:
        ck, kw = ARMS[arm]
        L = Trellis3R225(seed=seed, **{**cfg, **ck}, **kw)
    done, res = 0, {"seed": seed, "arm": arm, "baselines": base, "trellis": {}}
    for n in ns:
        for s in stream[done:n]:
            L.learn(s)
        done = n
        trees = [tree_spans(L.parse(s)["chosen"], len(s)) for s in held]
        res["trellis"][n] = score(trees, golds)
        if n == ns[-1] or n == 1000:
            res.setdefault("examples", {})[n] = [{"s": " ".join(s), "tree": sorted(t), "gold": sorted(g)} for s, t, g in zip(held[:8], trees[:8], golds[:8])]
        print(json.dumps({"arm": arm, "n": n, **res["trellis"][n]}), flush=True)
    os.makedirs("experiments/v3/results/story_gen/r233/unseen", exist_ok=True)
    json.dump(res, open(f"experiments/v3/results/story_gen/r233/unseen/goldparse_s{seed}" + ("" if arm == "default" else f"_{arm}") + ".json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
