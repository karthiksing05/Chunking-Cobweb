"""Rare-word convergence (docs/TEMPORAL_INTERDEP.md): with interdependent
representations a word seen a few times should already be represented like its
category, because its neighbours' representations are learned.

Toy language: S -> NP V NP, NP -> D N | D A N, with 4 determiners, 3 adjectives,
6 verbs and 40 nouns of Zipfian frequency (many seen 1-3 times). After learning,
for every word: its nearest other word by name overlap (sum of min weights) is
of the same category (gold only scores). Reported for nouns by how often they
were seen (1, 2-3, 4-7, 8-15, 16+), at checkpoints of the stream, together with
the mean overlap of a noun's name with the other nouns' (within) and with the
verbs' (across).

Usage: t3_rare.py <cfg json> <seed> [n sentences] [tag]
"""
import sys, json, random
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2.trellis3 import Trellis3, Elem

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
tag = sys.argv[4] if len(sys.argv) > 4 else ""
rng = random.Random(seed)
D, A = ["a", "the", "this", "that"], ["big", "red", "old"]
V = ["saw", "liked", "chased", "found", "heard", "met"]
N = [f"n{i}" for i in range(40)]
zipf = [1.0 / (k + 1) for k in range(len(N))]
cat = {**{w: "D" for w in D}, **{w: "A" for w in A}, **{w: "V" for w in V}, **{w: "N" for w in N}}


def np_():
    out = [rng.choice(D)]
    if rng.random() < 0.3:
        out.append(rng.choice(A))
    return out + rng.choices(N, weights=zipf)


corpus = [np_() + [rng.choice(V)] + np_() for _ in range(n)]


def as_bag(v):
    if isinstance(v, dict):
        t = sum(v.values()) or 1.0
        return {u: p / t for u, p in v.items()}
    if isinstance(v, list):
        real = [u for u in v if u]
        return {u: real.count(u) / len(real) for u in set(real)} if real else {}
    return {}


def overlap(a, b):
    return sum(min(p, b.get(u, 0.0)) for u, p in a.items())


L = Trellis3(seed=seed, **cfg)
seen = Counter()
checks = sorted({50, 100, 200, n})
out = {"tag": tag, "seed": seed, "sentences": n, "cfg": cfg}
for k, s in enumerate(corpus, 1):
    L.learn(s)
    seen.update(s)
    if k in checks:
        L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
        words = [w for w in seen]
        nm = {w: as_bag(L._ctx_name(Elem((0, 1), token=w)) or {}) for w in words}
        by = defaultdict(list)
        within, across = [], []
        for w in words:
            if cat[w] != "N":
                continue
            best = max((x for x in words if x != w), key=lambda x: overlap(nm[w], nm[x]))
            ok = cat[best] == "N" and overlap(nm[w], nm[best]) > 0
            c = seen[w]
            by["1" if c == 1 else "2-3" if c <= 3 else "4-7" if c <= 7 else "8-15" if c <= 15 else "16+"].append(ok)
            others = [x for x in words if cat[x] == "N" and x != w]
            within.append(sum(overlap(nm[w], nm[x]) for x in others) / max(len(others), 1))
            vs = [x for x in words if cat[x] == "V"]
            across.append(sum(overlap(nm[w], nm[x]) for x in vs) / max(len(vs), 1))
        order = ["1", "2-3", "4-7", "8-15", "16+"]
        out[str(k)] = {"noun nearest is a noun, by count": {b: (round(sum(by[b]) / len(by[b]), 3), len(by[b]))
                                                            for b in order if by[b]},
                       "all nouns": round(sum(sum(v) for v in by.values()) / max(sum(len(v) for v in by.values()), 1), 3),
                       "overlap noun~noun / noun~verb": [round(sum(within) / max(len(within), 1), 3),
                                                         round(sum(across) / max(len(across), 1), 3)]}
print(json.dumps(out), flush=True)
