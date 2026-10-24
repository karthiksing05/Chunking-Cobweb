"""The a/the test (docs/RELATIONAL.md): "a" and "the" are interchangeable before
eight nouns, but "dog" is only ever seen after "a" and "cat" only after "the".
Do dog and cat end up with related representations?

Reports, after learning:
  name overlap   sum_v min(name(x)[v], name(y)[v]) of the anchor names a content
                 instance uses for a word part, for (dog, cat), (dog, another
                 noun), (dog, a verb);
  tree sharing   for k = 1..3 levels above the leaf, the share of (x leaf,
                 y leaf) pairs whose ancestors k up coincide.

Usage: t3_athe.py <cfg json> <seed> [n sentences] [tag]
"""
import sys, json, random
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2.trellis3 import Trellis3, Elem

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
tag = sys.argv[4] if len(sys.argv) > 4 else ""
rng = random.Random(seed)
nouns = [f"n{i}" for i in range(8)]
verbs = ["saw", "liked", "chased", "found"]


def np_():
    r = rng.random()
    if r < 0.08:
        return ["a", "dog"]
    if r < 0.16:
        return ["the", "cat"]
    return [rng.choice(["a", "the"]), rng.choice(nouns)]


corpus = [np_() + [rng.choice(verbs)] + np_() for _ in range(n)]
L = Trellis3(seed=seed, **cfg)
for s in corpus:
    L.learn(s)
L._rel_cache, L._wc_cache = {}, {}


def name(w):
    e = Elem((0, 1), token=w)
    return L._ctx_name(e) or {}


def overlap(x, y):
    a, b = name(x), name(y)
    if isinstance(a, list) or isinstance(b, list):
        a = {v: 1 / len(a) for v in a} if isinstance(a, list) else a
        b = {v: 1 / len(b) for v in b} if isinstance(b, list) else b
    return round(sum(min(a.get(v, 0.0), b.get(v, 0.0)) for v in set(a) | set(b)), 3)


leaves = {}
for r in L.records:
    if r.token is not None and r.leaf is not None:
        leaves.setdefault(r.token, []).append(r.leaf)


def up(node, k):
    for _ in range(k):
        node = node.parent or node
    return node.id


def share(x, y, k):
    X, Y = leaves.get(x, [])[:60], leaves.get(y, [])[:60]
    pairs = [(a, b) for a in X for b in Y]
    return round(sum(up(a, k) == up(b, k) for a, b in pairs) / max(len(pairs), 1), 3)


pairs = {"dog~cat": ("dog", "cat"), "dog~n0": ("dog", "n0"), "cat~n1": ("cat", "n1"), "dog~saw": ("dog", "saw"), "a~the": ("a", "the")}
out = {"tag": tag, "seed": seed, "sentences": n, "cfg": cfg,
       "name overlap": {k: overlap(*p) for k, p in pairs.items()},
       "tree sharing": {k: {f"u{j}": share(*p, j) for j in (1, 2, 3)} for k, p in pairs.items()}}
print(json.dumps(out), flush=True)
