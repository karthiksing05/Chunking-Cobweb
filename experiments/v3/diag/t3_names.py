"""Where does a word's learned name lose category information? (TinyStories)

Learn n sentences once (t3_stories corpus), then score word-category accuracy
(nearest other word has the same coarse tag) under alternative name
constructions over the same trees:
  current      the learner's name (reservoir of type_k leaves, bands)
  res-path     the same reservoir, every concept on each leaf's path
  all-bands    every stored leaf of the word, bands
  all-path     every stored leaf of the word, every concept on the path
  all-parent   every stored leaf of the word, its parent concept only (finest category)
  leaf-window  the stored instances' raw window values (what the tree was given)
  raw-counts   cosine over raw left/right neighbour counts (baseline)
Usage: t3_names.py <cfg json> <seed> <n> [vocab]
"""
import json, math, random, sys
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import nltk
from trellis2.stories import read_sentences, simple_sentences
from trellis2.trellis3 import Trellis3

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
vocab = int(sys.argv[4]) if len(sys.argv) > 4 else 250
corpus = simple_sentences(read_sentences(), vocab_size=vocab)
random.Random(seed).shuffle(corpus)
train = corpus[:n]
coarse = lambda t: t[:2] if t not in ("TO",) else "IN"
gc = defaultdict(Counter)
for s in corpus[:20000]:
    for w, t in nltk.pos_tag(s):
        gc[w][coarse(t)] += 1
gold = {w: c.most_common(1)[0][0] for w, c in gc.items()}
L = Trellis3(seed=seed, **cfg)
raw = defaultdict(Counter)
for s in train:
    L.learn(s)
    p = ["<s>"] + list(s) + ["</s>"]
    for i in range(1, len(p) - 1):
        raw[p[i]]["L:" + p[i - 1]] += 1
        raw[p[i]]["R:" + p[i + 1]] += 1
leaves = defaultdict(list)
for r in L.records:
    if r.token is not None and r.leaf is not None:
        leaves[r.token].append(r)
words = [w for w in leaves if w in gold]


def norm(c):
    t = sum(c.values()) or 1.0
    return {k: v / t for k, v in c.items()}


def bag(paths, how):
    out = Counter()
    for path in paths:
        if how == "bands":
            ids = L._path_ids(path, "bands")
        elif how == "path":
            ids = [f"c{x.id}" for x in path[1:-1]] or ["-"]
        else:
            ids = [f"c{path[-2].id}"] if len(path) > 1 else ["-"]
        for v in ids:
            out[v] += 1.0 / len(ids)
    return norm(out)


def overlap(a, b):
    return sum(min(p, b.get(u, 0.0)) for u, p in a.items())


def cosine(a, b):
    num = sum(v * b.get(u, 0) for u, v in a.items())
    den = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return num / den if den else 0.0


def acc(names, sim=overlap):
    ok = []
    for w in words:
        best = max((x for x in words if x != w), key=lambda x: sim(names[w], names[x]))
        ok.append(gold[best] == gold[w])
    return round(sum(ok) / len(ok), 3)


L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
path_of = lambda leaf: L.ctx.ancestors(leaf)[::-1]
res = {w: [lf for _, lf in L.type_res.get((w,), [])] for w in words}
win = lambda r: Counter({f"{a}:{v}": 1 for a, v in r.ctx.items() if a[0] in "lr" and a[1:].isdigit() and isinstance(v, str)})
out = {"cfg": cfg, "seed": seed, "n": n, "words": len(words), "records": len(L.records), "ctx nodes": sum(1 for _ in L.ctx.nodes()) if hasattr(L.ctx, "nodes") else None,
       "accuracy": {
           "current": acc({w: norm(Counter(L._ctx_name(r))) if isinstance(L._ctx_name(r), list) else L._ctx_name(r) for w in words for r in [leaves[w][0]]}),
           "res-bands": acc({w: bag([path_of(lf) for lf in res[w]], "bands") for w in words}),
           "res-path": acc({w: bag([path_of(lf) for lf in res[w]], "path") for w in words}),
           "all-bands": acc({w: bag([path_of(r.leaf) for r in leaves[w]], "bands") for w in words}),
           "all-path": acc({w: bag([path_of(r.leaf) for r in leaves[w]], "path") for w in words}),
           "all-parent": acc({w: bag([path_of(r.leaf) for r in leaves[w]], "parent") for w in words}),
           "leaf-window": acc({w: sum((win(r) for r in leaves[w]), Counter()) for w in words}, cosine),
           "raw-counts": acc({w: raw[w] for w in words}, cosine)}}
print(json.dumps(out), flush=True)
