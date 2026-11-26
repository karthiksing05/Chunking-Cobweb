"""Round 233: why are TRELLIS context leaves category-impure? Train (cfg window w) and print the members of a few mixed leaves
with their context instances. usage: r233_leafprobe.py <seed> <n> <window>"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r230_grammar as G  # noqa
from trellis2.trellis3_r223 import Trellis3R223  # noqa
seed, n, w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfg = json.load(open("experiments/v3/results/story_gen/r214p01/cfg_w5.json")); cfg["window"] = w
L = Trellis3R223(seed=seed, **cfg)
for s, _ in G.load_corpus(seed, n)["train"]:
    L.learn(s)
print("ctx_attrs", L.ctx_attrs)
g = defaultdict(list)
for e in L.records:
    if e.token is not None and e.leaf is not None:
        g[e.leaf.id].append(e)
mixed = [v for v in g.values() if len({G.syncat(e.token) for e in v}) > 1]
print("leaves", len(g), "mixed", len(mixed), "occurrences in mixed", sum(len(v) for v in mixed))
for v in sorted(mixed, key=len, reverse=True)[:6]:
    print("--- leaf of", len(v))
    for e in v[:6]:
        print("  ", e.token, G.syncat(e.token), {k: (str(x)[:40]) for k, x in (e.ctx or {}).items()})
# leaf structure: how many records share a leaf and the depth of the tree
dep = []
for v in g.values():
    d, x = 0, v[0].leaf
    while x.parent is not None:
        d += 1; x = x.parent
    dep.append(d)
print("leaf depth mean", sum(dep) / len(dep), "max", max(dep))
