"""Does the content tree contribute to chunk factors? (evaluation only)

After learning, for the composites of held-out parses: the share whose best
content concept beats the root (content gain > 0), the mean content and context
gains, and the content and context trees' mean leaf depth.

Usage: t3_cgain.py <cond> <cfg json> <seed> [tag]
"""
import os, sys, json, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.trellis3 import Trellis3

cond, cfg, seed = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
tag = sys.argv[4] if len(sys.argv) > 4 else ""
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
t0 = time.time()
L = Trellis3(seed=seed, **cfg)
for e in train:
    L.learn(e.tokens)
learn_s = time.time() - t0
cg, xg = [], []
for e in test[:40]:
    r = L.parse(e.tokens)
    for x in r["comps"].values():
        cg.append(L._gain(L.cnt, x.cnt))
        xg.append(L._gain(L.ctx, x.ctx))


def mean_depth(tree):
    leaves = tree.leaves()
    return round(sum(len(tree.ancestors(n)) - 1 for n in leaves) / max(len(leaves), 1), 2)


m = lambda v: round(sum(v) / len(v), 3) if v else None
print(json.dumps({"tag": tag, "cond": cond, "seed": seed, "learn s": round(learn_s),
                  "content gain > 0": m([g > 0 for g in cg]), "mean content gain": m(cg),
                  "mean context gain": m(xg), "content leaf depth": mean_depth(L.cnt),
                  "context leaf depth": mean_depth(L.ctx)}), flush=True)
