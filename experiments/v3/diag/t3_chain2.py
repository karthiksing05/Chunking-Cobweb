"""The content tree's merge ladder (evaluation only).

Records, per tree: the weight distribution of fitted instances; operations by
instance weight; and how many merges wrap a node that an earlier merge made
(a ladder step). With 'noladder' as the 5th argument, merge and split are only
considered for instances of weight >= the 6th argument (default 0.5), and the
script reports the resulting tree shape.

Usage: t3_chain2.py <cond> <cfg json> [seed] [tag] [restructure_min_w]
"""
import os, sys, json, collections
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.trellis3 import Trellis3
from trellis2 import incremental_cobweb as ic

cond, cfg = sys.argv[1], json.loads(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
tag = sys.argv[4] if len(sys.argv) > 4 else ""
rmin = float(sys.argv[5]) if len(sys.argv) > 5 else None
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
L = Trellis3(seed=seed, **cfg)
T = lambda self: "ctx" if self is L.ctx else "cnt"
wb = lambda w: "<0.05" if w < 0.05 else "<0.2" if w < 0.2 else "<0.5" if w < 0.5 else ">=0.5"
weights = {"ctx": collections.Counter(), "cnt": collections.Counter()}
ops = {"ctx": collections.Counter(), "cnt": collections.Counter()}
merged, ladder = set(), {"ctx": 0, "cnt": 0}
ob, oi, om = ic.IncrementalCobwebTree._best_operation, ic.IncrementalCobwebTree.ifit, ic.IncrementalCobwebTree._merge
cur_w = {"w": 1.0}


def best(self, cur, x, w):
    op, b1, b2 = ob(self, cur, x, w)
    if rmin is not None and w < rmin and op in ("merge", "split"):
        op = max(((((sum(c.count * self.ec(c) for c in cur.children) + (b1.count + w) * self._ec_with(b1, x, w)
                     - b1.count * self.ec(b1)) / (cur.count + w) - self._ec_with(cur, x, w)) / len(cur.children), "best"),
                  (((sum(c.count * self.ec(c) for c in cur.children) + w * self._ec_alone(x)) / (cur.count + w)
                    - self._ec_with(cur, x, w)) / (len(cur.children) + 1), "new")))[1]
    ops[T(self)][(wb(w), op)] += 1
    return op, b1, b2


def ifit(self, x, w=1.0):
    weights[T(self)][wb(w)] += 1
    return oi(self, x, w)


def merge(self, cur, b1, b2):
    if id(b1) in merged or id(b2) in merged:
        ladder[T(self)] += 1
    m = om(self, cur, b1, b2)
    merged.add(id(m))
    return m


ic.IncrementalCobwebTree._best_operation, ic.IncrementalCobwebTree.ifit, ic.IncrementalCobwebTree._merge = best, ifit, merge
for e in train:
    L.learn(e.tokens)


def depth(tree):
    lv = tree.leaves()
    d = [len(tree.ancestors(n)) - 1 for n in lv]
    return {"mean leaf depth": round(sum(d) / len(d), 1), "max": max(d), "leaves": len(lv)}


def mix(c):
    out = collections.defaultdict(dict)
    for (b, op), k in c.items():
        out[b][op] = k
    return dict(out)


print(json.dumps({"tag": tag, "cond": cond, "seed": seed, "restructure min w": rmin,
                  "ctx": depth(L.ctx), "cnt": depth(L.cnt), "weights": {k: dict(v) for k, v in weights.items()},
                  "ops by weight": {k: mix(v) for k, v in ops.items()},
                  "merges wrapping a merged node": ladder}), flush=True)
