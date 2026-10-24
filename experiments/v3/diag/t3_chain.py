"""Why is the content tree a chain? (evaluation only)

Learns one corpus and records, for every ifit into each tree, which operation
Cobweb chose at each depth (best / new / merge / split, plus 'fork' when the
descent ends at a leaf). Then reports the tree shape: leaf depth, branching of
internal nodes, the share of internal nodes with exactly two children one of
which is a leaf, and node counts and child counts along the deepest path.

Usage: t3_chain.py <cond> <cfg json> [seed] [tag]
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
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
L = Trellis3(seed=seed, **cfg)
ops = {"ctx": collections.Counter(), "cnt": collections.Counter()}
name = {id(L.ctx): "ctx", id(L.cnt): "cnt"}
orig_best, orig_ifit = ic.IncrementalCobwebTree._best_operation, ic.IncrementalCobwebTree.ifit
state = {"depth": 0}


def best(self, cur, x, w):
    op, b1, b2 = orig_best(self, cur, x, w)
    d = len(self.ancestors(cur)) - 1
    ops[name.get(id(self), "?")][(min(d // 5 * 5, 50), op)] += 1
    return op, b1, b2


def ifit(self, x, w=1.0):
    leaf = orig_ifit(self, x, w)
    ops[name.get(id(self), "?")][("end", "leaf children" if leaf.parent and len(leaf.parent.children) == 2 else "other")] += 1
    return leaf


ic.IncrementalCobwebTree._best_operation = best
ic.IncrementalCobwebTree.ifit = ifit
for e in train:
    L.learn(e.tokens)
    name = {id(L.ctx): "ctx", id(L.cnt): "cnt"}


def shape(tree):
    nodes = list(tree.nodes())
    internal = [n for n in nodes if n.children]
    leaves = [n for n in nodes if not n.children]
    depth = {id(n): len(tree.ancestors(n)) - 1 for n in leaves}
    deep = max(leaves, key=lambda n: depth[id(n)])
    path = tree.ancestors(deep)[::-1]
    pairs = sum(1 for n in internal if len(n.children) == 2 and any(not c.children for c in n.children))
    bf = collections.Counter(min(len(n.children), 6) for n in internal)
    return {"nodes": len(nodes), "leaves": len(leaves),
            "mean leaf depth": round(sum(depth.values()) / len(leaves), 1), "max leaf depth": max(depth.values()),
            "internal branching (6 = 6+)": dict(sorted(bf.items())),
            "share of internal nodes = leaf + one subtree": round(pairs / max(len(internal), 1), 3),
            "deepest path (count, children) every 10th": [(round(n.count, 1), len(n.children)) for n in path[::10]]}


def opmix(c):
    by = collections.defaultdict(dict)
    for (d, op), k in c.items():
        by[str(d)][op] = k
    return dict(by)


print(json.dumps({"tag": tag, "cond": cond, "seed": seed, "ctx shape": shape(L.ctx), "cnt shape": shape(L.cnt),
                  "ctx ops by depth (5s)": opmix(ops["ctx"]), "cnt ops by depth (5s)": opmix(ops["cnt"])}), flush=True)
