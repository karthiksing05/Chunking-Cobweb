"""Where do role errors in generation come from? (evaluation only)

Generation with the leaf rule pools what is stored at one leaf: a part's
fillers are the records at its context leaf (FILL), and an expanded content
leaf's parts are pooled over the records stored there (LPART/RPART). A leaf
holding records of different gold categories lets one stand in for the other.

For each tree: the share of stored records (weighted) that sit at a leaf whose
records do not all share their gold category, and the leaf impurity
(1 - majority share), for composites and for words. Gold labels only score.

Usage: t3_purity.py <cond> <cfg json> [seed] [tag]
"""
import os, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3

cond, cfg = sys.argv[1], json.loads(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
tag = sys.argv[4] if len(sys.argv) > 4 else ""
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
g = CFG(target_grammar(cond))
L = Trellis3(seed=seed, **cfg)
label = {}
for e in train:
    n0 = len(L.records)
    L.learn(e.tokens)
    gl = g.gold_labels(e.tokens, e.tree)
    for r in L.records[n0:]:
        label[r.rid] = gl.get(r.span, "-")          # "-": not a gold constituent


def report(which):
    at = defaultdict(list)
    for r in L.records:
        leaf = r.leaf if which == "ctx" else r.cleaf
        if leaf is not None:
            at[leaf.id].append(r)
    out = {}
    for kind in ("composite", "word"):
        mixed = imp = tot = 0.0
        for recs in at.values():
            rs = [r for r in recs if (r.token is None) == (kind == "composite")]
            if not rs:
                continue
            c = Counter()
            for r in recs:                         # the pool generation draws from
                c[label[r.rid]] += r.weight
            w = sum(r.weight for r in rs)
            tot += w
            if len(c) > 1:
                mixed += w
            imp += w * (1 - max(c.values()) / sum(c.values()))
        out[kind] = {"records at mixed leaves": round(mixed / max(tot, 1e-9), 3),
                     "leaf impurity": round(imp / max(tot, 1e-9), 3)}
    return out


def parts_report():
    """Content leaves: do the stored composites agree on their parts' gold categories?
    Generation pools LPART/RPART over a content leaf's records, so a leaf holding
    differently built composites can splice a left part of one to a right part of another."""
    at = defaultdict(list)
    for r in L.records:
        if r.token is None and r.cleaf is not None and r.used > 0:
            at[r.cleaf.id].append(r)
    mixed = tot = 0.0
    for recs in at.values():
        shapes = Counter((label.get(r.left.rid, "?"), label.get(r.right.rid, "?")) for r in recs)
        w = sum(r.weight for r in recs)
        tot += w
        if len(shapes) > 1:
            mixed += w
    return {"chosen composites at content leaves mixing part categories": round(mixed / max(tot, 1e-9), 3)}


print(json.dumps({"tag": tag, "cond": cond, "seed": seed, "ctx": report("ctx"), "cnt": report("cnt"),
                  "cnt parts": parts_report()}), flush=True)
