"""TRELLIS v2 in lockstep on two trees: tree="python" vs tree="cu" (same
config and seed), comparing both hierarchies (context and content) node by
node after every learned sentence (ids, parents, children, counts, av, ss,
sstot, pav), floats with ==.

Usage: cu_equiv_trellis.py <cond|stories> <cfg json> <seed> <n> [fast]
  fast: 0 (default; reference arithmetic: Python fast_ec=False vs C++ fast=False)
        or 1 (Python fast_ec=True vs C++ fast=True)
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from trellis2.trellis3 import Trellis3  # noqa: E402

cond, cfg, seed, n = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
fast = len(sys.argv) > 5 and sys.argv[5] == "1"
if cond == "stories":
    from trellis2.stories import read_sentences, simple_sentences
    corpus = simple_sentences(read_sentences(), vocab_size=250)
    random.Random(seed).shuffle(corpus)
    train = corpus[:n]
else:
    from trellis2 import load_corpus, v1_split
    from trellis2.data import CONDITIONS, default_data_root
    tr, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    train = [e.tokens for e in tr][:n]


def snapshot(t):
    out = []
    for nd in t.nodes():
        p = nd.parent
        out.append((nd.id, None if p is None else p.id, [c.id for c in nd.children], nd.count, nd.sstot,
                    [(a, list(d.items())) for a, d in nd.av.items()], list(nd.ss.items()),
                    [(a, list(d.items())) for a, d in nd.pav.items()]))
    return out


Lp = Trellis3(seed=seed, **{**cfg, "tree": "python", "tree_fast": fast})
Lc = Trellis3(seed=seed, **{**cfg, "tree": "cu", "tree_fast": fast})
tp = tc = 0.0
for k, s in enumerate(train, 1):
    t0 = time.time()
    Lp.learn(s)
    t1 = time.time()
    Lc.learn(s)
    t2 = time.time()
    tp += t1 - t0
    tc += t2 - t1
    for which in ("ctx", "cnt"):
        a, b = snapshot(getattr(Lp, which)), snapshot(getattr(Lc, which))
        if a != b:
            d = next(((x, y) for x, y in zip(a, b) if x != y), (len(a), len(b)))
            names = ("id", "parent", "children", "count", "sstot", "av", "ss", "pav")
            parts = []
            if isinstance(d[0], tuple):
                for nm, x, y in zip(names, d[0], d[1]):
                    if x != y:
                        if nm in ("av", "pav"):
                            for (a1, l1), (a2, l2) in zip(x, y):
                                if (a1, l1) != (a2, l2):
                                    parts.append(f"{nm}[{a1}/{a2}]: python {[q for q in l1 if q not in l2]} cu {[q for q in l2 if q not in l1]} order-equal-sets {sorted(map(str, l1)) == sorted(map(str, l2))}")
                            if len(x) != len(y):
                                parts.append(f"{nm} attrs python {[q[0] for q in x]} cu {[q[0] for q in y]}")
                        else:
                            parts.append(f"{nm}: python {x} cu {y}")
            print(f"DIFFERENT after sentence {k} ({' '.join(s)}) in {which} at node {d[0][0] if isinstance(d[0], tuple) else d}: " + "; ".join(parts)[:3000], flush=True)
            sys.exit(1)
    if Lp.stats != Lc.stats:
        print(f"DIFFERENT stats after sentence {k}: {Lp.stats} vs {Lc.stats}", flush=True)
        sys.exit(1)
    if k % 10 == 0 or k == len(train):
        print(f"{k} sentences identical (ctx {len(snapshot(Lp.ctx))} nodes, cnt {len(snapshot(Lp.cnt))} nodes); "
              f"learn s python {tp:.1f} cu {tc:.1f}", flush=True)
print(f"IDENTICAL after {len(train)} sentences; stats {Lp.stats}", flush=True)
