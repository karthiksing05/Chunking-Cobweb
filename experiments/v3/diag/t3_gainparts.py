"""Why is the content-tree gain <= 0? Per attribute, the fit term
log2 P(x_a | C) - log2 P(x_a | root) along a composite's content path, with the
value-vocabulary size the Laplace smoothing uses (evaluation only).

Usage: t3_gainparts.py <cond> <cfg json> <seed> [n experiences]
"""
import os, sys, json, math
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.trellis3 import Trellis3, log_prob, LN2

cond, cfg, seed = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 320
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
L = Trellis3(seed=seed, **cfg)
for e in train[:n]:
    L.learn(e.tokens)

for tree, name in ((L.cnt, "content"), (L.ctx, "context")):
    root = tree.root
    print(f"== {name} tree: root count {root.count:.0f}; value vocabulary per attribute:",
          {a: len(root.av.get(a, {})) for a in root.av}, flush=True)

tot = {}
for e in test[:20]:
    r = L.parse(e.tokens)
    for key, x in list(r["comps"].items())[:40]:
        for tree, name, inst in ((L.cnt, "content", x.cnt), (L.ctx, "context", x.ctx)):
            root = tree.root
            path = tree.categorize(inst)
            for depth, node in enumerate(path[1:4], 1):
                for a, v in inst.items():
                    fa = (log_prob(node, {a: v}, root, L.alpha) - log_prob(root, {a: v}, root, L.alpha)) / LN2
                    # the same with the smoothing mass shrunk to one pseudo-count spread over the vocabulary
                    size = len(root.av.get(a, {})) + 1
                    fb = (log_prob(node, {a: v}, root, 1.0 / size) - log_prob(root, {a: v}, root, 1.0 / size)) / LN2
                    s = tot.setdefault((name, depth, a), [0, 0.0, 0.0, 0.0])
                    s[0] += 1; s[1] += fa; s[2] += fb
                    s[3] += math.log2(max(node.count, 1e-9) / root.count)
print("tree depth attr | mean fit (alpha=1) | mean fit (alpha=1/V) | mean log2 share")
for (name, depth, a), (c, fa, fb, sh) in sorted(tot.items()):
    print(f"{name:8s} {depth} {a:4s} | {fa / c:8.2f} | {fb / c:8.2f} | {sh / c:7.2f}  (n={c})")
