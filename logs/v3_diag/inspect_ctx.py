import os, sys, json
from collections import Counter
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.candidates import CandidateLearner
from trellis2.data import CONDITIONS, default_data_root
cond = sys.argv[1]; cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = CandidateLearner(seed=13, **cfg)
for e in train:
    L.learn(e.tokens)

def show(tree, node, depth, maxd, attrs):
    if depth > maxd:
        return
    summ = {a: [f"{v}:{c:.0f}" for v, c in sorted(node.av.get(a, {}).items(), key=lambda kv: -kv[1])[:6]] for a in attrs}
    print("  " * depth + f"n{node.id} count={node.count:.0f} " + " | ".join(f"{a}={summ[a]}" for a in attrs))
    for c in sorted(node.children, key=lambda c: -c.count):
        show(tree, c, depth + 1, maxd, attrs)
print("CONTEXT TREE")
show(L.ctx, L.ctx.root, 0, 3, ("kind", "first", "l1", "r1"))
print("CONTENT TREE")
show(L.cnt, L.cnt.root, 0, 2, ("L", "R", "L1", "L2", "R1", "R2", "S1"))
