"""What the top of the context tree splits on: for the root's children and
grandchildren, size, share that are words, the most common left and right
neighbours, and the gold part-of-speech mix of the words under it (gold used
only to diagnose)."""
import os, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3
cond, cfg = sys.argv[1], json.loads(sys.argv[2])
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = Trellis3(seed=13, **cfg); g = CFG(target_grammar(cond))
for e in train:
    n0 = len(L.records); L.learn(e.tokens); lab = g.gold_labels(e.tokens, e.tree)
    for r in L.records[n0:]:
        r.memo["lab"] = lab.get(r.span, "?") if r.token is not None else "chunk:" + lab.get(r.span, "-")
at = defaultdict(list)
for r in L.records:
    if r.leaf is not None: at[id(r.leaf)].append(r)
def members(n):
    out, st = [], [n]
    while st:
        m = st.pop()
        if m.children: st.extend(m.children)
        else: out.extend(at.get(id(m), []))
    return out
def show(n, ind):
    ms = members(n); W = sum(m.weight for m in ms) or 1e-9
    words = [m for m in ms if m.token is not None]
    pos = Counter(m.memo["lab"] for m in words).most_common(4)
    top = lambda a: [v for v, _ in sorted(n.av.get(a, {}).items(), key=lambda kv: -kv[1])[:3]]
    print(ind + f"c{n.id} count {n.count:.0f} words {len(words)}/{len(ms)} l1 {top('l1')} r1 {top('r1')} POS {pos}")
for c in sorted(L.ctx.root.children, key=lambda c: -c.count):
    show(c, "")
    for gc in sorted(c.children, key=lambda c: -c.count)[:4]:
        show(gc, "    ")
