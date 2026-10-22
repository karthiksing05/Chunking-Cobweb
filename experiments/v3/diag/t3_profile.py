"""Profile TRELLIS v2 learning on the first sentences of a corpus."""
import os, sys, json, cProfile, pstats, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.trellis3 import Trellis3
cond, cfg, n = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = Trellis3(seed=13, **cfg); pr = cProfile.Profile(); t = time.time(); pr.enable()
for e in train[:n]: L.learn(e.tokens)
pr.disable()
def shape(tree):
    k, d, st = [], [], [(tree.root, 0)]
    while st:
        x, dd = st.pop()
        if x.children: k.append(len(x.children)); st.extend((c, dd + 1) for c in x.children)
        else: d.append(dd)
    return len(d), sum(d) / len(d), max(d)
print("seconds %.0f" % (time.time() - t), L.stats, "ctx leaves/mean depth/max", shape(L.ctx), "cnt", shape(L.cnt))
pstats.Stats(pr).sort_stats("tottime").print_stats(12)
