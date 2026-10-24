"""Profile learning on the first n sentences of a condition (cProfile, top 30 by cumulative time)."""
import os, sys, json, cProfile, pstats
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.trellis3 import Trellis3
cond, cfg, n = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = Trellis3(seed=13, **cfg)
pr = cProfile.Profile()
pr.enable()
for e in train[:n]:
    L.learn(e.tokens)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(30)
pstats.Stats(pr).sort_stats("tottime").print_stats(15)
