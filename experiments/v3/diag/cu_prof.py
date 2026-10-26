"""Profile TRELLIS learning on the cu tree (cProfile; for perf use the C++ symbols).
Usage: cu_prof.py <cond> <cfg json> <seed> <n> [tree] [fast]"""
import cProfile
import json
import os
import pstats
import sys
import time

sys.path.insert(0, "src")
from trellis2 import load_corpus, v1_split  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cond, cfg, seed, n = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tree = sys.argv[5] if len(sys.argv) > 5 else "cu"
fast = (sys.argv[6] == "1") if len(sys.argv) > 6 else True
tr, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
L = Trellis3(seed=seed, **{**cfg, "tree": tree, "tree_fast": fast})
prof = os.environ.get("CPROF", "1") == "1"
pr = cProfile.Profile()
t0 = time.time()
if prof:
    pr.enable()
for e in tr[:n]:
    L.learn(e.tokens)
if prof:
    pr.disable()
print("learn s", round(time.time() - t0, 2))
if prof:
    pstats.Stats(pr).sort_stats("tottime").print_stats(20)
