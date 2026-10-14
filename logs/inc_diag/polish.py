"""Diagnostic: from a pickled state, sweep the local moves over every stored
experience (evaluation of the move set's power only)."""
import os, sys, pickle, time
from collections import Counter
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
sys.setrecursionlimit(100000)
L = pickle.load(open(sys.argv[1], "rb"))
from trellis2.incremental import DEFAULT_CONFIG
L.config = dict(DEFAULT_CONFIG, **{k: v for k, v in L.config.items() if k in ("compound",) and False})
L.config.update(dict(eval(sys.argv[2])) if len(sys.argv) > 2 else {})
t0 = time.time()
print("start", round(L.tables.bits(), 1), flush=True)
for sweep in range(int(sys.argv[3]) if len(sys.argv) > 3 else 2):
    before = Counter(L.stats)
    for i in range(len(L.experiences)):
        for _ in range(3):
            if not L._best_move(i):
                break
        L.reanalyse(i)
    acc = {k: v - before[k] for k, v in L.stats.items() if k.startswith("accepted") and v - before[k]}
    print("sweep", sweep, round(L.tables.bits(), 1), "K", L.tables.K, acc, round(time.time() - t0), flush=True)
L.verify()
