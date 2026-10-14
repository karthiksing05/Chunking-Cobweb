import sys, pickle, random, time
import numpy as np
SRC = sys.argv[2]
sys.path.insert(0, SRC + "/src"); sys.path.insert(0, SRC + "/experiments/v2")
sys.setrecursionlimit(100000)
from trellis2.incremental import DEFAULT_CONFIG
L = pickle.load(open(sys.argv[1], "rb"))
L.config = dict(DEFAULT_CONFIG, lookahead_dissolve=True)
if not hasattr(L, "slots"):
    L.slots = [[i] for i in range(len(L.elems))]; L.real_of = list(range(len(L.elems)))
L.np_rng = np.random.default_rng(0); L.tables.version = 0
for a in ("_decided", "_branched"):
    setattr(L, a, getattr(L, a, False))
L._branches = None; L._stall = 0; L._budget = int(sys.argv[4]) if len(sys.argv) > 4 else 1
rng = random.Random(2)
t0 = time.time()
print("start", round(L.tables.bits(), 1), flush=True)
for k in range(int(sys.argv[3])):
    j = rng.randrange(len(L.elems))
    before = L.tables.bits()
    acc = L._best_move(j); L._budget = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print(k, j, acc, round(L.tables.bits() - before, 1), {x: v for x, v in L.stats.items() if x.startswith("accepted")}, round(time.time() - t0), flush=True)
