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
L._decided = L._branched = False; L._branches = None
m = L.memory
j = 1957
sp = sorted([(d, mv) for d, mv in L._candidates(j) if mv[0] == "relabel" and len(mv) > 4], key=lambda x: x[0])
print(len(sp), "splits")
for r, (d, mv) in enumerate(sp[:40]):
    toks = sorted({m.token[e] for e in mv[1] if m.kind[e] == 0})
    src = sorted({L.sym[e] for e in mv[1]})
    print(r, round(d, 1), len(mv[1]), src[:4], toks[:7], flush=True)
for r, (d, mv) in enumerate(sp[:12]):
    t0 = time.time()
    v = L._rollout(j, mv, 6)
    print("rollout", r, round(d, 1), round(v, 1), round(time.time() - t0), flush=True)
