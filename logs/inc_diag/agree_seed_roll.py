import sys, pickle, time
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
L._decided = L._branched = False; L._branches = None; L._stall = 0; L._budget = 1
m = L.memory
j = 1957
sp = sorted([(d, mv) for d, mv in L._candidates(j) if mv[0] == "relabel" and len(mv) > 4 and mv[4] in ("word classes", "context")], key=lambda x: x[0])
for d, mv in sp[:10]:
    toks = sorted({m.token[e] for e in mv[1] if m.kind[e] == 0})[:6]
    t0 = time.time()
    v = L._rollout(j, mv, 6)
    print(round(d, 1), mv[4], toks, "rollout", round(v, 1), round(time.time() - t0), flush=True)
