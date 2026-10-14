import sys, pickle, random, time
import numpy as np
SRC = sys.argv[2]
sys.path.insert(0, SRC + "/src"); sys.path.insert(0, SRC + "/experiments/v2")
sys.setrecursionlimit(100000)
from trellis2.incremental import DEFAULT_CONFIG
import complex_grammars as CG
L = pickle.load(open(sys.argv[1], "rb"))
L.config = dict(DEFAULT_CONFIG, **L.config)
if not hasattr(L, "slots"):
    L.slots = [[i] for i in range(len(L.elems))]; L.real_of = list(range(len(L.elems)))
L.np_rng = np.random.default_rng(0); L.tables.version = 0
m = L.memory
g = CG.build("agree")
num = {}
for lhs, prods in g.items():
    for p in prods:
        if len(p) == 1 and p[0] not in g:
            num[p[0]] = "s" if lhs.endswith("_s") else ("p" if lhs.endswith("_p") else None)
rng = random.Random(1)
A = 40
seed = {e: L.next_sym for e in L.members[A] if m.kind[e] == 0 and num.get(m.token[e]) == "p"}
p, nc = L._relabel_patch(seed, L.own_classes(seed))
d0 = L.tables.delta(p)
tok = L._trial(("relabel", seed, nc, p, "split"))
total = d0
print("seed", round(d0, 1), flush=True)
t0 = time.time()
for step in range(int(sys.argv[3])):
    best = None
    for j in rng.sample(range(len(L.elems)), 25):
        for d, mv in L._candidates(j):
            if best is None or d < best[0]:
                best = (d, mv)
    if best is None or best[0] >= 0:
        print("no improving move"); break
    L._apply(best[1]); total += best[0]
    desc = best[1][0]
    if best[1][0] in ("relabel", "move"):
        els = best[1][1]
        toks = sorted({m.token[e] for e in els if m.kind[e] == 0})[:8]
        desc += f" {len(els)} elements {toks} syms {sorted({L.sym[e] for e in els})[:4]}"
    print(step, round(best[0], 1), "cumulative", round(total, 1), desc, round(time.time() - t0), flush=True)
L._undo(tok)
