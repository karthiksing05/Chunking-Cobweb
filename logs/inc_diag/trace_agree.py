import os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/logs/inc_snap/it6/src")
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/logs/inc_snap/it6/experiments/v2")
import numpy as np
import complex_grammars as CG
from trellis2.incremental import IncrementalLearner
train_all, held = CG.load("agree")
order = np.random.default_rng(13).permutation(len(train_all))
train = [train_all[i] for i in order[:2000]]
L = IncrementalLearner(seed=13, config=eval(sys.argv[2]) if len(sys.argv) > 2 else None)
def words(A):
    ks = sorted(L.rec_by_sym.get(A, ()), key=lambda k: -L.rec[(A, k)])[:5]
    return "/".join(k[1] if k[0] == "P" else f"({k[1]} {k[2]})" for k in ks)
orig = L._apply
def traced(move):
    d = move[0] + (f" then {move[2]}" if move[0] == "compound" else "")
    if L._journal is None:
        print("   apply", d, flush=True)
    return orig(move)
L._apply = traced
for k, e in enumerate(train[:int(sys.argv[1])]):
    print(k, " ".join(e.tokens), flush=True)
    L.observe(e.tokens)
    print("  bits", round(L.tables.bits(), 1), "K", L.tables.K, "tops", L._top_syms(k), flush=True)
for A in sorted(L.members):
    print(A, words(A))
