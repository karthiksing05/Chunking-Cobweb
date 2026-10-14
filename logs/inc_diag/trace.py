import os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/logs/inc_snap/it6/src")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.incremental import IncrementalLearner
cond, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ex = load_corpus(os.path.join(default_data_root(), CONDITIONS[cond]))
train, _ = v1_split(ex, seed)
L = IncrementalLearner(seed=seed, config=eval(sys.argv[4]) if len(sys.argv) > 4 else None)
orig = L._apply
def traced(move):
    m = L.memory
    desc = move[0]
    if move[0] in ("merge",):
        desc += f" {move[1]}<-{move[2]}"
    if move[0] == "compound":
        desc += f" then {move[2]}"
    if L._journal is None: print("   apply", desc, flush=True)
    return orig(move)
L._apply = traced
for k, e in enumerate(train[:n]):
    print(k, " ".join(e.tokens), flush=True)
    L.observe(e.tokens)
    tops = [L._top_syms(i) for i in range(len(L.experiences))]
    print("  bits", round(L.tables.bits(), 1), "K", L.tables.K, "tops", tops[-1], flush=True)
for A in sorted(L.members):
    print(A, sorted({(k if k[0] == 'P' else k) for k in L.rec_by_sym.get(A, ())}, key=str)[:8])
