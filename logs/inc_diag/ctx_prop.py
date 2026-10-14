import sys, pickle
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/logs/inc_snap/it8/src")
sys.setrecursionlimit(100000)
from trellis2.incremental import DEFAULT_CONFIG
L = pickle.load(open(sys.argv[1], "rb"))
L.config = dict(DEFAULT_CONFIG, **L.config); L.config.update(profile_depth=3)
m = L.memory
for A in sorted(L.members):
    for groups in L._context_splits(A):
        toks = sorted({m.token[e] if m.kind[e] == 0 else str(L._key(e)) for e in groups[0]})
        best = min((d for d, mv in L._split_candidates(groups)), default=None)
        print(A, toks[:8], round(best, 1))
