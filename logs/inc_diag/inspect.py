import os, sys, pickle
from collections import Counter
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
sys.setrecursionlimit(100000)
import numpy as np
from trellis2.data import target_grammar
from trellis2.evaluation import CFG
path, cond = sys.argv[1], sys.argv[2]
L = pickle.load(open(path, "rb"))
cfg = CFG(target_grammar(cond))
m = L.memory
n_roots = Counter(len(r) for r in L.roots)
print("bits", round(L.tables.bits(), 1), "K", L.tables.K, "M", L.tables.M, "pieces per sentence", sorted(n_roots.items()))
tops = Counter(tuple(L._top_syms(i)) for i in range(len(L.experiences)))
print("top sequences", tops.most_common(12))
for A in sorted(L.members):
    yields = Counter()
    for e in L.members[A]:
        i, j = m.span[e]
        toks = m.experiences[m.experience_of[e]][i:j]
        yields[" ".join(toks) if len(toks) <= 3 else " ".join(toks[:2]) + " .. " + toks[-1]] += 1
    keys = Counter({k: L.rec[(A, k)] for k in L.rec_by_sym.get(A, ())})
    print(f"S{A} n={len(L.members[A])} keys={[(k if k[0]=='P' else (k[1],k[2]), int(v)) for k, v in keys.most_common(6)]}")
g = L.grammar
rng = np.random.default_rng(0)
samples, _ = L.generate(300, rng)
ok = [cfg.recognizes(t) for t, _ in samples]
whole = [len(tr.roots) == 1 for _, tr in samples]
print("gen commission", 1 - np.mean(ok), "whole share", np.mean(whole), "commission among whole", 1 - np.mean([o for o, w in zip(ok, whole) if w] or [1]))
bad = [" ".join(t) for (t, tr), o in zip(samples, ok) if not o][:10]
print("bad samples", bad)
