import os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.incremental import IncrementalLearner
cond, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ex = load_corpus(os.path.join(default_data_root(), CONDITIONS[cond]))
train, _ = v1_split(ex, seed)
L = IncrementalLearner(seed=seed)
L._step_orig = L._step
L._step = lambda i: None
for e in train[:n]:
    L.observe(e.tokens)
    if len(L.experiences) < n:
        L._step_orig(len(L.experiences) - 1)
i = n - 1
print("state", round(L.tables.bits(), 1), "K", L.tables.K)
cands = list(L._candidates(i))
best = sorted(cands, key=lambda x: x[0])[:5]
for d, mv in best:
    print("cand", mv[0], round(d, 1))
splits = sorted([(d, mv) for d, mv in cands if mv[0] == "relabel" and len(mv) > 4 and any(L.memory.is_root[e] for e in mv[1])], key=lambda x: x[0])
print("n splits", len(splits))
res = []
for d, mv in splits:
    undo = L._trial(mv)
    f = min(L._followups(mv[1]), key=lambda x: x[0], default=None)
    groups = {}
    for e, s in mv[1].items():
        groups.setdefault(s, set()).add(L.memory.token[e])
    L._undo(undo)
    if f: res.append((d + f[0], d, f[0], f[1][0], [sorted(g) for g in groups.values()][:4]))
for r in sorted(res)[:10]:
    print("compound", [round(x, 1) if isinstance(x, float) else x for x in r])
