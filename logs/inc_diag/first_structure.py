"""When the first structure forms: the best compounds and chunks scored at
that step, with the symbols' words (diagnostic)."""
import os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.incremental import IncrementalLearner
cond, seed = sys.argv[1], int(sys.argv[2])
cfg = eval(sys.argv[3]) if len(sys.argv) > 3 else {}
ex = load_corpus(os.path.join(default_data_root(), CONDITIONS[cond]))
train, _ = v1_split(ex, seed)
L = IncrementalLearner(seed=seed, config=cfg)
m = L.memory
def words(A):
    ks = sorted(L.rec_by_sym.get(A, ()), key=lambda k: -L.rec[(A, k)])[:4]
    return "/".join(k[1] if k[0] == "P" else f"({k[1]} {k[2]})" for k in ks)
orig_best = L._best_move
def best_move(i):
    if L.tables.K > 1 or True:
        cands = list(L._candidates(i))
        structure = sorted([(d, mv) for d, mv in cands if mv[0] in ("chunk", "joint", "inner chunk")], key=lambda x: x[0])[:3]
        splits = sorted([(d, mv) for d, mv in cands if mv[0] == "relabel" and len(mv) > 4 and any(m.is_root[e] for e in mv[1])], key=lambda x: x[0])[:8]
        comp = []
        for d, mv in splits:
            undo = L._trial(mv)
            f = sorted(L._followups(mv[1]), key=lambda x: x[0])[:3]
            names = {s: words(s) for s in set(mv[1].values())}
            L._undo(undo)
            comp.append((round(d + (f[0][0] if f else 0), 1), round(d, 1), [(round(x[0], 1), x[1][0], x[1][1]) for x in f], names))
        if structure or comp:
            print(f"  step {i}: best structure", [(round(d, 1), mv[0]) for d, mv in structure])
            for c in sorted(comp, key=lambda x: x[0])[:4]:
                print("    compound", c)
    return orig_best(i)
L._best_move = best_move
for k, e in enumerate(train[:int(sys.argv[4]) if len(sys.argv) > 4 else 20]):
    L.observe(e.tokens)
    print(k, " ".join(e.tokens), "| bits", round(L.tables.bits(), 1), "K", L.tables.K,
          "syms", {A: words(A) for A in sorted(L.members)}, flush=True)
