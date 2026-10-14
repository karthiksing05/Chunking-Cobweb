"""At the step where the first structure forms: greedy rollouts after each of
the best first moves (copies of the learner; diagnostic)."""
import os, sys, copy
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.incremental import IncrementalLearner
cond, seed, cfg, upto, extra = sys.argv[1], int(sys.argv[2]), eval(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ex = load_corpus(os.path.join(default_data_root(), CONDITIONS[cond]))
train, _ = v1_split(ex, seed)
L = IncrementalLearner(seed=seed, config=cfg)
for e in train[:upto]:
    L.observe(e.tokens)
L._step_orig = L._step
L._step = lambda i: None
L.observe(train[upto].tokens)
i = upto
m = L.memory
def words(L, A):
    ks = sorted(L.rec_by_sym.get(A, ()), key=lambda k: -L.rec[(A, k)])[:3]
    return "/".join(k[1] if k[0] == "P" else f"({k[1]} {k[2]})" for k in ks)
cands = list(L._candidates(i))
firsts = []
for d, mv in cands:
    if mv[0] in ("chunk", "joint", "inner chunk"):
        firsts.append((d, mv))
    elif mv[0] == "relabel" and len(mv) > 4 and any(m.is_root[e] for e in mv[1]):
        undo = L._trial(mv)
        for f in sorted(L._followups(mv[1]), key=lambda x: x[0])[:3]:
            firsts.append((d + f[0], ("compound", mv, f[1])))
        L._undo(undo)
firsts.sort(key=lambda x: x[0])
print("state", round(L.tables.bits(), 1))
for d, mv in firsts[:8]:
    L2 = copy.deepcopy(L)
    L2._apply(mv)
    desc = mv[2] if mv[0] == "compound" else mv[0]
    # Continue: greedy moves over every stored experience (the rollout), plus `extra` new sentences
    for _ in range(3):
        for j in range(len(L2.experiences)):
            while L2._best_move(j):
                pass
    for e in train[upto + 1: upto + 1 + extra]:
        L2.observe(e.tokens)
    syms = {A: words(L2, A) for A in sorted(L2.members)}
    print(round(d, 1), desc, "-> after rollout", round(L2.tables.bits(), 1), "| +", extra, "sentences:", round(L2.tables.bits(), 1), flush=True)
    print("     ", syms)
