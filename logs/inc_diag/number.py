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
L.np_rng = np.random.default_rng(0)
m = L.memory
# gold number of words (diagnosis only)
g = CG.build("agree")
num = {}
for lhs, prods in g.items():
    for p in prods:
        if len(p) == 1 and p[0] not in g:
            if lhs.endswith("_s"): num[p[0]] = "s"
            if lhs.endswith("_p"): num[p[0]] = "p"
for A in sorted(L.members):
    prims = [e for e in L.members[A] if m.kind[e] == 0]
    if not prims: continue
    toks = {m.token[e] for e in prims}
    print("S", A, len(L.members[A]), sorted(toks)[:10], {n: sum(1 for t in toks if num.get(t) == n) for n in "sp"})
t0 = time.time()
for A in sorted(L.members):
    prims = [e for e in L.members[A] if m.kind[e] == 0 and num.get(m.token[e]) == "p"]
    rest = [e for e in L.members[A] if m.kind[e] == 0 and num.get(m.token[e]) == "s"]
    if not prims or not rest: continue
    seed = {e: L.next_sym for e in prims}
    p, nc = L._relabel_patch(seed, L.own_classes(seed))
    print("oracle seed", A, "alone", round(L.tables.delta(p), 1), flush=True)
    joint = L._propagate_graph(seed, int(sys.argv[3]))
    p, nc = L._relabel_patch(joint, L.own_classes(joint))
    print("   graph joint", len(seed), "->", len(joint), "symbols", len(set(joint.values())), round(L.tables.delta(p), 1), round(time.time() - t0), flush=True)
# data-driven proposals
for A in sorted(L.members):
    props = L._tree_splits(A) + L._word_class_splits(A) + L._context_splits(A)
    for groups in props[:8]:
        seed = {e: L.next_sym + k for k, g_ in enumerate(groups) for e in g_}
        joint = L._propagate_graph(seed, int(sys.argv[3]))
        p, nc = L._relabel_patch(joint, L.own_classes(joint))
        toks = sorted({m.token[e] for g_ in groups for e in g_ if m.kind[e] == 0})[:6]
        print("prop", A, toks, len(joint), round(L.tables.delta(p), 1), flush=True)
