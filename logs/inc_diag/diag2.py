import os, sys, copy
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/experiments/v2")
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
from inc_diag import run, gold_pos
from trellis2.mdl_search import word_classes
from collections import Counter
cond, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
L, train = run(cond, seed, n)
pos = gold_pos(cond)
m = L.memory
def compound(L, mapping, label):
    p, nc = L._relabel_patch(mapping, L.own_classes(mapping))
    d0 = L.tables.delta(p)
    L2 = copy.deepcopy(L)
    L2._apply(("relabel", mapping, nc, p))
    best = None
    for i in range(len(L2.experiences)):
        labs = L2._top_syms(i)
        for pair in set(zip(labs, labs[1:])):
            for d, mv in L2._chunk_moves(pair):
                if best is None or d < best[0]: best = (d, pair)
        if len(labs) >= 3:
            for d, mv in L2._joint_chunk_moves(labs, 4):
                if best is None or d < best[0]: best = (d, mv[1] and 'joint')
    print(label, "split", round(d0,1), "best next", best and round(best[0],1), "compound", best and round(d0+best[0],1))
A = sorted(L.members)[0]
for k, groups in enumerate(L._tree_splits(A)):
    mapping = {}
    for j, g in enumerate(groups):
        for e in g: mapping[e] = L.next_sym + j
    purity = [Counter(pos[m.token[e]] for e in g if m.kind[e]==m.PRIMITIVE).most_common(3) for g in groups]
    compound(L, mapping, f"tree {k} groups={len(groups)} sizes={[len(g) for g in groups]} {purity[:2]}")
path = word_classes([e.tokens for e in train[:n]], 0.001)
for cls in path[-8:]:
    groups = {}
    for e in L.members[A]:
        if m.kind[e] == m.PRIMITIVE:
            groups.setdefault(cls[m.token[e]], set()).add(e)
    gs = sorted(groups.values(), key=len, reverse=True)[1:]
    mapping = {e: L.next_sym + j for j, g in enumerate(gs) for e in g}
    compound(L, mapping, f"word classes K={len(set(cls.values()))}")
