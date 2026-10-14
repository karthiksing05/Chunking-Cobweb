import sys, pickle
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/logs/inc_snap/it8/src")
sys.setrecursionlimit(100000)
L = pickle.load(open(sys.argv[1], "rb"))
m = L.memory
preps = {"in", "on", "with", "under"}
A = max(L.members, key=lambda s: sum(1 for e in L.members[s] if m.kind[e] == 0 and m.token[e] in preps))
grp = {e for e in L.members[A] if m.kind[e] == 0 and m.token[e] in preps}
print("symbol", A, "preps", len(grp), "of", len(L.members[A]))
mapping = {e: L.next_sym for e in grp}
for depth in range(0, 7):
    mp = L._propagate(mapping, depth) if depth else mapping
    for name, remap in (("own", L.own_classes(mp)), ("plain", None)):
        p, nc = L._relabel_patch(mp, remap, fresh_renamed=depth > 0)
        d = L.tables.delta(p)
        tok = L._trial(("relabel", mp, nc, p, "split"))
        cont = 0.0
        for _ in range(8):
            best = None
            for j in list(L.queue)[:16] + [len(L.experiences) - 1]:
                for dd, mv in L._candidates(j):
                    if best is None or dd < best[0]:
                        best = (dd, mv)
            if best is None or best[0] >= 0:
                break
            L._apply(best[1]); cont += best[0]
        L._undo(tok)
        print("depth", depth, name, "moved", len(mp), "delta", round(d, 1), "then greedy", round(cont, 1))
