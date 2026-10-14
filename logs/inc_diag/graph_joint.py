import sys, pickle
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/logs/inc_snap/it11/src")
sys.setrecursionlimit(100000)
from trellis2.incremental import DEFAULT_CONFIG
L = pickle.load(open(sys.argv[1], "rb"))
L.config = dict(DEFAULT_CONFIG, **{k: v for k, v in L.config.items() if k in DEFAULT_CONFIG})
m = L.memory
def words(A):
    ks = sorted(L.rec_by_sym.get(A, ()), key=lambda k: -L.rec[(A, k)])[:6]
    return "/".join(k[1] if k[0] == "P" else f"({k[1]} {k[2]})" for k in ks)
for A in sorted(L.members):
    print(A, len(L.members[A]), words(A))
res = []
for A in sorted(L.members):
    props = L._tree_splits(A) + L._context_splits(A) + L._word_class_splits(A)
    prims = [e for e in L.members[A] if m.kind[e] == 0]
    plural = {e for e in prims if m.token[e].endswith("s") and m.token[e] not in ("is", "was", "this", "these", "sings")}
    if 0 < len(plural) < len(prims):
        props.append([plural])            # number oracle (diagnostic only)
    for k, groups in enumerate(props):
        mp = {e: L.next_sym + j for j, g in enumerate(groups) for e in g}
        d0 = L.tables.delta(L._relabel_patch(mp, L.own_classes(mp))[0])
        joint = L._propagate_graph(mp, 6)
        dj = L.tables.delta(L._relabel_patch(joint, L.own_classes(joint), fresh_renamed=True)[0])
        tag = "oracle" if k == len(props) - 1 and 0 < len(plural) < len(prims) else "proposal"
        res.append((dj, d0, A, tag, len(mp), len(joint)))
for r in sorted(res)[:15]:
    print([round(x, 1) if isinstance(x, float) else x for x in r])
