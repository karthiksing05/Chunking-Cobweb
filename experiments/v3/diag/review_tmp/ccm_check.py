"""Review diag: CCM init variants, model-vs-search error, held/train overlap."""
import math, sys, os, json
from collections import Counter
sys.path.insert(0, "src"); sys.path.insert(0, "experiments/v3/diag")
import r233_ccm as CC, r233_goldparse as GP, r230_grammar as G

def split_post(n):
    P = [[0.0]*(n+1) for _ in range(n+1)]; P[0][n] = 1.0
    for L in range(n, 1, -1):
        for i in range(n-L+1):
            j = i+L; p = P[i][j]
            if p == 0: continue
            for k in range(i+1, j):
                P[i][k] += p/(L-1); P[k][j] += p/(L-1)
    return P

def init_stats(m, mode, raw):
    cy, dy, cc, dc = [0.0]*m.ny, [0.0]*m.ny, [0.0]*m.nc, [0.0]*m.nc
    for (n, spans), s in zip(m.d.sents, raw):
        if mode == "split": P = split_post(n)
        else:
            g = GP.gold(s) | {(0, n)} | {(i, i+1) for i in range(n)}
            P = [[1.0 if (i, j) in g else 0.0 for j in range(n+1)] for i in range(n+1)]
        for i, j, y, c in spans:
            p = P[i][j]; cy[y] += p; dy[y] += 1-p; cc[c] += p; dc[c] += 1-p
    return cy, dy, cc, dc

def fit(m, mode, raw):
    if mode == "uniform":
        st, _ = m.epoch(True)
    else:
        st = init_stats(m, mode, raw)
    m.mstep(*st)
    for it in range(CC.ITERS+1):
        st, ll = m.epoch(False)
        if it < CC.ITERS: m.mstep(*st)
    return ll + m.objective_prior()

def gold_constrained_viterbi(m, s, g):
    n, spans = m.d.encode(s)
    lr = {(i, j): m.lcy[y]-m.ldy[y]+m.lcc[c]-m.ldc[c] for i, j, y, c in spans}
    V = {}
    for i in range(n): V[(i, i+1)] = lr[(i, i+1)]
    for L in range(2, n+1):
        for i in range(n-L+1):
            j = i+L
            if any(GP.cross((i, j), x) for x in g): V[(i, j)] = -1e18; continue
            V[(i, j)] = lr[(i, j)] + max(V[(i, k)]+V[(k, j)] for k in range(i+1, j))
    return V[(0, n)]

for seed in (13, 17):
    C = G.load_corpus(seed, 4000)
    stream = [s for s, _ in C["train"]]; held = C["held"][:500]; golds = [GP.gold(s) for s in held]
    tr = set(tuple(s) for s in stream)
    ov = [tuple(s) in tr for s in held]
    print(json.dumps({"seed": seed, "held_in_train_4000": round(sum(ov)/len(ov), 3),
                      "held_in_train_1000": round(sum(tuple(s) in set(map(tuple, stream[:1000])) for s in held)/500, 3),
                      "held_len": dict(sorted(Counter(len(s) for s in held).items())),
                      "overlap_by_len": {L: round(sum(o for o, s in zip(ov, held) if len(s) == L)/max(1, sum(len(s) == L for s in held)), 2) for L in range(3, 8)},
                      "distinct_tag_patterns_held": len({tuple(G.TAG[w] for w in s) for s in held})}), flush=True)
    for tags in (False, True):
        tr_s = [CC.tagify(s) for s in stream] if tags else stream
        he_s = [CC.tagify(s) for s in held] if tags else held
        for n in (1000, 4000):
            for mode in ("uniform", "split", "gold"):
                m = CC.CCM(CC.Data(tr_s[:n]))
                obj = fit(m, mode, stream[:n])
                trees = [m.viterbi(s) for s in he_s]
                sc = GP.score(trees, golds)
                # model error: is best gold-consistent tree scored below the viterbi tree?
                me = sum(gold_constrained_viterbi(m, s, g) < m.viterbi(s, True) - 1e-9 for s, g in zip(he_s, golds))
                top = Counter()
                for s, t in zip(held, trees):
                    for (i, j) in t: top[" ".join(G.TAG[w] for w in s[i:j])] += 1
                print(json.dumps({"seed": seed, "tags": tags, "n": n, "init": mode, "map_obj": round(obj, 1), **sc,
                                  "share_gold_consistent_tree_scored_lower": round(me/500, 3), "top_spans": top.most_common(5)}), flush=True)
