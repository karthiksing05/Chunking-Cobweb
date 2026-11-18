"""Round 207 mechanism-2 probe (no generation): called from t3_story.py with STORY_R207_M2=<dump dir>, after the learner is built.

(1) pivot_support(kind="left") AUC against the judge on the single-junction class C sentences of the dump dir's seed
    (distinct, not stored, exactly one novel interior trigram (a, b, c) over two stored pairs). The pivot is b, its left
    context l1..lW the words before b, as t3_story.py's `both` branch builds them; score = log P(c | left, p=b) / P(c | p=b).
    AUC = P(score of a grammatical sentence > score of an ungrammatical one). Overall, pronoun pivots (you / it / i), per arm.
(2) tree statistics of the context tree: leaf occupancy (word-filler occurrences per leaf), leaf depth, and per concept
    (internal node) the filler word types and their effective number, by depth and by size.
"""
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter

from trellis2.trellis3 import BOS, FILL

sys.path.insert(0, "experiments/v3")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rs, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            if allv[k][1]:
                rs += (i + j + 1) / 2.0
        i = j
    n1, n0 = len(pos), len(neg)
    return (rs - n1 * (n1 + 1) / 2) / (n1 * n0)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))] if v else float("nan")


def tree_stats(L, tag):
    by_leaf = {}
    for e in L.records:
        if e.leaf is None:
            continue
        for which, attr, value, w in e.pas:
            if which == "ctx" and attr == FILL and value[0] == "w":
                d = by_leaf.setdefault(e.leaf.id, Counter())
                d[value[1]] += w
    root = L.ctx.root
    order, stack = [], [(root, 0)]
    while stack:
        nd, d = stack.pop()
        order.append((nd, d))
        stack.extend((c, d + 1) for c in nd.children)
    types = {}
    for nd, d in reversed(order):                      # post-order: a node's filler counts = its children's
        if not nd.children:
            types[nd.id] = by_leaf.get(nd.id, Counter())
        else:
            c = Counter()
            for ch in nd.children:
                c.update(types[ch.id])
            if nd.id in by_leaf:
                c.update(by_leaf[nd.id])
            types[nd.id] = c
    leaves = [(nd, d) for nd, d in order if not nd.children]
    occ = [sum(types[nd.id].values()) for nd, d in leaves]
    occ_nz = [x for x in occ if x > 0]
    print(f"[{tag}] nodes {len(order)}, leaves {len(leaves)} (with word fillers {len(occ_nz)}), internal {len(order) - len(leaves)}, "
          f"root children {len(root.children)}, max depth {max(d for _, d in order)}", flush=True)
    print(f"[{tag}] leaf occupancy (word fillers): mean {sum(occ_nz) / max(len(occ_nz), 1):.2f} median {q(occ_nz, .5):.1f} "
          f"p90 {q(occ_nz, .9):.1f} p99 {q(occ_nz, .99):.1f} max {max(occ_nz) if occ_nz else 0:.0f}; share of leaves with 1 filler "
          f"{sum(1 for x in occ_nz if x <= 1.0001) / max(len(occ_nz), 1):.3f}", flush=True)
    ld = [d for (nd, d), o in zip(leaves, occ) if o > 0]
    lw = sum(o * d for (nd, d), o in zip(leaves, occ) if o > 0) / max(sum(occ_nz), 1)
    print(f"[{tag}] leaf depth: mean {sum(ld) / max(len(ld), 1):.2f} filler-weighted {lw:.2f} median {q(ld, .5)} p90 {q(ld, .9)} max {max(ld) if ld else 0}", flush=True)
    print(f"[{tag}] concepts (internal nodes) by depth: n, median size (fillers), median distinct types, median effective types (exp entropy), "
          f"median top-type share, share with >=5 types", flush=True)
    rows = {}
    for nd, d in order:
        if not nd.children:
            continue
        c = types[nd.id]
        tot = sum(c.values())
        if tot <= 0:
            continue
        H = -sum(x / tot * math.log(x / tot) for x in c.values() if x > 0)
        b = d if d <= 5 else 6
        rows.setdefault(b, []).append((tot, len(c), math.exp(H), max(c.values()) / tot))
    for b in sorted(rows):
        r = rows[b]
        print(f"[{tag}]   depth {b if b < 6 else '6+'}: n {len(r)}, size {statistics.median(x[0] for x in r):.0f}, types {statistics.median(x[1] for x in r):.0f}, "
              f"eff {statistics.median(x[2] for x in r):.1f}, top share {statistics.median(x[3] for x in r):.2f}, >=5 types {sum(1 for x in r if x[1] >= 5) / len(r):.2f}", flush=True)
    print(f"[{tag}] concepts by size (fillers): n, median distinct types, median effective types, median top-type share", flush=True)
    rows = {}
    for nd, d in order:
        if not nd.children:
            continue
        c = types[nd.id]
        tot = sum(c.values())
        if tot <= 0:
            continue
        H = -sum(x / tot * math.log(x / tot) for x in c.values() if x > 0)
        b = 0 if tot < 4 else 1 if tot < 16 else 2 if tot < 64 else 3 if tot < 256 else 4
        rows.setdefault(b, []).append((tot, len(c), math.exp(H), max(c.values()) / tot))
    names = ["1-3", "4-15", "16-63", "64-255", "256+"]
    for b in sorted(rows):
        r = rows[b]
        print(f"[{tag}]   size {names[b]}: n {len(r)}, types {statistics.median(x[1] for x in r):.0f}, eff {statistics.median(x[2] for x in r):.1f}, "
              f"top share {statistics.median(x[3] for x in r):.2f}", flush=True)
    top = sorted(root.children, key=lambda c: -c.count)[:8]
    for c in top:
        cc = types[c.id]
        print(f"[{tag}]   top concept id {c.id}: count {c.count:.0f}, fillers {sum(cc.values()):.0f}, types {len(cc)}, top words {cc.most_common(6)}", flush=True)


def run(L, train, seed, n, dump_dir, out_path):
    import story_frontier as sf
    L._quiet = True
    W = max(getattr(L, "window", 2), 2)
    tr = [tuple(s) for s, _ in train]
    tri = Counter(g for s in tr for g in sf.grams(s, 3))
    bi = Counter(g for s in tr for g in sf.grams(s, 2))
    whole = set(tr)
    print(f"[m2] seed {seed} n {n} window {W} stored trigram types {len(tri)} pairs {len(bi)}", flush=True)
    tree_stats(L, f"tree n={n}")
    res = {}
    rows = []
    for f in sorted(glob.glob(os.path.join(dump_dir, f"story_s{seed}_n{n}_arm*.jsonl"))):
        if f.endswith(".scores.jsonl"):
            continue
        sc = f[:-6] + ".scores.json"
        if not os.path.exists(sc):
            continue
        lines = [json.loads(l) for l in open(f, encoding="utf-8")]
        arm = lines[0]["arm"]
        t = arm.get("temp")
        ps = json.load(open(sc, encoding="utf-8"))
        seen = set()
        for ln, p in zip(lines[1:], ps):
            s = tuple(ln["gen"].split())
            if not s or s in whole or s in seen:
                continue
            seen.add(s)
            new3 = [x for x in sf.grams(s, 3) if x not in tri]
            if len(new3) != 1:
                continue
            a, b, c = new3[0]
            if "<s>" in (a, b, c) or "</s>" in (a, b, c) or (a, b) not in bi or (b, c) not in bi:
                continue
            j = next(i for i, x in enumerate(sf.grams(s, 3)) if x == new3[0])
            tt = [BOS] * max(5, W) + list(s)
            i = max(5, W) + j + 1                       # index of c
            left = {f"l{k}": (tt[i - 1 - k] if i - 1 - k >= 0 else BOS) for k in range(1, W + 1)}
            sup = L.pivot_support(left, [b], [c], kind="left")[(b, c)]
            sup_p = L.pivot_support(left, [b], [c], kind="pivot")[(b, c)]
            rows.append((t, p > .5, b, c, sup, sup_p, " ".join(s)))
    print(f"[m2] single-junction class C sentences scored: {len(rows)}", flush=True)
    def rep(name, sel):
        for kind, ix in (("left", 4), ("pivot", 5)):
            g = [r[ix] for r in sel if r[1]]
            u = [r[ix] for r in sel if not r[1]]
            print(f"[m2] {name:14s} {kind:5s} n {len(sel):5d} gram {len(g):5d} ungram {len(u):5d} gram-rate {len(g) / max(len(sel), 1):.3f} "
                  f"AUC(gram>ungram) {auc(g, u):.3f} mean gram {sum(g) / max(len(g), 1):+.3f} mean ungram {sum(u) / max(len(u), 1):+.3f}", flush=True)
    rep("all temps", rows)
    for t in sorted({r[0] for r in rows}):
        rep(f"t{t}", [r for r in rows if r[0] == t])
    PR = ("you", "it", "i")
    rep("pronoun pivots", [r for r in rows if r[2] in PR])
    rep("other pivots", [r for r in rows if r[2] not in PR])
    for pv in ("and", "was", "you", "it", "i", "the", "to", "is"):
        rep(f"pivot {pv}", [r for r in rows if r[2] == pv])
    with open(out_path, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write("\t".join(map(str, r)) + "\n")
