"""Round 213 (chess) analysis of r213_chess.py dumps (cs73L memory, seeds, checkpoints along the stream).

Usage (compute node, repo root): r213_chess_analyze.py [seeds comma list] -> results/chess_inspect/r213_chess_stats.txt
"""
import json
import math
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations

D = "experiments/v3/results/chess_inspect/"
SEEDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "13,17,19,23").split(",")]
R = {s: json.load(open(f"{D}r213_chess_s{s}.json", encoding="utf-8")) for s in SEEDS}
CPS = [c for c in R[SEEDS[0]]["checkpoints"] if all(str(c) in R[s]["boards"] for s in SEEDS)]
FIN = str(CPS[-1])
out = []


def P(*a):
    out.append(" ".join(str(x) for x in a))
    print(out[-1], flush=True)


def parse(key):
    res = []
    for it in key.split("|"):
        sq, tok = it.split(":")
        res.append((ord(sq[0]) - 97, int(sq[1]) - 1, tok))
    return res


def shape(key):
    ps = parse(key)
    f0, r0 = min(p[0] for p in ps), min(p[1] for p in ps)
    return "|".join(sorted("%d,%d:%s" % (f - f0, r - r0, t) for f, r, t in ps))


def counts(s, cp):
    return Counter({k: sum(1 for x in v if x < cp) for k, v in R[s]["chunks"].items() if any(x < cp for x in v)})


def shape_counts(s, cp):
    c = Counter()
    for k, v in counts(s, cp).items():
        c[shape(k)] += v
    return c


def topk(c, k):
    return [w for w, _ in sorted(c.items(), key=lambda t: (-t[1], t[0]))[:k]]


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def wjac(a, b):
    ks = set(a) | set(b)
    return sum(min(a.get(k, 0), b.get(k, 0)) for k in ks) / max(sum(max(a.get(k, 0), b.get(k, 0)) for k in ks), 1)


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


def classify(key):
    ps = parse(key)
    kinds = Counter(t for _, _, t in ps)
    pawns = [(f, r, t) for f, r, t in ps if t[1] == "P"]
    for color, kr in (("w", 0), ("b", 7)):
        kings = [(f, r) for f, r, t in ps if t == color + "K"]
        own = [(f, r) for f, r, t in ps if t == color + "P"]
        if kings:
            kf, krk = kings[0]
            shield = [p for p in own if abs(p[0] - kf) <= 1 and 0 < (p[1] - krk) * (1 if color == "w" else -1) <= 2]
            if krk == kr and kf in (5, 6, 7) and len(shield) >= 2:
                return "castled king (kingside) + pawn shield"
            if krk == kr and kf in (0, 1, 2) and len(shield) >= 2:
                return "castled king (queenside) + pawn shield"
            if krk == kr and kf in (5, 6, 7, 0, 1, 2) and len(shield) == 1:
                return "king in corner + one pawn"
    for color, bsq, ps_ in (("w", [(6, 1), (1, 1)], [(6, 2), (1, 2)]), ("b", [(6, 6), (1, 6)], [(6, 5), (1, 5)])):
        for b, pw in zip(bsq, ps_):
            if (b[0], b[1], color + "B") in [(f, r, t) for f, r, t in ps] and (pw[0], pw[1], color + "P") in [(f, r, t) for f, r, t in ps]:
                return "fianchetto (bishop behind pawn on the long diagonal)"
    if len(pawns) == len(ps):
        for color in "wb":
            own = [(f, r) for f, r, t in pawns if t[0] == color]
            if len(own) == len(ps) and len(own) >= 2:
                chain = any(abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1 for a, b in combinations(own, 2))
                duo = any(abs(a[0] - b[0]) == 1 and a[1] == b[1] for a, b in combinations(own, 2))
                if chain:
                    return "pawn chain (diagonal pawns)"
                if duo:
                    return "pawn duo / phalanx (side by side)"
                return "pawns of one colour (other)"
        return "pawns of both colours"
    for color, rk, cf in (("w", 0, (0, 7)), ("b", 7, (0, 7))):
        rooks = [(f, r) for f, r, t in ps if t == color + "R" and r == rk and f in cf]
        if rooks and any(t[1] == "P" for _, _, t in ps):
            return "corner rook with its pawns"
    if all((r in (0, 7)) for f, r, t in ps if t[1] != "P") and sum(1 for _, _, t in ps if t[1] != "P") >= 2:
        return "back-rank pieces"
    return "mixed piece group"


size_of = lambda k: k.count("|") + 1
P("# r213 chess chunk stats (cs73L memory, GraphTrellis, max chunk size 4, seeds %s, stream = train_long shuffled by seed, checkpoints %s)" % (SEEDS, CPS))
for s in SEEDS:
    P("seed", s, json.dumps(R[s]["info"]))
# ---------------------------------------------------------------- top chunks
P("\n## top chunks by use count at %s positions (chosen as a chunk in a learned parse; per seed top 15, with chess class)" % FIN)
for s in SEEDS:
    c = counts(s, int(FIN))
    sz = Counter(size_of(k) for k in c)
    P("seed %d: distinct chunks used >= 2 times: %d (sizes %s); total uses %d" % (s, len(c), dict(sorted(sz.items())), sum(c.values())))
    for k in topk(c, 15):
        P("   %-48s uses %-4d size %d  %s" % (k, c[k], size_of(k), classify(k)))
P("\n## class shares: of top-50 / top-200 chunks (per-seed lists, pooled over seeds) and of all uses, at final")
for kk in (50, 200, None):
    cl = Counter()
    tot = 0
    for s in SEEDS:
        c = counts(s, int(FIN))
        ks = topk(c, kk) if kk else list(c)
        for k in ks:
            cl[classify(k)] += c[k] if kk is None else 1
            tot += c[k] if kk is None else 1
    P("top-%s%s:" % (kk, "" if kk else " (all, weighted by uses)"), "; ".join("%s %.2f" % (a, b / tot) for a, b in cl.most_common()))
sz_hist = {kk: Counter(size_of(k) for s in SEEDS for k in topk(counts(s, int(FIN)), kk)) for kk in (50, 200)}
P("size histogram of per-seed top-50 / top-200 (pooled over seeds):", {k: dict(sorted(v.items())) for k, v in sz_hist.items()})
# ---------------------------------------------------------------- overlap
P("\n## cross-seed overlap of chunk inventories (mean over the %d seed pairs)" % len(list(combinations(SEEDS, 2))))
rng = random.Random(213)
pairbase = {}
for cp in CPS:
    for k in (50, 200):
        row = {}
        cs = {s: counts(s, cp) for s in SEEDS}
        sh = {s: shape_counts(s, cp) for s in SEEDS}
        row["exact (squares)"] = mean(jac(topk(cs[a], k), topk(cs[b], k)) for a, b in combinations(SEEDS, 2))
        row["translation-invariant (shape)"] = mean(jac(topk(sh[a], k), topk(sh[b], k)) for a, b in combinations(SEEDS, 2))
        # random: k chunks drawn uniformly from the chunks used >= 2 times by that seed; shape version from its shapes
        rj, rs = [], []
        pool = {s: sorted(cs[s]) for s in SEEDS}
        spool = {s: sorted(sh[s]) for s in SEEDS}
        for _ in range(60):
            l1 = {s: rng.sample(pool[s], min(k, len(pool[s]))) for s in SEEDS}
            l2 = {s: rng.sample(spool[s], min(k, len(spool[s]))) for s in SEEDS}
            rj.append(mean(jac(l1[a], l1[b]) for a, b in combinations(SEEDS, 2)))
            rs.append(mean(jac(l2[a], l2[b]) for a, b in combinations(SEEDS, 2)))
        row["random from each seed's own chunks (exact)"] = mean(rj)
        row["random (shape)"] = mean(rs)
        # all-chunk random pool: union over seeds
        union = sorted(set(x for s in SEEDS for x in cs[s]))
        ru = []
        for _ in range(60):
            l = {s: rng.sample(union, min(k, len(union))) for s in SEEDS}
            ru.append(mean(jac(l[a], l[b]) for a, b in combinations(SEEDS, 2)))
        row["random from the union pool of all seeds' chunks (exact)"] = mean(ru)
        # frequency baseline: pairs within distance 2, by raw count (size-2 chunks only comparable)
        pb = {s: [p for p, _ in R[s]["pairs"][str(cp)][:k]] for s in SEEDS}
        row["frequency baseline: piece pairs within 2 squares, top by raw count (exact)"] = mean(jac(pb[a], pb[b]) for a, b in combinations(SEEDS, 2))
        pbs = {s: topk(Counter(shape(p) for p, _ in R[s]["pairs"][str(cp)]), k) for s in SEEDS}
        row["frequency baseline (shape)"] = mean(jac(pbs[a], pbs[b]) for a, b in combinations(SEEDS, 2))
        c2 = {s: Counter({kx: v for kx, v in cs[s].items() if size_of(kx) == 2}) for s in SEEDS}
        row["TRELLIS size-2 chunks only (exact)"] = mean(jac(topk(c2[a], k), topk(c2[b], k)) for a, b in combinations(SEEDS, 2))
        P("cp %s top-%d Jaccard: " % (cp, k) + "; ".join("%s %.3f" % (a, b) for a, b in row.items()))
    cs = {s: counts(s, cp) for s in SEEDS}
    sh = {s: shape_counts(s, cp) for s in SEEDS}
    P("cp %s count-weighted overlap (sum min / sum max of use counts): exact %.3f, shape %.3f" % (
        cp, mean(wjac(cs[a], cs[b]) for a, b in combinations(SEEDS, 2)), mean(wjac(sh[a], sh[b]) for a, b in combinations(SEEDS, 2))))
# chunks every seed finds
cfin = {s: counts(s, int(FIN)) for s in SEEDS}
for kk in (50, 200):
    ev = set.intersection(*[set(topk(cfin[s], kk)) for s in SEEDS])
    P("\nchunks in every seed's top-%d (exact squares): %d" % (kk, len(ev)))
    for k in sorted(ev, key=lambda k: -sum(cfin[s][k] for s in SEEDS))[:25]:
        P("   %-48s uses %s  %s" % (k, "/".join(str(cfin[s][k]) for s in SEEDS), classify(k)))
    shs = {s: shape_counts(s, int(FIN)) for s in SEEDS}
    evs = set.intersection(*[set(topk(shs[s], kk)) for s in SEEDS])
    P("shapes in every seed's top-%d (translation-invariant): %d" % (kk, len(evs)))
allseeds = Counter(k for s in SEEDS for k in cfin[s])
P("\nchunks used >= 2 times by k of the seeds:", dict(sorted(Counter(allseeds.values()).items())))
# ---------------------------------------------------------------- stability
P("\n## stability along the stream (within seed, mean over seeds): top-k Jaccard of checkpoint vs final")
for kk in (50, 200):
    for cp in CPS[:-1]:
        P("top-%d exact: cp %s vs %s: %.3f | shape: %.3f | random reference (exact): %.3f" % (
            kk, cp, FIN, mean(jac(topk(counts(s, cp), kk), topk(cfin[s], kk)) for s in SEEDS),
            mean(jac(topk(shape_counts(s, cp), kk), topk(shape_counts(s, int(FIN)), kk)) for s in SEEDS),
            mean(len(set(topk(counts(s, cp), kk)) & set(topk(cfin[s], kk))) / max(len(set(topk(counts(s, cp), kk)) | set(topk(cfin[s], kk))), 1) for s in SEEDS) * 0
            + mean(jac(rng.sample(sorted(cfin[s]), min(kk, len(cfin[s]))), rng.sample(sorted(cfin[s]), min(kk, len(cfin[s])))) for s in SEEDS)))
P("share of the final top-50 chunks already chosen at least once at each checkpoint: " + "; ".join(
    "%s: %.2f" % (cp, mean(sum(1 for k in topk(cfin[s], 50) if counts(s, cp)[k] > 0) / 50 for s in SEEDS)) for cp in CPS[:-1]))
P("pair-frequency baseline top-50 checkpoint vs final (exact): " + "; ".join(
    "%s: %.3f" % (cp, mean(jac([p for p, _ in R[s]["pairs"][str(cp)][:50]], [p for p, _ in R[s]["pairs"][FIN][:50]]) for s in SEEDS)) for cp in CPS[:-1]))
# ---------------------------------------------------------------- recurrence of chunks on the example boards
P("\n## the example boards: units of seed %d at %s positions (size >= 2), with class, uses in the stream, recurrence and whether every seed finds it" % (SEEDS[0], FIN))
shfin = {s: shape_counts(s, int(FIN)) for s in SEEDS}
place = {}
for s in SEEDS:
    pl = defaultdict(Counter)
    for k, v in counts(s, int(FIN)).items():
        ps = parse(k)
        pl[shape(k)][(min(p[0] for p in ps), min(p[1] for p in ps))] += v
    place[s] = pl
s0 = SEEDS[0]
SQ = lambda f, r: "abcdefgh"[f] + str(r + 1)
for bi, b in enumerate(R[s0]["boards"][FIN]):
    units = [u for u in b["units"] if u["size"] >= 2]
    cov = sum(u["size"] for u in units)
    P("\nboard %d (%s, %d pieces): %d multi-piece units covering %d pieces; %d single pieces" % (bi, b["kind"], len(b["fen"]), len(units), cov, sum(1 for u in b["units"] if u["size"] == 1)))
    for u in sorted(units, key=lambda u: -u["size"]):
        key = "|".join("%s:%s" % (sq, t) for sq, t in sorted(u["pieces"]))
        found = [cfin[s].get(key, 0) for s in SEEDS]
        sh = shape(key)
        P("   unit %-46s class: %s | stored %s | uses per seed %s | shape uses %s, placements %d (e.g. %s) | first chosen at stream position %s (seed %d) | seeds finding it (>=2 uses): %d/%d"
          % (key, classify(key), u["stored"], "/".join(map(str, found)), "/".join(str(shfin[s].get(sh, 0)) for s in SEEDS),
             len(place[s0].get(sh, {})), ", ".join("%s x%d" % (SQ(*p), n) for p, n in place[s0].get(sh, Counter()).most_common(4)),
             (min(R[s0]["chunks"][key]) if key in R[s0]["chunks"] else "-"), s0, sum(1 for x in found if x >= 2), len(SEEDS)))
open(D + "r213_chess_stats.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
