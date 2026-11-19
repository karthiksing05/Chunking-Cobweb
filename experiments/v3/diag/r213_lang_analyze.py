"""Round 213 (language) analysis of r213_lang.py dumps (4 seeds, n=1000, checkpoints 250/500/1000).

Usage (compute node, repo root):
  r213_lang_analyze.py stats            -> results/story_gen/r213/stats.json, stats.txt, candidates.txt
  r213_lang_analyze.py cases <idx,...> <bio string|string|...>   -> cases.txt   (held-out sentence indices; chunk strings for biographies)
"""
import json
import math
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations

D = "experiments/v3/results/story_gen/r213/"
SEEDS = [13, 17, 19, 23]
CPS = [250, 500, 1000]
R = {s: json.load(open(f"{D}lang_s{s}.json", encoding="utf-8")) for s in SEEDS}
held = [tuple(x) for x in R[13]["held"]]
H = len(held)
out = []


def P(*a):
    out.append(" ".join(str(x) for x in a))
    print(out[-1], flush=True)


def counts(s, cp):
    return Counter({k: sum(1 for x in v if x < cp) for k, v in R[s]["pos"].items() if any(x < cp for x in v)})


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


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rank(xs), rank(ys)
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def ngram_counts(s, cp, lo=2, hi=8):
    c = Counter()
    for sent in R[s]["train"][:cp]:
        n = len(sent)
        for i in range(n):
            for j in range(i + lo, min(n, i + hi) + 1):
                if j - i < n:
                    c[" ".join(sent[i:j])] += 1
    return c


def length_matched(freq, ref_top):
    """Top n-grams by raw count with the same length histogram as ref_top (a list of strings)."""
    need = Counter(len(w.split()) for w in ref_top)
    got, res = Counter(), []
    for w, _ in sorted(freq.items(), key=lambda t: (-t[1], t[0])):
        l = len(w.split())
        if got[l] < need[l]:
            res.append(w)
            got[l] += 1
        if len(res) == len(ref_top):
            break
    return res


def fam(s, cp, k, kind):
    """Top-k chunk list of one family for seed s at checkpoint cp."""
    if kind == "TRELLIS":
        return topk(counts(s, cp), k)
    if kind == "freq":
        return topk(ngram_counts(s, cp), k)
    if kind == "freq-lenmatched":
        return length_matched(ngram_counts(s, cp), topk(counts(s, cp), k))
    if kind == "BPE":
        return R[s]["bpe"][str(cp)][:k]
    raise ValueError(kind)


KINDS = ["TRELLIS", "freq", "freq-lenmatched", "BPE"]
# ---------------------------------------------------------------- 1. inventories
P("# r213 language stats (seeds %s, n=1000, held-out %d sentences)" % (SEEDS, H))
P("\n## stream info")
for s in SEEDS:
    P(s, json.dumps(R[s]["info"]))
P("\n## top chunks by count at 1000 (per-seed top 25; count = learn-tree bracket occurrences)")
for s in SEEDS:
    c = counts(s, 1000)
    P("seed", s, "distinct bracketed strings", len(c), "total bracket occurrences", sum(c.values()))
    P("  ", "; ".join("%s (%d)" % (w, c[w]) for w in topk(c, 25)))
tot = Counter()
for s in SEEDS:
    tot.update(counts(s, 1000))
P("\npooled over 4 seeds, top 40:", "; ".join("%s (%d)" % (w, v) for w, v in tot.most_common(40)))
lenh = Counter(len(w.split()) for w, _ in tot.most_common(200))
P("length histogram of pooled top-200:", dict(sorted(lenh.items())))
# strings found by every seed
sets = {s: set(topk(counts(s, 1000), 200)) for s in SEEDS}
every = set.intersection(*sets.values())
P("\nchunks in the per-seed top-200 of ALL 4 seeds (%d):" % len(every))
P("  ", "; ".join("%s (%s)" % (w, "/".join(str(counts(s, 1000)[w]) for s in SEEDS)) for w in sorted(every, key=lambda w: -sum(counts(s, 1000)[w] for s in SEEDS))))
sets50 = {s: set(topk(counts(s, 1000), 50)) for s in SEEDS}
every50 = set.intersection(*sets50.values())
P("in the per-seed top-50 of all 4 seeds (%d):" % len(every50), "; ".join(sorted(every50)))
anyseed = Counter(w for s in SEEDS for w in counts(s, 1000) if counts(s, 1000)[w] >= 3)
P("strings with count >= 3 in k of 4 seeds:", dict(sorted(Counter(anyseed.values()).items())))
# ---------------------------------------------------------------- 2. cross-seed overlap with baselines
P("\n## cross-seed overlap of chunk inventories (mean over the 6 seed pairs)")
rng = random.Random(213)
res = {}
for cp in CPS:
    for k in (50, 200):
        row = {}
        for kind in KINDS:
            lists = {s: fam(s, cp, k, kind) for s in SEEDS}
            row[kind] = mean(jac(lists[a], lists[b]) for a, b in combinations(SEEDS, 2))
        # random: draw k strings uniformly from the pool of n-grams (2..8 words) that occur >= 2 times in the seed's stream prefix
        pool = {s: [w for w, v in ngram_counts(s, cp).items() if v >= 2] for s in SEEDS}
        allpool = sorted(set(w for s in SEEDS for w in pool[s]))
        rj = []
        for _ in range(100):
            ls = {s: rng.sample(pool[s], min(k, len(pool[s]))) for s in SEEDS}
            rj.append(mean(jac(ls[a], ls[b]) for a, b in combinations(SEEDS, 2)))
        row["random (pool: n-grams count>=2)"] = mean(rj)
        res[(cp, k)] = row
        P("cp %d top-%d Jaccard: " % (cp, k) + ", ".join("%s %.3f" % (a, b) for a, b in row.items()) + "  [pool sizes %s]" % [len(pool[s]) for s in SEEDS])
P("\ncount-weighted overlap (sum min / sum max of raw counts over the union of chunk strings), mean over pairs")
for cp in CPS:
    cs = {s: counts(s, cp) for s in SEEDS}
    ns = {s: ngram_counts(s, cp) for s in SEEDS}
    P("cp %d: TRELLIS bracketed strings %.3f; all n-grams (raw count, same stream) %.3f" % (
        cp, mean(wjac(cs[a], cs[b]) for a, b in combinations(SEEDS, 2)), mean(wjac(ns[a], ns[b]) for a, b in combinations(SEEDS, 2))))
    # restricted to strings that occur >= 3 times in the pooled training streams, to compare like with like
    P("   TRELLIS top-200 count-weighted %.3f; freq-lenmatched top-200 %.3f; BPE top-200 (merge order, unit weight = jaccard) %.3f" % (
        mean(wjac({w: cs[a][w] for w in fam(a, cp, 200, "TRELLIS")}, {w: cs[b][w] for w in fam(b, cp, 200, "TRELLIS")}) for a, b in combinations(SEEDS, 2)),
        mean(wjac({w: ns[a][w] for w in fam(a, cp, 200, "freq-lenmatched")}, {w: ns[b][w] for w in fam(b, cp, 200, "freq-lenmatched")}) for a, b in combinations(SEEDS, 2)),
        mean(jac(fam(a, cp, 200, "BPE"), fam(b, cp, 200, "BPE")) for a, b in combinations(SEEDS, 2))))
P("\nTRELLIS top-50 vs frequency baselines within a seed (Jaccard, mean over seeds), cp 1000")
for kind in ("freq", "freq-lenmatched", "BPE"):
    P("  TRELLIS top-50 vs %s top-50: %.3f; top-200 vs top-200: %.3f" % (kind, mean(jac(fam(s, 1000, 50, "TRELLIS"), fam(s, 1000, 50, kind)) for s in SEEDS),
                                                                 mean(jac(fam(s, 1000, 200, "TRELLIS"), fam(s, 1000, 200, kind)) for s in SEEDS)))
P("TRELLIS length histogram of top-200 at 1000, seed 13:", dict(sorted(Counter(len(w.split()) for w in fam(13, 1000, 200, "TRELLIS")).items())),
  " freq top-200:", dict(sorted(Counter(len(w.split()) for w in fam(13, 1000, 200, "freq")).items())))
# ---------------------------------------------------------------- 3. stability along the stream
P("\n## stability along the stream (within seed; mean over seeds)")
for k in (50, 200):
    for kind in KINDS:
        a = mean(jac(fam(s, 250, k, kind), fam(s, 1000, k, kind)) for s in SEEDS)
        b = mean(jac(fam(s, 500, k, kind), fam(s, 1000, k, kind)) for s in SEEDS)
        P("top-%d Jaccard of checkpoint vs final (1000): 250 %.3f, 500 %.3f  [%s]" % (k, a, b, kind))
for kind in ("TRELLIS", "freq"):
    rs = []
    for s in SEEDS:
        fin = counts(s, 1000) if kind == "TRELLIS" else ngram_counts(s, 1000)
        t50 = topk(fin, 50)
        c500 = counts(s, 500) if kind == "TRELLIS" else ngram_counts(s, 500)
        c250 = counts(s, 250) if kind == "TRELLIS" else ngram_counts(s, 250)
        rs.append((spearman([c250.get(w, 0) for w in t50], [fin[w] for w in t50]), spearman([c500.get(w, 0) for w in t50], [fin[w] for w in t50])))
    P("Spearman of count at 250 / 500 vs count at 1000 over the final top-50 (%s): %.3f / %.3f" % (kind, mean(r[0] for r in rs), mean(r[1] for r in rs)))
P("final top-50 already a bracket (count>=1) at 250: TRELLIS %.2f; at 500 %.2f" % (
    mean(sum(1 for w in fam(s, 1000, 50, "TRELLIS") if counts(s, 250)[w] > 0) / 50 for s in SEEDS),
    mean(sum(1 for w in fam(s, 1000, 50, "TRELLIS") if counts(s, 500)[w] > 0) / 50 for s in SEEDS)))
# ---------------------------------------------------------------- 4. parse consistency, shapes
n_of = [len(s) for s in held]


def inner(br, n):
    return {(i, j) for i, j in br if j - i >= 2 and (i, j) != (0, n)}


RB = [{(i, n) for i in range(1, n - 1)} for n in n_of]
LB = [{(0, j) for j in range(2, n)} for n in n_of]
parses = {s: {cp: [inner(p["brackets"], n) for p, n in zip(R[s]["parses"][str(cp)], n_of)] for cp in CPS} for s in SEEDS}
gold = [{(a, b) for a, b, _ in g} for g in R[13]["gold"]]
assert all(R[s]["gold"] == R[13]["gold"] for s in SEEDS)


def prf(brs, ref):
    tp = sum(len(a & b) for a, b in zip(brs, ref))
    p = tp / max(sum(len(a) for a in brs), 1)
    r = tp / max(sum(len(b) for b in ref), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def f1s(a, b):
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / max(len(a) + len(b), 1)


strings = defaultdict(list)
for h, s in enumerate(held):
    for i in range(len(s)):
        for j in range(i + 2, min(len(s) - 1, i + 4) + 1):
            strings[s[i:j]].append((h, i, j))
strings = {k: v for k, v in strings.items() if len(v) >= 3}


def cons_dec(brs):
    tot = hit = dec = 0
    for v in strings.values():
        b = sum((i, j) in brs[h] for h, i, j in v)
        tot += len(v)
        hit += b
        dec += max(b, len(v) - b)
    return hit / tot, dec / tot


P("\n## parse consistency on the shared held-out set (%d sentences, %d repeated 2-4 word strings, eval-script definition)" % (H, len(strings)))
P("gold (NLTK shallow phrases): consistency %.3f decisive %.3f" % cons_dec(gold))
P("right-branching: consistency %.3f decisive %.3f; left-branching: consistency %.3f decisive %.3f" % (cons_dec(RB) + cons_dec(LB)))
for cp in CPS:
    cd = [cons_dec(parses[s][cp]) for s in SEEDS]
    xs = mean(mean(f1s(a, b) for a, b in zip(parses[x][cp], parses[y][cp])) for x, y in combinations(SEEDS, 2))
    stab = mean(mean(f1s(a, b) for a, b in zip(parses[s][cp], parses[s][CPS[CPS.index(cp) - 1]])) for s in SEEDS) if cp > 250 else float("nan")
    f = [prf(parses[s][cp], gold)[2] for s in SEEDS]
    P("cp %d: consistency %.3f decisive %.3f | cross-seed bracket F1 %.3f | stability vs previous checkpoint %.3f | F1 vs NLTK %.3f (right-branching %.3f, left-branching %.3f) | brackets/sentence %.2f"
      % (cp, mean(c for c, _ in cd), mean(d for _, d in cd), xs, stab, mean(f), prf(RB, gold)[2], prf(LB, gold)[2],
         mean(mean(len(b) for b in parses[s][cp]) for s in SEEDS)))
P("per-seed cp 1000 cross-seed pairs:", {"%d-%d" % (x, y): round(mean(f1s(a, b) for a, b in zip(parses[x][1000], parses[y][1000])), 3) for x, y in combinations(SEEDS, 2)})
P("random-binary cross-seed reference (r203 table, seeds 13/17/19): .343 at 1000; BPE x1 .682; TRELLIS default chosen .742, parts .756 (story_parse_eval, 300 other held-out sentences)")
# bracketed-somewhere consistency (t3_parse_agree.py definition)


def somewhere(brs):
    occ = defaultdict(list)
    for s, br in zip(held, brs):
        n = len(s)
        for k in (2, 3, 4):
            for i in range(n - k + 1):
                if (i, i + k) != (0, n):
                    occ[tuple(s[i:i + k])].append((i, i + k) in br)
    rep = [v for v in occ.values() if len(v) >= 3 and any(v)]
    tt = sum(len(v) for v in rep)
    return sum(sum(v) for v in rep) / max(tt, 1), len(rep)


P("bracketed-somewhere consistency (t3_parse_agree definition) at 1000: " + ", ".join("%d: %.3f (%d strings)" % ((s,) + somewhere(parses[s][1000])) for s in SEEDS)
  + "; right-branching %.3f (%d); left-branching %.3f (%d)" % (somewhere(RB) + somewhere(LB)))
# ---------------------------------------------------------------- 5. non-right-branching share
P("\n## share of TRELLIS parses (binary chosen tree, whole sentence excluded) that differ from right- / left-branching")
nonrb_sets = {}
for cp in CPS:
    rows = []
    for s in SEEDS:
        ps = parses[s][cp]
        isrb = [p == r for p, r in zip(ps, RB)]
        islb = [p == l for p, l in zip(ps, LB)]
        neither = [not a and not b for a, b in zip(isrb, islb)]
        nonrb = [not a for a in isrb]
        rows.append((mean(nonrb), mean(islb), mean(neither), sum(nonrb)))
        if cp == 1000:
            nonrb_sets[s] = [h for h, v in enumerate(neither) if v]
        idx = [h for h, v in enumerate(nonrb) if v]
        fT = prf([ps[h] for h in idx], [gold[h] for h in idx])
        fR = prf([RB[h] for h in idx], [gold[h] for h in idx])
        fL = prf([LB[h] for h in idx], [gold[h] for h in idx])
        iN = [h for h, v in enumerate(neither) if v]
        fTn = prf([ps[h] for h in iN], [gold[h] for h in iN])
        fRn = prf([RB[h] for h in iN], [gold[h] for h in iN])
        fLn = prf([LB[h] for h in iN], [gold[h] for h in iN])
        P("cp %d seed %d: differs from RB %.3f (%d), equals LB %.3f, neither RB nor LB %.3f (%d) | on the %d non-RB parses F1 vs NLTK: TRELLIS %.3f, RB %.3f, LB %.3f | on the %d neither-parses: TRELLIS %.3f, RB %.3f, LB %.3f | on RB-equal parses RB F1 %.3f"
          % (cp, s, rows[-1][0], rows[-1][3], rows[-1][1], rows[-1][2], len(iN), len(idx), fT[2], fR[2], fL[2], len(iN), fTn[2], fRn[2], fLn[2],
             prf([RB[h] for h in range(H) if isrb[h]], [gold[h] for h in range(H) if isrb[h]])[2]))
    P("cp %d mean over seeds: differs from RB %.3f, equals LB %.3f, neither %.3f" % (cp, mean(r[0] for r in rows), mean(r[1] for r in rows), mean(r[2] for r in rows)))
P("overall F1 vs NLTK at 1000 by seed:", {s: round(prf(parses[s][1000], gold)[2], 3) for s in SEEDS})
# share of sentences non-RB by every seed
allnon = [h for h in range(H) if all(parses[s][1000][h] != RB[h] for s in SEEDS)]
P("sentences non-right-branching in all 4 seeds at 1000: %d of %d" % (len(allnon), H))
# ---------------------------------------------------------------- candidates for the showcase
rng2 = random.Random(213)
P("\n## candidates (seed 13, cp 1000): neither RB nor LB; index | F1 vs NLTK | nonend brackets | tree | gold")


def render(n_words, spans, words):
    sp = sorted({(i, j) for i, j in spans}, key=lambda t: (t[0], -t[1]))
    kept = []
    for a in sp:
        if all(not (a[0] < b[1] and b[0] < a[1]) or (a[0] >= b[0] and a[1] <= b[1]) or (b[0] >= a[0] and b[1] <= a[1]) for b in kept):
            kept.append(a)

    def rec(i, j):
        parts, k = [], i
        inside = [x for x in kept if i <= x[0] and x[1] <= j and x != (i, j)]
        tops = [x for x in inside if not any(y != x and y[0] <= x[0] and x[1] <= y[1] and y != (i, j) for y in inside)]
        for a in sorted(tops):
            while k < a[0]:
                parts.append(words[k])
                k += 1
            parts.append(rec(*a))
            k = a[1]
        while k < j:
            parts.append(words[k])
            k += 1
        return "[" + " ".join(parts) + "]" if (i, j) in kept else " ".join(parts)
    return rec(0, n_words)


s0 = 13
cand = []
for h in nonrb_sets[s0]:
    ps = parses[s0][1000][h]
    nonend = [b for b in ps if b[1] != n_of[h]]
    f = f1s(ps, gold[h])
    cand.append((h, f, len(nonend)))
cl = []
for h, f, ne in cand:
    cl.append("%3d | %.2f | %d | %s | %s" % (h, f, ne, R[s0]["parses"]["1000"][h]["tree"], render(n_of[h], gold[h], held[h])))
open(D + "candidates.txt", "w", encoding="utf-8").write("\n".join(cl) + "\n")
P("wrote candidates.txt with %d trees" % len(cl))
rand4 = rng2.sample(nonrb_sets[s0], 4)
P("random draw (random.Random(213).sample of the seed-13 cp-1000 neither-RB-nor-LB set, k=4):", rand4)
json.dump({"neither13": nonrb_sets[13], "random4": rand4}, open(D + "stats.json", "w"))
open(D + "stats.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")

# ---------------------------------------------------------------- 6. case studies
if len(sys.argv) > 1 and sys.argv[1] == "cases":
    pass
