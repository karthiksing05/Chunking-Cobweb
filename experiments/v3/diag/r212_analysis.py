"""Round 212: 8-seed analysis of the generation claim (seeds 13 17 19 23 from r200/r204/r205, 29 31 37 41 from r212).

Confirmatory settings (fixed in advance from r205): TRELLIS edge n250 w7 t3.5, n500 w6 t2.8, n1000 w7 t2.0; LSTM t0.7/0.8/0.8; GPT t0.7/0.8/0.8.
Paired per-seed differences TRELLIS-LSTM and TRELLIS-GPT: mean, 95% t-interval (df = seeds-1), 95% percentile bootstrap over seeds, wins.
Pre-registered: "matches LSTM" = CI of TRELLIS-LSTM includes 0 and lower bound > -.03; "beats GPT" = CI of TRELLIS-GPT above 0.
Run on a compute node from the repo root: python experiments/v3/diag/r212_analysis.py
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
sys.path.insert(0, "experiments/v3/diag")
import r204_fair as rf  # noqa: E402
import story_frontier as sf  # noqa: E402

GD = rf.GD
OLD, NEW = (13, 17, 19, 23), (29, 31, 37, 41)
SETTING = {250: ("w7 t3.50", "t0.70", "t0.70"), 500: ("w6 t2.80", "t0.80", "t0.80"), 1000: ("w7 t2.00", "t0.80", "t0.80")}
FAMS = ("TRELLIS edge", "LSTM", "GPT")
try:
    from scipy import stats
    tcrit = lambda df: float(stats.t.ppf(0.975, df))
except Exception:
    tcrit = lambda df: {3: 3.182, 7: 2.365}[df]
out = []


def P(s=""):
    out.append(s)
    print(s, flush=True)


def ci(d):
    d = np.asarray(d, float)
    k = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(k)
    t = tcrit(k - 1)
    rng = np.random.default_rng(0)
    bs = np.sort(d[rng.integers(0, k, (20000, k))].mean(1))
    return m, m - t * se, m + t * se, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))], int((d > 0).sum()), k


def fmt_ci(d):
    m, tl, th, bl, bh, w, k = ci(d)
    return "mean %+.4f, t95 [%+.4f, %+.4f], boot95 [%+.4f, %+.4f], wins %d/%d" % (m, tl, th, bl, bh, w, k)


def verdict_lstm(d):
    m, tl, th, bl, bh, w, k = ci(d)
    f = lambda lo, hi: lo <= 0 <= hi and lo > -0.03
    return "t: %s, boot: %s" % ("HOLDS" if f(tl, th) else "FAILS", "HOLDS" if f(bl, bh) else "FAILS")


def verdict_gpt(d):
    m, tl, th, bl, bh, w, k = ci(d)
    return "t: %s, boot: %s" % ("HOLDS" if tl > 0 else "FAILS", "HOLDS" if bl > 0 else "FAILS")


def collect(n):
    old_jls = [os.path.join(GD, f) for f in ("r200-frontier-n%d.jsonl" % n, "r204-frontier-n%d.jsonl" % n, "r205-frontier-n%d.jsonl" % n)]
    new_jl = os.path.join(GD, "r212-frontier-n%d.jsonl" % n)
    ro, ao = rf.per_seed(n, old_jls)
    rn, an = rf.per_seed(n, [new_jl])
    for r in rn:
        if r["seed"] in ao:        # part B rows on old seeds: attempted count from the old dumps, as published
            if an.get(r["seed"]) != ao[r["seed"]]:
                P("note n=%d seed %d: attempted old %s vs r212 dumps %s (old used)" % (n, r["seed"], ao[r["seed"]], an.get(r["seed"])))
            r["A"] = ao[r["seed"]]
    att = dict(ao)
    att.update({s: a for s, a in an.items() if s not in ao})
    val = {}
    for r in ro + rn:
        k = rf.key_of(r)
        if k:
            val.setdefault(k, {})[r["seed"]] = r
    return val, att


def v(val, k, s, metric):
    r = val[k][s]
    return r[metric] / r["A"]


def overlap():
    P("\n## Training-set overlap between seeds (share of seed a's distinct training sentences also in seed b's)")
    for n in (250, 500, 1000):
        sets = {s: set(sf.train_sentences(s, n)) for s in OLD + NEW}
        def avg(pairs):
            return np.mean([len(sets[a] & sets[b]) / len(sets[a]) for a, b in pairs])
        oo = [(a, b) for a, b in combinations(OLD, 2)] + [(b, a) for a, b in combinations(OLD, 2)]
        no = [(a, b) for a in NEW for b in OLD] + [(b, a) for a in NEW for b in OLD]
        nn = [(a, b) for a, b in combinations(NEW, 2)] + [(b, a) for a, b in combinations(NEW, 2)]
        P("n=%d  old-old %.3f  new-old %.3f  new-new %.3f" % (n, avg(oo), avg(no), avg(nn)))


def main():
    for n in (250, 500, 1000):
        val, att = collect(n)
        P("\n# n=%d  attempted per seed: %s" % (n, dict(sorted(att.items()))))
        sel = dict(zip(FAMS, SETTING[n]))
        keys = {}
        for f in FAMS:
            ks = [k for k in val if k[0] == f and k[1] == sel[f]]
            if len(ks) != 1:
                P("MISSING setting %s %s" % (f, sel[f]))
                sys.exit(1)
            keys[f] = ks[0]
            miss = [s for s in OLD + NEW if s not in val[keys[f]]]
            if miss:
                P("MISSING seeds %s for %s" % (miss, keys[f]))
        seeds = [s for s in OLD + NEW if all(s in val[keys[f]] for f in FAMS)]
        for metric, label in (("gramnovel", "FAIR count"), ("strict", "BOUNDARY-STRICT count")):
            P("\n## n=%d %s at the r205-selected settings (%s), seeds %s" % (n, label, ", ".join("%s %s" % (f, keys[f][1]) for f in FAMS), seeds))
            P("seed | " + " | ".join(FAMS))
            for s in seeds:
                P("%d | " % s + " | ".join("%.3f" % v(val, keys[f], s, metric) for f in FAMS))
            for name, ss in (("all 8", seeds), ("old 4", [s for s in seeds if s in OLD]), ("new 4", [s for s in seeds if s in NEW])):
                P("mean over %s: " % name + ", ".join("%s %.4f" % (f, np.mean([v(val, keys[f], s, metric) for s in ss])) for f in FAMS))
            for name, ss in (("all 8", seeds), ("old 4", [s for s in seeds if s in OLD]), ("new 4", [s for s in seeds if s in NEW])):
                for o in ("LSTM", "GPT"):
                    d = [v(val, keys["TRELLIS edge"], s, metric) - v(val, keys[o], s, metric) for s in ss]
                    P("TRELLIS-%s [%s]: %s" % (o, name, fmt_ci(d)))
            if metric == "gramnovel":
                dl = [v(val, keys["TRELLIS edge"], s, metric) - v(val, keys["LSTM"], s, metric) for s in seeds]
                dg = [v(val, keys["TRELLIS edge"], s, metric) - v(val, keys["GPT"], s, metric) for s in seeds]
                P("VERDICT n=%d fair: matches LSTM -> %s ; beats GPT -> %s" % (n, verdict_lstm(dl), verdict_gpt(dg)))
            else:
                dl = [v(val, keys["TRELLIS edge"], s, metric) - v(val, keys["LSTM"], s, metric) for s in seeds]
                dg = [v(val, keys["TRELLIS edge"], s, metric) - v(val, keys["GPT"], s, metric) for s in seeds]
                P("reading n=%d strict (same rules): matches LSTM -> %s ; beats GPT -> %s" % (n, verdict_lstm(dl), verdict_gpt(dg)))
        P("\n## n=%d selection re-check: fair count means per setting" % n)
        for name, ss in (("new 4 seeds", NEW), ("old 4 seeds", OLD), ("all 8 seeds", OLD + NEW)):
            P("-- %s" % name)
            for f in FAMS:
                rows = sorted(((k[1], np.mean([v(val, k, s, "gramnovel") for s in ss])) for k, d in val.items()
                               if k[0] == f and all(s in d for s in ss)), key=lambda x: x[0])
                if f != "TRELLIS edge":
                    rows = [r for r in rows if r[0] in ("t0.70", "t0.80", "t0.90")]
                else:
                    rows = [r for r in rows if r[0].startswith(SETTING[n][0][:2])]
                if rows:
                    best = max(rows, key=lambda x: x[1])
                    P("   %-13s %s   best %s" % (f, "  ".join("%s %.4f" % r for r in rows), best[0]))
    overlap()
    open(os.path.join(GD, "r212-analysis.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")


main()
