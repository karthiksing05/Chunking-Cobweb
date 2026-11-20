"""Round 224: re-score the existing r212 (n=250/500/1000, 8 seeds) and r204 (n=4000, seeds 13/17) generation dumps with judge2
(r214_judge2.py: Qwen3-8B + phi-4 ensemble; syntax logit > -5.25, semantics logit > 2.5). No regeneration.
Fair count = distinct new (not stored) sentences with an unstored trigram that the judge passes, per ATTEMPTED draw (r204_fair.per_seed logic,
copied here with the judge flag swapped). "gram" = judge2 syntax pass; "meaningful" = syntax pass and semantics pass.
Modes (repo root): prep | report.   prep (env trellis) writes r224/sentences.txt;  judge: r214_judge2.py score (env chunks, GPU);  report (env trellis).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
sys.path.insert(0, "experiments/v3/diag")
import r204_fair as rf  # noqa: E402
import story_frontier as sf  # noqa: E402

GD = rf.GD
R = os.path.join(GD, "r224")
OLD, NEW = (13, 17, 19, 23), (29, 31, 37, 41)
SEEDS = OLD + NEW
SETTING = {250: ("w7 t3.50", "t0.70", "t0.70"), 500: ("w6 t2.80", "t0.80", "t0.80"), 1000: ("w7 t2.00", "t0.80", "t0.80")}
FAMS = ("TRELLIS edge", "LSTM", "GPT")
SET4000 = {"TRELLIS edge": "w5 t2.00", "LSTM": "t1.00", "GPT": "t0.90"}   # r204-fair-n4000 best over its 2 seeds, old judge
NS = (250, 500, 1000, 4000)


def jls_for(n):
    if n == 4000:
        return [[os.path.join(GD, "r200-frontier-n4000.jsonl"), os.path.join(GD, "r204-frontier-n4000.jsonl")]]
    return [[os.path.join(GD, f) for f in ("r200-frontier-n%d.jsonl" % n, "r204-frontier-n%d.jsonl" % n, "r205-frontier-n%d.jsonl" % n)],
            [os.path.join(GD, "r212-frontier-n%d.jsonl" % n)]]


def selected(n, fam, ks):
    if n == 4000:
        return fam in FAMS   # judge every n=4000 setting of the three families (small)
    return fam in FAMS and ks == dict(zip(FAMS, SETTING[n]))[fam]


def files(n, jls):
    rows = []
    for jl in jls:
        if os.path.exists(jl):
            rows += [json.loads(l) for l in open(jl, encoding="utf-8")]
    rows = [r for r in rows if r["n"] == n]
    seenf, uniq = set(), []
    for r in rows:
        k = (r["dir"], r["file"])
        if k not in seenf and os.path.exists(os.path.join(GD, r["dir"], r["file"][:-len(".jsonl")] + ".scores.json")):
            seenf.add(k)
            uniq.append(r)
    return uniq


def norm(g):
    return " ".join(g.split())


def load_rows(r):
    f = os.path.join(GD, r["dir"], r["file"])
    lines = [json.loads(l) for l in open(f, encoding="utf-8")]
    ps = json.load(open(f[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
    return lines[1:], ps, len(lines) - 1


def want(n, r):
    k = rf.key_of(dict(fam=rf.family_of(r)[0], temp=rf.family_of(r)[1]))
    return k is not None and selected(n, k[0], k[1]), k


def prep():
    sents = set()
    for n in NS:
        for jls in jls_for(n):
            for r in files(n, jls):
                ok, k = want(n, r)
                if ok:
                    lines, ps, _ = load_rows(r)
                    sents.update(norm(l["gen"]) for l in lines)
    sents.discard("")
    done = set()
    for name in ("scores.tsv",):   # sentences already judged by an earlier pass of this round: judge only the rest
        p = os.path.join(R, name)
        if os.path.exists(p):
            done |= set(l.split("\t")[0] for l in open(p, encoding="utf-8"))
    sents -= done
    open(os.path.join(R, "sentences_b.txt" if done else "sentences.txt"), "w", encoding="utf-8").write("\n".join(sorted(sents)) + "\n")
    print("distinct sentences still to judge", len(sents), "already judged", len(done))


def load_flags():
    import csv
    flags = {}
    for name in ("scores.tsv", "scores_b.tsv"):
        p = os.path.join(R, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                flags[row["sentence"]] = (int(row["syn_pass"]), int(row["sem_pass"]), int(row["Qwen3-8B_syn_pass"]), int(row["phi-4_syn_pass"]))
    return flags


def records(n, jls_groups, flags):
    """per (family, setting, seed) counts. Returns dict key -> seed -> rec, attempted per seed, and the sentence-level table."""
    allrec, att_groups, tables = [], [], []
    for jls in jls_groups:
        uniq = files(n, jls)
        nrows = {(r["dir"], r["file"]): load_rows(r)[2] for r in uniq}
        att = defaultdict(int)
        for r in uniq:
            att[r["seed"]] = max(att[r["seed"]], nrows[(r["dir"], r["file"])])
        cache, recs = {}, []
        for r in uniq:
            ok, k = want(n, r)
            if not ok:
                continue
            seed = r["seed"]
            if seed not in cache:
                cache[seed] = sf.Train(sf.train_sentences(seed, n))
            T = cache[seed]
            lines, ps, _ = load_rows(r)
            assert len(ps) == len(lines)
            seen = {}
            for ln, p in zip(lines, ps):
                s = tuple(ln["gen"].split())
                if s in T.whole or s in seen:
                    continue
                f = flags.get(" ".join(s), (0, 0, 0, 0))
                seen[s] = (p > 0.5, bool(f[0]), bool(f[1]), any(g not in T.tri for g in sf.grams(s, 3)))
            V = list(seen.values())
            rec = dict(key=k, seed=seed, att_own=att[seed], distinct=len(V), tri=sum(v[3] for v in V),
                       old=sum(v[0] and v[3] for v in V), syn=sum(v[1] and v[3] for v in V), mean=sum(v[1] and v[2] and v[3] for v in V),
                       tri_syn=sum(v[1] for v in V if v[3]), tri_mean=sum(v[1] and v[2] for v in V if v[3]),
                       tri_sem=sum(v[2] for v in V if v[3]), tri_old=sum(v[0] for v in V if v[3]),
                       agree=sum(v[0] == v[1] for v in V), nsent=len(V), oy_ny=sum(v[0] and v[1] for v in V), oy_nn=sum(v[0] and not v[1] for v in V),
                       on_ny=sum((not v[0]) and v[1] for v in V), on_nn=sum((not v[0]) and (not v[1]) for v in V),
                       all_syn=sum(v[1] for v in V), all_old=sum(v[0] for v in V))
            recs.append(rec)
        allrec.append((recs, dict(att)))
    # merge as r212 collect: group 0 old seeds, group 1 new; old attempted counts override
    val = {}
    att = {}
    if len(allrec) == 2:
        (ro, ao), (rn, an) = allrec
        for r in rn:
            if r["seed"] in ao:
                r["att_own"] = ao[r["seed"]]
        att = dict(ao)
        att.update({s: a for s, a in an.items() if s not in ao})
        recs = ro + rn
    else:
        recs, att = allrec[0]
    for r in recs:
        val.setdefault(r["key"], {})[r["seed"]] = r
    return val, att


def ci(d):
    d = np.asarray(d, float)
    k = len(d)
    m = d.mean()
    if k < 3:
        return m, None, None, None, None, int((d > 0).sum()), k
    se = d.std(ddof=1) / np.sqrt(k)
    from scipy import stats
    t = float(stats.t.ppf(0.975, k - 1))
    rng = np.random.default_rng(0)
    bs = np.sort(d[rng.integers(0, k, (20000, k))].mean(1))
    return m, m - t * se, m + t * se, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))], int((d > 0).sum()), k


def fci(d):
    m, tl, th, bl, bh, w, k = ci(d)
    if tl is None:
        return "%+.4f | n/a | n/a | %d/%d" % (m, w, k)
    return "%+.4f | [%+.4f, %+.4f] | [%+.4f, %+.4f] | %d/%d" % (m, tl, th, bl, bh, w, k)


def kappa(a, b, c, d):   # a=yy b=yn c=ny d=nn (old, new)
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report():
    flags = load_flags()
    out = []
    P = out.append
    verdicts = {}
    for n in NS:
        val, att = records(n, jls_for(n), flags)
        sel = SET4000 if n == 4000 else dict(zip(FAMS, SETTING[n]))
        keys = {f: (f, sel[f]) for f in FAMS}
        for f in FAMS:
            assert keys[f] in val, (n, keys[f], list(val)[:5])
        seeds = [s for s in SEEDS if all(s in val[keys[f]] for f in FAMS)]
        A = lambda f, s: att[s]
        P("\n## n=%d (settings %s; seeds %s; attempted per seed %s)\n" % (n, ", ".join("%s %s" % (f, keys[f][1]) for f in FAMS), seeds, {s: att[s] for s in seeds}))
        met = {"old": "old judge (Qwen2.5-3B p>.5), as r212", "syn": "judge2 syntax pass (headline re-score)", "mean": "judge2 meaningful = syntax and semantics pass"}
        v = lambda f, s, m: val[keys[f]][s][m] / att[s]
        P("Fair count, mean over seeds:\n\n| metric | TRELLIS edge | LSTM | GPT |\n|---|---|---|---|")
        for m in ("old", "syn", "mean"):
            P("| %s | %s |" % (met[m], " | ".join("%.4f" % np.mean([v(f, s, m) for s in seeds]) for f in FAMS)))
        P("\nPer seed fair count (TRELLIS / LSTM / GPT), old -> judge2 syntax -> judge2 meaningful:\n")
        for s in seeds:
            P("- seed %d: " % s + "; ".join("%s -> %s -> %s" % tuple("%.3f/%.3f/%.3f" % tuple(v(f, s, m) for f in FAMS) for m in ("old", "syn", "mean")) for _ in (0,)))
        # per-family rates among distinct new sentences with unstored trigram
        P("\nAmong distinct new sentences with an unstored trigram (mean over seeds of the per-seed share): judged grammatical by old judge, by judge2 syntax, semantics pass (judge2), meaningful (syn and sem); distinct-per-draw and unstored-trigram share of distinct.\n")
        P("| family | old gram | judge2 gram | judge2 sem | meaningful | meaningful / judge2 gram | distinct/draw | tri/distinct |\n|---|---|---|---|---|---|---|---|")
        for f in FAMS:
            rs = [val[keys[f]][s] for s in seeds]
            mean = lambda g: np.mean([g(r) for r in rs])
            P("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (f, mean(lambda r: r["tri_old"] / r["tri"]), mean(lambda r: r["tri_syn"] / r["tri"]),
              mean(lambda r: r["tri_sem"] / r["tri"]), mean(lambda r: r["tri_mean"] / r["tri"]), mean(lambda r: r["tri_mean"] / max(r["tri_syn"], 1)),
              mean(lambda r: r["distinct"] / att[r["seed"]]), mean(lambda r: r["tri"] / r["distinct"])))
        P("\nJudge agreement on the same sentences (all distinct new generated sentences of the setting, pooled over seeds; old = Qwen2.5-3B p>.5, new = judge2 syntax pass):\n")
        P("| family | sentences | agree | old yes/new yes | old yes/new no | old no/new yes | old no/new no | kappa | old pass rate | new pass rate |\n|---|---|---|---|---|---|---|---|---|---|")
        tot = [0, 0, 0, 0]
        for f in FAMS:
            rs = [val[keys[f]][s] for s in seeds]
            a, b, c, d = (sum(r[k] for r in rs) for k in ("oy_ny", "oy_nn", "on_ny", "on_nn"))
            tot = [x + y for x, y in zip(tot, (a, b, c, d))]
            N = a + b + c + d
            P("| %s | %d | %.3f | %d | %d | %d | %d | %.3f | %.3f | %.3f |" % (f, N, (a + d) / N, a, b, c, d, kappa(a, b, c, d), (a + b) / N, (a + c) / N))
        a, b, c, d = tot
        N = sum(tot)
        P("| all | %d | %.3f | %d | %d | %d | %d | %.3f | %.3f | %.3f |" % (N, (a + d) / N, a, b, c, d, kappa(a, b, c, d), (a + b) / N, (a + c) / N))
        for m, mname in (("old", "OLD judge (r212 reproduction)"), ("syn", "JUDGE2 syntax pass"), ("mean", "JUDGE2 meaningful")):
            P("\nPaired fair-count differences, %s, %d seeds:\n\n| contrast | mean | t 95%% CI | bootstrap 95%% CI | wins |\n|---|---|---|---|---|" % (mname, len(seeds)))
            for o in ("LSTM", "GPT"):
                d = [v("TRELLIS edge", s, m) - v(o, s, m) for s in seeds]
                P("| TRELLIS-%s | %s |" % (o, fci(d)))
                verdicts[(n, m, o)] = ci(d)
        if n == 4000:
            P("\nn=4000 per-setting fair count (mean of seeds 13/17), old judge vs judge2 syntax vs judge2 meaningful (selection re-check under judge2):\n")
            P("| family | setting | old | judge2 syn | judge2 meaningful |\n|---|---|---|---|---|")
            for k in sorted(val):
                if all(s in val[k] for s in (13, 17)):
                    P("| %s | %s | %.4f | %.4f | %.4f |" % (k[0], k[1], *[np.mean([val[k][s][m] / att[s] for s in (13, 17)]) for m in ("old", "syn", "mean")]))
    # verdicts
    P("\n## Headline claims under judge2 (syntax pass as grammatical; same pre-registered rules as r212)\n")
    P("Rules: matches LSTM = CI of TRELLIS-LSTM includes 0 and lower bound > -.03; beats GPT = CI of TRELLIS-GPT above 0. Both t and bootstrap shown.\n")
    P("| n | judge | matches LSTM (t / boot) | beats GPT (t / boot) |\n|---|---|---|---|")
    for n in (250, 500, 1000):
        for m, lab in (("old", "old"), ("syn", "judge2 syntax"), ("mean", "judge2 meaningful")):
            _, tl, th, bl, bh, _, _ = verdicts[(n, m, "LSTM")]
            f = lambda lo, hi: "HOLDS" if (lo <= 0 <= hi and lo > -0.03) else "FAILS"
            _, gl, gh, gbl, gbh, _, _ = verdicts[(n, m, "GPT")]
            g = lambda lo: "HOLDS" if lo > 0 else "FAILS"
            P("| %d | %s | %s / %s | %s / %s |" % (n, lab, f(tl, th), f(bl, bh), g(gl), g(gbl)))
    open(os.path.join(R, "report_body.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"prep": prep, "report": report}[sys.argv[1]]()
