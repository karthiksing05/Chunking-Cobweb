"""Round 198: fair productivity frontier. Same story dumps as story_frontier.py, but every rate is per ATTEMPTED draw
(a missing/failed draw counts as a failure) and repeats count once (distinct sentences).

Rules, identical for every family:
  attempted(seed, n) = max number of generation rows over all dumps of that (seed, n) (= prompts x draws; 1600 or 1544).
  distinct new       = distinct generated strings that are not a stored training sentence.
  distinct gram      = those with judge P(yes) > 0.5.
  distinct gram-novel= distinct new, judged grammatical, with an unstored trigram.
  rates per attempted draw = count / attempted.  tri / gram / gram|tri are shares among distinct new sentences.
Dump directories and files are read from a frontier jsonl (story_frontier.py output); the training sentences and the
trigram / novelty test are story_frontier.Train / train_sentences / grams (imported, unchanged).

Usage (repo root, compute node): story_frontier_fair.py <frontier.jsonl> <n> <out prefix>
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
import story_frontier as sf  # noqa: E402
from story_frontier_plot_n import family, interp, LEVELS  # noqa: E402

GD = "experiments/v3/results/story_gen"


def mean(v):
    return sum(v) / len(v) if v else None


def main(jl, n, prefix):
    rows = [json.loads(l) for l in open(jl, encoding="utf-8")]
    rows = [r for r in rows if r["n"] == n]
    nrows = {}
    for r in rows:
        nrows[(r["dir"], r["file"])] = sum(1 for _ in open(os.path.join(GD, r["dir"], r["file"]), encoding="utf-8")) - 1
    att = defaultdict(int)
    for r in rows:
        att[r["seed"]] = max(att[r["seed"]], nrows[(r["dir"], r["file"])])
    cache = {}
    recs = []
    for r in rows:
        seed = r["seed"]
        if seed not in cache:
            cache[seed] = sf.Train(sf.train_sentences(seed, n))
        T = cache[seed]
        f = os.path.join(GD, r["dir"], r["file"])
        lines = [json.loads(l) for l in open(f, encoding="utf-8")][1:]
        ps = json.load(open(f[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
        assert len(ps) == len(lines)
        seen = {}
        for ln, p in zip(lines, ps):
            s = tuple(ln["gen"].split())
            if s in T.whole or s in seen:
                continue
            seen[s] = (p > 0.5, any(g not in T.tri for g in sf.grams(s, 3)))
        A = att[seed]
        d = len(seen)
        g = sum(v[0] for v in seen.values())
        tr = sum(v[1] for v in seen.values())
        gt = sum(v[0] and v[1] for v in seen.values())
        fam, temp = family(r)
        recs.append(dict(fam=fam, temp=temp, seed=seed, A=A, returned=len(lines), distinct=d, gram=g, tri=tr, gramnovel=gt,
                         dist_rate=d / A, gram_rate=g / A, gn_rate=gt / A,
                         tri_d=tr / d if d else None, gram_d=g / d if d else None,
                         gram_tri=(gt / tr) if tr else None))
    with open(prefix + ".perseed.tsv", "w", encoding="utf-8") as fh:
        ks = ["fam", "temp", "seed", "A", "returned", "distinct", "gram", "tri", "gramnovel"]
        fh.write("\t".join(ks) + "\n")
        for r in sorted(recs, key=lambda r: (r["fam"], r["temp"], r["seed"])):
            fh.write("\t".join(str(r[k]) for k in ks) + "\n")
    g = defaultdict(list)
    for r in recs:
        g[(r["fam"], r["temp"])].append(r)
    cols = [("attempted", "A"), ("returned", "returned"), ("distinct new", "distinct"),
            ("distinct new/draw", "dist_rate"), ("distinct gram/draw", "gram_rate"),
            ("distinct gram-novel/draw", "gn_rate"), ("tri|distinct", "tri_d"), ("gram|distinct", "gram_d"),
            ("gram|tri,distinct", "gram_tri")]
    fams = sorted({k[0] for k in g}, key=lambda f: (not f.startswith("TRELLIS"), f))
    st = {}
    with open(prefix + ".tsv", "w", encoding="utf-8") as fh:
        fh.write("family\ttemp\tseeds\t" + "\t".join(c for c, _ in cols) + "\n")
        for k in sorted(g, key=lambda k: (fams.index(k[0]), k[1])):
            vals = {c: mean([r[key] for r in g[k] if r[key] is not None]) for c, key in cols}
            st[k] = vals
            fh.write("%s\t%.1f\t%d\t" % (k[0], k[1], len(g[k])) + "\t".join(
                "-" if vals[c] is None else "%.3f" % vals[c] for c, _ in cols) + "\n")
    out = ["n=%d; attempted per seed: %s" % (n, ", ".join("%d:%d" % kv for kv in sorted(att.items()))),
           "", "max distinct gram-novel per attempted draw, per family (mean over seeds at the best temperature; per-seed range)"]
    for f in fams:
        ks = [k for k in st if k[0] == f]
        b = max(ks, key=lambda k: st[k]["distinct gram-novel/draw"])
        v = [r["gn_rate"] for r in g[b]]
        out.append("  %-22s %.3f  [%.3f, %.3f]  at temp %.1f (seeds %d)" % (f, st[b]["distinct gram-novel/draw"], min(v), max(v), b[1], len(v)))
    for title, xk, yk in [("distinct gram per attempted draw vs tri among distinct new", "tri|distinct", "distinct gram/draw"),
                          ("gram among distinct new vs tri among distinct new", "tri|distinct", "gram|distinct")]:
        out += ["", "upper envelope: " + title + " (linear interpolation along temperature, no extrapolation)",
                "level | " + " | ".join(fams)]
        curves = {f: [(st[k][xk], st[k][yk]) for k in st if k[0] == f and st[k][xk] is not None] for f in fams}
        for x in LEVELS:
            cells = []
            for f in fams:
                y = interp(curves[f], x)
                cells.append("-" if y is None else "%.3f" % y)
            out.append("%.1f | %s" % (x, " | ".join(cells)))
    open(prefix + ".txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
