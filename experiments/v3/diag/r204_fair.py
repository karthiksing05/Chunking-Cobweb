"""Round 204: fair productivity with the edge_exact family, plus boundary diagnostics and the boundary-strict count.

Same rules as story_frontier_fair.py (attempted = max rows over the dumps of that (seed, n); distinct new = distinct generated
strings that are not a stored training sentence; gram = judge P(yes) > .5; fair = distinct new, grammatical, with an unstored
trigram, per ATTEMPTED draw). Families: plain TRELLIS (r200 + earlier), "TRELLIS edge" (arm key "edge": "exact"), LSTM, GPT.
Extra per (family, temp, seed):
  strict   = fair count restricted to boundary-typical sentences (first word starts >= 2 and last word ends >= 2 of the 1200 real
             held-out sentences of the n=1000 reference dirs, seeds 13/17/19/23); applied identically to every family
  atyp     = share of distinct new sentences that are boundary-atypical
  gram_nov = judged-grammatical share among distinct new sentences with an unstored trigram (gram|tri)
  rep      = repeated rows (rows - distinct new, over new rows) and, of those, the share that is a boundary-cut copy: a string
             that is a proper contiguous span of a stored training sentence (headless or truncated) -> "repcut"
Selection (as story_fair_select.py): per setting (family, window, temp) mean over seeds 13/17; chosen setting reported on 19/23;
paired per seed against r200's LSTM / GPT best (same seeds). n=4000: seeds 13 and 17 reported separately.

Usage (repo root, compute node): r204_fair.py <n> <out prefix> <frontier.jsonl> [<frontier.jsonl> ...]
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
import story_frontier as sf  # noqa: E402
import story_frontier_plot_n as spn  # noqa: E402

GD = "experiments/v3/results/story_gen"
SEL, HELD = (13, 17), (19, 23)


def family_of(r):
    arm = json.loads(r["arm"])
    if arm.get("edge") == "exact":
        m = re.search(r"-w(\d+)$", r["dir"])
        return "TRELLIS edge w%s" % m.group(1), arm.get("temp", 1.0)
    return spn.family(r)


def lexicon():
    st, en = Counter(), Counter()
    for f in sorted(glob.glob(os.path.join(GD, "r200-n1000-refs", "story_s*_n1000_refs.jsonl"))):
        for ln in open(f, encoding="utf-8"):
            s = json.loads(ln)["real"].split()
            if s:
                st[s[0]] += 1
                en[s[-1]] += 1
    return st, en


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def f3(x):
    return "-" if x is None else "%.3f" % x


def per_seed(n, jls):
    rows = []
    for jl in jls:
        rows += [json.loads(l) for l in open(jl, encoding="utf-8")]
    rows = [r for r in rows if r["n"] == n]
    seenf, uniq = set(), []
    for r in rows:
        k = (r["dir"], r["file"])
        if k not in seenf and os.path.exists(os.path.join(GD, r["dir"], r["file"][:-len(".jsonl")] + ".scores.json")):
            seenf.add(k)
            uniq.append(r)
    nrows = {(r["dir"], r["file"]): sum(1 for _ in open(os.path.join(GD, r["dir"], r["file"]), encoding="utf-8")) - 1 for r in uniq}
    att = defaultdict(int)
    for r in uniq:
        att[r["seed"]] = max(att[r["seed"]], nrows[(r["dir"], r["file"])])
    st, en = lexicon()
    typ = lambda s: st[s[0]] >= 2 and en[s[-1]] >= 2
    cache, recs = {}, []
    for r in uniq:
        seed = r["seed"]
        if seed not in cache:
            cache[seed] = sf.Train(sf.train_sentences(seed, n))
        T = cache[seed]
        f = os.path.join(GD, r["dir"], r["file"])
        lines = [json.loads(l) for l in open(f, encoding="utf-8")][1:]
        ps = json.load(open(f[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
        assert len(ps) == len(lines)
        seen, newrows, cutrep = {}, 0, 0
        for ln, p in zip(lines, ps):
            s = tuple(ln["gen"].split())
            if s in T.whole:
                continue
            newrows += 1
            if s in seen:
                cutrep += s in T.sub
                continue
            seen[s] = (p > 0.5, any(g not in T.tri for g in sf.grams(s, 3)), typ(s), s in T.sub)
        A = att[seed]
        d = len(seen)
        V = list(seen.values())
        gt = sum(v[0] and v[1] for v in V)
        gts = sum(v[0] and v[1] and v[2] for v in V)
        tri = sum(v[1] for v in V)
        fam, temp = family_of(r)
        recs.append(dict(fam=fam, temp=temp, seed=seed, A=A, returned=len(lines), distinct=d, gram=sum(v[0] for v in V), tri=tri,
                         gramnovel=gt, strict=gts, atyp=(sum(not v[2] for v in V) / d) if d else None,
                         gram_tri=(gt / tri) if tri else None, tri_d=(tri / d) if d else None,
                         dist_rate=d / A, repeats=newrows - d, repcut=cutrep,
                         subspan=sum(v[3] for v in V)))
    return recs, dict(att)


def key_of(r):
    fam = r["fam"]
    if re.fullmatch(r"TRELLIS w\d+", fam):
        return ("TRELLIS", "%s t%.2f" % (fam[len("TRELLIS "):], r["temp"]))
    if re.fullmatch(r"TRELLIS edge w\d+", fam):
        return ("TRELLIS edge", "%s t%.2f" % (fam[len("TRELLIS edge "):], r["temp"]))
    if fam in ("LSTM", "GPT"):
        return (fam, "t%.2f" % r["temp"])
    return None


def main(n, prefix, jls):
    recs, att = per_seed(n, jls)
    if os.environ.get("R204_SEEDS"):                 # probe: only these seeds (selection = report)
        keep = {int(s) for s in os.environ["R204_SEEDS"].split(",")}
        recs = [r for r in recs if r["seed"] in keep]
    with open(prefix + ".perseed.tsv", "w", encoding="utf-8") as fh:
        ks = ["fam", "temp", "seed", "A", "returned", "distinct", "gram", "tri", "gramnovel", "strict", "repeats", "repcut", "subspan"]
        fh.write("\t".join(ks) + "\n")
        for r in sorted(recs, key=lambda r: (r["fam"], r["temp"], r["seed"])):
            fh.write("\t".join(str(r[k]) for k in ks) + "\n")
    val = defaultdict(dict)
    for r in recs:
        k = key_of(r)
        if k:
            val[k][r["seed"]] = r
    seeds = sorted({s for d in val.values() for s in d})
    full = {k: d for k, d in val.items() if all(s in d for s in seeds)}
    sel = [s for s in SEL if s in seeds]
    held = [s for s in HELD if s in seeds]
    out = ["n=%d; attempted per seed: %s; seeds %s" % (n, att, seeds)]
    out.append("\nper setting (mean over seeds present in every setting): fair, strict, dist/draw, tri|d, gram|tri, atypical share of distinct, repeats, repeats that are proper spans of stored sentences")
    out.append("family setting | fair strict | dist/draw tri|d gram|tri | atyp | repeats(rows) cutcopy(rows)")
    for k in sorted(full, key=lambda k: (k[0] != "TRELLIS edge", k[0] != "TRELLIS", k[0], k[1])):
        d = full[k]
        rs = [d[s] for s in seeds]
        out.append("%-13s %-9s | %.3f %.3f | %.3f %.3f %.3f | %.3f | %.1f %.1f" % (
            k[0], k[1], mean([r["gramnovel"] / r["A"] for r in rs]), mean([r["strict"] / r["A"] for r in rs]),
            mean([r["dist_rate"] for r in rs]), mean([r["tri_d"] for r in rs]), mean([r["gram_tri"] for r in rs]),
            mean([r["atyp"] for r in rs]), mean([r["repeats"] for r in rs]), mean([r["repcut"] for r in rs])))
    for metric, label in (("gramnovel", "fair count (distinct gram-novel per attempted draw)"), ("strict", "boundary-strict fair count")):
        out.append("\n=== %s ===" % label)
        v = lambda k, s: full[k][s][metric] / full[k][s]["A"]
        fams = ["TRELLIS", "TRELLIS edge", "LSTM", "GPT"]
        bestall, bestsel = {}, {}
        for f in fams:
            ks = [k for k in full if k[0] == f]
            if not ks:
                continue
            b = max(ks, key=lambda k: mean([v(k, s) for s in seeds]))
            bestall[f] = b
            xs = [v(b, s) for s in seeds]
            out.append("  best over all seeds  %-13s %.3f [%.3f, %.3f] at %s" % (f, mean(xs), min(xs), max(xs), b[1]))
            if sel:
                bs = max(ks, key=lambda k: mean([v(k, s) for s in sel]))
                bestsel[f] = bs
                out.append("  chosen on %s %-13s %-9s sel %.3f held %s  per-seed %s" % (
                    sel, f, bs[1], mean([v(bs, s) for s in sel]), f3(mean([v(bs, s) for s in held])) if held else "-",
                    ", ".join("%d:%.3f" % (s, v(bs, s)) for s in seeds)))
        for title, best, use in (("paired, best over all seeds", bestall, seeds), ("paired, chosen on %s, compared on %s" % (sel, held or seeds), bestsel, held or seeds)):
            out.append("  -- " + title)
            for t in ("TRELLIS edge", "TRELLIS"):
                if t not in best:
                    continue
                for o in ("LSTM", "GPT"):
                    if o not in best:
                        continue
                    dd = [(s, v(best[t], s), v(best[o], s)) for s in use]
                    out.append("    %s %s vs %s %s: wins %d of %d; mean diff %+.3f; %s" % (
                        t, best[t][1], o, best[o][1], sum(a > b for _, a, b in dd), len(dd), mean([a - b for _, a, b in dd]),
                        ", ".join("%d:%.3f/%.3f" % z for z in dd)))
    open(prefix + ".txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3:])
