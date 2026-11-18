"""Round 210: edge+same vs edge alone at the same temperature, paired by seed (w7, n=1000).
Reads r205-fair-n1000.perseed.tsv (edge) and r210-fair-n1000.perseed.tsv (same), plus the dumps for the repeated-word rate
(share of distinct new sentences with a word occurring twice anywhere; and share of judged-ungrammatical ones)."""
import glob
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
import story_frontier as sf  # noqa: E402

GD = "experiments/v3/results/story_gen"
N = 1000


def tsv(path, fam):
    out = {}
    for i, ln in enumerate(open(path, encoding="utf-8")):
        if i == 0:
            ks = ln.split()
            continue
        r = dict(zip(ks, ln.rstrip("\n").split("\t")))
        if r["fam"] == fam:
            out[(float(r["temp"]), int(r["seed"]))] = {k: float(v) for k, v in r.items() if k != "fam"}
    return out


def repwords(dirs):
    out = {}
    for d in dirs:
        for f in sorted(glob.glob("%s/%s/story_s*_n%d_arm*.jsonl" % (GD, d, N))):
            if "refs" in f or f.endswith(".scores.json"):
                continue
            lines = [json.loads(l) for l in open(f, encoding="utf-8")]
            arm, rows = lines[0]["arm"], lines[1:]
            seed = int(f.split("_s")[1].split("_")[0])
            ps = json.load(open(f[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
            T = sf.Train(sf.train_sentences(seed, N))
            seen = {}
            for ln, p in zip(rows, ps):
                s = tuple(ln["gen"].split())
                if s in T.whole or s in seen:
                    continue
                seen[s] = p > 0.5
            if not seen:
                continue
            rep = {s: len(set(s)) < len(s) for s in seen}
            ung = [s for s in seen if not seen[s]]
            out[(float(arm["temp"]), seed, "same" if arm.get("repeat") else "edge")] = (
                sum(rep.values()) / len(seen), (sum(rep[s] for s in ung) / len(ung)) if ung else 0.0, len(seen))
    return out


def m(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    E = tsv("%s/r205-fair-n%d.perseed.tsv" % (GD, N), "TRELLIS edge w7")
    S = tsv("%s/r210-fair-n%d.perseed.tsv" % (GD, N), "TRELLIS edge w7")
    R = repwords(["r204-n1000-w7", "r210-n1000-w7"])
    temps = sorted({t for t, _ in S})
    seeds = [13, 17, 19, 23]
    A = lambda r: r["A"]
    mets = [("fair", lambda r: r["gramnovel"] / r["A"]), ("strict", lambda r: r["strict"] / r["A"]),
            ("dist/draw", lambda r: r["distinct"] / r["A"]), ("tri|d", lambda r: r["tri"] / r["distinct"]),
            ("gram|tri", lambda r: r["gramnovel"] / r["tri"])]
    out = ["setting | metric: edge -> same (mean diff, wins same>edge of 4 seeds; sel=13/17, held=19/23)"]
    for t in temps:
        for name, f in mets:
            d = {s: f(S[(t, s)]) - f(E[(t, s)]) for s in seeds if (t, s) in S and (t, s) in E}
            e = m([f(E[(t, s)]) for s in d]); x = m([f(S[(t, s)]) for s in d])
            out.append("t%.1f %-9s edge %.3f same %.3f diff %+.3f wins %d/%d sel %+.3f held %+.3f | %s" % (
                t, name, e, x, m(list(d.values())), sum(v > 0 for v in d.values()), len(d),
                m([d[s] for s in (13, 17) if s in d]), m([d[s] for s in (19, 23) if s in d]),
                " ".join("%d:%+.3f" % (s, d[s]) for s in sorted(d))))
        for kind in ("edge", "same"):
            v = [R[(t, s, kind)] for s in seeds if (t, s, kind) in R]
            out.append("t%.1f repeated-word rate (all distinct new / among ungrammatical) %s: %.3f / %.3f" % (t, kind, m([a[0] for a in v]), m([a[1] for a in v])))
        out.append("")
    # best same setting chosen on 13/17 by fair
    sel = {t: m([S[(t, s)]["gramnovel"] / S[(t, s)]["A"] for s in (13, 17)]) for t in temps}
    b = max(sel, key=sel.get)
    out.append("best same setting on 13/17: t%.1f (sel fair %.3f); held 19/23 same %.3f, edge same t %.3f, best edge-alone t2.0 held %.3f" % (
        b, sel[b], m([S[(b, s)]["gramnovel"] / S[(b, s)]["A"] for s in (19, 23)]), m([E[(b, s)]["gramnovel"] / E[(b, s)]["A"] for s in (19, 23)]),
        m([E[(2.0, s)]["gramnovel"] / E[(2.0, s)]["A"] for s in (19, 23)])))
    open("%s/r210-compare-n%d.txt" % (GD, N), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


main()
