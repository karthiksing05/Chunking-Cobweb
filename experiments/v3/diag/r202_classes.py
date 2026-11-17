"""Round 202: the r192 failure classes on judged story dumps, from the stream alone (no learner needed).

Over every NEW generation (not a stored training sentence; rows, repeats included, as r192) that has a trigram the
stream never had (padded <s> ... </s>), each unstored trigram (a, b, c) is a step of kind
  end   c is </s>                          (a new sentence end)
  A     (b, c) never stood side by side    (an unstored pair; the type licence let it through)
  C     otherwise                          (both pairs stored, only the trigram new)
and the sentence's class is A if it has an A step, else B if it has an end step, else C (r192_summarize.py's
"has licensed bigram" / "has novel end" / "middle trigrams only"). Also new STARTS ((<s>, w1, w2) unstored) are
counted, as the share of class-B/C sentences whose only new trigram is at the start.

Usage (repo root, compute node): r202_classes.py <out.tsv> <dump dir> [<dump dir> ...]
  family names as r202_fair.py; writes one row per (family, temp, seed) and prints means over seeds.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
import story_frontier as sf  # noqa: E402

sys.path.insert(0, "experiments/v3/diag")
from r202_fair import family_of  # noqa: E402


def classify(s, T, pairs):
    g = sf.grams(s, 3)
    new = [t for t in g if t not in T.tri]
    if not new:
        return None, new
    kinds = []
    for t in new:
        if t[2] == "</s>":
            kinds.append("end")
        elif t[1] != "<s>" and (t[1], t[2]) not in pairs:
            kinds.append("A")
        else:
            kinds.append("C")
    cls = "A" if "A" in kinds else ("B" if "end" in kinds else "C")
    return cls, new


def main(out, dirs):
    cache = {}
    rows = []
    for d in dirs:
        d = d.rstrip("/")
        for f in sorted(glob.glob(os.path.join(d, "*_arm*.jsonl"))):
            m = re.search(r"_s(\d+)_n(\d+)_arm(\d+)\.jsonl$", f)
            sc = f[:-len(".jsonl")] + ".scores.json"
            if not m or not os.path.exists(sc):
                continue
            seed, n = int(m.group(1)), int(m.group(2))
            if os.environ.get("R202_SEEDS") and str(seed) not in os.environ["R202_SEEDS"].split(","):
                continue
            if (seed, n) not in cache:
                sents = sf.train_sentences(seed, n)
                cache[(seed, n)] = (sf.Train(sents), {(a, b) for s in sents for a, b in zip(s, s[1:])})
            T, pairs = cache[(seed, n)]
            ends = {g[1] for g in T.tri if g[2] == "</s>"}
            lines = [json.loads(l) for l in open(f, encoding="utf-8")]
            arm, gl = lines[0]["arm"], lines[1:]
            ps = json.load(open(sc, encoding="utf-8"))
            fam, temp = family_of({"arm": json.dumps(arm, sort_keys=True), "dir": os.path.basename(d)})
            by = defaultdict(list)
            start_only = 0
            for r, p in zip(gl, ps):
                s = tuple(r["gen"].split())
                if s in T.whole:
                    continue
                cls, new = classify(s, T, pairs)
                by["new"].append(p > 0.5)
                by[f"len{min(len(s), 8)}"].append(p > 0.5)
                rep = len(set(s)) < len(s)
                conj = any(s[i] == "and" and i - k >= 0 and s[i - k:i] == s[i + 1:i + 1 + k]
                           for i in range(len(s)) for k in (1, 2))
                prem = s[-1] not in ends                 # ends after a word memory never saw end a sentence
                by["rep"].append(rep)
                by["conj"].append(conj)
                by["prem"].append(prem)
                if rep:
                    by["gram rep"].append(p > 0.5)
                if prem:
                    by["gram prem"].append(p > 0.5)
                if cls is None:
                    continue
                by["tri"].append(p > 0.5)
                by[cls].append(p > 0.5)
                if cls != "A" and all(t[0] == "<s>" for t in new):
                    start_only += 1
            nt = max(len(by["tri"]), 1)
            rows.append(dict(fam=fam, temp=temp, seed=seed, n=n, new=len(by["new"]), tri=len(by["tri"]),
                             **{f"share {c}": len(by[c]) / nt for c in "ABC"},
                             **{f"gram {c}": (sum(by[c]) / len(by[c]) if by[c] else None) for c in "ABC"},
                             **{"gram tri": sum(by["tri"]) / nt, "start-only share": start_only / nt},
                             **{f"gram len{k}": (sum(by[f"len{k}"]) / len(by[f"len{k}"]) if by[f"len{k}"] else None) for k in range(3, 9)},
                             **{f"share len{k}": len(by[f"len{k}"]) / max(len(by["new"]), 1) for k in range(3, 9)},
                             **{"repeat word": sum(by["rep"]) / max(len(by["new"]), 1),
                                "repeat conjunct": sum(by["conj"]) / max(len(by["new"]), 1),
                                "premature end": sum(by["prem"]) / max(len(by["new"]), 1),
                                "gram repeat": (sum(by["gram rep"]) / len(by["gram rep"]) if by["gram rep"] else None),
                                "gram premature": (sum(by["gram prem"]) / len(by["gram prem"]) if by["gram prem"] else None)}))
    ks = ["fam", "temp", "seed", "n", "new", "tri", "share A", "share B", "share C", "gram A", "gram B", "gram C",
          "gram tri", "start-only share"] + [f"gram len{k}" for k in range(3, 9)] + [f"share len{k}" for k in range(3, 9)] + \
        ["repeat word", "repeat conjunct", "premature end", "gram repeat", "gram premature"]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\t".join(ks) + "\n")
        for r in sorted(rows, key=lambda r: (r["n"], r["fam"], r["temp"], r["seed"])):
            fh.write("\t".join("-" if r[k] is None else (f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])) for k in ks) + "\n")
    g = defaultdict(list)
    for r in rows:
        g[(r["n"], r["fam"], r["temp"])].append(r)
    print("n | family | temp | seeds | sentences w/ new trigram | share A / B / C | gram A / B / C | gram|tri | start-only")
    for k in sorted(g):
        rs = g[k]

        def mu(key):
            v = [r[key] for r in rs if r[key] is not None]
            return "%.3f" % (sum(v) / len(v)) if v else "-"
        print(f"{k[0]} | {k[1]} | {k[2]} | {len(rs)} | {sum(r['tri'] for r in rs)} | "
              f"{mu('share A')} / {mu('share B')} / {mu('share C')} | {mu('gram A')} / {mu('gram B')} / {mu('gram C')} | "
              f"{mu('gram tri')} | {mu('start-only share')}")
    print("\nn | family | temp | gram by length 3 / 4 / 5 / 6 / 7 / 8 (share of new 3 / 4 / 5) | repeat word | repeat conjunct | premature end | gram repeat | gram premature")
    for k in sorted(g):
        rs = g[k]

        def mu(key):
            v = [r[key] for r in rs if r[key] is not None]
            return "%.3f" % (sum(v) / len(v)) if v else "-"
        print(f"{k[0]} | {k[1]} | {k[2]} | " + " / ".join(mu(f"gram len{i}") for i in range(3, 9)) +
              " (" + " / ".join(mu(f"share len{i}") for i in range(3, 6)) + ") | " +
              " | ".join(mu(x) for x in ("repeat word", "repeat conjunct", "premature end", "gram repeat", "gram premature")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
