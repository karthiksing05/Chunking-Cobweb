"""Round 186: parse-time readouts of the SAME learned memory (TinyStories), scored as in round 185.

Run:  story_parse_variants.py <cfg json> <seed> <checkpoints e.g. 1000,4000> <tag>
Agg:  story_parse_variants.py agg <tag>   (reads results/story_parse_variants/<tag>_s*.json, writes <tag>_table.md/.json
                                           and <tag>_examples.md)

Stream, held-out set, gold and metrics are copied from story_parse_eval.py (same fixed held-out set). Learning is the r185
recipe, unchanged. After learning to each checkpoint the held-out sentences are parsed under parse-time variants (instance
attributes set, then restored; parse() only reads the memory). Brackets = chosen composites of L.parse (whole sentence excluded).
  V0 default; V1 content-only gain; V2 context-only gain; V3 tau 0.5; V4 tau 2; V5 readout regen; V6 factor code; V7 factor climb.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "story_parse_variants")
sys.path.insert(0, os.path.join(ROOT, "src"))

VARIANTS = ["V0_default", "V1_content", "V2_context", "V3_tau0.5", "V4_tau2", "V5_regen", "V6_code", "V7_climb"]


def prf(pred, gold):
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    npred, ngold = sum(len(p) for p in pred), sum(len(g) for g in gold)
    P = tp / npred if npred else 0.0
    R = tp / ngold if ngold else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def f1_between(a, b):
    return prf(a, b)[2]


def main():
    import nltk
    from trellis2.stories import WORD, default_tinystories_path, read_sentences
    from trellis2.trellis3 import Trellis3

    cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
    checkpoints = [int(x) for x in sys.argv[3].split(",")]
    tag = sys.argv[4]
    n = checkpoints[-1]
    N_TEST_STORIES, n_test = 1500, 300

    with open(default_tinystories_path(), encoding="utf-8") as f:
        raw = f.read()
    stories = []
    for st in raw.split("<|endoftext|>"):
        st = re.sub(r"\s+", " ", st.replace("’", "'").replace("‘", "'"))
        sents = [w for w in (WORD.findall(m.group(0).lower()) for m in re.finditer(r"[^.!?]+[.!?]", st)) if w]
        if sents:
            stories.append(sents)
    all_sents = read_sentences()
    vocab = {w for w, _ in Counter(w for s in all_sents for w in s).most_common(250)}
    keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
    random.Random(0).shuffle(stories)
    test, train_st = stories[:N_TEST_STORIES], stories[N_TEST_STORIES:]
    random.Random(seed).shuffle(train_st)
    train = []
    for st in train_st:
        for i, s in enumerate(st):
            if keep(s):
                train.append((s, any(keep(t) for t in st[:i])))
        if len(train) >= n:
            break
    train = train[:n]
    held = [s for st in test for s in st if keep(s)]
    random.Random(1000).shuffle(held)
    held = held[:n_test]
    del raw, all_sents, stories, test, train_st

    CHUNKER = nltk.RegexpParser(r"""
      NP: {<DT|PRP\$|POS|CD|PDT>*<JJ.*|RBS|RBR>*<NN.*>+}
          {<PRP|EX>}
      AP: {<RB.*>*<JJ.*>+}
      VG: {<MD|TO>?<RB.*>*<VB.*>+(<RB.*>*<VB.*>+)*<RP>?}
      PP: {<IN|TO><NP>}
      VP: {<VG><NP|PP|AP|RB.*>+}
    """)

    def phrases(words):
        tagged = nltk.pos_tag(["I" if w == "i" else w for w in words])
        tree = CHUNKER.parse(tagged)
        spans = defaultdict(set)

        def walk(t, i):
            if isinstance(t, nltk.Tree):
                j = i
                for c in t:
                    j = walk(c, j)
                if t.label() in ("NP", "PP", "VP", "AP", "VG"):
                    spans[(i, j)].add(t.label())
                return j
            return i + 1
        walk(tree, 0)
        for k, (_, t) in enumerate(tagged):
            if t == "CC":
                spans[(k, k + 1)].add("CC")
        return dict(spans)

    def crosses(i, j, spans):
        if sv(i, j, spans):
            return False
        return any(a < j and i < b and not (a <= i and j <= b) and not (i <= a and b <= j) for a, b in spans)

    def sv(i, j, spans):
        lead = 1 if j - i > 2 and "CC" in spans.get((i, i + 1), ()) else 0
        return any(("NP" in spans.get((i, k), ()) or (lead and "NP" in spans.get((i + 1, k), ()))) and "VG" in spans.get((k, j), ())
                   for k in range(i + 1, j))

    GS = [phrases(s) for s in held]
    PH = ("NP", "PP", "VP", "AP", "VG")
    gold_ph = [{ij for ij, v in sp.items() if ij[1] - ij[0] >= 2 and ij[1] - ij[0] < len(s) and v & set(PH)} for s, sp in zip(held, GS)]
    gold_sv = [g | {(i, j) for i in range(len(s)) for j in range(i + 2, len(s)) if sv(i, j, sp)} for g, s, sp in zip(gold_ph, held, GS)]
    gold_np = [{ij for ij, v in sp.items() if ij[1] - ij[0] >= 2 and "NP" in v and ij[1] - ij[0] < len(s)} for s, sp in zip(held, GS)]
    occ = defaultdict(list)
    for h, s in enumerate(held):
        for i in range(len(s)):
            for j in range(i + 2, min(len(s) - 1, i + 4) + 1):
                occ[tuple(s[i:j])].append((h, i, j))
    strings = {k: v for k, v in occ.items() if len(v) >= 3}

    def metrics(brs):
        P, R, F = prf(brs, gold_ph)
        P2, R2, F2 = prf(brs, gold_sv)
        nb = sum(len(b) for b in brs)
        cr = sum(crosses(i, j, GS[h]) for h, b in enumerate(brs) for i, j in b)
        npm = prf(brs, gold_np)[1]
        tot = hit = dec = 0
        for v in strings.values():
            b = sum((i, j) in brs[h] for h, i, j in v)
            tot += len(v)
            hit += b
            dec += max(b, len(v) - b)
        return {"brackets/sent": round(nb / len(held), 3), "P": round(P, 4), "R": round(R, 4), "F1": round(F, 4),
                "P+sv": round(P2, 4), "R+sv": round(R2, 4), "F1+sv": round(F2, 4), "NP recall": round(npm, 4),
                "crossing": round(cr / nb, 4) if nb else None, "consistency": round(hit / tot, 4),
                "decisive": round(dec / tot, 4), "strings": len(strings)}

    def parse_all(L):
        L._quiet, pv = True, L._pv
        L._pv = None
        chosen = []
        for s in held:
            r = L.parse(s)
            chosen.append({(i, j) for i, k, j in r["chosen"] if 2 <= j - i < len(s)})
        L._quiet, L._pv = False, pv
        return chosen

    def run_variant(L, v):
        saved = {"tau": L.tau, "readout": L.readout, "factor": L.factor}
        try:
            if v == "V1_content":
                L._factor = lambda e: L._gain(L.cnt, e.cnt)
            elif v == "V2_context":
                L._factor = lambda e: L._gain(L.ctx, e.ctx)
            elif v == "V3_tau0.5":
                L.tau = 0.5
            elif v == "V4_tau2":
                L.tau = 2.0
            elif v == "V5_regen":
                L.readout = "regen"
            elif v == "V6_code":
                L.factor = "code"
            elif v == "V7_climb":
                L.factor = "climb"
            return parse_all(L)
        finally:
            L.__dict__.pop("_factor", None)
            L.tau, L.readout, L.factor = saved["tau"], saved["readout"], saved["factor"]

    L = Trellis3(seed=seed, **cfg)
    out = {"tag": tag, "seed": seed, "cfg": cfg, "held": held, "gold_phrase": [sorted(g) for g in gold_ph],
           "n_gold_phrase": sum(len(g) for g in gold_ph), "strings": len(strings), "checkpoints": []}
    done, t0 = 0, time.time()
    for cp in checkpoints:
        for idx in range(done, cp):
            s, fol = train[idx]
            L.learn(s, follows=fol)
        done = cp
        row = {"learned": cp, "records": len(L.records), "variants": {}}
        for v in VARIANTS:
            try:
                brs = run_variant(L, v)
            except Exception as ex:   # report plainly
                import traceback
                row["variants"][v] = {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}
                print(v, "ERROR", row["variants"][v]["error"], flush=True)
                continue
            row["variants"][v] = {"metrics": metrics(brs), "brackets": [sorted(b) for b in brs]}
            print(cp, v, json.dumps(row["variants"][v]["metrics"]), f"{time.time() - t0:.0f}s", flush=True)
        out["checkpoints"].append(row)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{tag}_s{seed}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), separators=(",", ":"))
    print("wrote", p)


def fmt(b):
    return " ".join(f"[{i},{j})" for i, j in sorted(b))


def agg(tag):
    import glob
    files = sorted(glob.glob(os.path.join(OUT, f"{tag}_s*.json")))
    runs = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        runs[d["seed"]] = d
    seeds = sorted(runs)
    assert len({json.dumps(runs[s]["held"]) for s in seeds}) == 1
    held = runs[seeds[0]]["held"]
    cps = [r["learned"] for r in runs[seeds[0]]["checkpoints"]]
    keys = ["brackets/sent", "P", "R", "F1", "F1+sv", "NP recall", "crossing", "consistency", "decisive", "xseed F1"]
    res, md = {}, [f"seeds {seeds}; held-out {len(held)}; gold phrase spans {runs[seeds[0]]['n_gold_phrase']}; strings {runs[seeds[0]]['strings']}", ""]
    best = {}
    for ci, cp in enumerate(cps):
        md += [f"### {cp} sentences learned", "", "| variant | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for v in VARIANTS:
            vs = {s: runs[s]["checkpoints"][ci]["variants"].get(v, {}) for s in seeds}
            if any("error" in x for x in vs.values()):
                md.append(f"| {v} | ERROR: " + "; ".join(f"s{s}: {x.get('error')}" for s, x in vs.items() if "error" in x) + " |" + " |" * (len(keys) - 1))
                continue
            per = {s: dict(vs[s]["metrics"]) for s in seeds}
            brs = {s: [set(map(tuple, b)) for b in vs[s]["brackets"]] for s in seeds}
            for s in seeds:
                o = [round(f1_between(brs[s], brs[t]), 4) for t in seeds if t != s]
                per[s]["xseed F1"] = round(sum(o) / len(o), 4)
            res[f"{cp}|{v}"] = per
            cells = []
            for k in keys:
                x = [per[s][k] for s in seeds]
                cells.append("n/a" if any(y is None for y in x) else f"{sum(x) / len(x):.3f} [{' / '.join(f'{y:.3f}' for y in x)}]")
            md.append(f"| {v} | " + " | ".join(cells) + " |")
        md.append("")
        cand = [(sum(res[f"{cp}|{v}"][s]["F1"] for s in seeds) / len(seeds), v) for v in VARIANTS[1:] if f"{cp}|{v}" in res]
        best[cp] = max(cand)[1] if cand else None
    open(os.path.join(OUT, f"{tag}_table.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump(res, open(os.path.join(OUT, f"{tag}_table.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(md))
    print("best non-default variant by mean F1:", best)

    # examples + inconsistent strings (seed 13 only), V0 vs the best variant at each checkpoint
    s0 = seeds[0]
    gold = [set(map(tuple, g)) for g in runs[s0]["gold_phrase"]]
    ex = [f"seed {s0}; brackets are word spans [i,j); gold = shallow phrases (size>=2)", ""]
    occ = defaultdict(list)
    for h, s in enumerate(held):
        for i in range(len(s)):
            for j in range(i + 2, min(len(s) - 1, i + 4) + 1):
                occ[tuple(s[i:j])].append((h, i, j))
    strings = {k: v for k, v in occ.items() if len(v) >= 3}
    rng = random.Random(5)
    pick = sorted(rng.sample(range(len(held)), 15))
    for ci, cp in enumerate(cps):
        for v in ["V0_default", best[cp]]:
            vv = runs[s0]["checkpoints"][ci]["variants"][v]
            brs = [set(map(tuple, b)) for b in vv["brackets"]]
            ex += [f"## {cp} learned, {v}: 15 held-out sentences", ""]
            for h in pick:
                w = held[h]
                show = lambda spans: " ".join(f"({' '.join(w[i:j])})" for i, j in sorted(spans, key=lambda t: (t[0], -t[1])))
                ex.append(f"- {' '.join(w)}\n  - trellis: {show(brs[h])}\n  - gold: {show(gold[h])}")
            ex += ["", f"## {cp} learned, {v}: 10 repeated strings with lowest consistency (>=3 occurrences)", ""]
            rows = []
            for k, occs in strings.items():
                b = sum((i, j) in brs[h] for h, i, j in occs)
                rows.append((b / len(occs), -len(occs), k, occs))
            rows.sort(key=lambda r: (r[0], r[1], r[2]))
            for c, _, k, occs in rows[:10]:
                ex.append(f"- \"{' '.join(k)}\" consistency {c:.2f} ({len(occs)} occ)")
                for h, i, j in occs[:6]:
                    w = held[h]
                    # how this occurrence was bracketed: the smallest chosen bracket covering it
                    inside = (i, j) in brs[h]
                    cover = sorted((b for b in brs[h] if b[0] <= i and j <= b[1] and b != (i, j)), key=lambda b: b[1] - b[0])
                    cross = [b for b in brs[h] if b[0] < j and i < b[1] and not (b[0] <= i and j <= b[1]) and not (i <= b[0] and b[1] <= j)]
                    ex.append(f"  - {'ONE UNIT' if inside else 'split'}: " + " ".join(w[:i]) + f" <{' '.join(w[i:j])}> " + " ".join(w[j:]) +
                              (f"   crossed by {[' '.join(w[a:b_]) for a, b_ in cross]}" if cross else ""))
            ex.append("")
    open(os.path.join(OUT, f"{tag}_examples.md"), "w", encoding="utf-8").write("\n".join(ex))


if __name__ == "__main__":
    if sys.argv[1] == "agg":
        agg(sys.argv[2])
    else:
        main()
