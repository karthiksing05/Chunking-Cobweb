"""Round 214 Phase 0 level map (no generation). Reads the occurrence dump of r214_story.py (env R214_LEVELS) and, per read band b
(0, 1, 2, 4, 6, 8, 12 bits, the leaf = "inf", and the basic level), reports over the committed word occurrences:

  slot purity        weighted share of the majority universal POS (spaCy, tagged in the sentence) of the fillers under each C_b
  filler entropy     occurrence-weighted mean of H(filler | C_b), bits
  selection retained I_b(filler; l2) / I_leaf(filler; l2), where I_b is the mutual information of the class-model joint
                     P_b(f, v) = sum_C P(C) P(f | C) P(v | C) (what a read at level b can use: filler and the word two back are
                     independent given the concept) and I_leaf the same at the leaf. I_b / I_raw (the empirical I(f; l2)) is also
                     printed. b = 0 gives 0, the leaf gives 1.

The design's pass band: purity >= .8, entropy >= 3 bits and selection retained <= .5 (Kill A when no band meets it).

Usage (env cobint, CPU): r214_levels.py <levels.json> <out.md> [<out.json>]
"""
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np
import spacy
from spacy.tokens import Doc


def tag(sentences):
    nlp = spacy.load("en_core_web_sm")
    out = []
    for d in nlp.pipe((Doc(nlp.vocab, words=s) for s in sentences), batch_size=64):
        out.append([t.pos_ for t in d])
    return out


def mi(J):
    J = J / J.sum()
    pf, pv = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    nz = J > 0
    return float((J[nz] * np.log2(J[nz] / (pf @ pv)[nz])).sum())


def main(path, out_md, out_json=None):
    D = json.load(open(path, encoding="utf-8"))
    sents, occ, grid = D["sentences"], D["occ"], D["grid"]
    pos = tag(sents)
    toks = sorted({o["tok"] for o in occ})
    l2s = sorted({o["l2"] for o in occ}, key=str)
    ti, vi = {t: i for i, t in enumerate(toks)}, {v: i for i, v in enumerate(l2s)}
    for o in occ:
        o["pos"] = pos[o["sid"]][o["pos"]] if o["sid"] >= 0 else "?"
    W = sum(o["w"] for o in occ)
    raw = np.zeros((len(toks), len(l2s)))
    for o in occ:
        raw[ti[o["tok"]], vi[o["l2"]]] += o["w"]
    I_raw = mi(raw)
    rows, res = [], {}
    for b in grid:
        groups = defaultdict(list)
        for o in occ:
            groups[o["c"][b]].append(o)
        pur = ent = 0.0
        J = np.zeros((len(toks), len(l2s)))
        for c, os_ in groups.items():
            wc = sum(o["w"] for o in os_)
            pc = Counter()
            fc = Counter()
            vc = Counter()
            for o in os_:
                pc[o["pos"]] += o["w"]
                fc[o["tok"]] += o["w"]
                vc[o["l2"]] += o["w"]
            pur += max(pc.values())
            ent += wc * -sum(x / wc * math.log2(x / wc) for x in fc.values())
            fi = np.array([ti[t] for t in fc])
            pf = np.array([x / wc for x in fc.values()])
            vv = np.array([vi[v] for v in vc])
            pv = np.array([x / wc for x in vc.values()])
            J[np.ix_(fi, vv)] += (wc / W) * np.outer(pf, pv)
        res[b] = dict(concepts=len(groups), purity=pur / W, entropy=ent / W, I=mi(J),
                      level=float(np.average([D["node bits"][str(o["c"][b])] for o in occ], weights=[o["w"] for o in occ])))
    Ileaf = res["inf"]["I"]
    for b in grid:
        r = res[b]
        r["selection"] = r["I"] / Ileaf if Ileaf > 0 else None
        r["selection_raw"] = r["I"] / I_raw if I_raw > 0 else None
        r["pass"] = r["purity"] >= 0.8 and r["entropy"] >= 3.0 and r["selection"] <= 0.5
    L = ["# r214 Phase 0 level map", "",
         "seed %s, n=%s, window %s; %d committed word occurrences (weight %.1f), %d word types, %d sentences; "
         "I(filler; l2) empirical = %.3f bits, class-model at the leaf = %.3f bits; owner mismatches %s" % (
             D["seed"], D["n"], D["window"], len(occ), W, len(toks), len(sents), I_raw, Ileaf, D["mismatched owner"]), "",
         "| band b (bits) | concepts | mean level (bits) | slot purity | filler entropy (bits) | selection retained | / empirical | meets pass band |",
         "|---|---|---|---|---|---|---|---|"]
    for b in grid:
        r = res[b]
        L.append("| %s | %d | %.2f | %.3f | %.2f | %.3f | %.3f | %s |" % (
            b, r["concepts"], r["level"], r["purity"], r["entropy"], r["selection"], r["selection_raw"], "yes" if r["pass"] else "no"))
    passing = [b for b in grid if res[b]["pass"]]
    L += ["", "pass band (purity >= .8, entropy >= 3 bits, selection retained <= .5): %s" % (passing or "none (Kill A)")]
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    if out_json:
        json.dump({"I_raw": I_raw, "I_leaf": Ileaf, "bands": res, "passing": passing}, open(out_json, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
