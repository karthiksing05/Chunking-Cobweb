"""TRELLIS simulation of Knowlton & Squire (1996) Experiment 1 (artificial grammar learning), controls, N = 18.

Binding design: docs/PLAN_PSYCH.md "P4 pre-registration". Stimuli and human numbers: docs/AGL_PROTOCOL.md section 1.

Usage:
  psych_agl.py run --arm full|conceptsoff|acs|tp [--seeds 13,...,30] [--train26] [--out DIR]
  psych_agl.py summary [--out DIR]       -> DIR/SUMMARY.txt

One simulated subject per seed (13..30). Training = the 23 distinct strings, each presented twice (46 learn() calls), random
order per seed (the second pass is a fresh shuffle of all 46 presentations; one shuffle of the 46 -- see ASSUMPTION). Each
presentation = one learn() on the letter tokens. Learning off at test (L._quiet = True while parsing, as psych_digits.Mem.top).
Learner = psych_segment.Trellis3Seg with the psych_segment CFG (C3); arm conceptsoff = concepts_off + concepts_off_parse, as there.

Scores (per test item; endorsement = above the subject's own median, ties at the median split at random with the seed's rng):
  primary  span_w : mean over all sub-spans of length >= 2 of the test string of the stored posterior weight span_w of that
                    token tuple (summed over the records of every learn(); psych_segment bookkeeping).
  logZ            : parse logZ / n_tokens                                       (secondary)
  regen           : sum of L._deriv_bits over the chosen (MBR) composites / n    (secondary; endorse LOW bits, i.e. the
                    endorsement score is -bits)
  acs  (baseline) : printed KS96 chunk-strength value of each item (deterministic; ties broken by the seed's rng)
  tp   (baseline) : mean over the item's within-item transitions (bigram P(b|a) = c(ab)/c(a.), trigram P(c|ab) = c(abc)/c(ab.)),
                    counts from the same training presentations with # boundary symbols at both ends (as psych_segment.TP;
                    # transitions are not scored). Mean rather than product so strings of length 2..6 are comparable.
Median rule implemented as: rank by score with random tie-breaking (seed rng), endorse the top 16 of 32 (= above the median,
ties across the median split at random).

ASSUMPTION (also in every JSON): the 32 test rows are used exactly as printed in the KS96 Appendix, including the duplicate
rows (GL contains XVTVJ twice, rows 6 and 7, with printed CS 6.7 and 6.1, and VTVJJ twice, rows 2 and 8, CS 5.1 and 5.2; the
doc says one of each pair is a typesetting error in the original; strings kept as printed, printed CS kept for the acs arm).
Training: 23 distinct strings x 2 presentations, 46 learns, in a random order per seed. The 26-row printed list (3 strings
printed twice) is only the --train26 sensitivity run (26 x 2 = 52 learns). ACS is the printed values, not recomputed.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, os.path.join("experiments", "v3"))
from psych_segment import CFG, Trellis3Seg  # noqa: E402

OUT = os.path.join("experiments", "v3", "results", "psych_agl")
SEEDS = list(range(13, 31))

TRAIN26 = ("XXVT XXVXJJ VXJJ XVJTVJ XXVXJ XVX VXJJJJ XVT XXXVT VJ XVXJJJ VJTVTV "
           "XXVJ VTVJ VJTVX XXXVTV XXVJ VJTVX XXXVTV XVXJJ VT VJTVXJ XXXVX VJTXVJ XVXJ XXXXVX").split()
TRAIN23 = list(dict.fromkeys(TRAIN26))
assert len(TRAIN26) == 26 and len(TRAIN23) == 23

TEST = {  # cell -> [(string, printed CS)], order as printed
    "GH": [("XXVXJ", 10.4), ("XVTV", 6.8), ("VXJ", 9.3), ("XXVTV", 8.0), ("XVJTVX", 7.4), ("XXVTVJ", 7.7), ("VJTXVX", 6.8), ("VX", 12.0)],
    "GL": [("VJTVT", 6.0), ("VTVJJ", 5.1), ("VTVJ", 5.6), ("XVJTVT", 6.7), ("VTV", 5.0), ("XVTVJ", 6.7), ("XVTVJ", 6.1), ("VTVJJ", 5.2)],
    "NGH": [("VJTV", 7.0), ("XXV", 12.3), ("XVXV", 10.0), ("XVXVJ", 9.1), ("XXVJJJ", 7.8), ("XJJ", 7.0), ("VXVJ", 8.2), ("XVXT", 7.0)],
    "NGL": [("XXJJ", 6.8), ("VXJTJ", 4.9), ("XXVJJ", 6.2), ("JXVT", 5.0), ("XXTX", 2.8), ("TVJ", 6.7), ("VXJJX", 5.9), ("VJJXVT", 4.9)],
}
ITEMS = [(c, s, cs) for c in ("GH", "GL", "NGH", "NGL") for s, cs in TEST[c]]
assert len(ITEMS) == 32
HUMAN = {"GH": 64.6, "GL": 62.5, "NGH": 54.9, "NGL": 28.1}

ASSUMPTION = ("Test rows exactly as printed in KS96 Appendix incl. duplicate GL rows (XVTVJ x2, VTVJJ x2; CS as printed); "
              "training = 23 distinct strings x 2 presentations, one random order of the 46 per seed; 26-row list only in the "
              "sensitivity run; median rule = rank with random tie-break, top 16 of 32 endorsed; regen endorses low bits; "
              "tp = mean of within-item bigram and trigram forward TPs with # boundaries; acs = printed CS, ties by seed.")


def endorse(scores, rng):
    """Endorse items above the median; ties at the median split at random (rank with random tie-break, top n/2)."""
    order = list(range(len(scores)))
    rng.shuffle(order)
    order.sort(key=lambda i: -scores[i])          # stable: ties keep the random order
    en = [0] * len(scores)
    for i in order[: len(scores) // 2]:
        en[i] = 1
    return en


def tp_scores(train_seq):
    c1, c2, c3 = Counter(), Counter(), Counter()
    for s in train_seq:
        q = ["#"] + list(s) + ["#"]
        for a, b in zip(q, q[1:]):
            c1[a] += 1
            c2[(a, b)] += 1
        for a, b, c in zip(q, q[1:], q[2:]):
            c3[(a, b, c)] += 1
    out = []
    for _, s, _ in ITEMS:
        q = ["#"] + list(s) + ["#"]
        ps = [c2[(a, b)] / c1[a] if c1[a] else 0.0 for a, b in zip(s, s[1:])]
        # trigram TP within the item; the leading # is the context of the first letter, not scored (as psych_segment.TP)
        ps += [c3[(a, b, c)] / c2[(a, b)] if c2[(a, b)] else 0.0 for a, b, c in zip(s, s[1:], s[2:])]
        out.append(sum(ps) / len(ps))
    return out


def run_one(arm, seed, outdir, train):
    t0 = time.time()
    rng = random.Random(f"agl-{seed}")
    seq = [s for s in train for _ in range(2)]
    rng.shuffle(seq)
    res = {"arm": arm, "seed": seed, "assumption": ASSUMPTION, "cfg": CFG, "train_order": seq, "n_train_strings": len(train),
           "items": [{"cell": c, "string": s, "cs": cs} for c, s, cs in ITEMS]}
    erng = random.Random(f"agl-endorse-{seed}")
    scores = {}
    if arm == "acs":
        scores["acs"] = [cs for _, _, cs in ITEMS]
    elif arm == "tp":
        scores["tp"] = tp_scores(seq)
    else:
        kw = {"concepts_off": True, "concepts_off_parse": True} if arm == "conceptsoff" else {}
        L = Trellis3Seg(seed=seed, **CFG, **kw)
        tl = time.time()
        for s in seq:
            L.learn_buf(list(s))
        res["learn_s"] = time.time() - tl
        res["n_records"] = len(L.records)
        sp, lz, rg = [], [], []
        cache = {}
        for _, s, _ in ITEMS:
            if s not in cache:
                toks = list(s)
                n = len(toks)
                spans = [tuple(toks[i:j]) for i in range(n) for j in range(i + 2, n + 1)]
                spw = sum(L.S(x) for x in spans) / len(spans)
                L._quiet = True
                try:
                    r = L.parse(toks)
                    bits = sum(L._deriv_bits(r["comps"][k]) for k in r["chosen"])
                finally:
                    L._quiet = False
                cache[s] = (spw, r["logZ"] / n, -bits / n, bits / n)
            spw, lzn, nb, b = cache[s]
            sp.append(spw), lz.append(lzn), rg.append(nb)
        scores["span_w"], scores["logZ"], scores["regen"] = sp, lz, rg
        res["regen_bits_per_token"] = [cache[s][3] for _, s, _ in ITEMS]
    res["scores"] = scores
    res["endorse"] = {k: endorse(v, erng) for k, v in scores.items()}
    res["cell_endorse"] = {k: {c: 100 * sum(e for e, it in zip(v, ITEMS) if it[0] == c) / 8 for c in HUMAN}
                           for k, v in res["endorse"].items()}
    res["time_s"] = time.time() - t0
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"s{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    print(json.dumps({"arm": arm, "seed": seed, "cells": res["cell_endorse"], "time_s": round(res["time_s"], 1)}), flush=True)


# ------------------------------------------------------------------------------------------------ summary
def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float("nan")


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def paired_t(d):
    s = sd(d)
    if not s or math.isnan(s):
        return float("nan"), len(d) - 1
    return mean(d) / (s / math.sqrt(len(d))), len(d) - 1


def load(d):
    return [json.load(open(os.path.join(d, f), encoding="utf-8")) for f in sorted(os.listdir(d)) if f.startswith("s") and f.endswith(".json")]


def block(subjects, key):
    ce = [s["cell_endorse"][key] for s in subjects]
    cells = {c: [x[c] for x in ce] for c in HUMAN}
    m = {c: mean(v) for c, v in cells.items()}
    sdc = {c: sd(v) for c, v in cells.items()}
    gram = [(x["GH"] + x["GL"]) / 2 - (x["NGH"] + x["NGL"]) / 2 for x in ce]
    cs = [(x["GH"] + x["NGH"]) / 2 - (x["GL"] + x["NGL"]) / 2 for x in ce]
    inter = [(x["NGH"] - x["NGL"]) - (x["GH"] - x["GL"]) for x in ce]
    gH = [x["GH"] - x["NGH"] for x in ce]
    gL = [x["GL"] - x["NGL"] for x in ce]
    r = pearson([m[c] for c in HUMAN], [HUMAN[c] for c in HUMAN])
    lines = [f"    cells (mean +- SD across subjects, N={len(ce)}): " + "  ".join(f"{c} {m[c]:5.1f}+-{sdc[c]:4.1f}" for c in HUMAN),
             f"    grammaticality effect (G-NG) {mean(gram):6.1f}   chunk-strength effect (H-L) {mean(cs):6.1f}   "
             f"interaction (CS effect NG - CS effect G) {mean(inter):6.1f}",
             "    paired t, G - NG within H: {:6.1f} (t({})={:.2f})   within L: {:6.1f} (t({})={:.2f})".format(
                 mean(gH), len(gH) - 1, paired_t(gH)[0], mean(gL), len(gL) - 1, paired_t(gL)[0]),
             f"    Pearson r with human cell means: {r:.3f}"]
    return lines, {"gH": mean(gH), "gL": mean(gL), "gram_within": (mean(gH) + mean(gL)) / 2, "m": m}


def summary(outdir):
    arms = ["full", "conceptsoff", "acs", "tp"]
    out = ["Knowlton & Squire (1996) Exp 1, TRELLIS simulation (psych_agl.py). Endorsement % per cell; subjects = seeds 13..30.",
           f"Human (controls, KS96 Table 3): " + "  ".join(f"{c} {v}" for c, v in HUMAN.items()) +
           "  | G-NG 22.0, H-L 14.4, interaction (NG CS effect 26.8 - G CS effect 2.1) 24.7", ""]
    keep = {}
    for arm in arms:
        d = os.path.join(outdir, arm)
        if not os.path.isdir(d):
            continue
        subs = load(d)
        for key in subs[0]["endorse"]:
            tag = "PRIMARY" if key in ("span_w", "acs", "tp") else "secondary"
            out.append(f"[{arm}] score = {key} ({tag}), n subjects = {len(subs)}")
            lines, k = block(subs, key)
            out += lines + [""]
            keep[(arm, key)] = k
    d = os.path.join(outdir, "sens26_full")
    if os.path.isdir(d):
        subs = load(d)
        out.append(f"SENSITIVITY (secondary): 26-row printed training list x2 presentations, full arm, primary score span_w, n = {len(subs)}")
        out += block(subs, "span_w")[0] + [""]
    if ("full", "span_w") in keep:
        f = keep[("full", "span_w")]
        out.append("Pre-registered mechanism criterion (primary score; grammaticality effect within CS cells, mean of G-NG within H and within L):")
        for a in ("acs", "conceptsoff"):
            if (a, "span_w" if a != "acs" else "acs") in keep:
                o = keep[(a, "span_w" if a != "acs" else "acs")]
                out.append(f"  full {f['gram_within']:.1f} (H {f['gH']:.1f}, L {f['gL']:.1f}) vs {a} {o['gram_within']:.1f} (H {o['gH']:.1f}, L {o['gL']:.1f}) -> {'exceeds' if f['gram_within'] > o['gram_within'] else 'does NOT exceed'}")
    with open(os.path.join(outdir, "SUMMARY.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--arm", default="full", choices=["full", "conceptsoff", "acs", "tp"])
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--train26", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.mode == "summary":
        summary(a.out)
    else:
        sub = "sens26_full" if a.train26 else a.arm
        for s in [int(x) for x in a.seeds.split(",")]:
            run_one(a.arm, s, os.path.join(a.out, sub), TRAIN26 if a.train26 else TRAIN23)
