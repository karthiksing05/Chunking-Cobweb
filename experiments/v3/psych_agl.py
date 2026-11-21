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


def tp_scores(train_seq, items=None):
    items = ITEMS if items is None else items
    c1, c2, c3 = Counter(), Counter(), Counter()
    for s in train_seq:
        q = ["#"] + list(s) + ["#"]
        for a, b in zip(q, q[1:]):
            c1[a] += 1
            c2[(a, b)] += 1
        for a, b, c in zip(q, q[1:], q[2:]):
            c3[(a, b, c)] += 1
    out = []
    for _, s, _ in items:
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


# ---- MVdL97 stimuli, transcribed from docs/AGL_PROTOCOL.md section 2.3-2.4 (Appendix C/D); ACS = recomputed global ACS ----
MV_TRAIN = {
    "2a": ("MXRVVVV MVXRMXT MVRVVV VMRVXT VXTRRRR MXRVXVR VMRVVVM VMRMVR VXVRVMT MXRMVRV MVXTRX MVXRVMR MXRVMRM VMRVMRV VMRVVM VMRVXVT MVXTRR MVXRM MXRMVRM MXTRRRX VMRTMXR MXTRR MVR VMTX VXVTRR MXRMVR MVXRVXT VXVTRRX VMRVMTR MVXRMXR MXRVXTR MVXRVV").split(),
    "2b": ("VXVTX VXTRRX MVXRVVV VXVRVM VMRMXRM VXVT VXTRX MVXRMXT MXRVXVR MXRVXTR VXVTR VMRTVXT MVRVVM VMRMXTR MXRM VXVTRRR VXTX VXTRRR MXRMVXT MXRMXR VXVTRR MVXTRRX MXRMVR MXRVVV MXRVMT VMTRX VXVRVVV VXVRVVM MXRMXT MXRVV VMRTMXR VMTRR MVXRM MVRVM MVXRVVM VMTX MVRVVVV VXVRV VMRVXTX MXRVVM VMTRRRX MXTRRRX MXRVXVT MXRVMTR VMRVXTR VMTRRX MXTRRR MXRVMTX MVXTRR VMRVMT MXRTVMT MVRVV MXRVVVV MVXRVV MXRVXT MXTX MXRTVMR MXRMVRV MXRVVVM VMRMXRV MXTRX VXTRRRR MXTR VMT VMRMXR MXTRRX MXRMVRM VMRVXVT MXRMXRM VMRVVM VMRTMXT MVRVVVM VMRMVR VMRVXVR VMRVV MXRTVXT VXVR VXTRR MVXRV VMRM VMTRRRR VMTR MVXTR VXVRVMR VMRVMRM VMRTVMT VXVRVV VMRVMTX VMRMVXT VXT MVRM MVXTRRR VMRVVVV VXTR MXRVMRV VMTRRR MXTRR MXRVXTX MVXT MXRV MVRVVV VMRMVRM VMRMVRV VMRVVV VMRVMRV MVR VMRMXTX MXRMVXR VMRVMTR VMRVMR MVXTX VXVRMXR MVXRMXR VMRMXT VMRV VXTRRRX VMRTVMR MVXRVMT MXT MXRTMVR MXTRRRR MVRV MVXR VMRVVVM VMR").split(),
}
MV_TEST = {
    "2a": {
        "GA": [("MVXRVVV", 10.09), ("MXRMVXT", 9.91), ("MXRMXRM", 9.45), ("MXRVMT", 9.67), ("MXRVXVT", 8.91), ("VMRVMRM", 10.55), ("VMRVXTR", 10.36), ("VXVRVXT", 8.55), ],
        "GNA": [("MXRMXTX", 7.09), ("MXRTVMR", 6.82), ("VMRTVXT", 6.55), ("VMTRRRX", 5.82), ("VXTX", 6.60), ("VXVTR", 6.29), ("VXVTRX", 5.33), ("VXVTX", 4.86), ],
        "NGA": [("MVRMVXT", 8.73), ("MVRMXR", 8.44), ("MVRVMRM", 9.27), ("MVRVXRM", 9.64), ("MVRVXT", 9.44), ("MXRVXRM", 10.73), ("VMRVVMR", 10.55), ("VMRVXRM", 10.91), ],
        "NGNA": [("MVRVRRX", 6.36), ("MXTRVX", 8.67), ("VMRTMTR", 5.00), ("VMRTRXT", 5.64), ("VXVMRRR", 7.82), ("VXVRX", 5.71), ("VXVTMXT", 5.36), ("VXVTRMR", 5.82), ],
    },
    "2b": {
        "GA": [("MVXRVMR", 36.00), ("MVXRVXT", 32.45), ("MXRMXRV", 36.09), ("MXRMXTX", 28.18), ("MXRVMR", 41.11), ("VMRVM", 43.14), ("VMRVXT", 38.11), ("VXVRVXT", 28.45), ],
        "GNA": [("MVXRMVR", 26.18), ("MVXTRX", 25.89), ("VMRTMVR", 22.27), ("VXVRM", 22.14), ("VXVRMVR", 21.82), ("VXVRMXT", 24.73), ("VXVRVMT", 28.00), ("VXVTRX", 16.44), ],
        "NGA": [("MVRVMRV", 34.45), ("MVXRVXR", 31.82), ("MXRVXR", 36.00), ("MXRVXRM", 33.45), ("MXRVXRV", 36.27), ("VMRVXR", 37.33), ("VMRVXRM", 34.55), ("VMRVXRV", 37.36), ],
        "NGNA": [("MVRMVRM", 20.82), ("MVRMVRV", 24.00), ("MVRMXTX", 22.09), ("MVRVXT", 29.78), ("MVXVRV", 26.89), ("MVXVTR", 20.78), ("MVXVTRX", 18.18), ("VXVRVXR", 27.82), ],
    },
}


# ------------------------------------------------------------------------------------------------ MVdL97 Exps 2a / 2b
# Binding design: docs/PLAN_PSYCH.md "P4b pre-registration" (inherits P4). Stimuli: docs/AGL_PROTOCOL.md section 2.
MV_SEEDS = list(range(13, 33))
MV_CELLS = ("GA", "GNA", "NGA", "NGNA")
MV_HUMAN = {"2a": {"GA": 65.3, "GNA": 42.8, "NGA": 67.5, "NGNA": 45.0},
            "2b": {"GA": 67.8, "GNA": 62.8, "NGA": 56.3, "NGNA": 60.3}}
MV_STORE_S, MV_GAIN_S, MV_REC_W = 8.0, 3.0, 0.5     # limit arm: 8 s to store a new string, +3 s per presentation, REC_W as psych_digits
MV_ASSUMPTION = ("2a: one random order of the 32 strings per seed, shown twice in that order (64 learns); 2b: the 125 strings once in a "
                 "random order per seed (125 learns); full_x2 (2b sensitivity) = that order repeated twice. Test = the 32 strings of "
                 "each experiment, cells GA/GNA/NGA/NGNA, acs arm = recomputed global ACS per item (AGL_PROTOCOL 2.4). Primary score "
                 "span_w, median split with random tie-break (top 16 of 32). limit arm: clock += 3 s per presentation; a string with "
                 "span_w(whole string) >= 0.5 is reinforced +1 without learn(); else learn() costs 8 s and happens only if clock >= 8 "
                 "(clock starts at 0); otherwise the presentation is lost.")
MV_ITEMS = {e: [(c, s, acs) for c in MV_CELLS for s, acs in MV_TEST[e][c]] for e in MV_TEST}
for _e in MV_ITEMS:
    assert len(MV_ITEMS[_e]) == 32
assert len(MV_TRAIN["2a"]) == 32 and len(set(MV_TRAIN["2a"])) == 32
assert len(MV_TRAIN["2b"]) == 125 and len(set(MV_TRAIN["2b"])) == 125


def run_mv(exp, arm, seed, outdir, twice=False):
    t0 = time.time()
    e = exp[4:]                                              # "2a" / "2b"
    items = MV_ITEMS[e]
    rng = random.Random(f"mvdl{e}-{seed}")
    order = list(MV_TRAIN[e])
    rng.shuffle(order)
    reps = 2 if (e == "2a" or twice) else 1
    seq = order * reps
    res = {"exp": exp, "arm": arm, "twice": twice, "seed": seed, "assumption": MV_ASSUMPTION, "cfg": CFG, "train_order": seq,
           "n_presentations": len(seq), "items": [{"cell": c, "string": s, "acs": a} for c, s, a in items]}
    erng = random.Random(f"mvdl{e}-endorse-{seed}")
    scores = {}
    if arm == "acs":
        scores["acs"] = [a for _, _, a in items]
    elif arm == "tp":
        scores["tp"] = tp_scores(seq, items)
    else:
        kw = {"concepts_off": True, "concepts_off_parse": True} if arm == "conceptsoff" else {}
        L = Trellis3Seg(seed=seed, **CFG, **kw)
        tl = time.time()
        if arm == "limit":
            clock, nl, nr, nskip = 0.0, 0, 0, 0
            for s in seq:
                clock += MV_GAIN_S
                tup = tuple(s)
                if L.S(tup) >= MV_REC_W:
                    L.span_w[tup] += 1.0
                    nr += 1
                elif clock >= MV_STORE_S:
                    L.learn_buf(list(s))
                    clock -= MV_STORE_S
                    nl += 1
                else:
                    nskip += 1
            res["limit"] = {"learned": nl, "reinforced": nr, "lost": nskip, "clock_end": clock}
        else:
            for s in seq:
                L.learn_buf(list(s))
        res["learn_s"] = time.time() - tl
        res["n_records"] = len(L.records)
        sp, lz, rg, cache = [], [], [], {}
        for _, s, _ in items:
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
                cache[s] = (spw, r["logZ"] / n, -bits / n)
            sp.append(cache[s][0]), lz.append(cache[s][1]), rg.append(cache[s][2])
        scores["span_w"], scores["logZ"], scores["regen"] = sp, lz, rg
    res["scores"] = scores
    res["endorse"] = {k: endorse(v, erng) for k, v in scores.items()}
    res["cell_endorse"] = {k: {c: 100 * sum(x for x, it in zip(v, items) if it[0] == c) / 8 for c in MV_CELLS}
                           for k, v in res["endorse"].items()}
    res["time_s"] = time.time() - t0
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"s{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    print(json.dumps({"exp": exp, "arm": arm, "seed": seed, "cells": res["cell_endorse"], "time_s": round(res["time_s"], 1)}), flush=True)


def _mv_effects(subs, key):
    ce = [s["cell_endorse"][key] for s in subs]
    gram = [(x["GA"] + x["GNA"]) / 2 - (x["NGA"] + x["NGNA"]) / 2 for x in ce]
    assoc = [(x["GA"] + x["NGA"]) / 2 - (x["GNA"] + x["NGNA"]) / 2 for x in ce]
    m = {c: mean([x[c] for x in ce]) for c in MV_CELLS}
    return ce, m, gram, assoc


def summary_mv(outdir):
    arms = ["full", "conceptsoff", "acs", "tp", "limit"]
    out = ["Meulemans & Van der Linden (1997) Exps 2a/2b, TRELLIS simulation (psych_agl.py --exp mvdl2a|mvdl2b). Endorsement % per cell.",
           "Pre-registration: docs/PLAN_PSYCH.md P4b. Subjects = seeds 13..32 (N=20 per experiment). Primary score = span_w (arms full, conceptsoff),",
           "acs (recomputed global ACS), tp. limit arm and logZ/regen scores are secondary. Associativity effect = A - NA; grammaticality effect = G - NG.",
           "Human cells (GA/GNA/NGA/NGNA): 2a 65.3/42.8/67.5/45.0 (assoc +22.5, gram -2.2); 2b 67.8/62.8/56.3/60.3 (assoc +0.5, gram +7.0); "
           "human shift 2b-2a: gram +9.2, assoc -22.0", "", "ASSUMPTIONS: " + MV_ASSUMPTION, ""]
    st = {}
    for e in ("2a", "2b"):
        out.append(f"=== Experiment {e} ===")
        for sub in arms + (["full_x2"] if e == "2b" else []):
            d = os.path.join(outdir, f"mvdl{e}", sub)
            if not os.path.isdir(d) or not any(f.endswith(".json") for f in os.listdir(d)):
                continue
            subs = load(d)
            for key in subs[0]["endorse"]:
                tag = "PRIMARY" if key in ("span_w", "acs", "tp") and sub != "limit" else "secondary"
                if sub == "full_x2":
                    tag = "SENSITIVITY (shown twice)" + ("" if key == "span_w" else ", secondary")
                ce, m, gram, assoc = _mv_effects(subs, key)
                H = MV_HUMAN[e]
                r = pearson([m[c] for c in MV_CELLS], [H[c] for c in MV_CELLS])
                out.append(f"[{sub}] score = {key} ({tag}), n = {len(subs)}")
                out.append("    cells (mean +- SD): " + "  ".join(f"{c} {m[c]:5.1f}+-{sd([x[c] for x in ce]):4.1f}" for c in MV_CELLS))
                out.append(f"    grammaticality effect {mean(gram):6.1f} (SD {sd(gram):.1f}, paired t({len(gram)-1})={paired_t(gram)[0]:.2f})   "
                           f"associativity effect {mean(assoc):6.1f} (SD {sd(assoc):.1f}, t={paired_t(assoc)[0]:.2f})   Pearson r (4 cells) {r:.3f}")
                if sub == "limit":
                    lm = [s["limit"] for s in subs]
                    out.append("    limit bookkeeping (mean per subject): learned %.1f, reinforced %.1f, lost %.1f" % (
                        mean([x["learned"] for x in lm]), mean([x["reinforced"] for x in lm]), mean([x["lost"] for x in lm])))
                out.append("")
                st[(e, sub, key)] = {"m": m, "gram": gram, "assoc": assoc}
    out.append("=== Shift table: effect in 2b minus effect in 2a (percentage points; SE from the 20 seeds of each experiment, Welch) ===")
    out.append("Human shift: grammaticality +9.2, associativity -22.0")
    out.append(f"{'arm':12s} {'score':8s} {'gram 2a':>8s} {'gram 2b':>8s} {'gram shift':>10s} {'+-SE':>6s}   {'assoc 2a':>8s} {'assoc 2b':>8s} {'assoc shift':>11s} {'+-SE':>6s}   r8")
    shifts = {}
    for arm in arms:
        keys = sorted({k for (e, a, k) in st if a == arm}, key=lambda k: (k not in ("span_w", "acs", "tp"), k))
        for key in keys:
            if ("2a", arm, key) not in st or ("2b", arm, key) not in st:
                continue
            a, b = st[("2a", arm, key)], st[("2b", arm, key)]
            gs = mean(b["gram"]) - mean(a["gram"])
            asf = mean(b["assoc"]) - mean(a["assoc"])
            gse = math.sqrt(sd(a["gram"]) ** 2 / len(a["gram"]) + sd(b["gram"]) ** 2 / len(b["gram"]))
            ase = math.sqrt(sd(a["assoc"]) ** 2 / len(a["assoc"]) + sd(b["assoc"]) ** 2 / len(b["assoc"]))
            r8 = pearson([a["m"][c] for c in MV_CELLS] + [b["m"][c] for c in MV_CELLS],
                         [MV_HUMAN["2a"][c] for c in MV_CELLS] + [MV_HUMAN["2b"][c] for c in MV_CELLS])
            shifts[(arm, key)] = (gs, gse, asf, ase)
            out.append(f"{arm:12s} {key:8s} {mean(a['gram']):8.1f} {mean(b['gram']):8.1f} {gs:10.1f} {gse:6.1f}   {mean(a['assoc']):8.1f} "
                       f"{mean(b['assoc']):8.1f} {asf:11.1f} {ase:6.1f}   {r8:.3f}")
    out.append("(r8 = Pearson r of the 8 simulated cell means, 2a then 2b, with the 8 human cell means.)")
    out.append("")
    out.append("=== Pre-registered mechanism criterion (primary scores) ===")
    out.append("Claim: concepts carry the shift. Holds if the full model's grammaticality shift (2b - 2a) is positive and larger than conceptsoff's and acs's.")
    if ("full", "span_w") in shifts:
        gf, sef = shifts[("full", "span_w")][:2]
        ok = gf > 0
        out.append(f"  full span_w grammaticality shift {gf:+.1f} +- {sef:.1f} -> {'positive' if gf > 0 else 'NOT positive'}")
        for o, okey in (("conceptsoff", "span_w"), ("acs", "acs")):
            if (o, okey) in shifts:
                go, seo = shifts[(o, okey)][:2]
                diff, se = gf - go, math.sqrt(sef ** 2 + seo ** 2)
                ok = ok and gf > go
                out.append(f"  vs {o} {go:+.1f} +- {seo:.1f}: difference {diff:+.1f} +- {se:.1f} (z = {diff / se:.2f}) -> {'larger' if gf > go else 'NOT larger'}")
        if ("tp", "tp") in shifts:
            go, seo = shifts[("tp", "tp")][:2]
            out.append(f"  (reference, not in the criterion) tp {go:+.1f} +- {seo:.1f}")
        out.append(f"CRITERION {'HOLDS' if ok else 'DOES NOT HOLD'} (point estimates; the SEs above say whether the margins exceed across-seed noise).")
    with open(os.path.join(outdir, "SUMMARY_MVDL.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--exp", default="ks96", choices=["ks96", "mvdl2a", "mvdl2b"])
    ap.add_argument("--arm", default="full", choices=["full", "conceptsoff", "acs", "tp", "limit"])
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--train26", action="store_true")
    ap.add_argument("--twice", action="store_true", help="mvdl2b sensitivity: show the 125 strings twice (full arm only)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.mode == "summary":
        if a.exp == "ks96":
            summary(a.out)
        else:
            summary_mv(a.out)
    elif a.exp == "ks96":
        assert a.arm != "limit"
        sub = "sens26_full" if a.train26 else a.arm
        for s in [int(x) for x in (a.seeds or ",".join(map(str, SEEDS))).split(",")]:
            run_one(a.arm, s, os.path.join(a.out, sub), TRAIN26 if a.train26 else TRAIN23)
    else:
        assert not a.twice or (a.exp == "mvdl2b" and a.arm == "full")
        sub = "full_x2" if a.twice else a.arm
        for s in [int(x) for x in (a.seeds or ",".join(map(str, MV_SEEDS))).split(",")]:
            run_mv(a.exp, a.arm, s, os.path.join(a.out, a.exp, sub), twice=a.twice)
