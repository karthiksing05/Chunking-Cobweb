"""Gather experiments/v3/results/psych_segment/*.json: per experiment, model means per condition next to the human data.

Usage: psych_segment_summary.py [results dir]   (default experiments/v3/results/psych_segment; use ..._tp for the TP-only check)

Human numbers: docs/SAN96_PROTOCOL.md
  * condition coefficients (betas), section 2.3 (Frank et al. 2010)
  * condition MEANS (% correct), last section "Human condition means read from FGGT10 Figure 1" (digitised, +-1 point)
Correlations (Pearson, Spearman) of the model's condition means with the human condition means (as Frank et al. did) and,
as a secondary check, with the betas. Models: TRELLIS Luce score (arms full / conceptsoff), TRELLIS bits choice, TP baseline.
Infants: mean Luce, mean proportion better (human .06 for words vs part-words, doc 1.1) per condition.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join("experiments", "v3", "results", "psych_segment")

CONDS = {"f1": ["1", "2", "3", "4", "6", "8", "12", "24"], "f2": ["48", "100", "300", "600", "900", "1200"],
         "f3": ["3", "4", "5", "6", "9"]}
# docs/SAN96_PROTOCOL.md section 2.3 (betas) and last section (digitised means, % correct, Frank et al. Figure 1)
BETA = {"f1": [2.31, 1.56, 1.64, 1.39, 1.32, 1.31, .86, .45], "f2": [.082, .39, .75, .76, 1.03, 1.33],
        "f3": [2.12, 1.77, 1.17, .84, .39]}
MEAN = {"f1": [87.2, 80.5, 80.8, 79.0, 75.0, 77.0, 68.6, 60.3], "f2": [57.2, 63.8, 70.8, 70.5, 75.3, 80.0],
        "f3": [87.4, 81.1, 75.3, 71.8, 65.0]}
NAMES = {"f1": "Frank Exp 1 (words per sentence)", "f2": "Frank Exp 2 (tokens)", "f3": "Frank Exp 3 (word types)",
         "i1": "SAN96 infants Exp 1 (words vs nonwords)", "i2": "SAN96 infants Exp 2 (words vs part-words)"}


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def f(x, d=3):
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


rows = defaultdict(list)   # (exp, cond, arm) -> list of file dicts
for fn in sorted(glob.glob(os.path.join(D, "*.json"))):
    r = json.load(open(fn, encoding="utf-8"))
    rows[(r["exp"], r["cond"], r["arm"])].append(r)


def cmean(exp, cond, arm, key):
    rs = [r[key] for r in rows.get((exp, cond, arm), []) if r.get(key) is not None]
    return (sum(rs) / len(rs), len(rs)) if rs else (None, 0)


for exp in ("f1", "f2", "f3"):
    cs = CONDS[exp]
    print(f"\n=== {NAMES[exp]} ===  (model = mean over participants of the per-participant mean Luce / choice; human = % correct / 100)")
    cols = [("TRELLIS full Luce", "full", "luce_mean"), ("TRELLIS full bits", "full", "bits_choice_mean"),
            ("TRELLIS c-off Luce", "conceptsoff", "luce_mean"), ("TRELLIS c-off bits", "conceptsoff", "bits_choice_mean"),
            ("TP", "full", "tp_luce_mean")]
    # TP is stored in every run; take it from whichever arm exists (full, else conceptsoff, else tp)
    def get(arm, key, c):
        if key == "tp_luce_mean":
            for a in ("full", "conceptsoff", "tp"):
                m, n = cmean(exp, c, a, key)
                if n:
                    return m, n
            return None, 0
        return cmean(exp, c, arm, key)
    print(f"{'cond':>6} {'n':>3} {'human mean':>10} {'beta':>6} " + " ".join(f"{c[0]:>18}" for c in cols))
    series = {c[0]: [] for c in cols}
    for i, c in enumerate(cs):
        vals = [get(a, k, c) for _, a, k in cols]
        n = max(v[1] for v in vals)
        print(f"{c:>6} {n:>3} {MEAN[exp][i] / 100:>10.3f} {BETA[exp][i]:>6.2f} " + " ".join(f"{f(v[0]):>18}" for v in vals))
        for (nm, _, _), v in zip(cols, vals):
            series[nm].append(v[0])
    print("correlation with human condition means (Pearson / Spearman) | with betas (Pearson / Spearman)")
    for nm, _, _ in cols:
        s = series[nm]
        if any(v is None for v in s):
            print(f"  {nm:>20}: incomplete")
            continue
        print(f"  {nm:>20}: {f(pearson(s, [m / 100 for m in MEAN[exp]]), 2)} / {f(spearman(s, MEAN[exp]), 2)}   |   "
              f"{f(pearson(s, BETA[exp]), 2)} / {f(spearman(s, BETA[exp]), 2)}")
print("\nReference (doc 2.4, Frank et al. vs human means): TP .84 / .43 / -.99; lexical model .94 / .89 / -.98; PARSER .00 / .86 / .00.")
print("Human betas are condition coefficients (logit scale); Exp 2 order of the last four is presumed (doc 2.3).")

for exp in ("i1", "i2"):
    print(f"\n=== {NAMES[exp]} ===  (human: words-vs-part-words proportion better .06, looks longer to the novel foil; SAN96b 12/16 prefer nonwords)")
    print(f"{'cond':>5} {'arm':>12} {'n':>3} {'Luce':>7} {'prop better':>12} {'bits choice':>12} {'TP Luce':>8}")
    for c in ("A", "B", "all"):
        for arm in ("full", "conceptsoff"):
            rs = [r for cc in ("A", "B") if c in (cc, "all") for r in rows.get((exp, cc, arm), [])]
            if not rs:
                continue
            m = lambda k: sum(r[k] for r in rs) / len(rs)
            print(f"{c:>5} {arm:>12} {len(rs):>3} {m('luce_mean'):>7.3f} {m('prop_better_mean'):>12.3f} {m('bits_choice_mean'):>12.3f} {m('tp_luce_mean'):>8.3f}")
    n_pos = sum(1 for arm in ("full",) for cc in ("A", "B") for r in rows.get((exp, cc, arm), []) if r["prop_better_mean"] > 0)
    print(f"  infants with words more familiar (full arm): {n_pos} of {sum(len(rows.get((exp, cc, 'full'), [])) for cc in ('A', 'B'))}")
