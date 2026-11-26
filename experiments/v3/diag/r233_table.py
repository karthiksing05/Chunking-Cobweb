"""Tabulate r233/r234 unseen-sentence parse results: means over seeds and paired 95% t intervals on F1 differences.
usage: r233_table.py [g2]"""
import glob, json, os, statistics as st, sys
SUF = "_g2" if len(sys.argv) > 1 and sys.argv[1] == "g2" else ""
D = f"experiments/v3/results/story_gen/r233/unseen{SUF}"
SEEDS = [13, 17, 19, 23, 29, 31]
NS = ["125", "1000", "4000"]
T5 = 2.571

def get(fn, key):
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d[key]

rows = {}
for arm, lab in [("full", "TRELLIS"), ("nochunk", "chunks off"), ("minbits:3", "m>=3"), ("minbits:6", "m>=6")]:
    rows[lab] = {s: get(f"goldparse_s{s}_{arm}.json", "trellis") for s in SEEDS}
rows["right-branching"] = {s: (lambda b: b and {n: b["right-branching"] for n in NS})(get(f"goldparse_s{s}_full.json", "baselines") or get(f"goldparse_s{s}_nochunk.json", "baselines")) for s in SEEDS}
for fn, lab in [("ccm_it0", "CCM initial counts"), ("ccm_it1", "CCM one pass"), ("ccm", "CCM 20 passes")]:
    rows[lab] = {s: get(f"{fn}_s{s}.json", "ccm") for s in SEEDS}
rows["exact-chunk"] = {s: get(f"exactchunk_s{s}.json", "count") for s in SEEDS}

def cell(r, n, k):
    v = [r[s][n][k] for s in SEEDS if r[s] and n in r[s]]
    return (st.mean(v), len(v)) if v else (None, 0)

print("grammar", SUF or "g1")
for lab, r in rows.items():
    out = []
    for n in NS:
        R, c = cell(r, n, "recall"); X, _ = cell(r, n, "exact"); F, _ = cell(r, n, "micro_F1")
        out.append("--" if R is None else f"{R:.2f}/{X:.2f}/{F:.2f}({c})")
    print(f"{lab:20s}", "  ".join(out))

def diff(a, b, k="micro_F1"):
    for n in NS:
        d = [rows[a][s][n][k] - rows[b][s][n][k] for s in SEEDS if rows[a][s] and rows[b][s] and n in rows[a][s] and n in rows[b][s]]
        if len(d) < 2:
            print(f"  {a} - {b} {k} n={n}: --"); continue
        m, h = st.mean(d), T5 * st.stdev(d) / len(d) ** .5 if len(d) == 6 else float("nan")
        print(f"  {a} - {b} {k} n={n}: {m:+.3f} [{m-h:+.3f}, {m+h:+.3f}] (k={len(d)})")

print("paired differences")
for a, b in [("TRELLIS", "CCM 20 passes"), ("TRELLIS", "CCM one pass"), ("CCM initial counts", "right-branching"),
             ("exact-chunk", "right-branching"), ("TRELLIS", "chunks off"), ("TRELLIS", "m>=3"), ("TRELLIS", "right-branching")]:
    diff(a, b)
diff("TRELLIS", "chunks off", "recall"); diff("TRELLIS", "m>=3", "recall")
