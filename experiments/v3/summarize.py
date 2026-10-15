"""Compact comparison of v3 runs: python experiments/v3/summarize.py PREFIX [PREFIX...] (dirs under experiments/v3/results)."""
import json, os, sys
from collections import defaultdict
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for pre in sys.argv[1:]:
    for d in sorted(x for x in os.listdir(R) if x.startswith(pre)):
        f = os.path.join(R, d, "results.json")
        if not os.path.exists(f):
            continue
        rows = json.load(open(f, encoding="utf-8"))
        if rows and "condition" in rows[0]:
            agg = defaultdict(list)
            for r in rows:
                agg[r["condition"]].append(r)
            parts = []
            for c, rs in agg.items():
                om = sum(r["held-out omission"] for r in rs) / len(rs)
                rb = sum(r["right-branching omission"] for r in rs) / len(rs)
                gr = sum(r["grammatical"] for r in rs) / len(rs)
                parts.append(f"{c} {om:.2f}/{rb:.2f} g{gr:.2f}")
            print(f"{d:14s} " + " | ".join(parts))
        else:
            print(f"{d:14s} " + " | ".join(f"{r['train']}: om {r['held-out omission']:.2f}/{r['right-branching omission']:.2f} wf {r['well formed']:.2f} att {r['positions attested']:.2f}" for r in rows))
