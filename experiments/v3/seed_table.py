"""Seed-averaged comparison across result dirs sharing a prefix (one dir per seed)."""
import json, os, sys, statistics as st
from collections import defaultdict
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for pre in sys.argv[1:]:
    rows = []
    for d in os.listdir(R):
        if d.startswith(pre + "_s") and os.path.exists(os.path.join(R, d, "results.json")):
            rows += json.load(open(os.path.join(R, d, "results.json"), encoding="utf-8"))
    agg = defaultdict(list)
    for r in rows:
        agg[r["condition"]].append(r)
    out = []
    for c, rs in agg.items():
        om = [r["held-out omission"] for r in rs]; rb = [r["right-branching omission"] for r in rs]
        wins = sum(o < b for o, b in zip(om, rb))
        out.append(f"{c} {st.mean(om):.2f}±{st.pstdev(om):.2f} (RB {st.mean(rb):.2f}, {wins}/{len(om)} better) g{st.mean(r['grammatical'] for r in rs):.2f} pur{st.mean(r['category purity']['3']['purity'] for r in rs):.2f}")
    print(pre, "|", " | ".join(out))
