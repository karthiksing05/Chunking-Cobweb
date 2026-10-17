"""Seed-averaged table for run_trellis3 results: python experiments/v3/t3_table.py PREFIX [PREFIX...]
(each prefix names dirs PREFIX_s<seed> under experiments/v3/results)."""
import json, os, sys, statistics as st
from collections import defaultdict
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for pre in sys.argv[1:]:
    rows = []
    for d in sorted(os.listdir(R)):
        f = os.path.join(R, d, "results.json")
        if d.startswith(pre + "_s") and os.path.exists(f):
            rows += json.load(open(f, encoding="utf-8"))
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    cells, means, rbs = [], [], []
    for c in ("small", "med", "large", "term_low", "term_med", "term_high"):
        rs = by.get(c, [])
        if not rs:
            continue
        om = [r["held-out omission"] for r in rs]
        rb = [r["right-branching omission"] for r in rs]
        pur = [r["word POS purity by depth"]["3"][0] for r in rs]
        win = sum(o < b for o, b in zip(om, rb))
        means.append(st.mean(om)); rbs.append(st.mean(rb))
        cells.append(f"{c} {st.mean(om):.2f}±{st.pstdev(om):.2f}/RB{st.mean(rb):.2f} ({win}/{len(om)}) pos{st.mean(pur):.2f}")
    if means:
        print(f"{pre:14s} mean {st.mean(means):.3f} (RB {st.mean(rbs):.3f}) | " + " | ".join(cells))
