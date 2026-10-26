"""Phrase consistency of fixed-shape parses (right- / left-branching) on the held-out
set t3_parse_consistency.py uses (substitution and stability are 1 for them by
construction). Usage: cons_baseline.py"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from trellis2.stories import heldout_path  # noqa: E402

held = json.load(open(heldout_path(13), encoding="utf-8"))
for name, shape in (("right-branching", lambda n: {(i, n) for i in range(1, n - 1)}),
                    ("left-branching", lambda n: {(0, j) for j in range(2, n)})):
    occ = defaultdict(list)
    for s in held:
        n, br = len(s), shape(len(s))
        for k in (2, 3, 4):
            for i in range(0, n - k + 1):
                if (i, i + k) != (0, n):
                    occ[tuple(s[i:i + k])].append((i, i + k) in br)
    rep = {g: v for g, v in occ.items() if len(v) >= 3}
    tot = sum(len(v) for v in rep.values())
    print(name, json.dumps({"phrase consistency": round(sum(max(sum(v), len(v) - sum(v)) for v in rep.values()) / tot, 3),
                            "constituent rate": round(sum(sum(v) for v in rep.values()) / tot, 3)}))
