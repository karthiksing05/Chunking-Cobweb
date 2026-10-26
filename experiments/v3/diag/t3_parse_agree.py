"""Consistency of saved held-out parses (t3_parse_consistency.py), no gold:

  bracketed-somewhere consistency  of word strings (2-4 words, 3+ occurrences) that
                       the parser brackets at least once, the share of their
                       occurrences bracketed (1 = a chunk is always a chunk)
  cross-seed F1        bracket F1 between seeds at the same checkpoint (same
                       held-out sentences, independent learners)
Fixed right-/left-branching parses are reported as references.

Usage: t3_parse_agree.py <tag>[,<tag>...] [seeds=13,17]
"""
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, "src")
from trellis2.stories import heldout_path  # noqa: E402

tags = sys.argv[1].split(",")
seeds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["13", "17"]
held = json.load(open(heldout_path(13), encoding="utf-8"))
D = "experiments/v3/results/consistency"


def somewhere(parses):
    occ = defaultdict(list)
    for s, br in zip(held, parses):
        n = len(s)
        for k in (2, 3, 4):
            for i in range(n - k + 1):
                if (i, i + k) != (0, n):
                    occ[tuple(s[i:i + k])].append((i, i + k) in br)
    rep = [v for v in occ.values() if len(v) >= 3 and any(v)]
    tot = sum(len(v) for v in rep)
    return round(sum(sum(v) for v in rep) / max(tot, 1), 3), len(rep)


def f1(a, b):
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / max(len(a) + len(b), 1)


shapes = {"right-branching": lambda n: {(i, n) for i in range(1, n - 1)},
          "left-branching": lambda n: {(0, j) for j in range(2, n)}}
for name, sh in shapes.items():
    c, m = somewhere([sh(len(s)) for s in held])
    print(json.dumps({"tag": name, "bracketed-somewhere consistency": c, "strings": m}))
for tag in tags:
    runs = {}
    for s in seeds:
        p = os.path.join(D, f"{tag}_s{s}.json")
        if os.path.exists(p):
            runs[s] = {int(c): [{tuple(b) for b in br} for br in v] for c, v in json.load(open(p, encoding="utf-8")).items()}
    for s, r in runs.items():
        for c, parses in sorted(r.items()):
            cc, m = somewhere(parses)
            print(json.dumps({"tag": tag, "seed": s, "checkpoint": c, "bracketed-somewhere consistency": cc, "strings": m}))
    for a, b in combinations(sorted(runs), 2):
        for c in sorted(set(runs[a]) & set(runs[b])):
            x = sum(f1(p, q) for p, q in zip(runs[a][c], runs[b][c])) / len(held)
            print(json.dumps({"tag": tag, "seeds": [a, b], "checkpoint": c, "cross-seed bracket F1": round(x, 3)}))
