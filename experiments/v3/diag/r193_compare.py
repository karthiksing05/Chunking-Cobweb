"""Round 193 pre-test scoring: per judged pre-test dir (r193_pretest.py: arm0 original, arm1 repaired), the measures of
story_frontier.py for the original, and for the repair as generation would use it (a repaired sentence that is a stored
sentence falls back to the original).

Usage (compute node): r193_compare.py <pre-test dir> [...]
"""
import json
import os
import re
import sys
import glob

sys.path.insert(0, "experiments/v3")
from story_frontier import Train, grams, train_sentences  # noqa: E402


def stats(sents, judged, T):
    new = [(s, j) for s, j in zip(sents, judged) if s not in T.whole]
    tri = [any(g not in T.tri for g in grams(s, 3)) for s, _ in new]
    ok = [j > 0.5 for _, j in new]
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return m(ok), m(tri), m([a for a, t in zip(ok, tri) if t]), len(new)


for d in sys.argv[1:]:
    for f0 in sorted(glob.glob(os.path.join(d, "*_arm0.jsonl"))):
        seed, n = map(int, re.search(r"_s(\d+)_n(\d+)_arm0", f0).groups())
        T = Train(train_sentences(seed, n))
        f1 = f0.replace("_arm0", "_arm1")
        a = [tuple(json.loads(l)["gen"].split()) for l in list(open(f0, encoding="utf-8"))[1:]]
        b = [tuple(json.loads(l)["gen"].split()) for l in list(open(f1, encoding="utf-8"))[1:]]
        pa = json.load(open(f0[:-6] + ".scores.json", encoding="utf-8"))
        pb = json.load(open(f1[:-6] + ".scores.json", encoding="utf-8"))
        gen = [(y, q) if y not in T.whole else (x, p) for x, y, p, q in zip(a, b, pa, pb)]
        for name, (ss, jj) in (("original", (a, pa)), ("repaired (raw)", (b, pb)),
                               ("repaired, stored -> original", ([g for g, _ in gen], [j for _, j in gen]))):
            g, t, gt, nn = stats(ss, jj, T)
            print(f"{os.path.basename(d)} s{seed} {name:30s} new {nn:5d} gram {g:.3f} tri {t:.3f} gram|tri {gt:.3f}")
