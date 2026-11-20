"""Round 229: is the r220 chess realism judge file-symmetric?
Takes the real held-out positions the judge was validated on (train_long.fen lines [25459:26459], the "real held-out" control),
applies (a) every file shift of +/-1 and +/-2 for which all pieces stay on the board, (b) the horizontal mirror a<->h, keeps
the legal results (python-chess is_valid) and reports the share the judge classifies "real" (score >= TAU).
Usage (compute node, repo root): r229_judge_file_symmetry.py -> results/chess_regimes/r229_judge_file_symmetry.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r220_judge as J  # noqa: E402

OUT = os.path.join(J.ROOT, "experiments", "v3", "results", "chess_regimes", "r229_judge_file_symmetry.json")
jd = J.Judge()
held = [J.real(i) for i in range(25459, 26459)]


def shift(p, d):
    q = {(f + d, r): t for (f, r), t in p.items()}
    return q if all(0 <= f < 8 for f, _ in q) else None


def mirror(p):
    return {(7 - f, r): t for (f, r), t in p.items()}


transforms = {"identity": lambda p: p, "mirror a<->h": mirror}
for d in (-2, -1, 1, 2):
    transforms["shift %+d" % d] = (lambda dd: (lambda p: shift(p, dd)))(d)

res = {"tau": jd.tau, "n_held_out": len(held)}
for name, fn in transforms.items():
    tried = kept = real = 0
    scores = []
    for p in held:
        q = fn(p)
        if q is None:
            continue
        tried += 1
        if not J.legal(q):
            continue
        kept += 1
        s = float(jd.score(q))
        scores.append(s)
        real += s >= jd.tau
    res[name] = {"positions the transform applies to": tried, "legal kept": kept, "real": real,
                 "share real of kept": round(real / kept, 3), "share real of applicable": round(real / tried, 3),
                 "mean score of kept": round(float(np.mean(scores)), 3)}
    print(name, res[name], flush=True)
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
