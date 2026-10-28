"""Chase & Simon case study (docs/CHESS_CS.md): a few held-out positions followed
through experience. At each checkpoint memory is shown the real position and the
same pieces scrambled (as chess_recall.py, same split and recognition), and the
units kept in short-term memory (K largest recognised units) are recorded, so a
figure can show which pieces are recalled and in which chunks.

Writes experiments/v3/results/chess_case/<tag>_s<seed>.json.

Usage: chess_case.py <cfg json> <seed> <checkpoints> <tag> [theta K n_cases]
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
theta = float(sys.argv[5]) if len(sys.argv) > 5 else 16.0
K = int(sys.argv[6]) if len(sys.argv) > 6 else 7
n_cases = int(sys.argv[7]) if len(sys.argv) > 7 else 6
max_size = cfg.pop("max_size", 4)
n_test = 100

positions = C.load_positions(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen"))
random.Random(seed).shuffle(positions)
test_real, train = positions[-n_test:], positions[:-n_test]
rng = random.Random(seed + 1)


def randomize(pos):
    pieces = list(pos.values())
    squares = [(f, r) for f in range(8) for r in range(8)]
    while True:
        rng.shuffle(squares)
        out = dict(zip(squares, pieces))
        if all(not (p[1] == "P" and sq[1] in (0, 7)) for sq, p in out.items()):
            return out


test_random = [randomize(p) for p in test_real]          # same draws as chess_recall.py
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)


def bits(e):
    if "climb" not in e.memo:
        e.memo["climb"] = -1e9 if L.cnt.root.count <= 0 else \
            L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
    return e.memo["climb"]


def units(e):
    if e.token is not None or bits(e) >= theta:
        return [e]
    return units(e.left) + units(e.right)


def kept(pos):
    r = L.parse(pos)
    sqs = r["exp"].data[1]
    us = sorted((u for e in r["top"] for u in units(e)), key=lambda u: -len(u.span))
    show = lambda u: [[C.square_name(sqs[i]), pos[sqs[i]]] for i in sorted(u.span)]
    return {"kept": [show(u) for u in us[:K]], "dropped": [show(u) for u in us[K:]],
            "recall": round(sum(len(u.span) for u in us[:K]) / len(pos), 3)}


cases = [{"real": {C.square_name(s): t for s, t in test_real[j].items()},
          "scrambled": {C.square_name(s): t for s, t in test_random[j].items()}, "checkpoints": []}
         for j in range(n_cases)]
done = 0
for c in checks:
    for p in train[done:c]:
        L.learn(p)
    done = c
    for j in range(n_cases):
        cases[j]["checkpoints"].append({"learned": c, "real": kept(test_real[j]), "scrambled": kept(test_random[j])})
    print(json.dumps({"learned": c, "recall": [(x["checkpoints"][-1]["real"]["recall"], x["checkpoints"][-1]["scrambled"]["recall"])
                                               for x in cases]}), flush=True)
out = os.path.join(ROOT, "experiments", "v3", "results", "chess_case")
os.makedirs(out, exist_ok=True)
json.dump({"seed": seed, "theta": theta, "K": K, "max size": max_size, "cases": cases},
          open(os.path.join(out, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
