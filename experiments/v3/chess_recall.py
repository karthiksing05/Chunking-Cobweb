"""Chase & Simon (1973) with TRELLIS: brief-exposure recall of real vs random
positions, as a function of experience (docs/CHESS_CS.md).

Learning sees real positions (Lichess 1800+, after ply 30). At checkpoints (0 =
novice ... N = expert) memory is frozen and shown held-out positions once:

  real     a held-out game position
  random   the same pieces on random distinct squares (pawns off ranks 1 and 8)

Exposure: the position is parsed into a forest of units (chunks or single pieces).
Recognition (CHREST-like): a chunk counts as one unit only if long-term memory
recognises it: its descriptions are at least ``theta`` bits more probable under
the deepest concept on their paths holding 3+ instances than under the roots
(Trellis3._climb, content plus context; recognition="content": content only); an unfamiliar chunk falls apart into its parts until they are
familiar or single pieces. Reported for several theta. Short-term memory holds ``K`` units (7, Chase
& Simon's chunk span); the largest are kept. Recall: the share of the position's
pieces inside the kept units (each placed on its square). Also reported: mean
size of the recognised chunks and the number of units needed for the whole board.

Usage: chess_recall.py <cfg json> <seed> <checkpoints> <tag> [K] [thetas] [n test]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 7
thetas = [float(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "0,2,4,8").split(",")]
n_test = int(sys.argv[7]) if len(sys.argv) > 7 else 100
max_size = cfg.pop("max_size", 4)
RECOG = cfg.pop("recognition", "both")      # "both" hierarchies, or "content" only

path = os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")
positions = C.load_positions(path)
random.Random(seed).shuffle(positions)
test_real = positions[-n_test:]
train = positions[:-n_test]
rng = random.Random(seed + 1)


def randomize(pos):
    pieces = list(pos.values())
    squares = [(f, r) for f in range(8) for r in range(8)]
    while True:
        rng.shuffle(squares)
        out = dict(zip(squares, pieces))
        if all(not (p[1] == "P" and sq[1] in (0, 7)) for sq, p in out.items()):
            return out


test_random = [randomize(p) for p in test_real]
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)


def bits(e) -> float:
    """Recognition in both hierarchies: what the chunk is made of (content) and
    where it stands (context: its anchor square and the pieces along its rays)."""
    if "climb" not in e.memo:
        if L.cnt.root.count <= 0:
            e.memo["climb"] = -1e9
        else:
            e.memo["climb"] = L._climb(L.cnt, L._content(e)) + (L._climb(L.ctx, e.ctx) if RECOG == "both" else 0.0)
    return e.memo["climb"]


def units(e, theta):
    """Recognised units of an element: itself if familiar, else its parts' units."""
    if e.token is not None or bits(e) >= theta:
        return [e]
    return units(e.left, theta) + units(e.right, theta)


def recall(r, pos, theta):
    us = [u for e in r["top"] for u in units(e, theta)]
    sizes = sorted((len(u.span) for u in us), reverse=True)
    n = len(pos)
    kept = sum(sizes[:K])
    chunks = [s for s in sizes if s > 1]
    return {"recall": kept / n, "units for the board": len(us),
            "mean chunk size": sum(chunks) / len(chunks) if chunks else 1.0,
            "pieces in chunks": sum(chunks) / n}


t0, done, rows = time.time(), 0, []
for c in checks:
    for p in train[done:c]:
        L.learn(p)
    done = c
    learn_s = time.time() - t0
    row = {"tag": tag, "seed": seed, "positions learned": c, "learn s": round(learn_s), "K": K}
    for name, test in (("real", test_real), ("random", test_random)):
        parses = [(L.parse(p), p) for p in test]
        for th in thetas:
            rs = [recall(r, p, th) for r, p in parses]
            for k in rs[0]:
                row[f"{name} theta {th:g}: {k}"] = round(sum(x[k] for x in rs) / len(rs), 3)
    print(json.dumps(row), flush=True)
    rows.append(row)
os.makedirs(os.path.join(ROOT, "experiments", "v3", "results", "chess_recall"), exist_ok=True)
json.dump(rows, open(os.path.join(ROOT, "experiments", "v3", "results", "chess_recall", f"{tag}_s{seed}.json"), "w"))
