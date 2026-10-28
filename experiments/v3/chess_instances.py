"""What a chess instance looks like: after learning ``n`` positions, one held-out
position is parsed and the context and content descriptions of a few of its
elements are printed (a piece, and the chunks of its stored analysis), as stored
in the two Cobweb trees.

Usage: chess_instances.py <cfg json> <seed> <n>
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
pos = C.load_positions(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen"))
random.Random(seed).shuffle(pos)
L = GraphTrellis(seed=seed, **cfg)
for p in pos[:n]:
    L.learn(p)
L._quiet = True
r = L.learn(pos[n])                     # learned, so its instances are complete (as stored)
exp = r["exp"]
sqs = exp.data[1]


def show(d):
    out = {}
    for a, v in d.items():
        if isinstance(v, dict):
            v = {k: round(w, 3) for k, w in sorted(v.items(), key=lambda kv: -kv[1])[:8]}
        out[a] = v
    return out


def where(e):
    return " ".join(f"{exp.tokens[i]}@{C.square_name(sqs[i])}" for i in sorted(e.span))


print("position:", " ".join(f"{t}@{C.square_name(s)}" for t, s in zip(exp.tokens, sqs)))
print("stored analysis (top):", L.chunks(r))
w = r["words"][len(r["words"]) // 2]
print(json.dumps({"element": "piece " + where(w), "context instance": show(w.ctx)}, ensure_ascii=False, indent=1))
chosen = sorted((r["comps"][k] for k in r["chosen"]), key=lambda e: len(e.span))
for e in [chosen[0], chosen[-1]] if chosen else []:
    print(json.dumps({"element": f"chunk of {len(e.span)}: " + where(e), "relations": list(e.split),
                      "content instance": show(e.cnt or {}), "context instance": show(e.ctx or {})},
                     ensure_ascii=False, indent=1))
