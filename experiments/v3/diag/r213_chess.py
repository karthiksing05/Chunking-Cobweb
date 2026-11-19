"""Round 213 (chess): chunk inventory of the cs73L memory (long-stream Chase and Simon recipe) per seed, along the stream.

Same memory as chess_cs1973_exact.py with the cs73L config (logs/v3_cs73L_submit.sh): GraphTrellis, train_long.fen shuffled by
the seed, learned in order. Only the learning loop runs here (no recall trials), to the first max(checkpoints) positions of the
seed's stream (cs73L ran 27,459; learning cost grows with the stream, so this stops earlier; for seeds 13 and 17 the 4,000
positions are the same ones cs73L learned first).

Per seed it saves (experiments/v3/results/chess_inspect/r213_chess_s<seed>.json):
  chunks   committed chunk (pieces on squares, sorted "e4:wP|f4:wP") -> stream indices where it was a chosen chunk (size >= 2)
  pairs    frequency baseline: the commonest piece pairs within Chebyshev distance <= 2 by raw count at each checkpoint
  boards   decomposition of fixed boards (first 20 of middle.fen, first 10 of end.fen) into recognised units (theta 16, as
           chess_inspect.py) with each unit's pieces, size, whether it is a stored chunk
Usage (compute node, repo root): r213_chess.py <seed> <checkpoints e.g. 500,1000,2000,4000> [max_size]
"""
import json
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

seed = int(sys.argv[1])
checks = [int(x) for x in sys.argv[2].split(",")]
CFG = json.loads('{"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,'
                 '"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"any"}')
MAX_SIZE, THETA = 4, 16.0
D = os.path.join(ROOT, "data", "chess", "cs1973")
OUT = os.path.join(ROOT, "experiments", "v3", "results", "chess_inspect")


def load(name):
    with open(os.path.join(D, f"{name}.fen"), encoding="utf-8") as fh:
        return [C.parse_fen(l.strip()) for l in fh if l.strip()]


train = load("train_long")
random.Random(seed).shuffle(train)
boards = [("middle", p) for p in load("middle")[:20]] + [("end", p) for p in load("end")[:10]]
L = GraphTrellis(max_size=MAX_SIZE, seed=seed, **CFG)


def key_of(pieces):
    return "|".join(f"{C.square_name(sq)}:{tok}" for sq, tok in sorted(pieces))


def e_tok(e, i):
    while e.token is None:
        e = e.left if i in e.left.span else e.right
    return e.token


def bits(e):
    if "climb" not in e.memo:
        e.memo["climb"] = L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
    return e.memo["climb"]


def units(e):
    if e.token is not None or bits(e) >= THETA:
        return [e]
    return units(e.left) + units(e.right)


def pairs_of(pos):
    items = sorted(pos.items())
    out = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            (s1, t1), (s2, t2) = items[a], items[b]
            if max(abs(s1[0] - s2[0]), abs(s1[1] - s2[1])) <= 2:
                out.append(key_of([(s1, t1), (s2, t2)]))
    return out


chunks, pair_counts, stored = {}, Counter(), set()
res = {"seed": seed, "checkpoints": checks, "cfg": CFG, "max_size": MAX_SIZE, "theta": THETA, "chunks": {}, "pairs": {}, "boards": {}, "info": {}}
t0, done = time.time(), 0
for cp in checks:
    L._quiet = False
    for idx in range(done, cp):
        p = train[idx]
        r = L.learn(p)
        sqs = r["exp"].data[1]
        for e in r["comps"].values():
            if e.rid is not None and e.token is None:
                stored.add(key_of([(sqs[i], e_tok(e, i)) for i in e.span]))
        for k in r["chosen"]:
            e = r["comps"][k]
            if e.rid is not None and e.token is None:
                chunks.setdefault(key_of([(sqs[i], e_tok(e, i)) for i in e.span]), []).append(idx)
        pair_counts.update(set(pairs_of(p)))
    done = cp
    res["pairs"][str(cp)] = pair_counts.most_common(3000)
    res["info"][str(cp)] = {"records": len(L.records), "stored chunks": len(stored), "chosen distinct": len(chunks), "learn s": round(time.time() - t0)}
    # decomposition of the fixed boards with learning off
    L._quiet = True
    bd = []
    for kind, p in boards:
        r = L.parse(p)
        sqs = r["exp"].data[1]
        us = []
        for top in r["top"]:
            for u in units(top):
                pcs = [(sqs[i], e_tok(u, i)) for i in u.span]
                us.append({"size": len(u.span), "pieces": sorted([C.square_name(sq), tok] for sq, tok in pcs),
                           "stored": key_of(pcs) in stored, "bits": None if u.token is not None else round(bits(u), 1)})
        bd.append({"kind": kind, "fen": {C.square_name(s): t for s, t in p.items()}, "units": us})
    L._quiet = False
    res["boards"][str(cp)] = bd
    res["chunks"] = {k: v for k, v in chunks.items() if len(v) >= 2}
    print(json.dumps({"seed": seed, "cp": cp, **res["info"][str(cp)]}), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"r213_chess_s{seed}.json"), "w", encoding="utf-8"))
print("done", seed, round(time.time() - t0))
