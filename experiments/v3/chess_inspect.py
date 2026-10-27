"""Chess chunks at several granularities (docs/CHESS_CS.md): what GraphTrellis
stores and recognises after learning real positions, by chunk size.

Learning as in chess_recall.py (Lichess 1800+, ply 30, same split). Then:

  inventory   committed chunks in memory grouped by size (pieces), and within a
              size by TRELLIS's own content concept (the parent of the chunk's
              content leaf); per concept its count and its most frequent
              instance (pieces on squares, and the relations that joined them)
  recognised  over the held-out real and scrambled positions, the sizes of the
              units recognised at theta (as chess_recall.py; content + context)
  nested      held-out positions' stored analyses as trees: each chunk with its
              pieces, squares and depth, so small chunks can be seen inside large

Writes experiments/v3/results/chess_inspect/<tag>_s<seed>.json; plot with
experiments/v3/chess_inspect_plot.py.

Usage: chess_inspect.py <cfg json> <seed> <n learn> <tag> [theta] [n nested]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis3 import FILL  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed, n_learn, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
theta = float(sys.argv[5]) if len(sys.argv) > 5 else 16.0
n_nested = int(sys.argv[6]) if len(sys.argv) > 6 else 6
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


test_random = [randomize(p) for p in test_real]
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)


def tree(e, sqs):
    """An element as a nested dict: pieces on squares, relations, parts."""
    if e.token is not None:
        (i,) = tuple(e.span)
        return {"piece": e.token, "sq": C.square_name(sqs[i])}
    return {"size": len(e.span), "rel": sorted(set(e.split)),
            "pieces": sorted([C.square_name(sqs[i]), tok] for i, tok in ((i, e_tok(e, i)) for i in e.span)),
            "parts": [tree(e.left, sqs), tree(e.right, sqs)], "rid": e.rid}


def e_tok(e, i):
    while e.token is None:
        e = e.left if i in e.left.span else e.right
    return e.token


# learn, remembering each committed chunk's pieces on squares
inst = {}
t0 = time.time()
for p in train[:n_learn]:
    r = L.learn(p)
    sqs = r["exp"].data[1]
    for k in r["chosen"]:
        e = r["comps"][k]
        if e.rid is not None:
            inst[e.rid] = (tuple(sorted((C.square_name(sqs[i]), e_tok(e, i)) for i in e.span)), tuple(sorted(set(e.split))))
learn_s = time.time() - t0

# inventory: committed chunks by size, then by content concept (parent of the content leaf)
by_size = defaultdict(lambda: defaultdict(list))
for e in L.records:
    if e.token is None and e.cleaf is not None and e.rid in inst \
            and any(attr == FILL and v[0] == "c" for _, attr, v, _ in e.pas):
        concept = (e.cleaf.parent or e.cleaf).id
        by_size[len(e.span)][concept].append(inst[e.rid])
inventory = {}
for size, concepts in sorted(by_size.items()):
    rows = []
    for cid, members in sorted(concepts.items(), key=lambda kv: -len(kv[1]))[:12]:
        top, n_top = Counter(members).most_common(1)[0]
        kinds = Counter(tok for m, _ in members for _, tok in m)
        rows.append({"concept": cid, "count": len(members), "distinct": len(set(members)),
                     "example": [list(x) for x in top[0]], "example rel": list(top[1]), "example count": n_top,
                     "pieces": dict(kinds.most_common(6))})
    inventory[size] = {"committed": sum(len(v) for v in concepts.values()), "concepts": len(concepts), "top": rows}


def bits(e):
    if "climb" not in e.memo:
        e.memo["climb"] = L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
    return e.memo["climb"]


def units(e):
    if e.token is not None or bits(e) >= theta:
        return [e]
    return units(e.left) + units(e.right)


recognised = {}
nested = []
for name, test in (("real", test_real), ("random", test_random)):
    sizes = Counter()
    for j, p in enumerate(test):
        r = L.parse(p)
        sqs = r["exp"].data[1]
        for top in r["top"]:
            for u in units(top):
                sizes[len(u.span)] += 1
        if name == "real" and j < n_nested:
            nested.append({"fen": {C.square_name(s): t for s, t in p.items()},
                           "top": [tree(e, sqs) for e in r["top"]],
                           "recognised": [sorted(C.square_name(sqs[i]) for i in u.span)
                                          for e in r["top"] for u in units(e) if u.token is None]})
    tot = sum(sizes.values())
    recognised[name] = {str(k): round(v / len(test), 3) for k, v in sorted(sizes.items())}

out = {"tag": tag, "seed": seed, "positions learned": n_learn, "max size": max_size, "theta": theta,
       "learn s": round(learn_s), "inventory": inventory, "recognised units per position": recognised, "nested": nested}
os.makedirs(os.path.join(ROOT, "experiments", "v3", "results", "chess_inspect"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "experiments", "v3", "results", "chess_inspect", f"{tag}_s{seed}.json"), "w",
                    encoding="utf-8"), indent=1)
print(json.dumps({k: out[k] for k in ("tag", "seed", "positions learned", "max size", "learn s", "recognised units per position")}))
for size, v in inventory.items():
    print(json.dumps({"size": size, "committed": v["committed"], "concepts": v["concepts"],
                      "top": [(r["count"], r["distinct"], r["example"], r["example rel"]) for r in v["top"][:4]]}))
