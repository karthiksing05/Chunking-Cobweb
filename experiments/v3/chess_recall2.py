"""Chase & Simon with reproducible chunks: chess_recall.py counts every recognised
chunk as recalled, but a chunk helps recall only if memory can put its pieces back.
Here a recognised chunk counts only when long-term memory holds an identical chunk
(the same pieces on the same squares), and the same test is run on chunks mined by
co-occurrence alone (CHREST-style recall with a byte-pair dictionary).

Arms, each on real held-out positions and on the same pieces on random squares:
  recognised  as chess_recall.py: a chunk is one unit when recognised at ``theta``
  verified    recognised at ``theta`` and an identical chunk is stored; otherwise
              it falls apart into its parts
  stored      any chunk of the parse with an identical stored chunk (no theta)
  bpe         the position covered by byte-pair merges learned from the same
              training positions (merges of pairs that co-occur in 2+ positions,
              chunks of up to ``max_size`` pieces), applied in order
Short-term memory keeps the ``K`` largest units; recall = share of the position's
pieces inside them.

Usage: chess_recall2.py <cfg json> <seed> <checkpoints> <tag> [K] [thetas] [n test] [bpe merges]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 7
thetas = [float(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "8,12,16").split(",")]
n_test = int(sys.argv[7]) if len(sys.argv) > 7 else 100
n_merges = int(sys.argv[8]) if len(sys.argv) > 8 else 2000
max_size = cfg.pop("max_size", 4)

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
store = set()


def key(e, exp):
    sqs = exp.data[1]
    return frozenset((sqs[i], exp.tokens[i]) for i in e.span)


def bits(e) -> float:
    if "climb" not in e.memo:
        e.memo["climb"] = -1e9 if L.cnt.root.count <= 0 else L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
    return e.memo["climb"]


def units(e, exp, theta, need_stored):
    if e.token is not None:
        return [1]
    ok = (theta is None or bits(e) >= theta) and (not need_stored or key(e, exp) in store)
    if ok:
        return [len(e.span)]
    return units(e.left, exp, theta, need_stored) + units(e.right, exp, theta, need_stored)


def recall_of(sizes, n):
    sizes = sorted(sizes, reverse=True)
    chunks = [s for s in sizes if s > 1]
    return {"recall": sum(sizes[:K]) / n, "pieces in chunks": sum(chunks) / n,
            "mean chunk size": sum(chunks) / len(chunks) if chunks else 1.0}


def bpe(train_pos):
    tr = [{frozenset([x]) for x in p.items()} for p in train_pos]
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for items in tr:
            small = [i for i in items if len(i) < max_size]
            for a, b in combinations(sorted(small, key=sorted), 2):
                if len(a) + len(b) <= max_size:
                    pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), c = pairs.most_common(1)[0]
        if c < 2:
            break
        merges.append((a, b))
        for items in tr:
            if a in items and b in items:
                items -= {a, b}
                items.add(a | b)
    return merges


def bpe_sizes(pos, merges):
    items = {frozenset([x]) for x in pos.items()}
    for a, b in merges:
        if a in items and b in items:
            items -= {a, b}
            items.add(a | b)
    return [len(i) for i in items]


def mean(rs):
    return {k: round(sum(x[k] for x in rs) / len(rs), 3) for k in rs[0]}


t0, done = time.time(), 0
rows = []
for c in checks:
    for p in train[done:c]:
        r = L.learn(p)
        for k in r["chosen"]:
            store.add(key(r["comps"][k], r["exp"]))
    done = c
    row = {"tag": tag, "seed": seed, "positions learned": c, "learn s": round(time.time() - t0), "K": K,
           "stored chunks": len(store)}
    t1 = time.time()
    merges = bpe(train[:c]) if c > 0 else []
    row["bpe merges"], row["bpe s"] = len(merges), round(time.time() - t1)
    for name, test in (("real", test_real), ("random", test_random)):
        parses = [L.parse(p) for p in test]
        for th in thetas:
            for arm, need in (("recognised", False), ("verified", True)):
                rs = [recall_of([s for e in r["top"] for s in units(e, r["exp"], th, need)], len(p)) for r, p in zip(parses, test)]
                for k, v in mean(rs).items():
                    row[f"{name} {arm} theta {th:g}: {k}"] = v
        rs = [recall_of([s for e in r["top"] for s in units(e, r["exp"], None, True)], len(p)) for r, p in zip(parses, test)]
        for k, v in mean(rs).items():
            row[f"{name} stored: {k}"] = v
        rs = [recall_of(bpe_sizes(p, merges), len(p)) for p in test]
        for k, v in mean(rs).items():
            row[f"{name} bpe: {k}"] = v
    print(json.dumps(row), flush=True)
    rows.append(row)
os.makedirs(os.path.join(ROOT, "experiments", "v3", "results", "chess_recall"), exist_ok=True)
json.dump(rows, open(os.path.join(ROOT, "experiments", "v3", "results", "chess_recall", f"{tag}_s{seed}.json"), "w"))
