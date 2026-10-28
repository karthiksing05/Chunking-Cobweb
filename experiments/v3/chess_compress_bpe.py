"""Baseline for chess_compress_dict.py: a chunk dictionary mined by co-occurrence
alone (byte-pair encoding on piece-square sets), coded the same way.

A position is a set of piece-square items. Each merge takes the pair of items that
occur together in the most training positions and makes it one item (a chunk of
up to ``max_size`` pieces); the training positions are rewritten with it. A position
is coded by applying the merges in order; items are entropy-coded with their usage
frequencies on the training positions (Laplace, plus one stop item per position);
storing the dictionary costs each used chunk written as a piece-square list, as in
chess_compress_dict.py. Reported after 25 .. ``n_merges`` merges.

Usage: chess_compress_bpe.py <seed> <tag> [n_rep n_learn n_train n_test max_size n_merges]
"""
import json
import math
import os
import random
import sys
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2 import chess as C  # noqa: E402

seed, tag = int(sys.argv[1]), sys.argv[2]
n_rep, n_learn, n_train, n_test, max_size, n_merges = [int(x) for x in sys.argv[3:9]] if len(sys.argv) > 8 \
    else [4000, 2000, 2500, 1000, 4, 2000]
fens = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")) if l.strip()]
random.Random(seed).shuffle(fens)
train = [frozenset(((f, r), t) for (f, r), t in C.parse_fen(x).items()) for x in fens[:n_learn]]
held = [frozenset(((f, r), t) for (f, r), t in C.parse_fen(x).items()) for x in fens[n_rep:n_rep + n_train + n_test]]


def single(ps):
    return frozenset([ps])


def apply(pos, merges):
    """The position as items (frozensets of piece-squares) after the merges in order."""
    items = {single(p) for p in pos}
    for a, b in merges:
        if a in items and b in items:
            items -= {a, b}
            items.add(a | b)
    return items


tr = [{single(p) for p in pos} for pos in train]
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


def coder(counts, n_alpha):
    tot = sum(counts.values()) + n_learn + n_alpha + 1
    return (lambda it: -math.log2((counts.get(it, 0) + 1) / tot)), -math.log2((n_learn + 1) / tot)


raw_use = Counter(single(p) for pos in train for p in pos)
cost_r, stop_r = coder(raw_use, 768)
rbits = sum(sum(cost_r(single(p)) for p in pos) + stop_r for pos in held) / len(held)
print(json.dumps({"tag": tag, "seed": seed, "merges made": len(merges), "piece-square bits mean (held out)": round(rbits, 1)}), flush=True)
for k in (25, 50, 100, 200, 400, 800, 1600, n_merges):
    if k > len(merges):
        continue
    M = merges[:k]
    use = Counter(it for pos in train for it in apply(pos, M))
    chunks = [it for it in use if len(it) > 1]
    cost_d, stop_d = coder(use, 768 + len(chunks))
    dict_bits = sum(sum(cost_r(single(p)) for p in it) + stop_r for it in chunks)
    hb = [sum(cost_d(it) for it in apply(pos, M)) + stop_d for pos in held]
    b = sum(hb) / len(hb)
    print(json.dumps({"tag": tag, "merges": k, "chunks": len(chunks), "bits per position": round(b, 1),
                      "dictionary bits per held-out position": round(dict_bits / len(held), 1),
                      "total per position": round(b + dict_bits / len(held), 1),
                      "held-out positions": len(held)}), flush=True)
