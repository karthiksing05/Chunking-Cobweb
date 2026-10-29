"""Prequential code for the co-occurrence baseline (chess_compress_bpe.py), to compare
with chess_compress_gen.py under PREQ: each of the first n_learn positions is coded
by what was learned from the positions before it, then learned.

Merges are re-mined from all positions so far at 100, 200, 400, ... positions
(doubling, as BPE is a batch method); item counts under the current merges are
updated after every position. Each position's code is the even mixture of the
codes after each merge count in ``KS`` and the piece-square list (at most log2 of
their number above the best). The sum is the whole description, dictionary
included, with no separate dictionary term.

Usage: chess_compress_bpe_preq.py <seed> <tag> [n_learn max_size]
"""
import json
import math
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

seed, tag = int(sys.argv[1]), sys.argv[2]
n_learn, max_size = [int(x) for x in sys.argv[3:5]] if len(sys.argv) > 4 else [4000, 4]
KS = (50, 100, 200, 400, 800)
fens = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")) if l.strip()]
random.Random(seed).shuffle(fens)
data = [frozenset(((f, r), t) for (f, r), t in C.parse_fen(x).items()) for x in fens[:n_learn]]


def single(ps):
    return frozenset([ps])


def apply(pos, merges):
    items = {single(p) for p in pos}
    for a, b in merges:
        if a in items and b in items:
            items -= {a, b}
            items.add(a | b)
    return items


def mine(train, n_merges):
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
    return merges


def bits(items, use, n_seen, n_alpha):
    tot = sum(use.values()) + n_seen + n_alpha + 1
    return sum(-math.log2((use.get(it, 0) + 1) / tot) for it in items) - math.log2((n_seen + 1) / tot)


merges, codes = [], {}                     # codes: k -> (merges, usage counter)
raw = Counter()
tot, tot_raw, out, t0 = 0.0, 0.0, [], time.time()
marks = {300, 1000, 2000, 4000, n_learn}
for i, pos in enumerate(data):
    if i in (100, 200, 400, 800, 1600, 3200):
        merges = mine(data[:i], max(KS))
        codes = {k: (merges[:k], Counter(it for p in data[:i] for it in apply(p, merges[:k]))) for k in KS if k <= len(merges)}
    rb = bits([single(p) for p in pos], raw, i, 768)
    costs = [rb] + [bits(apply(pos, M), use, i, 768 + sum(1 for it in use if len(it) > 1)) for M, use in codes.values()]
    m = min(costs)
    b = m - math.log2(sum(2.0 ** (m - c) for c in costs) / len(costs))
    tot, tot_raw = tot + b, tot_raw + rb
    raw.update(single(p) for p in pos)
    for M, use in codes.values():
        use.update(apply(pos, M))
    if i + 1 in marks:
        row = {"positions": i + 1, "total bits": round(tot), "mean bits per position": round(tot / (i + 1), 1),
               "piece-square total bits": round(tot_raw), "piece-square mean": round(tot_raw / (i + 1), 1),
               "merges": len(merges), "s": round(time.time() - t0)}
        out.append(row)
        print(json.dumps(row), flush=True)
out_dir = os.path.join(ROOT, "experiments", "v3", "results", "chess_compress")
json.dump({"tag": tag, "seed": seed, "merge counts mixed": KS, "curve": out},
          open(os.path.join(out_dir, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
