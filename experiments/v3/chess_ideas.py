"""How chess chunking changes with learning: which chess ideas the chunks encode.

GraphTrellis learns real positions (Lichess 1800+, ply 30, the split of chess_inspect.py).
At each checkpoint it records:

  window     every chunk committed since the previous checkpoint, labelled with the
             chess ideas it contains (rules below, colour-relative: Black is mirrored)
  chance     the same labels on size-matched groups of pieces from the same positions:
             "random" (any pieces) and "nearby" (grown piece by piece, each within two
             squares of one already taken), so a lift > 1 means memory prefers the idea
  held-out   the chunks recognised (bits >= theta, as chess_inspect.py) in held-out real
             positions, labelled the same way: what the player sees at this checkpoint
  top        the most frequent chunks committed in the window, with their labels
  cases      a few held-out positions with their recognised chunks at every checkpoint

Ideas (own = one colour; ranks relative to that colour):
  pawn chain        two own pawns diagonally adjacent
  pawn phalanx      two own pawns side by side
  doubled pawns     two own pawns on one file
  pawn ram          an own pawn blocked by an enemy pawn directly in front
  king shelter      own king with 2+ own pawns on the three files around it, 1-2 ranks ahead
  castled king      own king on g1/h1 or a1/b1/c1 with an own rook or pawn on files f-h / a-c, ranks 1-2
  fianchetto        own bishop on b2/g2 with an own pawn on b3/g3
  pawn-guarded      an own piece (not a pawn) defended by an own pawn
  battery           own Q/R pair on one file or rank, or Q/B pair on one diagonal, nothing between in the chunk
  undeveloped       2+ own pieces (not pawns) on their starting squares
  attack            a piece in the chunk attacks an enemy piece in the chunk (line clear within the chunk)

Usage: chess_ideas.py <cfg json> <seed> <checkpoints, e.g. 100,300,1000,2000,4000> <tag> [theta n_cases]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checkpoints = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
theta = float(sys.argv[5]) if len(sys.argv) > 5 else 16.0
n_cases = int(sys.argv[6]) if len(sys.argv) > 6 else 4
max_size = cfg.pop("max_size", 4)
n_test = 100

positions = C.load_positions(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen"))
random.Random(seed).shuffle(positions)
test_real, train = positions[-n_test:], positions[:-n_test]
rng = random.Random(seed + 2)
IDEAS = ["pawn chain", "pawn phalanx", "doubled pawns", "pawn ram", "king shelter", "castled king", "fianchetto",
         "pawn-guarded", "battery", "undeveloped", "attack"]
HOME = {"R": {(0, 0), (7, 0)}, "N": {(1, 0), (6, 0)}, "B": {(2, 0), (5, 0)}, "Q": {(3, 0)}, "K": {(4, 0)}}


def rel(pieces, colour):
    """The pieces with ranks turned so that ``colour`` plays up the board: [(f, r, kind, own)]."""
    out = []
    for (f, r), tok in pieces:
        rr = r if colour == "w" else 7 - r
        out.append((f, rr, tok[1], tok[0] == colour))
    return out


def clear(pcs, a, b):
    """No piece of the chunk strictly between squares a and b (on a line)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max(abs(dx), abs(dy))
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    occ = {(f, r) for f, r, _, _ in pcs}
    return all((a[0] + sx * k, a[1] + sy * k) not in occ for k in range(1, n))


def ideas(pieces):
    """The set of ideas in a group of pieces [((file, rank), token)]."""
    found = set()
    for colour in "wb":
        pcs = rel(pieces, colour)
        own = [(f, r, k) for f, r, k, o in pcs if o]
        foe = [(f, r, k) for f, r, k, o in pcs if not o]
        pawns = {(f, r) for f, r, k in own if k == "P"}
        fpawns = {(f, r) for f, r, k in foe if k == "P"}
        for f, r in pawns:
            if (f + 1, r + 1) in pawns or (f - 1, r + 1) in pawns:
                found.add("pawn chain")
            if (f + 1, r) in pawns:
                found.add("pawn phalanx")
            if any(g == f and s != r for g, s in pawns):
                found.add("doubled pawns")
            if (f, r + 1) in fpawns:
                found.add("pawn ram")
        for f, r, k in own:
            if k == "K":
                shelter = [p for p in pawns if abs(p[0] - f) <= 1 and 1 <= p[1] - r <= 2]
                if len(shelter) >= 2:
                    found.add("king shelter")
                if r == 0 and f in (6, 7) and any(g >= 5 and s <= 1 and kk in "RP" for g, s, kk in own if kk != "K"):
                    found.add("castled king")
                if r == 0 and f in (0, 1, 2) and any(g <= 2 and s <= 1 and kk in "RP" for g, s, kk in own if kk != "K"):
                    found.add("castled king")
            if k == "B" and (f, r) in ((1, 1), (6, 1)) and (f, 2) in pawns:
                found.add("fianchetto")
            if k != "P" and ((f - 1, r - 1) in pawns or (f + 1, r - 1) in pawns):
                found.add("pawn-guarded")
        heavy = [(f, r, k) for f, r, k in own if k in "QRB"]
        for i, (f, r, k) in enumerate(heavy):
            for g, s, kk in heavy[i + 1:]:
                line = f == g or r == s
                diag = abs(f - g) == abs(r - s)
                if ((line and {k, kk} <= {"Q", "R"}) or (diag and {k, kk} == {"Q", "B"})) and clear(pcs, (f, r), (g, s)):
                    found.add("battery")
        if sum(1 for f, r, k in own if k in HOME and (f, r) in HOME[k]) >= 2:
            found.add("undeveloped")
    for (sa, ta) in pieces:
        for (sb, tb) in pieces:
            if ta[0] != tb[0] and C.hits(ta, sb[0] - sa[0], sb[1] - sa[1]):
                pcs = rel(pieces, "w")
                if ta[1] in "NKP" or clear(pcs, sa, sb):
                    found.add("attack")
    return found


def chance(pos, size, nearby):
    """A size-matched group of the position's pieces: any, or grown within two squares."""
    items = list(pos.items())
    if not nearby:
        return rng.sample(items, min(size, len(items)))
    group = [rng.choice(items)]
    while len(group) < size:
        cand = [it for it in items if it not in group
                and any(max(abs(it[0][0] - g[0][0]), abs(it[0][1] - g[0][1])) <= 2 for g in group)]
        if not cand:
            break
        group.append(rng.choice(cand))
    return group


def e_tok(e, i):
    while e.token is None:
        e = e.left if i in e.left.span else e.right
    return e.token


def rate(groups):
    n = len(groups)
    c = Counter(x for g in groups for x in ideas(g))
    return {k: round(c[k] / n, 4) if n else None for k in IDEAS}, n


def bits(e):
    if "climb" not in e.memo:
        e.memo["climb"] = L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
    return e.memo["climb"]


def units(e):
    if e.token is not None or bits(e) >= theta:
        return [e]
    return units(e.left) + units(e.right)


L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
out = {"tag": tag, "seed": seed, "theta": theta, "max size": max_size, "ideas": IDEAS, "checkpoints": []}
done, t0 = 0, time.time()
cases = {j: [] for j in range(n_cases)}
for cp in checkpoints:
    window, ch_rand, ch_near = [], [], []
    for p in train[done:cp]:
        r = L.learn(p)
        sqs = r["exp"].data[1]
        for k in r["chosen"]:
            e = r["comps"][k]
            if e.token is None and e.rid is not None:
                g = sorted((sqs[i], e_tok(e, i)) for i in e.span)
                window.append(g)
                ch_rand.append(chance(p, len(g), False))
                ch_near.append(chance(p, len(g), True))
    done = cp
    L._quiet = True
    held = []
    for j, p in enumerate(test_real):
        r = L.parse(p)
        sqs = r["exp"].data[1]
        rec = [sorted((sqs[i], e_tok(u, i)) for i in u.span) for top in r["top"] for u in units(top) if u.token is None]
        held += rec
        if j < n_cases:
            cases[j].append({"learned": cp, "chunks": [{"pieces": [[C.square_name(s), t] for s, t in g],
                                                        "ideas": sorted(ideas(g))} for g in sorted(rec, key=len, reverse=True)]})
    L._quiet = False
    rw, nw = rate(window)
    rr, _ = rate(ch_rand)
    rn, _ = rate(ch_near)
    rh, nh = rate(held)
    top = Counter(tuple(g) for g in window).most_common(15)
    row = {"learned": cp, "seconds": round(time.time() - t0), "window chunks": nw,
           "mean size": round(sum(map(len, window)) / max(1, nw), 3),
           "size counts": dict(Counter(map(len, window))),
           "rate": rw, "chance random": rr, "chance nearby": rn,
           "lift nearby": {k: (round(rw[k] / rn[k], 3) if rw[k] is not None and rn[k] else None) for k in IDEAS},
           "held-out recognised per position": round(nh / n_test, 3),
           "held-out mean size": round(sum(map(len, held)) / max(1, nh), 3), "held-out rate": rh,
           "top": [{"pieces": [[C.square_name(s), t] for s, t in g], "count": c, "ideas": sorted(ideas(list(g)))}
                   for g, c in top]}
    out["checkpoints"].append(row)
    print(json.dumps({k: row[k] for k in ("learned", "seconds", "window chunks", "mean size", "held-out recognised per position")}), flush=True)
    print(json.dumps({"rate": rw, "lift nearby": row["lift nearby"]}), flush=True)
out["cases"] = [{"fen": {C.square_name(s): t for s, t in test_real[j].items()}, "by checkpoint": cases[j]} for j in cases]
d = os.path.join(ROOT, "experiments", "v3", "results", "chess_ideas")
os.makedirs(d, exist_ok=True)
json.dump(out, open(os.path.join(d, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
print("wrote", os.path.join(d, f"{tag}_s{seed}.json"))
