"""Chase & Simon (1973), "Perception in chess", replicated with TRELLIS memories.

Stimuli as in the paper: middle-game positions after White's 21st move with 24-26
pieces, end-game positions after White's 41st move with 12-15 pieces, from games
between players rated 1,800+ (Lichess, January 2013), and random positions (the same
pieces on random squares). Training positions come from other games, at a random ply
from 20 to 100. Players are memories after different amounts of learning: B (300
positions), A (1,000) and M (4,000), with 0 as a reference.

Memory task (Figs. 1-2): a 5-s view lets short-term memory take K units (K = 7); a
unit is a chunk of the parse whose exact pieces are stored, else its pieces singly,
the largest first. On every further trial the player keeps the previous
reconstruction and takes K new units, ranked by the pieces they add (the paper:
"M typically added about four more pieces to his previous reconstruction").

Chunks (Table 8): sizes of the successive units recalled on trial 1, largest first,
and units per trial. Relations (Tables 4-6): the chess relations between pieces
recalled in one chunk (C&S: less than 2 s apart) and between pieces of successive
chunks (more than 2 s), against all pairs on the board (a priori): attack (A), defense
(D), proximity (P), same colour (C), same piece type (S). Fig. 5 has no counterpart
(no latencies); its proxy is the share of piece pairs that one chunk holds, by the
number of relations between them.

Usage: chess_cs1973.py <cfg json> <seed> <tag> [n_train n_test K]
"""
from __future__ import annotations

import io
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed, tag = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
n_train, n_test, K = [int(x) for x in sys.argv[4:7]] if len(sys.argv) > 6 else [4000, 100, 7]
max_size = cfg.pop("max_size", 4)
PLAYERS = [("novice", 0), ("B", 300), ("A", 1000), ("M", 4000)]
TRIALS = 7
D = os.path.join(ROOT, "data", "chess", "cs1973")


def build_sets():
    """Middle games, end games and training positions from the Lichess PGN (cached)."""
    import chess
    import chess.pgn
    os.makedirs(D, exist_ok=True)
    pgn = os.path.join(ROOT, "data", "chess", "lichess_db_standard_rated_2013-01.pgn.zst")
    proc = subprocess.Popen(["zstd", "-dc", pgn], stdout=subprocess.PIPE)
    stream = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
    rng = random.Random(0)
    mid, end, train = [], [], []
    while len(mid) < 1000 or len(end) < 1000 or len(train) < 8000:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        try:
            if min(int(g.headers.get("WhiteElo", 0)), int(g.headers.get("BlackElo", 0))) < 1800:
                continue
        except ValueError:
            continue
        b, fens = g.board(), []
        for mv in g.mainline_moves():
            b.push(mv)
            fens.append(b.board_fen())
        role = rng.random()
        if role < 0.5:                                   # a test game
            if len(fens) >= 41 and 24 <= len(chess.Board(fens[40] + " w - - 0 1").piece_map()) <= 26 and len(mid) < 1000:
                mid.append(fens[40])
            elif len(fens) >= 81 and 12 <= len(chess.Board(fens[80] + " w - - 0 1").piece_map()) <= 15 and len(end) < 1000:
                end.append(fens[80])
        elif len(fens) >= 20 and len(train) < 8000:      # a training game: one position from it
            train.append(fens[rng.randrange(19, min(len(fens), 100))])
    proc.kill()
    for name, fs in (("middle", mid), ("end", end), ("train", train)):
        with open(os.path.join(D, f"{name}.fen"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(fs) + "\n")


if not os.path.exists(os.path.join(D, "train.fen")):
    build_sets()


def load(name, n):
    return [C.parse_fen(l.strip()) for l in open(os.path.join(D, f"{name}.fen"), encoding="utf-8") if l.strip()][:n]


rng = random.Random(seed)
train = load("train", 8000)
rng.shuffle(train)
train = train[:n_train]
tests = {"middle": load("middle", n_test), "end": load("end", n_test)}


def randomize(pos):
    pieces = list(pos.values())
    squares = [(f, r) for f in range(8) for r in range(8)]
    while True:
        rng.shuffle(squares)
        out = dict(zip(squares, pieces))
        if all(not (p[1] == "P" and sq[1] in (0, 7)) for sq, p in out.items()):
            return out


tests["random middle"] = [randomize(p) for p in tests["middle"]]
tests["random end"] = [randomize(p) for p in tests["end"]]


# -------------------------------------------------------------- relations
def relations(pos, a, b):
    """C&S's code for two pieces: A attack, D defense, P proximity, C colour, S type."""
    pa, pb = pos[a], pos[b]
    same = pa[0] == pb[0]
    hit = any(t == b and (fwd != "sees" or back) for t, fwd, back in C.interactions(pos, a))
    code = ""
    if hit and not same:
        code += "A"
    if hit and same:
        code += "D"
    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1:
        code += "P"
    if same:
        code += "C"
    if pa[1] == pb[1]:
        code += "S"
    return code or "-"


# -------------------------------------------------------------- memory
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
store = set()


def key(e, exp):
    sqs = exp.data[1]
    return frozenset((sqs[i], exp.tokens[i]) for i in e.span)


def units_of(e, exp):
    """A stored chunk is one unit; otherwise its parts (down to single pieces)."""
    sqs = exp.data[1]
    if e.token is not None or key(e, exp) in store:
        return [frozenset(sqs[i] for i in e.span)]
    if e.left is None:
        return [frozenset([sqs[i]]) for i in e.span]
    return units_of(e.left, exp) + units_of(e.right, exp)


def trials(units, n):
    """Pieces placed after each trial: K new units a view, the previous reconstruction kept."""
    placed, out, first = set(), [], None
    for t in range(TRIALS):
        ranked = sorted(units, key=lambda u: -len(u - placed))
        take = [u for u in ranked[:K] if u - placed]
        if t == 0:
            first = take
        for u in take:
            placed |= u
        out.append(len(placed))
    return out, first


def analyse(pos):
    r = L.parse(pos)
    units = [u for e in r["top"] for u in units_of(e, r["exp"])]
    curve, first = trials(units, len(pos))
    within, between = Counter(), Counter()
    for u in first:
        for a, b in combinations(sorted(u), 2):
            within[relations(pos, a, b)] += 1
    for u, v in zip(first, first[1:]):
        for a in u:
            for b in v:
                between[relations(pos, a, b)] += 1.0 / (len(u) * len(v))
    unit_of = {sq: i for i, u in enumerate(units) for sq in u}
    prior, same_by_n = Counter(), Counter()
    for a, b in combinations(sorted(pos), 2):
        code = relations(pos, a, b)
        prior[code] += 1
        n_rel = len(code.replace("-", ""))
        same_by_n[(n_rel, unit_of[a] == unit_of[b])] += 1
    return {"curve": curve, "sizes": sorted((len(u) for u in first), reverse=True), "within": within,
            "between": between, "prior": prior, "same": same_by_n}


def summarise(rows):
    n = len(rows)
    out = {"pieces per trial": [round(sum(r["curve"][t] for r in rows) / n, 2) for t in range(TRIALS)],
           "successive chunk sizes": [round(sum(r["sizes"][i] for r in rows if len(r["sizes"]) > i) /
                                            max(1, sum(1 for r in rows if len(r["sizes"]) > i)), 2) for i in range(8)],
           "chunks per trial": round(sum(len(r["sizes"]) for r in rows) / n, 2),
           "multi-piece chunks per trial": round(sum(sum(1 for s in r["sizes"] if s > 1) for r in rows) / n, 2)}
    for part in ("within", "between", "prior"):
        tot = Counter()
        for r in rows:
            tot.update(r[part])
        z = sum(tot.values()) or 1
        out[f"relations {part}"] = {k: round(v / z, 4) for k, v in sorted(tot.items())}
    same = Counter()
    for r in rows:
        same.update(r["same"])
    out["share in one chunk by relations"] = {k: round(same[(k, True)] / max(1, same[(k, True)] + same[(k, False)]), 4)
                                              for k in range(5)}
    return out


t0, done, res = time.time(), 0, {"tag": tag, "seed": seed, "K": K, "positions tested": n_test, "players": {}}
for name, c in PLAYERS:
    for p in train[done:c]:
        r = L.learn(p)
        for k in r["chosen"]:
            store.add(key(r["comps"][k], r["exp"]))
    done = c
    L._quiet = True
    res["players"][name] = {"positions learned": c, "stored chunks": len(store),
                            **{s: summarise([analyse(p) for p in ps]) for s, ps in tests.items()}}
    L._quiet = False
    print(json.dumps({"player": name, "learned": c, "s": round(time.time() - t0),
                      **{s: res["players"][name][s]["pieces per trial"][:3] for s in tests}}), flush=True)
out_dir = os.path.join(ROOT, "experiments", "v3", "results", "chess_recall")
os.makedirs(out_dir, exist_ok=True)
json.dump(res, open(os.path.join(out_dir, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(res), flush=True)
