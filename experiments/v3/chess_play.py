"""Playing chess from memory: an Elo study of TRELLIS by experience (docs/CHESS_CS.md).

Players (legal moves from python-chess):
  random      a uniformly random legal move
  greedy1     the move that leaves the most material after it (1 ply), random ties
  minimax2    2-ply material minimax, random ties
  trellis-N   learned from N real positions (GraphTrellis), plays the move after which
              memory recognises the position best: the summed recognition bits of the
              parsed position's chunks in both hierarchies (Trellis3._climb, content +
              context); no search, no material
  trellis-N+safe  the same among the moves that do not lose material to the best 1-ply
              reply (if any)
Games: every pairing plays ``games`` games, colours alternating, at most 160 plies; a
game still running is adjudicated by material (a lead of 3+ pawns wins), else drawn.
Elo: Bradley-Terry fitted to all results (draws half), random anchored at 0.

Usage: chess_play.py <cfg json> <seed> <checkpoints> <tag> [games] [opponents]
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
G = int(sys.argv[5]) if len(sys.argv) > 5 else 20
opponents = (sys.argv[6] if len(sys.argv) > 6 else "random,greedy1,minimax2").split(",")
max_size = cfg.pop("max_size", 4)
rng = random.Random(seed)
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board, color):
    return sum(VAL[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values())


def position(board):
    return {(chess.square_file(sq), chess.square_rank(sq)): ("w" if p.color else "b") + p.symbol().upper()
            for sq, p in board.piece_map().items()}


def after(board, mv):
    b = board.copy(stack=False)
    b.push(mv)
    return b


def best_by(board, score):
    moves = list(board.legal_moves)
    rng.shuffle(moves)
    return max(moves, key=score)


def p_random(board):
    return rng.choice(list(board.legal_moves))


def p_greedy1(board):
    me = board.turn
    return best_by(board, lambda m: material(after(board, m), me))


def p_minimax2(board):
    me = board.turn

    def worst(b):
        if b.is_checkmate():
            return 1000
        rep = list(b.legal_moves)
        return min((material(after(b, r), me) for r in rep), default=material(b, me))
    return best_by(board, lambda m: worst(after(board, m)))


def make_trellis(L, safe):
    def bits(b):
        if b.is_checkmate():
            return 1e9
        r = L.parse(position(b))
        tot = 0.0
        for e in r["top"]:
            if e.token is None:
                tot += L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
        return tot

    def play(board):
        me = board.turn
        moves = list(board.legal_moves)
        if safe:
            def loss(m):
                b = after(board, m)
                base = material(b, me)
                return max((base - material(after(b, r), me) for r in b.legal_moves), default=0)
            ok = [m for m in moves if loss(m) <= 0]
            moves = ok or moves
        rng.shuffle(moves)
        return max(moves, key=lambda m: bits(after(board, m)))
    return play


def game(white, black):
    b = chess.Board()
    for _ in range(160):
        if b.is_game_over(claim_draw=True):
            break
        b.push((white if b.turn else black)(b))
    res = b.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return 0.0
    if res == "1/2-1/2":
        return 0.5
    d = material(b, chess.WHITE)
    return 1.0 if d >= 3 else 0.0 if d <= -3 else 0.5


def elo(results, names):
    """Bradley-Terry by minorise-maximise; draws count half a win each way; random = 0."""
    s = {n: 1.0 for n in names}
    wins = defaultdict(float)
    games = defaultdict(float)
    for a, b, r in results:
        wins[a] += r
        wins[b] += 1 - r
        games[(a, b)] += 1
        games[(b, a)] += 1
    for _ in range(500):
        new = {}
        for i in names:
            den = sum(games[(i, j)] / (s[i] + s[j]) for j in names if j != i and games[(i, j)])
            new[i] = (wins[i] + 0.5) / max(den, 1e-12)
        s = new
    ref = s.get("random", 1.0)
    return {n: round(400 * math.log10(s[n] / ref)) for n in names}


players = {"random": p_random, "greedy1": p_greedy1, "minimax2": p_minimax2}
results = []


def match(a, b, pa, pb, n):
    score = 0.0
    for g in range(n):
        if g % 2 == 0:
            r = game(pa, pb)
            results.append((a, b, r))
            score += r
        else:
            r = game(pb, pa)
            results.append((b, a, r))
            score += 1 - r
    return score / n


t0 = time.time()
table = {}
names = list(opponents)
for i, a in enumerate(opponents):
    for b in opponents[i + 1:]:
        table[f"{a} vs {b}"] = match(a, b, players[a], players[b], G)
print(json.dumps({"references": table, "s": round(time.time() - t0)}), flush=True)
path = os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")
positions = C.load_positions(path)
random.Random(seed).shuffle(positions)
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
done = 0
for c in checks:
    for p in positions[done:c]:
        L.learn(p)
    done = c
    L.samples = 0                           # play reads the best analysis only
    for safe in (False, True):
        name = f"trellis-{c}" + ("+safe" if safe else "")
        names.append(name)
        t1 = time.time()
        row = {"player": name, "positions learned": c}
        for o in opponents:
            row[f"score vs {o}"] = round(match(name, o, make_trellis(L, safe), players[o], G), 3)
        row["s"] = round(time.time() - t1)
        row["elo (random = 0)"] = elo(results, names)
        print(json.dumps(row), flush=True)
    L.samples = 16
print(json.dumps({"tag": tag, "seed": seed, "elo (random = 0)": elo(results, names)}))
