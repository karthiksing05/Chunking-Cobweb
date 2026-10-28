"""Chess state compression, stage A (docs/CHESS_CS.md): teacher values and the
representations that need the trellis environment.

Positions: Lichess 1800+, ply 30 (board only; white to move is assumed). Split by
seed: ``n_rep`` positions to learn representations from (unlabelled), then
``n_train`` + ``n_test`` positions labelled by the teacher.

Teacher: alpha-beta to ``depth`` plies with a capture-only quiescence search, on
material plus piece-square tables (python-chess); the value is its score for white
in pawns, clipped to [-10, 10]. It stands in for a strong engine (Stockfish could
not be installed).

Representations written here (the torch learners are stage B):
  raw       768 piece-square indicators
  trellis   GraphTrellis learned on the first ``n_learn`` of the representation
            positions; a position's code is the set of units it is parsed into
            when recognised at ``theta`` (as chess_recall.py): every recognised
            chunk names its content concept ``k`` levels above its leaf, for k in
            ``levels`` (one indicator per concept id and level), and every piece left
            unrecognised is coded raw (its piece-square indicator). Bits: each
            recognised chunk costs -log2 of its concept's share among the training
            positions' recognised chunks (level 0 here meaning the code word), each
            raw piece log2 768.

Writes experiments/v3/results/chess_compress/<tag>_s<seed>.npz (boards, values,
raw and trellis features, trellis bits) and a JSON summary.

Usage: chess_compress_a.py <cfg json> <seed> <tag> [n_rep n_learn n_train n_test depth theta levels]
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed, tag = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
a = [int(x) for x in sys.argv[4:9]] if len(sys.argv) > 8 else [4000, 2000, 2500, 1000, 3]
n_rep, n_learn, n_train, n_test, depth = a
theta = float(sys.argv[9]) if len(sys.argv) > 9 else 16.0
levels = [int(x) for x in (sys.argv[10] if len(sys.argv) > 10 else "1,2").split(",")]
max_size = cfg.pop("max_size", 4)

fens = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")) if l.strip()]
random.Random(seed).shuffle(fens)
rep_fens = fens[:n_rep]
lab_fens = fens[n_rep:n_rep + n_train + n_test]

# ---------------------------------------------------------------- teacher
VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.2, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}
# piece-square bonuses in pawns (simplified tables: centralisation, pawn advance, king shelter)
CENTER = [min(f, 7 - f) + min(r, 7 - r) for r in range(8) for f in range(8)]


def pst(piece, sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if piece.color == chess.BLACK:
        r = 7 - r
    c = CENTER[r * 8 + f] / 6.0                     # 0 edge .. 1 centre
    t = piece.piece_type
    if t == chess.PAWN:
        return 0.05 * r + 0.1 * c * (r >= 3)
    if t in (chess.KNIGHT, chess.BISHOP):
        return 0.3 * c
    if t == chess.QUEEN:
        return 0.1 * c
    if t == chess.KING:
        return 0.2 * (r == 0) - 0.2 * c
    return 0.1 * (r == 6)                            # rook on the seventh


def evaluate(board):
    if board.is_checkmate():
        return -100.0 if board.turn == chess.WHITE else 100.0
    s = 0.0
    for sq, p in board.piece_map().items():
        v = VAL[p.piece_type] + pst(p, sq)
        s += v if p.color == chess.WHITE else -v
    return s


def ordered(board, captures=False):
    moves = [m for m in board.legal_moves if not captures or board.is_capture(m)]

    def key(m):
        if board.is_capture(m):
            victim = board.piece_at(m.to_square)
            return -(VAL[victim.piece_type] if victim else 1.0) * 10 + VAL[board.piece_at(m.from_square).piece_type]
        return 0.0
    return sorted(moves, key=key)


def quiesce(board, alpha, beta, nodes):
    nodes[0] += 1
    stand = evaluate(board) * (1 if board.turn == chess.WHITE else -1)
    if stand >= beta:
        return stand
    alpha = max(alpha, stand)
    for m in ordered(board, captures=True):
        board.push(m)
        v = -quiesce(board, -beta, -alpha, nodes)
        board.pop()
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return alpha


def negamax(board, d, alpha, beta, nodes):
    if d == 0 or board.is_game_over():
        return quiesce(board, alpha, beta, nodes) if not board.is_game_over() else \
            evaluate(board) * (1 if board.turn == chess.WHITE else -1)
    best = -1e9
    for m in ordered(board):
        board.push(m)
        v = -negamax(board, d - 1, -beta, -alpha, nodes)
        board.pop()
        best = max(best, v)
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best


def teacher(fen):
    board = chess.Board(fen + " w - - 0 1")
    nodes = [0]
    v = negamax(board, depth, -1e9, 1e9, nodes)      # white to move: the score is white's
    return max(-10.0, min(10.0, v)), nodes[0]


t0 = time.time()
values, nodes = [], []
for f in lab_fens:
    v, n = teacher(f)
    values.append(v)
    nodes.append(n)
teach_s = time.time() - t0

# ---------------------------------------------------------------- raw
KINDS = sorted(C.TOKENS)


def raw(fen):
    x = np.zeros(768, dtype=np.float32)
    for (f, r), tok in C.parse_fen(fen).items():
        x[KINDS.index(tok) * 64 + r * 8 + f] = 1.0
    return x


boards = np.stack([raw(f) for f in rep_fens + lab_fens])

# ---------------------------------------------------------------- trellis
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
t1 = time.time()
for f in rep_fens[:n_learn]:
    L.learn(C.parse_fen(f))
learn_s = time.time() - t1


def bits(e):
    if "climb" not in e.memo:
        e.memo["climb"] = L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
    return e.memo["climb"]


def units(e):
    if e.token is not None or bits(e) >= theta:
        return [e]
    return units(e.left) + units(e.right)


def code(fen):
    """(concept ids per level of each recognised chunk, raw indices of the rest, leaf ids)"""
    r = L.parse(C.parse_fen(fen))
    sqs = r["exp"].data[1]
    chunks, rest = [], []
    for top in r["top"]:
        for u in units(top):
            if u.token is not None:
                (i,) = tuple(u.span)
                f_, r_ = sqs[i]
                rest.append(KINDS.index(u.token) * 64 + r_ * 8 + f_)
            else:
                path = L.cnt.categorize(L._content(u))
                chunks.append([path[max(0, len(path) - 1 - k)].id for k in [0] + levels])
    return chunks, rest


t2 = time.time()
codes = [code(f) for f in rep_fens[:n_learn] + lab_fens]
code_s = time.time() - t2
train_codes = codes[:n_learn]
vocab = {}
for lv in range(1, len(levels) + 1):
    for ch, _ in train_codes:
        for c in ch:
            vocab.setdefault((lv, c[lv]), len(vocab))
freq = Counter(c[0] for ch, _ in train_codes for c in ch)
tot = sum(freq.values())
n_words = len(freq) + 1
feats = np.zeros((len(lab_fens), len(vocab) + 768), dtype=np.float32)
tbits = np.zeros(len(lab_fens), dtype=np.float32)
n_units = np.zeros(len(lab_fens), dtype=np.float32)
for j, (ch, rest) in enumerate(codes[n_learn:]):
    for c in ch:
        for lv in range(1, len(levels) + 1):
            k = vocab.get((lv, c[lv]))
            if k is not None:
                feats[j, k] += 1.0
        tbits[j] += -math.log2((freq.get(c[0], 0) + 1) / (tot + n_words))
    for i in rest:
        feats[j, len(vocab) + i] = 1.0
        tbits[j] += math.log2(768)
    n_units[j] = len(ch) + len(rest)

out_dir = os.path.join(ROOT, "experiments", "v3", "results", "chess_compress")
os.makedirs(out_dir, exist_ok=True)
np.savez_compressed(os.path.join(out_dir, f"{tag}_s{seed}.npz"), boards=boards, n_rep=n_rep, n_train=n_train,
                    values=np.array(values, dtype=np.float32), trellis=feats, trellis_bits=tbits,
                    trellis_units=n_units, trellis_vocab=len(vocab))
summary = {"tag": tag, "seed": seed, "depth": depth, "teacher s": round(teach_s), "teacher nodes mean": round(float(np.mean(nodes))),
           "value mean": round(float(np.mean(values)), 3), "value sd": round(float(np.std(values)), 3),
           "learn s": round(learn_s), "code s": round(code_s), "max size": max_size, "theta": theta, "levels": levels,
           "trellis vocab": len(vocab), "trellis bits mean": round(float(tbits.mean()), 1),
           "raw bits mean": round(float(boards[n_rep:].sum(1).mean() * math.log2(768)), 1),
           "trellis units mean": round(float(n_units.mean()), 2)}
json.dump(summary, open(os.path.join(out_dir, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(summary), flush=True)
