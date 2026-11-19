"""Round 217: chess regime judge for the TRELLIS level-of-abstraction study (design r214 section 9, move version).

Regimes for one move in one position (python-chess; no engine: Stockfish is not on the cluster and a download was blocked):
  R-illegal  the move is not legal in the position (nonsense: breaks the rules)
  R-legal    legal but poor (syntax only): its material value (below) is at least POOR (2) pawns under the mover's material now
  R-sem      legal and reasonable: loses less than POOR pawns by that value
Value of a move m for the mover = material(mover) - material(opponent) after m, the opponent's reply that
minimises it and the mover's best recapture (captures only, standing pat allowed) (checkmating the mover = -1000; mating the opponent = +1000; stalemate = 0). Cheap and fixed, as the minimax2
player of chess_play.py. It sees hung pieces and lost exchanges, not positional sense, so R-sem is "does not blunder
material to a 3-ply capture search"; real moves of 1500-1800 players are the calibration (CONTROLS below).

Usage (compute node, repo root; trellis env):  r217_chess_judge.py            builds the test set (cached) and runs the controls
Imports: judge(board, uci) -> dict; load_testset().
"""
import json
import os
import random
import subprocess
import sys

import chess
import chess.pgn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "experiments", "v3", "results", "chess_regimes")
PGN = os.path.join(ROOT, "data", "chess", "lichess_db_standard_rated_2013-01.pgn.zst")
TESTSET = os.path.join(OUT, "r217_testset.json")
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
POOR = 2
PLY = 30          # positions after 30 plies (white to move), as the training positions of data/chess/settings


def material(board, color):
    return sum(VAL[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values())


def value(board, mv):
    """Material value of mv for the mover after the opponent's best reply and the mover's best recapture (captures only,
    standing pat allowed): min over replies r of max(material after r, material after each capture c of the mover)."""
    me = board.turn
    b = board.copy(stack=False)
    b.push(mv)
    if b.is_checkmate():
        return 1000
    reps = list(b.legal_moves)
    if not reps:
        return 0
    worst = 1e9
    for r in reps:
        b.push(r)
        if b.is_checkmate():
            v = -1000
        else:
            v = material(b, me)
            for c in list(b.legal_moves):
                if b.is_capture(c):
                    b.push(c)
                    v = max(v, material(b, me))
                    b.pop()
        b.pop()
        worst = min(worst, v)
    return worst


def values(board):
    return {m.uci(): value(board, m) for m in board.legal_moves}


def judge(board, uci, vals=None):
    """Regime of the move uci (a string) in board. vals = values(board) if already computed."""
    try:
        mv = chess.Move.from_uci(uci)
    except Exception:
        return {"regime": "R-illegal", "why": "unparseable"}
    if mv not in board.legal_moves:
        return {"regime": "R-illegal", "why": "not legal"}
    vals = vals or values(board)
    drop = material(board, board.turn) - vals[uci]
    return {"regime": "R-legal" if drop >= POOR else "R-sem", "drop": drop}


def load_testset(n=300, seed=0):
    """Real (fen, move) pairs at ply PLY from games with min Elo in [1500, 1800) (disjoint from the >= 1800 training games)."""
    if os.path.exists(TESTSET):
        with open(TESTSET, encoding="utf-8") as fh:
            return json.load(fh)
    p = subprocess.Popen(["zstd", "-dc", PGN], stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    rows = []
    while len(rows) < 4 * n:
        g = chess.pgn.read_game(p.stdout)
        if g is None:
            break
        try:
            lo = min(int(g.headers["WhiteElo"]), int(g.headers["BlackElo"]))
        except Exception:
            continue
        if not 1500 <= lo < 1800:
            continue
        b = g.board()
        mvs = list(g.mainline_moves())
        if len(mvs) <= PLY:
            continue
        for m in mvs[:PLY]:
            b.push(m)
        rows.append({"fen": b.fen(), "move": mvs[PLY].uci(), "min_elo": lo})
    p.stdout.close()
    p.kill()
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    os.makedirs(OUT, exist_ok=True)
    with open(TESTSET, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return rows


def wrong_square(board, uci, rng):
    """The real move with its destination shifted to a wrong square (resampled until the result is illegal)."""
    mv = chess.Move.from_uci(uci)
    for _ in range(100):
        to = rng.randrange(64)
        c = chess.Move(mv.from_square, to)
        if to != mv.to_square and c not in board.legal_moves:
            return c.uci()
    return chess.Move(mv.from_square, (mv.to_square + 9) % 64).uci()


def random_pair(board, rng):
    """A random from/to pair; illegal in all but a few cases (resampled until illegal)."""
    for _ in range(100):
        c = chess.Move(rng.randrange(64), rng.randrange(64))
        if c.from_square != c.to_square and c not in board.legal_moves:
            return c.uci()


def controls(rows, seed=0):
    rng = random.Random(seed)
    arms = {k: [] for k in ("real", "random legal", "worst legal", "wrong square", "random pair")}
    for r in rows:
        b = chess.Board(r["fen"])
        v = values(b)
        arms["real"].append(judge(b, r["move"], v)["regime"])
        arms["random legal"].append(judge(b, rng.choice(list(v)), v)["regime"])
        arms["worst legal"].append(judge(b, min(v, key=v.get), v)["regime"])
        arms["wrong square"].append(judge(b, wrong_square(b, r["move"], rng), v)["regime"])
        arms["random pair"].append(judge(b, random_pair(b, rng), v)["regime"])
    out = {}
    for k, l in arms.items():
        out[k] = {reg: round(l.count(reg) / len(l), 3) for reg in ("R-sem", "R-legal", "R-illegal")}
    return out


if __name__ == "__main__":
    rows = load_testset()
    res = {"n": len(rows), "poor_threshold": POOR, "controls": controls(rows)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "r217_judge_controls.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
