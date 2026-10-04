"""A first test of playing chess from the chess grammar (hypothesis 1 of
``docs/V2_DESIGN.md``, "typical positions are good positions"): play the
legal move after which the position has the shortest code.

The positions are those of ``run_chess.py`` (Lichess, January 2013, both
players rated 1800+, the position after ply 30), with the move played next.
The grammar's code is the structure search's plain code: the read's context
chosen on the 4,000 training positions, the chunk moves found on them
replayed on each candidate position. Reported on the 500 held-out positions:
how often each rule picks the move that was played (top 1 and top 3),
against a random legal move and a rule that captures the most valuable piece.

Usage:
    python experiments/v2/run_chess_play.py --extract     # the moves played next (needs python-chess, zstd)
    python experiments/v2/run_chess_play.py --out experiments/v2/results/chess_play
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2.chess import (BoardSearch, default_positions_path, forward_neighbours,  # noqa: E402
                            load_positions, parse_fen, select_context)

MOVES = os.path.join(os.path.dirname(default_positions_path()), "positions_1800_ply30_moves.jsonl")


def extract(min_elo: int = 1800, ply: int = 30, min_plies: int = 40) -> None:
    """The full position (side to move included) and the move played next,
    for the same games and in the same order as ``run_chess.py --extract``."""
    import chess.pgn
    src = os.path.join(os.path.dirname(MOVES), "lichess_db_standard_rated_2013-01.pgn.zst")
    elo = re.compile(r'\[(White|Black)Elo "(\d+)"\]')

    def games(stream):
        buf = []
        for line in stream:
            if line.startswith("[Event ") and buf:
                yield "".join(buf)
                buf = []
            buf.append(line)
        if buf:
            yield "".join(buf)

    proc = subprocess.Popen(["zstd", "-dc", src], stdout=subprocess.PIPE)
    kept = 0
    with open(MOVES, "w") as f:
        for text in games(io.TextIOWrapper(proc.stdout, encoding="utf-8")):
            ratings = dict(elo.findall(text))
            if len(ratings) < 2 or min(int(v) for v in ratings.values()) < min_elo:
                continue
            game = chess.pgn.read_game(io.StringIO(text))
            moves = list(game.mainline_moves())
            if len(moves) < min_plies:
                continue
            board = game.board()
            for m in moves[:ply]:
                board.push(m)
            f.write(json.dumps({"fen": board.fen(), "next": moves[ply].uci(),
                                "white": int(ratings["White"]), "black": int(ratings["Black"])}) + "\n")
            kept += 1
    print(f"{kept} positions with their next move written to {MOVES}")


def position_bits(search: BoardSearch, position, alpha: float) -> float:
    """Bits of one position under a search's counts, its chunk moves replayed."""
    a = BoardSearch([position], alpha, search.features)
    for B, rel, C, Y in search.moves:
        p = a.pos[0]
        pairs = [(0, sq, t) for sq, x in p["tops"].items() if x[0] == B
                 for r, t in forward_neighbours(p["position"], sq)
                 if r == rel and p["owner"][t] == t and t in p["tops"] and p["tops"][t][0] == C]
        chosen = a.select(pairs)
        if chosen:
            a.fresh = Y[1]
            a.apply((B, rel, C), chosen)
    return search.code_of(a)


def main():
    import chess
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "chess_play"))
    args = ap.parse_args()
    if args.extract:
        extract()
        return
    os.makedirs(args.out, exist_ok=True)
    positions = load_positions()
    with open(MOVES) as f:
        recs = [json.loads(line) for line in f]
    if len(recs) != len(positions) or any(parse_fen(r["fen"].split()[0]) != p for r, p in zip(recs, positions)):
        raise SystemExit("the moves file does not match the positions; run with --extract")
    order = np.random.default_rng(args.seed).permutation(len(positions))
    train = [positions[i] for i in order[:args.train]]
    test = [recs[i] for i in order[args.train:args.train + args.test]]

    t0 = time.time()
    features = select_context(train, args.alpha)
    full = BoardSearch(train, args.alpha, features)
    full.run(500)
    models = {"shortest code: chunks and the read's context": full,
              "shortest code: the read's context, no chunks": BoardSearch(train, args.alpha, features),
              "shortest code: each square on its own": BoardSearch(train, args.alpha)}
    value = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
    hits = {name: [0, 0] for name in models}
    rand, greedy, combined, n_moves = [], [], [], []
    for r in test:
        board = chess.Board(r["fen"])
        legal = list(board.legal_moves)
        played = chess.Move.from_uci(r["next"])
        n_moves.append(len(legal))
        rand.append(1 / len(legal))

        def gain(m):
            victim = board.piece_at(m.to_square)
            return (value[victim.piece_type] if victim else 1,           # en passant takes a pawn
                    -value[board.piece_at(m.from_square).piece_type])
        captures = [m for m in legal if board.is_capture(m)]
        if captures:
            best = max(gain(m) for m in captures)
            top = [m for m in captures if gain(m) == best]
            greedy.append((played in top) / len(top))
        else:
            greedy.append(1 / len(legal))
        # One stupid rule first: a capture that does not lose material (the
        # victim is worth at least the capturer), most valuable victim first.
        fair = [m for m in captures if gain(m)[0] >= -gain(m)[1]]
        after = []
        for m in legal:
            b2 = board.copy()
            b2.push(m)
            after.append(parse_fen(b2.board_fen()))
        for name, search in models.items():
            bits = np.array([position_bits(search, p, args.alpha) for p in after])
            ranked = [legal[i] for i in np.argsort(bits, kind="stable")]
            hits[name][0] += ranked[0] == played
            hits[name][1] += played in ranked[:3]
            if search is full:
                if fair:
                    best = max(gain(m) for m in fair)
                    pick = [m for m in fair if gain(m) == best]
                    combined.append((played in pick) / len(pick))
                else:
                    combined.append(float(ranked[0] == played))
    n = len(test)
    rows = {"a random legal move": (float(np.mean(rand)), float(np.mean([min(1.0, 3 * x) for x in rand]))),
            "capture the most valuable piece, else a random move": (float(np.mean(greedy)), None)}
    rows.update({name: (h1 / n, h3 / n) for name, (h1, h3) in hits.items()})
    rows["a capture that does not lose material, else the shortest code"] = (float(np.mean(combined)), None)
    results = {"train": args.train, "test": n, "seconds": time.time() - t0,
               "legal moves per position": float(np.mean(n_moves)),
               "context of the read": [f"at least {m} {k}" for k, m in features],
               "chunk moves": [f"[{B} {rel} {C}]" for B, rel, C, _ in full.moves],
               "agreement with the move played (top 1, top 3)": rows}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    lines = [f"Choosing White's move after ply 30 in {n} held-out Lichess positions (both players 1800+; "
             f"{np.mean(n_moves):.1f} legal moves on average). The grammar's code was learned from "
             f"{args.train} positions (the read's context: {', '.join(results['context of the read'])}; "
             f"chunks: {', '.join(results['chunk moves']) or 'none'}).", "",
             "| Rule | Picks the move played | Among its top 3 |", "|---|---|---|"]
    for name, (a, b) in rows.items():
        lines.append(f"| {name} | {a:.1%} | {'–' if b is None else f'{b:.1%}'} |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
