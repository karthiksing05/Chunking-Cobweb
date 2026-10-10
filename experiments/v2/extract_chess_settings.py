"""Positions for the chess chunk-setting sweep (Track 2, chess_chunks).

One pass over the Lichess January 2013 PGN (the file run_chess.py extracts
from) keeps every game in which both players are rated at least 1500 and
that lasts at least 40 plies (run_chess.extract's filter), and records per
game: the lower rating, the ECO code, the first six moves (UCI), and the
board after plies 12, 16, 20 and 30. It then writes one positions file per
(minimum rating, ply), one position per game in game order, as
run_chess.extract does:

    data/chess/settings/games.tsv
    data/chess/settings/positions_<elo>_ply<p>.fen   (elo 1500/1800/2000; p 12/16/20/30)

positions_1500_ply30.fen here must equal data/chess/positions_1500_ply30.fen
(checked; that file is not touched).
"""
from __future__ import annotations

import io
import os
import re
import subprocess

import chess.pgn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "chess", "lichess_db_standard_rated_2013-01.pgn.zst")
OUT = os.path.join(ROOT, "data", "chess", "settings")
ZSTD = os.path.expanduser("~/flash/miniconda3/envs/trellis/bin/zstd")
PLIES = (12, 16, 20, 30)
ELOS = (1500, 1800, 2000)


def games(stream):
    buf = []
    for line in stream:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            buf = []
        buf.append(line)
    if buf:
        yield "".join(buf)


def main():
    os.makedirs(OUT, exist_ok=True)
    elo = re.compile(r'\[(White|Black)Elo "(\d+)"\]')
    eco = re.compile(r'\[ECO "([^"]*)"\]')
    proc = subprocess.Popen([ZSTD, "-dc", SRC], stdout=subprocess.PIPE)
    rows = []
    n = 0
    for text in games(io.TextIOWrapper(proc.stdout, encoding="utf-8")):
        n += 1
        ratings = dict(elo.findall(text))
        if len(ratings) < 2 or min(int(v) for v in ratings.values()) < min(ELOS):
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        moves = list(game.mainline_moves())
        if len(moves) < 40:
            continue
        board = game.board()
        fens = {}
        for i, m in enumerate(moves[:max(PLIES)], start=1):
            board.push(m)
            if i in PLIES:
                fens[i] = board.board_fen()
        e = eco.search(text)
        rows.append((min(int(v) for v in ratings.values()), e.group(1) if e else "?",
                     " ".join(m.uci() for m in moves[:6]), fens))
    with open(os.path.join(OUT, "games.tsv"), "w") as f:
        f.write("min_elo\teco\topening6\t" + "\t".join(f"ply{p}" for p in PLIES) + "\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t" + "\t".join(r[3][p] for p in PLIES) + "\n")
    for lo in ELOS:
        for p in PLIES:
            with open(os.path.join(OUT, f"positions_{lo}_ply{p}.fen"), "w") as f:
                for r in rows:
                    if r[0] >= lo:
                        f.write(r[3][p] + "\n")
    print(f"{n} games read; kept (both 1500+, 40+ plies): {len(rows)}; "
          + ", ".join(f"{lo}+: {sum(r[0] >= lo for r in rows)}" for lo in ELOS))
    ref = os.path.join(ROOT, "data", "chess", "positions_1500_ply30.fen")
    same = open(ref).read() == open(os.path.join(OUT, "positions_1500_ply30.fen")).read()
    print("positions_1500_ply30.fen identical to the committed extraction:", same)


if __name__ == "__main__":
    main()
