"""A longer training stream for chess_cs1973_exact.py: every 10th ply of the same training games.

data/chess/cs1973/train.fen holds one position (a random ply in 19..99) from each of the 4,660
training games of the Lichess January 2013 dump (both players 1800+); the dump has no more such
games. Here the build of chess_cs1973.py is replayed with the same random stream, so the games split
the same way into test games (the middle/end pools) and training games, and the replay is checked
against train.fen. Each training game then gives its positions at plies 20, 30, ..., 100 (every 10th
ply while the game lasts). The stream is game-major, and the games are shuffled as wholes (seed 0),
so a learner reading a prefix studies whole games, and no test game is ever in it.

Writes data/chess/cs1973/train_long.fen (one position per line) and train_long_games.txt (the game
index of each line).
"""
import io
import os
import random
import subprocess

import chess
import chess.pgn

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = os.path.join(ROOT, "data", "chess", "cs1973")
pgn = os.path.join(ROOT, "data", "chess", "lichess_db_standard_rated_2013-01.pgn.zst")
proc = subprocess.Popen(["zstd", "-dc", pgn], stdout=subprocess.PIPE)
stream = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
rng = random.Random(0)
mid, end, train, games = [], [], [], []
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
    if role < 0.5:
        if len(fens) >= 41 and 24 <= len(chess.Board(fens[40] + " w - - 0 1").piece_map()) <= 26 and len(mid) < 1000:
            mid.append(fens[40])
        elif len(fens) >= 81 and 12 <= len(chess.Board(fens[80] + " w - - 0 1").piece_map()) <= 15 and len(end) < 1000:
            end.append(fens[80])
    elif len(fens) >= 20 and len(train) < 8000:
        train.append(fens[rng.randrange(19, min(len(fens), 100))])
        games.append([fens[p - 1] for p in range(20, min(len(fens), 100) + 1, 10)])
proc.kill()
old = [l.strip() for l in open(os.path.join(D, "train.fen"), encoding="utf-8") if l.strip()]
assert train == old, (len(train), len(old))
assert mid == [l.strip() for l in open(os.path.join(D, "middle.fen"), encoding="utf-8") if l.strip()]
order = list(range(len(games)))
random.Random(0).shuffle(order)
with open(os.path.join(D, "train_long.fen"), "w", encoding="utf-8") as fh, \
        open(os.path.join(D, "train_long_games.txt"), "w", encoding="utf-8") as gh:
    for gi in order:
        for f in games[gi]:
            fh.write(f + "\n")
            gh.write(f"{gi}\n")
print("replay matches train.fen:", len(train), "games;", sum(map(len, games)), "positions in train_long.fen")
