# Chess: Chase & Simon recall and an Elo study (2026-10-26)

## Chase & Simon (1973) replication (`experiments/v3/chess_recall.py`)

GraphTrellis learns real positions (Lichess 1800+, ply 30). At checkpoints it is shown 100 held-out positions, plus the same pieces scattered at random (random positions).
- **Exposure:** a parse into units (chunks and single pieces).
- **Recognition (CHREST-like):** a chunk is one unit if memory recognises it, i.e. its recognition bits (`_climb`, content + context) are at least θ. Otherwise it falls apart into its parts.
- **Short-term memory:** K = 7 units; the largest are kept.
- **Recall:** the share of pieces inside the kept units.

Recognising by content alone shows no effect: a chunk's content names only its pieces' categories, which random boards share. Recognising by where things stand (context) does.

| recall, K = 7 (smoke run, 20 positions) | novice | 300 learned | 800 learned |
|---|---|---|---|
| real, θ 12 / 16 | 0.27 / 0.27 | 0.57 / 0.30 | 0.76 / 0.47 |
| random, θ 12 / 16 | 0.27 / 0.27 | 0.39 / 0.27 | 0.60 / 0.31 |

The full run goes to 4000 positions, seeds 13/17 (`logs/chess-cs-*`).

## Elo study (`experiments/v3/chess_play.py`)

The TRELLIS player plays the legal move after which memory recognises the position best: summed chunk recognition bits, with no search and no material count. A `+safe` variant first excludes moves that lose material to the best 1-ply reply.

References are a random mover, a 1-ply material-greedy player and a 2-ply material minimax player. Elo comes from Bradley–Terry over all games, with random = 0. Checkpoints are 0 / 300 / 1000 / 4000 positions, with 20 games per pairing (`logs/chess-play-*`).

Absolute (FIDE-like) Elo needs a calibrated engine. Downloading a Stockfish binary was blocked by the permission policy, so the scale here is relative to the reference players.
