Chess positions (Lichess, both players 1800+, after ply 30): 4000 learned, 500 held out.

The read counts nothing: every square is read on its own (α = 0.001).

| Bits per position | The read alone, no chunks | TRELLIS v2 |
|---|---|---|
| training | 85.05 | 84.42 (search alone: 84.67) |
| held out | 82.75 | 81.93 (the grammar; the search's code with its chunks: 81.81) |

Symbols: 18; rule classes: 19; chunk types: 12. The grammar's size: 10,604 model bits (and 327,081 data bits for the training positions).

| Chunk type | Count | Most frequent anchors |
|---|---|---|
| `[bP N1 bB]` | 2164 | g6 (599), e6 (445), b6 (281) |
| `[bR E1 bK]` | 2110 | f8 (2083), d8 (20), e8 (5) |
| `[wB N1 wP]` | 1826 | g2 (396), d3 (338), b2 (209) |
| `[bR E2 bK]` | 536 | e8 (453), f8 (68), c8 (11) |
| `[bK E3 bR]` | 380 | e8 (374), c8 (4), a8 (1) |
| `[wK E3 wR]` | 298 | e1 (290), c1 (4), b1 (2) |
| `[wK E1 wR]` | 237 | c1 (227), e1 (3), b1 (3) |
| `[bK E1 bR]` | 181 | c8 (163), g8 (7), e8 (5) |

| Generated positions (1,000) | TRELLIS v2 | The read alone, no chunks |
|---|---|---|
| one king each | 37.7% | 38.3% |
| no pawn on a back rank | 100.0% | 100.0% |
| at most 8 pawns each | 73.1% | 70.5% |
| at most 16 pieces each | 79.9% | 79.2% |
| no more of any kind than at the start | 7.7% | 5.8% |
| passes every check | 4.3% | 3.8% |
| samples rejected (a chunk off the board or on an occupied square) | 7.7% | – |
| generated chunks found, piece for piece, in a held-out position | 93.0% | – |
