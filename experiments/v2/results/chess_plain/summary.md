Chess positions (Lichess, both players 1800+, after ply 30): 4000 learned, 500 held out.

The read has no context: every square is read on its own (α = 0.001).

| Bits per position | Squares on their own | TRELLIS v2 |
|---|---|---|
| training | 84.39 | 83.80 (search alone: 83.91) |
| held out | 82.69 | 81.74 (learned chunks) |

Symbols: 17; rule classes: 18; chunk types: 11. The grammar's size: 7,464 model bits (and 327,754 data bits for the training positions).

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

| Generated positions (1,000) | TRELLIS v2 | Squares on their own |
|---|---|---|
| one king each | 35.2% | 36.8% |
| no pawn on a back rank | 100.0% | 100.0% |
| at most 8 pawns each | 72.5% | 70.4% |
| at most 16 pieces each | 79.1% | 79.6% |
| no more of any kind than at the start | 7.2% | 6.4% |
| passes every check | 4.1% | 3.8% |
| samples rejected (a chunk off the board or on an occupied square) | 5.8% | – |
| generated chunks found, piece for piece, in a held-out position | 94.7% | – |
