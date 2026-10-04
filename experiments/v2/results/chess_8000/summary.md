Chess positions (Lichess, both players 1800+, after ply 30): 8000 learned, 560 held out.

The context of the square-by-square read (features of the pieces on earlier squares that pay for themselves): at least 2 bR, at least 1 wK, at least 1 bK, at least 2 wR, at least 1 wR, at least 1 wQ, at least 1 bQ, at least 7 bP. Dirichlet concentration α = 0.001. Without the context, squares on their own: 82.26 held-out bits per position.

| Bits per position | Squares on their own | TRELLIS v2 |
|---|---|---|
| training | 77.97 | 77.99 (search alone: 77.92) |
| held out | 74.51 | 74.42 (learned chunks) |

Symbols: 14; rule classes: 14; chunk types: 3. The grammar's size: 35,129 model bits (and 588,784 data bits for the training positions).

| Chunk type | Count | Most frequent anchors |
|---|---|---|
| `[bP N1 bB]` | 4376 | g6 (1202), e6 (922), b6 (589) |
| `[wB N1 wP]` | 3628 | g2 (809), d3 (704), b2 (397) |

| Generated positions (1,000) | TRELLIS v2 | Squares on their own |
|---|---|---|
| one king each | 99.6% | 100.0% |
| no pawn on a back rank | 100.0% | 100.0% |
| at most 8 pawns each | 82.0% | 81.5% |
| at most 16 pieces each | 86.8% | 87.2% |
| no more of any kind than at the start | 41.5% | 38.4% |
| passes every check | 41.5% | 38.4% |
| samples rejected (a chunk off the board or on an occupied square) | 0.2% | – |
| generated chunks found, piece for piece, in a held-out position | 98.5% | – |
