Chess positions (Lichess, both players 1800+, after ply 30): 4000 learned, 500 held out.

The context of the square-by-square read (features of the pieces on earlier squares that pay for themselves): at least 2 bR, at least 1 wK, at least 1 bK, at least 2 wR, at least 1 wR, at least 7 bP, at least 1 bR. Dirichlet concentration α = 0.001. Without the context, squares on their own: 82.69 held-out bits per position.

| Bits per position | Squares on their own | TRELLIS v2 |
|---|---|---|
| training | 79.69 | 80.05 (search alone: 79.64) |
| held out | 76.62 | 76.47 (learned chunks) |

Symbols: 14; rule classes: 14; chunk types: 3. The grammar's size: 17,591 model bits (and 302,599 data bits for the training positions).

| Chunk type | Count | Most frequent anchors |
|---|---|---|
| `[bP N1 bB]` | 2164 | g6 (599), e6 (445), b6 (281) |
| `[wB N1 wP]` | 1826 | g2 (396), d3 (338), b2 (209) |

| Generated positions (1,000) | TRELLIS v2 | Squares on their own |
|---|---|---|
| one king each | 96.3% | 100.0% |
| no pawn on a back rank | 99.9% | 100.0% |
| at most 8 pawns each | 82.7% | 80.2% |
| at most 16 pieces each | 84.8% | 85.5% |
| no more of any kind than at the start | 22.9% | 23.1% |
| passes every check | 22.9% | 23.1% |
| samples rejected (a chunk off the board or on an occupied square) | 0.1% | – |
| generated chunks found, piece for piece, in a held-out position | 97.0% | – |
