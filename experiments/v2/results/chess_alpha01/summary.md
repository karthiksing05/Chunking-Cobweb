Chess positions (Lichess, both players 1800+, after ply 30): 4000 learned, 500 held out.

The context of the square-by-square read (features of the pieces on earlier squares that pay for themselves): at least 2 bR, at least 1 wK, at least 1 bK, at least 2 wR, at least 1 wR, at least 1 bQ, at least 1 wQ, at least 7 bP. Dirichlet concentration α = 0.01. Without the context, squares on their own: 82.81 held-out bits per position.

| Bits per position | Squares on their own | TRELLIS v2 |
|---|---|---|
| training | 79.19 | 79.49 (search alone: 79.19) |
| held out | 75.27 | 75.27 (learned chunks) |

Symbols: 12; rule classes: 12; chunk types: 0. The grammar's size: 18,828 model bits (and 299,132 data bits for the training positions).

| Chunk type | Count | Most frequent anchors |
|---|---|---|

| Generated positions (1,000) | TRELLIS v2 | Squares on their own |
|---|---|---|
| one king each | 94.5% | 99.9% |
| no pawn on a back rank | 98.9% | 100.0% |
| at most 8 pawns each | 83.3% | 78.8% |
| at most 16 pieces each | 83.8% | 84.6% |
| no more of any kind than at the start | 37.5% | 39.2% |
| passes every check | 37.2% | 39.1% |
| samples rejected (a chunk off the board or on an occupied square) | 0.0% | – |
| generated chunks found, piece for piece, in a held-out position | 0.0% | – |
