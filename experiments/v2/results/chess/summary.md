Chess positions (Lichess, both players 1800+, after ply 30): 4000 learned, 500 held out.

The read asks, at each square, of each kind of piece in turn whether an element is anchored there on it, given how many pieces of that kind stand on earlier squares. Dirichlet concentration α = 0.001. Read without the counts, squares on their own: 82.75 held-out bits per position.

| Bits per position | The read alone, no chunks | TRELLIS v2 |
|---|---|---|
| training | 76.57 | 76.37 (search alone: 76.47) |
| held out | 73.47 | 73.33 (the grammar; the search's code with its chunks: 73.43) |

Symbols: 13; rule classes: 13; chunk types: 1. The grammar's size: 15,369 model bits (and 290,129 data bits for the training positions).

| Chunk type | Count | Most frequent anchors |
|---|---|---|
| `[wP N1 bP]` | 2788 | d4 (647), e4 (442), e5 (390) |

| Generated positions (1,000) | TRELLIS v2 | The read alone, no chunks |
|---|---|---|
| one king each | 100.0% | 100.0% |
| no pawn on a back rank | 100.0% | 100.0% |
| at most 8 pawns each | 100.0% | 100.0% |
| at most 16 pieces each | 100.0% | 99.9% |
| no more of any kind than at the start | 100.0% | 99.9% |
| passes every check | 100.0% | 99.9% |
| samples rejected (a chunk off the board or on an occupied square) | 0.1% | – |
| generated chunks found, piece for piece, in a held-out position | 98.7% | – |
