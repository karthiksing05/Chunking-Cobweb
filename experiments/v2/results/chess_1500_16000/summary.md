Chess positions (Lichess, both players 1500+, after ply 30): 16000 learned, 500 held out.

The read asks, at each square, of each kind of piece in turn whether an element is anchored there on it, given how many pieces of that kind stand on earlier squares. Dirichlet concentration α = 0.001. Read without the counts, squares on their own: 85.20 held-out bits per position.

| Bits per position | The read alone, no chunks | TRELLIS v2 |
|---|---|---|
| training | 76.43 | 76.28 (search alone: 76.32) |
| held out | 75.48 | 75.37 (the grammar; the search's code with its chunks: 75.38) |

Symbols: 13; rule classes: 13; chunk types: 1. The grammar's size: 18,679 model bits (and 1,201,739 data bits for the training positions).

| Chunk type | Count | Most frequent anchors |
|---|---|---|
| `[wP N1 bP]` | 10799 | d4 (2215), e4 (1934), e5 (1427) |

| Generated positions (1,000) | TRELLIS v2 | The read alone, no chunks |
|---|---|---|
| one king each | 100.0% | 100.0% |
| no pawn on a back rank | 99.9% | 100.0% |
| at most 8 pawns each | 99.9% | 100.0% |
| at most 16 pieces each | 99.9% | 100.0% |
| no more of any kind than at the start | 99.9% | 100.0% |
| passes every check | 99.9% | 100.0% |
| samples rejected (a chunk off the board or on an occupied square) | 0.0% | – |
| generated chunks found, piece for piece, in a held-out position | 98.0% | – |
