Chess positions (Lichess, both players 1800+, after ply 30): 8000 learned, 560 held out.

The read asks, at each square, of each kind of piece in turn whether an element is anchored there on it, given how many pieces of that kind stand on earlier squares. Dirichlet concentration α = 0.001. Read without the counts, squares on their own: 82.29 held-out bits per position.

| Bits per position | The read alone, no chunks | TRELLIS v2 |
|---|---|---|
| training | 75.10 | 74.87 (search alone: 74.94) |
| held out | 72.62 | 72.41 (the grammar; the search's code with its chunks: 72.45) |

Symbols: 14; rule classes: 14; chunk types: 3. The grammar's size: 17,264 model bits (and 581,682 data bits for the training positions).

| Chunk type | Count | Most frequent anchors |
|---|---|---|
| `[wP N1 bP]` | 5604 | d4 (1329), e4 (877), e5 (790) |
| `[wB N1 wP]` | 2860 | g2 (805), b2 (387), d3 (323) |

| Generated positions (1,000) | TRELLIS v2 | The read alone, no chunks |
|---|---|---|
| one king each | 100.0% | 100.0% |
| no pawn on a back rank | 99.8% | 99.8% |
| at most 8 pawns each | 99.9% | 100.0% |
| at most 16 pieces each | 99.9% | 100.0% |
| no more of any kind than at the start | 99.9% | 100.0% |
| passes every check | 99.8% | 99.8% |
| samples rejected (a chunk off the board or on an occupied square) | 0.2% | – |
| generated chunks found, piece for piece, in a held-out position | 98.6% | – |
