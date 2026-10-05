Chess positions (Lichess, both players 1800+, after ply 30): 4000 learned, 500 held out.

The read asks, at each square, of each kind of piece in turn whether an element is anchored there on it, given how many pieces of that kind stand on earlier squares. Dirichlet concentration α = 0.01. Read without the counts, squares on their own: 82.90 held-out bits per position.

| Bits per position | The read alone, no chunks | TRELLIS v2 |
|---|---|---|
| training | 76.86 | 77.56 (search alone: 76.86) |
| held out | 73.54 | 73.26 (the grammar; the search's code with its chunks: 73.54) |

Symbols: 12; rule classes: 12; chunk types: 0. The grammar's size: 12,207 model bits (and 298,016 data bits for the training positions).

| Chunk type | Count | Most frequent anchors |
|---|---|---|

| Generated positions (1,000) | TRELLIS v2 | The read alone, no chunks |
|---|---|---|
| one king each | 100.0% | 100.0% |
| no pawn on a back rank | 99.8% | 100.0% |
| at most 8 pawns each | 100.0% | 100.0% |
| at most 16 pieces each | 100.0% | 99.7% |
| no more of any kind than at the start | 100.0% | 99.7% |
| passes every check | 99.8% | 99.7% |
| samples rejected (a chunk off the board or on an occupied square) | 0.0% | – |
| generated chunks found, piece for piece, in a held-out position | 0.0% | – |
