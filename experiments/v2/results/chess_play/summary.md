Choosing White's move after ply 30 in 500 held-out Lichess positions (both players 1800+; 37.7 legal moves on average). The grammar's code was learned from 4000 positions (the read's context: at least 2 bR, at least 1 wK, at least 1 bK, at least 2 wR, at least 1 wR, at least 7 bP, at least 1 bR; chunks: [wB N1 wP], [bP N1 bB]).

| Rule | Picks the move played | Among its top 3 |
|---|---|---|
| a random legal move | 3.2% | 9.3% |
| capture the most valuable piece, else a random move | 19.4% | – |
| shortest code: chunks and the read's context | 9.0% | 21.8% |
| shortest code: the read's context, no chunks | 10.0% | 21.4% |
| shortest code: each square on its own | 8.8% | 22.2% |
| a capture that does not lose material, else the shortest code | 20.1% | – |
