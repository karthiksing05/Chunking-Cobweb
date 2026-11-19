# r217: chess regimes (C-real / C-legal / C-illegal) under TRELLIS's existing dials

Design: `experiments/v3/results/story_gen/r214_design.md` section 9. Code: `experiments/v3/diag/r217_chess_judge.py`,
`r217_chess_gen.py`, `r217_chess_report.py`. Data: `r217_testset.json`, `r217_judge_controls.json`, `r217_chess_gen_s13.json`,
and `r217_tables_s13.md` (every arm's table row and its first 5 example positions/moves).

## Setup

- **Existing chess generation.** TRELLIS has no move sampler. Its move policy (`chess_play.py`) plays the legal move after which
  memory recognises the resulting position best (summed content + context recognition bits of the parsed top units). Here every
  candidate move is scored the same way, with the content part (c) and context part (x) kept apart, and the dials are readings of
  those scores.
- **Memory.** cs73L recipe (r213 config: GraphTrellis, chess_flat, chess_join any, max_size 4), seed 13, first N positions of the
  shuffled `train_long.fen`; N = 0, 300, 1000.
- **Test positions.** 200 real positions at ply 30 (white to move) from Lichess Jan 2013 games with min Elo in [1500, 1800), so none is
  in the >= 1800 training games, plus the real next move.
- **Judge.** R-illegal: not legal (python-chess). R-legal: legal, but the move's value is at least 2 pawns below the mover's
  material now. Value = material (mover minus opponent) after the move, the opponent's worst reply and the mover's best
  recapture (captures only, standing pat allowed); mate = +-1000. R-sem: legal and loses less than 2 pawns by that value.
  No engine exists on the cluster (Stockfish not installed, download blocked), so this is a fixed material criterion: it catches
  hung pieces and lost exchanges, not positional sense. R-sem therefore means "does not blunder material", not "good".
  A first 2-ply version flagged 24-37% of real moves (recaptures lie beyond 2 plies), so the capture search was added before any
  arm was generated; the threshold (2) was fixed before the arms ran.

## Judge validation (300 real positions, same source)

| control | R-sem | R-legal | R-illegal |
|---|---|---|---|
| real game continuation | .983 | .017 | 0 |
| random legal move | .697 | .303 | 0 |
| worst legal move (lowest value) | .033 | .967 | 0 |
| real move with a wrong destination square | 0 | 0 | **1.000** |
| random from/to pair (resampled until illegal) | 0 | 0 | **1.000** |

Legality is exact (purity 1.0). Real moves are 98.3% R-sem. The sem/legal split is a weak separator for random moves: only 30% of
random legal moves are flagged, because most random moves in a quiet position hang nothing. So a flat R-legal share of about .30-.35
is what an undirected legal move gives; an arm only has a "legal-but-poor regime" if it exceeds that clearly.

## Per-arm regime purities (200 positions each, seed 13)

Legal-candidate arms (the candidates are python-chess's legal moves, as in `chess_play.py`):

| N | read | R-sem | R-legal | R-illegal |
|---|---|---|---|---|
| 0 | argmax c+x | .670 | .330 | 0 |
| 0 | argmax c / argmax x | .650 / .685 | .350 / .315 | 0 |
| 0 | argmin c+x | .720 | .280 | 0 |
| 0 | softmax T=2 / T=8 | .685 / .690 | .315 / .310 | 0 |
| 0 | uniform (random legal) | .675 | .325 | 0 |
| 0 | argmax c+x, safe | .730 | .270 | 0 |
| 300 | argmax c+x | .640 | .360 | 0 |
| 300 | argmax c / argmax x | .720 / .580 | .280 / .420 | 0 |
| 300 | argmin c+x | .640 | .360 | 0 |
| 300 | softmax T=2 / T=8 | .675 / .665 | .325 / .335 | 0 |
| 300 | uniform | .670 | .330 | 0 |
| 300 | argmax c+x, safe | .725 | .275 | 0 |
| 1000 | argmax c+x (the recipe) | .625 | .375 | 0 |
| 1000 | argmax c / argmax x | .640 / .640 | .360 / .360 | 0 |
| 1000 | argmin c+x | .685 | .315 | 0 |
| 1000 | softmax T=2 / T=8 | .680 / .635 | .320 / .365 | 0 |
| 1000 | uniform | .655 | .345 | 0 |
| 1000 | argmax c+x, safe | .700 | .300 | 0 |

Free-candidate arms (N = 1000; candidates = legal moves + 120 random piece displacements per position, so the model is never told
the rules; about three quarters of candidates are illegal; the uniform arm measured .70):

| read | R-sem | R-legal | R-illegal |
|---|---|---|---|
| argmax c+x | .125 | .105 | .770 |
| argmax c | .120 | .090 | .790 |
| argmax x | .120 | .080 | .800 |
| argmin c+x | .050 | .030 | **.920** |
| softmax T=2 | .160 | .095 | .745 |
| softmax T=8 | .195 | .075 | .730 |
| uniform | .205 | .095 | .700 |

Full table with the real-move match rate, and 5 example positions/moves per arm (all 31 arms): `r217_tables_s13.md`.
Representative examples (N=1000; position, move, regime):

- legal argmax c+x: `1r3rk1/2ppqpbp/b1p3p1/p1PnP3/P7/1P2Q3/1B1N1PPP/2KR1B1R w` g4 R-legal (loses 6); `2r2rk1/pp3ppp/1q2pn2/3pQ3/8/P1NP3P/1PP2PP1/2R2RK1 w` Qe4 R-legal (loses 8); `r2qk2r/4npbb/...` Nxb5 R-sem (also the real move).
- legal softmax T=8: same first position, Bc4 R-legal (loses 6); `r2qk2r/...` Rac1 R-sem.
- free argmax c+x: `1r3rk1/...` a4c6 R-illegal; `r2qk2r/...` e3d1 R-illegal; `2r2rk1/...` c2c4 R-illegal.
- free argmin c+x: a1d6, c5g1, g1e8, d3c8 all R-illegal (sliding or jumping through or off the piece's movement rules).

## Reading

1. **No existing dial gives an R-legal regime.** Over N, read (c, x, c+x), argmin, temperature and the safe filter, the legal-move
   R-legal share stays between .27 and .42, around the .33 of a uniform random legal move (N=1000: .375 for the recipe against .345
   uniform). The 95% sampling error on 200 positions is about +-.065, so every difference between legal-candidate arms except
   perhaps N=300 argmax x (.42) is inside it. The recognition score does not separate poor from reasonable legal moves; the only arms
   below uniform on R-legal are the safe filter (.27-.30), which tests the same material criterion as the judge and is not a
   representation dial.
2. **R-illegal is not reachable through the legal-move route**, since legality is imposed by python-chess there (0 illegal in all 18
   legal arms).
3. **With no rules supplied (free arms), the picks are mostly illegal, but only a little more than chance:** argmax .77-.80 against
   .70 uniform; argmin (the least-recognised position) .92. So recognition prefers reachable-looking boards weakly, and legality
   itself is nearly invisible to the score (a displaced piece makes a board that looks about as familiar as a legal one). The
   argmin arm is the only one that is clearly a nonsense regime. R-sem collapses to .05-.20 in all free arms, and R-legal is .03-.11:
   the free arms are a two-regime axis (illegal vs everything else), not a three-regime one.
4. **Conclusion for the three-regime study in chess:** the position-recognition read has no dial here that moves moves from
   reasonable to legal-but-poor. A C-legal regime would need the generative route of r214 section 9 (sampler over slot/level-b
   choices, judged by position) rather than move choice; this round answers the move-choice version only. The judge itself is
   validated: legality exact, real moves 98% R-sem, worst legal moves 97% R-legal.

## Caveats

- One seed (13), 200 test positions, three memory sizes; no confidence intervals beyond the binomial +-.065.
- Real-move match rates are 2-10% (the policy does not predict the human move); not a quality claim.
- The R-legal/R-sem split is a material heuristic, not an engine eval.
- The argmax c+x and argmax c arms share random tie-breaks; N=0 differences between arms are noise (the memory is empty and the
  scores are nearly constant).
