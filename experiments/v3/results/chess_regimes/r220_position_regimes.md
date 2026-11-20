# r220: chess regimes over generated POSITIONS, level of abstraction as the dial

Code: `experiments/v3/diag/r220_judge.py` (judge), `r220_position_gen.py` (sampler), `r220_report.py` (tables). Data:
`r220_judge.json`, `r220_gen_v1_s13.json`, `r220_gen_v1_s17.json`, `r220_gen_diag_s19.json` (illegal causes), full tables and boards
`r220_tables.md`. Jobs: judge 4806424; smoke 4806425; generation 4806433 (seed 13), 4806434 (seed 17), both cancelled after N=4000
finished (N=8000 not run); causes diag 4806443; report 4806751 (final).

## Verdict: NO-GO

The go rule (R-real >= .6, R-legal >= .5, R-non >= .6 at some settings) fails on R-real and R-legal at every setting and size.
R-non >= .6 holds everywhere (.88-.99). Generating chess positions from TRELLIS's own chunk and concept distributions gives
illegal boards at every level of the concept tree; the level moves R-real from .10 (leaf) to .01 (root) and R-legal never exceeds .015.
There is no meaningful / grammatical-but-empty / nonsense ladder in chess positions with this generator.

## How the sampler maps onto the original architecture (no new generation machinery)

The sampler runs forward, as draws, the distributions that `chess_compress_gen.py` already uses to code a position by "the sequence of
choices generation makes to produce exactly that position" (its docstring: root, anchor, parts, relation, copy, absolute, escape).
Representation is unchanged: the cs73L learner (GraphTrellis, chess_flat, chess_join any, max_size 4), the same stored top units.

| step in the sampler | original component |
|---|---|
| draw the next top unit from {stop, single piece-square, chunk concept} with probability by counts | the TOP alphabet (`top_bits`, `single_bits`, `stop_bits`): share of stored top chunks under the concept, single piece-square frequency, a stop item per position |
| the chunk's content concept is the ancestor `kc` levels above its leaf in the content tree (L.cnt path from `categorize`) | the code's "root" choice at content level kc (`lvl(cpath_of_record(rid), kc)`); **kc is the dial** (0 = leaf concept, 99 = root) |
| anchor drawn from the anchor squares of the stored top chunks under that concept, restricted to where the shape fits | `anchor_bits` / `fits` |
| body = a stored instance (pieces at offsets from the anchor) drawn from the stored top chunks under that concept | the "copy" route (`copy_bits`, instances by offsets) |
| arm abs0: a stored instance on its own squares, anchor free (kc=0) | the "absolute copy" route (`abs_copy_bits`) |
| arm list: pieces iid from the training piece-square frequencies, count from the training piece-count distribution | the "escape" route (piece-square list, `raw_bits`) |
| occupied squares excluded: a unit that does not fit is redrawn (<= 30 times), else dropped (reported as drop/board) | the context head's "occupied squares left out"; the code decodes only boards where units fit |

Not used: the context-tree head (CTXR, FILL at the context concept kx) and the part slots (LPART/RPART). So only the content dial is
turned; the context dial was not run (it would need sorting partial boards in the context tree; not attempted).
Nothing outside the existing distributions was added: no legality filter, no repair, no counts of kings.
Abstraction in the text analogue is "which ancestor concept's instances fill the slot"; here it is "which ancestor concept's stored
instances and anchors fill the unit".

## Judge (fixed before any TRELLIS output was generated)

- Legal (R-non otherwise): python-chess `Board.is_valid()` for the board with White to move, no castling or en passant.
- Realism: logistic regression (numpy, L2) on 768 piece-square indicators, 12 piece counts, 16 pawns-per-file counts, material
  difference, piece count. Real = `data/chess/cs1973/train_long.fen` lines 19459-25459; negatives = equal parts the same positions with
  pieces on random squares (legal ones kept) and random-play positions (20-70 plies). TAU = 5th percentile of the score on 1000 further
  real positions (lines 26459-27459). R-real = legal and score >= TAU (.702); R-legal = legal and below. The generator trains only on
  lines 0-19458 (disjoint games).

Controls (pass rates by class; `r220_judge.json`):

| control | R-real | R-legal | R-non | n |
|---|---|---|---|---|
| real held-out (lines 25459-26459) | .957 | .043 | 0 | 1000 |
| real validation (sets TAU) | .950 | .050 | 0 | 1000 |
| random legal, pieces shuffled | .030 | .970 | 0 | 500 |
| random legal, random play | .006 | .972 | .022 | 500 |
| random illegal (pieces on random squares) | 0 | 0 | 1.000 | 500 |
| real with a king added or removed | 0 | 0 | 1.000 | 500 |

The judge separates the three classes cleanly (purity .96-1.0). Caveat: realism is a statistical classifier of piece-square
statistics, so endgame boards with a few pieces can pass if legal (several showcase R-real boards at the root are such).

## Dial table (mean of seeds 13 and 17, 500 boards per cell each; seed spread <= .04 on every cell, see `r220_tables.md`)

N = 4000 positions:

| setting | R-real | R-legal | R-non | drop/board | pieces |
|---|---|---|---|---|---|
| kc=0 leaf concept (exact-ish stored chunks) | .085 | .000 | .915 | 2.4 | 12.1 |
| abs0 (stored instance on its squares) | .097 | .001 | .902 | 2.9 | 13.3 |
| kc=1 | .029 | .003 | .968 | 1.5 | 17.0 |
| kc=2 | .027 | .015 | .958 | 1.3 | 16.6 |
| kc=3 | .014 | .011 | .975 | 1.1 | 16.6 |
| kc=4 | .010 | .008 | .982 | 1.2 | 16.3 |
| kc=6 | .004 | .006 | .990 | 1.0 | 16.6 |
| kc=99 root | .006 | .005 | .989 | 1.1 | 16.2 |
| list (piece-square list) | .103 | .002 | .895 | 0 | 21.1 |

(Real boards have about 21 pieces; leaf-level boards are short because units that do not fit are dropped.)

## Learning curves (R-real / R-legal / R-non, mean of 2 seeds)

| N | kc=0 | kc=2 | kc=99 | list |
|---|---|---|---|---|
| 500 | .082 / .004 / .914 | .020 / .009 / .971 | .010 / .013 / .977 | .122 / .003 / .875 |
| 1000 | .081 / .003 / .916 | .037 / .006 / .957 | .008 / .003 / .989 | .112 / .001 / .887 |
| 2000 | .080 / .000 / .920 | .025 / .005 / .970 | .006 / .010 / .984 | .116 / .003 / .881 |
| 4000 | .085 / .000 / .915 | .027 / .015 / .958 | .006 / .005 / .989 | .103 / .002 / .895 |

N=8000 not run (learning 4000 took about 40 minutes of the 2-hour 8000 run; the curve is flat from 500 to 4000, so it was cancelled).
No setting changes with data: more training positions do not move any regime share.

## Why: illegality comes from counts, not from the level

Status flags of illegal boards (diag run, N=500, 300 boards): at kc=0, 44% have no white king, 44% no black king, 13% too many kings,
9-11% too many pawns; at the root, 61% have pawns on the back rank and 26% too many kings. The top alphabet draws units
independently, so "exactly one king each" and "at most 8 pawns" are properties of the whole board that no unit-level distribution
encodes (the same finding as r217 for moves: TRELLIS recognises positions but does not encode legality). The leaf level is better
only because units there are narrow, stored, locally real structures; boards that happen to contain both kings are real-looking
(R-real .08-.10, R-legal ~0). Abstraction trades pawn-structure realism for back-rank pawns and extra kings, so legal-but-unrealistic
boards (R-legal) are rare at every level (.000-.015): when a board is legal it is almost always at the level it was drawn from, real.

## Showcase (N=4000, seed 13; board field of the FEN, ranks 8 to 1)

- kc=0 R-real: `6k1/1p3pp1/p1p1p1p1/7p/3P1PP1/PBNB4/7P/6K1`, `rr2k2r/3p1ppp/2p1pp2/3P3P/1N6/3BP3/p1Q3PP/4RRK1`
- kc=0 R-non: `8/8/8/8/8/5N2/8/R2Q1R2` (no kings), `8/7p/6p1/8/8/3P2P1/5P1N/6K1` (no black king)
- kc=2 R-legal: `5k2/8/1pp1p3/1r1pNr1b/1pp5/7P/PP2QPP1/RNB1K2R`, `2r2q2/r2b4/2p3pp/np2pp2/p7/4k1p1/6P1/6K1`
- kc=2 R-non: `R7/8/8/8/8/8/Q1P1R3/BQNB1R2` (no kings)
- root R-legal: `8/1P4r1/PPp3k1/K...` style piles of pawns (`8/8/PPP2q2/bKp5/2pp4/1p3n1P/p1k1P3/2br4`)
- root R-non: `8/8/8/8/8/4p3/3p1p2/6p1`, `b4p2/PPQprk1p/p1Pbpb2/1pP1P1P1/1PPpbq2/rK1ppR1B/p1p1np1B/R1B2nK1` (pawns on rank 1/8)
- list R-real: `r6k/p6p/2n1pqp1/2q1p3/8/1PN5/K1P3P1/5R1R`

More in `r220_tables.md`.

## Connection to the text laws and what would be needed

In text, concept level orders the regimes because syntax (category sequences) is what the tree's abstract level carries, and
word-level concepts carry meaning. In chess the analogue of syntax is legality, and it is a global count constraint (one king per
side, pawn limits, no check on the side not to move) that none of TRELLIS's unit-level distributions carry. The abstraction dial
therefore changes realism (leaf .08-.10 to root .006) but not legality; no level gives "legal syntax, unreal content".
Candidate representation changes (allowed by the constraint, not tried here): a count facet in the context description (the old v2 read
counts gave 100% one-king boards, FRAMEWORK.md) or a king-anchored first unit; and the context head (kx) as a second dial.
A positive R-legal regime is unlikely without one of these; r217's move-level negative and this position-level negative agree.

## Caveats

Two seeds, 500 boards each; N=8000 not run; the realism score is a statistics classifier (endgames pass); only the content dial kc was
varied (the context dial and the unit-count dial were not); boards are assumed White to move. The first generation jobs (v1) were
launched before the causes tally was added to the script, so the causes come from the separate diag run (seed 19, N=500).
