# r226: where vs what abstraction (parse then regenerate with the anchor and the pieces abstracted separately)

Follows r221 (`r221_position_regimes.md`). Same judge (`r220_judge.py`, unchanged), same learner (cs73L recipe), same train/judge line splits,
held-out real positions = shuffled training lines [10000:10300] (never learned, disjoint from the judge's real lines). Two seeds (13, 17);
N = 500, 1000, 2000, 4000; 600 boards per cell per seed (300 positions x 2 draws). Code: `experiments/v3/diag/r226_where_what.py`,
`r226_report.py`. Data: `r226_v1_s13.json`, `r226_v1_s17.json`; full tables, illegal causes and boards: `r226_tables.md`.
Jobs: smoke 4808285; runs 4808297 (seed 13), 4808298 (seed 17); report 4808686. No waiters or jobs of mine left running.

## Verdict: partial; real and nonsense regimes yes, legal-unrealistic only as a thin band (R-legal <= .19, rule needs .5)

| rule | best cell | value |
|---|---|---|
| R-real >= .6 | where kc=0 (N=4000) | .833 |
| R-legal >= .5 | where kc=99 (N=4000) | **.186** (fails) |
| R-non >= .6 | where+what kc=6 (N=2000) | .985 |

Location abstraction alone gives a graded ladder real -> legal-unrealistic -> illegal on one dial (where kc: R-real .83 -> .05, R-legal .04 -> .19),
and adding identity abstraction gives the nonsense regime. The legal share peaks at .19 because 63% of the root-level boards place a pawn on rank 1 or 8.

## Mapping onto the original architecture

Parse = the existing parser (GraphTrellis.parse, learning off). Both dials are reads of the existing concept tree L.cnt, at kc levels above the
leaf the unit's content sorts to (99 = root; a concept with no stored top chunk backs off to its nearest ancestor that has some):
- **where (location abstraction)**: the unit's anchor is redrawn from the anchors of the stored top chunks under that concept (the code's anchor
  read, `anchor_bits`: "by the anchor squares of the stored top chunks under its concept"), at a square where the unit's own shape fits and the
  squares are free (30 tries, else the unit stays; share reported as "units kept"). kc=0: the anchors stored under its own leaf. kc=99 (root):
  any square where the shape fits (uniform). The unit keeps its own pieces and shape (the "relation": offsets from the anchor).
  A single piece is a 1-piece chunk: kc=0 the stored squares of singles of its own token, 0<kc<99 the stored squares of all single pieces, 99 any
  free square (single-piece-square items of the top alphabet). Kings are singles or chunk members and move like the rest.
- **what (identity abstraction, = r221b)**: the unit's pieces are replaced by a stored instance under the concept (copy route, instances by offsets);
  a single's token by a token of the stored singles on its square (kc=0 unchanged; root: all singles).
- Dial B is "where+what at the same kc". The 1-D path is leaf (kc=0) -> where-only at higher kc -> where+what.
No new generation machinery and no representation change; no legality filter, no repair.
Controls: identity (real board through the parser) and the piece-list controls from the escape route (counts off: pieces iid from piece-square frequencies;
counts on: the stored count profile's pieces on piece-square draws).

## Dial table (mean of 2 seeds; R-real / R-legal / R-non; units kept = share of units for which no draw fitted)

| cell | N=1000 | N=4000 | kept (N=4000) |
|---|---|---|---|
| identity (control) | .945 / .055 / 0 | .945 / .055 / 0 | 0 |
| where kc=0 | .809 / .046 / .145 | .833 / .043 / .123 | .199 |
| where kc=1 | .397 / .120 / .484 | .455 / .115 / .429 | .038 |
| where kc=2 | .309 / .164 / .526 | .324 / .178 / .498 | .035 |
| where kc=3 | .171 / .174 / .655 | .207 / .179 / .615 | .046 |
| where kc=4 | .139 / .141 / .721 | .150 / .137 / .712 | .048 |
| where kc=6 | .082 / .142 / .776 | .120 / .148 / .732 | .050 |
| where kc=99 | .051 / .176 / .773 | .053 / .186 / .760 | .083 |
| where+what kc=0 | .742 / .030 / .227 | .802 / .018 / .180 | .177 |
| where+what kc=1 | .161 / .040 / .799 | .183 / .048 / .768 | .014 |
| where+what kc=2 | .028 / .022 / .950 | .017 / .019 / .964 | .017 |
| where+what kc=3 | .014 / .018 / .969 | .014 / .010 / .976 | .014 |
| where+what kc=4 | .008 / .018 / .973 | .002 / .018 / .981 | .020 |
| where+what kc=6 | .005 / .015 / .979 | .003 / .016 / .982 | .024 |
| where+what kc=99 | .006 / .020 / .974 | .005 / .013 / .982 | .026 |
| list, counts off | .115 / .003 / .883 | .100 / .002 / .899 | 0 |
| list, counts on | .528 / .005 / .467 | .492 / .006 / .502 | 0 |

## Learning curves (R-real / R-legal / R-non), the three representative cells (best at N=4000 per regime, chosen after the grid)

| N | real: where kc=0 | legal: where kc=99 | non: where+what kc=99 |
|---|---|---|---|
| 500 | .809 / .053 / .138 | .060 / .152 / .786 | .000 / .018 / .982 |
| 1000 | .809 / .046 / .145 | .051 / .176 / .773 | .006 / .020 / .974 |
| 2000 | .833 / .047 / .121 | .055 / .163 / .782 | .003 / .016 / .981 |
| 4000 | .833 / .043 / .123 | .053 / .186 / .760 | .005 / .013 / .982 |

The curves are flat from 500 to 4000: the regimes depend on the abstraction level, and data barely moves them (R-real at kc=0 gains .02).

## Why R-legal is capped (illegal causes, N=4000, seed 13, share of all boards)

| cell | pawns on back rank | side not to move in check | other |
|---|---|---|---|
| where kc=0 | .07 | .10 | |
| where kc=1 | .31 | .17 | |
| where kc=3 | .40 | .25 | |
| where kc=6 | .61 | .35 | |
| where kc=99 | .63 | .38 | too-many-checkers / impossible check .02 |
| where+what kc=99 | .74 | .23 | no white king .41, no black king .36, too many kings .27, too many pawns |

Counts are preserved by where-only (no king, pawn or piece-count flags at all) so the prediction about counts holds: where-only boards are never illegal for
king or pawn counts. The illegal ones fail on placement rules the anchor read does not know: a pawn lands on rank 1 or 8 (the dominant cause, rising from .07 to .63
with the level) and the side not to move ends in check (.10 to .38). Adjacent kings are rare: 2-8% of the illegal boards (3-13% for where+what), so they are not the cause.
Identity abstraction then adds count failures (no king, two kings), taking R-non from .76 to .98.

## Boards (N=4000, seed 13; board field of the FEN, ranks 8 to 1)

- where kc=0, R-real: `1r4k1/2pR1pp1/r1p1n1p1/p1B2P2/2P3P1/7P/1PP5/2K4R`, `3q1rk1/1p2n1pp/2pb4/p5r1/2N1PQ1p/2P5/PP3PPP/R1B2RK1`
- where kc=0, R-legal: `1r4k1/2p3p1/r1p1n3/p4P2/2P3P1/2R1p2P/1PP2p2/1BK4R`, `r3k2r/pp3pp1/p2p1q2/1pp5/3n4/P1bP2P1/1PPBP3/R2QK1R1`, `rn2k2r/pp3ppp/5q2/4p3/5pp1/P1bP2P1/1PPBP3/R2QK1R1`
- where kc=0, R-non: `r1p1r3/pp1p2k1/p6p/2Q3P1/3P4/2P4P/PP3P2/6K1`, `r3rpk1/pppn2p1/5q2/4N2p/3P1P2/1Q4Pb/1P1P1P1P/2R1R1KB`
- where kc=99, R-real: `r2q1rk1/ppp1bpp1/1Pnpbn1p/1P2B3/1P1QP3/1R1KPN2/B5PP/7R`, `3q1rk1/1p1Bbpp1/P1Qpbn1p/R2Br3/2P1p1p1/2P1PNn1/5PPP/4K2R`, `8/1k6/p1p1p3/8/3P4/P1P1R3/4K3/6r1`
- where kc=99, R-legal: `2r5/1Q1pqrk1/5npp/2ppb1P1/2N1P3/1pPP4/2PPP3/R1BK1R2`, `6k1/r3r2p/pp1p4/1PP1pPP1/1pP4K/1P1Q2P1/8/8`, `8/2rp1kr1/PPpppPp1/1p2P1K1/1P1Q4/P4P2/8/8`
- where kc=99, R-non: `8/4P3/r1P2kP1/4RppP/PP1n3p/1K1B2pR/4r1p1/4p3` (pawns on rank 8 and 1), `1rR1p1k1/5pp1/1B2n3/5pP1/P2rPp2/3p1P2/2PP4/3K1R2`, `1rpq4/Q1p1n3/1ppb4/8/rkN3P1/1pp1P3/2PP1PPP/R1B2RK1` (pawn on rank 8)
- where+what kc=99, R-real (sparse endgames the classifier passes): `1k6/4P3/3P4/3pp3/1r6/4P3/6PP/6K1`, `5k2/4r1p1/4np2/8/4K3/5QP1/5PB1/2R3R1`
- where+what kc=99, R-legal: `8/2r3kb/8/1pN2p1n/PPPPp2p/2P5/4RP2/5K2`, `5r1k/3P1p2/PR6/2P3p1/1rr2PPP/3R2K1/2p5/2r5`, `8/2qnQ1n1/1k1bb3/1pp3p1/pp6/P2PpP2/2PpP3/3K4`
- where+what kc=99, R-non: `2P2N2/PP6/1K1R1B1R/3p4/3p4/8/1P1P1k2/K2R1b2`, `2pp3p/4p3/7b/2Q3n1/2Nq4/2R1B3/2Pn1p2/N3p3`, `6p1/7p/2rkQB1P/1r1PK1r1/1pnp2pN/1pppppP1/1pp1Pp2/8`

## Reading and the link to the text laws

1. **Where and what are separable in chess and they order the regimes.** Keeping each parsed unit's own pieces and shape, and moving only its anchor
   up the concept tree, takes boards from real (.83) to mostly illegal (.76) through a legal-unrealistic share that is thin (.12-.19) but real and
   monotone in neither direction (R-legal peaks at kc=2-3 and kc=99, .18-.19, five times the identity control's .055 judge noise). Swapping the pieces too
   (what) removes the legal band (R-legal .01-.05) and gives nonsense. So location is the "grammatical but meaningless" dial and identity is the "nonsense" dial.
2. **Why the legal band stays small.** Legality in chess has placement rules (no pawn on rank 1/8, no check on the side not to move) that no concept in the tree
   carries: the anchor read draws squares from where chunks stood, which at the root is every square. The text analogue (category order is preserved when words
   move between positions of the same category) fails here because a pawn chunk moved to rank 8 is not a pawn chunk in a legal position.
3. **Data does not move it** (flat 500-4000), as in r220/r221.

## Caveats

Two seeds; 600 boards per cell per seed; N=8000 not run; the judge is a piece-square classifier (passes some sparse endgames); "units kept" is 18-26% at kc=0
(no variant of the unit fits elsewhere) so kc=0 is partly copying the parse; the three representative cells were chosen after the grid; kings are moved as ordinary
singles/chunk members (no special treatment); adjacency tallies computed on the illegal boards only.
