# r227: file vs rank (the anchor split into its two facets) and what

Follows r226 (`r226_where_what.md`). Same judge (`r220_judge.py`, unchanged), learner (cs73L recipe), parser and held-out positions (shuffled training
lines [10000:10300]; 300 positions x 2 draws). Seeds 13 and 17; N = 500, 1000, 2000, 4000; 600 boards per cell per seed.
Code: `experiments/v3/diag/r227_file_rank.py`, `r227_report.py`. Data: `r227_v1_s13.json`, `r227_v1_s17.json`; full tables and boards: `r227_tables.md`.
Jobs: smoke 4808695; runs 4808718 (seed 13; requeued once by the cluster, final log from the second run), 4808719 (seed 17); report 4810943.

## Verdict: file-only abstraction gives the best legal-unrealistic band yet (R-legal up to .25), still short of .5; the cap is the check rate plus the judge

| rule | best cell | value |
|---|---|---|
| R-real >= .6 | where-file kc=0 (N=4000) | .890 |
| R-legal >= .5 | where-file kc=6 (N=500) | **.254** (fails); .233 at N=4000, kc=99 |
| R-non >= .6 | where-both+what kc=99 (N=4000) | .990 |

## Mapping onto the original architecture

Unchanged from r226: parse by the existing parser; anchors and concepts are reads of the existing chunk tree. The only change is that the redrawn
anchor from the stored anchors under the concept kc levels above the leaf (the code's anchor read) is used for ONE of its two coordinates:
- **where-file**: the unit keeps its pieces, shape and own rank; the file of its anchor is drawn from the stored anchors under the concept (kc=99: any file where the shape fits).
- **where-rank**: the mirror: keep the file, redraw the rank.
- where-both = r226's "where"; where-X+what additionally replaces the pieces by a stored instance of the concept (r226 B).
Singles (kings included) are 1-piece chunks: the new square is a stored square of singles of the same token (kc=0) or of any single (0<kc<99), uniform at 99; only the chosen facet is taken from it.
No filter and no new machinery. Check (side not to move in check) is read from python-chess only for reporting; no concept or feature in the trees carries it, so it is NOT used as a filter.

## Dial table (mean of 2 seeds; R-real / R-legal / R-non; "check" = share of ALL boards illegal ONLY by a check flag)

N = 4000:

| cell | R-real / R-legal / R-non | check-only |
|---|---|---|
| identity (control) | .945 / .055 / 0 | 0 |
| where-file kc=0 | .890 / .056 / .054 | .054 |
| where-file kc=1 | .790 / .139 / .073 | .073 |
| where-file kc=2 | .738 / .184 / .077 | .077 |
| where-file kc=3 | .718 / .206 / .076 | .076 |
| where-file kc=4 | .696 / .215 / .089 | .089 |
| where-file kc=6 | .696 / .216 / .087 | .087 |
| where-file kc=99 | .681 / .233 / .085 | .085 |
| where-rank kc=0 | .829 / .053 / .118 | .033 |
| where-rank kc=2 | .480 / .091 / .428 | .121 |
| where-rank kc=6 | .219 / .121 / .659 | .136 |
| where-rank kc=99 | .193 / .137 / .671 | .130 |
| where-both kc=99 (= r226 where) | .080 / .137 / .782 | .112 |
| where-file+what kc=1 | .171 / .061 / .768 | .051 |
| where-file+what kc=99 | .003 / .014 / .984 | .013 |
| where-both+what kc=99 | .002 / .009 / .990 | .018 |
| list, counts off / on | .099 / .003 / .898, .504 / .006 / .490 | .019, .061 |

N=1000: where-file kc=99 .706 / .219 / .074; where-rank kc=99 .212 / .150 / .637; where-both+what kc=99 .002 / .010 / .988.
All other cells are in `r227_tables.md`.

## Learning curves (R-real / R-legal / R-non), the three cells best at N=4000 per regime (chosen after the grid)

| N | real: where-file kc=0 | legal: where-file kc=99 | non: where-both+what kc=99 |
|---|---|---|---|
| 500 | .876 / .077 / .048 | .671 / .242 / .086 | .007 / .015 / .978 |
| 1000 | .873 / .069 / .058 | .706 / .219 / .074 | .002 / .010 / .988 |
| 2000 | .867 / .070 / .064 | .663 / .242 / .094 | .005 / .013 / .981 |
| 4000 | .890 / .056 / .054 | .681 / .233 / .085 | .002 / .009 / .990 |

Flat in N, as before.

## Findings

1. **The rank facet carries the placement rules; the file facet does not.** Redrawing the file leaves no pawn on rank 1/8 (zero PAWNS_ON_BACKRANK at every kc) and
   no count failure, so all illegal where-file boards are CHECK failures: the illegal share (.054 at kc=0 to .089) equals the check-only share to three decimals.
   Redrawing the rank reproduces r226's failures (back-rank pawns .10 to .53, check .04 to .31). Rank is "syntax" (it decides legality), file is "content".
2. **Check is what caps legality, quantified.** In where-file the check rate is .054 (kc=0) to .089 (kc=4-99) of all boards: a move of a chunk along its rank can put the king on an attacked line.
   Even if every check board were counted legal, where-file at kc=99 would give R-legal + R-non = .233 + .085 = .32; the other 68% are R-real by the judge. So check is
   not the binding cap on R-legal at .5: the judge is. Adjacent kings play no role (they are rarely among illegals, 2-8%).
3. **Why file shifts stay "real" for the judge.** The piece-square classifier is nearly file-symmetric for most pieces, so moving a chunk along its rank to another file mostly keeps the
   piece-square score above TAU (R-real .68 at kc=99). The boards that fail the judge's realism but stay legal (R-legal .22-.25) are those where kings or pawn structures land on
   the wrong wing (examples below). The "shifted-board" compression result (a shape-based coder handles shifted positions) is the same fact seen from the other side: file position is the weakly constrained coordinate.
4. **where-file+what and where-both+what stay the nonsense end** (R-non .77 at kc=1, .95-.99 for kc >= 2), because swapping pieces breaks king and pawn counts (no king / two kings / back-rank pawns).
5. **Net**: location-by-file is a clean dial from real (.89, kc=0) through legal-unrealistic (.23 at the root, 8.5% check-illegal) and the legal band is monotone in kc up to kc=3-4 (.14, .18, .21, .22, .22, .23), but it
   never approaches .5 because most file-shifted boards still look real to a piece-square judge.

## Boards (N=4000, seed 13; board field of the FEN, ranks 8 to 1; three per regime)

- where-file kc=0, R-real: `1r4k1/2pR1pp1/r1p1n1p1/p1B2P2/2P3P1/7P/1PP5/2K4R`, `1r1q1rk1/Qpp1n1pp/2pb4/p7/2N1P3/2P5/PP3PPP/R1B2RK1`
- where-file kc=0, R-legal: `r3k2r/pp3ppp/4pq2/5pp1/8/PnbP2P1/1PPBP3/R2QK1R1` (castled-looking queen side but king on e1 and rooks on a1 g1), `r3k2r/ppp2pp1/p4q2/1pp5/8/PnbP2P1/1PPBP3/R2QK1R1`, `r3k1nr/pp2ppbp/2p1b3/3p2p1/2PPP2n/1P3N2/1P1N1PPP/R1BK1B1R`
- where-file kc=99, R-real: `1r4k1/3R1ppp/4nrpp/2B2pP1/P3P3/5P2/2PP4/3K1R2`, `1r1qrk2/Q1ppnpp1/2pb4/2p5/4N1P1/4P3/2PP1PPP/2R1B1KR`
- where-file kc=99, R-legal: `1r1k4/pppnp3/1ppPn3/6q1/4PP2/2NPQ1P1/5R1P/4R1K1`, `r1k2r2/ppp1pp2/3q1p2/6pp/8/1P1bn1PP/2PP1PB1/3R1KQR`, `r3k1nr/ppp1p1bp/2p1b3/3p2p1/2PPPn2/1P3N2/1P1N1PPP/R1BK1B1R`
- where-file kc=99, R-non (check): `5q2/4pk2/p1R3n1/N3p3/1p6/2P5/1P1P1Q2/3K4`, `6q1/1p4k1/1pR4n/N4p2/2p5/2P5/2PP2Q1/3K4`, `5rk1/2ppp2q/4np2/2p4N/1pp3Q1/2P4P/PP2RPPK/8`
- where-both+what kc=99, R-real (sparse endgame the classifier passes): `1r6/2b1n3/4q3/4k3/8/8/P3P3/K7`
- where-both+what kc=99, R-legal: `1R6/1K1p2R1/3np3/1rk5/4p3/8/N7/8`, `8/2P1P3/P3r2q/2P1pp2/KPPPP2k/1RB1R1b1/8/1r2q3`, `3b2rk/8/5p2/2P5/1P4PP/6NK/4PP2/8`
- where-both+what kc=99, R-non: `3p1k2/3q1p2/5PpP/2pPPp2/3pk3/r1pp4/pbn5/8`, `p4pk1/4pBp1/2B1P3/P3r3/1r1b4/2b1n3/1p2q3/2p5`, `2PP1q2/2ppP3/4p2n/Rp1p2p1/p1prk2p/3Pp3/4Pp2/2PK4`

## Caveats

Two seeds; 600 boards per cell per seed; N=8000 not run; the realism judge is a piece-square classifier that is near file-symmetric (this limits R-legal, not the architecture);
"units kept" is 17-23% at where-file kc=0 (no variant fits elsewhere), under 2% at kc>=1; kings move as ordinary singles/chunk members; three representative cells chosen after the grid;
the check condition was not used as a filter (no existing concept or feature reads it) and is only reported.
