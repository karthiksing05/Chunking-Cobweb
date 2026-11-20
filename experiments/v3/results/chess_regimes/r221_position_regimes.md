# r221: count facet, context-level dial kx, and parse-then-regenerate (chess position regimes)

Follows `r220_position_regimes.md` (same judge, `r220_judge.py`, unchanged; same learner, cs73L recipe; same train/judge line splits).
Code: `experiments/v3/diag/r221_position_gen.py` (count facet x kc x kx), `r221_parse_regen.py` (parse then regenerate),
`r221_analyze.py`, `r221_pr_report.py`. Data: `r221_gen_v1_s13.json`, `r221_gen_v1_s17.json`, `r221_pr_v1_s13.json`, `r221_pr_v1_s17.json`.
Full tables: `r221_tables.md`, `r221_pr_tables.md`. Two seeds (13, 17); N = 500, 1000, 2000, 4000.
Jobs: r221 smoke 4806789; first full runs 4806898/4806899 cancelled (a kc=99 cell was too slow, sampler rewritten); final runs
4807047 (seed 13), 4807048 (seed 17); analysis 4807777 (first, later re-run on the final data as 4806xxx-era job, output in `r221_tables.md`);
parse-regen first runs 4807778/4807779 crashed (KeyError, fixed by backing off to the nearest ancestor concept with stored instances);
final parse-regen 4807838 (seed 13), 4807839 (seed 17); report 4808277.

## Verdict: NO-GO on the three-regime rule, with a clean partial result

| test | R-real >= .6 | R-legal >= .5 | R-non >= .6 |
|---|---|---|---|
| r220 content dial kc only | no (max .10) | no (max .015) | yes (.88-.99) |
| r221 count facet + kx dial | no (max .54, counts=1 list) | no (max .098) | yes |
| r221b parse then regenerate | **yes (.76-.79, kc=0)** | no (max .058) | yes (.93-.98 for kc >= 2) |

R-legal never appears. Parse-then-regenerate gives two regimes, R-real at the leaf concept and R-non at kc >= 2, with kc=1 in between
(R-real .18-.23, R-non .72-.79), and nothing legal in between.

## 1. Mapping onto the original architecture

**Count facet (representation change).** Each learned position stores its whole-board piece counts (12 piece kinds, kings included) as a
stored attribute of the position, learned by counting exactly as TOP counts and single_tops are (the "read counts" of the old v2 grammar,
FRAMEWORK.md: how many of each kind the board holds). I did NOT add a new Cobweb tree for it: the stored concept is the distribution
over the stored positions' profiles (the root concept of the position). Generation draws a profile from it, then draws units as before
and renormalises over the units whose pieces the profile still has (the existing renormalisation over what stays feasible, as for gaps
that keep a unit on the board; units that do not fit are redrawn or dropped as in r220). The stop item is replaced by the profile being
used up; after 12 failed unit draws in a row the rest of the profile stays unplaced (reported as drop/board). Not learned the same way
as the content/context features (those are Cobweb attributes); it is a counted facet, the closest existing analogue is TOP's counts.

**kx dial (existing context head).** The code's context head (CTXR=1 in `chess_compress_gen.py`): the unit's anchor is drawn from the anchor
squares of the stored top chunks (occupied squares left out), the star from the anchor over the units decoded so far (rays, jumps, square)
is sorted in the context tree L.ctx, and the unit's content is drawn by FILL at the concept kx levels above the sorted leaf: a chunk's
content concept at level kc by the share of FILL chunks under it, then a stored instance under that concept that fits at the anchor (25
draws); a single piece by FILL's piece fillers (square from the single-piece squares). kx=None is r220's concept-first route.
Not used: the learned-weight mixture of concept predictions (CTXR=3) and part slots. Caveat: the context tree learned full-board stars,
while generation sorts partial boards, as in the code's context head.

**Parse then regenerate (r221b).** Parse = the existing parser (GraphTrellis.parse, learning off) on 300 held-out real positions (shuffled
training lines [10000:10300], never learned, disjoint from the judge's real lines). Replacement = the existing concept-level sampling: each
parsed chunk's content is sorted in L.cnt, the concept kc levels above its leaf is taken (root = 99), a stored top-chunk instance under it is
drawn (the copy route, instances by offsets) and placed at the chunk's own anchor; if no draw fits in 30 tries the chunk stays (reported as
"chunks kept"; backing off to the nearest ancestor with stored instances when the sorted concept has none). Singles (kings included) either
stay ("keep") or are redrawn ("resample": kc=0 unchanged; 0<kc<root a token from the stored singles on that square; root a token from all stored
singles). Identity = the real board through the parser (control).

## 2. r221 grid: count facet x kc x kx (2 seeds, 200 boards per cell per seed; full grid for N=1000 and 4000 in `r221_tables.md`)

N = 4000, R-real / R-legal / R-non:

| counts | kc | kx=None | kx=0 | kx=2 | kx=root |
|---|---|---|---|---|---|
| off | 0 | .085 / .007 / .907 | .045 / .010 / .945 | .007 / .013 / .980 | .003 / .010 / .988 |
| off | 2 | .030 / .010 / .960 | .003 / .007 / .990 | .010 / .003 / .988 | .005 / .007 / .988 |
| off | root | .007 / .007 / .985 | .005 / .013 / .982 | .003 / .007 / .990 | .007 / .010 / .982 |
| on | 0 | .417 / .025 / .557 | .185 / .045 / .770 | .050 / .040 / .910 | .015 / .065 / .920 |
| on | 2 | .280 / .098 / .623 | .035 / .053 / .912 | .040 / .025 / .935 | .013 / .057 / .930 |
| on | root | .020 / .045 / .935 | .033 / .065 / .903 | .013 / .053 / .935 | .018 / .058 / .925 |
| list | - | counts off .098 / .003 / .900 | counts on .528 / .005 / .468 | | |

Findings: the count facet is what moves legality (R-non drops from .90-.99 to .47-.62 at counts on with kc <= 2, kx=None), and it makes boards
real-like (R-real .28-.42, list .53), but the legal-and-unreal class stays at .02-.10. Higher kx (the broader context concept) trades R-real for
back-rank pawns (PAWNS_ON_BACKRANK .84-.87 at kx >= 2 with counts on) and checks (OPPOSITE_CHECK .25-.49): the context head, sorting partial
boards, places pieces on squares a real position never has, so kx does not give a legal regime either.

Learning curves (R-real / R-legal / R-non; the three cells best at N=4000 on each regime, chosen after the grid):

| N | real: counts on, piece list | legal: counts on, kc=2, kx=None | non: counts off, kc=2, kx=0 |
|---|---|---|---|
| 500 | .468 / .003 / .530 | .225 / .082 / .693 | .007 / .005 / .988 |
| 1000 | .540 / .007 / .453 | .235 / .087 / .677 | .010 / .003 / .988 |
| 2000 | .512 / .003 / .485 | .250 / .077 / .672 | .010 / .000 / .990 |
| 4000 | .528 / .005 / .468 | .280 / .098 / .623 | .003 / .007 / .990 |

## 3. r221b: parse then regenerate (2 seeds, 600 boards per cell per seed)

R-real / R-legal / R-non; "kept" = share of parsed chunks for which no stored instance fitted at the anchor (the chunk stayed).

| cell | N=1000 | N=4000 | kept (N=4000) |
|---|---|---|---|
| identity (control) | .945 / .055 / 0 | .945 / .055 / 0 | 0 |
| keep kc=0 | .672 / .036 / .291 | .762 / .051 / .188 | .179 |
| keep kc=1 | .179 / .058 / .762 | .228 / .049 / .722 | .010 |
| keep kc=2 | .037 / .021 / .943 | .043 / .019 / .938 | .009 |
| keep kc=3 | .021 / .018 / .962 | .020 / .019 / .960 | .009 |
| keep kc=4 | .010 / .021 / .969 | .018 / .018 / .965 | .015 |
| keep kc=6 | .010 / .025 / .966 | .005 / .015 / .980 | .013 |
| keep kc=99 | .007 / .021 / .972 | .007 / .018 / .974 | .012 |
| resample kc=0 | .682 / .042 / .275 | .762 / .050 / .188 | .178 |
| resample kc=2 | .032 / .017 / .952 | .055 / .019 / .926 | .010 |
| resample kc=99 | .005 / .017 / .978 | .005 / .017 / .980 | .014 |

(keep and resample coincide at kc=0 by construction; the rest of the resample grid is in `r221_pr_tables.md`.)
Learning curves (R-real / R-legal / R-non): resample kc=0 .667/.041/.291 (500), .682/.042/.275 (1k), .790/.048/.161 (2k), .762/.050/.188 (4k);
resample kc=6 non .972, .968, .976, .984. R-legal at kc=0 equals the judge's own false-reject rate on real boards (.055 in the identity row).

What the result says. The real structure is kept (anchors, king and pawn placement come from the parse) yet legality does NOT survive abstraction:
at kc=1, 72-79% of boards are illegal, and at kc >= 2 more than 93%. The replaced chunk instances carry their own kings, pawns and pieces
(a chunk drawn under a concept with a king puts a second king; one without removes the real king; chunk pawns land on rank 1/8), so the parser's
whole-board counts are not preserved by the replacement, only by the singles. kc=0 retains legality because the leaf concept's instances
are near copies of the chunk (and 18-26% of chunks stay because no variant fits). So the dial gives real (kc=0) then nonsense (kc >= 1),
with the legal-unrealistic step absent: what is lost between kc=0 and kc=1 is the exact chunk content (which pieces, where), and legality
breaks together with realism.

## 4. Which whole-board property separates legal-unrealistic from real

Across all kept legal and real boards of the r221 grid (both seeds, all N; 1072 R-legal boards, 3014 R-real boards; `r221_tables.md`),
decomposed by the unchanged judge's feature groups (logit contribution), with real held-out and shuffled-legal controls:

| property | generated R-real | generated R-legal | real held-out | shuffled legal |
|---|---|---|---|---|
| piece-square group (logit) | +3.25 | -3.12 | +4.48 | -6.94 |
| kind-count group | 1.34 | 0.90 | 0.45 | 0.73 |
| pawns-per-file group | 0.09 | -0.31 | 0.16 | -0.20 |
| white king rank (mean) | 1.55 | 2.87 | 1.52 | 4.35 |
| black king rank (mean) | 7.25 | 5.57 | 7.38 | 4.41 |
| pieces | 16.8 | 18.2 | 21.4 | 19.9 |
| \|material diff\| | 6.7 | 5.6 | 1.7 | 1.8 |

The separating property is piece placement, and above all king placement: the piece-square group of the judge flips sign (+3.3 to -3.1) and
the kings leave their back ranks (white king rank 1.6 to 2.9, black king rank 7.3 to 5.6; real positions 1.5 and 7.4). Piece counts, pawn
counts per file and material do not separate the two classes (kind-count group .9-1.3, pawns .09 vs -.31, material difference high in BOTH generated classes
at 5.6-6.7 against 1.7 for real boards, so material balance is not what the judge's R-real tests; it is a weak feature of the classifier and
generated boards of both classes are imbalanced). So legal-but-unrealistic boards are those with the right counts and no illegal placements
whose pieces stand where real positions never have them (kings walked out, pieces on unusual squares). Honest limit: this is from the r221
grid; I did not run the property decomposition on the r221b boards, and the 'real' class includes some sparse endgames the classifier passes.

## 5. Connection to the text laws and conclusion

In text, legality (syntax) is carried by category sequences that the abstract concept level keeps; here the count constraints are whole-board and no
concept level in the content or context tree carries them. Giving the generator the counts (facet) restores most legality at the leaf level, and
parse-then-regenerate keeps the real structure, but replacing a chunk at any level above its leaf swaps its pieces, so the counts break with it.
In every route, abstraction moves boards from real to illegal in one step; the legal-unrealistic regime is a narrow band (R-legal <= .10, never
above .06 for parse-regen) that no existing dial lands on. The text analogue's middle regime exists because categories are interchangeable at
fixed syntax; in chess a chunk's category is not interchangeable at fixed counts. Stopped as instructed: no further machinery.

## Caveats

Two seeds; 200 (r221) or 600 (r221b) boards per cell per seed; N=8000 not run; the realism judge is a piece-square classifier (passes some
sparse endgames, weak on material balance); the count facet is a counted profile, not a Cobweb attribute; the context head sorts partial boards
in a tree learned on full-board stars; r221b backs off to the nearest ancestor concept that has stored instances; the three "representative"
cells were chosen after the grid; the first analysis/parse-regen runs were superseded (job list above).
