# Round 214 Phases 0-1: result (Phase 0 Kill A fired; Phase 1 not run)

## Verdict: NO-GO for knob A as designed (Kill A); Phase 1 was not launched

The design's Phase 0c stop rule: "no band meets purity >= .8, entropy >= 3 bits and selection retained <= .5 -> R-syn by level is
not available; switch knob A to the `window_names: "far"` / `nb_bits` fallback and re-plan". It fires on seed 13 and on seed 17
(n=1000, paper recipe: Trellis3R204 edge exact, window 7, bands naming, cu tree, posterior store). No band comes close: slot purity
reaches .8 only at the leaf, where filler entropy is 0.03 bits; at the one band where entropy is still >= 4 bits (8 bits) purity is .39-.40.
So the read-level dial does not pass through a "syntactic slot" band in this learned tree, and the Phase 1 probe (15 arms), the judge
step and the go rule were not run. The go rule is therefore not evaluated: per-arm judge purities, novelty quantities and sample
sentences do not exist for this round.

## Phase 0 code (done, tests pass)
- `src/trellis2/trellis3_r214.py`: `Trellis3R214(Trellis3R208)` with `read_bits`, `read_reach`, `parse_bits` (all None = Trellis3R208).
  Also `lift(leaf, b)`, `level_bits`, `word_classes(b)`. Parent caches (`_gap_cache`, `_edge_cache`, `_r208_cache`) are swapped for a
  lifted read and restored, because their keys do not hold b.
- `tests/trellis2/test_r214_level.py` (7 tests: default byte-identical for _gap_fills, hier_fills, _gain, _code, parse; infinite bits and
  full reach equal default; b=0 is the unigram with edge_exact boundaries; monotone bounded lift; reach 1; parse_bits 0; restoration and
  no cache poisoning; the dial changes the mass). Full `tests/trellis2` on a compute node: 134 passed (job 4793372).
- Driver: `experiments/v3/diag/r214_story.py` wraps `t3_story.py` (read, patched in memory at four exact-match anchors, executed as
  `__main__`); `t3_story.py` is unedited. Arm keys `bits`, `wreach`, `pbits`. No running or pending job used t3_story.py
  (squeue checked: r214eval runs r214_judge2.py, v3-r213c runs the chess/lang r213 scripts). Env `R214_LEVELS=<out.json>` dumps the
  occurrence table without generation. `experiments/v3/diag/r214_p1.sh` is the Phase 1 launcher (15 arms in 3 jobs); it was written and NOT submitted.
- `experiments/v3/diag/r214_levels.py` (env cobint, spaCy en_core_web_sm tags each training sentence in context).

## Level map (committed word occurrences read at C_b; seed 13 then seed 17, n=1000)
Slot purity = occurrence-weighted share of the majority universal POS (spaCy) of the fillers under each C_b. Filler entropy =
occurrence-weighted mean H(filler | C_b), bits. Selection retained = I_b(filler; l2) / I_leaf(filler; l2), where I_b is the mutual
information of the class-model joint sum_C P(C) P(f|C) P(v|C) (my operationalisation of "selection by the word two back that a read at
level b can use"; I(f; l2) empirical = 2.566 bits at seed 13, 2.568 at seed 17, equal to the leaf model). Pass band: purity >= .8,
entropy >= 3, selection <= .5.

### seed 13

| band b (bits) | concepts | mean level (bits) | slot purity | filler entropy (bits) | selection retained | / empirical | meets pass band |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.00 | 0.177 | 6.94 | 0.000 | 0.000 | no |
| 1 | 1 | 0.00 | 0.177 | 6.94 | 0.000 | 0.000 | no |
| 2 | 3 | 1.51 | 0.224 | 6.51 | 0.101 | 0.101 | no |
| 4 | 11 | 3.26 | 0.274 | 5.98 | 0.186 | 0.186 | no |
| 6 | 38 | 5.28 | 0.327 | 5.20 | 0.238 | 0.238 | no |
| 8 | 136 | 6.96 | 0.397 | 4.01 | 0.361 | 0.361 | no |
| 12 | 1523 | 10.60 | 0.676 | 1.40 | 0.747 | 0.747 | no |
| inf | 5112 | 13.15 | 0.995 | 0.03 | 1.000 | 1.000 | no |
| basic | 7 | 2.17 | 0.249 | 6.25 | 0.122 | 0.122 | no |

seed 13, n=1000, window 7; 5306 committed word occurrences (weight 5306.0), 239 word types, 1000 sentences; I(filler; l2) empirical = 2.566 bits, class-model at the leaf = 2.566 bits; owner mismatches 0

pass band: none (Kill A)

### seed 17

| band b (bits) | concepts | mean level (bits) | slot purity | filler entropy (bits) | selection retained | / empirical | meets pass band |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.00 | 0.176 | 6.89 | 0.000 | 0.000 | no |
| 1 | 1 | 0.00 | 0.176 | 6.89 | 0.000 | 0.000 | no |
| 2 | 3 | 1.47 | 0.217 | 6.48 | 0.091 | 0.091 | no |
| 4 | 10 | 3.25 | 0.293 | 5.91 | 0.165 | 0.165 | no |
| 6 | 40 | 5.21 | 0.311 | 5.23 | 0.221 | 0.221 | no |
| 8 | 143 | 6.99 | 0.388 | 4.05 | 0.345 | 0.345 | no |
| 12 | 1550 | 10.62 | 0.663 | 1.48 | 0.731 | 0.731 | no |
| inf | 5269 | 13.18 | 0.996 | 0.03 | 1.000 | 1.000 | no |
| basic | 7 | 2.23 | 0.251 | 6.17 | 0.114 | 0.114 | no |

seed 17, n=1000, window 7; 5502 committed word occurrences (weight 5502.0), 245 word types, 1000 sentences; I(filler; l2) empirical = 2.568 bits, class-model at the leaf = 2.568 bits; owner mismatches 0

pass band: none (Kill A)

## Reading
- Coarse concepts of the context tree (the 2-8 bit bands) are not part-of-speech slots: purity .22-.40 against .18 for the root. The tree
  groups occurrences by their whole description (window words, neighbour bags, sentence facets), not by what fills the gap, so a
  coarse concept mixes noun, verb and function-word fillers.
- Selection by the word two back falls smoothly with b (retained .10 at 2 bits, .36 at 8, .74 at 12, 1 at the leaf); there is no band
  where selection is gone and the category is kept. The requirements purity >= .8 and selection <= .5 are never met together:
  purity is >= .66 only where selection retained is >= .73.
- The basic level (7 concepts, mean 2.2 bits) is as coarse as the 2-4 bit band: purity .25, entropy 6.2 bits.
- Deviations from the design: (1) seed 13 is the specified map; seed 17 was added as a replicate (the design asks for 13 and 17); (2) the
  selection-retained statistic is the class-model MI ratio above (the design's I(filler; l2 | C_b) read literally rises with b and cannot
  "survive" a lift, so the model-joint reading was used); (3) Phase 1 and the judge gate were not run because Kill A fired.
- Next step per the design: re-plan knob A with the `window_names: "far"` / `nb_bits` fallback (relearn per setting), or change the
  hypothesis (the level map says the tree has no slot level to read at). Knob B (read_reach) and parse_bits are implemented and tested
  but untouched by this verdict.

Files: `experiments/v3/results/story_gen/r214p01/levels_s{13,17}_n1000.json` (occurrence dumps), `r214_levelmap_s{13,17}.{md,json}`.
