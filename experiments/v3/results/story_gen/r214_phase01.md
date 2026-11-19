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

# Addendum: Phase 0' (word-class map for the window_names / nb_bits fallback of knob A) — gate fails; Phase 1' not run

Learner: seed 13 and seed 17, n=1000, the paper recipe plus `type_names: "sample"` (the only added flag; it fills the per-type
reservoirs that `_res_bag` reads, `type_k` 8 and `type_top` 40 at their defaults; `name_level`, `name_mix_k` and `window_names` stay off, so
the tree is the token-window tree). Name of word w at nb_bits b = the rule `Trellis3._window_name` / `_named_window` uses: the most common
concept in the bag of the concept ids on w's reservoir paths, bands up to b (b = 0: no band, all words share one name; None: every band).
Words are grouped by that name; purity = token-weighted share of the majority spaCy UPOS (tagged in the sentence) per class; entropy =
token-weighted H(word | class). Gate: purity >= .8 and >= 8 classes with >= 3 words. Code: `experiments/v3/diag/r214_wordmap.py`
(env cobint) on the dump of `r214_story.py` with env `R214_WORDMAP`.

### seed 13

seed 13, n=1000, 239 word types, 5306 tokens, 7077 reservoir types

| nb_bits b | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 239 | 0.177 | 6.94 | no |
| 1 | 3 | 3 | 117 | 0.248 | 5.67 | no |
| 2 | 3 | 3 | 117 | 0.248 | 5.67 | no |
| 4 | 4 | 4 | 106 | 0.280 | 5.29 | no |
| 6 | 8 | 5 | 104 | 0.284 | 5.16 | no |
| 8 | 15 | 5 | 99 | 0.294 | 5.06 | no |
| 12 | 15 | 6 | 98 | 0.299 | 4.97 | no |
| none | 15 | 6 | 98 | 0.299 | 4.97 | no |

b=4, five biggest classes by tokens: they was the he it a lily said | play you with fun together friends sad day | and to were her in what saw at | happy very are unexpected scared felt feel many
b=8, five biggest classes by tokens: they was the he it a lily said | play you with fun together friends sad too | and to were her in what saw at | happy very are unexpected scared felt feel my | park boy long sun sky
b=none, five biggest classes by tokens: they was the he it a lily said | play you with fun together friends sad too | and to were big her in what saw | happy very so are unexpected scared felt an | park boy long sun

passing b: none

### seed 17

seed 17, n=1000, 245 word types, 5502 tokens, 7587 reservoir types

| nb_bits b | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 245 | 0.176 | 6.89 | no |
| 1 | 3 | 3 | 119 | 0.254 | 5.45 | no |
| 2 | 3 | 3 | 119 | 0.254 | 5.45 | no |
| 4 | 4 | 4 | 119 | 0.254 | 5.31 | no |
| 6 | 11 | 5 | 113 | 0.275 | 5.14 | no |
| 8 | 22 | 5 | 111 | 0.323 | 4.84 | no |
| 12 | 23 | 5 | 109 | 0.297 | 4.85 | no |
| none | 23 | 5 | 109 | 0.297 | 4.85 | no |

b=4, five biggest classes by tokens: and they the he tim a were she | to happy it had play with fun friends | was very sad so not something unexpected an | you yes felt let's just both kind learned
b=8, five biggest classes by tokens: and they the he tim a were she | to happy it had play with fun friends | was sad so not an surprised man careful | you yes felt let's just both kind learned | very
b=none, five biggest classes by tokens: was and they the he a were she | to happy it had play with fun friends | sad so not bird an man careful proud | tim you yes felt let's just kind learned | very

passing b: none

## Verdict (Phase 0')
No nb_bits passes on either seed: POS purity is at most .30 (seed 13) and .32 (seed 17) against .18 for one class, and at most 6
classes hold >= 3 words (the gate needs 8). The learned names group the 240 words into 3 to 23 classes whose biggest holds 98-119 words;
b = 4 to None change little (15 vs 23 classes). The names come from the token-window tree whose coarse concepts are not part-of-speech
slots (the Phase 0 level map), so naming the window by them does not give syntactic classes. Per the coordinator's rule I stopped:
no window_names learners (about 6), no knob-B reach arms, no temperature control, no judge step, no go rule, no per-arm purity or novelty
table, no samples. Nothing was generated.

Caveat on the gate: the name is the single most common concept of the bag, which tends to be the coarsest shared concept; a name built from
the full bag (several concepts per word) or from a learner trained with `window_names` on (names inside the tree, so a name is refined by
earlier names) is a different map and was not tried. Both are options for the re-plan, as are classes from another source (for example
type-level distributional clustering) with the same read.

## Code added in this step
- `Trellis3R214._named_query`: with `window_names` set, `_gap_fills` names the query's window words as the stored contexts are named
  (default, `window_names` None, unchanged); test 7 checks it against the parent read of the named query. Full `tests/trellis2`: 135 passed (job 4793529).
- `r214_story.py` env `R214_WORDMAP=<out.json>` (no generation) and `r214_wordmap.py`.

# Addendum 2: track 2, a type-level word-category tree (Cobweb only): gate fails; probe arms not run

Tree: `trellis2.trellis3_r214.WordTree`. One instance per word type of the training stream (n=1000), described by two bag attributes,
L = the type's left-neighbour distribution and R = its right-neighbour distribution (probabilities over the words that stood there,
`<s>` / `</s>` included), the Brown / Schuetze description. It is sorted by the repo's `cu_cobweb.CUCobwebTree` (compiled Fisher Cobweb,
the tree TRELLIS uses; bags as TRELLIS's facets; `exact_alone` on), types inserted in order of first occurrence, seed = the run seed. No
clustering library, no gradient, no new Cobweb. A type's class at level b = its leaf's deepest ancestor holding at least a 2^-b share of
the types (level -log2(count / root count) <= b); `dK` = the ancestor at depth K. Batch over the stream's types (the incremental variant
was not built). Same columns as the word-class map; purity = token-weighted spaCy UPOS majority per class.

Opt-in wiring (built, tested): `Trellis3R214.word_names` (word -> class name) replaces the reservoir names as the `window_names` source for
the learner's stored contexts and for the read's query alike (`_named_window`; None = unchanged); driver cfg key `"wtree": b | "leaf"`
builds the tree from the stream and sets it (window_names "all" by default). Test 8 checks that every stored window slot is a class
name and that the read of a token query equals the read of its named query. Full `tests/trellis2`: 136 passed. Default path byte-identical (tests 1-2).

### seed 13

| level (bits, or dK = depth K) | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 239 | 0.177 | 6.94 | no |
| 1 | 2 | 2 | 128 | 0.284 | 5.94 | no |
| 2 | 4 | 4 | 96 | 0.365 | 5.17 | no |
| 3 | 8 | 8 | 53 | 0.439 | 4.21 | no |
| 4 | 14 | 12 | 53 | 0.503 | 3.62 | no |
| 5 | 27 | 23 | 22 | 0.574 | 2.82 | no |
| 6 | 50 | 40 | 14 | 0.648 | 2.09 | no |
| 6.5 | 69 | 50 | 10 | 0.722 | 1.46 | no |
| 7 | 97 | 37 | 5 | 0.775 | 1.13 | no |
| 7.25 | 97 | 37 | 5 | 0.775 | 1.13 | no |
| 7.5 | 97 | 37 | 5 | 0.775 | 1.13 | no |
| 7.75 | 97 | 37 | 5 | 0.775 | 1.13 | no |
| leaf | 239 | 0 | 1 | 0.965 | 0.00 | no |
| d1 | 3 | 3 | 128 | 0.365 | 5.47 | no |
| d2 | 12 | 9 | 96 | 0.427 | 4.59 | no |
| d3 | 55 | 17 | 43 | 0.509 | 3.38 | no |
| d4 | 109 | 17 | 30 | 0.647 | 2.37 | no |
| d5 | 157 | 13 | 24 | 0.735 | 1.60 | no |
| d6 | 198 | 10 | 14 | 0.812 | 0.79 | pass |
| d8 | 238 | 0 | 2 | 0.965 | 0.01 | no |

level d6, five biggest classes by tokens: to a is mom not what will look | and was said | he she tim amy | together friends too friend sorry him again | they
level 7, five biggest classes by tokens: and was said | they but lucy | he she tim amy | lily tom sue max anna | happy sad surprised
level 7.25, five biggest classes by tokens: and was said | they but lucy | he she tim amy | lily tom sue max anna | happy sad surprised
passing b: ['d6']

### seed 17

| level (bits, or dK = depth K) | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 245 | 0.176 | 6.89 | no |
| 1 | 2 | 2 | 141 | 0.278 | 5.90 | no |
| 2 | 6 | 6 | 75 | 0.332 | 4.90 | no |
| 3 | 8 | 8 | 54 | 0.363 | 4.47 | no |
| 4 | 13 | 12 | 38 | 0.477 | 3.81 | no |
| 5 | 33 | 27 | 17 | 0.623 | 2.75 | no |
| 6 | 54 | 40 | 12 | 0.675 | 1.88 | no |
| 6.5 | 70 | 52 | 9 | 0.729 | 1.55 | no |
| 7 | 98 | 39 | 6 | 0.795 | 1.12 | no |
| 7.25 | 98 | 39 | 6 | 0.795 | 1.12 | no |
| 7.5 | 98 | 39 | 6 | 0.795 | 1.12 | no |
| 7.75 | 98 | 39 | 6 | 0.795 | 1.12 | no |
| leaf | 242 | 1 | 4 | 0.966 | 0.00 | no |
| d1 | 3 | 3 | 141 | 0.325 | 5.46 | no |
| d2 | 13 | 10 | 94 | 0.457 | 4.28 | no |
| d3 | 43 | 14 | 79 | 0.521 | 3.76 | no |
| d4 | 89 | 17 | 75 | 0.575 | 3.10 | no |
| d5 | 146 | 12 | 41 | 0.698 | 1.92 | no |
| d6 | 183 | 7 | 27 | 0.779 | 1.24 | no |
| d8 | 221 | 6 | 7 | 0.878 | 0.36 | no |

level d8, five biggest classes by tokens: was and a felt is just feel | were had all are | they | the | to for
level 7, five biggest classes by tokens: was and a felt feel | the can this | were had all are | they let's it's | he she lucy
level 7.25, five biggest classes by tokens: was and a felt feel | the can this | were had all are | they let's it's | he she lucy
passing b: none

## Verdict (track 2)
The word tree is far purer than the learned-name map (Phase 0'): at 7 bits purity is .775 (seed 13) and .795 (seed 17) with 97-98
classes, 37-39 of them holding >= 3 words (largest 5-6 words); at 6 bits .648 / .675 with 50-54 classes. But the gate (purity >= .8 and >= 8
classes of >= 3 words) is not passed on both seeds at any bit level: the closest level is **7 bits** (a finer grid, 7.25-7.75, gives the
same classes as 7 because the tree has no node between 6.5 and 8 bits' worth of share). By depth, depth 6 passes on seed 13 only (.812, 10
classes of >= 3 words) and misses on seed 17 (.779). Purity rises with fineness while the classes shrink: .8 purity arrives only when classes hold
2 to 3 words on average, so there is no band of "slots that hold many words". Per the coordinator's rule (fail: report the purity table and
the closest level) the window_names probe arms were not run. What the table does show: the level whose classes are about half-pure
(4-5 bits, 14-33 classes) has large classes (22-53 words) of mixed function words, names and nouns; the Cobweb partition of
neighbour distributions is a clean syntax/semantics cut only near the leaves.

# Addendum 3: track 1 (knob B, temperature control) and track 2 probe (word-tree window names), judged by r214_judge2

Judge: `r214_judge2.py score` (ensemble Qwen3-8B + phi-4, thresholds `r214_judge2_thresholds.json`: syntax pass at the low thresholds, see `r214_judge2.md`);
regimes R-sem = syn_pass and sem_pass, R-syn = syn_pass and not sem_pass, R-non = not syn_pass; purity = share of the arm's distinct new
sentences (not a stored training sentence). Novelty as r204_fair: A = 1600 attempted draws, tri|d, fair_X = distinct new in regime X with an
unstored trigram per attempted draw, repeats (cut = proper spans of stored sentences). Seed 13, n=1000, edge exact, gate off, tau 1 unless
noted; the token learner is the paper recipe (r212/r204 edge, window 7). Drivers `r214_p1b.sh` (track 1), `r214_p1c.sh` (probe),
`r214_phase01.py` (collect / report / report211); dumps in `r214p01/{b1,b2,t_<b>}`, judge outputs in `r214p01/judge{1,2,211}`.
The wrapper's arm key `wreach` is the read reach W' (query keeps l1..lW'); `wtree` b = window names from the word tree at b bits, "leaf" = each
word its own class (identical to the token learner: the wtree-leaf rows equal the token rows to every digit, a check of the wiring).

## Track 1: knob B (reach) and temperature, token learner

| group | arm | seed | distinct new | dist/draw | R-sem | R-syn | R-non | (non with sem_pass) | tri\|d | tri\|d in sem / syn / non | fair sem / syn / non | repeats (cut) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reach | W'=1 t=1 | 13 | 1544 | 0.965 | 0.157 | 0.159 | 0.684 | 0.003 | 0.955 | 0.786 / 0.951 / 0.994 | 0.119 / 0.146 / 0.656 | 56 (10) |
| reach | W'=2 t=1 | 13 | 1463 | 0.914 | 0.334 | 0.174 | 0.491 | 0.005 | 0.645 | 0.260 / 0.655 / 0.904 | 0.079 / 0.104 / 0.406 | 137 (47) |
| reach | W'=3 t=1 | 13 | 1432 | 0.895 | 0.290 | 0.155 | 0.555 | 0.005 | 0.779 | 0.376 / 0.793 / 0.986 | 0.098 / 0.110 / 0.490 | 168 (16) |
| reach | W'=5 t=1 | 13 | 1474 | 0.921 | 0.196 | 0.145 | 0.659 | 0.008 | 0.925 | 0.671 / 0.953 / 0.995 | 0.121 / 0.127 / 0.604 | 126 (1) |
| temp | W'=7 t=1 | 13 | 1468 | 0.917 | 0.193 | 0.140 | 0.668 | 0.009 | 0.934 | 0.696 / 0.966 / 0.996 | 0.123 / 0.124 / 0.610 | 132 (1) |
| temp | W'=7 t=2 | 13 | 1583 | 0.989 | 0.056 | 0.097 | 0.846 | 0.002 | 0.990 | 0.854 / 0.987 / 0.999 | 0.048 / 0.095 / 0.837 | 17 (0) |
| temp | W'=7 t=4 | 13 | 1600 | 1.000 | 0.016 | 0.041 | 0.943 | 0.001 | 0.999 | 0.920 / 1.000 / 1.000 | 0.014 / 0.041 / 0.943 | 0 (0) |
| temp | W'=7 t=8 | 13 | 1600 | 1.000 | 0.007 | 0.030 | 0.963 | 0.001 | 1.000 | 1.000 / 1.000 / 1.000 | 0.007 / 0.030 / 0.963 | 0 (0) |


### Samples (5 per arm)

**reach / W'=1 t=1**
- they did it is not careful  (syn_p 0.00, sem_p 0.00, non)
- will be careful  (syn_p 0.00, sem_p 0.03, non)
- sara and ben tom and the animals  (syn_p 0.00, sem_p 0.00, non)
- sky opened new friends  (syn_p 0.01, sem_p 0.01, non)
- max loved to were very sad  (syn_p 0.00, sem_p 0.00, non)

**reach / W'=2 t=1**
- tim's was happy and excited  (syn_p 0.00, sem_p 0.53, non)
- they loved to play with lily  (syn_p 0.94, sem_p 0.93, sem)
- the car to it  (syn_p 0.00, sem_p 0.00, non)
- it was so happy  (syn_p 1.00, sem_p 1.00, sem)
- sue not too  (syn_p 0.00, sem_p 0.00, non)

**reach / W'=3 t=1**
- but house was not too  (syn_p 0.00, sem_p 0.00, non)
- lily liked to play in the animals  (syn_p 0.59, sem_p 0.00, syn)
- it was big happy shiny  (syn_p 0.00, sem_p 0.04, non)
- lily felt scared  (syn_p 1.00, sem_p 1.00, sem)
- he could not like it  (syn_p 0.96, sem_p 0.94, sem)

**reach / W'=5 t=1**
- we have be friends  (syn_p 0.00, sem_p 0.01, non)
- his mom said no lily  (syn_p 0.06, sem_p 0.01, syn)
- lily was so excited  (syn_p 0.99, sem_p 0.99, sem)
- tom is bad happy  (syn_p 0.00, sem_p 0.00, non)
- they played together all the long  (syn_p 0.00, sem_p 0.00, non)

**temp / W'=7 t=1**
- they were to play too  (syn_p 0.01, sem_p 0.36, syn)
- tom was his new home very much  (syn_p 0.00, sem_p 0.00, non)
- can want play with you  (syn_p 0.00, sem_p 0.00, non)
- let's said find some food  (syn_p 0.00, sem_p 0.00, non)
- tim dog was playing with a ball  (syn_p 0.47, sem_p 0.98, sem)

**temp / W'=7 t=2**
- then a boy came in the park  (syn_p 0.29, sem_p 0.92, syn)
- tom had no the day long  (syn_p 0.00, sem_p 0.00, non)
- bear do the walked to go good fun  (syn_p 0.00, sem_p 0.00, non)
- yes we go  (syn_p 0.00, sem_p 0.04, non)
- give have be away  (syn_p 0.00, sem_p 0.00, non)

**temp / W'=7 t=4**
- tim sam help were were at cat  (syn_p 0.00, sem_p 0.00, non)
- lily what the good  (syn_p 0.00, sem_p 0.00, non)
- they feel can for laughed  (syn_p 0.00, sem_p 0.00, non)
- tim could that what dog  (syn_p 0.00, sem_p 0.00, non)
- tom like liked loved them you  (syn_p 0.00, sem_p 0.00, non)

**temp / W'=7 t=8**
- look is to toy  (syn_p 0.00, sem_p 0.00, non)
- he became his is take all the long  (syn_p 0.00, sem_p 0.00, non)
- and mom no fast  (syn_p 0.00, sem_p 0.00, non)
- no be together  (syn_p 0.00, sem_p 0.00, non)
- said car anna the much  (syn_p 0.00, sem_p 0.00, non)

## Track 2 probe: window_names "all" from the word tree (wtree b), read reach W' = 7, 3, 1

| group | arm | seed | distinct new | dist/draw | R-sem | R-syn | R-non | (non with sem_pass) | tri\|d | tri\|d in sem / syn / non | fair sem / syn / non | repeats (cut) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wtree-0 | W'=7 t=1 | 13 | 1595 | 0.997 | 0.010 | 0.026 | 0.964 | 0.001 | 0.999 | 0.875 / 1.000 / 1.000 | 0.009 / 0.026 / 0.961 | 5 (0) |
| wtree-0 | W'=3 t=1 | 13 | 1598 | 0.999 | 0.008 | 0.035 | 0.957 | 0.002 | 0.999 | 0.917 / 1.000 / 1.000 | 0.007 / 0.035 / 0.956 | 2 (0) |
| wtree-0 | W'=1 t=1 | 13 | 1599 | 0.999 | 0.005 | 0.018 | 0.977 | 0.001 | 1.000 | 1.000 / 1.000 / 1.000 | 0.005 / 0.018 / 0.977 | 1 (0) |
| wtree-4 | W'=7 t=1 | 13 | 1548 | 0.968 | 0.054 | 0.097 | 0.849 | 0.010 | 0.984 | 0.735 / 0.993 / 0.998 | 0.038 / 0.093 / 0.821 | 52 (0) |
| wtree-4 | W'=3 t=1 | 13 | 1563 | 0.977 | 0.052 | 0.086 | 0.861 | 0.010 | 0.988 | 0.793 / 0.993 / 0.999 | 0.041 / 0.084 / 0.841 | 37 (1) |
| wtree-4 | W'=1 t=1 | 13 | 1580 | 0.988 | 0.027 | 0.056 | 0.918 | 0.002 | 0.996 | 0.857 / 0.989 / 1.000 | 0.022 / 0.054 / 0.906 | 20 (0) |
| wtree-5 | W'=7 t=1 | 13 | 1525 | 0.953 | 0.110 | 0.117 | 0.773 | 0.009 | 0.970 | 0.744 / 0.989 / 1.000 | 0.078 / 0.110 / 0.737 | 75 (1) |
| wtree-5 | W'=3 t=1 | 13 | 1532 | 0.958 | 0.091 | 0.108 | 0.801 | 0.010 | 0.980 | 0.807 / 0.982 / 0.999 | 0.071 / 0.101 / 0.766 | 68 (3) |
| wtree-5 | W'=1 t=1 | 13 | 1564 | 0.978 | 0.038 | 0.058 | 0.904 | 0.004 | 0.992 | 0.800 / 1.000 / 1.000 | 0.030 / 0.056 / 0.884 | 36 (1) |
| wtree-6 | W'=7 t=1 | 13 | 1462 | 0.914 | 0.132 | 0.134 | 0.734 | 0.007 | 0.969 | 0.782 / 0.980 / 1.000 | 0.094 / 0.120 / 0.671 | 138 (2) |
| wtree-6 | W'=3 t=1 | 13 | 1491 | 0.932 | 0.137 | 0.123 | 0.740 | 0.005 | 0.954 | 0.716 / 0.951 / 0.999 | 0.091 / 0.109 / 0.689 | 109 (4) |
| wtree-6 | W'=1 t=1 | 13 | 1551 | 0.969 | 0.055 | 0.086 | 0.859 | 0.003 | 0.992 | 0.884 / 1.000 / 0.998 | 0.048 / 0.083 / 0.831 | 49 (1) |
| wtree-7 | W'=7 t=1 | 13 | 1386 | 0.866 | 0.229 | 0.134 | 0.636 | 0.004 | 0.932 | 0.742 / 0.952 / 0.997 | 0.147 / 0.111 / 0.549 | 214 (7) |
| wtree-7 | W'=3 t=1 | 13 | 1425 | 0.891 | 0.255 | 0.150 | 0.594 | 0.004 | 0.899 | 0.695 / 0.879 / 0.992 | 0.158 / 0.117 / 0.525 | 175 (12) |
| wtree-7 | W'=1 t=1 | 13 | 1523 | 0.952 | 0.096 | 0.120 | 0.784 | 0.007 | 0.974 | 0.774 / 0.978 / 0.997 | 0.071 / 0.112 / 0.744 | 77 (5) |
| wtree-leaf | W'=7 t=1 | 13 | 1468 | 0.917 | 0.193 | 0.140 | 0.668 | 0.009 | 0.934 | 0.696 / 0.966 / 0.996 | 0.123 / 0.124 / 0.610 | 132 (1) |
| wtree-leaf | W'=3 t=1 | 13 | 1432 | 0.895 | 0.290 | 0.155 | 0.555 | 0.005 | 0.779 | 0.376 / 0.793 / 0.986 | 0.098 / 0.110 / 0.490 | 168 (16) |
| wtree-leaf | W'=1 t=1 | 13 | 1544 | 0.965 | 0.157 | 0.159 | 0.684 | 0.003 | 0.955 | 0.786 / 0.951 / 0.994 | 0.119 / 0.146 / 0.656 | 56 (10) |


### Samples (5 per arm)

**wtree-0 / W'=7 t=1**
- then have want your ball  (syn_p 0.00, sem_p 0.00, non)
- tom had not had you  (syn_p 0.16, sem_p 0.00, syn)
- one was sad  (syn_p 0.83, sem_p 0.96, sem)
- she will the safe  (syn_p 0.00, sem_p 0.00, non)
- lily was to were anymore  (syn_p 0.00, sem_p 0.00, non)

**wtree-0 / W'=3 t=1**
- they they to their fast  (syn_p 0.00, sem_p 0.00, non)
- they and away were and sun  (syn_p 0.00, sem_p 0.00, non)
- you go with to what dog  (syn_p 0.00, sem_p 0.00, non)
- sue saw again  (syn_p 0.00, sem_p 0.00, non)
- is became very much play  (syn_p 0.00, sem_p 0.00, non)

**wtree-0 / W'=1 t=1**
- she want said sun  (syn_p 0.00, sem_p 0.00, non)
- but said were home  (syn_p 0.00, sem_p 0.00, non)
- let's played friend  (syn_p 0.00, sem_p 0.00, non)
- is the new mom ball  (syn_p 0.00, sem_p 0.00, non)
- yes play the are said sam day  (syn_p 0.00, sem_p 0.00, non)

**wtree-4 / W'=7 t=1**
- lily and max went scared  (syn_p 0.00, sem_p 0.01, non)
- he and the boy  (syn_p 0.00, sem_p 0.00, non)
- tim saw tom and a tree  (syn_p 0.99, sem_p 0.99, sem)
- his mom smiled at the box  (syn_p 1.00, sem_p 1.00, sem)
- lily saw not mom idea  (syn_p 0.00, sem_p 0.00, non)

**wtree-4 / W'=3 t=1**
- this bird play the home  (syn_p 0.00, sem_p 0.00, non)
- they had an of other  (syn_p 0.00, sem_p 0.00, non)
- they played had surprised  (syn_p 0.00, sem_p 0.00, non)
- tim did a back bird  (syn_p 0.01, sem_p 0.00, syn)
- but made sam lot of fun  (syn_p 0.00, sem_p 0.00, non)

**wtree-4 / W'=1 t=1**
- yes i mommy all friends  (syn_p 0.00, sem_p 0.00, non)
- it were came happened  (syn_p 0.00, sem_p 0.00, non)
- they wanted a go happy together  (syn_p 0.00, sem_p 0.00, non)
- they liked idea  (syn_p 0.00, sem_p 0.00, non)
- tim were so and sky said sad  (syn_p 0.00, sem_p 0.00, non)

**wtree-5 / W'=7 t=1**
- can cat was yes am sorry  (syn_p 0.00, sem_p 0.00, non)
- he was surprised  (syn_p 1.00, sem_p 0.93, sem)
- tim was with the new toys all day  (syn_p 1.00, sem_p 1.00, sem)
- one go was see  (syn_p 0.00, sem_p 0.00, non)
- tim all good friends  (syn_p 0.00, sem_p 0.00, non)

**wtree-5 / W'=3 t=1**
- tim wanted outside and laughed  (syn_p 0.19, sem_p 1.00, sem)
- tom saw it bear  (syn_p 0.00, sem_p 0.00, non)
- she had so that play him  (syn_p 0.00, sem_p 0.00, non)
- is mom happy too  (syn_p 0.08, sem_p 0.20, syn)
- but had the toys to much  (syn_p 0.00, sem_p 0.00, non)

**wtree-5 / W'=1 t=1**
- sue was sam big friends  (syn_p 0.00, sem_p 0.00, non)
- the was so pretty  (syn_p 0.00, sem_p 0.01, non)
- the girl fast  (syn_p 0.00, sem_p 0.00, non)
- as day all a big outside  (syn_p 0.00, sem_p 0.00, non)
- they was we i big box  (syn_p 0.00, sem_p 0.00, non)

**wtree-6 / W'=7 t=1**
- do don't go with there  (syn_p 0.00, sem_p 0.00, non)
- they all happy it proud  (syn_p 0.00, sem_p 0.00, non)
- it girl was not little girl amy  (syn_p 0.00, sem_p 0.00, non)
- sue and very happy  (syn_p 0.00, sem_p 0.00, non)
- tim wanted to have it  (syn_p 1.00, sem_p 0.99, sem)

**wtree-6 / W'=3 t=1**
- the have lots of fun  (syn_p 0.00, sem_p 0.00, non)
- at the end house  (syn_p 0.00, sem_p 0.00, non)
- at the park they saw a big friend  (syn_p 0.99, sem_p 0.45, syn)
- he liked to play with it ball together  (syn_p 0.00, sem_p 0.01, non)
- they want their and and yes lily  (syn_p 0.00, sem_p 0.00, non)

**wtree-6 / W'=1 t=1**
- they were and fast with the have fun  (syn_p 0.00, sem_p 0.00, non)
- they sue said  (syn_p 0.00, sem_p 0.00, non)
- tim put get good  (syn_p 0.00, sem_p 0.00, non)
- thank it had to girl  (syn_p 0.00, sem_p 0.00, non)
- no are best cake away  (syn_p 0.00, sem_p 0.00, non)

**wtree-7 / W'=7 t=1**
- it was the bird became good friends  (syn_p 0.00, sem_p 0.00, non)
- tim is not happy  (syn_p 1.00, sem_p 1.00, sem)
- max was happy  (syn_p 1.00, sem_p 1.00, sem)
- but saw her mom for help  (syn_p 0.00, sem_p 0.00, non)
- his mom said yes too  (syn_p 0.94, sem_p 0.99, sem)

**wtree-7 / W'=3 t=1**
- she was they loved sorry  (syn_p 0.00, sem_p 0.00, non)
- the dog felt to all day  (syn_p 0.00, sem_p 0.00, non)
- they we will play home it  (syn_p 0.00, sem_p 0.00, non)
- but are good friends  (syn_p 0.00, sem_p 0.00, non)
- the they went happy  (syn_p 0.00, sem_p 0.01, non)

**wtree-7 / W'=1 t=1**
- tom had fun  (syn_p 1.00, sem_p 1.00, sem)
- i don't be sorry  (syn_p 0.00, sem_p 0.02, non)
- they saw a pretty  (syn_p 0.00, sem_p 0.00, non)
- but played very happy  (syn_p 0.00, sem_p 0.00, non)
- tom was yes lily  (syn_p 0.00, sem_p 0.00, non)

**wtree-leaf / W'=7 t=1**
- he was the happy to be home  (syn_p 0.00, sem_p 0.35, non)
- tim we have to go home soon  (syn_p 0.00, sem_p 0.08, non)
- she said what what happened  (syn_p 0.02, sem_p 0.04, syn)
- they fish all day  (syn_p 0.01, sem_p 0.48, syn)
- they saw a it box  (syn_p 0.00, sem_p 0.00, non)

**wtree-leaf / W'=3 t=1**
- tim said thank what happened  (syn_p 0.00, sem_p 0.00, non)
- they were not share  (syn_p 0.00, sem_p 0.00, non)
- she said who you lily me  (syn_p 0.00, sem_p 0.00, non)
- a big laughed came down fun  (syn_p 0.00, sem_p 0.00, non)
- tom and tim were so happy and excited  (syn_p 1.00, sem_p 1.00, sem)

**wtree-leaf / W'=1 t=1**
- it there was very happy  (syn_p 0.00, sem_p 0.04, non)
- he the dog became good bird  (syn_p 0.00, sem_p 0.00, non)
- he asked her better soon  (syn_p 0.00, sem_p 0.01, non)
- what it too  (syn_p 0.00, sem_p 0.00, non)
- the mommy said yes it is  (syn_p 0.29, sem_p 0.96, sem)

## r211 dumps re-judged with the ensemble (learned-W curve and baselines)
At most 100 distinct new sentences per (family, temp, seed) were judged (a fixed random sample; `R214_CAP211`); the columns dist/draw and
distinct new come from the full sets; tri|d, R-sem, R-syn and R-non over the judged sample, mean over the seeds present. TRELLIS wN = plain
TRELLIS at learned window N, TRELLIS edge wN = edge-exact; ngram k = maximum-likelihood k-gram (tri|d 0 by construction).

| family | temp | seeds | distinct new/seed | dist/draw | judged/seed | tri\|d | R-sem | R-syn | R-non |
|---|---|---|---|---|---|---|---|---|---|
| GPT | 0.15 | 4 | 38 | 0.025 | 38 | 0.654 | 0.780 | 0.133 | 0.087 |
| GPT | 0.2 | 4 | 86 | 0.056 | 86 | 0.630 | 0.771 | 0.143 | 0.086 |
| GPT | 0.3 | 4 | 208 | 0.137 | 100 | 0.643 | 0.730 | 0.155 | 0.115 |
| GPT | 0.4 | 4 | 422 | 0.277 | 100 | 0.688 | 0.660 | 0.190 | 0.150 |
| GPT | 0.5 | 4 | 667 | 0.439 | 100 | 0.703 | 0.625 | 0.200 | 0.175 |
| GPT | 0.6 | 4 | 908 | 0.598 | 100 | 0.735 | 0.573 | 0.205 | 0.223 |
| GPT | 0.7 | 4 | 1116 | 0.735 | 100 | 0.810 | 0.493 | 0.217 | 0.290 |
| GPT | 0.8 | 4 | 1283 | 0.845 | 100 | 0.850 | 0.440 | 0.215 | 0.345 |
| GPT | 0.9 | 4 | 1384 | 0.911 | 100 | 0.890 | 0.357 | 0.242 | 0.400 |
| GPT | 1.0 | 4 | 1449 | 0.953 | 100 | 0.917 | 0.282 | 0.203 | 0.515 |
| GPT | 1.2 | 4 | 1502 | 0.988 | 100 | 0.940 | 0.155 | 0.155 | 0.690 |
| LSTM | 0.15 | 4 | 33 | 0.022 | 33 | 0.331 | 0.837 | 0.108 | 0.055 |
| LSTM | 0.2 | 4 | 71 | 0.047 | 71 | 0.419 | 0.849 | 0.113 | 0.039 |
| LSTM | 0.3 | 4 | 192 | 0.127 | 100 | 0.475 | 0.865 | 0.083 | 0.052 |
| LSTM | 0.4 | 4 | 411 | 0.271 | 100 | 0.500 | 0.825 | 0.098 | 0.077 |
| LSTM | 0.5 | 4 | 684 | 0.450 | 100 | 0.603 | 0.745 | 0.135 | 0.120 |
| LSTM | 0.6 | 4 | 940 | 0.619 | 100 | 0.652 | 0.730 | 0.130 | 0.140 |
| LSTM | 0.7 | 4 | 1144 | 0.752 | 100 | 0.730 | 0.622 | 0.168 | 0.210 |
| LSTM | 0.8 | 4 | 1316 | 0.866 | 100 | 0.785 | 0.490 | 0.198 | 0.312 |
| LSTM | 0.9 | 4 | 1413 | 0.929 | 100 | 0.855 | 0.432 | 0.208 | 0.360 |
| LSTM | 1.0 | 4 | 1448 | 0.953 | 100 | 0.875 | 0.345 | 0.190 | 0.465 |
| LSTM | 1.2 | 4 | 1502 | 0.988 | 100 | 0.948 | 0.195 | 0.158 | 0.647 |
| TRELLIS edge w5 | 0.15 | 3 | 2 | 0.001 | 2 | 0.000 | 1.000 | 0.000 | 0.000 |
| TRELLIS edge w5 | 0.2 | 4 | 4 | 0.002 | 4 | 0.250 | 0.850 | 0.000 | 0.150 |
| TRELLIS edge w5 | 0.3 | 4 | 43 | 0.028 | 43 | 0.404 | 0.763 | 0.065 | 0.173 |
| TRELLIS edge w5 | 0.5 | 4 | 362 | 0.239 | 100 | 0.485 | 0.708 | 0.113 | 0.180 |
| TRELLIS edge w5 | 0.7 | 4 | 708 | 0.466 | 100 | 0.655 | 0.603 | 0.193 | 0.205 |
| TRELLIS edge w5 | 1.0 | 4 | 992 | 0.653 | 100 | 0.680 | 0.590 | 0.185 | 0.225 |
| TRELLIS edge w5 | 1.3 | 4 | 1131 | 0.744 | 100 | 0.777 | 0.532 | 0.208 | 0.260 |
| TRELLIS edge w5 | 1.6 | 4 | 1194 | 0.786 | 100 | 0.797 | 0.483 | 0.205 | 0.312 |
| TRELLIS edge w5 | 2.0 | 4 | 1244 | 0.819 | 100 | 0.850 | 0.535 | 0.188 | 0.278 |
| TRELLIS edge w6 | 1.3 | 4 | 1118 | 0.736 | 100 | 0.732 | 0.513 | 0.210 | 0.277 |
| TRELLIS edge w6 | 1.6 | 4 | 1180 | 0.776 | 100 | 0.775 | 0.502 | 0.230 | 0.268 |
| TRELLIS edge w6 | 2.0 | 4 | 1236 | 0.814 | 100 | 0.815 | 0.492 | 0.208 | 0.300 |
| TRELLIS edge w6 | 2.4 | 4 | 1268 | 0.835 | 100 | 0.865 | 0.415 | 0.205 | 0.380 |
| TRELLIS edge w6 | 2.8 | 4 | 1294 | 0.852 | 100 | 0.858 | 0.360 | 0.235 | 0.405 |
| TRELLIS edge w6 | 3.5 | 4 | 1328 | 0.874 | 100 | 0.902 | 0.335 | 0.247 | 0.417 |
| TRELLIS edge w7 | 0.2 | 3 | 3 | 0.002 | 3 | 0.567 | 0.350 | 0.083 | 0.567 |
| TRELLIS edge w7 | 0.3 | 4 | 36 | 0.024 | 36 | 0.528 | 0.721 | 0.075 | 0.204 |
| TRELLIS edge w7 | 0.5 | 4 | 342 | 0.225 | 100 | 0.600 | 0.670 | 0.138 | 0.193 |
| TRELLIS edge w7 | 0.7 | 4 | 719 | 0.473 | 100 | 0.677 | 0.647 | 0.145 | 0.208 |
| TRELLIS edge w7 | 1.0 | 4 | 1007 | 0.663 | 100 | 0.748 | 0.610 | 0.180 | 0.210 |
| TRELLIS edge w7 | 1.3 | 4 | 1125 | 0.740 | 100 | 0.738 | 0.562 | 0.200 | 0.237 |
| TRELLIS edge w7 | 1.6 | 4 | 1184 | 0.779 | 100 | 0.792 | 0.537 | 0.193 | 0.270 |
| TRELLIS edge w7 | 2.0 | 4 | 1238 | 0.814 | 100 | 0.883 | 0.397 | 0.228 | 0.375 |
| TRELLIS edge w7 | 2.4 | 4 | 1272 | 0.836 | 100 | 0.840 | 0.388 | 0.245 | 0.367 |
| TRELLIS edge w7 | 2.8 | 4 | 1299 | 0.855 | 100 | 0.873 | 0.348 | 0.225 | 0.427 |
| TRELLIS edge w7 | 3.5 | 4 | 1340 | 0.882 | 100 | 0.905 | 0.315 | 0.240 | 0.445 |
| TRELLIS edge w8 | 2.4 | 4 | 1261 | 0.830 | 100 | 0.853 | 0.422 | 0.182 | 0.395 |
| TRELLIS edge w8 | 2.8 | 2 | 1312 | 0.854 | 100 | 0.865 | 0.405 | 0.175 | 0.420 |
| TRELLIS edge w8 | 3.5 | 1 | 1384 | 0.865 | 100 | 0.970 | 0.310 | 0.260 | 0.430 |
| TRELLIS w2 | 1.0 | 4 | 1296 | 0.853 | 100 | 0.348 | 0.583 | 0.198 | 0.220 |
| TRELLIS w3 | 0.5 | 4 | 496 | 0.327 | 100 | 0.130 | 0.812 | 0.133 | 0.055 |
| TRELLIS w3 | 0.7 | 4 | 817 | 0.538 | 100 | 0.230 | 0.770 | 0.138 | 0.092 |
| TRELLIS w3 | 1.0 | 4 | 1094 | 0.720 | 100 | 0.443 | 0.647 | 0.202 | 0.150 |
| TRELLIS w4 | 0.5 | 4 | 380 | 0.251 | 100 | 0.430 | 0.685 | 0.143 | 0.172 |
| TRELLIS w4 | 0.7 | 4 | 720 | 0.473 | 100 | 0.480 | 0.645 | 0.172 | 0.182 |
| TRELLIS w4 | 1.0 | 4 | 1017 | 0.669 | 100 | 0.590 | 0.625 | 0.182 | 0.193 |
| TRELLIS w4 | 1.3 | 4 | 1152 | 0.758 | 100 | 0.670 | 0.568 | 0.203 | 0.230 |
| TRELLIS w5 | 0.5 | 4 | 444 | 0.292 | 100 | 0.677 | 0.528 | 0.125 | 0.347 |
| TRELLIS w5 | 0.7 | 4 | 808 | 0.532 | 100 | 0.690 | 0.523 | 0.155 | 0.323 |
| TRELLIS w5 | 1.0 | 4 | 1059 | 0.696 | 100 | 0.718 | 0.473 | 0.153 | 0.375 |
| TRELLIS w5 | 1.3 | 4 | 1176 | 0.774 | 100 | 0.845 | 0.463 | 0.152 | 0.385 |
| TRELLIS w5 | 1.6 | 4 | 1232 | 0.811 | 100 | 0.833 | 0.420 | 0.177 | 0.402 |
| TRELLIS w5 | 2.0 | 4 | 1281 | 0.843 | 100 | 0.870 | 0.378 | 0.200 | 0.422 |
| TRELLIS w6 | 1.3 | 4 | 1156 | 0.761 | 100 | 0.830 | 0.375 | 0.143 | 0.482 |
| TRELLIS w6 | 1.6 | 4 | 1216 | 0.800 | 100 | 0.845 | 0.333 | 0.145 | 0.522 |
| TRELLIS w6 | 2.0 | 4 | 1258 | 0.828 | 100 | 0.865 | 0.307 | 0.200 | 0.492 |
| TRELLIS w7 | 1.3 | 4 | 1183 | 0.779 | 100 | 0.805 | 0.417 | 0.133 | 0.450 |
| TRELLIS w7 | 1.6 | 4 | 1220 | 0.802 | 100 | 0.860 | 0.335 | 0.177 | 0.487 |
| TRELLIS w7 | 2.0 | 4 | 1274 | 0.838 | 100 | 0.867 | 0.295 | 0.218 | 0.487 |
| ngram 3 | 0.7 | 4 | 992 | 0.653 | 100 | 0.000 | 0.652 | 0.200 | 0.148 |
| ngram 3 | 1.0 | 4 | 1163 | 0.765 | 100 | 0.000 | 0.637 | 0.235 | 0.128 |
| ngram 4 | 0.7 | 4 | 610 | 0.402 | 100 | 0.000 | 0.833 | 0.117 | 0.050 |
| ngram 4 | 1.0 | 4 | 703 | 0.463 | 100 | 0.000 | 0.820 | 0.138 | 0.043 |
| ngram 5 | 0.7 | 4 | 190 | 0.125 | 100 | 0.000 | 0.875 | 0.105 | 0.020 |
| ngram 5 | 1.0 | 4 | 198 | 0.130 | 100 | 0.000 | 0.855 | 0.120 | 0.025 |

## Go rule and verdict
Go rule: each regime has a setting with purity >= .6, R-syn peaks at an interior position along the level dial, and the temperature path
does not reproduce R-syn.
- **R-non** reaches .6 and above everywhere on the coarse side (wtree b=0: .964; temperature 4 and 8: .94 and .96).
- **R-sem** >= .6 occurs only in the r211 families at low-to-mid temperature (edge w7 at tau 1: .610 with tri|d .748; edge w5: .59), not in any
  r214 arm: the best r214 arm is the token read at W'=2 (.334, tri|d among its R-sem sentences only .26) and the word tree at 7 bits (.229).
  The r214 arms use gate off and tau 1 (the r211 arms have the pair gate, which this design switches off as a confound).
- **R-syn** never reaches .6; the largest R-syn share in any arm, any family, any temperature is about .26 (r211 edge w5 tau 1.3: .21;
  TRELLIS edge w7 tau 3.5: .24; GPT tau 0.9: .24; r214 arms: at most .159, token W'=1 and 7-bit tree W'=1 .120). No setting is "syntax without
  meaning": when the output stops being meaningful it stops being grammatical too.
- **No interior R-syn peak**: along wtree b (root, 4, 5, 6, 7, leaf) at W'=7 R-syn goes .026, .097, .117, .134, .134, .140 (monotone), R-sem
  .010, .054, .110, .132, .229, .193, R-non .964 down to .668. The word tree behaves as a dial between R-non and R-sem/R-syn in a fixed
  ratio (R-sem : R-syn about 1 : 1 to 2 : 1), with R-non shrinking; it is not a syntax-only band.
- **Reach**: on the token learner R-sem peaks at W'=2 (.334), R-syn stays at .14-.17 for every W', R-non is lowest at W'=2 (.49). Knob B gives a
  partial R-sem / R-non axis, as the design predicted, not R-syn. The word-tree names sit on the same curve (at 7 bits W'=3: .255 / .150 / .594).
- **Temperature** (token learner, W'=7): R-sem .193 -> .056 -> .016 -> .007 and R-syn .140 -> .097 -> .041 -> .030 for tau 1, 2, 4, 8; the path
  goes straight to R-non and does not reproduce R-syn (criterion met trivially, because there is nothing to reproduce).
- The word-salad cell (R-non with sem_pass) is below .011 in every arm.

**Verdict: NO-GO.** R-syn is not available as a regime on this corpus with this judge: not through read level (Phase 0 level map), learned names
(Phase 0'), a Cobweb word-category tree (.775 / .795 purity at 7 bits, and its probe), read reach, or temperature, and the existing
families (TRELLIS, LSTM, GPT, n-grams) also give at most about .26. Of the three regimes, R-non and R-sem are produced by the dials (R-sem
is meaningful output only at the pair-gate recipe of r211, with novelty .6-.75 tri|d); the middle regime is a transition, with R-syn between .03
and .26 whatever the setting. Phase 2 is not warranted. Interim (track 1) numbers were reported to the coordinator before the word-tree probe finished.
