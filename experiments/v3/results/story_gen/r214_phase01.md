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

# Addendum 4: output-side abstraction (class emission, `emit_bits` / driver key `wemit`)

Diagnosis from the coordinator: the earlier dials abstracted only how the context is read; the filler came from stored token occurrences, so lexical
identity survived and syntax never decoupled from lexical choice. New opt-in `Trellis3R214.emit_bits` (word tree `WordTree` at level b, built
from the training stream; `emit_counts` = training-stream word counts): every step's filler masses, for the go-on read and the end read alike, become
P(w | ctx) = P(C_b(w) | ctx) * P(w | C_b(w)), with P(C | ctx) = the read's mass summed over the words of the class and P(w | C) = the context-free
share of the word's training count within its class (over the class's words the read offers). Leaf (one class per word, or `emit_bits` None / inf)
= the existing model exactly; root = the unigram share of the read's total mass. Tests (test 9: off, inf and singleton classes equal the read
exactly; root gives mass proportional to the counts and conserves the total; a middle level conserves every class's mass and splits it by
count) pass; the default path is byte-identical (tests 1-2) and the full `tests/trellis2` suite passes (137). The pair gate is applied to the
emitted word (t3_story's `_ok(l1, w)` filters the post-emission fillers, i.e. rejection inside the class). Drivers `r214_p1d.sh` (seed, b list),
`r214_p1e.sh`; dumps `r214p01/e13_*`, `e17_*`; judge outputs `judge3` (seed 13), `judge4` (follow-ups and seed 17). Seed 13 and 17, n=1000, edge exact, tau 1, W' 7 unless noted.
Input (i) = token window (`e-i-*`); input (ii) = window named by the same word tree at the same b (`e-ii-*`, a learner per b). `emit=leaf` with the gate is the paper-recipe
token model with the gate on, gate on being the primary run as asked.

## Seed 13 (all arms)

| group | arm | seed | distinct new | dist/draw | R-sem | R-syn | R-non | (non with sem_pass) | tri\|d | tri\|d in sem / syn / non | fair sem / syn / non | repeats (cut) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e-i-off | W'=7 emit=0 t=1 | 13 | 1600 | 1.000 | 0.001 | 0.012 | 0.988 | 0.001 | 1.000 | 1.000 / 1.000 / 1.000 | 0.001 / 0.012 / 0.988 | 0 (0) |
| e-i-off | W'=7 emit=4 t=1 | 13 | 1585 | 0.991 | 0.021 | 0.032 | 0.947 | 0.001 | 0.999 | 0.939 / 1.000 / 1.000 | 0.019 / 0.032 / 0.938 | 15 (0) |
| e-i-off | W'=7 emit=5 t=1 | 13 | 1574 | 0.984 | 0.029 | 0.049 | 0.922 | 0.001 | 0.994 | 0.778 / 1.000 / 1.000 | 0.022 / 0.048 / 0.907 | 26 (0) |
| e-i-off | W'=7 emit=6 t=1 | 13 | 1574 | 0.984 | 0.041 | 0.062 | 0.897 | 0.005 | 0.992 | 0.862 / 0.969 / 1.000 | 0.035 / 0.059 / 0.882 | 26 (0) |
| e-i-off | W'=7 emit=6.5 t=1 | 13 | 1533 | 0.958 | 0.067 | 0.100 | 0.833 | 0.001 | 0.984 | 0.824 / 0.968 / 0.999 | 0.052 / 0.093 / 0.797 | 67 (1) |
| e-i-off | W'=7 emit=7 t=1 | 13 | 1499 | 0.937 | 0.110 | 0.129 | 0.761 | 0.001 | 0.976 | 0.818 / 0.979 / 0.998 | 0.084 / 0.118 / 0.712 | 101 (3) |
| e-i-on | W'=7 emit=0 t=1 gate | 13 | 1543 | 0.964 | 0.069 | 0.116 | 0.815 | 0.004 | 0.990 | 0.916 / 0.972 / 0.998 | 0.061 / 0.109 / 0.784 | 57 (9) |
| e-i-on | W'=7 emit=4 t=1 gate | 13 | 1436 | 0.897 | 0.212 | 0.186 | 0.602 | 0.012 | 0.951 | 0.826 / 0.955 / 0.993 | 0.157 / 0.159 / 0.537 | 164 (8) |
| e-i-on | W'=7 emit=5 t=1 gate | 13 | 1391 | 0.869 | 0.268 | 0.193 | 0.538 | 0.014 | 0.938 | 0.818 / 0.944 / 0.996 | 0.191 / 0.159 / 0.466 | 209 (16) |
| e-i-on | W'=7 emit=6 t=1 gate | 13 | 1352 | 0.845 | 0.286 | 0.209 | 0.505 | 0.022 | 0.926 | 0.811 / 0.929 / 0.990 | 0.196 / 0.164 / 0.422 | 248 (10) |
| e-i-on | W'=7 emit=6.5 t=1 gate | 13 | 1262 | 0.789 | 0.409 | 0.213 | 0.378 | 0.010 | 0.895 | 0.793 / 0.944 / 0.977 | 0.256 / 0.159 / 0.291 | 338 (9) |
| e-i-on | W'=7 emit=7 t=1 gate | 13 | 1172 | 0.733 | 0.432 | 0.217 | 0.352 | 0.010 | 0.871 | 0.755 / 0.925 / 0.981 | 0.239 / 0.147 / 0.253 | 428 (16) |
| e-i-on | W'=7 emit=leaf t=1 gate | 13 | 1032 | 0.645 | 0.558 | 0.206 | 0.235 | 0.012 | 0.766 | 0.639 / 0.887 / 0.963 | 0.230 / 0.118 / 0.146 | 566 (16) |
| e-ii-0 | W'=7 emit=0 t=1 gate | 13 | 1596 | 0.998 | 0.005 | 0.012 | 0.983 | 0.001 | 0.999 | 0.875 / 0.947 / 1.000 | 0.004 / 0.011 / 0.981 | 4 (1) |
| e-ii-0 | W'=7 emit=0 t=1 | 13 | 1600 | 1.000 | 0.002 | 0.012 | 0.986 | 0.000 | 1.000 | 1.000 / 1.000 / 1.000 | 0.002 / 0.012 / 0.986 | 0 (0) |
| e-ii-4 | W'=7 emit=4 t=1 gate | 13 | 1451 | 0.907 | 0.179 | 0.168 | 0.653 | 0.012 | 0.955 | 0.819 / 0.947 / 0.995 | 0.133 / 0.144 / 0.589 | 149 (21) |
| e-ii-4 | W'=7 emit=4 t=1 | 13 | 1592 | 0.995 | 0.013 | 0.025 | 0.962 | 0.003 | 0.999 | 0.952 / 1.000 / 1.000 | 0.013 / 0.025 / 0.957 | 8 (0) |
| e-ii-5 | W'=7 emit=5 t=1 gate | 13 | 1434 | 0.896 | 0.231 | 0.179 | 0.591 | 0.008 | 0.944 | 0.807 / 0.949 / 0.995 | 0.167 / 0.152 / 0.527 | 166 (10) |
| e-ii-5 | W'=7 emit=5 t=1 | 13 | 1586 | 0.991 | 0.017 | 0.045 | 0.938 | 0.004 | 0.997 | 0.889 / 0.972 / 1.000 | 0.015 / 0.044 / 0.929 | 14 (0) |
| e-ii-6 | W'=7 emit=6 t=1 gate | 13 | 1364 | 0.853 | 0.296 | 0.177 | 0.526 | 0.015 | 0.927 | 0.800 / 0.926 / 0.999 | 0.202 / 0.140 / 0.448 | 236 (13) |
| e-ii-6 | W'=7 emit=6 t=1 | 13 | 1571 | 0.982 | 0.028 | 0.057 | 0.915 | 0.003 | 0.994 | 0.818 / 0.978 / 1.000 | 0.022 / 0.055 / 0.898 | 29 (0) |
| e-ii-6.5 | W'=7 emit=6.5 t=1 gate | 13 | 1316 | 0.823 | 0.371 | 0.224 | 0.405 | 0.007 | 0.891 | 0.758 / 0.939 / 0.987 | 0.231 / 0.173 / 0.329 | 284 (20) |
| e-ii-6.5 | W'=7 emit=6.5 t=1 | 13 | 1552 | 0.970 | 0.057 | 0.079 | 0.863 | 0.003 | 0.985 | 0.764 / 0.976 / 1.000 | 0.043 / 0.075 / 0.838 | 48 (0) |
| e-ii-7 | W'=7 emit=7 t=1 gate | 13 | 1250 | 0.781 | 0.407 | 0.189 | 0.404 | 0.004 | 0.871 | 0.723 / 0.928 / 0.994 | 0.230 / 0.137 / 0.314 | 350 (17) |
| e-ii-7 | W'=7 emit=7 t=1 | 13 | 1514 | 0.946 | 0.096 | 0.104 | 0.800 | 0.006 | 0.976 | 0.788 / 0.981 / 0.998 | 0.072 / 0.096 / 0.756 | 86 (4) |


### Samples, seed 13 arms (5 per arm)

**e-i-off / W'=7 emit=0 t=1**
- so toy she he that  (syn_p 0.00, sem_p 0.00, non)
- we scared much he am  (syn_p 0.00, sem_p 0.00, non)
- lily so sad some friends  (syn_p 0.00, sem_p 0.00, non)
- now i friend happy know you idea with  (syn_p 0.00, sem_p 0.00, non)
- it every play many tom  (syn_p 0.00, sem_p 0.00, non)

**e-i-off / W'=7 emit=4 t=1**
- they was not got of to smiled  (syn_p 0.00, sem_p 0.00, non)
- they all and very friend became mom around  (syn_p 0.00, sem_p 0.00, non)
- you yes it's friends mom  (syn_p 0.00, sem_p 0.00, non)
- sue found the to fun other  (syn_p 0.00, sem_p 0.00, non)
- it felt excited happy wanted happy  (syn_p 0.00, sem_p 0.00, non)

**e-i-off / W'=7 emit=5 t=1**
- she so look with careful  (syn_p 0.00, sem_p 0.00, non)
- bob were together thanked toy  (syn_p 0.00, sem_p 0.00, non)
- just big car laughed day  (syn_p 0.00, sem_p 0.00, non)
- is mia were big  (syn_p 0.00, sem_p 0.00, non)
- you it was inside  (syn_p 0.00, sem_p 0.00, non)

**e-i-off / W'=7 emit=6 t=1**
- it was red said excited  (syn_p 0.00, sem_p 0.00, non)
- come see good him  (syn_p 0.00, sem_p 0.00, non)
- tim played the the and played can fun  (syn_p 0.00, sem_p 0.00, non)
- he were can was was good go  (syn_p 0.00, sem_p 0.00, non)
- lily and ben played together all it park  (syn_p 0.00, sem_p 0.01, non)

**e-i-off / W'=7 emit=6.5 t=1**
- tim and not fun  (syn_p 0.00, sem_p 0.00, non)
- they feel happy  (syn_p 0.31, sem_p 0.98, sem)
- they play happy  (syn_p 0.00, sem_p 0.05, non)
- tim have tom in fun  (syn_p 0.00, sem_p 0.00, non)
- but then looked had happened  (syn_p 0.00, sem_p 0.00, non)

**e-i-off / W'=7 emit=7 t=1**
- it was a fun soon is her  (syn_p 0.00, sem_p 0.00, non)
- they were surprised toy the cat happy  (syn_p 0.00, sem_p 0.00, non)
- they said i'm were sorry  (syn_p 0.00, sem_p 0.06, non)
- we to want to go  (syn_p 0.00, sem_p 0.00, non)
- but with careful  (syn_p 0.00, sem_p 0.00, non)

**e-i-on / W'=7 emit=0 t=1 gate**
- you will not a car went fast  (syn_p 0.00, sem_p 0.00, non)
- can be happy with the dog went back  (syn_p 0.00, sem_p 0.00, non)
- tim said look i was the sky  (syn_p 0.16, sem_p 0.03, syn)
- lily look ben and tim  (syn_p 0.00, sem_p 0.00, non)
- they look ben are you  (syn_p 0.00, sem_p 0.00, non)

**e-i-on / W'=7 emit=4 t=1 gate**
- tom was sad  (syn_p 1.00, sem_p 1.00, sem)
- you mom to play  (syn_p 0.00, sem_p 0.00, non)
- she liked the toys were so much fun  (syn_p 0.00, sem_p 0.18, non)
- it said they had fun together  (syn_p 1.00, sem_p 0.97, sem)
- ben were very scared  (syn_p 0.00, sem_p 0.01, non)

**e-i-on / W'=7 emit=5 t=1 gate**
- the car was very fast  (syn_p 1.00, sem_p 1.00, sem)
- they and smiled and said yes it too  (syn_p 0.00, sem_p 0.00, non)
- then he saw a big tree  (syn_p 1.00, sem_p 1.00, sem)
- and play together  (syn_p 0.00, sem_p 0.00, non)
- tom and then the ball together  (syn_p 0.00, sem_p 0.00, non)

**e-i-on / W'=7 emit=6 t=1 gate**
- he and it together  (syn_p 0.00, sem_p 0.00, non)
- they like to play in it water  (syn_p 0.00, sem_p 0.00, non)
- let's find something fun to play  (syn_p 0.01, sem_p 0.99, sem)
- the cat and i am sorry  (syn_p 0.00, sem_p 0.01, non)
- tom were the cat was fun together  (syn_p 0.00, sem_p 0.00, non)

**e-i-on / W'=7 emit=6.5 t=1 gate**
- they do not find him  (syn_p 0.89, sem_p 0.23, syn)
- as they all became best friends  (syn_p 0.00, sem_p 0.22, non)
- at the park mia saw his friend sue  (syn_p 0.99, sem_p 0.96, sem)
- he wanted to do it  (syn_p 1.00, sem_p 1.00, sem)
- we did not get it  (syn_p 1.00, sem_p 0.98, sem)

**e-i-on / W'=7 emit=7 t=1 gate**
- they liked to play at the friends  (syn_p 0.01, sem_p 0.01, syn)
- she went to see what it  (syn_p 0.00, sem_p 0.01, non)
- lily and ben were playing in the sun  (syn_p 1.00, sem_p 1.00, sem)
- see you look  (syn_p 0.00, sem_p 0.01, non)
- come play on a good friends  (syn_p 0.00, sem_p 0.00, non)

**e-i-on / W'=7 emit=leaf t=1 gate**
- lily and ben let's go  (syn_p 0.00, sem_p 0.00, non)
- they played together all the park  (syn_p 0.00, sem_p 0.05, non)
- tim was big again  (syn_p 0.99, sem_p 0.29, syn)
- can i play with me  (syn_p 0.00, sem_p 0.01, non)
- tom and to play outside in the sun  (syn_p 0.00, sem_p 0.01, non)

**e-ii-0 / W'=7 emit=0 t=1 gate**
- tim played and girl and what the can  (syn_p 0.00, sem_p 0.00, non)
- go with had and lots of the with  (syn_p 0.00, sem_p 0.00, non)
- the was had to was  (syn_p 0.00, sem_p 0.00, non)
- and bad too little  (syn_p 0.00, sem_p 0.00, non)
- tim you to a tree  (syn_p 0.00, sem_p 0.00, non)

**e-ii-0 / W'=7 emit=0 t=1**
- but a happy her  (syn_p 0.00, sem_p 0.00, non)
- they he day happy max you so was  (syn_p 0.00, sem_p 0.00, non)
- lily play and knew in but happy  (syn_p 0.00, sem_p 0.00, non)
- you he yes fun  (syn_p 0.00, sem_p 0.00, non)
- spot she little  (syn_p 0.00, sem_p 0.00, non)

**e-ii-4 / W'=7 emit=4 t=1 gate**
- they played something shiny and pretty  (syn_p 0.99, sem_p 0.81, syn)
- they were the rock and so were sorry  (syn_p 0.00, sem_p 0.01, non)
- they said the rock and friends and friends  (syn_p 0.00, sem_p 0.00, non)
- she liked to go home was so small  (syn_p 0.00, sem_p 0.00, non)
- a bad man smiled excited  (syn_p 0.00, sem_p 0.15, non)

**e-ii-4 / W'=7 emit=4 t=1**
- they upon to time look and is back  (syn_p 0.00, sem_p 0.00, non)
- it bear play in you  (syn_p 0.00, sem_p 0.00, non)
- he so look do came the and animals  (syn_p 0.00, sem_p 0.00, non)
- can with what was be outside  (syn_p 0.00, sem_p 0.00, non)
- tom laughed it both don't go his tim  (syn_p 0.00, sem_p 0.00, non)

**e-ii-5 / W'=7 emit=5 t=1 gate**
- then a car was not good friends  (syn_p 0.04, sem_p 0.00, syn)
- who was scared and happy  (syn_p 0.00, sem_p 0.00, non)
- she said let's do not his ball  (syn_p 0.00, sem_p 0.00, non)
- she is your dog  (syn_p 0.96, sem_p 0.74, syn)
- one day he were they liked to do  (syn_p 0.00, sem_p 0.00, non)

**e-ii-5 / W'=7 emit=5 t=1**
- the dog were laughed was good  (syn_p 0.00, sem_p 0.00, non)
- i his help fun  (syn_p 0.00, sem_p 0.00, non)
- tim's do and fish away  (syn_p 0.00, sem_p 0.00, non)
- ben was and happy  (syn_p 0.00, sem_p 0.05, non)
- sue see to big ball  (syn_p 0.00, sem_p 0.00, non)

**e-ii-6 / W'=7 emit=6 t=1 gate**
- it too big and shiny  (syn_p 0.00, sem_p 0.00, non)
- spot was very scared and pretty  (syn_p 0.28, sem_p 0.62, syn)
- sue said lily and sad  (syn_p 0.00, sem_p 0.00, non)
- one day he made a little bird  (syn_p 1.00, sem_p 0.71, syn)
- she was her best friend  (syn_p 0.61, sem_p 0.39, syn)

**e-ii-6 / W'=7 emit=6 t=1**
- lily and so sad  (syn_p 0.00, sem_p 0.00, non)
- they then something took idea  (syn_p 0.00, sem_p 0.00, non)
- they told mom and was together  (syn_p 0.00, sem_p 0.01, non)
- she like it park  (syn_p 0.00, sem_p 0.00, non)
- tim was with i toys are fun  (syn_p 0.00, sem_p 0.00, non)

**e-ii-6.5 / W'=7 emit=6.5 t=1 gate**
- but wanted to play with it  (syn_p 0.00, sem_p 0.02, non)
- yes it a shiny  (syn_p 0.00, sem_p 0.00, non)
- his mom and the friends  (syn_p 0.00, sem_p 0.01, non)
- it a happy  (syn_p 0.00, sem_p 0.00, non)
- lily and sue loved the friends  (syn_p 0.78, sem_p 0.06, syn)

**e-ii-6.5 / W'=7 emit=6.5 t=1**
- they thought tom bird was home  (syn_p 0.27, sem_p 0.00, syn)
- everyone are you  (syn_p 0.00, sem_p 0.00, non)
- it and flew was fun  (syn_p 0.00, sem_p 0.00, non)
- can a the good ball careful  (syn_p 0.00, sem_p 0.00, non)
- that they have fun  (syn_p 0.00, sem_p 0.00, non)

**e-ii-7 / W'=7 emit=7 t=1 gate**
- can we play with you  (syn_p 0.52, sem_p 0.83, syn)
- i had to play it  (syn_p 0.99, sem_p 0.97, sem)
- tom and very surprised and excited  (syn_p 0.00, sem_p 0.00, non)
- the dog was surprised  (syn_p 1.00, sem_p 1.00, sem)
- they were they wanted to look ben look  (syn_p 0.00, sem_p 0.00, non)

**e-ii-7 / W'=7 emit=7 t=1**
- max was lily you at friends  (syn_p 0.00, sem_p 0.00, non)
- but were very happy  (syn_p 0.00, sem_p 0.01, non)
- tim found share sam play  (syn_p 0.00, sem_p 0.00, non)
- the dog went to make the bird  (syn_p 0.13, sem_p 0.00, syn)
- said played mia lots good there  (syn_p 0.00, sem_p 0.00, non)

## Follow-ups on seed 13 (tau and W' at the top b) and seed 17 replicates (b = 6, 6.5, 7: the three b values with the highest R-syn)

| group | arm | seed | distinct new | dist/draw | R-sem | R-syn | R-non | (non with sem_pass) | tri\|d | tri\|d in sem / syn / non | fair sem / syn / non | repeats (cut) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e17-i-off | W'=7 emit=6 t=1 | 17 | 1507 | 0.942 | 0.063 | 0.046 | 0.891 | 0.001 | 0.980 | 0.705 / 0.971 / 1.000 | 0.042 / 0.042 / 0.839 | 37 (2) |
| e17-i-off | W'=7 emit=6.5 t=1 | 17 | 1500 | 0.938 | 0.065 | 0.069 | 0.866 | 0.003 | 0.983 | 0.745 / 0.990 / 1.000 | 0.046 / 0.064 / 0.812 | 44 (10) |
| e17-i-off | W'=7 emit=7 t=1 | 17 | 1484 | 0.927 | 0.080 | 0.083 | 0.837 | 0.003 | 0.980 | 0.790 / 0.992 / 0.997 | 0.059 / 0.076 / 0.774 | 60 (4) |
| e17-i-on | W'=7 emit=6 t=1 gate | 17 | 1291 | 0.807 | 0.352 | 0.203 | 0.445 | 0.009 | 0.900 | 0.767 / 0.939 / 0.988 | 0.218 / 0.154 / 0.354 | 253 (36) |
| e17-i-on | W'=7 emit=6.5 t=1 gate | 17 | 1268 | 0.792 | 0.335 | 0.215 | 0.450 | 0.016 | 0.894 | 0.739 / 0.949 / 0.984 | 0.196 / 0.161 / 0.351 | 276 (27) |
| e17-i-on | W'=7 emit=7 t=1 gate | 17 | 1207 | 0.754 | 0.369 | 0.216 | 0.415 | 0.014 | 0.886 | 0.742 / 0.939 / 0.988 | 0.206 / 0.153 / 0.309 | 337 (27) |
| e17-i-on | W'=7 emit=leaf t=1 gate | 17 | 1057 | 0.661 | 0.577 | 0.171 | 0.252 | 0.011 | 0.737 | 0.597 / 0.851 / 0.981 | 0.228 / 0.096 / 0.163 | 487 (11) |
| e17-ii-6 | W'=7 emit=6 t=1 gate | 17 | 1277 | 0.798 | 0.306 | 0.178 | 0.516 | 0.013 | 0.907 | 0.752 / 0.925 / 0.992 | 0.184 / 0.131 / 0.409 | 267 (29) |
| e17-ii-6 | W'=7 emit=6 t=1 | 17 | 1509 | 0.943 | 0.052 | 0.041 | 0.907 | 0.001 | 0.991 | 0.846 / 1.000 / 0.999 | 0.041 / 0.039 / 0.855 | 35 (0) |
| e17-ii-6.5 | W'=7 emit=6.5 t=1 gate | 17 | 1259 | 0.787 | 0.355 | 0.178 | 0.467 | 0.011 | 0.892 | 0.720 / 0.969 / 0.993 | 0.201 / 0.136 / 0.365 | 285 (35) |
| e17-ii-6.5 | W'=7 emit=6.5 t=1 | 17 | 1487 | 0.929 | 0.057 | 0.054 | 0.889 | 0.000 | 0.989 | 0.800 / 1.000 / 1.000 | 0.043 / 0.050 / 0.826 | 57 (2) |
| e17-ii-7 | W'=7 emit=7 t=1 gate | 17 | 1202 | 0.751 | 0.369 | 0.197 | 0.433 | 0.009 | 0.883 | 0.734 / 0.924 / 0.990 | 0.204 / 0.137 / 0.323 | 342 (20) |
| e17-ii-7 | W'=7 emit=7 t=1 | 17 | 1452 | 0.907 | 0.083 | 0.090 | 0.828 | 0.003 | 0.977 | 0.742 / 0.985 / 1.000 | 0.056 / 0.080 / 0.751 | 92 (3) |
| follow | W'=7 emit=6.5 t=0.7 gate | 13 | 995 | 0.622 | 0.445 | 0.198 | 0.357 | 0.013 | 0.870 | 0.745 / 0.959 / 0.977 | 0.206 / 0.118 / 0.217 | 605 (22) |
| follow | W'=7 emit=6.5 t=2 gate | 13 | 1403 | 0.877 | 0.301 | 0.249 | 0.450 | 0.004 | 0.928 | 0.823 / 0.946 / 0.989 | 0.217 / 0.206 / 0.390 | 197 (14) |
| follow | W'=2 emit=6.5 t=1 gate | 13 | 1338 | 0.836 | 0.470 | 0.197 | 0.333 | 0.009 | 0.726 | 0.595 / 0.787 / 0.874 | 0.234 / 0.129 / 0.244 | 262 (52) |
| follow | W'=2 emit=7 t=1 gate | 13 | 1324 | 0.828 | 0.469 | 0.231 | 0.300 | 0.005 | 0.682 | 0.557 / 0.712 / 0.854 | 0.216 / 0.136 / 0.212 | 276 (75) |


### Samples (5 per arm)

**e17-i-off / W'=7 emit=6 t=1**
- they are not with little dog anymore  (syn_p 0.50, sem_p 0.02, syn)
- you am be be the dog  (syn_p 0.00, sem_p 0.00, non)
- she laughed look the said to do fun  (syn_p 0.00, sem_p 0.00, non)
- sue wanted to find his what cat  (syn_p 0.00, sem_p 0.00, non)
- mom playing was smiled  (syn_p 0.00, sem_p 0.03, non)

**e17-i-off / W'=7 emit=6.5 t=1**
- tim and sue all with had the bird  (syn_p 0.00, sem_p 0.00, non)
- they had can were very surprised  (syn_p 0.00, sem_p 0.00, non)
- the boy feel found sad  (syn_p 0.00, sem_p 0.03, non)
- it and sue mom smiled  (syn_p 0.00, sem_p 0.00, non)
- sue was so sad  (syn_p 1.00, sem_p 0.99, sem)

**e17-i-off / W'=7 emit=7 t=1**
- but can one unexpected happened  (syn_p 0.00, sem_p 0.00, non)
- lily a proud unexpected happened  (syn_p 0.00, sem_p 0.00, non)
- i saw will now  (syn_p 0.03, sem_p 0.09, syn)
- let's go soon  (syn_p 0.99, sem_p 0.59, syn)
- do you came to go with friend  (syn_p 0.00, sem_p 0.00, non)

**e17-i-on / W'=7 emit=6 t=1 gate**
- i will do you  (syn_p 0.11, sem_p 0.22, syn)
- sara and lily had no friends  (syn_p 1.00, sem_p 1.00, sem)
- tim and sue were very happy  (syn_p 1.00, sem_p 1.00, sem)
- tom and his friends were excited  (syn_p 1.00, sem_p 1.00, sem)
- but the friends were scared of the tree  (syn_p 1.00, sem_p 0.99, sem)

**e17-i-on / W'=7 emit=6.5 t=1 gate**
- mia wanted to go away  (syn_p 1.00, sem_p 1.00, sem)
- while they played in a big house  (syn_p 0.00, sem_p 0.77, non)
- tom and ben and said to help them  (syn_p 0.00, sem_p 0.00, non)
- from that day on they played with him  (syn_p 1.00, sem_p 0.93, sem)
- his toys now  (syn_p 0.00, sem_p 0.00, non)

**e17-i-on / W'=7 emit=7 t=1 gate**
- sue was happy to do  (syn_p 0.00, sem_p 0.02, non)
- the bird asked can we play with you  (syn_p 0.01, sem_p 0.59, syn)
- lily and sue wanted to do it  (syn_p 1.00, sem_p 1.00, sem)
- but she could not see what was inside  (syn_p 1.00, sem_p 0.65, syn)
- they and the girl became good friends  (syn_p 0.01, sem_p 0.52, syn)

**e17-i-on / W'=7 emit=leaf t=1 gate**
- tom and tom thanked her  (syn_p 1.00, sem_p 0.99, sem)
- tim smiled and said yes you lily  (syn_p 0.25, sem_p 0.04, syn)
- anna and lily were playing in the park  (syn_p 1.00, sem_p 1.00, sem)
- he loved to play outside  (syn_p 1.00, sem_p 1.00, sem)
- we be scared lily  (syn_p 0.00, sem_p 0.00, non)

**e17-ii-6 / W'=7 emit=6 t=1 gate**
- lily were not happy to find his ball  (syn_p 0.00, sem_p 0.03, non)
- then he looked at lily sam  (syn_p 0.21, sem_p 0.09, syn)
- they had so excited to your ball together  (syn_p 0.00, sem_p 0.00, non)
- ben was happy and said yes  (syn_p 1.00, sem_p 1.00, sem)
- ben were so happy to see their cat  (syn_p 0.00, sem_p 0.25, non)

**e17-ii-6 / W'=7 emit=6 t=1**
- he and scared and will anymore  (syn_p 0.00, sem_p 0.00, non)
- they then did food happened  (syn_p 0.00, sem_p 0.00, non)
- tim a so sad  (syn_p 0.00, sem_p 0.00, non)
- they very friends  (syn_p 0.00, sem_p 0.00, non)
- tim want with go are you better  (syn_p 0.00, sem_p 0.00, non)

**e17-ii-6.5 / W'=7 emit=6.5 t=1 gate**
- they and friends and scared  (syn_p 0.00, sem_p 0.00, non)
- she was very happy to do  (syn_p 0.00, sem_p 0.04, non)
- lily had an idea  (syn_p 1.00, sem_p 1.00, sem)
- tom wanted to play with it  (syn_p 1.00, sem_p 1.00, sem)
- she and proud of her little girl  (syn_p 0.00, sem_p 0.00, non)

**e17-ii-6.5 / W'=7 emit=6.5 t=1**
- but then something unexpected long  (syn_p 0.00, sem_p 0.00, non)
- he wanted mom a lily  (syn_p 0.00, sem_p 0.00, non)
- so they could the went have play  (syn_p 0.00, sem_p 0.00, non)
- the dog went to see i cake  (syn_p 0.01, sem_p 0.00, syn)
- they played liked friends  (syn_p 0.00, sem_p 0.67, non)

**e17-ii-7 / W'=7 emit=7 t=1 gate**
- but he saw the man was a girl  (syn_p 0.92, sem_p 0.75, syn)
- they played and sad  (syn_p 0.00, sem_p 0.00, non)
- lily and happy to help  (syn_p 0.00, sem_p 0.00, non)
- what was proud of fun  (syn_p 0.00, sem_p 0.00, non)
- do that was very happy and thanked sam  (syn_p 0.00, sem_p 0.00, non)

**e17-ii-7 / W'=7 emit=7 t=1**
- who all you  (syn_p 0.00, sem_p 0.00, non)
- he was and to with his ball me  (syn_p 0.00, sem_p 0.00, non)
- tim liked for play with together long  (syn_p 0.00, sem_p 0.00, non)
- soon he saw was girl my max  (syn_p 0.00, sem_p 0.00, non)
- they had and lot of fun  (syn_p 0.00, sem_p 0.06, non)

**follow / W'=7 emit=6.5 t=0.7 gate**
- the car and i was playing lily  (syn_p 0.00, sem_p 0.00, non)
- the little girl was happy  (syn_p 1.00, sem_p 1.00, sem)
- the cat and i don't do that ball  (syn_p 0.00, sem_p 0.00, non)
- they was very happy to help his mom  (syn_p 0.00, sem_p 0.86, non)
- this is big  (syn_p 0.99, sem_p 0.98, sem)

**follow / W'=7 emit=6.5 t=2 gate**
- once upon a high rock and a girl  (syn_p 0.00, sem_p 0.00, non)
- a big bird came down from the toys  (syn_p 1.00, sem_p 0.03, syn)
- tom wanted to go now  (syn_p 1.00, sem_p 1.00, sem)
- it a boy came to the park  (syn_p 0.00, sem_p 0.02, non)
- go in a lot too  (syn_p 0.00, sem_p 0.00, non)

**follow / W'=2 emit=6.5 t=1 gate**
- he and sad  (syn_p 0.00, sem_p 0.00, non)
- they and lily and sue were happy  (syn_p 0.09, sem_p 0.68, syn)
- so they found it under a new friend  (syn_p 1.00, sem_p 0.00, syn)
- she wanted to play  (syn_p 0.99, sem_p 1.00, sem)
- she asked a car  (syn_p 0.00, sem_p 0.01, non)

**follow / W'=2 emit=7 t=1 gate**
- they asked lily said scared  (syn_p 0.00, sem_p 0.00, non)
- it a kind bird to do it  (syn_p 0.00, sem_p 0.00, non)
- but then they saw a big red ball  (syn_p 1.00, sem_p 1.00, sem)
- the man smiled and smiled  (syn_p 1.00, sem_p 1.00, sem)
- it was not happy  (syn_p 1.00, sem_p 1.00, sem)

## Is class impurity the bottleneck? (seed 13, `r214_impurity.py`)
An emitted word is IMPURE when its majority POS (spaCy UPOS over the training stream) differs from the token-weighted majority POS of its class at b.
Per arm: share of syntax-fail and of syntax-pass sentences with at least one impure emission, impure words per sentence (fail / pass), and the
commonest (word POS <- class POS) among the fails.

| arm | b | classes | distinct new | syn-fail | impure share in syn-fail | impure share in syn-pass | impure words per sentence (fail / pass) | top impure (word POS <- class POS) among fails |
|---|---|---|---|---|---|---|---|---|
| e-i-on gate W'=7 | 0 | 1 | 1543 | 1257 | 1.000 | 1.000 | 4.93 / 4.14 | PRON<-VERB 1152, NOUN<-VERB 772, CCONJ<-VERB 753, DET<-VERB 698 |
| e-i-on gate W'=7 | 4 | 14 | 1436 | 865 | 0.984 | 0.921 | 2.97 / 2.36 | CCONJ<-AUX 564, PRON<-DET 244, PROPN<-PRON 193, ADV<-VERB 158 |
| e-i-on gate W'=7 | 5 | 27 | 1391 | 749 | 0.971 | 0.916 | 2.58 / 2.09 | CCONJ<-AUX 486, PRON<-DET 188, PROPN<-PRON 164, ADV<-NOUN 155 |
| e-i-on gate W'=7 | 6 | 50 | 1352 | 683 | 0.941 | 0.827 | 2.25 / 1.66 | CCONJ<-AUX 477, PRON<-DET 165, VERB<-AUX 133, ADV<-PROPN 107 |
| e-i-on gate W'=7 | 6.5 | 69 | 1262 | 477 | 0.782 | 0.746 | 1.29 / 1.26 | AUX<-CCONJ 109, AUX<-VERB 48, NOUN<-ADV 41, ADV<-PROPN 39 |
| e-i-on gate W'=7 | 7 | 97 | 1172 | 412 | 0.711 | 0.684 | 1.11 / 1.09 | AUX<-CCONJ 103, AUX<-VERB 53, ADV<-PROPN 33, VERB<-CCONJ 27 |
| e-i-off W'=7 | 0 | 1 | 1600 | 1580 | 0.999 | 1.000 | 4.41 / 3.65 | PRON<-VERB 1675, NOUN<-VERB 840, ADV<-VERB 812, ADJ<-VERB 783 |
| e-i-off W'=7 | 4 | 14 | 1585 | 1501 | 0.973 | 0.881 | 2.78 / 1.93 | PRON<-DET 347, ADP<-VERB 329, CCONJ<-AUX 318, PROPN<-PRON 290 |
| e-i-off W'=7 | 5 | 27 | 1574 | 1452 | 0.955 | 0.820 | 2.38 / 1.66 | CCONJ<-AUX 352, PRON<-DET 327, VERB<-AUX 285, PROPN<-PRON 268 |
| e-i-off W'=7 | 6 | 50 | 1574 | 1412 | 0.899 | 0.790 | 2.05 / 1.48 | CCONJ<-AUX 346, PRON<-DET 335, VERB<-AUX 303, ADV<-PROPN 180 |
| e-i-off W'=7 | 6.5 | 69 | 1533 | 1277 | 0.836 | 0.758 | 1.52 / 1.20 | AUX<-CCONJ 191, PRON<-DET 124, ADV<-PROPN 122, VERB<-CCONJ 91 |
| e-i-off W'=7 | 7 | 97 | 1499 | 1141 | 0.738 | 0.676 | 1.23 / 1.06 | AUX<-CCONJ 192, ADV<-PROPN 98, VERB<-CCONJ 88, PRON<-NOUN 79 |
| e-ii-4 gate W'=7 | 4 | 14 | 1451 | 947 | 0.965 | 0.907 | 2.75 / 2.12 | CCONJ<-AUX 485, PRON<-DET 236, ADV<-VERB 175, VERB<-AUX 158 |
| e-ii-4 W'=7 | 4 | 14 | 1592 | 1531 | 0.970 | 0.803 | 2.66 / 1.66 | PRON<-DET 351, CCONJ<-AUX 345, PROPN<-PRON 305, VERB<-AUX 291 |
| e-ii-5 gate W'=7 | 5 | 27 | 1434 | 847 | 0.933 | 0.867 | 2.20 / 1.91 | CCONJ<-AUX 452, PRON<-DET 189, PROPN<-PRON 152, VERB<-AUX 145 |
| e-ii-5 W'=7 | 5 | 27 | 1586 | 1487 | 0.951 | 0.899 | 2.33 / 1.98 | PRON<-DET 321, CCONJ<-AUX 308, VERB<-AUX 281, PROPN<-PRON 268 |
| e-ii-6 gate W'=7 | 6 | 50 | 1364 | 718 | 0.935 | 0.802 | 2.18 / 1.53 | CCONJ<-AUX 391, PRON<-DET 193, VERB<-AUX 140, ADV<-PROPN 92 |
| e-ii-6 W'=7 | 6 | 50 | 1571 | 1437 | 0.917 | 0.754 | 1.96 / 1.40 | CCONJ<-AUX 327, VERB<-AUX 299, PRON<-DET 293, ADV<-PROPN 156 |
| e-ii-6.5 gate W'=7 | 6.5 | 69 | 1316 | 533 | 0.799 | 0.728 | 1.35 / 1.16 | AUX<-CCONJ 118, VERB<-CCONJ 57, NOUN<-ADV 54, ADV<-PROPN 54 |
| e-ii-6.5 W'=7 | 6.5 | 69 | 1552 | 1340 | 0.845 | 0.708 | 1.65 / 1.08 | AUX<-CCONJ 267, PRON<-DET 141, ADV<-PROPN 118, VERB<-CCONJ 117 |
| e-ii-7 gate W'=7 | 7 | 97 | 1250 | 505 | 0.667 | 0.663 | 1.05 / 1.00 | AUX<-CCONJ 104, PROPN<-PRON 50, VERB<-CCONJ 40, AUX<-VERB 38 |
| e-ii-7 W'=7 | 7 | 97 | 1514 | 1211 | 0.752 | 0.670 | 1.29 / 0.90 | AUX<-CCONJ 231, ADV<-PROPN 108, VERB<-CCONJ 97, PROPN<-PRON 95 |

Reading: at coarse b (<= 6) every sentence carries impure emissions (2-5 per sentence) and the fails carry more than the passes (.94-.98 vs .83-.92
of sentences); at 6.5-7 bits the two groups carry the same amount (.711 vs .684, 1.11 vs 1.09 words per sentence with the gate; .667 vs .663 and 1.05 vs 1.00
in the named-input arm), so at the fine levels a syntax fail is not explained by impurity alone. The impure pairs are function-word mixtures
(CCONJ <- AUX: "and" in the class of "was / were / said"; AUX <- CCONJ; PRON <- DET; PROPN <- PRON), the word tree's "and was said" class; content
classes (nouns, emotion words) rarely appear among the top impure pairs. So the bottleneck is the function-word class, not the content
classes; a class source that separates auxiliaries, conjunctions, determiners and pronouns is what the R-syn regime would need.

## Go rule and verdict (class emission)
- **R-non >= .6**: yes (gate off, b <= 6: .90-.99; gate on b = 0: .815).
- **R-sem >= .6**: no in any class-emission arm; the gated token model (emit = leaf) reaches .558 (seed 13) and .577 (seed 17) with tri|d .766 / .737; R-sem rises
  monotonically with b (gate on, token input, seed 13: .069, .212, .268, .286, .409, .432, leaf .558).
- **R-syn >= .6**: no. Gate on, token input, seed 13: .116 (b 0), .186, .193, .209, .213, .217 (b 7), .206 (leaf); seed 17: .203 (6), .215 (6.5), .216 (7), .171 (leaf);
  maximum anywhere .249 (b 6.5, tau 2). The curve rises from the root and then sits at .2-.22 up to the leaf: no interior peak (the maximum is .01 above the leaf on
  seed 13 and .045 on seed 17, far from the pre-registered .2 above both ends), and the kill threshold (R-syn max < .4) applies.
  Gate off the R-syn share is lower (.012 to .129).
- **Input named at the same b** (ii) is no better than token input (i) at every b (R-sem .179-.407 against .212-.432; R-syn .168-.224 against .186-.217): naming the window loses R-sem and gains nothing in R-syn.
- **Temperature at b 6.5 (gate on)**: .7 / 1 / 2 give R-sem .445 / .409 / .301, R-syn .198 / .213 / .249, R-non .357 / .378 / .450. R-syn is nearly flat in tau (the gain
  from 1 to 2 is .036 while R-non grows .07), so the level and not the temperature does the little there is, and temperature alone (token model without emission, previous addendum) falls to R-syn .03-.14.
- **W' = 2** at b 6.5 / 7 (gate on): R-sem .470 / .469 (tri|d among them .595 / .557, above the .30 floor), R-syn .197 / .231, R-non .333 / .300: the best R-sem of any class-emission arm, still below .6.

**Verdict: NO-GO.** Output-side abstraction moves the generator from R-sem to R-non as b falls, and the gate keeps R-syn at about .2 for every b from 4 to the leaf, a level that the
leaf itself also has (.17-.21). The abstraction decouples word identity from context (the emitted words differ from the context-licensed word) but not syntax
from meaning: the sentences that lose meaning lose grammar too, because the class the word tree gives for function words is mixed. R-syn is not a regime that either dial
(input read level, window names, read reach, temperature, output class level) produces on this corpus with this judge; the largest purity anywhere is .249. Phase 2 is not warranted.
