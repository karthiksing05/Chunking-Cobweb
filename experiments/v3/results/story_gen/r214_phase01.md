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
