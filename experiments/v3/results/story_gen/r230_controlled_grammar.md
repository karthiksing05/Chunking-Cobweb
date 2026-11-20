# r230: three regimes on a controlled grammar (syntax and meaning defined exactly)

Verdict: GO on dial C6 (class-level pair gate and emission at tree level b, reach W' = 5). At n = 1000 and both seeds, one dial gives R-sem at the leaf (.72/.73), R-syn at b = 6 and 7 (.64-.67 / .60-.65) and R-non at b <= 2 (.88-.98). R-syn needs n >= 1000 (.44-.49 at n <= 500). Dial C9 (context level) gives no R-syn (max .41).

## Setup
- Grammar and judge: `experiments/v3/diag/r230_grammar.py`; driver `r230_story.py` (r223_story.py with only t3_story.py's TinyStories data block replaced; `new: true` as in earlier rounds, so no generation is a stored training sentence). Learner, dials and generator are Trellis3R223 unchanged.
- Syntax: S -> NP VP; NP -> Det (Adj) N; VP -> Vi | Vt NP | Vpp Prep NP; a+sg, some+pl, the+either; verb agrees with subject; each verb has one frame, Vpp verbs fix their preposition (to/in). Semantics: 5 noun types (person, animal, food, place, object, 8 nouns each), verbs restrict subject/object/complement type, 14 adjectives restrict noun type. Lexicon 147 words (80 noun forms, 48 verb forms, 14 adj, 3 det, 2 prep); sentence length 3-8; 14,720 distinct sentences in 20,000 samples. Nouns pluralise by "+s" ("womans", "foxs" are legal tokens).
- Judge (fixed before any generation): R-non = does not parse (NP structure, agreement, frame, preposition); R-syn = parses, some selectional restriction fails; R-sem = parses and all hold.
- TinyStories-study configuration: cfg_w5 of r214p01 (window 5, edge exact, token input, word pair gate, tau 1), word tree `wtree_feats` function-word features. n = 1000 (and 250 to 4000), seeds 13, 17; 1,600 unconditioned generations per arm (400 prompts x 4 draws).

## Judge controls (n = 2000 each; controls.json)
| set | R-sem | R-syn | R-non |
|---|---|---|---|
| grammar samples | 1.000 | 0 | 0 |
| same-category swap, 1 word | .337 | .663 | 0 |
| same-category swap, 2 words | .151 | .849 | 0 |
| same-category swap, 3 words | .094 | .906 | 0 |
| word shuffle | .003 | .003 | .994 |
| verb number flipped | 0 | 0 | 1.000 |
| random words, same length | 0 | 0 | 1.000 |
(A swap lands in R-sem when it stays inside the semantic type, hence .337 at one swap.)

## Word-taxonomy purity per level (purity.json; pairs of class mates)
syn = same syntactic category (category = POS x number x verb frame/preposition); sem = same semantic type among class mates of the same kind (noun type, verb restriction profile, adjective restriction set). Chance over all word pairs: syn .17, sem .158. Features: top-5 frequent words (`func:5`); n = 1000, seed 13 / seed 17.
| b (bits) | classes | syn purity | sem purity |
|---|---|---|---|
| 0 | 1 | .17 / .17 | .158 / .158 |
| 2 | 3 | .469 / .456 | .158 / .150 |
| 3 | 9 | .590 / .709 | .147 / .135 |
| 4 | 17-18 | .778 / .817 | .164 / .153 |
| 5 | 34 | .872 / .806 | .170 / .178 |
| 6 | 50-53 | .912 / .925 | .176 / .152 |
| 7 | 69-71 | .906 / .916 | .205 / .152 |
| leaf | 106-107 | .902 / .928 | .230 / .193 |
So a level exists: from b = 5 to 7 the classes are about 90% syntactic-category pure and semantically at chance. The feature set matters: with all-neighbour features (`all`) sem purity rises with b (b=5: .50, b=6: .63/.50, b=7: .58/.64) while syn stays .94, so there the taxonomy is semantic; func:8 / 15 / 30 behave like func:5 up to b = 6 (sem .18-.34, rising with K). func:5 was chosen after seeing these purity tables (a representation choice, K = number of function-word types; C6 `wemitk` = 5 too).

## Dial table, n = 1000, window 5 (seed 13 / seed 17; shares of 1,600 generations: sem / syn / non)
C6 on func:5 tree, class gate and emission at b, W' = 5 (leaf = word gate, exact emission):
| setting | R-sem | R-syn | R-non |
|---|---|---|---|
| leaf, W'=5 | .728 / .719 | .239 / .249 | .033 / .032 |
| b=7 | .209 / .263 | .646 / .600 | .146 / .137 |
| b=6 | .193 / .234 | .667 / .636 | .140 / .130 |
| b=5 | .146 / .154 | .639 / .548 | .215 / .297 |
| b=4 | .090 / .116 | .504 / .562 | .406 / .322 |
| b=3 | .036 / .083 | .268 / .524 | .697 / .393 |
| b=2 | .016 / .011 | .100 / .108 | .884 / .881 |
| b=0 | .003 / .003 | .016 / .028 | .981 / .969 |
Reach at leaf (word gate): W'=0 .065 / .074 sem, non .892 / .854; W'=1 sem .219 / .210, syn .230 / .258, non .551 / .532; W'=2 sem .658 / .713, non .201 / .144; W'=3 sem .828 / .846 (best); W'=5 .728 / .719.
C9 (context window read at b, exact emission, word pair gate), W'=5 (W'=3 similar): b=leaf sem .728/.719; 7: .723/.686; 6: .713/.709; 5: .702/.684; 4: .634/.616; 3: .636/.617; 0: .584/.592 sem with syn .398/.391 and non .018. Max R-syn .41 (b = 0 to 4); non only appears at W' <= 2 (W'=1: non ~.52 at every b, sem ~.23-.26). C9 therefore does not separate syntax from meaning: the unchanged word-level pair gate keeps attested noun-verb and adjective-noun adjacencies, which carry the selectional restrictions, so abstracting the context leaves R-sem ~.6.

Criterion (R-syn >= .5 with R-sem >= .6 and R-non >= .6 elsewhere on the same dial): met on C6, W'=5: R-syn .64-.67 at b=6 (seed 13: .667, seed 17: .636), R-sem .72 at the leaf, R-non .97-.98 at b=0 (.88 at b=2; b=3 is .70 / .39, so the R-non end is b <= 2).

## Learning curves (C6, W'=5; sem / syn / non; seed 13 | seed 17)
| n | leaf (R-sem) | b=6 (R-syn) | b=0 (R-non) |
|---|---|---|---|
| 250 | .668/.223/.109 \| .639/.291/.069 | .212/.435/.353 \| .126/.371/.502 | .003/.015/.983 \| .003/.021/.976 |
| 500 | .669/.263/.069 \| .693/.255/.052 | .259/.487/.254 \| .111/.374/.515 | .007/.014/.979 \| .005/.026/.969 |
| 1000 | .728/.239/.033 \| .719/.249/.032 | .193/.667/.140 \| .234/.636/.130 | .003/.016/.981 \| .003/.028/.969 |
| 2000 | .709/.280/.011 \| .723/.269/.009 | .221/.641/.138 \| .270/.558/.172 | .001/.021/.978 \| .004/.021/.974 |
| 4000 | .748/.249/.003 \| .721/.277/.003 | .210/.685/.105 \| .184/.659/.156 | .001/.025/.974 \| .005/.028/.967 |
R-syn at b=6 grows with data (.37-.49 at n <= 500, .56-.69 at n >= 1000) as the word taxonomy sharpens; R-sem at the leaf does not reach 1: about a quarter of the leaf generations violate a selectional restriction (mostly object type) at every n, and R-non falls to .003.

## Examples (n = 1000, seed 13, W'=5; random novel generations, each judged by the rule)
R-sem (leaf): the sleepy woman eats the cake; a tall teacher buys the book; the big mans buy a box; the happy man sleeps; a old boy goes to some big parks.
R-syn (b=6): a pie bakes a cow [subject+object]; the woman returns to some cakes [object]; a boy feeds a sleepy park [object+adjective]; the dog stays in a old teacher [object]; a cat stays in the old teacher [object].
R-non (b=0): some runs returns some farmer; a laughs egg sleep; a dogs cookie man; some cat eats a stays dogs; the pears schools.
(Seed 17 and the b=0 R-syn / R-sem tails are in r230/examples.jsonl; at b=0, R-sem sentences are chance hits, e.g. "the man goes to a market", R-sem .003.)
At b=6 the R-non share (.13-.14) is mostly agreement ("the lemon spoil") and preposition errors.

## Caveats
- The word taxonomy for the R-syn dial uses function-word features (func:5), picked after reading purity for 5 feature sets; with all-neighbour features the same b levels are semantically pure (C6 with `all` was not run).
- Leaf R-sem tops at .75 (not 1.0), so "R-sem" here means the largest, not exclusive, share; the criterion .6 is met.
- At b=6 / 7 the average class has ~2 members; the regime comes from context-free within-class emission plus class-level gating (r214 `_emit`, `class_pair_ok`), no new machinery.
- 2 seeds; learning-curve seeds 13 and 17 only.

## Files and jobs
- Code: experiments/v3/diag/r230_grammar.py, r230_story.py, r230_p0_job.sh, r230_p1.sh, r230_judge_job.sh, r230_ex_job.sh.
- Data: experiments/v3/results/story_gen/r230/ (controls.json, purity.json, wt/, judge_p1.json, judge_p1b.json, judge_p1c.json, p1/ dumps).
- Jobs: step 0 (controls, word trees, purity) 4814018, 4814033; grid n=1000: C6/C9 4814047-4814062, C6 per-level 4814876, 4814884, ... (b = 6..0, both seeds, ids 4814900-4814964 range step 8 and 4814876-4814892); learning curves 4816007-4816030; judges 4814086, 4814087, 4814150, 4815764, 4816333, 4816904; examples 4816035. (A script bug at submission launched 80 duplicate C9 jobs, cancelled within seconds; the dumps they did not write are unaffected.)
