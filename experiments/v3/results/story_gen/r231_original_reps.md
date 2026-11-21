# r231: three regimes on one dial with all-neighbour word representations

Verdict (updated after P3): NO-GO under the pre-registered criterion. At n = 1000, 2000 and 4000 no allsuf b in {5,6,7} reaches R-syn >= .5 on both seeds (seed 13 reaches .53-.63 at n >= 2000, seed 17 stays at .40-.49); the suffix facet, not data alone, is what lifts R-syn (control: all at n = 4000 gives R-syn .22-.45). Original P1/P2 verdict follows.

Verdict: NO-GO. With every neighbour kept, the r230 class-level dial (C6, W' = 5) gives R-sem at the leaf (.72-.85) and R-non at b <= 2 (.90-.98), but no setting reaches R-syn >= .5 on both seeds. The original 'all' tree peaks at R-syn .39-.42 (b = 4-6); the three representation variants peak at .33 (abs:4), .31 (pos) and .52 / .45 (allsuf, b = 6, with R-sem only .32). The syntactic regime of r230 (.64-.67 at b = 6, 7) came from the function-word-only features (func:5).

## Setup
Same grammar, judge, driver (`r231_story.py` = r230_story.py with the word-tree factory swapped), config cfg_w5, n = 1000, seeds 13 and 17, 1,600 generations per arm, temperature 1.0, W' = 5, as r230. Only the attributes fed to the word tree change (`r231_wordtree.py`). The dial is C6 (class pair gate and class emission at tree level b; leaf = word gate and exact emission).

Variants, fixed before running (one-line rationale each):
- abs:4: a word's L / R attributes are its neighbours' classes at b = 4 of the original 'all' tree (two passes; markers stay), so every neighbour is in but described by its class; class patterns are syntactic, specific co-occurrents carry meaning.
- pos: facets L, R (immediate neighbours, identities) plus W (positions +-2, +-3, identities); immediate context is separated from wide context, all neighbours kept.
- allsuf: the original L, R facets plus a suffix facet S (last 1-3 characters), added not substituted (nouns end in "s" when plural, verbs in "s" when singular).

## Word-taxonomy purity (purity.json; syn / sem; chance syn .17, sem .158; seed 13 | seed 17)
| b | classes (s13) | all | abs:4 | pos | allsuf |
|---|---|---|---|---|---|
| 3 | 7 | .64/.27 \| .64/.28 | .45/.27 \| .47/.32 | .63/.28 \| .62/.28 | .61/.26 \| .64/.26 |
| 4 | 12-14 | .80/.34 \| .77/.41 | .65/.35 \| .65/.40 | .85/.36 \| .79/.34 | .78/.31 \| .85/.30 |
| 5 | 23-30 | .82/.50 \| .87/.51 | .72/.37 \| .73/.38 | .84/.58 \| .80/.54 | .86/.38 \| .86/.37 |
| 6 | 41-51 | .95/.63 \| .94/.50 | .85/.33 \| .82/.36 | .93/.61 \| .95/.53 | .91/.47 \| .89/.46 |
| 7 | 61-75 | .94/.58 \| .91/.64 | .93/.26 \| .84/.31 | .91/.62 \| .93/.68 | .93/.40 \| .93/.53 |
abs:4 is the only variant that lowers sem purity at b = 6-7 (.26-.36 vs .50-.64) while syn stays .82-.93; at b <= 5 its syn purity is lower than 'all'. allsuf lowers sem somewhat (.37-.53). pos is the same as 'all'. The full table including b = 0 and 2 is in r231/purity.json.

## P1: dial on the original 'all' tree (sem / syn / non; seed 13 | seed 17)
| setting | R-sem | R-syn | R-non |
|---|---|---|---|
| leaf, W'=3 | .828 / .846 | .159 / .146 | .013 / .007 |
| leaf, W'=5 | .728 / .719 | .239 / .249 | .033 / .032 |
| b=7 | .646 / .562 | .316 / .318 | .039 / .121 |
| b=6 | .546 / .470 | .304 / .415 | .149 / .115 |
| b=5 | .284 / .302 | .307 / .412 | .409 / .286 |
| b=4 | .172 / .204 | .394 / .363 | .434 / .433 |
| b=3 | .074 / .088 | .201 / .203 | .725 / .709 |
| b=2 | .021 / .030 | .066 / .066 | .913 / .904 |
| b=0 | .003 / .003 | .016 / .028 | .981 / .969 |
R-sem at the leaf is .72-.85 and R-non >= .6 at b <= 3, but R-syn never exceeds .42: the 'all' classes at b = 4-6 are syntactically about 80-95% pure, but also semantically mixed (sem .34-.63), so a within-class swap often keeps the type and the class gate keeps typed adjacencies, and the share of non-parsing outputs at b = 4-5 is large (.29-.43). The P1 criterion (R-syn >= .5) fails. This corrects the r230 caveat: "C6 with all was not run" is now run, and it does not give R-syn.

## P2: variants (sem / syn / non; seed 13 | seed 17). Leaf and b = 0 as in P1 (leaf W'=5 .728/.719 sem; b=0 non .981/.969)
abs:4
| b | R-sem | R-syn | R-non |
|---|---|---|---|
| 7 | .549 / .548 | .331 / .289 | .119 / .163 |
| 6 | .438 / .489 | .300 / .278 | .262 / .234 |
| 5 | .334 / .304 | .339 / .339 | .327 / .356 |
pos
| b | R-sem | R-syn | R-non |
|---|---|---|---|
| 6 | .648 / .631 | .308 / .273 | .044 / .097 |
| 5 | .370 / .345 | .301 / .291 | .329 / .364 |
| 4 | .197 / .226 | .347 / .326 | .456 / .448 |
allsuf
| b | R-sem | R-syn | R-non |
|---|---|---|---|
| 6 | .316 / .321 | .519 / .448 | .165 / .231 |
| 5 | .227 / .239 | .479 / .398 | .294 / .363 |
| 4 | .155 / .127 | .354 / .331 | .491 / .542 |
Best for R-syn: allsuf at b = 6 (.519 seed 13, .448 seed 17; below .5 on seed 17), and allsuf is also the only representation where R-syn grows with b up to 6 (.35 -> .48 -> .52). It still falls short of func:5 (.64-.67), and its leaf-to-b = 6 sem drop is a real dial (.73 -> .32) but R-syn does not exceed R-sem enough to form a regime. abs:4 lowers sem purity as designed, but the generated R-syn does not follow: the class gate at b = 6-7 over abs:4 classes admits many non-parsing and many semantically valid outputs (R-sem .44-.55), so purity of the class mates is not what limits R-syn. pos gives the 'all' pattern.

## Examples (allsuf, seed 13, first three per regime in file order; setting: R-sem from leaf W'=5, R-syn from b=6, R-non from b=0; file r231/examples.jsonl holds each arm's three regimes)
R-sem (leaf): the boy helps some boys; the boys buy a cup; the girl cooks the pie.
R-syn (b=6): some cats help the quiet boy [subject]; some red chairs smile [subject]; the cat returns to some carrots [object].
R-non (b=0): some buy mans chase a cake; some farmers gardens to a fresh boys; the carry hungry the carries the dogs.

## Selections made after seeing results
- The b levels of each variant's dial (abs:4: 7, 6, 5; pos and allsuf: 6, 5, 4; plus leaf and 0) were chosen after the purity tables (syn high, sem low) and before any dial result. Variant b = 3, 2 were not run (R-non there is representation independent).
- The "best" variant for the examples (allsuf, b = 6) was chosen after seeing the dial tables.
- The variants were decided before running; no variant was added, dropped or tuned after results (abs level B = 4 was fixed in advance from the 'all' purity table, b = 3 not tried).

## Caveats
- 2 seeds, n = 1000 only; no learning curve (r230 showed R-syn at b = 6 grows with n, .37-.49 at n <= 500, so allsuf could cross .5 at larger n; not tested).
- Leaf arms are word-level and independent of the tree; the leaf rows repeat across variants by construction.
- abs:4 leaf classes merge words with identical class descriptions (110 / 117 classes), which affects only class names, not the leaf arm.
- Purity here is over class mates of the 147-word lexicon; the 'all' and variant trees of the same seed see the same 1000 sentences.
- b = 2 and 3 for the variants and the C9 context-level dial were not run.

## Files and jobs
- Code: experiments/v3/diag/r231_wordtree.py, r231_story.py, r231_p1.sh, r231_p0_job.sh, r231_examples.py, r231_ex_job.sh (judge: r230_judge_job.sh).
- Data: experiments/v3/results/story_gen/r231/ (purity.json, wt/, judge_all.json, judge_var.json, examples.jsonl, p1/ dumps).
- Jobs: word trees + purity 4816987; P1 'all' 4816981 (seed 13), 4816982 (seed 17); P2 4816996-4817001 (abs:4 s13/s17, pos, allsuf; earlier submissions 4816990-4816995 cancelled before starting to add b=0); judges 4817422 (variants), 4817798 (all); examples 4817804.

## P3: allsuf at larger n (sem / syn / non; seed 13 | seed 17)
Pre-registered criterion (set before running): GO if some b in {5,6,7} gives R-syn >= .5 on BOTH seeds at the same n, with leaf R-sem >= .6 and b=0 R-non >= .6 at that n. Setup as P2 (W'=5, 1,600 generations, word tree rebuilt at each n). Control: 'all' (no suffix) at n = 4000, b = 4-6.

allsuf, n = 2000
| setting | R-sem | R-syn | R-non |
|---|---|---|---|
| leaf | .709 / .723 | .280 / .269 | .011 / .009 |
| b=7 | .398 / .438 | .528 / .472 | .074 / .091 |
| b=6 | .361 / .382 | .556 / .482 | .083 / .136 |
| b=5 | .265 / .268 | .533 / .494 | .202 / .239 |
| b=0 | .001 / .004 | .021 / .021 | .978 / .974 |
allsuf, n = 4000
| setting | R-sem | R-syn | R-non |
|---|---|---|---|
| leaf | .748 / .721 | .249 / .277 | .003 / .003 |
| b=7 | .448 / .475 | .537 / .404 | .015 / .121 |
| b=6 | .355 / .435 | .626 / .419 | .019 / .146 |
| b=5 | .253 / .269 | .580 / .441 | .168 / .290 |
| b=0 | .001 / .005 | .025 / .028 | .974 / .967 |
Control, 'all' at n = 4000
| setting | R-sem | R-syn | R-non |
|---|---|---|---|
| b=6 | .574 / .743 | .245 / .225 | .181 / .033 |
| b=5 | .381 / .459 | .256 / .244 | .364 / .297 |
| b=4 | .251 / .231 | .373 / .452 | .376 / .317 |
(Seed 13 b=6 came from the first job, b=5 and 4 from a rerun; see jobs.)

Purity at b = 5 / 6 / 7 (syn / sem; seed 13 | seed 17; purity_p3.json)
| n | rep | b=5 | b=6 | b=7 |
|---|---|---|---|---|
| 2000 | allsuf | .92/.37 \| .92/.36 | .95/.50 \| .92/.55 | .94/.46 \| .93/.53 |
| 4000 | allsuf | .94/.40 \| .90/.40 | .97/.44 \| .93/.55 | .97/.45 \| .92/.57 |
| 4000 | all | .86/.65 \| .85/.62 | .95/.69 \| .97/.75 | .94/.80 \| .95/.74 |

Reading: R-syn under allsuf rises with n only on seed 13 (b=6: .519 -> .556 -> .626); seed 17 peaks at .494 (n=2000, b=5) and falls to .40-.44 at n = 4000, so the criterion fails at every n (seed 17 is the limiting seed; at n=2000 it misses by .006-.028). The leaf R-sem (.72-.75) and b=0 R-non (.97) conditions hold. In the control, 'all' at n = 4000 has class sem purity .62-.80 and R-syn only .22-.45, so more data alone does not produce the syntactic regime; the suffix facet does (allsuf sem purity stays .36-.57). The seed-17 shortfall is a mix of non-parsing outputs (R-non .12-.29 at n=4000, b=5-7) rather than semantic outputs, unlike seed 13 (.02-.17).
Caveats: 2 seeds; the criterion is a hard threshold at .5 and seed 17 at n = 2000 is within noise of it (.472-.494); no b = 3-4 for allsuf at these n; the Killed-by-memory jobs (4817839, 4817841; 32G) were rerun with 72G and tag `_r`, so the arms of those settings come from a re-learned learner with the same seed.
Jobs P3: word trees + purity 4817836; dials 4817837-4817842, reruns 4819223, 4819224; judges 4819217 (partial), 4820651 (final, judge_p3.json). Code: r231p3_p1.sh, r231p3_p0_job.sh. Data: r231/p3/, r231/wt3/, r231/purity_p3.json.
