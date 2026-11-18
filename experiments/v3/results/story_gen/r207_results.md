# Round 207: mechanism 1 (history missing mass, "escape":"n1") and the mechanism-2 probe, n=4000 seed 13

Code: `src/trellis2/trellis3_r207.py` (`Trellis3R207(Trellis3R204)`, flag `escape_mass`, `escape_split`), arm key `"escape":"n1"` in
`experiments/v3/diag/t3_story.py` (`_fills`, additive; used with `"edge":"exact"`, `units:"words"`, pair gate), mechanism-2 hook `STORY_R207_M2` +
`experiments/v3/diag/r207_m2.py`, `experiments/v3/diag/r207_fair.py` (r204_fair with escape arms reported as the edge family). Tests:
`tests/trellis2/test_r207_escape.py` (6 tests: byte-identical default and edge reads, flag leaves `_gap_fills` unchanged, c/N1 equal direct counting,
toy N1/(c+N1), unseen history / start step / degenerate cases unchanged, pair-level cap); full `tests/trellis2` suite 121 passed (sbatch 4783075).

Rule as implemented: for history (l2, l1) with c > 0 (word occurrences in memory) the fillers the read admits are split into attested, novel over a
stored pair, and novel over an unstored pair (the type-licence admits). Novel total = e = N1/(c+N1); the licence-admitted share is
min(e x its natural share of the novel mass, N1(l1)/(c(l1)+N1(l1))), the rest of e goes to novel trigrams over stored pairs, the remainder to attested
fillers; each side keeps the tree read's relative masses, the total mass is unchanged, and each filler's word / end split is kept (one factor on both).
c = 0 or start step (l1 = <s>) or nothing novel / attested admitted: read unchanged. The rescale is applied before the temperature power, so
temperature flattens the shares afterwards (this matters below).

## 1. Count pre-test (no generation), r204-n4000-w5 seed-13 dumps, training-stream counts

AUC = P(score of a grammatical sentence > score of an ungrammatical one). Class C: exactly one novel interior trigram over two stored pairs,
score = N1/(c+N1) of history (a, b). Class A: exactly one unstored interior pair, score = N1/(c+N1) of the left word a (word fillers only).

| class | t1.3 | t1.6 | t2.0 | all temps | n (gram / ungram, all) |
|---|---|---|---|---|---|
| C single junction | .573 | .544 | .590 | **.568** | 848 / 600 |
| C, pronoun pivots (you, it, i) | .800 | .597 | .821 | .700 | 28 / 120 |
| A | .615 | .664 | .619 | **.632** | 52 / 243 |

Kill rule (C < .55 or A < .60): not triggered (C .568 all, .590 at t2.0; A .632 all, .619 at t2.0). Class C is close to the line at t1.6 (.544); A rests
on 52 grammatical sentences. Generation ran.

## 2. Generation probe, n=4000 seed 13, edge + escape, w5 (200G, 1 seed, 1600 draws)

| arm | fair | strict | dist/draw | tri\|d | gram\|tri | atyp share | repeats |
|---|---|---|---|---|---|---|---|
| **edge + escape t2.0** | **.324** | .258 | .972 | .938 | **.355** | .244 | 45 |
| edge + escape t2.4 | .296 | .234 | .953 | .961 | .323 | .272 | 75 |
| r204 edge w5 t2.0 (same seed) | .318 | .264 | .954 | .810 | .411 | - | 73 |
| LSTM t1.0 / GPT t0.9 (same seed) | .383 / .372 | .347 / .333 | .966 / .917 | .726 / .691 | .546 / .587 | | |

Pass rule: fair >= .335 and gram|tri >= .44 at tri >= .75: **failed** (fair .324, gram|tri .355 at tri .938). Fair-only kill (<= .322): not met at t2.0 (+.006
over r204 edge, inside seed noise, SE about .011), but t2.4 is below it (.296) and the design's second kill (gram|tri < .43 at tri|d >= .75) is met at both
temperatures. Not measured: per-step novelty at history count 32-63 (pass condition); skipped because the other conditions already fail.

Reading: the escape made the sampler innovate more, not less. Distinct novel share tri|d rose .810 -> .938 and every-trigram-stored fell to .06 / .04, while
gram|tri fell .411 -> .355. Two causes visible without further runs: (1) most generated steps sit after rare histories, where N1/(c+N1) is large (about .5 at
c = 1), and r206's own table puts real text at .55 there, so the rule hands out the mass real text uses, but the tree read then spreads it over novel words whose
grammaticality at those histories is low; (2) temperature t2.0-2.4 flattens the (1-e, e) split again after the rescale, so the temperature that was tuned for the
old read now over-innovates. The optimum for this arm is probably at t <= 1.3-1.6, outside the brief's grid (t2.0, t2.4). I did not run it, because pass was
not reached and the brief's step 3 is conditional on a pass. A cheap follow-up is one n=4000 seed-13 run at t1.0/1.3/1.6 (about 25 min learn + 1.2 h per arm).

Steps 3 (seed 17 at n=4000, n=1000 on 4 seeds) were not run (probe failed).

Files: `r207-fair-n4000.{txt,perseed.tsv}`, `r207-frontier-n4000.jsonl`, dumps `r207-n4000-w5` (t2.0), `r207b-n4000-w5` (t2.4).

## 3. Mechanism-2 probe (no generation): pivot_support kind "left", seed 13

Rebuilt learner at n=4000 (and n=1000 for comparison), 1456 single-junction class C sentences (n=4000; 1399 at n=1000) from the r204 w5 dumps (t1.3/1.6/2.0).
AUC (grammatical > ungrammatical); `r207_m2_n{1000,4000}.tsv` has each sentence's scores, logs `logs/v3-r207-m2-n*.log`.

| subset (n=4000) | n | left AUC | pivot AUC (r192 kind) |
|---|---|---|---|
| all | 1456 | **.525** | .562 |
| t1.3 / t1.6 / t2.0 | 497 / 487 / 472 | .532 / .556 / .480 | .568 / .556 / .557 |
| pronoun pivots (you, it, i) | 150 (28 gram) | **.665** | .645 |
| other pivots | 1306 | .505 | .554 |
| pivot you / it / i | 95 / 42 / 13 | .578 / .623 / .800 | .541 / .647 / .650 |
| pivot and / was / the / to | 198 / 125 / 82 / 61 | .526 / .684 / .323 / .404 | |

n=1000 left AUC: all .519, pronoun pivots .692 (88 sentences). Pass was >= .62 overall and >= .65 on pronoun pivots; kill < .57. **Killed**: overall .525 (the
pronoun AUC passes alone, but 150 sentences with 28 grammatical is thin, and the pronoun pivots are 10% of class C). The "left" read is anti-predictive on
"the" (.32) and "to" (.40): a soft left weight would hurt there. Mechanism 2 generation was not implemented.

### Tree statistics (direct test of r206 explanation (a)), seed 13 w5

| | n=1000 | n=4000 |
|---|---|---|
| nodes / leaves / internal | 37.8k / 27.6k / 10.2k | 132k / 98k / 34k |
| leaves holding word fillers | 5087 | 19746 |
| leaf occupancy, mean / p99 / max | 1.04 / 2 / 5 | 1.09 / 3 / 32 |
| leaves with exactly 1 filler | .970 | .957 |
| leaf depth mean (filler-weighted) / max | 8.4 (8.4) / 13 | 9.8 (9.7) / 17 |
| root children (top concepts) | 3 | 3 |
| concepts by size (16-63 fillers): median distinct types / effective types | 17 / 14.6 | 11 / 8.7 |
| concepts by size (64-255): types / effective | 46 / 34.3 | 45 / 32.0 |
| concepts by size (256+): types / effective | 114 / 63.8 | 101 / 53.8 |
| depth-5 concepts: median size / types | 17 / 7 | 88 / 22 |
| top three concepts (fillers; types) | 2773;199, 2412;210, 121;40 | 11267;238, 9780;238, 471;83 |

Leaves stay singletons (96-97%), so leaf occupancy does not grow; the tree gains depth (+1.4 levels) and about 3.4x more concepts. At equal size, concepts at n=4000
hold fewer types (more specific), not more mixed ones. What grows is the share of mass in the three top concepts, which stay nearly vocabulary-wide (238 of 249 types):
about 21k fillers of 21.5k sit in two concepts with 238 types each. This does not support "coarser or more mixed concepts" at fixed size; it does show that
the first two levels carry no word-class information at either n, so context similarity beyond depth 4-5 does the work.

## 4. Verdict

- Mechanism 1: pre-test passed (C .568 / A .632), generation failed. The history-missing-mass rule fixed what r206 diagnosed (novelty now responds to evidence in
  principle) but, applied through the unchanged read and the t2.0 temperature, raised novelty to tri|d .94 and lowered gram|tri to .355; fair .324 vs .318 (noise).
  The design as specified (t2.0 / t2.4) is closed. Untested: a lower temperature grid, which would bring tri|d back toward .8 and is the one cell worth one run; I put
  its fair above .335 at below 25%, since gram|tri at matched novelty (r206: .47 at tri .70) is what it would have to beat.
- Mechanism 2: killed by the probe (AUC .525 overall).
- n=4000 parity with LSTM/GPT is not reached; both candidate mechanisms from r206 are now tested or killed on seed 13.

## 5. Rule slips

- One `python3` empty-heredoc command was run on the login node by mistake (no script, nothing computed; the permission classifier refused it). No other python ran there.
- t3_story.py was edited (three additive hunks plus the STORY_R207_M2 hook); default path unchanged (new keys popped only in the causal branch, branches guarded by the arm key).
- A first waiter used `rg` inside `bash -c` (not found) and returned at once; the later waiters were capped with `timeout`. No background process is left running.
- The logs/ directory is git-ignored, so the r207 .sh scripts and logs are not committed (as in earlier rounds).
