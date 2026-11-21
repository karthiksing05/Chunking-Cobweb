# r232: does the word representation limit the syntax-only regime on TinyStories? (func:5 vs func:100)

**Verdict: the representation does NOT limit R-syn on TinyStories.** func:5 changes best-interior-b R-syn by -.046 (seed 13: .209 vs .255) and -.046 (seed 17: .189 vs .235) relative to func:100, so the gain is below .05 on both seeds (the pre-registered "does not limit" branch; it is in fact a small loss, not a gain). The .5 target is not reached by any cell (maximum .255). The control reproduces r214h (b=7 seed 13 R-syn .255).

## Pre-registration (written before any result)
Question: on TinyStories, does the word representation (func:100 taxonomy) limit R-syn (grammatical but meaningless) of the class-level abstraction dial? Best R-syn of C6a is about .27.
- Arms: word taxonomy func:5 (5 function-word features) vs control func:100 (= C6a, rerun in the same job batch and the same judge pass), seeds 13 and 17, n=1000 sentences.
- Everything else as r214_p1h.sh: cfg.json (window 7, paper recipe), edge exact, token input, tau 1, W' 7, class-level pair gate (wgate b) and class emission (wemit b) at the same b, 50 most frequent types exact (K=50, wemitk 50), 400 prompts x 4 draws (1600 attempted), judge r214_judge2 (Qwen3-8B + phi-4, thresholds r214_judge2_thresholds.json).
- Dial levels, fixed before generation: leaf (gated word leaf, tau 1), b = 7, 6, 5, 4, 0. TinyStories has no semantic labels, so only POS purity (spaCy UPOS, r214_wordmap.py) is tabulated; b is not chosen from it (the set is the C6a set).
- "Best interior b" = the b in {7,6,5,4} with the highest R-syn, per tree and seed.
- Criterion (fixed now): the representation limits R-syn on TinyStories if func:5 raises best-interior-b R-syn by >= .10 over func:100 on both seeds; it does NOT limit it if the gain is < .05 on both seeds; anything else is inconclusive. The .5 target is also reported.
- Code: experiments/v3/diag/r232_p0_job.sh (purity), r232_p1.sh (generation, a copy of r214_p1h.sh with the tree as a parameter), r232_judge_job.sh (collect, judge, report with r214_phase01.py).

## Setup as run
Word trees func:5 and func:100 (n=1000 TinyStories sentences, seeds 13 and 17; 239 / 245 word types), all else as C6a: cfg.json (window 7), edge exact, token input, tau 1, W' 7, class-level gate and emission at b, K=50 exact types, 400 prompts x 4 draws, one judge pass (Qwen3-8B + phi-4, `r214_judge2.py score`, thresholds of `r214_judge2_thresholds.json`) over all four generation dirs, so judge drift is shared. Leaf arm = gated word leaf (no tree), identical for both trees by construction (it is a within-pair sanity check: the rows match exactly). Arms: leaf, b = 7, 6, 5, 4, 0.

## Purity table (POS purity = token-weighted majority spaCy UPOS per class; classes in brackets; seed 13 | seed 17)
No semantic labels exist for TinyStories, so only POS purity is available; b was fixed in advance, not chosen from this table.

| level | func:5 classes | func:5 POS purity | func:100 classes | func:100 POS purity |
|---|---|---|---|---|
| b=0 | 1 \| 1 | .177 \| .176 | 1 \| 1 | .177 \| .176 |
| b=4 | 20 \| 16 | .562 \| .520 | 15 \| 12 | .464 \| .461 |
| b=5 | 35 \| 32 | .665 \| .669 | 25 \| 29 | .554 \| .634 |
| b=6 | 59 \| 59 | .750 \| .758 | 49 \| 49 | .644 \| .676 |
| b=7 | 92 \| 93 | .806 \| .823 | 103 \| 110 | .796 \| .860 |
| leaf | 141 \| 142 | .868 \| .880 | 237 \| 240 | .964 \| .966 |

func:5 classes are purer in POS than func:100 at b=4-6 (+.06 to +.11), equal at 7. The purity gain does not carry into R-syn.

## Regime table (R-sem / R-syn / R-non; seed 13 | seed 17; judged distinct new sentences, 1600 attempted draws)

| setting | func:5 R-sem | func:5 R-syn | func:5 R-non | func:100 R-sem | func:100 R-syn | func:100 R-non |
|---|---|---|---|---|---|---|
| leaf (word gate) | .558 \| .577 | .206 \| .171 | .235 \| .252 | .558 \| .577 | .206 \| .171 | .235 \| .252 |
| b=7 | .180 \| .179 | .188 \| .189 | .632 \| .632 | .331 \| .330 | **.255** \| **.235** | .415 \| .434 |
| b=6 | .153 \| .160 | **.209** \| .170 | .638 \| .670 | .197 \| .192 | .208 \| .193 | .595 \| .615 |
| b=5 | .166 \| .145 | .195 \| .145 | .639 \| .710 | .156 \| .183 | .169 \| .148 | .676 \| .669 |
| b=4 | .094 \| .084 | .141 \| .143 | .766 \| .773 | .107 \| .093 | .139 \| .127 | .754 \| .780 |
| b=0 | .034 \| .034 | .095 \| .103 | .871 \| .863 | .034 \| .034 | .095 \| .103 | .871 \| .863 |

Best interior b (b in 4-7) by R-syn: func:5 .209 (b=6) | .189 (b=7); func:100 .255 (b=7) | .235 (b=7).

## Control comparison against the criterion
| seed | func:5 best interior R-syn | func:100 best interior R-syn (control) | gain | branch |
|---|---|---|---|---|
| 13 | .209 (b=6) | .255 (b=7) | -.046 | < .05 |
| 17 | .189 (b=7) | .235 (b=7) | -.046 | < .05 |

Both seeds fall in "gain < .05": no limit from the word representation. The control reproduces the r214h row for seed 13 (b=7: R-sem .331, R-syn .255, R-non .415, identical to Addendum 7), so the judge pass is consistent with the earlier one. No cell comes near the .5 target (maximum over all 24 rows: .255). The contrast with the controlled grammar (func:5 .60-.67) is therefore not a representation effect on TinyStories: even with POS-purer classes, func:5 loses meaning (R-sem .18 vs .33 at b=7) but converts the loss into R-non (.63 vs .42), not R-syn.

## Judged examples (func:5 arms, from r232/report.md random samples with fixed seed 214; syn_p / sem_p from the judge)
R-sem (syntax and meaning pass)
- tom felt sad near the water (1.00 / 1.00); tim was scared (1.00 / 1.00); he went to his mom (1.00 / 1.00)

R-syn (syntax passes, meaning fails)
- they said sorry to each box (0.72 / 0.05); he wanted to play for his small water (0.71 / 0.00); she said lily (0.02 / 0.07, labelled syn by the low syntax threshold)

R-non (syntax fails)
- they had a says fun (0.00 / 0.00); a big bird outside in a dog (0.00 / 0.00); inside he are a big cat (0.00 / 0.00)

## Caveats
- The R-syn regime here is defined by the judge's low syntax-pass threshold (see `r214_judge2.md`); borderline items such as "she said lily" (syn_p .02) count as syn. The same judge is used for both trees.
- func:5 K=50: the number of exact types is 50 for both trees (50 most frequent types, ~.64 of tokens), as in C6a; K was not varied.
- n=1000, two seeds, one draw count; differences of about .02 are within seed noise (the b=7 control differs by .02 between seeds). The -.046 gap is consistent in sign and size across seeds, but this experiment is powered to detect gains of .05-.10, not to say whether func:5 is worse.
- POS purity uses spaCy majority UPOS; TinyStories has no semantic labels, so the controlled-grammar "syntax high, semantics low" purity criterion could not be applied and b was fixed to the C6a set.
- The earlier oracle (r214h Addendum 7) put the R-syn ceiling for within-class swapping at about .2 on this setup; the func:5 result agrees with that: R-syn stays at .14-.21 across all b.
- The leaf rows are identical for func:5 and func:100 because the leaf arm does not use the word tree.

## Files and jobs
Code: `experiments/v3/diag/r232_p0_job.sh`, `r232_p1.sh`, `r232_judge_job.sh` (reuse `r214_story.py`, `r214_wordmap.py`, `r214_phase01.py`, `r214_judge2.py`). Results: `experiments/v3/results/story_gen/r232/{report.md,purity_*.md,purity_*.json,summary.json}` (generation dumps and scores.tsv left uncommitted, in `r232/g_*` and `r232/scores.tsv`).
Jobs (overcap): purity 4821192; generation func:5 s13 4821193, func:100 s13 4821194, func:5 s17 4821195, func:100 s17 4821196; judge (A40) 4821197.
