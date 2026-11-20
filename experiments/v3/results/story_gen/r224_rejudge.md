# Round 224: the generation comparison re-judged with judge2

The paper's headline (TRELLIS edge vs LSTM vs GPT on TinyStories, fair count) was scored with the Qwen2.5-3B judge (p > .5). This round re-scores the existing dumps with judge2 (r214: Qwen3-8B + phi-4 ensemble, syntax logit > -5.25, semantics logit > 2.5), no regeneration, judge2 code unchanged (`r214_judge2.py score`, thresholds file unchanged).

Code: `experiments/v3/diag/r224_rejudge.py` (`prep`, `report`), `r224_job.sh`, `r224_job_b.sh`. Sentences and judge2 outputs: `r224/sentences*.txt`, `r224/scores.tsv`, `r224/scores_b.tsv`, full report `r224/report_body.md` (same text as the tables below). Jobs (A40, overcap): 4808134 failed at the report step (n=4000 needed the r200 dumps too), 4808144 judged 74,149 distinct sentences (34 min), 4808463 judged the 17,507 additional n=4000 sentences and wrote the report. Logs `logs/r224-rejudge*.log`.

## Definitions (unchanged from `PAPER_GENERATION.md`, r204_fair.per_seed logic copied with the judge flag swapped)

- Dumps: n=250/500/1000 at the r212 confirmatory settings (TRELLIS edge n250 w7 t3.5, n500 w6 t2.8, n1000 w7 t2.0; LSTM t.7/.8/.8; GPT t.7/.8/.8), seeds 13 17 19 23 (r200/r204/r205 dumps) and 29 31 37 41 (r212 dumps), attempted draws per seed as in r212. n=4000: r200 + r204 dumps, seeds 13 and 17, at the r204 best settings (TRELLIS edge w5 t2.00, LSTM t1.00, GPT t0.90, chosen under the old judge; the n=4000 per-setting table below shows judge2 would move the GPT choice to t1.00 and nothing else).
- Fair count = distinct, not-stored generated sentences with an unstored trigram that the judge passes, per attempted draw. "judge2 syntax" = ensemble syntax pass used as grammatical. "meaningful" = syntax pass and semantics pass (the paper's separate semantic axis).
- Reproduction check: the old-judge column recomputed here equals r212 and r204 to the printed digit (e.g. n=1000 .3281/.3346/.3282; n=4000 .3187/.3580/.3621), so the pipeline reproduces the published numbers.
- Statistics: as r212 (paired per-seed difference TRELLIS minus the neural model, t 95% CI df=7, 20,000-draw percentile bootstrap over seeds, wins). n=4000 has 2 seeds, so only means and wins.

## Summary tables (fair count, mean over seeds)

| n | judge | TRELLIS edge | LSTM | GPT |
|---|---|---|---|---|
| 250 | old | .2419 | .2323 | .2068 |
| 250 | judge2 syntax | .3030 | .2877 | .2828 |
| 250 | judge2 meaningful | .1890 | .1828 | .1671 |
| 500 | old | .2982 | .2990 | .2738 |
| 500 | judge2 syntax | .3718 | .3731 | .3720 |
| 500 | judge2 meaningful | .2304 | .2435 | .2265 |
| 1000 | old | .3281 | .3346 | .3282 |
| 1000 | judge2 syntax | .4096 | .4200 | .4284 |
| 1000 | judge2 meaningful | .2584 | .2840 | .2711 |
| 4000 (2 seeds) | old | .3187 | .3580 | .3621 |
| 4000 (2 seeds) | judge2 syntax | .4163 | .4729 | .4716 |
| 4000 (2 seeds) | judge2 meaningful | .2220 | .2916 | .3042 |

Judge agreement (syntax, same distinct new sentences, old p>.5 vs judge2 syntax pass): 80.8% at n=250, 80.5% at 500, 80.1% at 1000, 79.0% at 4000 (kappa .55-.62). Disagreement is one-sided: judge2 passes more (overall pass rate .51/.58/.68/.72 against .41/.47/.56/.58). Of the disagreements, 77-83% are old-no/judge2-yes (e.g. n=1000: 5,005 against 1,203). The pass-rate shift is similar across families (+.08 to +.15; TRELLIS +.11 at 250 and 1000), largest for GPT at 4000 (.663 to .812).

## Headline claims

Rules as r212 (matches LSTM: CI includes 0 and lower bound > -.03; beats GPT: CI above 0). "FAILS" for matches-LSTM at n=250 is because TRELLIS is above the LSTM (CI excludes 0 from above), not below it.

| n | judge | TRELLIS-LSTM mean [t CI] wins | matches LSTM (t / boot) | TRELLIS-GPT mean [t CI] wins | beats GPT (t / boot) |
|---|---|---|---|---|---|
| 250 | old | +.0096 [-.0110, +.0301] 5/8 | HOLDS / HOLDS | +.0351 [+.0163, +.0539] 8/8 | HOLDS / HOLDS |
| 250 | judge2 syntax | +.0153 [+.0001, +.0305] 6/8 | no: TRELLIS ahead (boot [+.0037, +.0271]) | +.0202 [-.0037, +.0442] 7/8 | FAILS / FAILS (boot lower -.0002) |
| 250 | judge2 meaningful | +.0061 [-.0064, +.0187] 4/8 | HOLDS / HOLDS | +.0219 [+.0001, +.0436] 7/8 | HOLDS / HOLDS (borderline) |
| 500 | old | -.0008 [-.0129, +.0113] 4/8 | HOLDS / HOLDS | +.0244 [+.0098, +.0390] 8/8 | HOLDS / HOLDS |
| 500 | judge2 syntax | -.0013 [-.0127, +.0102] 3/8 | HOLDS / HOLDS | -.0002 [-.0278, +.0274] 4/8 | FAILS / FAILS |
| 500 | judge2 meaningful | -.0132 [-.0301, +.0038] 2/8 | FAILS / HOLDS (t lower -.0301) | +.0039 [-.0105, +.0183] 5/8 | FAILS / FAILS |
| 1000 | old | -.0065 [-.0279, +.0149] 3/8 | HOLDS / HOLDS | -.0002 [-.0227, +.0223] 4/8 | FAILS / FAILS |
| 1000 | judge2 syntax | -.0104 [-.0417, +.0208] 4/8 | FAILS / FAILS (t lower -.0417, boot -.0346) | -.0188 [-.0426, +.0051] 2/8 | FAILS / FAILS (boot upper CI excludes 0 below: [-.0367, -.0002]) |
| 1000 | judge2 meaningful | -.0256 [-.0518, +.0005] 1/8 | FAILS / FAILS (boot [-.0475, -.0075], LSTM ahead) | -.0127 [-.0374, +.0119] 2/8 | FAILS / FAILS |
| 4000 (2 seeds) | old | -.0393, 0/2 | n/a | -.0434, 0/2 | n/a |
| 4000 (2 seeds) | judge2 syntax | -.0566, 0/2 | n/a | -.0553, 0/2 | n/a |
| 4000 (2 seeds) | judge2 meaningful | -.0695, 0/2 | n/a | -.0821, 0/2 | n/a |

Verdicts per claim, with judge2 syntax pass as grammatical (the like-for-like re-score):

- Ties LSTM at 250-1000: partly survives. The point differences keep their size (+.015, -.001, -.010) and the sign pattern, and at n=500 the tie holds under both CIs. At n=250 TRELLIS is slightly ahead (t CI lower bound +.0001, 6/8 wins), so "ties" becomes "level or slightly ahead". At n=1000 the point difference is -.010 and 4/8 wins, but the CI is wider than under the old judge (t [-.0417, +.0208]) and its lower bound passes the pre-registered -.03 limit, so the pre-registered "matches" rule fails. On the stricter meaningful count the LSTM is ahead at n=1000 (-.0256, 1/8 wins, bootstrap CI excludes 0) and at n=500 (-.0132, 2/8).
- Beats GPT at 250 and 500: does not survive as stated. At n=250 TRELLIS is ahead on 7/8 seeds (+.0202) but the CIs include 0 on syntax (t lower -.0037, boot -.0002); on the meaningful count it clears 0 narrowly (t +.0001, boot +.0048). At n=500 the difference vanishes on syntax (-.0002, 4/8) and on meaningful (+.0039, 5/8). The old-judge 8/8 wins at both sizes shrink because the GPT fair count rises more than TRELLIS under judge2 (n=250: GPT +.076, TRELLIS +.061; n=500: GPT +.098, TRELLIS +.074); the share of unstored-trigram sentences that pass rises by the same amount for both at 250 (+.12 vs +.11), but GPT has more distinct draws per attempt.
- Level at 1000: survives and is slightly weaker for TRELLIS. TRELLIS is below GPT (-.0188 syntax, 2/8 wins, bootstrap CI [-.0367, -.0002] excludes 0 on the low side, t CI includes 0) and below LSTM (-.0104).
- Both pull ahead at 4000: survives, and is wider. With 2 seeds, TRELLIS trails the LSTM by .057 and GPT by .055 (syntax), 0/2 wins at every metric; on the meaningful count by .070 and .082. TRELLIS's judge2 meaningful share among unstored-trigram sentences is .286, against .405 (LSTM) and .479 (GPT).

## Syntax and semantics per family (distinct new sentences with an unstored trigram, mean over seeds of per-seed shares)

| n | family | old gram | judge2 gram | judge2 sem | meaningful (syn and sem) | meaningful / judge2 gram |
|---|---|---|---|---|---|---|
| 250 | TRELLIS edge | .451 | .565 | .359 | .352 | .624 |
| 250 | LSTM | .348 | .430 | .281 | .274 | .636 |
| 250 | GPT | .317 | .434 | .261 | .256 | .591 |
| 500 | TRELLIS edge | .468 | .584 | .369 | .362 | .619 |
| 500 | LSTM | .407 | .508 | .336 | .332 | .653 |
| 500 | GPT | .367 | .499 | .308 | .303 | .608 |
| 1000 | TRELLIS edge | .484 | .604 | .388 | .381 | .631 |
| 1000 | LSTM | .494 | .620 | .425 | .419 | .676 |
| 1000 | GPT | .471 | .614 | .394 | .389 | .633 |
| 4000 | TRELLIS edge | .410 | .535 | .287 | .286 | .534 |
| 4000 | LSTM | .498 | .657 | .406 | .405 | .616 |
| 4000 | GPT | .570 | .742 | .481 | .479 | .645 |

Reading: about 60% of the sentences each family writes in the unstored-trigram regime that pass syntax also pass semantics (.59-.68), nearly constant across n and family up to 1000. TRELLIS is level on that ratio at 250 and 500, slightly below the LSTM at 1000 (.631 against .676), and clearly lower at 4000 (.534 against .616 and .645). Semantic pass among all unstored-trigram sentences is almost entirely contained in syntax pass (.286 against .287 at 4000: semantic pass without syntax pass is rare under the ensemble).

### Detail: n=250 (settings TRELLIS edge w7 t3.50, LSTM t0.70, GPT t0.70; seeds [13, 17, 19, 23, 29, 31, 37, 41]; attempted per seed {13: 1600, 17: 1544, 19: 1460, 23: 1476, 29: 1600, 31: 1592, 37: 1524, 41: 1600})

Fair count, mean over seeds:

| metric | TRELLIS edge | LSTM | GPT |
|---|---|---|---|
| old judge (Qwen2.5-3B p>.5), as r212 | 0.2419 | 0.2323 | 0.2068 |
| judge2 syntax pass (headline re-score) | 0.3030 | 0.2877 | 0.2828 |
| judge2 meaningful = syntax and semantics pass | 0.1890 | 0.1828 | 0.1671 |

Per seed fair count (TRELLIS / LSTM / GPT), old -> judge2 syntax -> judge2 meaningful:

- seed 13: 0.253/0.244/0.223 -> 0.330/0.309/0.289 -> 0.193/0.200/0.170
- seed 17: 0.245/0.232/0.196 -> 0.296/0.280/0.245 -> 0.193/0.184/0.153
- seed 19: 0.247/0.277/0.225 -> 0.305/0.306/0.303 -> 0.195/0.199/0.176
- seed 23: 0.218/0.222/0.207 -> 0.259/0.259/0.299 -> 0.165/0.174/0.190
- seed 29: 0.239/0.219/0.229 -> 0.311/0.291/0.298 -> 0.188/0.172/0.175
- seed 31: 0.263/0.220/0.183 -> 0.312/0.277/0.274 -> 0.215/0.185/0.148
- seed 37: 0.235/0.198/0.196 -> 0.322/0.280/0.290 -> 0.188/0.167/0.167
- seed 41: 0.234/0.247/0.196 -> 0.289/0.300/0.263 -> 0.174/0.181/0.158

Among distinct new sentences with an unstored trigram (mean over seeds of the per-seed share): judged grammatical by old judge, by judge2 syntax, semantics pass (judge2), meaningful (syn and sem); distinct-per-draw and unstored-trigram share of distinct.

| family | old gram | judge2 gram | judge2 sem | meaningful | meaningful / judge2 gram | distinct/draw | tri/distinct |
|---|---|---|---|---|---|---|---|
| TRELLIS edge | 0.451 | 0.565 | 0.359 | 0.352 | 0.624 | 0.600 | 0.895 |
| LSTM | 0.348 | 0.430 | 0.281 | 0.274 | 0.636 | 0.745 | 0.898 |
| GPT | 0.317 | 0.434 | 0.261 | 0.256 | 0.591 | 0.703 | 0.927 |

Judge agreement on the same sentences (all distinct new generated sentences of the setting, pooled over seeds; old = Qwen2.5-3B p>.5, new = judge2 syntax pass):

| family | sentences | agree | old yes/new yes | old yes/new no | old no/new yes | old no/new no | kappa | old pass rate | new pass rate |
|---|---|---|---|---|---|---|---|---|---|
| TRELLIS edge | 7433 | 0.789 | 3290 | 368 | 1198 | 2577 | 0.580 | 0.492 | 0.604 |
| LSTM | 9240 | 0.817 | 3220 | 460 | 1233 | 4327 | 0.631 | 0.398 | 0.482 |
| GPT | 8718 | 0.814 | 2783 | 311 | 1311 | 4313 | 0.621 | 0.355 | 0.470 |
| all | 25391 | 0.808 | 9293 | 1139 | 3742 | 11217 | 0.617 | 0.411 | 0.513 |

Paired fair-count differences, OLD judge (r212 reproduction), 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | +0.0096 | [-0.0110, +0.0301] | [-0.0064, +0.0252] | 5/8 |
| TRELLIS-GPT | +0.0351 | [+0.0163, +0.0539] | [+0.0217, +0.0505] | 8/8 |

Paired fair-count differences, JUDGE2 syntax pass, 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | +0.0153 | [+0.0001, +0.0305] | [+0.0037, +0.0271] | 6/8 |
| TRELLIS-GPT | +0.0202 | [-0.0037, +0.0442] | [-0.0002, +0.0366] | 7/8 |

Paired fair-count differences, JUDGE2 meaningful, 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | +0.0061 | [-0.0064, +0.0187] | [-0.0032, +0.0161] | 4/8 |
| TRELLIS-GPT | +0.0219 | [+0.0001, +0.0436] | [+0.0048, +0.0385] | 7/8 |

### Detail: n=500 (settings TRELLIS edge w6 t2.80, LSTM t0.80, GPT t0.80; seeds [13, 17, 19, 23, 29, 31, 37, 41]; attempted per seed {13: 1600, 17: 1544, 19: 1460, 23: 1476, 29: 1600, 31: 1592, 37: 1524, 41: 1600})

Fair count, mean over seeds:

| metric | TRELLIS edge | LSTM | GPT |
|---|---|---|---|
| old judge (Qwen2.5-3B p>.5), as r212 | 0.2982 | 0.2990 | 0.2738 |
| judge2 syntax pass (headline re-score) | 0.3718 | 0.3731 | 0.3720 |
| judge2 meaningful = syntax and semantics pass | 0.2304 | 0.2435 | 0.2265 |

Per seed fair count (TRELLIS / LSTM / GPT), old -> judge2 syntax -> judge2 meaningful:

- seed 13: 0.307/0.298/0.290 -> 0.374/0.367/0.374 -> 0.246/0.239/0.233
- seed 17: 0.294/0.295/0.265 -> 0.375/0.357/0.336 -> 0.233/0.245/0.217
- seed 19: 0.308/0.339/0.298 -> 0.384/0.390/0.418 -> 0.238/0.275/0.258
- seed 23: 0.296/0.286/0.278 -> 0.376/0.360/0.401 -> 0.245/0.222/0.249
- seed 29: 0.300/0.298/0.269 -> 0.371/0.393/0.394 -> 0.232/0.238/0.223
- seed 31: 0.291/0.300/0.270 -> 0.364/0.368/0.356 -> 0.220/0.248/0.214
- seed 37: 0.289/0.275/0.227 -> 0.371/0.379/0.315 -> 0.210/0.238/0.180
- seed 41: 0.300/0.301/0.293 -> 0.359/0.371/0.383 -> 0.217/0.242/0.237

Among distinct new sentences with an unstored trigram (mean over seeds of the per-seed share): judged grammatical by old judge, by judge2 syntax, semantics pass (judge2), meaningful (syn and sem); distinct-per-draw and unstored-trigram share of distinct.

| family | old gram | judge2 gram | judge2 sem | meaningful | meaningful / judge2 gram | distinct/draw | tri/distinct |
|---|---|---|---|---|---|---|---|
| TRELLIS edge | 0.468 | 0.584 | 0.369 | 0.362 | 0.619 | 0.731 | 0.873 |
| LSTM | 0.407 | 0.508 | 0.336 | 0.332 | 0.653 | 0.857 | 0.858 |
| GPT | 0.367 | 0.499 | 0.308 | 0.303 | 0.608 | 0.828 | 0.900 |

Judge agreement on the same sentences (all distinct new generated sentences of the setting, pooled over seeds; old = Qwen2.5-3B p>.5, new = judge2 syntax pass):

| family | sentences | agree | old yes/new yes | old yes/new no | old no/new yes | old no/new no | kappa | old pass rate | new pass rate |
|---|---|---|---|---|---|---|---|---|---|
| TRELLIS edge | 9050 | 0.793 | 4246 | 416 | 1455 | 2933 | 0.583 | 0.515 | 0.630 |
| LSTM | 10624 | 0.811 | 4551 | 479 | 1529 | 4065 | 0.625 | 0.473 | 0.572 |
| GPT | 10268 | 0.810 | 3970 | 329 | 1624 | 4345 | 0.625 | 0.419 | 0.545 |
| all | 29942 | 0.805 | 12767 | 1224 | 4608 | 11343 | 0.614 | 0.467 | 0.580 |

Paired fair-count differences, OLD judge (r212 reproduction), 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0008 | [-0.0129, +0.0113] | [-0.0108, +0.0075] | 4/8 |
| TRELLIS-GPT | +0.0244 | [+0.0098, +0.0390] | [+0.0146, +0.0368] | 8/8 |

Paired fair-count differences, JUDGE2 syntax pass, 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0013 | [-0.0127, +0.0102] | [-0.0101, +0.0076] | 3/8 |
| TRELLIS-GPT | -0.0002 | [-0.0278, +0.0274] | [-0.0201, +0.0222] | 4/8 |

Paired fair-count differences, JUDGE2 meaningful, 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0132 | [-0.0301, +0.0038] | [-0.0253, +0.0007] | 2/8 |
| TRELLIS-GPT | +0.0039 | [-0.0105, +0.0183] | [-0.0074, +0.0148] | 5/8 |

### Detail: n=1000 (settings TRELLIS edge w7 t2.00, LSTM t0.80, GPT t0.80; seeds [13, 17, 19, 23, 29, 31, 37, 41]; attempted per seed {13: 1600, 17: 1544, 19: 1460, 23: 1476, 29: 1600, 31: 1592, 37: 1524, 41: 1600})

Fair count, mean over seeds:

| metric | TRELLIS edge | LSTM | GPT |
|---|---|---|---|
| old judge (Qwen2.5-3B p>.5), as r212 | 0.3281 | 0.3346 | 0.3282 |
| judge2 syntax pass (headline re-score) | 0.4096 | 0.4200 | 0.4284 |
| judge2 meaningful = syntax and semantics pass | 0.2584 | 0.2840 | 0.2711 |

Per seed fair count (TRELLIS / LSTM / GPT), old -> judge2 syntax -> judge2 meaningful:

- seed 13: 0.337/0.342/0.317 -> 0.428/0.421/0.404 -> 0.267/0.292/0.261
- seed 17: 0.337/0.332/0.356 -> 0.423/0.392/0.440 -> 0.259/0.264/0.295
- seed 19: 0.353/0.395/0.351 -> 0.421/0.490/0.477 -> 0.256/0.345/0.288
- seed 23: 0.326/0.308/0.274 -> 0.409/0.402/0.397 -> 0.269/0.272/0.225
- seed 29: 0.335/0.303/0.329 -> 0.422/0.383/0.432 -> 0.275/0.267/0.284
- seed 31: 0.299/0.320/0.330 -> 0.378/0.418/0.423 -> 0.233/0.268/0.285
- seed 37: 0.304/0.340/0.327 -> 0.392/0.417/0.437 -> 0.236/0.281/0.253
- seed 41: 0.334/0.338/0.342 -> 0.404/0.436/0.416 -> 0.273/0.282/0.278

Among distinct new sentences with an unstored trigram (mean over seeds of the per-seed share): judged grammatical by old judge, by judge2 syntax, semantics pass (judge2), meaningful (syn and sem); distinct-per-draw and unstored-trigram share of distinct.

| family | old gram | judge2 gram | judge2 sem | meaningful | meaningful / judge2 gram | distinct/draw | tri/distinct |
|---|---|---|---|---|---|---|---|
| TRELLIS edge | 0.484 | 0.604 | 0.388 | 0.381 | 0.631 | 0.809 | 0.838 |
| LSTM | 0.494 | 0.620 | 0.425 | 0.419 | 0.676 | 0.872 | 0.777 |
| GPT | 0.471 | 0.614 | 0.394 | 0.389 | 0.633 | 0.839 | 0.832 |

Judge agreement on the same sentences (all distinct new generated sentences of the setting, pooled over seeds; old = Qwen2.5-3B p>.5, new = judge2 syntax pass):

| family | sentences | agree | old yes/new yes | old yes/new no | old no/new yes | old no/new no | kappa | old pass rate | new pass rate |
|---|---|---|---|---|---|---|---|---|---|
| TRELLIS edge | 10025 | 0.805 | 5075 | 396 | 1560 | 2994 | 0.598 | 0.546 | 0.662 |
| LSTM | 10809 | 0.807 | 5849 | 419 | 1666 | 2875 | 0.589 | 0.580 | 0.695 |
| GPT | 10389 | 0.791 | 5213 | 388 | 1779 | 3009 | 0.571 | 0.539 | 0.673 |
| all | 31223 | 0.801 | 16137 | 1203 | 5005 | 8878 | 0.586 | 0.555 | 0.677 |

Paired fair-count differences, OLD judge (r212 reproduction), 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0065 | [-0.0279, +0.0149] | [-0.0228, +0.0101] | 3/8 |
| TRELLIS-GPT | -0.0002 | [-0.0227, +0.0223] | [-0.0161, +0.0184] | 4/8 |

Paired fair-count differences, JUDGE2 syntax pass, 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0104 | [-0.0417, +0.0208] | [-0.0346, +0.0131] | 4/8 |
| TRELLIS-GPT | -0.0188 | [-0.0426, +0.0051] | [-0.0367, -0.0002] | 2/8 |

Paired fair-count differences, JUDGE2 meaningful, 8 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0256 | [-0.0518, +0.0005] | [-0.0475, -0.0075] | 1/8 |
| TRELLIS-GPT | -0.0127 | [-0.0374, +0.0119] | [-0.0304, +0.0073] | 2/8 |

### Detail: n=4000 (settings TRELLIS edge w5 t2.00, LSTM t1.00, GPT t0.90; seeds [13, 17]; attempted per seed {13: 1600, 17: 1544})

Fair count, mean over seeds:

| metric | TRELLIS edge | LSTM | GPT |
|---|---|---|---|
| old judge (Qwen2.5-3B p>.5), as r212 | 0.3187 | 0.3580 | 0.3621 |
| judge2 syntax pass (headline re-score) | 0.4163 | 0.4729 | 0.4716 |
| judge2 meaningful = syntax and semantics pass | 0.2220 | 0.2916 | 0.3042 |

Per seed fair count (TRELLIS / LSTM / GPT), old -> judge2 syntax -> judge2 meaningful:

- seed 13: 0.318/0.383/0.372 -> 0.421/0.496/0.479 -> 0.220/0.316/0.317
- seed 17: 0.319/0.333/0.352 -> 0.411/0.449/0.464 -> 0.224/0.267/0.291

Among distinct new sentences with an unstored trigram (mean over seeds of the per-seed share): judged grammatical by old judge, by judge2 syntax, semantics pass (judge2), meaningful (syn and sem); distinct-per-draw and unstored-trigram share of distinct.

| family | old gram | judge2 gram | judge2 sem | meaningful | meaningful / judge2 gram | distinct/draw | tri/distinct |
|---|---|---|---|---|---|---|---|
| TRELLIS edge | 0.410 | 0.535 | 0.287 | 0.286 | 0.534 | 0.952 | 0.816 |
| LSTM | 0.498 | 0.657 | 0.406 | 0.405 | 0.616 | 0.972 | 0.742 |
| GPT | 0.570 | 0.742 | 0.481 | 0.479 | 0.645 | 0.923 | 0.688 |

Judge agreement on the same sentences (all distinct new generated sentences of the setting, pooled over seeds; old = Qwen2.5-3B p>.5, new = judge2 syntax pass):

| family | sentences | agree | old yes/new yes | old yes/new no | old no/new yes | old no/new no | kappa | old pass rate | new pass rate |
|---|---|---|---|---|---|---|---|---|---|
| TRELLIS edge | 2994 | 0.784 | 1322 | 130 | 518 | 1024 | 0.570 | 0.485 | 0.615 |
| LSTM | 3056 | 0.798 | 1729 | 94 | 523 | 710 | 0.556 | 0.597 | 0.737 |
| GPT | 2903 | 0.787 | 1833 | 93 | 524 | 453 | 0.466 | 0.663 | 0.812 |
| all | 8953 | 0.790 | 4884 | 317 | 1565 | 2187 | 0.547 | 0.581 | 0.720 |

Paired fair-count differences, OLD judge (r212 reproduction), 2 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0393 | n/a | n/a | 0/2 |
| TRELLIS-GPT | -0.0434 | n/a | n/a | 0/2 |

Paired fair-count differences, JUDGE2 syntax pass, 2 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0566 | n/a | n/a | 0/2 |
| TRELLIS-GPT | -0.0553 | n/a | n/a | 0/2 |

Paired fair-count differences, JUDGE2 meaningful, 2 seeds:

| contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|
| TRELLIS-LSTM | -0.0695 | n/a | n/a | 0/2 |
| TRELLIS-GPT | -0.0821 | n/a | n/a | 0/2 |

n=4000 per-setting fair count (mean of seeds 13/17), old judge vs judge2 syntax vs judge2 meaningful (selection re-check under judge2):

| family | setting | old | judge2 syn | judge2 meaningful |
|---|---|---|---|---|
| GPT | t0.30 | 0.0310 | 0.0373 | 0.0312 |
| GPT | t0.40 | 0.0762 | 0.0917 | 0.0742 |
| GPT | t0.50 | 0.1440 | 0.1700 | 0.1388 |
| GPT | t0.70 | 0.2714 | 0.3399 | 0.2472 |
| GPT | t0.80 | 0.3424 | 0.4293 | 0.2918 |
| GPT | t0.90 | 0.3621 | 0.4716 | 0.3042 |
| GPT | t1.00 | 0.3569 | 0.4856 | 0.2941 |
| GPT | t1.20 | 0.3096 | 0.4692 | 0.2333 |
| LSTM | t0.30 | 0.0242 | 0.0289 | 0.0251 |
| LSTM | t0.40 | 0.0774 | 0.0889 | 0.0751 |
| LSTM | t0.50 | 0.1452 | 0.1710 | 0.1422 |
| LSTM | t0.70 | 0.2766 | 0.3430 | 0.2529 |
| LSTM | t0.80 | 0.3400 | 0.4219 | 0.3010 |
| LSTM | t0.90 | 0.3546 | 0.4666 | 0.3076 |
| LSTM | t1.00 | 0.3580 | 0.4729 | 0.2916 |
| LSTM | t1.20 | 0.2975 | 0.4496 | 0.2336 |
| TRELLIS edge | w5 t1.30 | 0.2967 | 0.3880 | 0.2308 |
| TRELLIS edge | w5 t1.60 | 0.2939 | 0.3942 | 0.2193 |
| TRELLIS edge | w5 t2.00 | 0.3187 | 0.4163 | 0.2220 |

### Detail: Headline claims under judge2 (syntax pass as grammatical; same pre-registered rules as r212)

Rules: matches LSTM = CI of TRELLIS-LSTM includes 0 and lower bound > -.03; beats GPT = CI of TRELLIS-GPT above 0. Both t and bootstrap shown.

| n | judge | matches LSTM (t / boot) | beats GPT (t / boot) |
|---|---|---|---|
| 250 | old | HOLDS / HOLDS | HOLDS / HOLDS |
| 250 | judge2 syntax | FAILS / FAILS | FAILS / FAILS |
| 250 | judge2 meaningful | HOLDS / HOLDS | HOLDS / HOLDS |
| 500 | old | HOLDS / HOLDS | HOLDS / HOLDS |
| 500 | judge2 syntax | HOLDS / HOLDS | FAILS / FAILS |
| 500 | judge2 meaningful | FAILS / HOLDS | FAILS / FAILS |
| 1000 | old | HOLDS / HOLDS | FAILS / FAILS |
| 1000 | judge2 syntax | FAILS / FAILS | FAILS / FAILS |
| 1000 | judge2 meaningful | FAILS / FAILS | FAILS / FAILS |

## Notes and caveats

- Only the three confirmatory settings per n were judged (and every n=4000 setting of the three families); the r212 selection re-check (other temperatures) was not re-run with judge2, so the settings are still the old-judge choices. The n=4000 table shows judge2's best settings agree except GPT (t1.00 .4856 against t0.90 .4716 on syntax; under it the TRELLIS-GPT gap would widen to -.069).
- judge2's syntax threshold is deliberately permissive (logit > -5.25, calibrated so that well-formed nonsense passes), so the "grammatical" pass rates are higher than the old judge's by 10-14 points overall, and a sentence's meaning is counted only on the semantics axis. Controls (r214): C1 real sentences 1.00 syntax pass, C3adj mild adjacent swap .127 pass. The mild-swap miss means judge2 inflates the count somewhat for models that emit near-grammatical word order; this applies to all three families.
- Everything above is a re-score of the same dumps, so any difference from r212 is judge choice, not generation noise. The 8 seeds are the same as r212 (training sets overlap, as documented there).
- Process note: during setup one stray `python3` invocation (an empty stdin, no script) ran on the login node by mistake; it ran nothing. One later attempt was blocked by the harness.
