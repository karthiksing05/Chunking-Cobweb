
## n=250 (settings TRELLIS edge w7 t3.50, LSTM t0.70, GPT t0.70; seeds [13, 17, 19, 23, 29, 31, 37, 41]; attempted per seed {13: 1600, 17: 1544, 19: 1460, 23: 1476, 29: 1600, 31: 1592, 37: 1524, 41: 1600})

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

## n=500 (settings TRELLIS edge w6 t2.80, LSTM t0.80, GPT t0.80; seeds [13, 17, 19, 23, 29, 31, 37, 41]; attempted per seed {13: 1600, 17: 1544, 19: 1460, 23: 1476, 29: 1600, 31: 1592, 37: 1524, 41: 1600})

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

## n=1000 (settings TRELLIS edge w7 t2.00, LSTM t0.80, GPT t0.80; seeds [13, 17, 19, 23, 29, 31, 37, 41]; attempted per seed {13: 1600, 17: 1544, 19: 1460, 23: 1476, 29: 1600, 31: 1592, 37: 1524, 41: 1600})

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

## n=4000 (settings TRELLIS edge w5 t2.00, LSTM t1.00, GPT t0.90; seeds [13, 17]; attempted per seed {13: 1600, 17: 1544})

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

## Headline claims under judge2 (syntax pass as grammatical; same pre-registered rules as r212)

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
