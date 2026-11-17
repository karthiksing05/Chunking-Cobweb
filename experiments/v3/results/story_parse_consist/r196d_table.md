seeds [13, 17, 19]; held-out 300; strings 125

### 250 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.273 [0.273 / 0.278 / 0.270] | 0.461 [0.459 / 0.468 / 0.456] | 0.343 [0.342 / 0.348 / 0.340] | 0.449 [0.453 / 0.457 / 0.437] | 0.591 [0.622 / 0.614 / 0.535] | 0.294 [0.287 / 0.280 / 0.316] | 0.358 [0.377 / 0.382 / 0.315] | 0.847 [0.841 / 0.844 / 0.857] | 0.587 [0.602 / 0.595 / 0.565] |
| Vwin | 3.303 [3.303 / 3.303 / 3.303] | 0.292 [0.300 / 0.308 / 0.268] | 0.492 [0.505 / 0.519 / 0.452] | 0.366 [0.376 / 0.386 / 0.337] | 0.448 [0.458 / 0.465 / 0.420] | 0.585 [0.606 / 0.591 / 0.559] | 0.297 [0.285 / 0.276 / 0.331] | 0.322 [0.341 / 0.317 / 0.310] | 0.830 [0.826 / 0.830 / 0.833] | 0.623 [0.630 / 0.630 / 0.608] |
| Vnowin | 3.303 [3.303 / 3.303 / 3.303] | 0.319 [0.319 / 0.319 / 0.319] | 0.537 [0.537 / 0.537 / 0.537] | 0.400 [0.400 / 0.400 / 0.400] | 0.362 [0.362 / 0.362 / 0.362] | 0.465 [0.465 / 0.465 / 0.465] | 0.329 [0.329 / 0.329 / 0.329] | 0.212 [0.212 / 0.212 / 0.212] | 0.873 [0.873 / 0.873 / 0.873] | 1.000 [1.000 / 1.000 / 1.000] |
| Vpmi | 3.303 [3.303 / 3.303 / 3.303] | 0.146 [0.078 / 0.072 / 0.288] | 0.245 [0.131 / 0.121 / 0.485] | 0.183 [0.098 / 0.090 / 0.361] | 0.257 [0.219 / 0.190 / 0.363] | 0.220 [0.126 / 0.110 / 0.425] | 0.507 [0.595 / 0.601 / 0.324] | 0.347 [0.426 / 0.424 / 0.190] | 0.811 [0.772 / 0.793 / 0.868] | 0.375 [0.457 / 0.441 / 0.227] |
| Vccm | 3.303 [3.303 / 3.303 / 3.303] | 0.179 [0.169 / 0.160 / 0.208] | 0.302 [0.284 / 0.270 / 0.350] | 0.225 [0.211 / 0.201 / 0.261] | 0.267 [0.285 / 0.253 / 0.263] | 0.168 [0.165 / 0.150 / 0.189] | 0.473 [0.479 / 0.495 / 0.445] | 0.293 [0.265 / 0.319 / 0.295] | 0.833 [0.864 / 0.828 / 0.806] | 0.424 [0.405 / 0.453 / 0.413] |
| Vzero | 3.303 [3.303 / 3.303 / 3.303] | 0.319 [0.319 / 0.319 / 0.319] | 0.537 [0.537 / 0.537 / 0.537] | 0.400 [0.400 / 0.400 / 0.400] | 0.362 [0.362 / 0.362 / 0.362] | 0.465 [0.465 / 0.465 / 0.465] | 0.329 [0.329 / 0.329 / 0.329] | 0.212 [0.212 / 0.212 / 0.212] | 0.873 [0.873 / 0.873 / 0.873] | 1.000 [1.000 / 1.000 / 1.000] |

diagnosis (V0, pooled over seeds): strings always bracketed 57, never 152, mixed 166; pairs of occurrences of one string with the SAME cc (content concept bands) 1653/3948 = 0.419

bracket rate by position: interior 243/1122 = 0.217; start 350/534 = 0.655

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.280 | 0.99 |
| ctx | +1.034 | 0.89 |
| cnt | +0.000 | 0.00 |
| t_share | -0.646 | 0.13 |
| t_win | +1.732 | 0.88 |
| t_cc | -0.025 | 0.46 |
| t_pv | -0.027 | 0.22 |
| g_win | +1.069 | 0.89 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.679, between-string sd 0.623
- t_win: within-string sd 1.333, between-string sd 1.232
- t_cc: within-string sd 0.157, between-string sd 0.216
- g_win: within-string sd 0.744, between-string sd 0.665
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "wanted to" (2/18 bracketed)
  - ONE UNIT: mia <wanted to> help tim find his mom  | post 0.31 ctx 1.09 [share -1.68 win 3.18 cc -0.16 pv -0.25] cnt 0.00 g_win 1.50 g_nowin 0.00 | cc [['c1762', 0.2], ['c25149', 0.2], ['c33358', 0.2], ['c4078', 0.2], ['c470', 0.2]]
  - split   : tom <wanted to> help  | post 0.15 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c14428', 0.2], ['c2128', 0.2], ['c2698', 0.2], ['c30750', 0.2], ['c470', 0.2]]
    - crossed by <tom wanted>: post 0.69 ctx 1.30 [share -1.68 win 3.11 cc 0.12 pv -0.25] cnt 0.00 g_win 1.44 g_nowin 0.00
    - crossed by <to help>: post 0.66 ctx 1.17 [share -0.94 win 2.06 cc 0.09 pv -0.04] cnt 0.00 g_win 1.12 g_nowin 0.00
- "to play" (7/15 bracketed)
  - ONE UNIT: tom and anna liked <to play> with water  | post 0.81 ctx 5.90 [share -3.44 win 9.32 cc -0.63 pv 0.65] cnt 0.00 g_win 4.34 g_nowin 0.00 | cc [['c16805', 0.2], ['c19147', 0.2], ['c4078', 0.2], ['c470', 0.2], ['c5118', 0.2]]
  - split   : they liked <to play> together  | post 0.43 ctx 1.12 [share -0.94 win 2.06 cc 0.04 pv -0.04] cnt 0.00 g_win 1.12 g_nowin 0.00 | cc [['c12608', 0.2], ['c4078', 0.2], ['c4079', 0.2], ['c470', 0.2], ['c9814', 0.2]]
    - crossed by <play together>: post 0.46 ctx 1.17 [share -0.94 win 2.06 cc 0.09 pv -0.04] cnt 0.00 g_win 1.12 g_nowin 0.00
- "they were" (10/12 bracketed)
  - ONE UNIT:  <they were> happy  | post 0.53 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c14428', 0.2], ['c2128', 0.2], ['c2698', 0.2], ['c470', 0.2], ['c8798', 0.2]]
  - split   : and <they were> always happy  | post 0.19 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c1762', 0.2], ['c32699', 0.2], ['c33358', 0.2], ['c4078', 0.2], ['c470', 0.2]]
    - crossed by <and they>: post 0.76 ctx 1.97 [share -1.68 win 3.78 cc 0.12 pv -0.25] cnt 0.00 g_win 2.11 g_nowin 0.00
- "tim was" (9/11 bracketed)
  - ONE UNIT:  <tim was> sad  | post 0.51 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c14428', 0.2], ['c2128', 0.2], ['c2698', 0.2], ['c470', 0.2], ['c8798', 0.2]]
  - split   :  <tim was> excited because he had many toys  | post 0.26 ctx 1.19 [share -1.68 win 3.00 cc 0.12 pv -0.25] cnt 0.00 g_win 1.33 g_nowin 0.00 | cc [['c2128', 0.2], ['c2339', 0.2], ['c34531', 0.2], ['c37575', 0.2], ['c470', 0.2]]
    - crossed by <was excited>: post 0.65 ctx 2.50 [share -1.68 win 4.59 cc -0.16 pv -0.25] cnt 0.00 g_win 2.91 g_nowin 0.00
- "were very" (3/10 bracketed)
  - ONE UNIT: tim and his mom <were very> surprised  | post 0.47 ctx 1.86 [share -1.66 win 3.96 cc -0.20 pv -0.25] cnt 0.00 g_win 2.30 g_nowin 0.00 | cc [['c12608', 0.2], ['c37484', 0.2], ['c4078', 0.2], ['c4079', 0.2], ['c470', 0.2]]
  - split   : they <were very> happy  | post 0.10 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c14428', 0.2], ['c2128', 0.2], ['c2698', 0.2], ['c32380', 0.2], ['c470', 0.2]]
    - crossed by <they were>: post 0.79 ctx 2.09 [share -1.68 win 3.90 cc 0.12 pv -0.25] cnt 0.00 g_win 2.23 g_nowin 0.00
    - crossed by <very happy>: post 0.67 ctx 1.12 [share -0.94 win 2.06 cc 0.03 pv -0.04] cnt 0.00 g_win 1.12 g_nowin 0.00
- "a big" (5/10 bracketed)
  - ONE UNIT:  <a big> bird came and took it away  | post 0.51 ctx 0.79 [share -1.68 win 2.60 cc 0.12 pv -0.25] cnt 0.00 g_win 0.92 g_nowin 0.00 | cc [['c2128', 0.2], ['c2339', 0.2], ['c34531', 0.2], ['c37575', 0.2], ['c470', 0.2]]
  - split   : at the park they saw <a big> tree  | post 0.16 ctx 0.14 [share -0.94 win 1.08 cc 0.04 pv -0.04] cnt 0.00 g_win 0.14 g_nowin 0.00 | cc [['c12608', 0.2], ['c18404', 0.2], ['c4078', 0.2], ['c4079', 0.2], ['c470', 0.2]]
    - crossed by <big tree>: post 0.66 ctx 1.91 [share -1.66 win 3.91 cc -0.09 pv -0.25] cnt 0.00 g_win 2.25 g_nowin 0.00
    - crossed by <they saw a>: post 0.30 ctx 0.86 [share -3.44 win 5.08 cc -1.43 pv 0.65] cnt 0.00 g_win 1.66 g_nowin 0.00
- "play with" (2/9 bracketed)
  - ONE UNIT: now i can <play with> the toy  | post 0.37 ctx 1.33 [share -2.28 win 3.89 cc -0.15 pv -0.12] cnt 0.00 g_win 1.61 g_nowin 0.00 | cc [['c11438', 0.6000000000000001], ['c4078', 0.2], ['c470', 0.2]]
  - split   : tom and anna liked to <play with> water  | post 0.05 ctx 1.88 [share -0.94 win 2.81 cc 0.04 pv -0.04] cnt 0.00 g_win 1.88 g_nowin 0.00 | cc [['c12608', 0.2], ['c25154', 0.2], ['c4078', 0.2], ['c4079', 0.2], ['c470', 0.2]]
    - crossed by <to play>: post 0.81 ctx 5.90 [share -3.44 win 9.32 cc -0.63 pv 0.65] cnt 0.00 g_win 4.34 g_nowin 0.00
    - crossed by <liked to play>: post 0.62 ctx 3.07 [share -3.44 win 7.29 cc -1.43 pv 0.65] cnt 0.00 g_win 4.12 g_nowin 0.00
- "to help" (5/8 bracketed)
  - ONE UNIT: mia was happy <to help> her mom  | post 0.44 ctx 2.00 [share -2.28 win 4.55 cc -0.15 pv -0.12] cnt 0.00 g_win 2.27 g_nowin 0.00 | cc [['c19147', 0.2], ['c25760', 0.2], ['c4078', 0.2], ['c470', 0.2], ['c5118', 0.2]]
  - split   : they wanted <to help> more  | post 0.37 ctx 1.88 [share -0.94 win 2.81 cc 0.04 pv -0.04] cnt 0.00 g_win 1.88 g_nowin 0.00 | cc [['c12608', 0.2], ['c37484', 0.2], ['c4078', 0.2], ['c4079', 0.2], ['c470', 0.2]]
    - crossed by <they wanted to>: post 0.42 ctx 1.22 [share -1.68 win 3.37 cc -0.22 pv -0.25] cnt 0.00 g_win 1.69 g_nowin 0.00

### 1000 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.293 [0.289 / 0.288 / 0.303] | 0.494 [0.486 / 0.485 / 0.510] | 0.368 [0.362 / 0.361 / 0.380] | 0.483 [0.478 / 0.483 / 0.487] | 0.690 [0.724 / 0.693 / 0.653] | 0.249 [0.245 / 0.243 / 0.257] | 0.324 [0.339 / 0.319 / 0.313] | 0.868 [0.857 / 0.882 / 0.864] | 0.742 [0.754 / 0.761 / 0.710] |
| Vwin | 3.303 [3.303 / 3.303 / 3.303] | 0.298 [0.304 / 0.295 / 0.297] | 0.503 [0.512 / 0.497 / 0.500] | 0.375 [0.381 / 0.370 / 0.372] | 0.471 [0.477 / 0.478 / 0.459] | 0.688 [0.740 / 0.669 / 0.653] | 0.265 [0.248 / 0.266 / 0.280] | 0.320 [0.321 / 0.337 / 0.302] | 0.845 [0.848 / 0.853 / 0.835] | 0.724 [0.737 / 0.729 / 0.706] |
| Vnowin | 3.303 [3.303 / 3.303 / 3.303] | 0.319 [0.319 / 0.319 / 0.319] | 0.537 [0.537 / 0.537 / 0.537] | 0.400 [0.400 / 0.400 / 0.400] | 0.362 [0.362 / 0.362 / 0.362] | 0.465 [0.465 / 0.465 / 0.465] | 0.329 [0.329 / 0.329 / 0.329] | 0.212 [0.212 / 0.212 / 0.212] | 0.873 [0.873 / 0.873 / 0.873] | 1.000 [1.000 / 1.000 / 1.000] |
| Vpmi | 3.303 [3.303 / 3.303 / 3.303] | 0.199 [0.149 / 0.117 / 0.331] | 0.336 [0.252 / 0.197 / 0.558] | 0.250 [0.188 / 0.147 / 0.415] | 0.331 [0.315 / 0.276 / 0.402] | 0.333 [0.299 / 0.213 / 0.488] | 0.421 [0.456 / 0.511 / 0.297] | 0.312 [0.402 / 0.409 / 0.125] | 0.824 [0.790 / 0.781 / 0.902] | 0.361 [0.430 / 0.412 / 0.241] |
| Vccm | 3.303 [3.303 / 3.303 / 3.303] | 0.221 [0.221 / 0.198 / 0.243] | 0.372 [0.372 / 0.333 / 0.410] | 0.277 [0.277 / 0.248 / 0.305] | 0.300 [0.289 / 0.264 / 0.347] | 0.320 [0.299 / 0.213 / 0.449] | 0.428 [0.428 / 0.453 / 0.402] | 0.251 [0.261 / 0.257 / 0.235] | 0.832 [0.846 / 0.815 / 0.833] | 0.476 [0.497 / 0.472 / 0.459] |
| Vzero | 3.303 [3.303 / 3.303 / 3.303] | 0.319 [0.319 / 0.319 / 0.319] | 0.537 [0.537 / 0.537 / 0.537] | 0.400 [0.400 / 0.400 / 0.400] | 0.362 [0.362 / 0.362 / 0.362] | 0.465 [0.465 / 0.465 / 0.465] | 0.329 [0.329 / 0.329 / 0.329] | 0.212 [0.212 / 0.212 / 0.212] | 0.873 [0.873 / 0.873 / 0.873] | 1.000 [1.000 / 1.000 / 1.000] |

diagnosis (V0, pooled over seeds): strings always bracketed 52, never 184, mixed 139; pairs of occurrences of one string with the SAME cc (content concept bands) 1588/3948 = 0.402

bracket rate by position: interior 157/1122 = 0.140; start 379/534 = 0.710

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.345 | 1.00 |
| ctx | +1.170 | 0.88 |
| cnt | +0.000 | 0.00 |
| t_share | -0.563 | 0.18 |
| t_win | +1.683 | 0.88 |
| t_cc | +0.012 | 0.45 |
| t_pv | +0.037 | 0.29 |
| g_win | +1.059 | 0.86 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.751, between-string sd 0.746
- t_win: within-string sd 1.310, between-string sd 1.164
- t_cc: within-string sd 0.163, between-string sd 0.225
- g_win: within-string sd 0.716, between-string sd 0.723
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "to play" (7/15 bracketed)
  - ONE UNIT: tom and anna liked <to play> with water  | post 0.83 ctx 5.18 [share -3.60 win 8.23 cc -0.52 pv 1.07] cnt 0.00 g_win 3.47 g_nowin 0.00 | cc [['c16805', 0.2], ['c4078', 0.2], ['c470', 0.2], ['c5079', 0.2], ['c5118', 0.2]]
  - split   : they liked <to play> together  | post 0.46 ctx 1.69 [share -0.94 win 2.50 cc 0.11 pv 0.02] cnt 0.00 g_win 1.56 g_nowin 0.00 | cc [['c116647', 0.2], ['c12608', 0.2], ['c4078', 0.2], ['c4079', 0.2], ['c470', 0.2]]
    - crossed by <play together>: post 0.47 ctx 1.59 [share -0.94 win 2.36 cc 0.15 pv 0.02] cnt 0.00 g_win 1.42 g_nowin 0.00
- "they were" (10/12 bracketed)
  - ONE UNIT:  <they were> happy  | post 0.52 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.07 g_nowin 0.00 | cc [['c14428', 0.2], ['c159726', 0.2], ['c2128', 0.2], ['c2698', 0.2], ['c470', 0.2]]
  - split   : and <they were> always happy  | post 0.12 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c148522', 0.2], ['c1762', 0.2], ['c2631', 0.2], ['c4078', 0.2], ['c470', 0.2]]
    - crossed by <and they>: post 0.86 ctx 2.82 [share -1.65 win 4.39 cc 0.19 pv -0.12] cnt 0.00 g_win 2.74 g_nowin 0.00
- "tim was" (9/11 bracketed)
  - ONE UNIT:  <tim was> sad  | post 0.56 ctx 0.14 [share -2.65 win 3.46 cc -0.42 pv -0.24] cnt 0.00 g_win 0.81 g_nowin 0.00 | cc [['c14428', 0.2], ['c2128', 0.2], ['c2698', 0.2], ['c39115', 0.2], ['c470', 0.2]]
  - split   :  <tim was> excited because he had many toys  | post 0.14 ctx 0.63 [share -1.65 win 2.07 cc 0.33 pv -0.12] cnt 0.00 g_win 0.42 g_nowin 0.00 | cc [['c115571', 0.3333333333333333], ['c116605', 0.16666666666666666], ['c2128', 0.16666666666666666], ['c2339', 0.16666666666666666], ['c470', 0.16666666666666666]]
    - crossed by <was excited>: post 0.80 ctx 2.87 [share -1.65 win 4.66 cc -0.03 pv -0.12] cnt 0.00 g_win 3.06 g_nowin 0.00
- "a big" (5/10 bracketed)
  - ONE UNIT:  <a big> bird came and took it away  | post 0.76 ctx 2.13 [share -1.65 win 3.62 cc 0.27 pv -0.12] cnt 0.00 g_win 1.97 g_nowin 0.00 | cc [['c159835', 0.2], ['c161226', 0.2], ['c2128', 0.2], ['c2339', 0.2], ['c470', 0.2]]
  - split   : at the park they saw <a big> tree  | post 0.19 ctx 1.80 [share -0.94 win 2.57 cc 0.15 pv 0.02] cnt 0.00 g_win 1.63 g_nowin 0.00 | cc [['c111317', 0.16666666666666666], ['c12608', 0.16666666666666666], ['c147303', 0.16666666666666666], ['c4078', 0.16666666666666666], ['c4079', 0.16666666666666666], ['c470', 0.16666666666666666]]
    - crossed by <big tree>: post 0.53 ctx 2.03 [share -1.59 win 3.48 cc 0.24 pv -0.11] cnt 0.00 g_win 1.90 g_nowin 0.00
    - crossed by <they saw a>: post 0.61 ctx 3.73 [share -2.40 win 6.75 cc -0.69 pv 0.07] cnt 0.00 g_win 4.35 g_nowin 0.00
- "to help" (4/8 bracketed)
  - ONE UNIT: they all wanted <to help> her feel happy  | post 0.60 ctx 2.96 [share -3.60 win 6.01 cc -0.52 pv 1.07] cnt 0.00 g_win 2.46 g_nowin 0.00 | cc [['c4078', 0.2], ['c470', 0.2], ['c5079', 0.2], ['c5118', 0.2], ['c80868', 0.2]]
  - split   : they wanted <to help> more  | post 0.41 ctx 1.64 [share -0.94 win 2.40 cc 0.15 pv 0.02] cnt 0.00 g_win 1.46 g_nowin 0.00 | cc [['c116647', 0.16666666666666666], ['c126067', 0.16666666666666666], ['c12608', 0.16666666666666666], ['c4078', 0.16666666666666666], ['c4079', 0.16666666666666666], ['c470', 0.16666666666666666]]
    - crossed by <help more>: post 0.47 ctx 1.59 [share -0.94 win 2.36 cc 0.15 pv 0.02] cnt 0.00 g_win 1.42 g_nowin 0.00
- "he was" (7/8 bracketed)
  - ONE UNIT:  <he was> so happy  | post 0.88 ctx 2.96 [share -1.65 win 4.46 cc 0.27 pv -0.12] cnt 0.00 g_win 2.81 g_nowin 0.00 | cc [['c2128', 0.2], ['c2339', 0.2], ['c40139', 0.4], ['c470', 0.2]]
  - split   :  <he was> scared and excited  | post 0.22 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c2128', 0.2], ['c2339', 0.2], ['c40139', 0.4], ['c470', 0.2]]
    - crossed by <was scared>: post 0.68 ctx 2.30 [share -1.65 win 4.09 cc -0.03 pv -0.12] cnt 0.00 g_win 2.44 g_nowin 0.00
- "his mom" (4/7 bracketed)
  - ONE UNIT:  <his mom> said no tim  | post 0.83 ctx 2.30 [share -1.65 win 3.79 cc 0.27 pv -0.12] cnt 0.00 g_win 2.14 g_nowin 0.00 | cc [['c148918', 0.2], ['c159835', 0.2], ['c2128', 0.2], ['c2339', 0.2], ['c470', 0.2]]
  - split   : tim and <his mom> were very surprised  | post 0.17 ctx 0.71 [share -2.40 win 3.09 cc -0.05 pv 0.07] cnt 0.00 g_win 2.79 g_nowin 0.00 | cc [['c4078', 0.2], ['c470', 0.2], ['c5079', 0.2], ['c84020', 0.4]]
    - crossed by <and his>: post 0.65 ctx 2.17 [share -1.65 win 3.96 cc -0.03 pv -0.12] cnt 0.00 g_win 2.31 g_nowin 0.00
- "to be" (2/6 bracketed)
  - ONE UNIT: it wanted <to be> with its friend again  | post 0.55 ctx 3.12 [share -2.40 win 5.50 cc -0.05 pv 0.07] cnt 0.00 g_win 3.10 g_nowin 0.00 | cc [['c4078', 0.2], ['c470', 0.2], ['c5079', 0.2], ['c5118', 0.2], ['c80868', 0.2]]
  - split   : they wanted <to be> pretty like her  | post 0.12 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c4078', 0.2], ['c470', 0.2], ['c5079', 0.2], ['c5118', 0.2], ['c80868', 0.2]]
    - crossed by <be pretty like her>: post 0.42 ctx 1.32 [share -0.94 win 2.36 cc -0.12 pv 0.02] cnt 0.00 g_win 1.42 g_nowin 0.00
    - crossed by <they wanted to>: post 0.39 ctx 1.51 [share -1.65 win 3.31 cc -0.04 pv -0.12] cnt 0.00 g_win 1.66 g_nowin 0.00
