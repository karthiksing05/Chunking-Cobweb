seeds [13, 17, 19]; held-out 300; strings 125

### 250 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.233 [0.237 / 0.233 / 0.230] | 0.393 [0.400 / 0.393 / 0.388] | 0.293 [0.298 / 0.293 / 0.289] | 0.340 [0.337 / 0.341 / 0.341] | 0.493 [0.504 / 0.496 / 0.480] | 0.403 [0.414 / 0.406 / 0.390] | 0.452 [0.449 / 0.446 / 0.460] | 0.827 [0.828 / 0.835 / 0.819] | 0.480 [0.474 / 0.484 / 0.483] |

diagnosis (V0, pooled over seeds): strings always bracketed 68, never 134, mixed 173; pairs of occurrences of one string with the SAME cc (content concept bands) 2711/3948 = 0.687

bracket rate by position: interior 514/1122 = 0.458; start 234/534 = 0.438

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.517 | 0.99 |
| ctx | +0.402 | 0.64 |
| cnt | +0.000 | 0.00 |
| coh | -0.000 | 0.03 |
| t_share | -0.333 | 0.28 |
| t_win | +0.831 | 0.65 |
| t_cc | -0.113 | 0.29 |
| t_pv | +0.017 | 0.34 |
| g_win | +0.462 | 0.64 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.636, between-string sd 0.624
- t_win: within-string sd 1.354, between-string sd 1.241
- t_cc: within-string sd 0.196, between-string sd 0.200
- g_win: within-string sd 0.720, between-string sd 0.677
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000
- coh: within-string sd 0.003, between-string sd 2.765

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "wanted to" (12/18 bracketed)
  - ONE UNIT: tom <wanted to> help  | post 0.69 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c21477', 0.2], ['c25698', 0.2], ['c909', 0.6000000000000001]]
  - split   : it <wanted to> be with its friend again  | post 0.25 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c22237', 0.2], ['c25698', 0.2], ['c909', 0.6000000000000001]]
    - crossed by <to be>: post 0.74 ctx 1.84 [share -2.28 win 4.45 cc -0.18 pv -0.15] cnt 0.00 g_win 2.17 g_nowin 0.00
- "to play" (12/15 bracketed)
  - ONE UNIT: they liked <to play> together  | post 0.97 ctx 1.29 [share -0.98 win 2.21 cc 0.07 pv -0.02] cnt 0.00 g_win 1.24 g_nowin 0.00 | cc [['c25698', 0.25], ['c4947', 0.5], ['c4948', 0.25]]
  - split   : she wanted <to play> with her friend tim  | post 0.23 ctx 2.15 [share -2.28 win 4.62 cc -0.04 pv -0.15] cnt 0.00 g_win 2.35 g_nowin 0.00 | cc [['c25698', 0.2], ['c26518', 0.2], ['c4947', 0.4], ['c4948', 0.2]]
    - crossed by <play with her friend>: post 0.23 ctx 0.70 [share -0.98 win 1.55 cc 0.15 pv -0.02] cnt 0.00 g_win 0.60 g_nowin 0.00
    - crossed by <she wanted to>: post 0.53 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "they were" (4/12 bracketed)
  - ONE UNIT:  <they were> all excited  | post 0.92 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c2615', 0.2], ['c27591', 0.4], ['c29149', 0.2], ['c4', 0.2]]
  - split   :  <they were> happy  | post 0.41 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c2615', 0.2], ['c27591', 0.4], ['c29149', 0.2], ['c4', 0.2]]
    - crossed by <were happy>: post 0.58 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "tim was" (6/11 bracketed)
  - ONE UNIT:  <tim was> sad  | post 0.51 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c12440', 0.4], ['c27591', 0.4], ['c4042', 0.2]]
  - split   :  <tim was> very happy  | post 0.31 ctx 0.98 [share -1.67 win 2.91 cc 0.02 pv -0.28] cnt 0.00 g_win 1.25 g_nowin 0.00 | cc [['c12440', 0.4], ['c27591', 0.4], ['c4042', 0.2]]
    - crossed by <was very happy>: post 0.66 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "a big" (8/10 bracketed)
  - ONE UNIT: he saw tom near <a big> tree  | post 0.94 ctx 0.51 [share -1.73 win 2.91 cc -0.48 pv -0.20] cnt 0.00 g_win 1.18 g_nowin 0.00 | cc [['c15824', 0.2], ['c25698', 0.2], ['c2588', 0.4], ['c26377', 0.2]]
  - split   : at the park they saw <a big> tree  | post 0.16 ctx 0.25 [share -0.98 win 1.12 cc 0.12 pv -0.02] cnt 0.00 g_win 0.15 g_nowin 0.00 | cc [['c15824', 0.2], ['c25698', 0.2], ['c2588', 0.4], ['c26377', 0.2]]
    - crossed by <they saw a>: post 0.82 ctx 1.68 [share -2.28 win 4.17 cc -0.06 pv -0.15] cnt 0.00 g_win 2.16 g_nowin 0.00
- "was so" (5/10 bracketed)
  - ONE UNIT: sue <was so> surprised  | post 0.86 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c13150', 0.3333333333333333], ['c22430', 0.16666666666666666], ['c9929', 0.5]]
  - split   : he <was so> happy  | post 0.09 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c13150', 0.3333333333333333], ['c22430', 0.16666666666666666], ['c9929', 0.5]]
    - crossed by <he was>: post 0.78 ctx 1.24 [share -1.67 win 3.21 cc -0.02 pv -0.28] cnt 0.00 g_win 1.54 g_nowin 0.00
    - crossed by <so happy>: post 0.75 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.88 g_nowin 0.00
- "play with" (7/9 bracketed)
  - ONE UNIT: now i can <play with> the toy  | post 0.99 ctx 1.56 [share -2.28 win 4.17 cc -0.19 pv -0.15] cnt 0.00 g_win 1.90 g_nowin 0.00 | cc [['c12251', 0.2], ['c25698', 0.2], ['c26377', 0.2], ['c6607', 0.4]]
  - split   : tom and anna liked to <play with> water  | post 0.03 ctx 1.98 [share -0.98 win 2.86 cc 0.12 pv -0.02] cnt 0.00 g_win 1.88 g_nowin 0.00 | cc [['c12251', 0.2], ['c25698', 0.2], ['c26377', 0.2], ['c6607', 0.4]]
    - crossed by <to play>: post 0.97 ctx 6.17 [share -3.47 win 9.42 cc -0.41 pv 0.64] cnt 0.00 g_win 4.44 g_nowin 0.00
    - crossed by <liked to play>: post 0.69 ctx 3.95 [share -2.28 win 6.41 cc -0.03 pv -0.15] cnt 0.00 g_win 4.13 g_nowin 0.00
- "was very" (1/9 bracketed)
  - ONE UNIT: tim <was very> sad  | post 0.65 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c21685', 0.4], ['c9929', 0.6000000000000001]]
  - split   : tim <was very> happy  | post 0.09 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c21685', 0.4], ['c9929', 0.6000000000000001]]
    - crossed by <very happy>: post 0.89 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.88 g_nowin 0.00

### 1000 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.249 [0.250 / 0.254 / 0.241] | 0.419 [0.422 / 0.429 / 0.406] | 0.312 [0.314 / 0.319 / 0.303] | 0.367 [0.377 / 0.364 / 0.359] | 0.564 [0.512 / 0.653 / 0.528] | 0.377 [0.368 / 0.388 / 0.375] | 0.469 [0.469 / 0.464 / 0.475] | 0.818 [0.804 / 0.819 / 0.832] | 0.550 [0.547 / 0.537 / 0.565] |

diagnosis (V0, pooled over seeds): strings always bracketed 77, never 120, mixed 178; pairs of occurrences of one string with the SAME cc (content concept bands) 2576/3948 = 0.652

bracket rate by position: interior 523/1122 = 0.466; start 254/534 = 0.476

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.549 | 1.00 |
| ctx | +0.375 | 0.66 |
| cnt | +0.000 | 0.00 |
| coh | -0.000 | 0.02 |
| t_share | -0.310 | 0.30 |
| t_win | +0.680 | 0.67 |
| t_cc | -0.024 | 0.39 |
| t_pv | +0.029 | 0.40 |
| g_win | +0.344 | 0.63 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.691, between-string sd 0.795
- t_win: within-string sd 1.313, between-string sd 1.326
- t_cc: within-string sd 0.180, between-string sd 0.183
- g_win: within-string sd 0.675, between-string sd 0.748
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000
- coh: within-string sd 0.000, between-string sd 2.610

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "wanted to" (10/18 bracketed)
  - ONE UNIT: tom <wanted to> help  | post 0.69 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c25698', 0.2], ['c51421', 0.2], ['c909', 0.4], ['c9970', 0.2]]
  - split   : it <wanted to> be with its friend again  | post 0.21 ctx 0.45 [share -1.66 win 2.13 cc 0.14 pv -0.15] cnt 0.00 g_win 0.47 g_nowin 0.00 | cc [['c25698', 0.2], ['c51421', 0.2], ['c909', 0.4], ['c9970', 0.2]]
    - crossed by <to be>: post 0.78 ctx 3.36 [share -2.38 win 5.89 cc -0.20 pv 0.06] cnt 0.00 g_win 3.51 g_nowin 0.00
- "to play" (12/15 bracketed)
  - ONE UNIT: they liked <to play> together  | post 0.98 ctx 1.70 [share -1.00 win 2.44 cc 0.22 pv 0.05] cnt 0.00 g_win 1.44 g_nowin 0.00 | cc [['c100337', 0.2], ['c25698', 0.2], ['c4947', 0.4], ['c4948', 0.2]]
  - split   : the cat said i want <to play> too  | post 0.39 ctx 1.50 [share -1.00 win 2.23 cc 0.22 pv 0.05] cnt 0.00 g_win 1.24 g_nowin 0.00 | cc [['c100337', 0.2], ['c25698', 0.2], ['c4947', 0.4], ['c4948', 0.2]]
    - crossed by <i want to>: post 0.59 ctx 3.26 [share -2.38 win 5.43 cc 0.15 pv 0.06] cnt 0.00 g_win 3.05 g_nowin 0.00
- "they were" (4/12 bracketed)
  - ONE UNIT:  <they were> happy  | post 0.55 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c2615', 0.2], ['c4', 0.6000000000000001], ['c98695', 0.2]]
  - split   :  <they were> very happy  | post 0.23 ctx 1.44 [share -1.66 win 3.17 cc 0.08 pv -0.15] cnt 0.00 g_win 1.51 g_nowin 0.00 | cc [['c2615', 0.2], ['c4', 0.6000000000000001], ['c98695', 0.2]]
    - crossed by <were very>: post 0.69 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "tim was" (5/11 bracketed)
  - ONE UNIT:  <tim was> sad  | post 0.62 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c12440', 0.4], ['c4042', 0.2], ['c59223', 0.4]]
  - split   :  <tim was> so surprised and happy  | post 0.09 ctx 0.95 [share -1.66 win 3.00 cc -0.23 pv -0.15] cnt 0.00 g_win 1.34 g_nowin 0.00 | cc [['c12440', 0.4], ['c4042', 0.2], ['c59223', 0.4]]
    - crossed by <was so>: post 0.91 ctx 1.25 [share -1.66 win 2.83 cc 0.24 pv -0.15] cnt 0.00 g_win 1.17 g_nowin 0.00
- "were very" (9/10 bracketed)
  - ONE UNIT: tim and his mom <were very> surprised  | post 0.50 ctx 2.02 [share -1.69 win 3.77 cc 0.01 pv -0.07] cnt 0.00 g_win 2.08 g_nowin 0.00 | cc [['c109601', 0.2], ['c11052', 0.4], ['c25698', 0.2], ['c44310', 0.2]]
  - split   : tom and sue <were very> surprised  | post 0.49 ctx 0.80 [share -2.90 win 3.67 cc 0.16 pv -0.14] cnt 0.00 g_win 0.78 g_nowin 0.00 | cc [['c109601', 0.2], ['c11052', 0.4], ['c25698', 0.2], ['c44310', 0.2]]
    - crossed by <very surprised>: post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 1.07 g_nowin 0.00
    - crossed by <and sue were>: post 0.50 ctx 2.12 [share -1.66 win 3.65 cc 0.28 pv -0.15] cnt 0.00 g_win 1.99 g_nowin 0.00
- "a big" (8/10 bracketed)
  - ONE UNIT: he saw tom near <a big> tree  | post 0.99 ctx 2.65 [share -2.90 win 5.90 cc -0.22 pv -0.14] cnt 0.00 g_win 3.01 g_nowin 0.00 | cc [['c25698', 0.2], ['c2588', 0.4], ['c44310', 0.2], ['c95376', 0.2]]
  - split   : at the park they saw <a big> tree  | post 0.37 ctx 1.81 [share -1.00 win 2.54 cc 0.22 pv 0.05] cnt 0.00 g_win 1.54 g_nowin 0.00 | cc [['c25698', 0.2], ['c2588', 0.4], ['c44310', 0.2], ['c95376', 0.2]]
    - crossed by <they saw a>: post 0.62 ctx 4.01 [share -2.38 win 6.37 cc -0.04 pv 0.06] cnt 0.00 g_win 4.00 g_nowin 0.00
- "was so" (5/10 bracketed)
  - ONE UNIT: sue <was so> surprised  | post 0.79 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c105415', 0.16666666666666666], ['c13150', 0.16666666666666666], ['c76061', 0.16666666666666666], ['c93264', 0.16666666666666666], ['c9929', 0.3333333333333333]]
  - split   : he <was so> happy  | post 0.10 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c105415', 0.16666666666666666], ['c13150', 0.16666666666666666], ['c76061', 0.16666666666666666], ['c93264', 0.16666666666666666], ['c9929', 0.3333333333333333]]
    - crossed by <he was>: post 0.87 ctx 1.50 [share -1.66 win 3.38 cc -0.07 pv -0.15] cnt 0.00 g_win 1.72 g_nowin 0.00
    - crossed by <so happy>: post 0.64 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 1.07 g_nowin 0.00
- "play with" (5/9 bracketed)
  - ONE UNIT: now i can <play with> the toy  | post 0.96 ctx 2.43 [share -2.38 win 4.90 cc -0.14 pv 0.06] cnt 0.00 g_win 3.39 g_nowin 0.00 | cc [['c133178', 0.16666666666666666], ['c25698', 0.16666666666666666], ['c30464', 0.16666666666666666], ['c44310', 0.16666666666666666], ['c6607', 0.3333333333333333]]
  - split   : tom and anna liked to <play with> water  | post 0.08 ctx 2.12 [share -1.00 win 2.83 cc 0.24 pv 0.05] cnt 0.00 g_win 1.83 g_nowin 0.00 | cc [['c131202', 0.16666666666666666], ['c132355', 0.16666666666666666], ['c25698', 0.16666666666666666], ['c44310', 0.16666666666666666], ['c6607', 0.3333333333333333]]
    - crossed by <to play>: post 0.92 ctx 5.14 [share -3.59 win 8.18 cc -0.53 pv 1.07] cnt 0.00 g_win 3.39 g_nowin 0.00
