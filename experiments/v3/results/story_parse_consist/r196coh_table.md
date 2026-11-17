seeds [13, 17, 19]; held-out 300; strings 125

### 250 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.230 [0.229 / 0.221 / 0.239] | 0.387 [0.386 / 0.372 / 0.403] | 0.288 [0.287 / 0.277 / 0.300] | 0.380 [0.386 / 0.382 / 0.372] | 0.483 [0.512 / 0.449 / 0.488] | 0.368 [0.369 / 0.368 / 0.366] | 0.423 [0.431 / 0.393 / 0.444] | 0.866 [0.893 / 0.859 / 0.846] | 0.476 [0.475 / 0.496 / 0.458] |
| Vc:none | 3.303 [3.303 / 3.303 / 3.303] | 0.283 [0.297 / 0.287 / 0.265] | 0.477 [0.500 / 0.483 / 0.447] | 0.355 [0.372 / 0.360 / 0.333] | 0.438 [0.450 / 0.431 / 0.432] | 0.591 [0.638 / 0.606 / 0.528] | 0.313 [0.291 / 0.333 / 0.316] | 0.322 [0.332 / 0.306 / 0.330] | 0.833 [0.826 / 0.813 / 0.861] | 0.603 [0.604 / 0.604 / 0.602] |
| Vc:coh_only | 3.303 [3.303 / 3.303 / 3.303] | 0.202 [0.203 / 0.212 / 0.193] | 0.341 [0.342 / 0.357 / 0.325] | 0.254 [0.255 / 0.266 / 0.242] | 0.332 [0.336 / 0.358 / 0.301] | 0.375 [0.425 / 0.354 / 0.346] | 0.424 [0.434 / 0.403 / 0.435] | 0.434 [0.413 / 0.417 / 0.471] | 0.870 [0.900 / 0.875 / 0.833] | 0.387 [0.393 / 0.402 / 0.368] |

diagnosis (V0, pooled over seeds): strings always bracketed 75, never 157, mixed 143; pairs of occurrences of one string with the SAME cc (content concept bands) 2707/3948 = 0.686

bracket rate by position: interior 386/1122 = 0.344; start 314/534 = 0.588

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.435 | 1.00 |
| ctx | +0.517 | 0.71 |
| cnt | +0.000 | 0.00 |
| coh | +0.029 | 0.03 |
| t_share | -0.270 | 0.31 |
| t_win | +0.838 | 0.65 |
| t_cc | -0.053 | 0.41 |
| t_pv | +0.002 | 0.38 |
| g_win | +0.552 | 0.70 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.699, between-string sd 0.648
- t_win: within-string sd 1.319, between-string sd 1.192
- t_cc: within-string sd 0.182, between-string sd 0.178
- g_win: within-string sd 0.765, between-string sd 0.674
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000
- coh: within-string sd 0.189, between-string sd 1.878

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "wanted to" (16/18 bracketed)
  - ONE UNIT: tom <wanted to> help  | post 0.71 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c1110', 0.4], ['c1386', 0.2], ['c16257', 0.2], ['c27037', 0.2]]
  - split   : he <wanted to> go outside and play  | post 0.34 ctx 0.32 [share -1.69 win 2.35 cc -0.11 pv -0.23] cnt 0.00 g_win 0.66 g_nowin 0.00 | cc [['c1110', 0.4], ['c1386', 0.2], ['c16257', 0.2], ['c30606', 0.2]]
    - crossed by <to go>: post 0.60 ctx 1.97 [share -2.30 win 4.41 cc 0.03 pv -0.16] cnt 0.00 g_win 2.10 g_nowin 0.00
- "to play" (1/15 bracketed)
  - ONE UNIT: the cat said i want <to play> too  | post 0.53 ctx 1.47 [share -0.90 win 2.45 cc -0.04 pv -0.04] cnt 0.00 g_win 1.55 g_nowin 0.00 | cc [['c1395', 0.25], ['c3474', 0.5], ['c6276', 0.25]]
  - split   : they liked <to play> together  | post 0.03 ctx 1.06 [share -0.90 win 2.04 cc -0.04 pv -0.04] cnt 0.00 g_win 1.14 g_nowin 0.00 | cc [['c1395', 0.25], ['c3474', 0.5], ['c6276', 0.25]]
    - crossed by <they liked to>: post 0.58 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
    - crossed by <liked to>: post 0.64 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.09 g_nowin 0.00
- "were very" (4/10 bracketed)
  - ONE UNIT: tim and his mom <were very> surprised  | post 0.63 ctx 2.26 [share -1.58 win 3.92 cc 0.14 pv -0.22] cnt 0.00 g_win 2.34 g_nowin 0.00 | cc [['c12540', 0.2], ['c1395', 0.2], ['c17636', 0.4], ['c6276', 0.2]]
  - split   : they <were very> happy  | post 0.02 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c10741', 0.16666666666666666], ['c15222', 0.3333333333333333], ['c15409', 0.16666666666666666], ['c1816', 0.16666666666666666], ['c6276', 0.16666666666666666]]
    - crossed by <they were>: post 0.97 ctx 2.24 [share -1.69 win 4.02 cc 0.14 pv -0.23] cnt 0.00 g_win 2.33 g_nowin 0.00
    - crossed by <very happy>: post 0.94 ctx 0.91 [share -0.90 win 2.04 cc -0.18 pv -0.04] cnt 0.00 g_win 1.14 g_nowin 0.00
- "was very" (1/9 bracketed)
  - ONE UNIT: the little boy <was very> happy  | post 0.63 ctx 1.77 [share -1.58 win 3.45 cc 0.13 pv -0.22] cnt 0.00 g_win 1.86 g_nowin 0.00 | cc [['c1395', 0.2], ['c16488', 0.2], ['c17636', 0.4], ['c6276', 0.2]]
  - split   : tim <was very> happy  | post 0.03 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c10741', 0.16666666666666666], ['c15222', 0.3333333333333333], ['c15409', 0.16666666666666666], ['c1816', 0.16666666666666666], ['c6276', 0.16666666666666666]]
    - crossed by <tim was>: post 0.97 ctx 2.21 [share -1.69 win 4.02 cc 0.12 pv -0.23] cnt 0.00 g_win 2.33 g_nowin 0.00
    - crossed by <very happy>: post 0.89 ctx 0.91 [share -0.90 win 2.04 cc -0.18 pv -0.04] cnt 0.00 g_win 1.14 g_nowin 0.00
- "to help" (2/8 bracketed)
  - ONE UNIT: the people came <to help> the car  | post 0.77 ctx 5.34 [share -3.50 win 8.57 cc -0.36 pv 0.63] cnt 0.00 g_win 4.23 g_nowin 0.00 | cc [['c1110', 0.3333333333333333], ['c15880', 0.16666666666666666], ['c30608', 0.16666666666666666], ['c3922', 0.3333333333333333]]
  - split   : they wanted <to help> more  | post 0.06 ctx 1.75 [share -0.90 win 2.73 cc -0.04 pv -0.04] cnt 0.00 g_win 1.83 g_nowin 0.00 | cc [['c1395', 0.25], ['c3474', 0.5], ['c6276', 0.25]]
    - crossed by <wanted to>: post 0.93 ctx 1.82 [share -1.69 win 3.86 cc -0.11 pv -0.23] cnt 0.00 g_win 2.17 g_nowin 0.00
    - crossed by <they wanted to>: post 0.83 ctx 1.63 [share -1.69 win 3.53 cc 0.02 pv -0.23] cnt 0.00 g_win 1.84 g_nowin 0.00
- "he was" (7/8 bracketed)
  - ONE UNIT:  <he was> so happy  | post 0.95 ctx 2.52 [share -1.69 win 4.28 cc 0.16 pv -0.23] cnt 0.00 g_win 2.59 g_nowin 0.00 | cc [['c22756', 0.2], ['c23072', 0.2], ['c3082', 0.4], ['c7606', 0.2]]
  - split   :  <he was> happy again  | post 0.32 ctx 0.49 [share -1.69 win 2.25 cc 0.16 pv -0.23] cnt 0.00 g_win 0.56 g_nowin 0.00 | cc [['c22756', 0.2], ['c23072', 0.2], ['c3082', 0.4], ['c7606', 0.2]]
    - crossed by <was happy>: post 0.64 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
    - crossed by <was happy again>: post 0.54 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "to be" (3/6 bracketed)
  - ONE UNIT: you have <to be> nice to me  | post 0.50 ctx 0.40 [share -0.90 win 1.54 cc -0.20 pv -0.04] cnt 0.00 g_win 0.64 g_nowin 0.00 | cc [['c1110', 0.4], ['c24064', 0.2], ['c3922', 0.4]]
  - split   : it wanted <to be> with its friend again  | post 0.42 ctx 1.69 [share -2.30 win 4.12 cc 0.03 pv -0.16] cnt 0.00 g_win 1.82 g_nowin 0.00 | cc [['c1110', 0.4], ['c24064', 0.2], ['c3922', 0.4]]
    - crossed by <it wanted to>: post 0.29 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
    - crossed by <wanted to>: post 0.49 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "liked to" (3/6 bracketed)
  - ONE UNIT: they <liked to> play together  | post 0.64 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.09 g_nowin 0.00 | cc [['c1110', 0.4], ['c1386', 0.2], ['c28858', 0.2], ['c7521', 0.2]]
  - split   : they <liked to> play in the park  | post 0.23 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.09 g_nowin 0.00 | cc [['c1110', 0.4], ['c1386', 0.2], ['c28858', 0.2], ['c7521', 0.2]]
    - crossed by <to play in the park>: post 0.69 ctx 1.13 [share -0.90 win 2.04 cc 0.04 pv -0.04] cnt 0.00 g_win 1.14 g_nowin 0.00
    - crossed by <they liked>: post 0.63 ctx 0.99 [share -1.69 win 2.98 cc -0.06 pv -0.23] cnt 0.00 g_win 1.29 g_nowin 0.00

### 1000 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.251 [0.284 / 0.258 / 0.211] | 0.423 [0.478 / 0.435 / 0.355] | 0.315 [0.356 / 0.324 / 0.265] | 0.414 [0.456 / 0.415 / 0.370] | 0.551 [0.598 / 0.622 / 0.433] | 0.335 [0.279 / 0.340 / 0.386] | 0.442 [0.435 / 0.411 / 0.480] | 0.879 [0.888 / 0.880 / 0.868] | 0.567 [0.595 / 0.588 / 0.519] |
| Vc:none | 3.303 [3.303 / 3.303 / 3.303] | 0.285 [0.304 / 0.288 / 0.263] | 0.480 [0.512 / 0.485 / 0.444] | 0.358 [0.381 / 0.361 / 0.331] | 0.464 [0.479 / 0.460 / 0.452] | 0.688 [0.748 / 0.740 / 0.575] | 0.274 [0.244 / 0.284 / 0.294] | 0.336 [0.330 / 0.326 / 0.353] | 0.853 [0.850 / 0.851 / 0.859] | 0.728 [0.745 / 0.738 / 0.702] |
| Vc:coh_only | 3.303 [3.303 / 3.303 / 3.303] | 0.208 [0.245 / 0.213 / 0.166] | 0.350 [0.413 / 0.359 / 0.279] | 0.261 [0.308 / 0.267 / 0.208] | 0.346 [0.396 / 0.354 / 0.289] | 0.433 [0.496 / 0.488 / 0.315] | 0.414 [0.354 / 0.410 / 0.478] | 0.457 [0.446 / 0.409 / 0.514] | 0.862 [0.861 / 0.879 / 0.848] | 0.453 [0.477 / 0.471 / 0.411] |

diagnosis (V0, pooled over seeds): strings always bracketed 95, never 152, mixed 128; pairs of occurrences of one string with the SAME cc (content concept bands) 2647/3948 = 0.670

bracket rate by position: interior 409/1122 = 0.365; start 323/534 = 0.605

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.493 | 1.00 |
| ctx | +0.683 | 0.80 |
| cnt | +0.000 | 0.00 |
| coh | +0.059 | 0.08 |
| t_share | -0.390 | 0.28 |
| t_win | +1.056 | 0.75 |
| t_cc | +0.003 | 0.55 |
| t_pv | +0.014 | 0.36 |
| g_win | +0.616 | 0.73 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.782, between-string sd 0.802
- t_win: within-string sd 1.316, between-string sd 1.204
- t_cc: within-string sd 0.190, between-string sd 0.205
- g_win: within-string sd 0.741, between-string sd 0.738
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000
- coh: within-string sd 0.185, between-string sd 1.968

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "wanted to" (6/18 bracketed)
  - ONE UNIT: they <wanted to> help more  | post 0.62 ctx 1.22 [share -1.64 win 3.07 cc -0.10 pv -0.11] cnt 0.00 g_win 1.43 g_nowin 0.00 | cc [['c1110', 0.3333333333333333], ['c1386', 0.3333333333333333], ['c30606', 0.16666666666666666], ['c85707', 0.16666666666666666]]
  - split   : tom <wanted to> help  | post 0.10 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c1110', 0.4], ['c1386', 0.4], ['c27037', 0.2]]
    - crossed by <tom wanted>: post 0.88 ctx 2.03 [share -1.64 win 3.49 cc 0.30 pv -0.11] cnt 0.00 g_win 1.84 g_nowin 0.00
    - crossed by <to help>: post 0.89 ctx 1.65 [share -0.90 win 2.32 cc 0.20 pv 0.02] cnt 0.00 g_win 1.42 g_nowin 0.00
- "to play" (8/15 bracketed)
  - ONE UNIT: they liked <to play> together  | post 0.87 ctx 1.70 [share -0.90 win 2.49 cc 0.08 pv 0.02] cnt 0.00 g_win 1.60 g_nowin 0.00 | cc [['c122303', 0.2], ['c1395', 0.2], ['c3474', 0.4], ['c6276', 0.2]]
  - split   : tim loved <to play> outside with his friends  | post 0.42 ctx 3.46 [share -3.60 win 6.34 cc -0.35 pv 1.07] cnt 0.00 g_win 2.40 g_nowin 0.00 | cc [['c1110', 0.4], ['c3922', 0.4], ['c7523', 0.2]]
    - crossed by <play outside>: post 0.58 ctx 5.57 [share -3.60 win 8.20 cc -0.09 pv 1.07] cnt 0.00 g_win 4.13 g_nowin 0.00
- "they were" (11/12 bracketed)
  - ONE UNIT:  <they were> happy  | post 0.77 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c3082', 0.4], ['c35260', 0.2], ['c4', 0.2], ['c9298', 0.2]]
  - split   :  <they were> happy to find a new friend  | post 0.27 ctx 0.83 [share -1.64 win 2.24 cc 0.35 pv -0.11] cnt 0.00 g_win 0.60 g_nowin 0.00 | cc [['c3082', 0.4], ['c35260', 0.2], ['c4', 0.2], ['c9298', 0.2]]
    - crossed by <were happy>: post 0.67 ctx 2.49 [share -1.64 win 3.98 cc 0.27 pv -0.11] cnt 0.00 g_win 2.46 g_nowin 0.00
- "tim was" (7/11 bracketed)
  - ONE UNIT:  <tim was> sad  | post 0.65 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.89 g_nowin 0.00 | cc [['c15874', 0.2], ['c3082', 0.4], ['c41192', 0.2], ['c4437', 0.2]]
  - split   : a little boy named <tim was> sad  | post 0.35 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 2.82 g_nowin 0.00 | cc [['c15874', 0.2], ['c3082', 0.4], ['c41192', 0.2], ['c4437', 0.2]]
    - crossed by <was sad>: post 0.38 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "were very" (3/10 bracketed)
  - ONE UNIT: tim and his mom <were very> surprised  | post 0.72 ctx 2.06 [share -1.52 win 3.52 cc 0.16 pv -0.10] cnt 0.00 g_win 2.00 g_nowin 0.00 | cc [['c12540', 0.2], ['c1395', 0.2], ['c17636', 0.4], ['c6276', 0.2]]
  - split   : they <were very> happy  | post 0.01 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c12540', 0.16666666666666666], ['c1395', 0.16666666666666666], ['c146999', 0.16666666666666666], ['c17636', 0.3333333333333333], ['c6276', 0.16666666666666666]]
    - crossed by <they were>: post 0.99 ctx 2.96 [share -1.64 win 4.37 cc 0.35 pv -0.11] cnt 0.00 g_win 2.73 g_nowin 0.00
    - crossed by <very happy>: post 0.97 ctx 1.76 [share -1.52 win 3.11 cc 0.27 pv -0.10] cnt 0.00 g_win 1.59 g_nowin 0.00
- "play with" (7/9 bracketed)
  - ONE UNIT: now i can <play with> the toy  | post 0.94 ctx 2.89 [share -2.40 win 4.92 cc 0.30 pv 0.06] cnt 0.00 g_win 2.71 g_nowin 0.00 | cc [['c1110', 0.3333333333333333], ['c149143', 0.16666666666666666], ['c7776', 0.3333333333333333], ['c99503', 0.16666666666666666]]
  - split   : tom and anna liked to <play with> water  | post 0.28 ctx 1.95 [share -0.90 win 2.74 cc 0.08 pv 0.02] cnt 0.00 g_win 1.85 g_nowin 0.00 | cc [['c1395', 0.2], ['c6276', 0.2], ['c7132', 0.6000000000000001]]
    - crossed by <to play>: post 0.70 ctx 5.53 [share -3.60 win 8.42 cc -0.35 pv 1.07] cnt 0.00 g_win 3.61 g_nowin 0.00
- "was very" (3/9 bracketed)
  - ONE UNIT: the little boy <was very> happy  | post 0.62 ctx 2.29 [share -2.81 win 5.23 cc 0.07 pv -0.21] cnt 0.00 g_win 2.42 g_nowin 0.00 | cc [['c1395', 0.2], ['c16488', 0.2], ['c17636', 0.4], ['c6276', 0.2]]
  - split   : tim <was very> happy  | post 0.01 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c1395', 0.16666666666666666], ['c16488', 0.16666666666666666], ['c17636', 0.3333333333333333], ['c56118', 0.16666666666666666], ['c6276', 0.16666666666666666]]
    - crossed by <tim was>: post 0.99 ctx 2.81 [share -1.64 win 4.37 cc 0.19 pv -0.11] cnt 0.00 g_win 2.73 g_nowin 0.00
    - crossed by <very happy>: post 0.97 ctx 1.64 [share -1.52 win 2.99 cc 0.27 pv -0.10] cnt 0.00 g_win 1.47 g_nowin 0.00
- "to help" (4/8 bracketed)
  - ONE UNIT: mia was happy <to help> her mom  | post 0.46 ctx 2.26 [share -2.40 win 4.37 cc 0.23 pv 0.06] cnt 0.00 g_win 1.97 g_nowin 0.00 | cc [['c1110', 0.4], ['c15880', 0.2], ['c3922', 0.4]]
  - split   : they wanted <to help> more  | post 0.34 ctx 1.72 [share -0.90 win 2.51 cc 0.08 pv 0.02] cnt 0.00 g_win 1.61 g_nowin 0.00 | cc [['c1395', 0.2], ['c3474', 0.4], ['c36309', 0.2], ['c6276', 0.2]]
    - crossed by <they wanted to>: post 0.56 ctx 1.40 [share -1.64 win 2.78 cc 0.38 pv -0.11] cnt 0.00 g_win 1.14 g_nowin 0.00
    - crossed by <wanted to>: post 0.62 ctx 1.22 [share -1.64 win 3.07 cc -0.10 pv -0.11] cnt 0.00 g_win 1.43 g_nowin 0.00
