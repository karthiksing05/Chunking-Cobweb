seeds [13, 17, 19]; held-out 300; strings 125

### 250 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.237 [0.265 / 0.236 / 0.209] | 0.399 [0.447 / 0.398 / 0.352] | 0.297 [0.333 / 0.296 / 0.262] | 0.336 [0.366 / 0.331 / 0.312] | 0.378 [0.425 / 0.441 / 0.268] | 0.405 [0.346 / 0.425 / 0.445] | 0.281 [0.274 / 0.265 / 0.304] | 0.823 [0.830 / 0.826 / 0.813] | 0.448 [0.426 / 0.442 / 0.476] |

diagnosis (V0, pooled over seeds): strings always bracketed 25, never 161, mixed 189; pairs of occurrences of one string with the SAME cc (content concept bands) 2992/3948 = 0.758

bracket rate by position: interior 316/1122 = 0.282; start 149/534 = 0.279

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.234 | 0.98 |
| ctx | +0.990 | 0.86 |
| cnt | +0.000 | 0.00 |
| coh | -0.075 | 0.06 |
| t_share | -0.884 | 0.13 |
| t_win | +1.970 | 0.84 |
| t_cc | -0.080 | 0.36 |
| t_pv | -0.016 | 0.26 |
| g_win | +1.016 | 0.88 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.644, between-string sd 0.668
- t_win: within-string sd 1.348, between-string sd 1.287
- t_cc: within-string sd 0.204, between-string sd 0.211
- g_win: within-string sd 0.698, between-string sd 0.660
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000
- coh: within-string sd 0.285, between-string sd 1.747

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "to play" (6/15 bracketed)
  - ONE UNIT: they liked <to play> together  | post 0.48 ctx 1.30 [share -0.97 win 2.02 cc 0.28 pv -0.02] cnt 0.00 g_win 1.05 g_nowin 0.00 | cc [['c6957', 0.6000000000000001], ['c7603', 0.4]]
  - split   : tim loved <to play> outside with his friends  | post 0.27 ctx 3.33 [share -2.21 win 5.46 cc 0.20 pv -0.12] cnt 0.00 g_win 3.25 g_nowin 0.00 | cc [['c6957', 0.6000000000000001], ['c7603', 0.4]]
    - crossed by <play outside>: post 0.65 ctx 4.02 [share -3.38 win 7.26 cc -0.53 pv 0.67] cnt 0.00 g_win 3.16 g_nowin 0.00
- "they were" (6/12 bracketed)
  - ONE UNIT:  <they were> very happy  | post 0.57 ctx 0.95 [share -1.75 win 3.05 cc -0.07 pv -0.28] cnt 0.00 g_win 1.30 g_nowin 0.00 | cc [['c2848', 0.4], ['c4', 0.6000000000000001]]
  - split   :  <they were> happy  | post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c2848', 0.4], ['c4', 0.6000000000000001]]
    - crossed by <were happy>: post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "tim was" (6/11 bracketed)
  - ONE UNIT:  <tim was> very happy  | post 0.57 ctx 0.94 [share -1.75 win 3.05 cc -0.08 pv -0.28] cnt 0.00 g_win 1.30 g_nowin 0.00 | cc [['c14008', 0.2], ['c15903', 0.4], ['c852', 0.4]]
  - split   :  <tim was> sad  | post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c14008', 0.2], ['c15903', 0.4], ['c852', 0.4]]
    - crossed by <was sad>: post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "were very" (4/10 bracketed)
  - ONE UNIT: tim and his mom <were very> surprised  | post 0.56 ctx 2.07 [share -1.76 win 4.07 cc -0.02 pv -0.23] cnt 0.00 g_win 2.32 g_nowin 0.00 | cc [['c27202', 0.4], ['c29603', 0.4], ['c33108', 0.2]]
  - split   : they <were very> happy  | post 0.26 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c18310', 0.8], ['c5549', 0.2]]
    - crossed by <they were>: post 0.57 ctx 0.95 [share -1.75 win 3.05 cc -0.07 pv -0.28] cnt 0.00 g_win 1.30 g_nowin 0.00
    - crossed by <very happy>: post 0.48 ctx 0.32 [share -0.97 win 1.78 cc -0.47 pv -0.02] cnt 0.00 g_win 0.81 g_nowin 0.00
- "a big" (4/10 bracketed)
  - ONE UNIT: they saw <a big> door  | post 0.62 ctx 2.02 [share -0.97 win 2.77 cc 0.24 pv -0.02] cnt 0.00 g_win 1.80 g_nowin 0.00 | cc [['c13735', 0.4], ['c18030', 0.2], ['c27202', 0.4]]
  - split   : at the park they saw <a big> tree  | post 0.25 ctx 0.38 [share -0.97 win 1.13 cc 0.24 pv -0.02] cnt 0.00 g_win 0.16 g_nowin 0.00 | cc [['c13735', 0.4], ['c18030', 0.2], ['c27202', 0.4]]
    - crossed by <big tree>: post 0.46 ctx 0.95 [share -0.97 win 1.70 cc 0.24 pv -0.02] cnt 0.00 g_win 1.31 g_nowin 0.00
    - crossed by <they saw a>: post 0.34 ctx 1.66 [share -3.38 win 5.11 cc -0.74 pv 0.67] cnt 0.00 g_win 1.98 g_nowin 0.00
- "to help" (7/8 bracketed)
  - ONE UNIT: they wanted <to help> more  | post 0.59 ctx 2.05 [share -0.97 win 2.77 cc 0.28 pv -0.02] cnt 0.00 g_win 1.80 g_nowin 0.00 | cc [['c15320', 0.4], ['c6957', 0.6000000000000001]]
  - split   : mia wanted <to help> tim find his mom  | post 0.21 ctx 0.54 [share -3.38 win 3.52 cc -0.27 pv 0.67] cnt 0.00 g_win 0.46 g_nowin 0.00 | cc [['c15320', 0.4], ['c6957', 0.6000000000000001]]
    - crossed by <help tim find>: post 0.42 ctx 2.16 [share -2.21 win 4.73 cc -0.24 pv -0.12] cnt 0.00 g_win 2.52 g_nowin 0.00
    - crossed by <help tim>: post 0.49 ctx 1.78 [share -2.21 win 4.06 cc 0.05 pv -0.12] cnt 0.00 g_win 1.85 g_nowin 0.00
- "he was" (3/8 bracketed)
  - ONE UNIT:  <he was> so happy  | post 0.63 ctx 1.16 [share -1.75 win 3.34 cc -0.15 pv -0.28] cnt 0.00 g_win 1.59 g_nowin 0.00 | cc [['c15903', 0.4], ['c24552', 0.4], ['c3171', 0.2]]
  - split   :  <he was> playing outside with his toy car  | post 0.41 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c15903', 0.4], ['c24552', 0.4], ['c3171', 0.2]]
    - crossed by <was playing outside with his toy car>: post 0.41 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "his mom" (6/7 bracketed)
  - ONE UNIT: tim and <his mom> were very surprised  | post 0.38 ctx 0.29 [share -2.21 win 2.58 cc 0.05 pv -0.12] cnt 0.00 g_win 0.36 g_nowin 0.00 | cc [['c26561', 0.4], ['c27202', 0.4], ['c36024', 0.2]]
  - split   :  <his mom> said no tim  | post 0.41 ctx 0.03 [share -1.75 win 2.72 cc -0.66 pv -0.28] cnt 0.00 g_win 0.97 g_nowin 0.00 | cc [['c10020', 0.25], ['c15903', 0.5], ['c32431', 0.25]]
    - crossed by <mom said no tim>: post 0.40 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00

### 1000 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.259 [0.257 / 0.273 / 0.246] | 0.436 [0.434 / 0.459 / 0.415] | 0.325 [0.323 / 0.342 / 0.309] | 0.373 [0.390 / 0.365 / 0.366] | 0.483 [0.457 / 0.606 / 0.386] | 0.377 [0.349 / 0.388 / 0.394] | 0.269 [0.290 / 0.257 / 0.261] | 0.839 [0.830 / 0.841 / 0.846] | 0.521 [0.519 / 0.501 / 0.543] |

diagnosis (V0, pooled over seeds): strings always bracketed 28, never 178, mixed 169; pairs of occurrences of one string with the SAME cc (content concept bands) 2423/3948 = 0.614

bracket rate by position: interior 294/1122 = 0.262; start 152/534 = 0.285

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.255 | 0.98 |
| ctx | +1.005 | 0.92 |
| cnt | +0.000 | 0.00 |
| coh | -0.018 | 0.06 |
| t_share | -0.737 | 0.12 |
| t_win | +1.747 | 0.92 |
| t_cc | -0.043 | 0.40 |
| t_pv | +0.038 | 0.33 |
| g_win | +0.958 | 0.91 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.682, between-string sd 0.810
- t_win: within-string sd 1.263, between-string sd 1.313
- t_cc: within-string sd 0.176, between-string sd 0.189
- g_win: within-string sd 0.662, between-string sd 0.739
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000
- coh: within-string sd 0.259, between-string sd 1.742

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "to play" (8/15 bracketed)
  - ONE UNIT: they liked <to play> together  | post 0.54 ctx 1.76 [share -0.98 win 2.38 cc 0.32 pv 0.04] cnt 0.00 g_win 1.40 g_nowin 0.00 | cc [['c131541', 0.4], ['c29612', 0.2], ['c6957', 0.2], ['c7603', 0.2]]
  - split   : tim loved <to play> outside with his friends  | post 0.17 ctx 3.35 [share -3.53 win 6.42 cc -0.64 pv 1.09] cnt 0.00 g_win 2.44 g_nowin 0.00 | cc [['c131541', 0.4], ['c29612', 0.2], ['c6957', 0.2], ['c7603', 0.2]]
    - crossed by <play outside>: post 0.76 ctx 5.26 [share -3.53 win 8.03 cc -0.33 pv 1.09] cnt 0.00 g_win 4.04 g_nowin 0.00
- "they were" (6/12 bracketed)
  - ONE UNIT:  <they were> very happy  | post 0.63 ctx 1.30 [share -1.73 win 3.35 cc -0.15 pv -0.17] cnt 0.00 g_win 1.62 g_nowin 0.00 | cc [['c2848', 0.4], ['c4', 0.6000000000000001]]
  - split   :  <they were> happy  | post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c2848', 0.4], ['c4', 0.6000000000000001]]
    - crossed by <were happy>: post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.08 g_nowin 0.00
- "tim was" (6/11 bracketed)
  - ONE UNIT:  <tim was> very happy  | post 0.64 ctx 1.28 [share -1.73 win 3.35 cc -0.17 pv -0.17] cnt 0.00 g_win 1.62 g_nowin 0.00 | cc [['c14008', 0.4], ['c852', 0.2], ['c89102', 0.4]]
  - split   :  <tim was> sad  | post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.06 g_nowin 0.00 | cc [['c14008', 0.4], ['c852', 0.2], ['c89102', 0.4]]
    - crossed by <was sad>: post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "a big" (5/10 bracketed)
  - ONE UNIT:  <a big> bird came and took it away  | post 0.50 ctx 0.38 [share -1.73 win 2.49 cc -0.21 pv -0.17] cnt 0.00 g_win 0.76 g_nowin 0.00 | cc [['c101613', 0.4], ['c89102', 0.4], ['c95360', 0.2]]
  - split   : at the park they saw <a big> tree  | post 0.20 ctx 1.87 [share -0.98 win 2.48 cc 0.32 pv 0.04] cnt 0.00 g_win 1.50 g_nowin 0.00 | cc [['c131541', 0.4], ['c162157', 0.2], ['c37822', 0.4]]
    - crossed by <they saw a>: post 0.68 ctx 4.55 [share -2.32 win 6.56 cc 0.24 pv 0.07] cnt 0.00 g_win 4.24 g_nowin 0.00
- "play with" (1/9 bracketed)
  - ONE UNIT: can we <play with> you  | post 0.35 ctx 0.02 [share -0.98 win 0.64 cc 0.32 pv 0.04] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c131541', 0.4], ['c53772', 0.6000000000000001]]
  - split   : now i can <play with> the toy  | post 0.39 ctx 2.34 [share -2.32 win 4.42 cc 0.17 pv 0.07] cnt 0.00 g_win 2.97 g_nowin 0.00 | cc [['c131541', 0.4], ['c7433', 0.2], ['c7668', 0.4]]
    - crossed by <can play>: post 0.52 ctx 2.38 [share -2.32 win 4.46 cc 0.17 pv 0.07] cnt 0.00 g_win 2.14 g_nowin 0.00
- "to help" (7/8 bracketed)
  - ONE UNIT: they wanted <to help> more  | post 0.56 ctx 1.83 [share -0.98 win 2.45 cc 0.32 pv 0.04] cnt 0.00 g_win 1.46 g_nowin 0.00 | cc [['c131541', 0.3333333333333333], ['c15320', 0.3333333333333333], ['c59517', 0.16666666666666666], ['c6957', 0.16666666666666666]]
  - split   : mia wanted <to help> tim find his mom  | post 0.13 ctx 1.32 [share -0.98 win 1.93 cc 0.32 pv 0.04] cnt 0.00 g_win 0.95 g_nowin 0.00 | cc [['c131541', 0.4], ['c15320', 0.4], ['c6957', 0.2]]
    - crossed by <help tim>: post 0.77 ctx 3.75 [share -2.32 win 5.82 cc 0.17 pv 0.07] cnt 0.00 g_win 3.50 g_nowin 0.00
- "he was" (3/8 bracketed)
  - ONE UNIT:  <he was> so happy  | post 0.67 ctx 1.46 [share -1.73 win 3.52 cc -0.16 pv -0.17] cnt 0.00 g_win 1.79 g_nowin 0.00 | cc [['c24552', 0.4], ['c3171', 0.2], ['c89102', 0.4]]
  - split   :  <he was> playing outside with his toy car  | post 0.39 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c24552', 0.4], ['c3171', 0.2], ['c89102', 0.4]]
    - crossed by <was playing outside with his toy car>: post 0.39 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "his mom" (4/7 bracketed)
  - ONE UNIT:  <his mom> said no tim  | post 0.50 ctx 0.65 [share -1.73 win 2.87 cc -0.32 pv -0.17] cnt 0.00 g_win 1.14 g_nowin 0.00 | cc [['c10020', 0.3333333333333333], ['c108455', 0.16666666666666666], ['c89102', 0.3333333333333333], ['c95360', 0.16666666666666666]]
  - split   : tim and <his mom> were very surprised  | post 0.29 ctx 0.95 [share -2.32 win 3.03 cc 0.17 pv 0.07] cnt 0.00 g_win 1.49 g_nowin 0.00 | cc [['c131541', 0.4], ['c133449', 0.2], ['c26561', 0.4]]
    - crossed by <and his>: post 0.59 ctx 2.11 [share -1.73 win 3.98 cc 0.02 pv -0.17] cnt 0.00 g_win 2.25 g_nowin 0.00
