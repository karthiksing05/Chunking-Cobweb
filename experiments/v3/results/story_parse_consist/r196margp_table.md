seeds [13, 17, 19]; held-out 300; strings 125

### 250 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.284 [0.290 / 0.298 / 0.264] | 0.478 [0.488 / 0.502 / 0.446] | 0.356 [0.363 / 0.374 / 0.332] | 0.450 [0.459 / 0.461 / 0.431] | 0.577 [0.606 / 0.606 / 0.520] | 0.295 [0.289 / 0.277 / 0.320] | 0.329 [0.350 / 0.321 / 0.317] | 0.847 [0.844 / 0.839 / 0.859] | 0.606 [0.598 / 0.620 / 0.599] |
| Vmarg | 3.303 [3.303 / 3.303 / 3.303] | 0.343 [0.324 / 0.352 / 0.352] | 0.578 [0.546 / 0.594 / 0.594] | 0.430 [0.407 / 0.442 / 0.442] | 0.382 [0.364 / 0.391 / 0.391] | 0.528 [0.465 / 0.559 / 0.559] | 0.303 [0.330 / 0.290 / 0.290] | 0.171 [0.201 / 0.156 / 0.156] | 0.885 [0.873 / 0.891 / 0.891] | 0.872 [0.808 / 0.904 / 0.904] |

diagnosis (V0, pooled over seeds): strings always bracketed 48, never 160, mixed 167; pairs of occurrences of one string with the SAME cc (content concept bands) 2987/3948 = 0.757

bracket rate by position: interior 226/1122 = 0.201; start 319/534 = 0.597

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.280 | 0.98 |
| ctx | +1.093 | 0.90 |
| cnt | +0.000 | 0.00 |
| coh | -0.027 | 0.02 |
| t_share | -0.675 | 0.16 |
| t_win | +1.875 | 0.89 |
| t_cc | -0.076 | 0.34 |
| t_pv | -0.031 | 0.24 |
| g_win | +1.167 | 0.91 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.679, between-string sd 0.644
- t_win: within-string sd 1.326, between-string sd 1.186
- t_cc: within-string sd 0.183, between-string sd 0.239
- g_win: within-string sd 0.751, between-string sd 0.680
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.000
- coh: within-string sd 0.196, between-string sd 1.894

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "wanted to" (1/18 bracketed)
  - ONE UNIT: amy <wanted to> make something with it  | post 0.52 ctx 2.79 [share -3.09 win 6.10 cc -1.00 pv 0.79] cnt 0.00 g_win 2.19 g_nowin 0.00 | cc [['c30854', 0.2], ['c31143', 0.2], ['c3625', 0.4], ['c5985', 0.2]]
  - split   : tom <wanted to> help  | post 0.21 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c14900', 0.2], ['c17975', 0.4], ['c25649', 0.2], ['c37808', 0.2]]
    - crossed by <tom wanted>: post 0.59 ctx 0.70 [share -1.68 win 3.08 cc -0.46 pv -0.25] cnt 0.00 g_win 1.41 g_nowin 0.00
    - crossed by <to help>: post 0.63 ctx 0.94 [share -0.94 win 2.05 cc -0.13 pv -0.04] cnt 0.00 g_win 1.11 g_nowin 0.00
- "to play" (8/15 bracketed)
  - ONE UNIT: they liked <to play> together  | post 0.45 ctx 1.23 [share -0.94 win 2.07 cc 0.14 pv -0.04] cnt 0.00 g_win 1.13 g_nowin 0.00 | cc [['c1099', 0.4], ['c5985', 0.2], ['c6695', 0.4]]
  - split   : tim loved <to play> outside with his friends  | post 0.30 ctx 3.19 [share -2.28 win 5.69 cc -0.10 pv -0.12] cnt 0.00 g_win 3.41 g_nowin 0.00 | cc [['c1099', 0.4], ['c5985', 0.2], ['c6695', 0.4]]
    - crossed by <play outside>: post 0.65 ctx 4.05 [share -3.44 win 7.31 cc -0.47 pv 0.65] cnt 0.00 g_win 3.25 g_nowin 0.00
- "they were" (8/12 bracketed)
  - ONE UNIT:  <they were> very happy  | post 0.78 ctx 1.92 [share -1.68 win 3.91 cc -0.07 pv -0.25] cnt 0.00 g_win 2.24 g_nowin 0.00 | cc [['c13914', 0.4], ['c25649', 0.2], ['c4', 0.4]]
  - split   :  <they were> happy  | post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c13914', 0.4], ['c25649', 0.2], ['c4', 0.4]]
    - crossed by <were happy>: post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "tim was" (6/11 bracketed)
  - ONE UNIT:  <tim was> very happy  | post 0.78 ctx 1.98 [share -1.68 win 3.91 cc -0.01 pv -0.25] cnt 0.00 g_win 2.24 g_nowin 0.00 | cc [['c14920', 0.4], ['c25649', 0.2], ['c4140', 0.4]]
  - split   :  <tim was> sad  | post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c14920', 0.4], ['c25649', 0.2], ['c4140', 0.4]]
    - crossed by <was sad>: post 0.50 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "were very" (3/10 bracketed)
  - ONE UNIT: tim and his mom <were very> surprised  | post 0.47 ctx 1.99 [share -1.66 win 3.97 cc -0.06 pv -0.26] cnt 0.00 g_win 2.31 g_nowin 0.00 | cc [['c22893', 0.2], ['c24041', 0.4], ['c37817', 0.2], ['c5985', 0.2]]
  - split   : they <were very> happy  | post 0.12 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c25649', 0.2], ['c33312', 0.4], ['c37808', 0.2], ['c8227', 0.2]]
    - crossed by <they were>: post 0.78 ctx 1.92 [share -1.68 win 3.91 cc -0.07 pv -0.25] cnt 0.00 g_win 2.24 g_nowin 0.00
    - crossed by <very happy>: post 0.65 ctx 0.77 [share -0.94 win 2.05 cc -0.31 pv -0.04] cnt 0.00 g_win 1.11 g_nowin 0.00
- "a big" (4/10 bracketed)
  - ONE UNIT:  <a big> bird came and took it away  | post 0.57 ctx 0.40 [share -1.68 win 2.67 cc -0.34 pv -0.25] cnt 0.00 g_win 0.99 g_nowin 0.00 | cc [['c25649', 0.2], ['c28700', 0.2], ['c31969', 0.2], ['c31978', 0.2], ['c37808', 0.2]]
  - split   : at the park they saw <a big> tree  | post 0.13 ctx 0.29 [share -0.94 win 1.13 cc 0.14 pv -0.04] cnt 0.00 g_win 0.19 g_nowin 0.00 | cc [['c11963', 0.2], ['c22893', 0.2], ['c5985', 0.2], ['c7175', 0.4]]
    - crossed by <big tree>: post 0.70 ctx 1.92 [share -1.66 win 3.90 cc -0.06 pv -0.26] cnt 0.00 g_win 2.23 g_nowin 0.00
    - crossed by <they saw a>: post 0.41 ctx 1.93 [share -3.44 win 5.14 cc -0.42 pv 0.65] cnt 0.00 g_win 1.68 g_nowin 0.00
- "play with" (1/9 bracketed)
  - ONE UNIT: would you like to <play with> me  | post 0.39 ctx 1.54 [share -0.94 win 2.37 cc 0.14 pv -0.04] cnt 0.00 g_win 1.44 g_nowin 0.00 | cc [['c22893', 0.2], ['c24041', 0.4], ['c37817', 0.2], ['c5985', 0.2]]
  - split   : now i can <play with> the toy  | post 0.36 ctx 1.39 [share -2.28 win 3.85 cc -0.06 pv -0.12] cnt 0.00 g_win 1.58 g_nowin 0.00 | cc [['c22893', 0.2], ['c35242', 0.2], ['c5985', 0.2], ['c7225', 0.4]]
    - crossed by <with the toy>: post 0.33 ctx 1.22 [share -1.66 win 3.00 cc 0.13 pv -0.26] cnt 0.00 g_win 1.34 g_nowin 0.00
- "to help" (7/8 bracketed)
  - ONE UNIT: they wanted <to help> more  | post 0.44 ctx 1.97 [share -0.94 win 2.81 cc 0.14 pv -0.04] cnt 0.00 g_win 1.87 g_nowin 0.00 | cc [['c1099', 0.4], ['c15389', 0.4], ['c5985', 0.2]]
  - split   : mia wanted <to help> tim find his mom  | post 0.27 ctx 0.46 [share -3.44 win 3.78 cc -0.53 pv 0.65] cnt 0.00 g_win 1.13 g_nowin 0.00 | cc [['c1099', 0.4], ['c15389', 0.4], ['c5985', 0.2]]
    - crossed by <help tim>: post 0.37 ctx 1.40 [share -2.28 win 3.86 cc -0.06 pv -0.12] cnt 0.00 g_win 1.59 g_nowin 0.00

### 1000 sentences learned

| variant | brackets/sent | P | R | F1 | F1+sv | NP recall | crossing | consistency | decisive | xseed F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0_default | 3.303 [3.303 / 3.303 / 3.303] | 0.294 [0.302 / 0.304 / 0.276] | 0.495 [0.508 / 0.512 / 0.464] | 0.369 [0.379 / 0.381 / 0.346] | 0.481 [0.483 / 0.489 / 0.471] | 0.724 [0.756 / 0.756 / 0.661] | 0.251 [0.240 / 0.239 / 0.274] | 0.335 [0.337 / 0.323 / 0.346] | 0.857 [0.835 / 0.875 / 0.862] | 0.756 [0.770 / 0.781 / 0.718] |
| Vmarg | 3.303 [3.303 / 3.303 / 3.303] | 0.337 [0.351 / 0.337 / 0.324] | 0.569 [0.592 / 0.568 / 0.546] | 0.423 [0.441 / 0.423 / 0.407] | 0.379 [0.390 / 0.383 / 0.364] | 0.517 [0.551 / 0.535 / 0.465] | 0.311 [0.292 / 0.310 / 0.331] | 0.185 [0.158 / 0.196 / 0.201] | 0.878 [0.889 / 0.871 / 0.873] | 0.818 [0.852 / 0.825 / 0.778] |

diagnosis (V0, pooled over seeds): strings always bracketed 51, never 175, mixed 149; pairs of occurrences of one string with the SAME cc (content concept bands) 2540/3948 = 0.643

bracket rate by position: interior 173/1122 = 0.154; start 382/534 = 0.715

mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the share of strings where the difference is positive

| quantity | mean diff (bits) | share > 0 |
|---|---|---|
| post | +0.368 | 1.00 |
| ctx | +1.069 | 0.81 |
| cnt | +0.000 | 0.01 |
| coh | +0.005 | 0.04 |
| t_share | -0.497 | 0.19 |
| t_win | +1.557 | 0.79 |
| t_cc | -0.036 | 0.36 |
| t_pv | +0.045 | 0.28 |
| g_win | +1.010 | 0.83 |
| g_nowin | +0.000 | 0.00 |

- ctx: within-string sd 0.766, between-string sd 0.779
- t_win: within-string sd 1.285, between-string sd 1.176
- t_cc: within-string sd 0.197, between-string sd 0.210
- g_win: within-string sd 0.733, between-string sd 0.747
- g_nowin: within-string sd 0.000, between-string sd 0.000
- cnt: within-string sd 0.000, between-string sd 0.004
- coh: within-string sd 0.255, between-string sd 1.869

case studies (seed 13): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept

- "to play" (8/15 bracketed)
  - ONE UNIT: they liked <to play> together  | post 0.47 ctx 1.82 [share -0.92 win 2.52 cc 0.21 pv 0.02] cnt 0.00 g_win 1.59 g_nowin 0.00 | cc [['c1099', 0.4], ['c48181', 0.2], ['c5985', 0.2], ['c6695', 0.2]]
  - split   : tim loved <to play> outside with his friends  | post 0.21 ctx 3.64 [share -3.58 win 6.79 cc -0.64 pv 1.07] cnt 0.00 g_win 2.75 g_nowin 0.00 | cc [['c1099', 0.4], ['c48181', 0.2], ['c5985', 0.2], ['c6695', 0.2]]
    - crossed by <play outside>: post 0.74 ctx 5.32 [share -3.58 win 8.15 cc -0.32 pv 1.07] cnt 0.00 g_win 4.13 g_nowin 0.00
- "they were" (9/12 bracketed)
  - ONE UNIT:  <they were> very happy  | post 0.87 ctx 2.78 [share -1.65 win 4.34 cc 0.20 pv -0.11] cnt 0.00 g_win 2.69 g_nowin 0.00 | cc [['c13914', 0.2], ['c151047', 0.2], ['c4', 0.6000000000000001]]
  - split   :  <they were> happy  | post 0.48 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c13914', 0.2], ['c151047', 0.2], ['c4', 0.6000000000000001]]
    - crossed by <were happy>: post 0.52 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.21 g_nowin 0.00
- "tim was" (8/11 bracketed)
  - ONE UNIT:  <tim was> sad  | post 0.51 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.79 g_nowin 0.00 | cc [['c117316', 0.2], ['c14920', 0.4], ['c151258', 0.2], ['c4140', 0.2]]
  - split   : a little boy named <tim was> sad  | post 0.37 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 2.90 g_nowin 0.00 | cc [['c117316', 0.2], ['c14920', 0.4], ['c151258', 0.2], ['c4140', 0.2]]
    - crossed by <was sad>: post 0.39 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "a big" (5/10 bracketed)
  - ONE UNIT:  <a big> bird came and took it away  | post 0.80 ctx 1.97 [share -1.65 win 3.66 cc 0.07 pv -0.11] cnt 0.00 g_win 2.00 g_nowin 0.00 | cc [['c117316', 0.16666666666666666], ['c130728', 0.16666666666666666], ['c151258', 0.16666666666666666], ['c160978', 0.16666666666666666], ['c32003', 0.16666666666666666], ['c32043', 0.16666666666666666]]
  - split   : at the park they saw <a big> tree  | post 0.28 ctx 1.89 [share -0.92 win 2.58 cc 0.21 pv 0.02] cnt 0.00 g_win 1.66 g_nowin 0.00 | cc [['c22549', 0.2], ['c22893', 0.2], ['c5985', 0.2], ['c7175', 0.4]]
    - crossed by <big tree>: post 0.50 ctx 1.96 [share -1.57 win 3.40 cc 0.25 pv -0.12] cnt 0.00 g_win 1.83 g_nowin 0.00
- "to help" (4/8 bracketed)
  - ONE UNIT: they all wanted <to help> her feel happy  | post 0.58 ctx 2.41 [share -3.58 win 5.90 cc -0.98 pv 1.07] cnt 0.00 g_win 2.38 g_nowin 0.00 | cc [['c1099', 0.4], ['c15389', 0.4], ['c5985', 0.2]]
  - split   : they wanted <to help> more  | post 0.42 ctx 1.69 [share -0.92 win 2.36 cc 0.23 pv 0.02] cnt 0.00 g_win 1.43 g_nowin 0.00 | cc [['c1099', 0.3333333333333333], ['c15389', 0.3333333333333333], ['c5985', 0.16666666666666666], ['c60614', 0.16666666666666666]]
    - crossed by <help more>: post 0.48 ctx 1.50 [share -0.92 win 2.33 cc 0.07 pv 0.02] cnt 0.00 g_win 1.41 g_nowin 0.00
- "he was" (6/8 bracketed)
  - ONE UNIT:  <he was> so happy  | post 0.90 ctx 2.76 [share -1.65 win 4.49 cc 0.02 pv -0.11] cnt 0.00 g_win 2.84 g_nowin 0.00 | cc [['c117316', 0.2], ['c4853', 0.2], ['c52873', 0.2], ['c729', 0.4]]
  - split   :  <he was> happy  | post 0.48 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c117316', 0.2], ['c4853', 0.2], ['c52873', 0.2], ['c729', 0.4]]
    - crossed by <was happy>: post 0.52 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00
- "his mom" (4/7 bracketed)
  - ONE UNIT:  <his mom> said no tim  | post 0.81 ctx 1.77 [share -1.65 win 3.82 cc -0.30 pv -0.11] cnt 0.00 g_win 2.17 g_nowin 0.00 | cc [['c117316', 0.2], ['c151258', 0.2], ['c160978', 0.2], ['c9834', 0.4]]
  - split   : tim and <his mom> were very surprised  | post 0.17 ctx 0.81 [share -2.38 win 3.02 cc 0.10 pv 0.07] cnt 0.00 g_win 2.72 g_nowin 0.00 | cc [['c109107', 0.16666666666666666], ['c159133', 0.16666666666666666], ['c22893', 0.16666666666666666], ['c5985', 0.16666666666666666], ['c75699', 0.3333333333333333]]
    - crossed by <and his>: post 0.68 ctx 2.48 [share -1.65 win 4.03 cc 0.21 pv -0.11] cnt 0.00 g_win 2.38 g_nowin 0.00
- "to be" (3/6 bracketed)
  - ONE UNIT: it wanted <to be> with its friend again  | post 0.48 ctx 2.80 [share -2.38 win 5.44 cc -0.32 pv 0.07] cnt 0.00 g_win 3.05 g_nowin 0.00 | cc [['c1099', 0.4], ['c12363', 0.2], ['c18878', 0.2], ['c5985', 0.2]]
  - split   : they wanted <to be> pretty like her  | post 0.12 ctx 0.00 [share 0.00 win 0.00 cc 0.00 pv 0.00] cnt 0.00 g_win 0.00 g_nowin 0.00 | cc [['c1099', 0.4], ['c12363', 0.2], ['c18878', 0.2], ['c5985', 0.2]]
    - crossed by <they wanted to>: post 0.39 ctx 1.75 [share -1.65 win 3.34 cc 0.17 pv -0.11] cnt 0.00 g_win 1.69 g_nowin 0.00
    - crossed by <be pretty>: post 0.47 ctx 1.79 [share -2.38 win 4.00 cc 0.10 pv 0.07] cnt 0.00 g_win 1.62 g_nowin 0.00
