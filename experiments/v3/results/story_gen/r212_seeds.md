# Round 212: eight seeds for the generation claim

Seeds 13, 17, 19, 23 (r200/r204/r205) plus new seeds 29, 31, 37, 41 (this round). Same pipeline, same Qwen2.5-3B judge, same fair count (distinct new, judged-grammatical, unstored-trigram sentences per attempted draw). Full output: `r212-analysis.txt`. Code: `experiments/v3/diag/r212_analysis.py`, `experiments/v3/diag/r212_grid.sh` (copy of the gitignored `logs/v3_r212_grid.sh`).

Seeds: a seed shuffles the stories and the first n kept sentences after the seed's 1,500 test stories are the training set (`story_frontier.train_sentences`); the sampling seed is derived from it. New seeds are independent shuffles, built the same way as 13/17/19/23. Training sets are not disjoint (shared sentences, mean overlap with another seed: n=250 .077, n=500 .107-.117, n=1000 .136-.142), and old-old overlap equals new-old overlap, so the new seeds are no more or less independent than the old ones. Attempted draws per seed come from the seed's own dumps (1460-1600). The strict lexicon is unchanged (the n=1000 reference sentences of seeds 13/17/19/23).

Confirmatory settings, fixed from r205: TRELLIS edge n250 w7 t3.5, n500 w6 t2.8, n1000 w7 t2.0; LSTM t.7/.8/.8; GPT t.7/.8/.8.

## Fair count, mean over 8 seeds (old 4 / new 4 in brackets)

| n | TRELLIS edge | LSTM | GPT |
|---|---|---|---|
| 250 | .2419 (.2408 / .2430) | .2323 (.2436 / .2211) | .2068 (.2127 / .2010) |
| 500 | .2982 (.3011 / .2952) | .2990 (.3044 / .2935) | .2738 (.2827 / .2649) |
| 1000 | .3281 (.3381 / .3180) | .3346 (.3439 / .3252) | .3282 (.3244 / .3321) |

## Paired differences, fair count, 8 seeds

| n | contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|---|
| 250 | TRELLIS-LSTM | +.0096 | [-.0110, +.0301] | [-.0064, +.0252] | 5/8 |
| 250 | TRELLIS-GPT | +.0351 | [+.0163, +.0539] | [+.0217, +.0505] | 8/8 |
| 500 | TRELLIS-LSTM | -.0008 | [-.0129, +.0113] | [-.0108, +.0075] | 4/8 |
| 500 | TRELLIS-GPT | +.0244 | [+.0098, +.0390] | [+.0146, +.0368] | 8/8 |
| 1000 | TRELLIS-LSTM | -.0065 | [-.0279, +.0149] | [-.0228, +.0101] | 3/8 |
| 1000 | TRELLIS-GPT | -.0002 | [-.0227, +.0223] | [-.0161, +.0184] | 4/8 |

Per seed (TRELLIS / LSTM / GPT), n=1000: 13 .337/.342/.317; 17 .337/.332/.356; 19 .353/.395/.351; 23 .326/.308/.274; 29 .335/.303/.329; 31 .299/.320/.330; 37 .304/.340/.327; 41 .334/.338/.342. Per-seed tables for n=250 and 500 are in `r212-analysis.txt`.

## Pre-registered verdicts (fair count, 8 seeds; the same result under t and bootstrap CIs)

- "Matches LSTM" (CI includes 0 and lower bound > -.03): HOLDS at n=250, 500 and 1000 (lower bounds -.0110/-.0064, -.0129/-.0108, -.0279/-.0228 for t/bootstrap). At n=1000 the t lower bound -.0279 clears -.03 narrowly.
- "Beats GPT" (CI above 0): HOLDS at n=250 and 500 (t lower bounds +.0163 and +.0098). FAILS at n=1000 (CI [-.0227, +.0223], 4/8 wins). The r205 n=1000 edge over GPT (+.014 on 4 seeds) does not replicate: the new seeds give -.014 (GPT t.8 .332 against TRELLIS .318). The paper's "beats a matched GPT" now holds at 250 and 500 only.

## Boundary-strict count, 8 seeds

| n | TRELLIS | LSTM | GPT | TRELLIS-LSTM mean, t CI, boot CI, wins | TRELLIS-GPT mean, t CI, boot CI, wins |
|---|---|---|---|---|---|
| 250 | .2236 | .2154 | .1959 | +.0082 [-.0102, +.0266] [-.0059, +.0221] 5/8 | +.0278 [+.0104, +.0452] [+.0146, +.0415] 7/8 |
| 500 | .2725 | .2819 | .2617 | -.0094 [-.0211, +.0023] [-.0193, -.0016] 2/8 | +.0108 [-.0055, +.0270] [-.0017, +.0235] 5/8 |
| 1000 | .3003 | .3147 | .3139 | -.0144 [-.0392, +.0104] [-.0334, +.0049] 2/8 | -.0136 [-.0363, +.0091] [-.0297, +.0052] 2/8 |

Same pre-registered rules applied to the strict count (an extension, not pre-registered for strict): matches LSTM holds at 250 (both CIs); at 500 holds under t, fails under bootstrap (upper bound -.0016); at 1000 fails both (t lower bound -.0392). Beats GPT holds at 250 only; fails at 500 and 1000.

## Selection re-check (fair count means per setting)

- n=500: TRELLIS t2.8 best on 8 seeds (.2982 against t2.4 .2918; t3.2 only exists on new seeds). On the new seeds alone t2.4 (.2971) edges t2.8 (.2952). Interior. LSTM best t.8 (8 seeds); on new seeds t.7 (.3000 against .2935). GPT t.8.
- n=1000: TRELLIS t2.0 best on new seeds, old seeds and 8 seeds; interior. LSTM t.8, GPT t.8 on all subsets (GPT t.9 .3298 is close on new seeds).
- n=250: TRELLIS t3.5 is no longer the best on either subset: new seeds t4.0 (.2451, against t3.5 .2430, t3.0 .2389); old seeds t4.5 (.2579). LSTM and GPT t.7 on every subset.

## n=250 temperature extension (TRELLIS edge w7, old seeds 13/17/19/23, fair count)

t2.8 .2196; t3.5 .2408; t4.5 .2579; t6.0 .2561. The peak is at t4.5 and t6.0 is slightly lower, so the optimum is interior but flat from 4.5 to 6.0. On the new seeds t4.0 is the top of their grid (.2451). Post hoc, with t4.5 on the old seeds TRELLIS edge is .2579 against LSTM .2436 (+.0143) and GPT .2127 (+.045). That setting was chosen after seeing these seeds, so it is not confirmatory, and t4.5 was not run on the new seeds. Because t3.5 was fixed in advance, the confirmatory n=250 comparison above uses t3.5.

## Reading

- The tie with the LSTM survives 8 seeds at the fair count at every n (differences +.0096, -.0008, -.0065, all inside noise; per-seed spread is large).
- The GPT win is solid at n=250 (8/8) and n=500 (8/8) and gone at n=1000.
- On the boundary-strict count the picture is weaker: TRELLIS trails the LSTM at 500 and 1000 and is level with or below GPT at 1000.

## Rule slips

None known. Waiter was one capped bash loop and has finished; no jobs are left running.
