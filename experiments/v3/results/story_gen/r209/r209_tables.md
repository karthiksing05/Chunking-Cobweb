
## n=1000: real-sentence ceiling and calibration (share judged grammatical, 1200 real held-out sentences)

| judge | real (ceiling) | scrambled | content swapped | real >= .95? |
|---|---|---|---|---|
| 3B/v1 | 0.891 | 0.107 | 0.058 | no |
| Qwen3-8B/v1 | 0.863 | 0.039 | 0.039 | no |
| Qwen3-8B/strict | 0.958 | 0.058 | 0.060 | YES |
| Llama-3.1-8B/v1 | 0.918 | 0.132 | 0.068 | no |
| Llama-3.1-8B/strict | 0.976 | 0.163 | 0.117 | YES |
| phi-4/v1 | 0.914 | 0.089 | 0.052 | no |
| phi-4/strict | 0.970 | 0.068 | 0.075 | YES |

Real sentences failed by the highest-ceiling judge (Llama-3.1-8B/strict), first 15: i will be so happy if you do; do you want some; go tom go; lily and tom look up; said the car; his mom said no tim; ben and lily said; i will help you it said; tom and lily said; i see him lily; lily and sam asked; they are too hard for me too; tim said excited; and the dog; she wanted it bad

### n=1000: per family (mean over seeds [13, 17]); gram = share of distinct new judged grammatical, gram|tri = same among distinct new with an unstored trigram, norm = / real ceiling, fair = distinct gram-novel per attempted draw

| judge | family | distinct | gram | gram norm | gram\|tri | gram\|tri norm | fair/draw |
|---|---|---|---|---|---|---|---|
| 3B/v1 | TRELLIS edge w7 t2.0 | 1302 | 0.541 | 0.608 | 0.482 | 0.541 | 0.337 |
| 3B/v1 | TRELLIS w6 t1.6 | 1255 | 0.515 | 0.578 | 0.466 | 0.523 | 0.323 |
| 3B/v1 | LSTM t0.8 | 1366 | 0.591 | 0.663 | 0.503 | 0.564 | 0.337 |
| 3B/v1 | GPT t0.8 | 1314 | 0.556 | 0.624 | 0.487 | 0.547 | 0.336 |
| Qwen3-8B/v1 | TRELLIS edge w7 t2.0 | 1302 | 0.485 | 0.562 | 0.422 | 0.489 | 0.295 |
| Qwen3-8B/v1 | TRELLIS w6 t1.6 | 1255 | 0.391 | 0.453 | 0.334 | 0.387 | 0.231 |
| Qwen3-8B/v1 | LSTM t0.8 | 1366 | 0.522 | 0.606 | 0.426 | 0.494 | 0.286 |
| Qwen3-8B/v1 | GPT t0.8 | 1314 | 0.508 | 0.589 | 0.430 | 0.498 | 0.297 |
| Qwen3-8B/strict | TRELLIS edge w7 t2.0 | 1302 | 0.581 | 0.606 | 0.520 | 0.543 | 0.363 |
| Qwen3-8B/strict | TRELLIS w6 t1.6 | 1255 | 0.469 | 0.489 | 0.407 | 0.425 | 0.282 |
| Qwen3-8B/strict | LSTM t0.8 | 1366 | 0.615 | 0.642 | 0.522 | 0.545 | 0.350 |
| Qwen3-8B/strict | GPT t0.8 | 1314 | 0.597 | 0.623 | 0.523 | 0.546 | 0.361 |
| Llama-3.1-8B/v1 | TRELLIS edge w7 t2.0 | 1302 | 0.597 | 0.650 | 0.542 | 0.590 | 0.379 |
| Llama-3.1-8B/v1 | TRELLIS w6 t1.6 | 1255 | 0.551 | 0.600 | 0.506 | 0.551 | 0.350 |
| Llama-3.1-8B/v1 | LSTM t0.8 | 1366 | 0.611 | 0.666 | 0.526 | 0.573 | 0.352 |
| Llama-3.1-8B/v1 | GPT t0.8 | 1314 | 0.583 | 0.635 | 0.515 | 0.561 | 0.356 |
| Llama-3.1-8B/strict | TRELLIS edge w7 t2.0 | 1302 | 0.659 | 0.676 | 0.611 | 0.626 | 0.427 |
| Llama-3.1-8B/strict | TRELLIS w6 t1.6 | 1255 | 0.607 | 0.622 | 0.564 | 0.578 | 0.390 |
| Llama-3.1-8B/strict | LSTM t0.8 | 1366 | 0.659 | 0.676 | 0.576 | 0.591 | 0.386 |
| Llama-3.1-8B/strict | GPT t0.8 | 1314 | 0.621 | 0.637 | 0.555 | 0.569 | 0.383 |
| phi-4/v1 | TRELLIS edge w7 t2.0 | 1302 | 0.534 | 0.584 | 0.478 | 0.522 | 0.334 |
| phi-4/v1 | TRELLIS w6 t1.6 | 1255 | 0.434 | 0.475 | 0.374 | 0.409 | 0.258 |
| phi-4/v1 | LSTM t0.8 | 1366 | 0.580 | 0.635 | 0.486 | 0.532 | 0.326 |
| phi-4/v1 | GPT t0.8 | 1314 | 0.561 | 0.614 | 0.484 | 0.530 | 0.334 |
| phi-4/strict | TRELLIS edge w7 t2.0 | 1302 | 0.579 | 0.597 | 0.521 | 0.537 | 0.364 |
| phi-4/strict | TRELLIS w6 t1.6 | 1255 | 0.459 | 0.474 | 0.397 | 0.409 | 0.274 |
| phi-4/strict | LSTM t0.8 | 1366 | 0.605 | 0.624 | 0.511 | 0.527 | 0.342 |
| phi-4/strict | GPT t0.8 | 1314 | 0.582 | 0.600 | 0.510 | 0.526 | 0.352 |

Family ranking by fair/draw at n=1000:

- 3B/v1: TRELLIS edge w7 (0.337) > LSTM (0.337) > GPT (0.336) > TRELLIS w6 (0.323)
- Qwen3-8B/v1: GPT (0.297) > TRELLIS edge w7 (0.295) > LSTM (0.286) > TRELLIS w6 (0.231)
- Qwen3-8B/strict: TRELLIS edge w7 (0.363) > GPT (0.361) > LSTM (0.350) > TRELLIS w6 (0.282)
- Llama-3.1-8B/v1: TRELLIS edge w7 (0.379) > GPT (0.356) > LSTM (0.352) > TRELLIS w6 (0.350)
- Llama-3.1-8B/strict: TRELLIS edge w7 (0.427) > TRELLIS w6 (0.390) > LSTM (0.386) > GPT (0.383)
- phi-4/v1: GPT (0.334) > TRELLIS edge w7 (0.334) > LSTM (0.326) > TRELLIS w6 (0.258)
- phi-4/strict: TRELLIS edge w7 (0.364) > GPT (0.352) > LSTM (0.342) > TRELLIS w6 (0.274)

Ranking by gram|tri (normalised by ceiling) at n=1000:

- 3B/v1: LSTM (0.564) > GPT (0.547) > TRELLIS edge w7 (0.541) > TRELLIS w6 (0.523)
- Qwen3-8B/v1: GPT (0.498) > LSTM (0.494) > TRELLIS edge w7 (0.489) > TRELLIS w6 (0.387)
- Qwen3-8B/strict: GPT (0.546) > LSTM (0.545) > TRELLIS edge w7 (0.543) > TRELLIS w6 (0.425)
- Llama-3.1-8B/v1: TRELLIS edge w7 (0.590) > LSTM (0.573) > GPT (0.561) > TRELLIS w6 (0.551)
- Llama-3.1-8B/strict: TRELLIS edge w7 (0.626) > LSTM (0.591) > TRELLIS w6 (0.578) > GPT (0.569)
- phi-4/v1: LSTM (0.532) > GPT (0.530) > TRELLIS edge w7 (0.522) > TRELLIS w6 (0.409)
- phi-4/strict: TRELLIS edge w7 (0.537) > LSTM (0.527) > GPT (0.526) > TRELLIS w6 (0.409)

## n=4000: real-sentence ceiling and calibration (share judged grammatical, 600 real held-out sentences)

| judge | real (ceiling) | scrambled | content swapped | real >= .95? |
|---|---|---|---|---|
| 3B/v1 | 0.905 | 0.113 | 0.057 | no |
| Qwen3-8B/v1 | 0.865 | 0.045 | 0.033 | no |
| Qwen3-8B/strict | 0.960 | 0.057 | 0.053 | YES |
| Llama-3.1-8B/v1 | 0.923 | 0.142 | 0.065 | no |
| Llama-3.1-8B/strict | 0.978 | 0.157 | 0.108 | YES |
| phi-4/v1 | 0.918 | 0.097 | 0.053 | no |
| phi-4/strict | 0.972 | 0.070 | 0.070 | YES |

Real sentences failed by the highest-ceiling judge (Llama-3.1-8B/strict), first 15: i will be so happy if you do; do you want some; go tom go; lily and tom look up; said the car; his mom said no tim; ben and lily said; i will help you it said; tom and lily said; i see him lily; lily and sam asked; they are too hard for me too; tim said excited

### n=4000: per family (mean over seeds [13]); gram = share of distinct new judged grammatical, gram|tri = same among distinct new with an unstored trigram, norm = / real ceiling, fair = distinct gram-novel per attempted draw

| judge | family | distinct | gram | gram norm | gram\|tri | gram\|tri norm | fair/draw |
|---|---|---|---|---|---|---|---|
| 3B/v1 | TRELLIS edge w5 t2.0 | 1527 | 0.490 | 0.541 | 0.411 | 0.455 | 0.318 |
| 3B/v1 | LSTM t1.0 | 1546 | 0.642 | 0.710 | 0.546 | 0.603 | 0.383 |
| 3B/v1 | GPT t0.9 | 1467 | 0.678 | 0.749 | 0.587 | 0.648 | 0.372 |
| Qwen3-8B/v1 | TRELLIS edge w5 t2.0 | 1527 | 0.400 | 0.463 | 0.311 | 0.360 | 0.241 |
| Qwen3-8B/v1 | LSTM t1.0 | 1546 | 0.582 | 0.673 | 0.487 | 0.563 | 0.342 |
| Qwen3-8B/v1 | GPT t0.9 | 1467 | 0.628 | 0.727 | 0.539 | 0.624 | 0.342 |
| Qwen3-8B/strict | TRELLIS edge w5 t2.0 | 1527 | 0.539 | 0.561 | 0.455 | 0.474 | 0.352 |
| Qwen3-8B/strict | LSTM t1.0 | 1546 | 0.715 | 0.745 | 0.630 | 0.656 | 0.442 |
| Qwen3-8B/strict | GPT t0.9 | 1467 | 0.754 | 0.785 | 0.674 | 0.702 | 0.427 |
| Llama-3.1-8B/v1 | TRELLIS edge w5 t2.0 | 1527 | 0.538 | 0.583 | 0.459 | 0.497 | 0.355 |
| Llama-3.1-8B/v1 | LSTM t1.0 | 1546 | 0.671 | 0.727 | 0.581 | 0.629 | 0.407 |
| Llama-3.1-8B/v1 | GPT t0.9 | 1467 | 0.696 | 0.754 | 0.615 | 0.666 | 0.390 |
| Llama-3.1-8B/strict | TRELLIS edge w5 t2.0 | 1527 | 0.639 | 0.653 | 0.572 | 0.584 | 0.442 |
| Llama-3.1-8B/strict | LSTM t1.0 | 1546 | 0.735 | 0.752 | 0.655 | 0.670 | 0.460 |
| Llama-3.1-8B/strict | GPT t0.9 | 1467 | 0.776 | 0.793 | 0.698 | 0.714 | 0.443 |
| phi-4/v1 | TRELLIS edge w5 t2.0 | 1527 | 0.481 | 0.523 | 0.398 | 0.433 | 0.307 |
| phi-4/v1 | LSTM t1.0 | 1546 | 0.629 | 0.685 | 0.539 | 0.587 | 0.378 |
| phi-4/v1 | GPT t0.9 | 1467 | 0.686 | 0.747 | 0.599 | 0.652 | 0.379 |
| phi-4/strict | TRELLIS edge w5 t2.0 | 1527 | 0.539 | 0.555 | 0.451 | 0.464 | 0.349 |
| phi-4/strict | LSTM t1.0 | 1546 | 0.712 | 0.732 | 0.618 | 0.636 | 0.434 |
| phi-4/strict | GPT t0.9 | 1467 | 0.748 | 0.770 | 0.656 | 0.675 | 0.416 |

Family ranking by fair/draw at n=4000:

- 3B/v1: LSTM (0.383) > GPT (0.372) > TRELLIS edge w5 (0.318)
- Qwen3-8B/v1: LSTM (0.342) > GPT (0.342) > TRELLIS edge w5 (0.241)
- Qwen3-8B/strict: LSTM (0.442) > GPT (0.427) > TRELLIS edge w5 (0.352)
- Llama-3.1-8B/v1: LSTM (0.407) > GPT (0.390) > TRELLIS edge w5 (0.355)
- Llama-3.1-8B/strict: LSTM (0.460) > GPT (0.443) > TRELLIS edge w5 (0.442)
- phi-4/v1: GPT (0.379) > LSTM (0.378) > TRELLIS edge w5 (0.307)
- phi-4/strict: LSTM (0.434) > GPT (0.416) > TRELLIS edge w5 (0.349)

Ranking by gram|tri (normalised by ceiling) at n=4000:

- 3B/v1: GPT (0.648) > LSTM (0.603) > TRELLIS edge w5 (0.455)
- Qwen3-8B/v1: GPT (0.624) > LSTM (0.563) > TRELLIS edge w5 (0.360)
- Qwen3-8B/strict: GPT (0.702) > LSTM (0.656) > TRELLIS edge w5 (0.474)
- Llama-3.1-8B/v1: GPT (0.666) > LSTM (0.629) > TRELLIS edge w5 (0.497)
- Llama-3.1-8B/strict: GPT (0.714) > LSTM (0.670) > TRELLIS edge w5 (0.584)
- phi-4/v1: GPT (0.652) > LSTM (0.587) > TRELLIS edge w5 (0.433)
- phi-4/strict: GPT (0.675) > LSTM (0.636) > TRELLIS edge w5 (0.464)

## Agreement with the 3B v1 judge (Cohen's kappa on binary P(yes)>.5)

| judge | kappa, generated distinct new (pooled) | kappa, real+scrambled+swapped refs | agreement rate generated |
|---|---|---|---|
| Qwen3-8B/v1 | 0.652 | 0.801 | 0.825 |
| Qwen3-8B/strict | 0.627 | 0.807 | 0.816 |
| Llama-3.1-8B/v1 | 0.662 | 0.791 | 0.833 |
| Llama-3.1-8B/strict | 0.602 | 0.744 | 0.805 |
| phi-4/v1 | 0.633 | 0.774 | 0.817 |
| phi-4/strict | 0.596 | 0.790 | 0.800 |

(generated pooled: 13355 distinct sentences; refs: 3350)
