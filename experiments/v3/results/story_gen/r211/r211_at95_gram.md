# Round 211: the gram >= .95 operating point at n=1000 (strict judges)

Code: `experiments/v3/diag/r211_collect_judge.py` (collect + judge), `r211_analyse.py` (this file); grid `logs/v3_r211_grid.sh`; per-sentence flags `r211/r211_sets.json`, scores `r211/r211_scores.json`, curves `r211/r211_curves.tsv`.

* Qwen3-8B/strict: real-sentence ceiling 0.958 (1200 held-out refs), scrambled false-pass 0.058, content-swapped 0.060.
* phi-4/strict: real-sentence ceiling 0.970 (1200 held-out refs), scrambled false-pass 0.068, content-swapped 0.075.


## Curves under Qwen3-8B/strict (mean of seeds 13/17/19/23; gram = share of distinct new judged grammatical; tri|d = share of distinct new with an unstored trigram; dist = distinct new per attempted draw; fair = distinct new, grammatical, unstored trigram per attempted draw)

| family | temp | gram | tri\|d | dist/draw | fair | gram\|tri | distinct/seed |
|---|---|---|---|---|---|---|---|
| TRELLIS edge w5 | 0.10 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0 (below MIN_D, not used) |
| TRELLIS edge w5 | 0.15 | 0.750 | 0.000 | 0.001 | 0.000 | nan | 1 (below MIN_D, not used) |
| TRELLIS edge w5 | 0.20 | 0.850 | 0.250 | 0.002 | 0.000 | 0.559 | 4 (below MIN_D, not used) |
| TRELLIS edge w5 | 0.30 | 0.807 | 0.404 | 0.028 | 0.007 | 0.572 | 43 |
| TRELLIS edge w5 | 0.50 | 0.767 | 0.553 | 0.239 | 0.081 | 0.616 | 362 |
| TRELLIS edge w5 | 0.70 | 0.748 | 0.626 | 0.466 | 0.184 | 0.629 | 708 |
| TRELLIS edge w5 | 1.00 | 0.704 | 0.688 | 0.653 | 0.268 | 0.597 | 992 |
| TRELLIS edge w5 | 1.30 | 0.676 | 0.737 | 0.744 | 0.320 | 0.583 | 1131 |
| TRELLIS edge w5 | 1.60 | 0.643 | 0.779 | 0.786 | 0.344 | 0.562 | 1194 |
| TRELLIS edge w5 | 2.00 | 0.594 | 0.821 | 0.819 | 0.352 | 0.523 | 1244 |
| TRELLIS edge w6 | 1.30 | 0.676 | 0.750 | 0.736 | 0.324 | 0.587 | 1118 |
| TRELLIS edge w6 | 1.60 | 0.645 | 0.793 | 0.776 | 0.352 | 0.571 | 1180 |
| TRELLIS edge w6 | 2.00 | 0.601 | 0.828 | 0.814 | 0.360 | 0.535 | 1236 |
| TRELLIS edge w6 | 2.40 | 0.557 | 0.857 | 0.835 | 0.357 | 0.499 | 1268 |
| TRELLIS edge w6 | 2.80 | 0.525 | 0.877 | 0.852 | 0.352 | 0.472 | 1294 |
| TRELLIS edge w6 | 3.50 | 0.478 | 0.903 | 0.874 | 0.340 | 0.430 | 1328 |
| TRELLIS edge w7 | 0.10 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0 (below MIN_D, not used) |
| TRELLIS edge w7 | 0.15 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0 (below MIN_D, not used) |
| TRELLIS edge w7 | 0.20 | 0.325 | 0.425 | 0.002 | 0.000 | 0.229 | 2 (below MIN_D, not used) |
| TRELLIS edge w7 | 0.30 | 0.759 | 0.528 | 0.024 | 0.007 | 0.592 | 36 |
| TRELLIS edge w7 | 0.50 | 0.764 | 0.632 | 0.225 | 0.093 | 0.655 | 342 |
| TRELLIS edge w7 | 0.70 | 0.744 | 0.665 | 0.473 | 0.202 | 0.640 | 719 |
| TRELLIS edge w7 | 1.00 | 0.709 | 0.726 | 0.663 | 0.299 | 0.621 | 1007 |
| TRELLIS edge w7 | 1.30 | 0.678 | 0.756 | 0.740 | 0.334 | 0.596 | 1125 |
| TRELLIS edge w7 | 1.60 | 0.644 | 0.791 | 0.779 | 0.352 | 0.572 | 1184 |
| TRELLIS edge w7 | 2.00 | 0.596 | 0.836 | 0.814 | 0.361 | 0.531 | 1238 |
| TRELLIS edge w7 | 2.40 | 0.559 | 0.860 | 0.836 | 0.362 | 0.503 | 1272 |
| TRELLIS edge w7 | 2.80 | 0.523 | 0.881 | 0.855 | 0.355 | 0.471 | 1299 |
| TRELLIS edge w7 | 3.50 | 0.467 | 0.913 | 0.882 | 0.343 | 0.426 | 1340 |
| TRELLIS edge w8 | 2.40 | 0.567 | 0.852 | 0.830 | 0.358 | 0.506 | 1261 |
| TRELLIS w2 | 1.00 | 0.714 | 0.368 | 0.853 | 0.168 | 0.536 | 1296 |
| TRELLIS w3 | 0.50 | 0.914 | 0.155 | 0.327 | 0.038 | 0.742 | 496 |
| TRELLIS w3 | 0.70 | 0.867 | 0.261 | 0.538 | 0.099 | 0.702 | 817 |
| TRELLIS w3 | 1.00 | 0.807 | 0.392 | 0.720 | 0.181 | 0.642 | 1094 |
| TRELLIS w4 | 0.50 | 0.790 | 0.439 | 0.251 | 0.066 | 0.598 | 380 |
| TRELLIS w4 | 0.70 | 0.770 | 0.512 | 0.473 | 0.148 | 0.609 | 720 |
| TRELLIS w4 | 1.00 | 0.722 | 0.595 | 0.669 | 0.233 | 0.584 | 1017 |
| TRELLIS w4 | 1.30 | 0.681 | 0.664 | 0.758 | 0.286 | 0.567 | 1152 |
| TRELLIS w5 | 0.50 | 0.618 | 0.656 | 0.292 | 0.086 | 0.450 | 444 |
| TRELLIS w5 | 0.70 | 0.607 | 0.683 | 0.532 | 0.167 | 0.460 | 808 |
| TRELLIS w5 | 1.00 | 0.583 | 0.738 | 0.696 | 0.238 | 0.464 | 1059 |
| TRELLIS w5 | 1.30 | 0.548 | 0.786 | 0.774 | 0.273 | 0.449 | 1176 |
| TRELLIS w5 | 1.60 | 0.523 | 0.821 | 0.811 | 0.292 | 0.438 | 1232 |
| TRELLIS w5 | 2.00 | 0.493 | 0.863 | 0.843 | 0.312 | 0.429 | 1281 |
| TRELLIS w6 | 1.30 | 0.509 | 0.832 | 0.761 | 0.271 | 0.429 | 1156 |
| TRELLIS w6 | 1.60 | 0.476 | 0.855 | 0.800 | 0.277 | 0.405 | 1216 |
| TRELLIS w6 | 2.00 | 0.437 | 0.881 | 0.828 | 0.274 | 0.375 | 1258 |
| TRELLIS w7 | 1.30 | 0.514 | 0.817 | 0.779 | 0.275 | 0.433 | 1183 |
| TRELLIS w7 | 1.60 | 0.481 | 0.846 | 0.802 | 0.276 | 0.407 | 1220 |
| TRELLIS w7 | 2.00 | 0.433 | 0.884 | 0.838 | 0.279 | 0.377 | 1274 |
| GPT | 0.15 | 0.844 | 0.654 | 0.025 | 0.012 | 0.746 | 38 |
| GPT | 0.20 | 0.852 | 0.630 | 0.056 | 0.027 | 0.775 | 86 |
| GPT | 0.30 | 0.809 | 0.648 | 0.137 | 0.064 | 0.721 | 208 |
| GPT | 0.40 | 0.767 | 0.682 | 0.277 | 0.129 | 0.682 | 422 |
| GPT | 0.50 | 0.747 | 0.699 | 0.439 | 0.202 | 0.659 | 667 |
| GPT | 0.60 | 0.699 | 0.747 | 0.598 | 0.277 | 0.620 | 908 |
| GPT | 0.70 | 0.643 | 0.799 | 0.735 | 0.335 | 0.571 | 1116 |
| GPT | 0.80 | 0.576 | 0.840 | 0.845 | 0.360 | 0.507 | 1283 |
| GPT | 0.90 | 0.491 | 0.879 | 0.911 | 0.345 | 0.430 | 1384 |
| GPT | 1.00 | 0.383 | 0.909 | 0.953 | 0.284 | 0.327 | 1449 |
| GPT | 1.20 | 0.245 | 0.954 | 0.988 | 0.199 | 0.211 | 1502 |
| LSTM | 0.15 | 0.888 | 0.331 | 0.022 | 0.007 | 0.903 | 33 |
| LSTM | 0.20 | 0.914 | 0.419 | 0.047 | 0.018 | 0.896 | 71 |
| LSTM | 0.30 | 0.906 | 0.467 | 0.127 | 0.051 | 0.867 | 192 |
| LSTM | 0.40 | 0.880 | 0.514 | 0.271 | 0.115 | 0.821 | 411 |
| LSTM | 0.50 | 0.822 | 0.598 | 0.450 | 0.199 | 0.740 | 684 |
| LSTM | 0.60 | 0.783 | 0.655 | 0.619 | 0.285 | 0.703 | 940 |
| LSTM | 0.70 | 0.716 | 0.695 | 0.752 | 0.326 | 0.623 | 1144 |
| LSTM | 0.80 | 0.629 | 0.779 | 0.866 | 0.366 | 0.543 | 1316 |
| LSTM | 0.90 | 0.539 | 0.832 | 0.929 | 0.356 | 0.460 | 1413 |
| LSTM | 1.00 | 0.461 | 0.864 | 0.953 | 0.318 | 0.386 | 1448 |
| LSTM | 1.20 | 0.278 | 0.942 | 0.988 | 0.220 | 0.237 | 1502 |
| ngram 3 | 0.70 | 0.811 | 0.000 | 0.653 | 0.000 | nan | 992 |
| ngram 3 | 1.00 | 0.781 | 0.000 | 0.765 | 0.000 | nan | 1163 |
| ngram 4 | 0.70 | 0.919 | 0.000 | 0.402 | 0.000 | nan | 610 |
| ngram 4 | 1.00 | 0.910 | 0.000 | 0.463 | 0.000 | nan | 703 |
| ngram 5 | 0.70 | 0.926 | 0.000 | 0.125 | 0.000 | nan | 190 |
| ngram 5 | 1.00 | 0.924 | 0.000 | 0.130 | 0.000 | nan | 198 |

## Curves under phi-4/strict (mean of seeds 13/17/19/23; gram = share of distinct new judged grammatical; tri|d = share of distinct new with an unstored trigram; dist = distinct new per attempted draw; fair = distinct new, grammatical, unstored trigram per attempted draw)

| family | temp | gram | tri\|d | dist/draw | fair | gram\|tri | distinct/seed |
|---|---|---|---|---|---|---|---|
| TRELLIS edge w5 | 0.10 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0 (below MIN_D, not used) |
| TRELLIS edge w5 | 0.15 | 0.500 | 0.000 | 0.001 | 0.000 | nan | 1 (below MIN_D, not used) |
| TRELLIS edge w5 | 0.20 | 0.600 | 0.250 | 0.002 | 0.000 | 0.559 | 4 (below MIN_D, not used) |
| TRELLIS edge w5 | 0.30 | 0.802 | 0.404 | 0.028 | 0.007 | 0.572 | 43 |
| TRELLIS edge w5 | 0.50 | 0.759 | 0.553 | 0.239 | 0.081 | 0.616 | 362 |
| TRELLIS edge w5 | 0.70 | 0.733 | 0.626 | 0.466 | 0.181 | 0.619 | 708 |
| TRELLIS edge w5 | 1.00 | 0.700 | 0.688 | 0.653 | 0.269 | 0.598 | 992 |
| TRELLIS edge w5 | 1.30 | 0.677 | 0.737 | 0.744 | 0.323 | 0.588 | 1131 |
| TRELLIS edge w5 | 1.60 | 0.641 | 0.779 | 0.786 | 0.343 | 0.561 | 1194 |
| TRELLIS edge w5 | 2.00 | 0.596 | 0.821 | 0.819 | 0.355 | 0.528 | 1244 |
| TRELLIS edge w6 | 1.30 | 0.668 | 0.750 | 0.736 | 0.321 | 0.581 | 1118 |
| TRELLIS edge w6 | 1.60 | 0.643 | 0.793 | 0.776 | 0.353 | 0.573 | 1180 |
| TRELLIS edge w6 | 2.00 | 0.593 | 0.828 | 0.814 | 0.357 | 0.530 | 1236 |
| TRELLIS edge w6 | 2.40 | 0.550 | 0.857 | 0.835 | 0.352 | 0.493 | 1268 |
| TRELLIS edge w6 | 2.80 | 0.522 | 0.877 | 0.852 | 0.349 | 0.468 | 1294 |
| TRELLIS edge w6 | 3.50 | 0.470 | 0.903 | 0.874 | 0.333 | 0.422 | 1328 |
| TRELLIS edge w7 | 0.10 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0 (below MIN_D, not used) |
| TRELLIS edge w7 | 0.15 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0 (below MIN_D, not used) |
| TRELLIS edge w7 | 0.20 | 0.325 | 0.425 | 0.002 | 0.000 | 0.229 | 2 (below MIN_D, not used) |
| TRELLIS edge w7 | 0.30 | 0.753 | 0.528 | 0.024 | 0.007 | 0.591 | 36 |
| TRELLIS edge w7 | 0.50 | 0.749 | 0.632 | 0.225 | 0.091 | 0.640 | 342 |
| TRELLIS edge w7 | 0.70 | 0.735 | 0.665 | 0.473 | 0.201 | 0.637 | 719 |
| TRELLIS edge w7 | 1.00 | 0.704 | 0.726 | 0.663 | 0.298 | 0.619 | 1007 |
| TRELLIS edge w7 | 1.30 | 0.677 | 0.756 | 0.740 | 0.335 | 0.599 | 1125 |
| TRELLIS edge w7 | 1.60 | 0.644 | 0.791 | 0.779 | 0.353 | 0.573 | 1184 |
| TRELLIS edge w7 | 2.00 | 0.593 | 0.836 | 0.814 | 0.362 | 0.532 | 1238 |
| TRELLIS edge w7 | 2.40 | 0.555 | 0.860 | 0.836 | 0.357 | 0.497 | 1272 |
| TRELLIS edge w7 | 2.80 | 0.523 | 0.881 | 0.855 | 0.356 | 0.472 | 1299 |
| TRELLIS edge w7 | 3.50 | 0.464 | 0.913 | 0.882 | 0.340 | 0.422 | 1340 |
| TRELLIS edge w8 | 2.40 | 0.563 | 0.852 | 0.830 | 0.354 | 0.501 | 1261 |
| TRELLIS w2 | 1.00 | 0.697 | 0.368 | 0.853 | 0.164 | 0.522 | 1296 |
| TRELLIS w3 | 0.50 | 0.878 | 0.155 | 0.327 | 0.038 | 0.749 | 496 |
| TRELLIS w3 | 0.70 | 0.834 | 0.261 | 0.538 | 0.095 | 0.675 | 817 |
| TRELLIS w3 | 1.00 | 0.789 | 0.392 | 0.720 | 0.181 | 0.641 | 1094 |
| TRELLIS w4 | 0.50 | 0.757 | 0.439 | 0.251 | 0.063 | 0.574 | 380 |
| TRELLIS w4 | 0.70 | 0.741 | 0.512 | 0.473 | 0.141 | 0.580 | 720 |
| TRELLIS w4 | 1.00 | 0.714 | 0.595 | 0.669 | 0.231 | 0.580 | 1017 |
| TRELLIS w4 | 1.30 | 0.672 | 0.664 | 0.758 | 0.281 | 0.559 | 1152 |
| TRELLIS w5 | 0.50 | 0.609 | 0.656 | 0.292 | 0.087 | 0.455 | 444 |
| TRELLIS w5 | 0.70 | 0.588 | 0.683 | 0.532 | 0.163 | 0.447 | 808 |
| TRELLIS w5 | 1.00 | 0.574 | 0.738 | 0.696 | 0.237 | 0.461 | 1059 |
| TRELLIS w5 | 1.30 | 0.543 | 0.786 | 0.774 | 0.271 | 0.446 | 1176 |
| TRELLIS w5 | 1.60 | 0.515 | 0.821 | 0.811 | 0.287 | 0.432 | 1232 |
| TRELLIS w5 | 2.00 | 0.478 | 0.863 | 0.843 | 0.301 | 0.413 | 1281 |
| TRELLIS w6 | 1.30 | 0.500 | 0.832 | 0.761 | 0.265 | 0.419 | 1156 |
| TRELLIS w6 | 1.60 | 0.462 | 0.855 | 0.800 | 0.267 | 0.391 | 1216 |
| TRELLIS w6 | 2.00 | 0.431 | 0.881 | 0.828 | 0.270 | 0.370 | 1258 |
| TRELLIS w7 | 1.30 | 0.499 | 0.817 | 0.779 | 0.263 | 0.413 | 1183 |
| TRELLIS w7 | 1.60 | 0.470 | 0.846 | 0.802 | 0.269 | 0.397 | 1220 |
| TRELLIS w7 | 2.00 | 0.427 | 0.884 | 0.838 | 0.274 | 0.370 | 1274 |
| GPT | 0.15 | 0.820 | 0.654 | 0.025 | 0.011 | 0.706 | 38 |
| GPT | 0.20 | 0.850 | 0.630 | 0.056 | 0.027 | 0.756 | 86 |
| GPT | 0.30 | 0.821 | 0.648 | 0.137 | 0.066 | 0.737 | 208 |
| GPT | 0.40 | 0.766 | 0.682 | 0.277 | 0.128 | 0.677 | 422 |
| GPT | 0.50 | 0.750 | 0.699 | 0.439 | 0.203 | 0.661 | 667 |
| GPT | 0.60 | 0.692 | 0.747 | 0.598 | 0.271 | 0.607 | 908 |
| GPT | 0.70 | 0.632 | 0.799 | 0.735 | 0.326 | 0.556 | 1116 |
| GPT | 0.80 | 0.567 | 0.840 | 0.845 | 0.353 | 0.497 | 1283 |
| GPT | 0.90 | 0.481 | 0.879 | 0.911 | 0.336 | 0.419 | 1384 |
| GPT | 1.00 | 0.379 | 0.909 | 0.953 | 0.281 | 0.324 | 1449 |
| GPT | 1.20 | 0.238 | 0.954 | 0.988 | 0.193 | 0.204 | 1502 |
| LSTM | 0.15 | 0.926 | 0.331 | 0.022 | 0.006 | 0.878 | 33 |
| LSTM | 0.20 | 0.909 | 0.419 | 0.047 | 0.017 | 0.869 | 71 |
| LSTM | 0.30 | 0.906 | 0.467 | 0.127 | 0.051 | 0.867 | 192 |
| LSTM | 0.40 | 0.879 | 0.514 | 0.271 | 0.115 | 0.828 | 411 |
| LSTM | 0.50 | 0.820 | 0.598 | 0.450 | 0.200 | 0.742 | 684 |
| LSTM | 0.60 | 0.782 | 0.655 | 0.619 | 0.285 | 0.704 | 940 |
| LSTM | 0.70 | 0.709 | 0.695 | 0.752 | 0.321 | 0.615 | 1144 |
| LSTM | 0.80 | 0.625 | 0.779 | 0.866 | 0.364 | 0.539 | 1316 |
| LSTM | 0.90 | 0.535 | 0.832 | 0.929 | 0.352 | 0.456 | 1413 |
| LSTM | 1.00 | 0.456 | 0.864 | 0.953 | 0.316 | 0.383 | 1448 |
| LSTM | 1.20 | 0.280 | 0.942 | 0.988 | 0.224 | 0.240 | 1502 |
| ngram 3 | 0.70 | 0.789 | 0.000 | 0.653 | 0.000 | nan | 992 |
| ngram 3 | 1.00 | 0.762 | 0.000 | 0.765 | 0.000 | nan | 1163 |
| ngram 4 | 0.70 | 0.878 | 0.000 | 0.402 | 0.000 | nan | 610 |
| ngram 4 | 1.00 | 0.870 | 0.000 | 0.463 | 0.000 | nan | 703 |
| ngram 5 | 0.70 | 0.898 | 0.000 | 0.125 | 0.000 | nan | 190 |
| ngram 5 | 1.00 | 0.899 | 0.000 | 0.130 | 0.000 | nan | 198 |

## Maximum tri|d and maximum fair count per family at a gram constraint (mean of 4 seeds; temperature in brackets; '-' = gram constraint not met by any measured or interpolated point of the curve, no extrapolation)


### Qwen3-8B/strict, gram >= .95

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.807 (t0.30) | - | - | - |
| TRELLIS edge w6 | 0.676 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.759 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.567 (t2.40) | - | - | - |
| TRELLIS w2 | 0.714 (t1.00) | - | - | - |
| TRELLIS w3 | 0.914 (t0.50) | - | - | - |
| TRELLIS w4 | 0.790 (t0.50) | - | - | - |
| TRELLIS w5 | 0.618 (t0.50) | - | - | - |
| TRELLIS w6 | 0.509 (t1.30) | - | - | - |
| TRELLIS w7 | 0.514 (t1.30) | - | - | - |
| GPT | 0.844 (t0.15) | - | - | - |
| LSTM | 0.888 (t0.15) | - | - | - |
| ngram 3 | 0.811 (t0.70) | - | - | - |
| ngram 4 | 0.919 (t0.70) | - | - | - |
| ngram 5 | 0.926 (t0.70) | - | - | - |

### Qwen3-8B/strict, gram >= .90

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.807 (t0.30) | - | - | - |
| TRELLIS edge w6 | 0.676 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.759 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.567 (t2.40) | - | - | - |
| TRELLIS w2 | 0.714 (t1.00) | - | - | - |
| TRELLIS w3 | 0.914 (t0.50) | 0.187 (t0.56, 0.900) | 0.056 (t0.56, 0.900) | 0.391 |
| TRELLIS w4 | 0.790 (t0.50) | - | - | - |
| TRELLIS w5 | 0.618 (t0.50) | - | - | - |
| TRELLIS w6 | 0.509 (t1.30) | - | - | - |
| TRELLIS w7 | 0.514 (t1.30) | - | - | - |
| GPT | 0.844 (t0.15) | - | - | - |
| LSTM | 0.888 (t0.15) | 0.478 (t0.32, 0.900) | 0.066 (t0.32, 0.900) | 0.159 |
| ngram 3 | 0.811 (t0.70) | - | - | - |
| ngram 4 | 0.919 (t0.70) | 0.000 (t0.70, 0.919) | 0.000 (t0.70, 0.919) | 0.402 |
| ngram 5 | 0.926 (t0.70) | 0.000 (t0.70, 0.926) | 0.000 (t0.70, 0.926) | 0.125 |

### Qwen3-8B/strict, gram >= .85

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.807 (t0.30) | - | - | - |
| TRELLIS edge w6 | 0.676 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.759 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.567 (t2.40) | - | - | - |
| TRELLIS w2 | 0.714 (t1.00) | - | - | - |
| TRELLIS w3 | 0.914 (t0.50) | 0.297 (t0.78, 0.850) | 0.122 (t0.78, 0.850) | 0.589 |
| TRELLIS w4 | 0.790 (t0.50) | - | - | - |
| TRELLIS w5 | 0.618 (t0.50) | - | - | - |
| TRELLIS w6 | 0.509 (t1.30) | - | - | - |
| TRELLIS w7 | 0.514 (t1.30) | - | - | - |
| GPT | 0.844 (t0.15) | 0.636 (t0.19, 0.850) | 0.029 (t0.20, 0.850) | 0.060 |
| LSTM | 0.888 (t0.15) | 0.557 (t0.45, 0.850) | 0.158 (t0.45, 0.850) | 0.363 |
| ngram 3 | 0.811 (t0.70) | - | - | - |
| ngram 4 | 0.919 (t0.70) | 0.000 (t0.70, 0.919) | 0.000 (t0.70, 0.919) | 0.402 |
| ngram 5 | 0.926 (t0.70) | 0.000 (t0.70, 0.926) | 0.000 (t0.70, 0.926) | 0.125 |

### Qwen3-8B/strict, gram >= .80

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.807 (t0.30) | 0.429 (t0.33, 0.800) | 0.019 (t0.33, 0.800) | 0.064 |
| TRELLIS edge w6 | 0.676 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.759 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.567 (t2.40) | - | - | - |
| TRELLIS w2 | 0.714 (t1.00) | - | - | - |
| TRELLIS w3 | 0.914 (t0.50) | 0.392 (t1.00, 0.807) | 0.181 (t1.00, 0.807) | 0.720 |
| TRELLIS w4 | 0.790 (t0.50) | - | - | - |
| TRELLIS w5 | 0.618 (t0.50) | - | - | - |
| TRELLIS w6 | 0.509 (t1.30) | - | - | - |
| TRELLIS w7 | 0.514 (t1.30) | - | - | - |
| GPT | 0.844 (t0.15) | 0.656 (t0.32, 0.800) | 0.078 (t0.32, 0.800) | 0.168 |
| LSTM | 0.888 (t0.15) | 0.630 (t0.56, 0.800) | 0.247 (t0.56, 0.800) | 0.545 |
| ngram 3 | 0.811 (t0.70) | 0.000 (t0.70, 0.811) | 0.000 (t0.70, 0.811) | 0.653 |
| ngram 4 | 0.919 (t0.70) | 0.000 (t0.70, 0.919) | 0.000 (t0.70, 0.919) | 0.402 |
| ngram 5 | 0.926 (t0.70) | 0.000 (t0.70, 0.926) | 0.000 (t0.70, 0.926) | 0.125 |

### Qwen3-8B/strict, gram >= .75

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.807 (t0.30) | 0.618 (t0.68, 0.750) | 0.171 (t0.68, 0.750) | 0.439 |
| TRELLIS edge w6 | 0.676 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.759 (t0.30) | 0.655 (t0.64, 0.750) | 0.170 (t0.64, 0.750) | 0.400 |
| TRELLIS edge w8 | 0.567 (t2.40) | - | - | - |
| TRELLIS w2 | 0.714 (t1.00) | - | - | - |
| TRELLIS w3 | 0.914 (t0.50) | 0.392 (t1.00, 0.807) | 0.181 (t1.00, 0.807) | 0.720 |
| TRELLIS w4 | 0.790 (t0.50) | 0.547 (t0.82, 0.750) | 0.183 (t0.82, 0.750) | 0.555 |
| TRELLIS w5 | 0.618 (t0.50) | - | - | - |
| TRELLIS w6 | 0.509 (t1.30) | - | - | - |
| TRELLIS w7 | 0.514 (t1.30) | - | - | - |
| GPT | 0.844 (t0.15) | 0.697 (t0.48, 0.750) | 0.190 (t0.48, 0.750) | 0.413 |
| LSTM | 0.888 (t0.15) | 0.675 (t0.65, 0.750) | 0.305 (t0.65, 0.750) | 0.684 |
| ngram 3 | 0.811 (t0.70) | 0.000 (t0.70, 0.811) | 0.000 (t0.70, 0.811) | 0.653 |
| ngram 4 | 0.919 (t0.70) | 0.000 (t0.70, 0.919) | 0.000 (t0.70, 0.919) | 0.402 |
| ngram 5 | 0.926 (t0.70) | 0.000 (t0.70, 0.926) | 0.000 (t0.70, 0.926) | 0.125 |

### Qwen3-8B/strict, gram >= .70

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.807 (t0.30) | 0.695 (t1.04, 0.700) | 0.275 (t1.04, 0.700) | 0.665 |
| TRELLIS edge w6 | 0.676 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.759 (t0.30) | 0.735 (t1.09, 0.700) | 0.309 (t1.09, 0.700) | 0.685 |
| TRELLIS edge w8 | 0.567 (t2.40) | - | - | - |
| TRELLIS w2 | 0.714 (t1.00) | 0.368 (t1.00, 0.714) | 0.168 (t1.00, 0.714) | 0.853 |
| TRELLIS w3 | 0.914 (t0.50) | 0.392 (t1.00, 0.807) | 0.181 (t1.00, 0.807) | 0.720 |
| TRELLIS w4 | 0.790 (t0.50) | 0.632 (t1.16, 0.700) | 0.261 (t1.16, 0.700) | 0.717 |
| TRELLIS w5 | 0.618 (t0.50) | - | - | - |
| TRELLIS w6 | 0.509 (t1.30) | - | - | - |
| TRELLIS w7 | 0.514 (t1.30) | - | - | - |
| GPT | 0.844 (t0.15) | 0.746 (t0.60, 0.700) | 0.275 (t0.60, 0.700) | 0.594 |
| LSTM | 0.888 (t0.15) | 0.710 (t0.72, 0.700) | 0.333 (t0.72, 0.700) | 0.773 |
| ngram 3 | 0.811 (t0.70) | 0.000 (t0.70, 0.811) | 0.000 (t0.70, 0.811) | 0.653 |
| ngram 4 | 0.919 (t0.70) | 0.000 (t0.70, 0.919) | 0.000 (t0.70, 0.919) | 0.402 |
| ngram 5 | 0.926 (t0.70) | 0.000 (t0.70, 0.926) | 0.000 (t0.70, 0.926) | 0.125 |

### Qwen3-8B/strict, gram >= 0.95 x ceiling (0.910)

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.807 (t0.30) | - | - | - |
| TRELLIS edge w6 | 0.676 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.759 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.567 (t2.40) | - | - | - |
| TRELLIS w2 | 0.714 (t1.00) | - | - | - |
| TRELLIS w3 | 0.914 (t0.50) | 0.164 (t0.52, 0.910) | 0.043 (t0.52, 0.910) | 0.345 |
| TRELLIS w4 | 0.790 (t0.50) | - | - | - |
| TRELLIS w5 | 0.618 (t0.50) | - | - | - |
| TRELLIS w6 | 0.509 (t1.30) | - | - | - |
| TRELLIS w7 | 0.514 (t1.30) | - | - | - |
| GPT | 0.844 (t0.15) | - | - | - |
| LSTM | 0.888 (t0.15) | 0.442 (t0.25, 0.910) | 0.034 (t0.25, 0.910) | 0.085 |
| ngram 3 | 0.811 (t0.70) | - | - | - |
| ngram 4 | 0.919 (t0.70) | 0.000 (t0.70, 0.919) | 0.000 (t0.70, 0.919) | 0.402 |
| ngram 5 | 0.926 (t0.70) | 0.000 (t0.70, 0.926) | 0.000 (t0.70, 0.926) | 0.125 |

### phi-4/strict, gram >= .95

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.802 (t0.30) | - | - | - |
| TRELLIS edge w6 | 0.668 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.753 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.563 (t2.40) | - | - | - |
| TRELLIS w2 | 0.697 (t1.00) | - | - | - |
| TRELLIS w3 | 0.878 (t0.50) | - | - | - |
| TRELLIS w4 | 0.757 (t0.50) | - | - | - |
| TRELLIS w5 | 0.609 (t0.50) | - | - | - |
| TRELLIS w6 | 0.500 (t1.30) | - | - | - |
| TRELLIS w7 | 0.499 (t1.30) | - | - | - |
| GPT | 0.820 (t0.15) | - | - | - |
| LSTM | 0.926 (t0.15) | - | - | - |
| ngram 3 | 0.789 (t0.70) | - | - | - |
| ngram 4 | 0.878 (t0.70) | - | - | - |
| ngram 5 | 0.898 (t0.70) | - | - | - |

### phi-4/strict, gram >= .90

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.802 (t0.30) | - | - | - |
| TRELLIS edge w6 | 0.668 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.753 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.563 (t2.40) | - | - | - |
| TRELLIS w2 | 0.697 (t1.00) | - | - | - |
| TRELLIS w3 | 0.878 (t0.50) | - | - | - |
| TRELLIS w4 | 0.757 (t0.50) | - | - | - |
| TRELLIS w5 | 0.609 (t0.50) | - | - | - |
| TRELLIS w6 | 0.500 (t1.30) | - | - | - |
| TRELLIS w7 | 0.499 (t1.30) | - | - | - |
| GPT | 0.820 (t0.15) | - | - | - |
| LSTM | 0.926 (t0.15) | 0.478 (t0.32, 0.900) | 0.066 (t0.32, 0.900) | 0.161 |
| ngram 3 | 0.789 (t0.70) | - | - | - |
| ngram 4 | 0.878 (t0.70) | - | - | - |
| ngram 5 | 0.898 (t0.70) | - | - | - |

### phi-4/strict, gram >= .85

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.802 (t0.30) | - | - | - |
| TRELLIS edge w6 | 0.668 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.753 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.563 (t2.40) | - | - | - |
| TRELLIS w2 | 0.697 (t1.00) | - | - | - |
| TRELLIS w3 | 0.878 (t0.50) | 0.222 (t0.63, 0.850) | 0.074 (t0.63, 0.850) | 0.460 |
| TRELLIS w4 | 0.757 (t0.50) | - | - | - |
| TRELLIS w5 | 0.609 (t0.50) | - | - | - |
| TRELLIS w6 | 0.500 (t1.30) | - | - | - |
| TRELLIS w7 | 0.499 (t1.30) | - | - | - |
| GPT | 0.820 (t0.15) | 0.631 (t0.20, 0.850) | 0.027 (t0.20, 0.850) | 0.056 |
| LSTM | 0.926 (t0.15) | 0.556 (t0.45, 0.850) | 0.157 (t0.45, 0.850) | 0.360 |
| ngram 3 | 0.789 (t0.70) | - | - | - |
| ngram 4 | 0.878 (t0.70) | 0.000 (t0.70, 0.878) | 0.000 (t0.70, 0.878) | 0.402 |
| ngram 5 | 0.898 (t0.70) | 0.000 (t0.70, 0.898) | 0.000 (t0.70, 0.898) | 0.125 |

### phi-4/strict, gram >= .80

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.802 (t0.30) | 0.411 (t0.31, 0.800) | 0.010 (t0.31, 0.800) | 0.038 |
| TRELLIS edge w6 | 0.668 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.753 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.563 (t2.40) | - | - | - |
| TRELLIS w2 | 0.697 (t1.00) | - | - | - |
| TRELLIS w3 | 0.878 (t0.50) | 0.359 (t0.93, 0.800) | 0.159 (t0.93, 0.800) | 0.675 |
| TRELLIS w4 | 0.757 (t0.50) | - | - | - |
| TRELLIS w5 | 0.609 (t0.50) | - | - | - |
| TRELLIS w6 | 0.500 (t1.30) | - | - | - |
| TRELLIS w7 | 0.499 (t1.30) | - | - | - |
| GPT | 0.820 (t0.15) | 0.661 (t0.34, 0.800) | 0.089 (t0.34, 0.800) | 0.190 |
| LSTM | 0.926 (t0.15) | 0.628 (t0.55, 0.800) | 0.245 (t0.55, 0.800) | 0.540 |
| ngram 3 | 0.789 (t0.70) | - | - | - |
| ngram 4 | 0.878 (t0.70) | 0.000 (t0.70, 0.878) | 0.000 (t0.70, 0.878) | 0.402 |
| ngram 5 | 0.898 (t0.70) | 0.000 (t0.70, 0.898) | 0.000 (t0.70, 0.898) | 0.125 |

### phi-4/strict, gram >= .75

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.802 (t0.30) | 0.579 (t0.57, 0.750) | 0.116 (t0.57, 0.750) | 0.318 |
| TRELLIS edge w6 | 0.668 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.753 (t0.30) | 0.616 (t0.47, 0.750) | 0.078 (t0.47, 0.750) | 0.194 |
| TRELLIS edge w8 | 0.563 (t2.40) | - | - | - |
| TRELLIS w2 | 0.697 (t1.00) | - | - | - |
| TRELLIS w3 | 0.878 (t0.50) | 0.392 (t1.00, 0.789) | 0.181 (t1.00, 0.789) | 0.720 |
| TRELLIS w4 | 0.757 (t0.50) | 0.471 (t0.59, 0.750) | 0.097 (t0.59, 0.750) | 0.349 |
| TRELLIS w5 | 0.609 (t0.50) | - | - | - |
| TRELLIS w6 | 0.500 (t1.30) | - | - | - |
| TRELLIS w7 | 0.499 (t1.30) | - | - | - |
| GPT | 0.820 (t0.15) | 0.700 (t0.50, 0.750) | 0.204 (t0.50, 0.750) | 0.440 |
| LSTM | 0.926 (t0.15) | 0.672 (t0.64, 0.750) | 0.301 (t0.64, 0.750) | 0.677 |
| ngram 3 | 0.789 (t0.70) | 0.000 (t0.70, 0.789) | 0.000 (t0.70, 0.789) | 0.653 |
| ngram 4 | 0.878 (t0.70) | 0.000 (t0.70, 0.878) | 0.000 (t0.70, 0.878) | 0.402 |
| ngram 5 | 0.898 (t0.70) | 0.000 (t0.70, 0.898) | 0.000 (t0.70, 0.898) | 0.125 |

### phi-4/strict, gram >= .70

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.802 (t0.30) | 0.687 (t1.00, 0.700) | 0.268 (t1.00, 0.700) | 0.651 |
| TRELLIS edge w6 | 0.668 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.753 (t0.30) | 0.731 (t1.05, 0.700) | 0.304 (t1.05, 0.700) | 0.675 |
| TRELLIS edge w8 | 0.563 (t2.40) | - | - | - |
| TRELLIS w2 | 0.697 (t1.00) | - | - | - |
| TRELLIS w3 | 0.878 (t0.50) | 0.392 (t1.00, 0.789) | 0.181 (t1.00, 0.789) | 0.720 |
| TRELLIS w4 | 0.757 (t0.50) | 0.618 (t1.10, 0.700) | 0.247 (t1.10, 0.700) | 0.698 |
| TRELLIS w5 | 0.609 (t0.50) | - | - | - |
| TRELLIS w6 | 0.500 (t1.30) | - | - | - |
| TRELLIS w7 | 0.499 (t1.30) | - | - | - |
| GPT | 0.820 (t0.15) | 0.741 (t0.59, 0.700) | 0.262 (t0.59, 0.700) | 0.576 |
| LSTM | 0.926 (t0.15) | 0.703 (t0.71, 0.700) | 0.326 (t0.71, 0.700) | 0.764 |
| ngram 3 | 0.789 (t0.70) | 0.000 (t0.70, 0.789) | 0.000 (t0.70, 0.789) | 0.653 |
| ngram 4 | 0.878 (t0.70) | 0.000 (t0.70, 0.878) | 0.000 (t0.70, 0.878) | 0.402 |
| ngram 5 | 0.898 (t0.70) | 0.000 (t0.70, 0.898) | 0.000 (t0.70, 0.898) | 0.125 |

### phi-4/strict, gram >= 0.95 x ceiling (0.921)

| family | lowest-temp gram (T) | max tri\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |
|---|---|---|---|---|
| TRELLIS edge w5 | 0.802 (t0.30) | - | - | - |
| TRELLIS edge w6 | 0.668 (t1.30) | - | - | - |
| TRELLIS edge w7 | 0.753 (t0.30) | - | - | - |
| TRELLIS edge w8 | 0.563 (t2.40) | - | - | - |
| TRELLIS w2 | 0.697 (t1.00) | - | - | - |
| TRELLIS w3 | 0.878 (t0.50) | - | - | - |
| TRELLIS w4 | 0.757 (t0.50) | - | - | - |
| TRELLIS w5 | 0.609 (t0.50) | - | - | - |
| TRELLIS w6 | 0.500 (t1.30) | - | - | - |
| TRELLIS w7 | 0.499 (t1.30) | - | - | - |
| GPT | 0.820 (t0.15) | - | - | - |
| LSTM | 0.926 (t0.15) | 0.353 (t0.16, 0.921) | 0.009 (t0.16, 0.921) | 0.028 |
| ngram 3 | 0.789 (t0.70) | - | - | - |
| ngram 4 | 0.878 (t0.70) | - | - | - |
| ngram 5 | 0.898 (t0.70) | - | - | - |

## Paired comparison at the operating point (family and temperature chosen on seeds 13/17; per-seed values on 19/23 are the held-out report; each seed's own curve is interpolated at the chosen temperature)

TRELLIS candidates = every plain and edge window; the one with the best selection-seed value is carried forward. Held-out gram is the value on seeds 19/23 at the chosen temperature (the constraint was only enforced on 13/17).


### Qwen3-8B/strict, gram >= .95, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | none reaches the constraint | - | - | - | - | - |
| LSTM | none reaches the constraint | - | - | - | - | - |
| GPT | none reaches the constraint | - | - | - | - | - |

### Qwen3-8B/strict, gram >= .95, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | none reaches the constraint | - | - | - | - | - |
| LSTM | none reaches the constraint | - | - | - | - | - |
| GPT | none reaches the constraint | - | - | - | - | - |

### Qwen3-8B/strict, gram >= .90, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.56 | 0.054 | 0.900 | 0.057 | 0.901 | 0.051 / 0.057 / 0.053 / 0.060 |
| LSTM | LSTM t0.36 | 0.074 | 0.900 | 0.105 | 0.880 | 0.081 / 0.068 / 0.098 / 0.113 |
| GPT | GPT t0.16 | 0.014 | 0.900 | 0.014 | 0.790 | 0.015 / 0.014 / 0.017 / 0.012 |

TRELLIS (TRELLIS w3 t0.56) minus LSTM (t0.36): selection seeds -0.020 (wins 0/2); held-out seeds -0.049 (wins 0/2; per seed 19:-0.045, 23:-0.052)

TRELLIS (TRELLIS w3 t0.56) minus GPT (t0.16): selection seeds +0.040 (wins 2/2); held-out seeds +0.042 (wins 2/2; per seed 19:+0.036, 23:+0.048)

### Qwen3-8B/strict, gram >= .90, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.56 | 0.188 | 0.900 | 0.184 | 0.901 | 0.190 / 0.185 / 0.165 / 0.202 |
| LSTM | LSTM t0.36 | 0.455 | 0.900 | 0.536 | 0.880 | 0.459 / 0.452 / 0.516 / 0.557 |
| GPT | GPT t0.15 | 0.626 | 0.903 | 0.682 | 0.784 | 0.549 / 0.704 / 0.658 / 0.706 |

TRELLIS (TRELLIS w3 t0.56) minus LSTM (t0.36): selection seeds -0.268 (wins 0/2); held-out seeds -0.353 (wins 0/2; per seed 19:-0.351, 23:-0.355)

TRELLIS (TRELLIS w3 t0.56) minus GPT (t0.15): selection seeds -0.439 (wins 0/2); held-out seeds -0.498 (wins 0/2; per seed 19:-0.493, 23:-0.504)

### Qwen3-8B/strict, gram >= .85, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.78 | 0.124 | 0.850 | 0.115 | 0.853 | 0.123 / 0.126 / 0.122 / 0.107 |
| LSTM | LSTM t0.45 | 0.137 | 0.850 | 0.178 | 0.850 | 0.141 / 0.133 / 0.182 / 0.174 |
| GPT | GPT t0.32 | 0.079 | 0.850 | 0.078 | 0.750 | 0.081 / 0.078 / 0.075 / 0.081 |

TRELLIS (TRELLIS w3 t0.78) minus LSTM (t0.45): selection seeds -0.013 (wins 0/2); held-out seeds -0.064 (wins 0/2; per seed 19:-0.060, 23:-0.067)

TRELLIS (TRELLIS w3 t0.78) minus GPT (t0.32): selection seeds +0.045 (wins 2/2); held-out seeds +0.037 (wins 2/2; per seed 19:+0.047, 23:+0.026)

### Qwen3-8B/strict, gram >= .85, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.78 | 0.310 | 0.850 | 0.278 | 0.853 | 0.317 / 0.302 / 0.287 / 0.270 |
| LSTM | LSTM t0.45 | 0.521 | 0.850 | 0.592 | 0.850 | 0.523 / 0.519 / 0.598 / 0.587 |
| GPT | GPT t0.15 | 0.626 | 0.903 | 0.682 | 0.784 | 0.549 / 0.704 / 0.658 / 0.706 |

TRELLIS (TRELLIS w3 t0.78) minus LSTM (t0.45): selection seeds -0.212 (wins 0/2); held-out seeds -0.314 (wins 0/2; per seed 19:-0.311, 23:-0.317)

TRELLIS (TRELLIS w3 t0.78) minus GPT (t0.15): selection seeds -0.317 (wins 0/2); held-out seeds -0.404 (wins 0/2; per seed 19:-0.371, 23:-0.436)

### Qwen3-8B/strict, gram >= .80, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t1.00 | 0.190 | 0.806 | 0.172 | 0.807 | 0.182 / 0.198 / 0.180 / 0.163 |
| LSTM | LSTM t0.55 | 0.228 | 0.800 | 0.263 | 0.802 | 0.233 / 0.223 / 0.284 / 0.242 |
| GPT | GPT t0.40 | 0.129 | 0.800 | 0.122 | 0.738 | 0.135 / 0.123 / 0.116 / 0.129 |

TRELLIS (TRELLIS w3 t1.00) minus LSTM (t0.55): selection seeds -0.038 (wins 0/2); held-out seeds -0.091 (wins 0/2; per seed 19:-0.104, 23:-0.079)

TRELLIS (TRELLIS w3 t1.00) minus GPT (t0.40): selection seeds +0.061 (wins 2/2); held-out seeds +0.049 (wins 2/2; per seed 19:+0.064, 23:+0.034)

### Qwen3-8B/strict, gram >= .80, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t1.00 | 0.409 | 0.806 | 0.374 | 0.807 | 0.416 / 0.401 / 0.397 / 0.352 |
| LSTM | LSTM t0.55 | 0.604 | 0.800 | 0.654 | 0.802 | 0.594 / 0.613 / 0.685 / 0.622 |
| GPT | GPT t0.40 | 0.660 | 0.800 | 0.700 | 0.738 | 0.656 / 0.664 / 0.643 / 0.757 |

TRELLIS (TRELLIS w3 t1.00) minus LSTM (t0.55): selection seeds -0.195 (wins 0/2); held-out seeds -0.279 (wins 0/2; per seed 19:-0.288, 23:-0.271)

TRELLIS (TRELLIS w3 t1.00) minus GPT (t0.40): selection seeds -0.251 (wins 0/2); held-out seeds -0.326 (wins 0/2; per seed 19:-0.246, 23:-0.405)

### Qwen3-8B/strict, gram >= .75, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t1.00 | 0.190 | 0.806 | 0.172 | 0.807 | 0.182 / 0.198 / 0.180 / 0.163 |
| LSTM | LSTM t0.65 | 0.299 | 0.750 | 0.310 | 0.750 | 0.306 / 0.293 / 0.323 / 0.298 |
| GPT | GPT t0.56 | 0.249 | 0.750 | 0.243 | 0.687 | 0.239 / 0.259 / 0.242 / 0.244 |

TRELLIS (TRELLIS w3 t1.00) minus LSTM (t0.65): selection seeds -0.109 (wins 0/2); held-out seeds -0.139 (wins 0/2; per seed 19:-0.143, 23:-0.135)

TRELLIS (TRELLIS w3 t1.00) minus GPT (t0.56): selection seeds -0.058 (wins 0/2); held-out seeds -0.071 (wins 0/2; per seed 19:-0.061, 23:-0.081)

### Qwen3-8B/strict, gram >= .75, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS edge w7 t0.51 | 0.643 | 0.750 | 0.626 | 0.775 | 0.609 / 0.677 / 0.621 / 0.631 |
| LSTM | LSTM t0.65 | 0.668 | 0.750 | 0.682 | 0.750 | 0.656 / 0.679 / 0.690 / 0.673 |
| GPT | GPT t0.56 | 0.708 | 0.750 | 0.747 | 0.687 | 0.705 / 0.710 / 0.706 / 0.788 |

TRELLIS (TRELLIS edge w7 t0.51) minus LSTM (t0.65): selection seeds -0.024 (wins 0/2); held-out seeds -0.055 (wins 0/2; per seed 19:-0.069, 23:-0.042)

TRELLIS (TRELLIS edge w7 t0.51) minus GPT (t0.56): selection seeds -0.064 (wins 0/2); held-out seeds -0.121 (wins 0/2; per seed 19:-0.085, 23:-0.157)

### Qwen3-8B/strict, gram >= .70, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS edge w7 t0.90 | 0.265 | 0.700 | 0.266 | 0.742 | 0.271 / 0.258 / 0.262 / 0.270 |
| LSTM | LSTM t0.71 | 0.331 | 0.700 | 0.331 | 0.710 | 0.336 / 0.325 / 0.333 / 0.329 |
| GPT | GPT t0.65 | 0.305 | 0.700 | 0.302 | 0.646 | 0.290 / 0.320 / 0.295 / 0.309 |

TRELLIS (TRELLIS edge w7 t0.90) minus LSTM (t0.71): selection seeds -0.066 (wins 0/2); held-out seeds -0.064 (wins 0/2; per seed 19:-0.070, 23:-0.059)

TRELLIS (TRELLIS edge w7 t0.90) minus GPT (t0.65): selection seeds -0.040 (wins 0/2); held-out seeds -0.036 (wins 0/2; per seed 19:-0.033, 23:-0.039)

### Qwen3-8B/strict, gram >= .70, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS edge w7 t0.90 | 0.723 | 0.700 | 0.688 | 0.742 | 0.733 / 0.713 / 0.710 / 0.665 |
| LSTM | LSTM t0.71 | 0.710 | 0.700 | 0.701 | 0.710 | 0.702 / 0.717 / 0.683 / 0.719 |
| GPT | GPT t0.65 | 0.756 | 0.700 | 0.786 | 0.646 | 0.749 / 0.764 / 0.747 / 0.825 |

TRELLIS (TRELLIS edge w7 t0.90) minus LSTM (t0.71): selection seeds +0.013 (wins 1/2); held-out seeds -0.014 (wins 1/2; per seed 19:+0.027, 23:-0.054)

TRELLIS (TRELLIS edge w7 t0.90) minus GPT (t0.65): selection seeds -0.034 (wins 0/2); held-out seeds -0.098 (wins 0/2; per seed 19:-0.036, 23:-0.160)

### Qwen3-8B/strict, gram >= 0.95 x ceiling (0.910), metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.52 | 0.040 | 0.910 | 0.045 | 0.911 | 0.036 / 0.043 / 0.038 / 0.052 |
| LSTM | LSTM t0.33 | 0.059 | 0.910 | 0.086 | 0.884 | 0.064 / 0.054 / 0.077 / 0.094 |
| GPT | none reaches the constraint | - | - | - | - | - |

TRELLIS (TRELLIS w3 t0.52) minus LSTM (t0.33): selection seeds -0.019 (wins 0/2); held-out seeds -0.041 (wins 0/2; per seed 19:-0.039, 23:-0.042)

### Qwen3-8B/strict, gram >= 0.95 x ceiling (0.910), metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.52 | 0.162 | 0.910 | 0.165 | 0.911 | 0.163 / 0.160 / 0.140 / 0.190 |
| LSTM | LSTM t0.33 | 0.442 | 0.910 | 0.524 | 0.884 | 0.441 / 0.442 / 0.495 / 0.553 |
| GPT | none reaches the constraint | - | - | - | - | - |

TRELLIS (TRELLIS w3 t0.52) minus LSTM (t0.33): selection seeds -0.280 (wins 0/2); held-out seeds -0.359 (wins 0/2; per seed 19:-0.355, 23:-0.363)

### phi-4/strict, gram >= .95, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | none reaches the constraint | - | - | - | - | - |
| LSTM | LSTM t0.17 | 0.006 | 0.950 | 0.013 | 0.891 | 0.004 / 0.008 / 0.009 / 0.018 |
| GPT | none reaches the constraint | - | - | - | - | - |

### phi-4/strict, gram >= .95, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | none reaches the constraint | - | - | - | - | - |
| LSTM | LSTM t0.17 | 0.280 | 0.950 | 0.436 | 0.891 | 0.150 / 0.409 / 0.330 / 0.542 |
| GPT | none reaches the constraint | - | - | - | - | - |

### phi-4/strict, gram >= .90, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | none reaches the constraint | - | - | - | - | - |
| LSTM | LSTM t0.33 | 0.058 | 0.900 | 0.086 | 0.896 | 0.063 / 0.052 / 0.078 / 0.093 |
| GPT | none reaches the constraint | - | - | - | - | - |

### phi-4/strict, gram >= .90, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | none reaches the constraint | - | - | - | - | - |
| LSTM | LSTM t0.33 | 0.441 | 0.900 | 0.523 | 0.896 | 0.440 / 0.442 / 0.494 / 0.553 |
| GPT | none reaches the constraint | - | - | - | - | - |

### phi-4/strict, gram >= .85, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.64 | 0.079 | 0.850 | 0.078 | 0.843 | 0.077 / 0.080 / 0.079 / 0.078 |
| LSTM | LSTM t0.43 | 0.121 | 0.850 | 0.164 | 0.871 | 0.124 / 0.117 / 0.168 / 0.160 |
| GPT | GPT t0.31 | 0.068 | 0.850 | 0.072 | 0.784 | 0.069 / 0.067 / 0.072 / 0.072 |

TRELLIS (TRELLIS w3 t0.64) minus LSTM (t0.43): selection seeds -0.042 (wins 0/2); held-out seeds -0.085 (wins 0/2; per seed 19:-0.089, 23:-0.082)

TRELLIS (TRELLIS w3 t0.64) minus GPT (t0.31): selection seeds +0.011 (wins 2/2); held-out seeds +0.006 (wins 2/2; per seed 19:+0.007, 23:+0.006)

### phi-4/strict, gram >= .85, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.64 | 0.241 | 0.850 | 0.221 | 0.843 | 0.246 / 0.235 / 0.216 / 0.226 |
| LSTM | LSTM t0.43 | 0.504 | 0.850 | 0.578 | 0.871 | 0.508 / 0.499 / 0.578 / 0.578 |
| GPT | GPT t0.15 | 0.626 | 0.856 | 0.682 | 0.784 | 0.549 / 0.704 / 0.658 / 0.706 |

TRELLIS (TRELLIS w3 t0.64) minus LSTM (t0.43): selection seeds -0.263 (wins 0/2); held-out seeds -0.357 (wins 0/2; per seed 19:-0.362, 23:-0.352)

TRELLIS (TRELLIS w3 t0.64) minus GPT (t0.15): selection seeds -0.386 (wins 0/2); held-out seeds -0.461 (wins 0/2; per seed 19:-0.442, 23:-0.480)

### phi-4/strict, gram >= .80, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.93 | 0.166 | 0.800 | 0.156 | 0.798 | 0.165 / 0.168 / 0.163 / 0.148 |
| LSTM | LSTM t0.53 | 0.203 | 0.800 | 0.245 | 0.819 | 0.202 / 0.205 / 0.269 / 0.221 |
| GPT | GPT t0.37 | 0.109 | 0.800 | 0.110 | 0.765 | 0.116 / 0.101 / 0.111 / 0.110 |

TRELLIS (TRELLIS w3 t0.93) minus LSTM (t0.53): selection seeds -0.037 (wins 0/2); held-out seeds -0.089 (wins 0/2; per seed 19:-0.106, 23:-0.073)

TRELLIS (TRELLIS w3 t0.93) minus GPT (t0.37): selection seeds +0.058 (wins 2/2); held-out seeds +0.046 (wins 2/2; per seed 19:+0.052, 23:+0.039)

### phi-4/strict, gram >= .80, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t0.93 | 0.378 | 0.800 | 0.345 | 0.798 | 0.386 / 0.371 / 0.363 / 0.327 |
| LSTM | LSTM t0.53 | 0.586 | 0.800 | 0.642 | 0.819 | 0.579 / 0.593 / 0.668 / 0.616 |
| GPT | GPT t0.37 | 0.644 | 0.800 | 0.699 | 0.765 | 0.634 / 0.654 / 0.644 / 0.754 |

TRELLIS (TRELLIS w3 t0.93) minus LSTM (t0.53): selection seeds -0.208 (wins 0/2); held-out seeds -0.297 (wins 0/2; per seed 19:-0.305, 23:-0.290)

TRELLIS (TRELLIS w3 t0.93) minus GPT (t0.37): selection seeds -0.266 (wins 0/2); held-out seeds -0.354 (wins 0/2; per seed 19:-0.281, 23:-0.427)

### phi-4/strict, gram >= .75, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t1.00 | 0.187 | 0.789 | 0.174 | 0.789 | 0.185 / 0.189 / 0.182 / 0.166 |
| LSTM | LSTM t0.64 | 0.291 | 0.750 | 0.308 | 0.758 | 0.296 / 0.285 / 0.327 / 0.288 |
| GPT | GPT t0.54 | 0.227 | 0.750 | 0.228 | 0.709 | 0.222 / 0.232 / 0.236 / 0.219 |

TRELLIS (TRELLIS w3 t1.00) minus LSTM (t0.64): selection seeds -0.104 (wins 0/2); held-out seeds -0.133 (wins 0/2; per seed 19:-0.145, 23:-0.122)

TRELLIS (TRELLIS w3 t1.00) minus GPT (t0.54): selection seeds -0.040 (wins 0/2); held-out seeds -0.054 (wins 0/2; per seed 19:-0.054, 23:-0.053)

### phi-4/strict, gram >= .75, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS w3 t1.00 | 0.409 | 0.789 | 0.374 | 0.789 | 0.416 / 0.401 / 0.397 / 0.352 |
| LSTM | LSTM t0.64 | 0.661 | 0.750 | 0.680 | 0.758 | 0.648 / 0.673 / 0.696 / 0.664 |
| GPT | GPT t0.54 | 0.696 | 0.750 | 0.737 | 0.709 | 0.693 / 0.699 / 0.691 / 0.782 |

TRELLIS (TRELLIS w3 t1.00) minus LSTM (t0.64): selection seeds -0.252 (wins 0/2); held-out seeds -0.306 (wins 0/2; per seed 19:-0.299, 23:-0.313)

TRELLIS (TRELLIS w3 t1.00) minus GPT (t0.54): selection seeds -0.287 (wins 0/2); held-out seeds -0.363 (wins 0/2; per seed 19:-0.295, 23:-0.431)

### phi-4/strict, gram >= .70, metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS edge w7 t0.87 | 0.254 | 0.700 | 0.256 | 0.736 | 0.260 / 0.247 / 0.252 / 0.259 |
| LSTM | LSTM t0.70 | 0.321 | 0.700 | 0.322 | 0.717 | 0.327 / 0.316 / 0.322 / 0.321 |
| GPT | GPT t0.63 | 0.289 | 0.700 | 0.288 | 0.647 | 0.280 / 0.298 / 0.296 / 0.279 |

TRELLIS (TRELLIS edge w7 t0.87) minus LSTM (t0.70): selection seeds -0.068 (wins 0/2); held-out seeds -0.066 (wins 0/2; per seed 19:-0.070, 23:-0.062)

TRELLIS (TRELLIS edge w7 t0.87) minus GPT (t0.63): selection seeds -0.036 (wins 0/2); held-out seeds -0.032 (wins 0/2; per seed 19:-0.044, 23:-0.020)

### phi-4/strict, gram >= .70, metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | TRELLIS edge w7 t0.87 | 0.714 | 0.700 | 0.684 | 0.736 | 0.723 / 0.706 / 0.707 / 0.661 |
| LSTM | LSTM t0.70 | 0.701 | 0.700 | 0.689 | 0.717 | 0.691 / 0.710 / 0.664 / 0.714 |
| GPT | GPT t0.63 | 0.747 | 0.700 | 0.779 | 0.647 | 0.741 / 0.753 / 0.743 / 0.816 |

TRELLIS (TRELLIS edge w7 t0.87) minus LSTM (t0.70): selection seeds +0.013 (wins 1/2); held-out seeds -0.005 (wins 1/2; per seed 19:+0.043, 23:-0.053)

TRELLIS (TRELLIS edge w7 t0.87) minus GPT (t0.63): selection seeds -0.033 (wins 0/2); held-out seeds -0.096 (wins 0/2; per seed 19:-0.036, 23:-0.155)

### phi-4/strict, gram >= 0.95 x ceiling (0.921), metric fair (new, gram, unstored trigram)/draw

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | none reaches the constraint | - | - | - | - | - |
| LSTM | LSTM t0.20 | 0.011 | 0.921 | 0.021 | 0.899 | 0.010 / 0.013 / 0.015 / 0.027 |
| GPT | none reaches the constraint | - | - | - | - | - |

### phi-4/strict, gram >= 0.95 x ceiling (0.921), metric tri|d

| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |
|---|---|---|---|---|---|---|---|
| TRELLIS | none reaches the constraint | - | - | - | - | - |
| LSTM | LSTM t0.20 | 0.356 | 0.921 | 0.470 | 0.899 | 0.269 / 0.442 / 0.390 / 0.550 |
| GPT | none reaches the constraint | - | - | - | - | - |

## Compact verdict table: TRELLIS minus opponent, held-out seeds 19/23

| judge | gram constraint | metric | vs | TRELLIS pick | sel diff | held-out diff | held-out wins |
|---|---|---|---|---|---|---|---|
| Qwen3-8B/strict | .70 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS edge w7 | -0.040 | -0.036 | 0/2 |
| Qwen3-8B/strict | .70 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS edge w7 | -0.066 | -0.064 | 0/2 |
| Qwen3-8B/strict | .70 | tri|d | GPT | TRELLIS edge w7 | -0.034 | -0.098 | 0/2 |
| Qwen3-8B/strict | .70 | tri|d | LSTM | TRELLIS edge w7 | +0.013 | -0.014 | 1/2 |
| Qwen3-8B/strict | .75 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | -0.058 | -0.071 | 0/2 |
| Qwen3-8B/strict | .75 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.109 | -0.139 | 0/2 |
| Qwen3-8B/strict | .75 | tri|d | GPT | TRELLIS edge w7 | -0.064 | -0.121 | 0/2 |
| Qwen3-8B/strict | .75 | tri|d | LSTM | TRELLIS edge w7 | -0.024 | -0.055 | 0/2 |
| Qwen3-8B/strict | .80 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.061 | +0.049 | 2/2 |
| Qwen3-8B/strict | .80 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.038 | -0.091 | 0/2 |
| Qwen3-8B/strict | .80 | tri|d | GPT | TRELLIS w3 | -0.251 | -0.326 | 0/2 |
| Qwen3-8B/strict | .80 | tri|d | LSTM | TRELLIS w3 | -0.195 | -0.279 | 0/2 |
| Qwen3-8B/strict | .85 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.045 | +0.037 | 2/2 |
| Qwen3-8B/strict | .85 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.013 | -0.064 | 0/2 |
| Qwen3-8B/strict | .85 | tri|d | GPT | TRELLIS w3 | -0.317 | -0.404 | 0/2 |
| Qwen3-8B/strict | .85 | tri|d | LSTM | TRELLIS w3 | -0.212 | -0.314 | 0/2 |
| Qwen3-8B/strict | .90 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.040 | +0.042 | 2/2 |
| Qwen3-8B/strict | .90 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.020 | -0.049 | 0/2 |
| Qwen3-8B/strict | .90 | tri|d | GPT | TRELLIS w3 | -0.439 | -0.498 | 0/2 |
| Qwen3-8B/strict | .90 | tri|d | LSTM | TRELLIS w3 | -0.268 | -0.353 | 0/2 |
| Qwen3-8B/strict | 0.95 x ceiling (0.910) | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.019 | -0.041 | 0/2 |
| Qwen3-8B/strict | 0.95 x ceiling (0.910) | tri|d | LSTM | TRELLIS w3 | -0.280 | -0.359 | 0/2 |
| phi-4/strict | .70 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS edge w7 | -0.036 | -0.032 | 0/2 |
| phi-4/strict | .70 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS edge w7 | -0.068 | -0.066 | 0/2 |
| phi-4/strict | .70 | tri|d | GPT | TRELLIS edge w7 | -0.033 | -0.096 | 0/2 |
| phi-4/strict | .70 | tri|d | LSTM | TRELLIS edge w7 | +0.013 | -0.005 | 1/2 |
| phi-4/strict | .75 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | -0.040 | -0.054 | 0/2 |
| phi-4/strict | .75 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.104 | -0.133 | 0/2 |
| phi-4/strict | .75 | tri|d | GPT | TRELLIS w3 | -0.287 | -0.363 | 0/2 |
| phi-4/strict | .75 | tri|d | LSTM | TRELLIS w3 | -0.252 | -0.306 | 0/2 |
| phi-4/strict | .80 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.058 | +0.046 | 2/2 |
| phi-4/strict | .80 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.037 | -0.089 | 0/2 |
| phi-4/strict | .80 | tri|d | GPT | TRELLIS w3 | -0.266 | -0.354 | 0/2 |
| phi-4/strict | .80 | tri|d | LSTM | TRELLIS w3 | -0.208 | -0.297 | 0/2 |
| phi-4/strict | .85 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.011 | +0.006 | 2/2 |
| phi-4/strict | .85 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.042 | -0.085 | 0/2 |
| phi-4/strict | .85 | tri|d | GPT | TRELLIS w3 | -0.386 | -0.461 | 0/2 |
| phi-4/strict | .85 | tri|d | LSTM | TRELLIS w3 | -0.263 | -0.357 | 0/2 |

## Same comparison with the constraint enforced on each seed's own curve (family chosen on 13/17 as above)

Removes seed-to-seed differences in grammaticality from the comparison: every seed is evaluated at its own interpolated point where its gram first reaches the constraint (or its best feasible point); seeds where either arm never reaches it are dropped (listed as n).

| judge | gram constraint | metric | vs | TRELLIS pick | mean diff all seeds (n) | mean diff 13/17 | mean diff 19/23 | wins on 19/23 | per-seed diffs |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B/strict | .70 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS edge w7 | +0.031 (4) | -0.044 | +0.106 | 2/2 | 13:+0.006, 17:-0.094, 19:+0.006, 23:+0.205 |
| Qwen3-8B/strict | .70 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS edge w7 | -0.054 (4) | -0.067 | -0.041 | 1/2 | 13:-0.051, 17:-0.084, 19:-0.107, 23:+0.026 |
| Qwen3-8B/strict | .70 | tri|d | GPT | TRELLIS edge w7 | -0.024 (4) | -0.035 | -0.013 | 0/2 | 13:+0.007, 17:-0.077, 19:-0.010, 23:-0.016 |
| Qwen3-8B/strict | .70 | tri|d | LSTM | TRELLIS edge w7 | +0.007 (4) | +0.013 | +0.001 | 1/2 | 13:+0.034, 17:-0.008, 19:-0.035, 23:+0.037 |
| Qwen3-8B/strict | .75 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | -0.003 (4) | -0.059 | +0.053 | 1/2 | 13:-0.053, 17:-0.064, 19:-0.017, 23:+0.123 |
| Qwen3-8B/strict | .75 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.121 (4) | -0.108 | -0.133 | 0/2 | 13:-0.136, 17:-0.081, 19:-0.146, 23:-0.121 |
| Qwen3-8B/strict | .75 | tri|d | GPT | TRELLIS edge w7 | -0.066 (4) | -0.080 | -0.052 | 1/2 | 13:-0.049, 17:-0.111, 19:-0.114, 23:+0.010 |
| Qwen3-8B/strict | .75 | tri|d | LSTM | TRELLIS edge w7 | -0.043 (4) | -0.042 | -0.043 | 1/2 | 13:-0.019, 17:-0.065, 19:-0.158, 23:+0.071 |
| Qwen3-8B/strict | .80 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.067 (3) | +0.051 | +0.097 | 1/1 | 13:+0.017, 17:+0.086, 19:+0.097 |
| Qwen3-8B/strict | .80 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.081 (4) | -0.059 | -0.102 | 0/2 | 13:-0.109, 17:-0.009, 19:-0.146, 23:-0.058 |
| Qwen3-8B/strict | .80 | tri|d | GPT | TRELLIS w3 | -0.284 (3) | -0.283 | -0.287 | 0/1 | 13:-0.264, 17:-0.302, 19:-0.287 |
| Qwen3-8B/strict | .80 | tri|d | LSTM | TRELLIS w3 | -0.254 (4) | -0.210 | -0.299 | 0/2 | 13:-0.223, 17:-0.197, 19:-0.335, 23:-0.262 |
| Qwen3-8B/strict | .85 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.055 (3) | +0.046 | +0.072 | 1/1 | 13:+0.023, 17:+0.070, 19:+0.072 |
| Qwen3-8B/strict | .85 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.034 (4) | -0.007 | -0.061 | 0/2 | 13:-0.034, 17:+0.021, 19:-0.118, 23:-0.004 |
| Qwen3-8B/strict | .85 | tri|d | GPT | TRELLIS w3 | -0.359 (3) | -0.341 | -0.394 | 0/1 | 13:-0.302, 17:-0.381, 19:-0.394 |
| Qwen3-8B/strict | .85 | tri|d | LSTM | TRELLIS w3 | -0.261 (4) | -0.205 | -0.318 | 0/2 | 13:-0.229, 17:-0.181, 19:-0.378, 23:-0.258 |
| Qwen3-8B/strict | .90 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.022 (2) | +0.029 | +0.015 | 1/1 | 17:+0.029, 19:+0.015 |
| Qwen3-8B/strict | .90 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.047 (3) | -0.026 | -0.088 | 0/1 | 13:-0.068, 17:+0.016, 19:-0.088 |
| Qwen3-8B/strict | .90 | tri|d | GPT | TRELLIS w3 | -0.512 (2) | -0.499 | -0.524 | 0/1 | 17:-0.499, 19:-0.524 |
| Qwen3-8B/strict | .90 | tri|d | LSTM | TRELLIS w3 | -0.323 (3) | -0.281 | -0.406 | 0/1 | 13:-0.320, 17:-0.242, 19:-0.406 |
| Qwen3-8B/strict | 0.95 x ceiling (0.910) | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | +0.017 (1) | +0.017 | - | 0/0 | 17:+0.017 |
| Qwen3-8B/strict | 0.95 x ceiling (0.910) | tri|d | LSTM | TRELLIS w3 | -0.265 (1) | -0.265 | - | 0/0 | 17:-0.265 |
| phi-4/strict | .70 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS edge w7 | +0.030 (4) | -0.040 | +0.100 | 1/2 | 13:-0.001, 17:-0.080, 19:-0.039, 23:+0.239 |
| phi-4/strict | .70 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS edge w7 | -0.059 (4) | -0.071 | -0.046 | 1/2 | 13:-0.053, 17:-0.089, 19:-0.131, 23:+0.038 |
| phi-4/strict | .70 | tri|d | GPT | TRELLIS edge w7 | -0.027 (4) | -0.032 | -0.022 | 0/2 | 13:+0.001, 17:-0.064, 19:-0.032, 23:-0.013 |
| phi-4/strict | .70 | tri|d | LSTM | TRELLIS edge w7 | +0.005 (4) | +0.011 | -0.001 | 1/2 | 13:+0.034, 17:-0.012, 19:-0.046, 23:+0.044 |
| phi-4/strict | .75 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | -0.046 (3) | -0.035 | -0.069 | 0/1 | 13:-0.051, 17:-0.019, 19:-0.069 |
| phi-4/strict | .75 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.117 (4) | -0.105 | -0.130 | 0/2 | 13:-0.123, 17:-0.087, 19:-0.149, 23:-0.111 |
| phi-4/strict | .75 | tri|d | GPT | TRELLIS w3 | -0.299 (3) | -0.295 | -0.307 | 0/1 | 13:-0.288, 17:-0.302, 19:-0.307 |
| phi-4/strict | .75 | tri|d | LSTM | TRELLIS w3 | -0.281 (4) | -0.255 | -0.308 | 0/2 | 13:-0.248, 17:-0.261, 19:-0.319, 23:-0.297 |
| phi-4/strict | .80 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | +0.001 (3) | +0.048 | -0.093 | 0/1 | 13:-0.006, 17:+0.102, 19:-0.093 |
| phi-4/strict | .80 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.077 (4) | -0.031 | -0.124 | 0/2 | 13:-0.044, 17:-0.018, 19:-0.213, 23:-0.035 |
| phi-4/strict | .80 | tri|d | GPT | TRELLIS w3 | -0.329 (3) | -0.302 | -0.383 | 0/1 | 13:-0.302, 17:-0.302, 19:-0.383 |
| phi-4/strict | .80 | tri|d | LSTM | TRELLIS w3 | -0.273 (4) | -0.202 | -0.343 | 0/2 | 13:-0.212, 17:-0.193, 19:-0.430, 23:-0.256 |
| phi-4/strict | .85 | fair (new, gram, unstored trigram)/draw | GPT | TRELLIS w3 | -0.004 (3) | +0.005 | -0.023 | 0/1 | 13:-0.018, 17:+0.028, 19:-0.023 |
| phi-4/strict | .85 | fair (new, gram, unstored trigram)/draw | LSTM | TRELLIS w3 | -0.087 (4) | -0.033 | -0.142 | 0/2 | 13:-0.066, 17:+0.001, 19:-0.213, 23:-0.071 |
| phi-4/strict | .85 | tri|d | GPT | TRELLIS w3 | -0.439 (3) | -0.413 | -0.490 | 0/1 | 13:-0.372, 17:-0.455, 19:-0.490 |
| phi-4/strict | .85 | tri|d | LSTM | TRELLIS w3 | -0.336 (4) | -0.255 | -0.418 | 0/2 | 13:-0.292, 17:-0.218, 19:-0.499, 23:-0.337 |

## Copy and splice share (distinct new sentences judged grammatical)

For each family and judge: the measured setting with the lowest temperature, and the measured setting with the highest gram that is closest to .95 from above if any (else the highest-gram setting). `tri=0` = distinct new sentence all of whose trigrams are stored (a splice of stored pieces or a stored span, no new trigram). `stored span` = it is a proper contiguous span of one stored sentence (a headless or truncated cut of a stored sentence). Exact stored copies: the dumps already exclude them (generation draws are 'new' only, so the share is 0 by construction in this column); they never enter any count here.

| judge | family | T | gram | grammatical distinct new per draw | of those: tri=0 | of those: stored span | exact stored copies, share of draws | grammatical AND unstored trigram per draw |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B/strict | TRELLIS edge w5 | 0.30 | 0.807 | 0.023 | 0.712 | 0.079 | 0.000 | 0.007 |
| Qwen3-8B/strict | TRELLIS edge w6 | 1.30 | 0.676 | 0.497 | 0.348 | 0.024 | 0.000 | 0.324 |
| Qwen3-8B/strict | TRELLIS edge w7 | 0.30 | 0.759 | 0.018 | 0.591 | 0.073 | 0.000 | 0.007 |
| Qwen3-8B/strict | TRELLIS edge w7 | 0.50 | 0.764 | 0.172 | 0.457 | 0.026 | 0.000 | 0.093 |
| Qwen3-8B/strict | TRELLIS edge w8 | 2.40 | 0.567 | 0.470 | 0.239 | 0.019 | 0.000 | 0.358 |
| Qwen3-8B/strict | TRELLIS w2 | 1.00 | 0.714 | 0.609 | 0.724 | 0.075 | 0.000 | 0.168 |
| Qwen3-8B/strict | TRELLIS w3 | 0.50 | 0.914 | 0.298 | 0.874 | 0.068 | 0.000 | 0.038 |
| Qwen3-8B/strict | TRELLIS w4 | 0.50 | 0.790 | 0.198 | 0.668 | 0.047 | 0.000 | 0.066 |
| Qwen3-8B/strict | TRELLIS w5 | 0.50 | 0.618 | 0.180 | 0.522 | 0.106 | 0.000 | 0.086 |
| Qwen3-8B/strict | TRELLIS w6 | 1.30 | 0.509 | 0.387 | 0.299 | 0.098 | 0.000 | 0.271 |
| Qwen3-8B/strict | TRELLIS w7 | 1.30 | 0.514 | 0.398 | 0.310 | 0.098 | 0.000 | 0.275 |
| Qwen3-8B/strict | GPT | 0.15 | 0.844 | 0.021 | 0.421 | 0.127 | 0.000 | 0.012 |
| Qwen3-8B/strict | GPT | 0.20 | 0.852 | 0.048 | 0.422 | 0.107 | 0.000 | 0.027 |
| Qwen3-8B/strict | LSTM | 0.15 | 0.888 | 0.019 | 0.664 | 0.216 | 0.000 | 0.007 |
| Qwen3-8B/strict | LSTM | 0.20 | 0.914 | 0.042 | 0.589 | 0.143 | 0.000 | 0.018 |
| Qwen3-8B/strict | ngram 3 | 0.70 | 0.811 | 0.530 | 1.000 | 0.146 | 0.000 | 0.000 |
| Qwen3-8B/strict | ngram 4 | 0.70 | 0.919 | 0.369 | 1.000 | 0.148 | 0.000 | 0.000 |
| Qwen3-8B/strict | ngram 5 | 0.70 | 0.926 | 0.115 | 1.000 | 0.174 | 0.000 | 0.000 |
| phi-4/strict | TRELLIS edge w5 | 0.30 | 0.802 | 0.023 | 0.710 | 0.080 | 0.000 | 0.007 |
| phi-4/strict | TRELLIS edge w6 | 1.30 | 0.668 | 0.491 | 0.346 | 0.025 | 0.000 | 0.321 |
| phi-4/strict | TRELLIS edge w7 | 0.30 | 0.753 | 0.018 | 0.587 | 0.073 | 0.000 | 0.007 |
| phi-4/strict | TRELLIS edge w8 | 2.40 | 0.563 | 0.467 | 0.241 | 0.020 | 0.000 | 0.354 |
| phi-4/strict | TRELLIS w2 | 1.00 | 0.697 | 0.594 | 0.725 | 0.079 | 0.000 | 0.164 |
| phi-4/strict | TRELLIS w3 | 0.50 | 0.878 | 0.286 | 0.868 | 0.072 | 0.000 | 0.038 |
| phi-4/strict | TRELLIS w4 | 0.50 | 0.757 | 0.189 | 0.668 | 0.049 | 0.000 | 0.063 |
| phi-4/strict | TRELLIS w5 | 0.50 | 0.609 | 0.177 | 0.509 | 0.108 | 0.000 | 0.087 |
| phi-4/strict | TRELLIS w6 | 1.30 | 0.500 | 0.381 | 0.302 | 0.086 | 0.000 | 0.265 |
| phi-4/strict | TRELLIS w7 | 1.30 | 0.499 | 0.386 | 0.321 | 0.085 | 0.000 | 0.263 |
| phi-4/strict | GPT | 0.15 | 0.820 | 0.020 | 0.439 | 0.130 | 0.000 | 0.011 |
| phi-4/strict | GPT | 0.20 | 0.850 | 0.047 | 0.434 | 0.111 | 0.000 | 0.027 |
| phi-4/strict | LSTM | 0.15 | 0.926 | 0.020 | 0.681 | 0.218 | 0.000 | 0.006 |
| phi-4/strict | ngram 3 | 0.70 | 0.789 | 0.515 | 1.000 | 0.148 | 0.000 | 0.000 |
| phi-4/strict | ngram 4 | 0.70 | 0.878 | 0.353 | 1.000 | 0.157 | 0.000 | 0.000 |
| phi-4/strict | ngram 5 | 0.70 | 0.898 | 0.112 | 1.000 | 0.182 | 0.000 | 0.000 |
| phi-4/strict | ngram 5 | 1.00 | 0.899 | 0.117 | 1.000 | 0.176 | 0.000 | 0.000 |
