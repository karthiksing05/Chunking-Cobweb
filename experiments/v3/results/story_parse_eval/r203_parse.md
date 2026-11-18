# Round 203: boundary-once and Goodman readouts for TRELLIS parsing (verdict: neither beats default/parts)

Code: `src/trellis2/trellis3_r203.py` (subclass `Trellis3R203`; trellis3.py untouched, defaults byte-identical), eval `experiments/v3/story_parse_r203.py`
(r185 eval + extra readouts), summary `experiments/v3/story_parse_r203_summary.py`, jobs `logs/v3_r203_run.sh`, tests `tests/trellis2/test_r203_parse.py` (14 pass; full tests/trellis2: 107 pass).
Same eval as rounds 196/198: seeds 13/17/19, 300 fixed held-out sentences, learned 250/1000/4000, gold = NLTK shallow phrases. Arms use the r196parts recipe (consist="parts") unless noted.

## What was built

* (a) `edge_w=1`: count the sentence edge once, slots kept. A span at the edge has l1 = l2 = BOS (or r1 = r2 = EOS); those m slots are one observation, so each
  slot's log-ratio term in the context gain is weighted edge_w/m (default: weight 1 each = counted twice). Instance, sorting path and storage are unchanged.
  (r196b's edge_once removed the slot from the scored instance.) edge_w=2 reproduces the default gain (tested).
* (b) Goodman labelled recall: binary tree maximising the sum of span posteriors, without the split-posterior tie-break the existing `_mbr` adds (so the
  existing readout was already nearly Goodman; this is the pure form). Also used while learning (arm r203b). Additionally an n-ary Goodman readout from the
  posteriors, non-crossing spans maximising sum(P - lam) (`gm_p50` lam .5, `gm_bud` lam set so the bracket budget equals BPE x1's), and `post_bud` (posterior
  threshold at the same budget). Each is checked against brute force on small n (tests).
* (c) = a + b (arm r203c).
* Reruns r203def / r203base reproduce r185 / r196parts exactly (post rows identical to the published tables), so the new eval path is a faithful copy.

## Verdict

* (a) fails, as in r196b: F1 falls from .357 to .278 (post, 4000), F1+sv .466 to .382, crossing .209 to .262, cross-seed F1 .80 to .67. The edge double count is
  not noise the parser can drop for free: gold phrases sit at sentence edges (right-branching, which brackets end-anchored spans, gets F1 .466). Consistency |d|
  improves only at small n and for the posterior readout (250: .100 to .067 post; 1000: .164 to .109; at 4000 .172 to .141 post but .216 to .223 chosen, a
  tie), and decisive |d| gets worse (.106 to .117 at 4000). Keeping the slots (weights 1/2 each) did not rescue it versus r196b's dropping the slot: edge_once
  (4000, post) F1 .284, this .278.
* (b) is a wash. Chosen-tree F1 .363/.369/.366 vs parts .356/.369/.369 at 250/1000/4000 (per seed: +.009,0,+.011; -.001,+.003,0; -.009,+.003,-.002), F1+sv
  .466/.487/.478 vs .450/.481/.475, cross-seed F1 .639/.765/.814 vs .606/.756/.798 (a small consistent gain), but consistency |d| .229/.221/.222 vs
  .213/.219/.216 and decisive |d| .132/.120/.109 vs .135/.124/.117 (no change within noise). n-ary Goodman at lam .5 equals the existing posterior >= .5 readout
  (identical at 1000 and 4000; the spans are already non-crossing), and at the BPE-matched budget gm_bud = post_bud (identical): there is no gain from
  the labelled-recall CKY beyond thresholding the posterior.
* (c) is worse than (b) alone (F1 .315 chosen at 4000 vs .366).
* No arm beats default/parts on F1 while moving consistency and decisive closer to gold. Best parser stays default/parts.

## Honest wins that stay (new measurement, not a new parser)

Matched budget (posterior threshold set so brackets/sentence = BPE x1's; `post_bud`, r203base, parts): crossing .171 vs BPE .299 (250), .169 vs .281 (1000), .210 vs .301 (4000).
F1 at that budget: .309 vs .275 (250, TRELLIS ahead), .344 vs .355 (1000, BPE ahead by .011), .358 vs .369 (4000, BPE ahead by .011); F1+sv .363/.452/.468 vs .296/.384/.417 (TRELLIS ahead
at all three). Consistency |d| .038/.132/.175 vs BPE .248/.327/.373 (TRELLIS closer to gold); decisive |d| .091/.103/.106 vs BPE .009/.040/.059 (BPE closer). Cross-seed F1 .548/.733/.801 vs BPE
.510/.682/.839 (TRELLIS ahead at 250 and 1000, BPE at 4000). The consistency metric rewards over-bracketing; read it against the gold ceiling (.116 / .982), not as a raw rate.

## Full table (mean over seeds; per-seed in the `r203*_table.md` files)

### 250 sentences learned

| method | brackets/sent | F1 | F1+sv | crossing | xseed F1 | abs(cons-gold) | abs(dec-gold) | consistency | decisive |
|---|---|---|---|---|---|---|---|---|---|
| GOLD (shallow phrases) | - | 1 | 1 | 0 | 1 | 0 | 0 | 0.116 | 0.982 |
| default r185 / post | 1.754 | 0.308 | 0.421 | 0.171 | 0.586 | 0.118 | 0.097 | 0.234 | 0.885 |
| default r185 / chosen | 3.303 | 0.343 | 0.449 | 0.294 | 0.587 | 0.242 | 0.135 | 0.358 | 0.847 |
| parts r196 / post | 1.934 | 0.353 | 0.436 | 0.184 | 0.594 | 0.100 | 0.103 | 0.216 | 0.879 |
| parts r196 / chosen | 3.303 | 0.356 | 0.450 | 0.295 | 0.606 | 0.213 | 0.135 | 0.329 | 0.847 |
| parts+cohesion r196 / post | 2.501 | 0.292 | 0.385 | 0.319 | 0.466 | 0.270 | 0.107 | 0.386 | 0.875 |
| parts+edge_once r196 / post | 1.266 | 0.247 | 0.296 | 0.307 | 0.468 | 0.066 | 0.114 | 0.182 | 0.868 |
| r203 default (rerun) / post | 1.754 | 0.308 | 0.421 | 0.171 | 0.586 | 0.118 | 0.097 | 0.234 | 0.885 |
| r203 default (rerun) / post_bud | 1.441 | 0.296 | 0.377 | 0.166 | 0.553 | 0.069 | 0.105 | 0.185 | 0.877 |
| r203 default (rerun) / gm_bud | 1.441 | 0.296 | 0.377 | 0.166 | 0.553 | 0.069 | 0.105 | 0.185 | 0.877 |
| r203 default (rerun) / gm_p50 | 1.711 | 0.302 | 0.409 | 0.174 | 0.575 | 0.111 | 0.099 | 0.227 | 0.883 |
| r203 parts (rerun) / post | 1.934 | 0.353 | 0.436 | 0.184 | 0.594 | 0.100 | 0.103 | 0.216 | 0.879 |
| r203 parts (rerun) / chosen | 3.303 | 0.356 | 0.450 | 0.295 | 0.606 | 0.213 | 0.135 | 0.329 | 0.847 |
| r203 parts (rerun) / post_bud | 1.452 | 0.309 | 0.363 | 0.171 | 0.548 | 0.038 | 0.091 | 0.154 | 0.891 |
| r203 parts (rerun) / gm_bud | 1.452 | 0.309 | 0.363 | 0.171 | 0.548 | 0.038 | 0.091 | 0.154 | 0.891 |
| r203 parts (rerun) / gm_p50 | 1.867 | 0.341 | 0.416 | 0.187 | 0.580 | 0.089 | 0.107 | 0.205 | 0.874 |
| r203a edge_w=1 / post | 1.260 | 0.244 | 0.307 | 0.284 | 0.498 | 0.067 | 0.109 | 0.183 | 0.873 |
| r203a edge_w=1 / chosen | 3.303 | 0.324 | 0.368 | 0.357 | 0.497 | 0.164 | 0.160 | 0.280 | 0.822 |
| r203a edge_w=1 / post_bud | 1.452 | 0.267 | 0.336 | 0.280 | 0.521 | 0.086 | 0.114 | 0.202 | 0.868 |
| r203a edge_w=1 / gm_bud | 1.452 | 0.281 | 0.317 | 0.294 | 0.469 | 0.068 | 0.114 | 0.184 | 0.868 |
| r203a edge_w=1 / gm_p50 | 0.996 | 0.191 | 0.231 | 0.328 | 0.364 | 0.034 | 0.101 | 0.150 | 0.880 |
| r203b goodman tree / post | 1.934 | 0.353 | 0.436 | 0.184 | 0.594 | 0.100 | 0.103 | 0.216 | 0.879 |
| r203b goodman tree / chosen | 3.303 | 0.363 | 0.466 | 0.285 | 0.639 | 0.229 | 0.132 | 0.345 | 0.850 |
| r203b goodman tree / post_bud | 1.452 | 0.309 | 0.363 | 0.171 | 0.548 | 0.038 | 0.091 | 0.154 | 0.891 |
| r203b goodman tree / gm_bud | 1.452 | 0.309 | 0.363 | 0.171 | 0.548 | 0.038 | 0.091 | 0.154 | 0.891 |
| r203b goodman tree / gm_p50 | 1.867 | 0.341 | 0.416 | 0.187 | 0.580 | 0.089 | 0.107 | 0.205 | 0.874 |
| r203c edge_w=1+goodman / post | 1.260 | 0.244 | 0.307 | 0.284 | 0.498 | 0.067 | 0.109 | 0.183 | 0.873 |
| r203c edge_w=1+goodman / chosen | 3.303 | 0.350 | 0.412 | 0.318 | 0.529 | 0.182 | 0.160 | 0.298 | 0.822 |
| r203c edge_w=1+goodman / post_bud | 1.452 | 0.267 | 0.336 | 0.280 | 0.521 | 0.086 | 0.114 | 0.202 | 0.868 |
| r203c edge_w=1+goodman / gm_bud | 1.452 | 0.281 | 0.317 | 0.294 | 0.469 | 0.068 | 0.114 | 0.184 | 0.868 |
| r203c edge_w=1+goodman / gm_p50 | 0.996 | 0.191 | 0.231 | 0.328 | 0.364 | 0.034 | 0.101 | 0.150 | 0.880 |
| BPE x1 / bpe_x1 | 1.452 | 0.275 | 0.296 | 0.299 | 0.510 | 0.248 | 0.009 | 0.363 | 0.973 |
| PMI (post budget) / pmi_post | 1.932 | 0.235 | 0.268 | 0.399 | 0.645 | 0.157 | 0.041 | 0.273 | 0.941 |
| right-branching | 3.303 | 0.466 | 0.397 | 0.249 | 1.000 | 0.116 | 0.018 | 0.000 | 1.000 |
| random binary | 3.303 | 0.275 | 0.355 | 0.402 | 0.338 | 0.229 | 0.243 | 0.345 | 0.739 |

### 1000 sentences learned

| method | brackets/sent | F1 | F1+sv | crossing | xseed F1 | abs(cons-gold) | abs(dec-gold) | consistency | decisive |
|---|---|---|---|---|---|---|---|---|---|
| GOLD (shallow phrases) | - | 1 | 1 | 0 | 1 | 0 | 0 | 0.116 | 0.982 |
| default r185 / post | 2.240 | 0.347 | 0.463 | 0.173 | 0.734 | 0.146 | 0.107 | 0.261 | 0.875 |
| default r185 / chosen | 3.303 | 0.368 | 0.483 | 0.249 | 0.742 | 0.208 | 0.114 | 0.324 | 0.868 |
| parts r196 / post | 2.408 | 0.352 | 0.469 | 0.185 | 0.747 | 0.164 | 0.109 | 0.280 | 0.873 |
| parts r196 / chosen | 3.303 | 0.369 | 0.481 | 0.251 | 0.756 | 0.219 | 0.124 | 0.335 | 0.857 |
| parts+cohesion r196 / post | 2.829 | 0.317 | 0.415 | 0.296 | 0.560 | 0.301 | 0.094 | 0.417 | 0.888 |
| parts+edge_once r196 / post | 1.574 | 0.275 | 0.331 | 0.297 | 0.559 | 0.093 | 0.107 | 0.209 | 0.875 |
| r203 default (rerun) / post | 2.240 | 0.347 | 0.463 | 0.173 | 0.734 | 0.146 | 0.107 | 0.261 | 0.875 |
| r203 default (rerun) / post_bud | 2.107 | 0.344 | 0.456 | 0.166 | 0.733 | 0.131 | 0.104 | 0.247 | 0.878 |
| r203 default (rerun) / gm_bud | 2.107 | 0.344 | 0.456 | 0.166 | 0.733 | 0.131 | 0.104 | 0.247 | 0.878 |
| r203 default (rerun) / gm_p50 | 2.240 | 0.347 | 0.463 | 0.173 | 0.734 | 0.146 | 0.107 | 0.261 | 0.875 |
| r203 parts (rerun) / post | 2.408 | 0.352 | 0.469 | 0.185 | 0.747 | 0.164 | 0.109 | 0.280 | 0.873 |
| r203 parts (rerun) / chosen | 3.303 | 0.369 | 0.481 | 0.251 | 0.756 | 0.219 | 0.124 | 0.335 | 0.857 |
| r203 parts (rerun) / post_bud | 2.107 | 0.344 | 0.452 | 0.169 | 0.733 | 0.132 | 0.103 | 0.248 | 0.879 |
| r203 parts (rerun) / gm_bud | 2.107 | 0.344 | 0.452 | 0.169 | 0.733 | 0.132 | 0.103 | 0.248 | 0.879 |
| r203 parts (rerun) / gm_p50 | 2.408 | 0.352 | 0.469 | 0.185 | 0.747 | 0.164 | 0.109 | 0.280 | 0.873 |
| r203a edge_w=1 / post | 1.720 | 0.290 | 0.360 | 0.258 | 0.596 | 0.109 | 0.109 | 0.225 | 0.873 |
| r203a edge_w=1 / chosen | 3.303 | 0.328 | 0.396 | 0.343 | 0.597 | 0.180 | 0.154 | 0.296 | 0.828 |
| r203a edge_w=1 / post_bud | 2.106 | 0.323 | 0.399 | 0.267 | 0.642 | 0.141 | 0.121 | 0.257 | 0.861 |
| r203a edge_w=1 / gm_bud | 2.107 | 0.321 | 0.381 | 0.281 | 0.607 | 0.128 | 0.129 | 0.244 | 0.853 |
| r203a edge_w=1 / gm_p50 | 1.502 | 0.256 | 0.309 | 0.280 | 0.534 | 0.082 | 0.109 | 0.197 | 0.873 |
| r203b goodman tree / post | 2.408 | 0.352 | 0.469 | 0.185 | 0.747 | 0.164 | 0.109 | 0.280 | 0.873 |
| r203b goodman tree / chosen | 3.303 | 0.369 | 0.487 | 0.247 | 0.765 | 0.221 | 0.120 | 0.337 | 0.862 |
| r203b goodman tree / post_bud | 2.107 | 0.344 | 0.452 | 0.169 | 0.733 | 0.132 | 0.103 | 0.248 | 0.879 |
| r203b goodman tree / gm_bud | 2.107 | 0.344 | 0.452 | 0.169 | 0.733 | 0.132 | 0.103 | 0.248 | 0.879 |
| r203b goodman tree / gm_p50 | 2.408 | 0.352 | 0.469 | 0.185 | 0.747 | 0.164 | 0.109 | 0.280 | 0.873 |
| r203c edge_w=1+goodman / post | 1.720 | 0.290 | 0.360 | 0.258 | 0.596 | 0.109 | 0.109 | 0.225 | 0.873 |
| r203c edge_w=1+goodman / chosen | 3.303 | 0.345 | 0.427 | 0.319 | 0.639 | 0.190 | 0.152 | 0.306 | 0.830 |
| r203c edge_w=1+goodman / post_bud | 2.106 | 0.323 | 0.399 | 0.267 | 0.642 | 0.141 | 0.121 | 0.257 | 0.861 |
| r203c edge_w=1+goodman / gm_bud | 2.107 | 0.321 | 0.381 | 0.281 | 0.607 | 0.128 | 0.129 | 0.244 | 0.853 |
| r203c edge_w=1+goodman / gm_p50 | 1.502 | 0.256 | 0.309 | 0.280 | 0.534 | 0.082 | 0.109 | 0.197 | 0.873 |
| BPE x1 / bpe_x1 | 2.107 | 0.355 | 0.384 | 0.281 | 0.682 | 0.327 | 0.040 | 0.443 | 0.942 |
| PMI (post budget) / pmi_post | 2.412 | 0.289 | 0.374 | 0.343 | 0.736 | 0.306 | 0.107 | 0.421 | 0.874 |
| right-branching | 3.303 | 0.466 | 0.397 | 0.249 | 1.000 | 0.116 | 0.018 | 0.000 | 1.000 |
| random binary | 3.303 | 0.270 | 0.345 | 0.405 | 0.343 | 0.242 | 0.262 | 0.357 | 0.720 |

### 4000 sentences learned

| method | brackets/sent | F1 | F1+sv | crossing | xseed F1 | abs(cons-gold) | abs(dec-gold) | consistency | decisive |
|---|---|---|---|---|---|---|---|---|---|
| GOLD (shallow phrases) | - | 1 | 1 | 0 | 1 | 0 | 0 | 0.116 | 0.982 |
| default r185 / post | 2.564 | 0.362 | 0.468 | 0.206 | 0.801 | 0.156 | 0.109 | 0.272 | 0.873 |
| default r185 / chosen | 3.303 | 0.367 | 0.478 | 0.253 | 0.807 | 0.199 | 0.113 | 0.315 | 0.869 |
| parts r196 / post | 2.600 | 0.357 | 0.466 | 0.209 | 0.799 | 0.172 | 0.106 | 0.288 | 0.876 |
| parts r196 / chosen | 3.303 | 0.369 | 0.475 | 0.250 | 0.798 | 0.216 | 0.117 | 0.332 | 0.865 |
| parts+cohesion r196 / post | 2.995 | 0.320 | 0.406 | 0.316 | 0.597 | 0.303 | 0.100 | 0.418 | 0.882 |
| parts+edge_once r196 / post | 1.839 | 0.284 | 0.351 | 0.299 | 0.545 | 0.110 | 0.131 | 0.226 | 0.851 |
| r203 default (rerun) / post | 2.564 | 0.362 | 0.468 | 0.206 | 0.801 | 0.156 | 0.109 | 0.272 | 0.873 |
| r203 default (rerun) / post_bud | 2.635 | 0.363 | 0.471 | 0.209 | 0.810 | 0.164 | 0.109 | 0.280 | 0.873 |
| r203 default (rerun) / gm_bud | 2.635 | 0.363 | 0.471 | 0.208 | 0.810 | 0.164 | 0.109 | 0.280 | 0.873 |
| r203 default (rerun) / gm_p50 | 2.564 | 0.362 | 0.468 | 0.206 | 0.801 | 0.156 | 0.109 | 0.272 | 0.873 |
| r203 parts (rerun) / post | 2.600 | 0.357 | 0.466 | 0.209 | 0.799 | 0.172 | 0.106 | 0.288 | 0.876 |
| r203 parts (rerun) / chosen | 3.303 | 0.369 | 0.475 | 0.250 | 0.798 | 0.216 | 0.117 | 0.332 | 0.865 |
| r203 parts (rerun) / post_bud | 2.635 | 0.358 | 0.468 | 0.210 | 0.801 | 0.175 | 0.106 | 0.291 | 0.876 |
| r203 parts (rerun) / gm_bud | 2.635 | 0.358 | 0.468 | 0.210 | 0.801 | 0.175 | 0.106 | 0.291 | 0.876 |
| r203 parts (rerun) / gm_p50 | 2.600 | 0.357 | 0.466 | 0.209 | 0.799 | 0.172 | 0.106 | 0.288 | 0.876 |
| r203a edge_w=1 / post | 2.070 | 0.278 | 0.382 | 0.262 | 0.670 | 0.141 | 0.117 | 0.257 | 0.865 |
| r203a edge_w=1 / chosen | 3.303 | 0.295 | 0.408 | 0.340 | 0.677 | 0.223 | 0.140 | 0.339 | 0.842 |
| r203a edge_w=1 / post_bud | 2.635 | 0.318 | 0.426 | 0.279 | 0.689 | 0.190 | 0.132 | 0.306 | 0.850 |
| r203a edge_w=1 / gm_bud | 2.635 | 0.307 | 0.419 | 0.288 | 0.685 | 0.190 | 0.133 | 0.306 | 0.848 |
| r203a edge_w=1 / gm_p50 | 2.057 | 0.278 | 0.383 | 0.262 | 0.668 | 0.139 | 0.117 | 0.255 | 0.865 |
| r203b goodman tree / post | 2.600 | 0.357 | 0.466 | 0.209 | 0.799 | 0.172 | 0.106 | 0.288 | 0.876 |
| r203b goodman tree / chosen | 3.303 | 0.366 | 0.478 | 0.250 | 0.814 | 0.222 | 0.109 | 0.338 | 0.873 |
| r203b goodman tree / post_bud | 2.635 | 0.358 | 0.468 | 0.210 | 0.801 | 0.175 | 0.106 | 0.291 | 0.876 |
| r203b goodman tree / gm_bud | 2.635 | 0.358 | 0.468 | 0.210 | 0.801 | 0.175 | 0.106 | 0.291 | 0.876 |
| r203b goodman tree / gm_p50 | 2.600 | 0.357 | 0.466 | 0.209 | 0.799 | 0.172 | 0.106 | 0.288 | 0.876 |
| r203c edge_w=1+goodman / post | 2.070 | 0.278 | 0.382 | 0.262 | 0.670 | 0.141 | 0.117 | 0.257 | 0.865 |
| r203c edge_w=1+goodman / chosen | 3.303 | 0.315 | 0.432 | 0.316 | 0.689 | 0.238 | 0.130 | 0.354 | 0.852 |
| r203c edge_w=1+goodman / post_bud | 2.635 | 0.318 | 0.426 | 0.279 | 0.689 | 0.190 | 0.132 | 0.306 | 0.850 |
| r203c edge_w=1+goodman / gm_bud | 2.635 | 0.307 | 0.419 | 0.288 | 0.685 | 0.190 | 0.133 | 0.306 | 0.848 |
| r203c edge_w=1+goodman / gm_p50 | 2.057 | 0.278 | 0.383 | 0.262 | 0.668 | 0.139 | 0.117 | 0.255 | 0.865 |
| BPE x1 / bpe_x1 | 2.635 | 0.369 | 0.417 | 0.301 | 0.839 | 0.373 | 0.059 | 0.489 | 0.923 |
| PMI (post budget) / pmi_post | 2.600 | 0.294 | 0.454 | 0.302 | 0.846 | 0.357 | 0.109 | 0.473 | 0.873 |
| right-branching | 3.303 | 0.466 | 0.397 | 0.249 | 1.000 | 0.116 | 0.018 | 0.000 | 1.000 |
| random binary | 3.303 | 0.280 | 0.362 | 0.401 | 0.328 | 0.217 | 0.233 | 0.333 | 0.749 |

