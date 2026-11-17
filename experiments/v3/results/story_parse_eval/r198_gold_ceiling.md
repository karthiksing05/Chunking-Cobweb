Round 198B: gold ceiling for parse consistency / decisive (mean over seeds 13/17/19; [per-seed]).
Strings: 2-4 word proper sub-spans of the 300 held-out sentences with >= 3 occurrences (story_parse_eval's rule); consistency = share of occurrences bracketed; decisive = mean of max(bracketed, unbracketed) share.
Gold = the NLTK shallow-phrase spans (NP/PP/VP/AP/VG, >= 2 words, not the whole sentence) used as the F1 reference.
Recomputed method statistics match the stored metrics (max abs diff 4.9e-05).

### 250 sentences learned

| method | consistency | decisive | abs(consistency - gold) | abs(decisive - gold) |
|---|---|---|---|---|
| GOLD shallow phrases | 0.116 [0.116 / 0.116 / 0.116] | 0.982 [0.982 / 0.982 / 0.982] | 0 | 0 |
| TRELLIS default (r185) / trellis_chosen | 0.358 [0.377 / 0.382 / 0.315] | 0.847 [0.841 / 0.844 / 0.857] | 0.242 | 0.135 |
| TRELLIS default (r185) / trellis_post | 0.234 [0.245 / 0.241 / 0.216] | 0.885 [0.888 / 0.880 / 0.886] | 0.118 | 0.097 |
| TRELLIS parts (r196parts) / trellis_chosen | 0.329 [0.350 / 0.321 / 0.317] | 0.847 [0.844 / 0.839 / 0.859] | 0.213 | 0.135 |
| TRELLIS parts (r196parts) / trellis_post | 0.216 [0.245 / 0.210 / 0.194] | 0.879 [0.871 / 0.882 / 0.884] | 0.100 | 0.103 |
| TRELLIS parts+edge_once (r196pe) / trellis_chosen | 0.281 [0.274 / 0.264 / 0.304] | 0.823 [0.830 / 0.826 / 0.813] | 0.165 | 0.159 |
| TRELLIS parts+edge_once (r196pe) / trellis_post | 0.182 [0.197 / 0.174 / 0.174] | 0.868 [0.873 / 0.862 / 0.868] | 0.066 | 0.114 |
| BPE x1 (r185) / bpe_x1 | 0.364 [0.380 / 0.362 / 0.348] | 0.973 [0.973 / 0.973 / 0.973] | 0.248 | 0.009 |
| right-branching / right_branching | 0.000 [0.000 / 0.000 / 0.000] | 1.000 [1.000 / 1.000 / 1.000] | 0.116 | 0.018 |

### 1000 sentences learned

| method | consistency | decisive | abs(consistency - gold) | abs(decisive - gold) |
|---|---|---|---|---|
| GOLD shallow phrases | 0.116 [0.116 / 0.116 / 0.116] | 0.982 [0.982 / 0.982 / 0.982] | 0 | 0 |
| TRELLIS default (r185) / trellis_chosen | 0.324 [0.339 / 0.319 / 0.313] | 0.868 [0.857 / 0.882 / 0.864] | 0.208 | 0.114 |
| TRELLIS default (r185) / trellis_post | 0.261 [0.283 / 0.263 / 0.239] | 0.875 [0.868 / 0.891 / 0.866] | 0.146 | 0.107 |
| TRELLIS parts (r196parts) / trellis_chosen | 0.335 [0.337 / 0.322 / 0.346] | 0.857 [0.835 / 0.875 / 0.862] | 0.219 | 0.124 |
| TRELLIS parts (r196parts) / trellis_post | 0.280 [0.277 / 0.279 / 0.284] | 0.873 [0.861 / 0.882 / 0.877] | 0.164 | 0.109 |
| TRELLIS parts+edge_once (r196pe) / trellis_chosen | 0.269 [0.290 / 0.257 / 0.261] | 0.839 [0.830 / 0.841 / 0.846] | 0.153 | 0.143 |
| TRELLIS parts+edge_once (r196pe) / trellis_post | 0.209 [0.221 / 0.221 / 0.185] | 0.875 [0.873 / 0.868 / 0.884] | 0.093 | 0.107 |
| BPE x1 (r185) / bpe_x1 | 0.443 [0.442 / 0.458 / 0.428] | 0.942 [0.955 / 0.922 / 0.949] | 0.327 | 0.040 |
| right-branching / right_branching | 0.000 [0.000 / 0.000 / 0.000] | 1.000 [1.000 / 1.000 / 1.000] | 0.116 | 0.018 |

### 4000 sentences learned

| method | consistency | decisive | abs(consistency - gold) | abs(decisive - gold) |
|---|---|---|---|---|
| GOLD shallow phrases | 0.116 [0.116 / 0.116 / 0.116] | 0.982 [0.982 / 0.982 / 0.982] | 0 | 0 |
| TRELLIS default (r185) / trellis_chosen | 0.315 [0.335 / 0.317 / 0.292] | 0.869 [0.855 / 0.882 / 0.870] | 0.199 | 0.113 |
| TRELLIS default (r185) / trellis_post | 0.272 [0.279 / 0.286 / 0.250] | 0.873 [0.871 / 0.879 / 0.870] | 0.156 | 0.109 |
| TRELLIS parts (r196parts) / trellis_chosen | 0.332 [0.322 / 0.337 / 0.335] | 0.865 [0.859 / 0.857 / 0.879] | 0.216 | 0.117 |
| TRELLIS parts (r196parts) / trellis_post | 0.288 [0.275 / 0.288 / 0.301] | 0.876 [0.857 / 0.880 / 0.889] | 0.172 | 0.106 |
| TRELLIS parts+edge_once (r196pe) / trellis_chosen | 0.304 [0.346 / 0.303 / 0.264] | 0.839 [0.844 / 0.824 / 0.848] | 0.188 | 0.143 |
| TRELLIS parts+edge_once (r196pe) / trellis_post | 0.226 [0.234 / 0.234 / 0.210] | 0.851 [0.859 / 0.844 / 0.850] | 0.110 | 0.131 |
| BPE x1 (r185) / bpe_x1 | 0.489 [0.493 / 0.480 / 0.495] | 0.923 [0.924 / 0.922 / 0.922] | 0.373 | 0.059 |
| right-branching / right_branching | 0.000 [0.000 / 0.000 / 0.000] | 1.000 [1.000 / 1.000 / 1.000] | 0.116 | 0.018 |

