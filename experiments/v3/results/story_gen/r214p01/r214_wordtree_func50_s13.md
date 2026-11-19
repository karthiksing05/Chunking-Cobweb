seed 13, n=1000, 239 word types, 5306 tokens, 239 reservoir types

| nb_bits b | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 239 | 0.177 | 6.94 | no |
| 1 | 1 | 1 | 239 | 0.177 | 6.94 | no |
| 2 | 5 | 5 | 76 | 0.403 | 5.01 | no |
| 3 | 8 | 8 | 47 | 0.442 | 4.34 | no |
| 4 | 15 | 12 | 32 | 0.502 | 3.64 | no |
| 5 | 26 | 23 | 20 | 0.594 | 2.82 | no |
| 6 | 60 | 46 | 10 | 0.722 | 1.74 | no |
| 6.5 | 78 | 52 | 7 | 0.753 | 1.46 | no |
| 7 | 109 | 32 | 5 | 0.825 | 0.97 | pass |
| 7.25 | 109 | 32 | 5 | 0.825 | 0.97 | pass |
| 7.5 | 109 | 32 | 5 | 0.825 | 0.97 | pass |
| 7.75 | 109 | 32 | 5 | 0.825 | 0.97 | pass |
| leaf | 229 | 5 | 3 | 0.962 | 0.01 | no |
| d1 | 3 | 3 | 116 | 0.400 | 5.42 | no |
| d2 | 9 | 7 | 100 | 0.430 | 4.80 | no |
| d3 | 29 | 17 | 68 | 0.475 | 3.85 | no |
| d4 | 68 | 21 | 31 | 0.619 | 2.65 | no |
| d5 | 127 | 19 | 29 | 0.705 | 1.87 | no |
| d6 | 167 | 17 | 10 | 0.846 | 0.94 | pass |
| d8 | 224 | 6 | 3 | 0.939 | 0.11 | no |

level leaf, five biggest classes by tokens: they | and | was | the | to
level d8, five biggest classes by tokens: and said is | they | was | the | to
level d6, five biggest classes by tokens: and a said is mom liked good will | happy sad scared excited surprised proud fast small | he she tim amy lucy | they spot | lily tom sue max mia anna

passing b: ['7', '7.25', '7.5', '7.75', 'd6']
