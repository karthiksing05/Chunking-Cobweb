seed 13, n=1000, 239 word types, 5306 tokens, 239 reservoir types

| nb_bits b | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 239 | 0.177 | 6.94 | no |
| 1 | 2 | 2 | 130 | 0.195 | 6.06 | no |
| 2 | 3 | 3 | 130 | 0.203 | 5.76 | no |
| 3 | 7 | 7 | 46 | 0.360 | 4.47 | no |
| 4 | 20 | 15 | 27 | 0.562 | 3.02 | no |
| 5 | 35 | 25 | 27 | 0.665 | 2.26 | no |
| 6 | 59 | 31 | 27 | 0.750 | 1.54 | no |
| 6.5 | 70 | 37 | 27 | 0.778 | 1.37 | no |
| 7 | 92 | 20 | 27 | 0.806 | 1.04 | pass |
| 7.25 | 92 | 20 | 27 | 0.806 | 1.04 | pass |
| 7.5 | 92 | 20 | 27 | 0.806 | 1.04 | pass |
| 7.75 | 92 | 20 | 27 | 0.806 | 1.04 | pass |
| leaf | 141 | 10 | 27 | 0.868 | 0.64 | pass |
| d1 | 3 | 3 | 130 | 0.203 | 5.76 | no |
| d2 | 8 | 8 | 44 | 0.423 | 4.29 | no |
| d3 | 18 | 16 | 29 | 0.541 | 3.14 | no |
| d4 | 44 | 23 | 28 | 0.684 | 2.20 | no |
| d5 | 81 | 23 | 27 | 0.753 | 1.54 | no |
| d6 | 120 | 12 | 27 | 0.826 | 0.98 | pass |
| d8 | 140 | 11 | 27 | 0.868 | 0.65 | pass |

level leaf, five biggest classes by tokens: day too good friend idea much away careful | they | to of for near from gave into | and | he she everyone who
level d8, five biggest classes by tokens: day too good friend idea much away careful | they | to of for near from gave into | and | he she everyone who
level d6, five biggest classes by tokens: they you can yes no will me that | day too happened good friend idea much away | lily tom sue max mia anna | to of for near from gave into | and

passing b: ['7', '7.25', '7.5', '7.75', 'leaf', 'd6', 'd8']
