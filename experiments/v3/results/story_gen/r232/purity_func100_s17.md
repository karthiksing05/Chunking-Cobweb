seed 17, n=1000, 245 word types, 5502 tokens, 245 reservoir types

| nb_bits b | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 245 | 0.176 | 6.89 | no |
| 1 | 2 | 2 | 136 | 0.299 | 5.89 | no |
| 2 | 4 | 4 | 94 | 0.364 | 5.03 | no |
| 3 | 8 | 8 | 48 | 0.418 | 4.46 | no |
| 4 | 12 | 12 | 42 | 0.461 | 3.82 | no |
| 5 | 29 | 29 | 22 | 0.634 | 2.70 | no |
| 6 | 49 | 43 | 15 | 0.676 | 2.09 | no |
| 6.5 | 73 | 61 | 8 | 0.750 | 1.47 | no |
| 7 | 110 | 35 | 5 | 0.860 | 0.94 | pass |
| 7.25 | 110 | 35 | 5 | 0.860 | 0.94 | pass |
| 7.5 | 110 | 35 | 5 | 0.860 | 0.94 | pass |
| 7.75 | 110 | 35 | 5 | 0.860 | 0.94 | pass |
| leaf | 240 | 1 | 4 | 0.966 | 0.00 | no |
| d1 | 3 | 3 | 136 | 0.364 | 5.42 | no |
| d2 | 11 | 9 | 94 | 0.403 | 4.41 | no |
| d3 | 38 | 17 | 46 | 0.490 | 3.42 | no |
| d4 | 91 | 21 | 41 | 0.656 | 2.35 | no |
| d5 | 153 | 18 | 14 | 0.805 | 1.25 | pass |
| d6 | 202 | 9 | 8 | 0.878 | 0.61 | pass |
| d8 | 238 | 2 | 4 | 0.964 | 0.04 | no |

level d8, five biggest classes by tokens: was | and | they | the | to
level d6, five biggest classes by tokens: was and to felt | happy sad scared excited surprised sorry safe shiny | fun friends day that now some soon mommy | they | the
level 7, five biggest classes by tokens: was and | they let's | he she bob | happy excited surprised | to felt

passing b: ['7', '7.25', '7.5', '7.75', 'd5', 'd6']
