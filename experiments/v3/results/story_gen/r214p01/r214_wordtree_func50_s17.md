seed 17, n=1000, 245 word types, 5502 tokens, 245 reservoir types

| nb_bits b | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 245 | 0.176 | 6.89 | no |
| 1 | 2 | 2 | 129 | 0.296 | 5.89 | no |
| 2 | 4 | 4 | 104 | 0.360 | 5.18 | no |
| 3 | 7 | 7 | 60 | 0.407 | 4.49 | no |
| 4 | 13 | 13 | 34 | 0.497 | 3.46 | no |
| 5 | 25 | 25 | 22 | 0.583 | 2.79 | no |
| 6 | 55 | 44 | 10 | 0.739 | 1.82 | no |
| 6.5 | 79 | 56 | 6 | 0.814 | 1.32 | pass |
| 7 | 105 | 38 | 5 | 0.855 | 0.97 | pass |
| 7.25 | 105 | 38 | 5 | 0.855 | 0.97 | pass |
| 7.5 | 105 | 38 | 5 | 0.855 | 0.97 | pass |
| 7.75 | 105 | 38 | 5 | 0.855 | 0.97 | pass |
| leaf | 235 | 3 | 4 | 0.964 | 0.01 | no |
| d1 | 3 | 3 | 129 | 0.360 | 5.42 | no |
| d2 | 10 | 10 | 104 | 0.405 | 4.62 | no |
| d3 | 37 | 20 | 60 | 0.500 | 3.50 | no |
| d4 | 90 | 26 | 29 | 0.700 | 2.06 | no |
| d5 | 157 | 19 | 22 | 0.778 | 1.22 | no |
| d6 | 198 | 12 | 7 | 0.919 | 0.41 | pass |
| d8 | 234 | 3 | 4 | 0.964 | 0.02 | no |

level leaf, five biggest classes by tokens: was | and | they | the | to
level d8, five biggest classes by tokens: was | and | they | the | to
level d6, five biggest classes by tokens: was saw asked found thought says made | to is will for just | and thanked took gave | they | the

passing b: ['6.5', '7', '7.25', '7.5', '7.75', 'd6']
