seed 17, n=1000, 245 word types, 5502 tokens, 245 reservoir types

| nb_bits b | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 245 | 0.176 | 6.89 | no |
| 1 | 1 | 1 | 245 | 0.176 | 6.89 | no |
| 2 | 3 | 3 | 110 | 0.328 | 5.32 | no |
| 3 | 7 | 7 | 49 | 0.385 | 4.46 | no |
| 4 | 16 | 16 | 26 | 0.520 | 3.23 | no |
| 5 | 32 | 27 | 20 | 0.669 | 2.18 | no |
| 6 | 59 | 39 | 20 | 0.758 | 1.40 | no |
| 6.5 | 72 | 40 | 20 | 0.791 | 1.22 | no |
| 7 | 93 | 21 | 20 | 0.823 | 0.94 | pass |
| 7.25 | 93 | 21 | 20 | 0.823 | 0.94 | pass |
| 7.5 | 93 | 21 | 20 | 0.823 | 0.94 | pass |
| 7.75 | 93 | 21 | 20 | 0.823 | 0.94 | pass |
| leaf | 142 | 16 | 20 | 0.880 | 0.54 | pass |
| d1 | 3 | 3 | 110 | 0.328 | 5.32 | no |
| d2 | 7 | 7 | 49 | 0.382 | 4.21 | no |
| d3 | 19 | 18 | 31 | 0.552 | 3.05 | no |
| d4 | 46 | 27 | 26 | 0.690 | 2.09 | no |
| d5 | 91 | 27 | 20 | 0.795 | 1.13 | no |
| d6 | 120 | 21 | 20 | 0.840 | 0.76 | pass |
| d8 | 141 | 16 | 20 | 0.878 | 0.54 | pass |

level leaf, five biggest classes by tokens: was | and | to of named into about | they | he she lucy
level d8, five biggest classes by tokens: was | and | to of named into about | they | he she lucy
level d6, five biggest classes by tokens: was | and | to of named into about | they | he she lucy

passing b: ['7', '7.25', '7.5', '7.75', 'leaf', 'd6', 'd8']
