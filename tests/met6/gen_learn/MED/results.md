# gen_learn_test — MED (τ=2, n_train=150, epochs=6)

## Head-to-head vs supervised reference

| metric | SUPERVISED | UNSUPERVISED |
|---|--:|--:|
| parse F1 | 84.6% | 46.3% |
| exact-match | 50.0% | 27.5% |
| generation grammatical | 88.3% | 98.3% |
| generation novel | 73.3% | 38.3% |

## Nonterminal analysis (unsupervised)

- **17 nonterminals** discovered (content basic-level categories)
- representation consistency (one meaning per NT): **49.3%**
- expansion consistency (one production per NT): **55.0%**
- assignment consistency (one NT per surface POS-span): **86.0%**

| nonterminal | uses | dominant POS-span | repr% | #spans | expand% | #prods |
|---|--:|---|--:|--:|--:|--:|
| 40850697 | 263 | `Det N` | 48% | 10 | 48% | 10 |
| 40850579 | 195 | `Det Adj` | 78% | 8 | 78% | 9 |
| 40852067 | 174 | `Det N V` | 43% | 5 | 71% | 5 |
| 40858988 | 59 | `Det N V Det Adj Adj N` | 10% | 37 | 19% | 19 |
| 40859450 | 56 | `Det Adj Adj N V` | 88% | 6 | 79% | 5 |
| 40854026 | 47 | `Det N V Det Adj` | 28% | 17 | 40% | 13 |
| 40850645 | 46 | `Adj Adj N` | 37% | 13 | 39% | 13 |
| 40850584 | 46 | `Det Adj Adj N` | 48% | 19 | 43% | 17 |
| 40852164 | 43 | `Det Adj Adj` | 23% | 17 | 26% | 13 |
| 40850570 | 30 | `N V` | 47% | 5 | 47% | 5 |
| 40851817 | 28 | `Det N V Det N` | 36% | 7 | 46% | 8 |
| 40850576 | 21 | `Det Adj Adj N V Det Adj Adj Adj N` | 10% | 18 | 10% | 17 |
| 40850760 | 8 | `Det N V Det Adj Adj N P Det Adj Adj Adj N` | 25% | 6 | 38% | 6 |
| 40850570 | 4 | `Det Adj Adj N` | 100% | 1 | 75% | 2 |
| 40851247 | 2 | `Det Adj Adj N V` | 50% | 2 | 50% | 2 |
| 40852517 | 2 | `Det N V Det Adj Adj N` | 100% | 1 | 100% | 1 |
| 40858225 | 1 | `Det Adj Adj N V Det Adj Adj Adj N P Det N` | 100% | 1 | 100% | 1 |

## Sample generated sentences (unsupervised)

- `a woman liked the big quick park under the woman`  (grammatical)  → `trees/gen0.png`
- `a quick big cat chased`  (grammatical, novel)  → `trees/gen1.png`
- `a red red big woman saw a woman under the dog`  (grammatical, novel)  → `trees/gen2.png`
- `the small red man found`  (grammatical)  → `trees/gen3.png`

## Test parse trees (unsupervised vs gold)

- `the man chased a park with the park`  →  unsup `trees/test0_unsup.png`  vs  gold `trees/test0_gold.png`
- `the park liked the big small woman with the small red man`  →  unsup `trees/test1_unsup.png`  vs  gold `trees/test1_gold.png`
- `the dog liked the small quick cat in a telescope`  →  unsup `trees/test2_unsup.png`  vs  gold `trees/test2_gold.png`
- `a dog admired a quick quick big park`  →  unsup `trees/test3_unsup.png`  vs  gold `trees/test3_gold.png`

## Summary graphics

- `summary.png` — head-to-head + nonterminal analysis
- `trees_montage.png` — all parse trees in one figure
