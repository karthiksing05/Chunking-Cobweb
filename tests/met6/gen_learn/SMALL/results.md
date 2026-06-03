# gen_learn_test — SMALL (τ=8, n_train=150, epochs=6)

## Head-to-head vs supervised reference

| metric | SUPERVISED | UNSUPERVISED |
|---|--:|--:|
| parse F1 | 100.0% | 99.3% |
| exact-match | 100.0% | 97.5% |
| generation grammatical | 100.0% | 100.0% |
| generation novel | 13.3% | 21.7% |

## Nonterminal analysis (unsupervised)

- **7 nonterminals** discovered (content basic-level categories)
- representation consistency (one meaning per NT): **100.0%**
- expansion consistency (one production per NT): **95.7%**
- assignment consistency (one NT per surface POS-span): **61.4%**

| nonterminal | uses | dominant POS-span | repr% | #spans | expand% | #prods |
|---|--:|---|--:|--:|--:|--:|
| 40847270 | 150 | `Det N` | 100% | 1 | 100% | 1 |
| 40847270 | 125 | `Det N V` | 100% | 1 | 100% | 1 |
| 40847283 | 65 | `Det N V Det N` | 100% | 1 | 89% | 4 |
| 40847269 | 64 | `Det N` | 100% | 1 | 100% | 1 |
| 40847290 | 63 | `Det N` | 100% | 1 | 100% | 1 |
| 40847272 | 62 | `Det N V Det N` | 100% | 1 | 73% | 4 |
| 40847279 | 25 | `Det N V` | 100% | 1 | 100% | 1 |

## Sample generated sentences (unsupervised)

- `a woman runs the man`  (grammatical)  → `trees/gen0.png`
- `the man likes a woman`  (grammatical)  → `trees/gen1.png`
- `the dog sees a dog`  (grammatical, novel)  → `trees/gen2.png`
- `the man likes a man`  (grammatical)  → `trees/gen3.png`

## Test parse trees (unsupervised vs gold)

- `the man chases the woman`  →  unsup `trees/test0_unsup.png`  vs  gold `trees/test0_gold.png`
- `a woman likes a dog`  →  unsup `trees/test1_unsup.png`  vs  gold `trees/test1_gold.png`
- `a woman likes the cat`  →  unsup `trees/test2_unsup.png`  vs  gold `trees/test2_gold.png`
- `the dog runs the cat`  →  unsup `trees/test3_unsup.png`  vs  gold `trees/test3_gold.png`

## Summary graphics

- `summary.png` — head-to-head + nonterminal analysis
- `trees_montage.png` — all parse trees in one figure
