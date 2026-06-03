# gen_learn_test — SMALL (τ=8, n_train=150, epochs=6)

## Head-to-head vs supervised reference

| metric | SUPERVISED | UNSUPERVISED |
|---|--:|--:|
| parse F1 | 100.0% | 99.3% |
| exact-match | 100.0% | 97.5% |
| generation grammatical | 100.0% | 100.0% |
| generation novel | 13.3% | 21.7% |

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
