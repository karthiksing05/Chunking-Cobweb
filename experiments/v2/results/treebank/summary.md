Penn Treebank WSJ10 (NLTK sample, 434 training / 108 test sentences per seed; mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | – | – | – |
| left-branching | 82.9% | 87.6% | – | – | – |
| unigram tag model | – | – | 33.4 | – | – |
| bigram tag model | – | – | 27.2 | – | – |
| TRELLIS v2, tags only (unsupervised) | 51.7% | 64.9% | 30.1 | 12.0 | 15.5 |
| TRELLIS v2, binarized gold trees (supervised) | 19.1% | 41.3% | 30.9 | 9.0 | 25.5 |
