Penn Treebank, NLTK sample: trained on 434 sentences of up to 10 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 33.4 | – | – |
| bigram tag model | – | – | – | 27.2 | – | – |
| TRELLIS v2, tags only (unsupervised) | 55.3% | 67.6% | 33.1% | 30.1 | 11.5 | 11.5 |
| TRELLIS v2, binarized gold trees (supervised) | 19.1% | 41.3% | 20.7% | 30.9 | 9.0 | 25.5 |
