Penn Treebank, NLTK sample: trained on 434 sentences of up to 10 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 33.4 | – | – |
| bigram tag model | – | – | – | 27.2 | – | – |
| TRELLIS v2, tags only (unsupervised) | 48.0% | 62.3% | 24.7% | 26.6 | 6.5 | 3.5 |
| TRELLIS v2, binarized gold trees (supervised) | 16.9% | 39.7% | 15.7% | 26.0 | 9.0 | 24.5 |
