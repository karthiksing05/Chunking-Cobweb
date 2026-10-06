Penn Treebank, NLTK sample: trained on 1910 sentences of up to 20 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 34.0 | – | – |
| bigram tag model | – | – | – | 27.3 | – | – |
| TRELLIS v2, tags only (unsupervised) | 53.5% | 66.3% | 28.5% | 26.7 | 22.5 | 46.5 |
| TRELLIS v2, binarized gold trees (supervised) | 14.8% | 38.2% | 12.6% | 26.8 | 16.0 | 67.0 |
