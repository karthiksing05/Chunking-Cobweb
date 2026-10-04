Penn Treebank, NLTK sample: trained on 1109 sentences of up to 15 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 33.7 | – | – |
| bigram tag model | – | – | – | 27.1 | – | – |
| TRELLIS v2, tags only (unsupervised) | 51.0% | 64.5% | 24.9% | 29.7 | 16.5 | 24.5 |
| TRELLIS v2, binarized gold trees (supervised) | 17.9% | 40.4% | 18.4% | 31.1 | 14.5 | 59.5 |
