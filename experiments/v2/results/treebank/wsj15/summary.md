Penn Treebank, NLTK sample: trained on 1109 sentences of up to 15 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 33.7 | – | – |
| bigram tag model | – | – | – | 27.1 | – | – |
| TRELLIS v2, tags only (unsupervised) | 50.8% | 64.3% | 28.2% | 26.3 | 15.0 | 13.0 |
| TRELLIS v2, binarized gold trees (supervised) | 16.6% | 39.4% | 15.2% | 26.1 | 13.0 | 43.5 |
