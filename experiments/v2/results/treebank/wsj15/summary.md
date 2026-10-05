Penn Treebank, NLTK sample: trained on 1109 sentences of up to 15 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 33.7 | – | – |
| bigram tag model | – | – | – | 27.1 | – | – |
| TRELLIS v2, tags only (unsupervised) | 51.9% | 65.1% | 28.2% | 26.7 | 12.5 | 11.0 |
| TRELLIS v2, binarized gold trees (supervised) | 16.4% | 39.3% | 15.2% | 26.8 | 13.0 | 42.0 |
