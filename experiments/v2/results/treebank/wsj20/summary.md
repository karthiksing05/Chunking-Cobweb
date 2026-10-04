Penn Treebank, NLTK sample: trained on 1910 sentences of up to 20 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 34.0 | – | – |
| bigram tag model | – | – | – | 27.3 | – | – |
| TRELLIS v2, tags only (unsupervised) | 52.1% | 65.3% | 29.5% | 29.6 | 22.5 | 34.0 |
| TRELLIS v2, binarized gold trees (supervised) | 17.5% | 40.1% | 16.4% | 31.6 | 18.5 | 195.5 |
