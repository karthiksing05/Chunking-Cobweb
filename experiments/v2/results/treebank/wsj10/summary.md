Penn Treebank, NLTK sample: trained on 434 sentences of up to 10 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 33.4 | – | – |
| bigram tag model | – | – | – | 27.2 | – | – |
| TRELLIS v2, tags only (unsupervised) | 49.0% | 63.0% | 27.8% | 26.8 | 7.0 | 4.5 |
| TRELLIS v2, binarized gold trees (supervised) | 16.9% | 39.7% | 15.7% | 26.0 | 9.0 | 24.5 |

| Generated tag sequences (1,000 each) | Of the training length | Real, among those | Every tag triple attested, among those |
|---|---|---|---|
| tag bigram | 66.9% | 7.5% | 82.1% |
| TRELLIS v2, tags only: its own sequences | 77.5% | 28.8% | 93.8% |
| TRELLIS v2, tags only: all samples | 80.7% | 5.1% | 68.4% |
| TRELLIS v2, gold trees: its own sequences | 89.1% | 2.8% | 64.5% |
