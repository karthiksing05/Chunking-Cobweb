Penn Treebank, NLTK sample: trained on 434 sentences of up to 10 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 33.4 | – | – |
| bigram tag model | – | – | – | 27.2 | – | – |
| TRELLIS v2, tags only (unsupervised) | 50.7% | 64.2% | 29.1% | 26.5 | 5.0 | 2.5 |
| TRELLIS v2, binarized gold trees (supervised) | 16.9% | 39.7% | 15.7% | 26.0 | 9.0 | 24.5 |

| Generated tag sequences (1,000 each) | Of the training length | Real, among those | Every tag triple attested, among those |
|---|---|---|---|
| tag bigram | 66.9% | 7.5% | 82.1% |
| TRELLIS v2, tags only: its own sequences | 13.0% | 28.2% | 92.3% |
| TRELLIS v2, tags only: all samples | 81.8% | 4.7% | 69.1% |
| TRELLIS v2, gold trees: its own sequences | 89.5% | 3.0% | 65.1% |
