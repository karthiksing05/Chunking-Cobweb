Penn Treebank, NLTK sample: trained on 1109 sentences of up to 15 tags, tested on 108 held-out WSJ10 sentences (mean over seeds 13,17).

| Model | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram tag model | – | – | – | 33.7 | – | – |
| bigram tag model | – | – | – | 27.1 | – | – |
| TRELLIS v2, tags only (unsupervised) | 48.0% | 62.2% | 25.5% | 26.3 | 15.5 | 11.5 |
| TRELLIS v2, binarized gold trees (supervised) | 16.6% | 39.4% | 15.2% | 26.1 | 13.0 | 43.5 |

| Generated tag sequences (1,000 each) | Of the training length | Real, among those | Every tag triple attested, among those |
|---|---|---|---|
| tag bigram | 71.4% | 3.9% | 77.3% |
| TRELLIS v2, tags only: its own sequences | 81.8% | 24.1% | 94.2% |
| TRELLIS v2, tags only: all samples | 81.0% | 1.8% | 69.7% |
| TRELLIS v2, gold trees: its own sequences | 86.7% | 0.5% | 57.7% |
