Chinese characters (IDS), 2000 training / 500 held-out characters, seeds 13.

| Model | Held-out bits/character | Bracket omission | Bracket commission | Well formed | Components in attested positions | Real: rediscovered held-out | Novel | Symbols | Chunk types |
|---|---|---|---|---|---|---|---|---|---|
| unigram tokens | 44.1 | – | – | – | – | – | – | – | – |
| bigram tokens | 35.0 | – | – | – | – | – | – | – | – |
| TRELLIS v2 from IDS structures, operators as relations | 30.3 | – | – | 100.0% | 83.8% | 5.3% | 77.4% | 21 | 280 |
| TRELLIS v2 from IDS structures, operators as tokens | 24.9 | 0.0% | 0.0% | 97.1% | 70.6% | 7.3% | 56.6% | 27 | 89 |
| TRELLIS v2 from sequences alone | 25.9 | 28.1% | 28.1% | 53.3% | 40.3% | 5.5% | 30.1% | 23 | 54 |
