Chinese characters (IDS), 2000 training / 500 held-out characters, seeds 13.

| Model | Held-out bits/character | Bracket omission | Bracket commission | Well formed | Components in attested positions | Real: rediscovered held-out | Novel | Symbols | Chunk types |
|---|---|---|---|---|---|---|---|---|---|
| unigram tokens | 44.1 | – | – | – | – | – | – | – | – |
| bigram tokens | 35.0 | – | – | – | – | – | – | – | – |
| TRELLIS v2 from IDS structures, operators as relations | 30.3 | – | – | 100.0% | 83.8% | 5.3% | 77.4% | 21 | 280 |
| TRELLIS v2 from IDS structures, operators as tokens | 27.2 | 0.0% | 0.0% | 98.0% | 61.5% | 5.2% | 53.1% | 27 | 76 |
| TRELLIS v2 from sequences alone | 30.5 | 33.8% | 33.8% | 15.3% | 12.1% | 1.3% | 9.7% | 39 | 63 |
