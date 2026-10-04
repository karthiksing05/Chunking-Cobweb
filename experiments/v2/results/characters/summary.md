Chinese characters (IDS), 2000 training / 500 held-out characters, seeds 13.

| Model | Held-out bits/character | Bracket omission | Bracket commission | Well formed | Components in attested positions | Real: rediscovered held-out | Novel | Symbols | Chunk types |
|---|---|---|---|---|---|---|---|---|---|
| unigram tokens | 44.1 | – | – | – | – | – | – | – | – |
| bigram tokens | 35.0 | – | – | – | – | – | – | – | – |
| TRELLIS v2 from IDS structures, operators as relations | 30.3 | – | – | 100.0% | 83.8% | 5.3% | 77.4% | 21 | 280 |
| TRELLIS v2 from IDS structures, operators as tokens | 31.9 | 0.0% | 0.0% | 95.8% | 52.1% | 2.5% | 48.9% | 36 | 132 |
| TRELLIS v2 from sequences alone | 36.8 | 31.7% | 31.7% | 8.2% | 5.7% | 0.7% | 4.9% | 41 | 67 |
