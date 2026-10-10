# Chinese characters: are the chunks strong? (Track 3)

Seed 23, 2000 learned / 500 held-out characters (`run_characters.py` defaults). Bits per character. Shuffled: atomic components permuted across all slots of the corpus (operator skeletons kept); slot-shuffled: permuted only among slots of the same operator and position.

## 1. Memorization and compression

| Method | real: total | real: held-out | shuffled: total | shuffled: held-out | slot_shuffled: total | slot_shuffled: held-out | real − shuffled (held-out) | real − slot-shuffled (held-out) |
|---|---|---|---|---|---|---|---|---|
| gzip -9 (corpus; held-out = C(train+test) − C(train)) | 47.1 | 42.3 | 68.2 | 64.1 | 60.9 | 57.8 | -21.8 | -15.5 |
| xz -9e (corpus; held-out = C(train+test) − C(train)) | 43.0 | 36.2 | 62.5 | 57.9 | 55.6 | 51.3 | -21.7 | -15.0 |
| raw deflate (corpus; held-out = C(train+test) − C(train)) | 47.0 | 42.3 | 68.1 | 64.1 | 60.8 | 57.8 | -21.8 | -15.5 |
| raw lzma2 (corpus; held-out = C(train+test) − C(train)) | 42.7 | 36.3 | 62.3 | 57.9 | 55.4 | 51.3 | -21.6 | -15.0 |
| gzip -9, each character alone | 339.1 | – | 342.2 | – | 341.6 | – | – | – |
| xz -9e, each character alone | 640.2 | – | 640.7 | – | 640.6 | – | – | – |
| token 1-gram (add-α, α by CV; total = prequential code) | 44.8 | 45.2 | 45.0 | 44.8 | 44.9 | 45.2 | 0.4 | 0.0 |
| token 2-gram (add-α, α by CV; total = prequential code) | 34.9 | 32.1 | 46.2 | 44.2 | 39.9 | 38.1 | -12.1 | -6.0 |
| token 3-gram (add-α, α by CV; total = prequential code) | 35.1 | 30.5 | 52.8 | 50.4 | 47.0 | 43.9 | -19.9 | -13.4 |
| GRU, hidden 8 (total = train NLL + 8-bit parameters) | 46.5 | 29.9 | 56.0 | 37.8 | 50.5 | 33.4 | -7.9 | -3.5 |
| GRU, hidden 16 (total = train NLL + 8-bit parameters) | 63.3 | 27.6 | 76.9 | 37.4 | 71.1 | 33.0 | -9.8 | -5.5 |
| GRU, hidden 32 (total = train NLL + 8-bit parameters) | 112.1 | 25.4 | 128.3 | 37.4 | 121.6 | 32.3 | -12.0 | -6.9 |
| GRU, hidden 64 (total = train NLL + 8-bit parameters) | 248.8 | 24.8 | 268.9 | 37.4 | 261.7 | 32.2 | -12.6 | -7.4 |
| GRU, hidden 128 (total = train NLL + 8-bit parameters) | 674.0 | 23.6 | 697.4 | 37.5 | 687.2 | 32.1 | -13.8 | -8.5 |
| TRELLIS, structures given (operators as relations) (total = model + data) | 34.4 | 31.7 | 38.1 | 36.8 | 34.7 | 32.2 | -5.1 | -0.5 |
| TRELLIS, structures given (operators as tokens) (total = model + data) | 31.5 | 25.7 | 39.3 | 37.6 | 36.0 | 33.7 | -11.9 | -8.0 |
| TRELLIS, sequences alone (total = model + data) | 32.3 | 26.8 | 41.9 | 40.5 | 38.9 | 35.9 | -13.6 | -9.0 |

Held-out gain over the token unigram of the same corpus (bits per character; the unigram codes all three corpora alike, so this separates what a method takes from structure from how badly it overfits a corpus without any):

| Method | real | shuffled | slot-shuffled | real − shuffled | real − slot-shuffled |
|---|---|---|---|---|---|
| xz -9e | 9.0 | -13.1 | -6.1 | 22.1 | 15.1 |
| gzip -9 | 3.0 | -19.3 | -12.6 | 22.2 | 15.6 |
| token 2-gram | 13.2 | 0.6 | 7.1 | 12.5 | 6.0 |
| token 3-gram | 14.7 | -5.6 | 1.3 | 20.4 | 13.4 |
| GRU, hidden 8 | 15.3 | 7.0 | 11.8 | 8.3 | 3.6 |
| GRU, hidden 16 | 17.7 | 7.4 | 12.2 | 10.2 | 5.5 |
| GRU, hidden 32 | 19.8 | 7.4 | 12.9 | 12.4 | 7.0 |
| GRU, hidden 64 | 20.5 | 7.4 | 13.0 | 13.1 | 7.4 |
| GRU, hidden 128 | 21.6 | 7.3 | 13.1 | 14.3 | 8.5 |
| TRELLIS, structures given (operators as relations) | 13.5 | 7.9 | 13.0 | 5.6 | 0.5 |
| TRELLIS, structures given (operators as tokens) | 19.6 | 7.2 | 11.5 | 12.3 | 8.0 |
| TRELLIS, sequences alone | 18.4 | 4.3 | 9.4 | 14.1 | 9.1 |

TRELLIS two-part codes (bits per training character: model + data = total):

| Model | real | shuffled | slot_shuffled |
|---|---|---|---|
| TRELLIS, structures given (operators as relations) | 3.6 + 30.8 = 34.4 (18 symbols, 136 chunk types) | 1.8 + 36.3 = 38.1 (5 symbols, 114 chunk types) | 3.6 + 31.1 = 34.7 (14 symbols, 107 chunk types) |
| TRELLIS, structures given (operators as tokens) | 13.0 + 18.5 = 31.5 (29 symbols, 113 chunk types) | 2.0 + 37.3 = 39.3 (15 symbols, 47 chunk types) | 6.6 + 29.3 = 36.0 (17 symbols, 52 chunk types) |
| TRELLIS, sequences alone | 13.3 + 19.0 = 32.3 (29 symbols, 57 chunk types) | 4.7 + 37.2 = 41.9 (15 symbols, 44 chunk types) | 8.2 + 30.7 = 38.9 (20 symbols, 51 chunk types) |

## 2. Chunk validity

Each chunk type (symbol, left symbol, relation, right symbol) is represented by its modal training yield.

| Model | cond | chunk types | modal yield a database component | operator + leading parts | complete, not listed | crosses components | mean purity | types with purity ≥ 0.5 (of which component) | specific reused unit: purity ≥ 0.5, a component used as a proper part | modal yield only ever a whole character | composite instances that are components |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRELLIS, structures given (operators as relations) | real | 136 | 89.7% | 0.0% | 10.3% | 0.0% | 0.58 | 82 (87.8%) | 49.3% | 17.6% | 91.1% |
| TRELLIS, structures given (operators as relations) | shuffled | 114 | 2.6% | 0.0% | 97.4% | 0.0% | 0.23 | 25 (0.0%) | 0.0% | 26.3% | 4.6% |
| TRELLIS, structures given (operators as relations) | slot_shuffled | 107 | 20.6% | 0.0% | 79.4% | 0.0% | 0.35 | 34 (20.6%) | 6.5% | 23.4% | 15.1% |
| TRELLIS, structures given (operators as tokens) | real | 113 | 23.0% | 72.6% | 4.4% | 0.0% | 0.60 | 70 (21.4%) | 13.3% | 0.9% | 46.0% |
| TRELLIS, structures given (operators as tokens) | shuffled | 47 | 4.3% | 83.0% | 12.8% | 0.0% | 0.28 | 12 (0.0%) | 0.0% | 2.1% | 2.2% |
| TRELLIS, structures given (operators as tokens) | slot_shuffled | 52 | 11.5% | 73.1% | 15.4% | 0.0% | 0.24 | 11 (0.0%) | 0.0% | 3.8% | 7.5% |
| TRELLIS, sequences alone | real | 57 | 36.8% | 24.6% | 0.0% | 38.6% | 0.48 | 24 (25.0%) | 7.0% | 12.3% | 38.3% |
| TRELLIS, sequences alone | shuffled | 44 | 6.8% | 0.0% | 27.3% | 65.9% | 0.11 | 3 (0.0%) | 0.0% | 9.1% | 0.9% |
| TRELLIS, sequences alone | slot_shuffled | 51 | 13.7% | 13.7% | 29.4% | 43.1% | 0.29 | 12 (0.0%) | 0.0% | 11.8% | 4.4% |

Recall of the K most frequent multi-part database components (proper parts of the training characters):

| Model | K | modal yield of a chunk type (any purity) | … with purity ≥ 0.5 | modal yield of a category (any purity) | … with purity ≥ 0.5 | occurrences analysed as one constituent |
|---|---|---|---|---|---|---|
| TRELLIS, structures given (operators as relations) | 50 (50) | 42.0% | 22.0% | 4.0% | 0.0% | 100.0% |
| TRELLIS, structures given (operators as relations) | 100 (100) | 31.0% | 15.0% | 2.0% | 0.0% | 100.0% |
| TRELLIS, structures given (operators as relations) | 200 (200) | 22.0% | 11.5% | 1.0% | 0.0% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 50 (50) | 20.0% | 8.0% | 6.0% | 0.0% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 100 (100) | 13.0% | 6.0% | 4.0% | 1.0% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 200 (200) | 7.0% | 3.5% | 2.0% | 0.5% | 100.0% |
| TRELLIS, sequences alone | 50 (50) | 16.0% | 4.0% | 4.0% | 2.0% | 61.2% |
| TRELLIS, sequences alone | 100 (100) | 9.0% | 3.0% | 3.0% | 2.0% | 66.6% |
| TRELLIS, sequences alone | 200 (200) | 5.5% | 1.5% | 1.5% | 1.0% | 69.4% |

## 3. Completion and cued recall (held-out real characters)

Masked: one atomic component of each held-out character hidden, every learned component ranked by the probability of the filled character (246 candidates; chance top-1 0.4%). Cue: the operator and the first part. Cue bits = −log2 P(character | cue). Open: 100 draws given the cue, ranked by frequency. Closed: the real corpus characters sharing the cue, ranked by probability. Percentile rank: 0 = first, chance 0.5.

| Method | masked top-1 / top-5 | masked percentile rank | cue bits (mean / median) | open: top-1 / top-5 | open: draws real / real unlearned / well formed | closed (all): top-1 / top-5 (chance) | closed: percentile rank |
|---|---|---|---|---|---|---|---|
| token 2-gram | 38.4% / 67.2% | 0.064 | 21.4 / 19.7 | 0.4% / 2.0% | 4.9% / 2.2% / 19.2% | 9.2% / 19.8% (10.0% / 22.5%) | 0.526 |
| token 3-gram | 53.0% / 71.8% | 0.079 | 21.3 / 20.0 | 1.4% / 3.6% | 8.3% / 3.5% / 17.6% | 9.8% / 19.8% (10.0% / 22.5%) | 0.542 |
| GRU, hidden 128 | 57.6% / 73.4% | 0.052 | 14.9 / 13.2 | 2.0% / 4.6% | 23.9% / 11.2% / 85.0% | 10.0% / 19.4% (nan% / nan%) | 0.554 |
| TRELLIS, structures given (operators as relations) | 24.6% / 48.0% | 0.083 | 20.2 / 18.4 | 0.2% / 1.4% | 6.6% / 5.5% / 100.0% | 10.2% / 20.4% (10.0% / 22.5%) | 0.514 |
| TRELLIS, structures given (operators as tokens) | 49.2% / 69.8% | 0.066 | 16.4 / 14.1 | 1.2% / 2.2% | 14.3% / 6.0% / 94.4% | 9.4% / 19.6% (10.0% / 22.5%) | 0.531 |
| TRELLIS, sequences alone | 53.0% / 70.4% | 0.062 | 17.6 / 15.9 | 1.4% / 2.2% | 10.5% / 5.6% / 52.8% | 8.8% / 19.6% (10.0% / 22.5%) | 0.538 |

## Caveats

- One seed (13), one split. No confidence intervals.
- With structures given, every composite is a gold IDS subtree, so a chunk type's modal yield is a database component almost by construction (relational: 91% of composite instances already are). The shuffled rows give the floor; the measurement is informative mainly for sequences alone, and through purity and recall for all modes.
- Recall "modal yield of a chunk type (any purity)" counts a component even when it is a small share of its type's instances (e.g. ⿱十一 at purity 0.01 with operators as tokens); the purity ≥ 0.5 column is the strict version.
- Closed-set cue ranking is at chance for every method, by construction: the candidates are all real characters with the same cue and the held-out truth is a random one of them. Read cue bits instead.
- Relational cue bits condition on a coarser cue (⿲ A is written ⿰ A …, so binary and ternary cues merge), which can only raise its cue bits.
- The n-gram total is the prequential (adaptive add-α) code; the GRU total is training NLL at 8-bit weights plus 8 bits per parameter; TRELLIS's total is its own model + data code. gzip/xz include their headers; per-character compression is dominated by them.

