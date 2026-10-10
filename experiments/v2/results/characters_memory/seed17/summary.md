# Chinese characters: are the chunks strong? (Track 3)

Seed 17, 2000 learned / 500 held-out characters (`run_characters.py` defaults). Bits per character. Shuffled: atomic components permuted across all slots of the corpus (operator skeletons kept); slot-shuffled: permuted only among slots of the same operator and position.

## 1. Memorization and compression

| Method | real: total | real: held-out | shuffled: total | shuffled: held-out | slot_shuffled: total | slot_shuffled: held-out | real − shuffled (held-out) | real − slot-shuffled (held-out) |
|---|---|---|---|---|---|---|---|---|
| gzip -9 (corpus; held-out = C(train+test) − C(train)) | 46.5 | 40.6 | 68.0 | 64.8 | 60.7 | 56.3 | -24.2 | -15.6 |
| xz -9e (corpus; held-out = C(train+test) − C(train)) | 42.4 | 35.6 | 62.5 | 58.3 | 55.3 | 50.0 | -22.7 | -14.4 |
| raw deflate (corpus; held-out = C(train+test) − C(train)) | 46.4 | 40.6 | 68.0 | 64.8 | 60.6 | 56.3 | -24.2 | -15.6 |
| raw lzma2 (corpus; held-out = C(train+test) − C(train)) | 42.2 | 35.5 | 62.3 | 58.3 | 55.1 | 50.0 | -22.7 | -14.4 |
| gzip -9, each character alone | 338.9 | – | 341.7 | – | 341.5 | – | – | – |
| xz -9e, each character alone | 639.7 | – | 640.3 | – | 640.3 | – | – | – |
| token 1-gram (add-α, α by CV; total = prequential code) | 45.0 | 44.4 | 44.9 | 44.8 | 45.0 | 44.5 | -0.5 | -0.1 |
| token 2-gram (add-α, α by CV; total = prequential code) | 34.4 | 31.3 | 46.0 | 44.1 | 39.7 | 37.0 | -12.8 | -5.7 |
| token 3-gram (add-α, α by CV; total = prequential code) | 34.5 | 29.5 | 52.6 | 50.3 | 46.6 | 43.1 | -20.8 | -13.6 |
| GRU, hidden 8 (total = train NLL + 8-bit parameters) | 45.8 | 29.1 | 55.2 | 37.6 | 50.6 | 33.2 | -8.5 | -4.1 |
| GRU, hidden 16 (total = train NLL + 8-bit parameters) | 62.9 | 26.5 | 76.0 | 37.6 | 69.8 | 31.8 | -11.1 | -5.3 |
| GRU, hidden 32 (total = train NLL + 8-bit parameters) | 110.2 | 24.6 | 126.4 | 37.5 | 120.5 | 31.6 | -12.9 | -7.0 |
| GRU, hidden 64 (total = train NLL + 8-bit parameters) | 245.7 | 23.6 | 265.0 | 37.6 | 258.1 | 31.2 | -14.0 | -7.6 |
| GRU, hidden 128 (total = train NLL + 8-bit parameters) | 667.3 | 22.9 | 689.7 | 37.8 | 682.6 | 31.3 | -14.9 | -8.4 |
| TRELLIS, structures given (operators as relations) (total = model + data) | 34.5 | 30.6 | 38.0 | 37.1 | 34.7 | 31.3 | -6.6 | -0.8 |
| TRELLIS, structures given (operators as tokens) (total = model + data) | 31.0 | 24.7 | 39.1 | 37.6 | 36.1 | 32.9 | -12.9 | -8.1 |
| TRELLIS, sequences alone (total = model + data) | 32.2 | 26.3 | 41.9 | 40.7 | 37.8 | 34.4 | -14.4 | -8.1 |

Held-out gain over the token unigram of the same corpus (bits per character; the unigram codes all three corpora alike, so this separates what a method takes from structure from how badly it overfits a corpus without any):

| Method | real | shuffled | slot-shuffled | real − shuffled | real − slot-shuffled |
|---|---|---|---|---|---|
| xz -9e | 8.8 | -13.5 | -5.5 | 22.3 | 14.3 |
| gzip -9 | 3.8 | -20.0 | -11.8 | 23.7 | 15.5 |
| token 2-gram | 13.1 | 0.8 | 7.5 | 12.3 | 5.6 |
| token 3-gram | 14.9 | -5.4 | 1.4 | 20.3 | 13.4 |
| GRU, hidden 8 | 15.3 | 7.3 | 11.3 | 8.0 | 4.0 |
| GRU, hidden 16 | 17.9 | 7.2 | 12.7 | 10.6 | 5.2 |
| GRU, hidden 32 | 19.8 | 7.3 | 12.9 | 12.5 | 6.9 |
| GRU, hidden 64 | 20.7 | 7.2 | 13.3 | 13.5 | 7.5 |
| GRU, hidden 128 | 21.5 | 7.0 | 13.2 | 14.5 | 8.3 |
| TRELLIS, structures given (operators as relations) | 13.8 | 7.7 | 13.1 | 6.1 | 0.7 |
| TRELLIS, structures given (operators as tokens) | 19.7 | 7.2 | 11.6 | 12.5 | 8.0 |
| TRELLIS, sequences alone | 18.0 | 4.1 | 10.1 | 13.9 | 8.0 |

TRELLIS two-part codes (bits per training character: model + data = total):

| Model | real | shuffled | slot_shuffled |
|---|---|---|---|
| TRELLIS, structures given (operators as relations) | 4.1 + 30.4 = 34.5 (22 symbols, 201 chunk types) | 1.6 + 36.4 = 38.0 (4 symbols, 96 chunk types) | 3.8 + 30.9 = 34.7 (17 symbols, 112 chunk types) |
| TRELLIS, structures given (operators as tokens) | 12.4 + 18.6 = 31.0 (29 symbols, 87 chunk types) | 1.8 + 37.3 = 39.1 (13 symbols, 43 chunk types) | 6.7 + 29.4 = 36.1 (17 symbols, 63 chunk types) |
| TRELLIS, sequences alone | 13.6 + 18.6 = 32.2 (23 symbols, 42 chunk types) | 4.5 + 37.4 = 41.9 (14 symbols, 44 chunk types) | 12.5 + 25.2 = 37.8 (19 symbols, 56 chunk types) |

## 2. Chunk validity

Each chunk type (symbol, left symbol, relation, right symbol) is represented by its modal training yield.

| Model | cond | chunk types | modal yield a database component | operator + leading parts | complete, not listed | crosses components | mean purity | types with purity ≥ 0.5 (of which component) | specific reused unit: purity ≥ 0.5, a component used as a proper part | modal yield only ever a whole character | composite instances that are components |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRELLIS, structures given (operators as relations) | real | 201 | 87.1% | 0.0% | 12.9% | 0.0% | 0.65 | 138 (84.1%) | 48.8% | 20.9% | 91.8% |
| TRELLIS, structures given (operators as relations) | shuffled | 96 | 2.1% | 0.0% | 97.9% | 0.0% | 0.21 | 16 (0.0%) | 0.0% | 31.2% | 4.7% |
| TRELLIS, structures given (operators as relations) | slot_shuffled | 112 | 19.6% | 0.0% | 80.4% | 0.0% | 0.43 | 45 (17.8%) | 7.1% | 28.6% | 15.1% |
| TRELLIS, structures given (operators as tokens) | real | 87 | 19.5% | 79.3% | 1.1% | 0.0% | 0.50 | 40 (17.5%) | 6.9% | 5.7% | 46.4% |
| TRELLIS, structures given (operators as tokens) | shuffled | 43 | 4.7% | 69.8% | 25.6% | 0.0% | 0.38 | 14 (0.0%) | 0.0% | 2.3% | 2.2% |
| TRELLIS, structures given (operators as tokens) | slot_shuffled | 63 | 9.5% | 69.8% | 20.6% | 0.0% | 0.37 | 22 (13.6%) | 4.8% | 1.6% | 7.5% |
| TRELLIS, sequences alone | real | 42 | 38.1% | 23.8% | 0.0% | 38.1% | 0.41 | 15 (33.3%) | 9.5% | 11.9% | 38.1% |
| TRELLIS, sequences alone | shuffled | 44 | 6.8% | 0.0% | 29.5% | 63.6% | 0.11 | 3 (0.0%) | 0.0% | 6.8% | 1.0% |
| TRELLIS, sequences alone | slot_shuffled | 56 | 17.9% | 21.4% | 25.0% | 35.7% | 0.24 | 14 (7.1%) | 1.8% | 5.4% | 5.4% |

Recall of the K most frequent multi-part database components (proper parts of the training characters):

| Model | K | modal yield of a chunk type (any purity) | … with purity ≥ 0.5 | modal yield of a category (any purity) | … with purity ≥ 0.5 | occurrences analysed as one constituent |
|---|---|---|---|---|---|---|
| TRELLIS, structures given (operators as relations) | 50 (50) | 64.0% | 44.0% | 8.0% | 4.0% | 100.0% |
| TRELLIS, structures given (operators as relations) | 100 (100) | 46.0% | 30.0% | 4.0% | 2.0% | 100.0% |
| TRELLIS, structures given (operators as relations) | 200 (200) | 32.5% | 21.5% | 2.0% | 1.0% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 50 (50) | 12.0% | 6.0% | 4.0% | 0.0% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 100 (100) | 6.0% | 3.0% | 2.0% | 0.0% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 200 (200) | 4.5% | 2.5% | 1.5% | 0.5% | 100.0% |
| TRELLIS, sequences alone | 50 (50) | 12.0% | 4.0% | 6.0% | 2.0% | 63.1% |
| TRELLIS, sequences alone | 100 (100) | 7.0% | 2.0% | 3.0% | 1.0% | 60.9% |
| TRELLIS, sequences alone | 200 (200) | 4.0% | 1.0% | 1.5% | 0.5% | 66.5% |

## 3. Completion and cued recall (held-out real characters)

Masked: one atomic component of each held-out character hidden, every learned component ranked by the probability of the filled character (246 candidates; chance top-1 0.4%). Cue: the operator and the first part. Cue bits = −log2 P(character | cue). Open: 100 draws given the cue, ranked by frequency. Closed: the real corpus characters sharing the cue, ranked by probability. Percentile rank: 0 = first, chance 0.5.

| Method | masked top-1 / top-5 | masked percentile rank | cue bits (mean / median) | open: top-1 / top-5 | open: draws real / real unlearned / well formed | closed (all): top-1 / top-5 (chance) | closed: percentile rank |
|---|---|---|---|---|---|---|---|
| token 2-gram | 36.4% / 67.6% | 0.057 | 20.5 / 19.2 | 0.4% / 2.2% | 4.5% / 2.3% / 18.6% | 9.4% / 20.8% (9.1% / 22.0%) | 0.494 |
| token 3-gram | 53.6% / 72.2% | 0.079 | 20.2 / 19.1 | 1.4% / 3.6% | 9.5% / 4.5% / 18.4% | 8.6% / 21.0% (9.1% / 22.0%) | 0.515 |
| GRU, hidden 128 | 57.2% / 71.6% | 0.056 | 14.1 / 12.5 | 2.6% / 5.8% | 24.8% / 11.5% / 86.6% | 7.2% / 21.6% (nan% / nan%) | 0.525 |
| TRELLIS, structures given (operators as relations) | 24.6% / 48.0% | 0.080 | 19.1 / 18.0 | 1.2% / 3.8% | 8.3% / 6.7% / 100.0% | 9.2% / 21.8% (9.1% / 22.0%) | 0.495 |
| TRELLIS, structures given (operators as tokens) | 52.8% / 70.6% | 0.070 | 15.3 / 13.8 | 2.2% / 4.4% | 15.8% / 9.1% / 97.2% | 7.0% / 20.2% (9.1% / 22.0%) | 0.519 |
| TRELLIS, sequences alone | 49.2% / 69.6% | 0.073 | 16.9 / 15.6 | 1.2% / 4.0% | 11.4% / 6.1% / 47.9% | 7.4% / 21.4% (9.1% / 22.0%) | 0.520 |

## Caveats

- One seed (13), one split. No confidence intervals.
- With structures given, every composite is a gold IDS subtree, so a chunk type's modal yield is a database component almost by construction (relational: 91% of composite instances already are). The shuffled rows give the floor; the measurement is informative mainly for sequences alone, and through purity and recall for all modes.
- Recall "modal yield of a chunk type (any purity)" counts a component even when it is a small share of its type's instances (e.g. ⿱十一 at purity 0.01 with operators as tokens); the purity ≥ 0.5 column is the strict version.
- Closed-set cue ranking is at chance for every method, by construction: the candidates are all real characters with the same cue and the held-out truth is a random one of them. Read cue bits instead.
- Relational cue bits condition on a coarser cue (⿲ A is written ⿰ A …, so binary and ternary cues merge), which can only raise its cue bits.
- The n-gram total is the prequential (adaptive add-α) code; the GRU total is training NLL at 8-bit weights plus 8 bits per parameter; TRELLIS's total is its own model + data code. gzip/xz include their headers; per-character compression is dominated by them.

