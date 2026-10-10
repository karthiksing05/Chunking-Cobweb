# Chinese characters: are the chunks strong? (Track 3)

Seed 13, 2000 learned / 500 held-out characters (`run_characters.py` defaults). Bits per character. Shuffled: atomic components permuted across all slots of the corpus (operator skeletons kept); slot-shuffled: permuted only among slots of the same operator and position.

## 1. Memorization and compression

| Method | real: total | real: held-out | shuffled: total | shuffled: held-out | slot_shuffled: total | slot_shuffled: held-out | real − shuffled (held-out) | real − slot-shuffled (held-out) |
|---|---|---|---|---|---|---|---|---|
| gzip -9 (corpus; held-out = C(train+test) − C(train)) | 46.9 | 40.5 | 68.5 | 62.9 | 61.0 | 55.6 | -22.4 | -15.2 |
| xz -9e (corpus; held-out = C(train+test) − C(train)) | 42.9 | 35.0 | 62.9 | 56.6 | 55.7 | 49.5 | -21.6 | -14.5 |
| raw deflate (corpus; held-out = C(train+test) − C(train)) | 46.8 | 40.5 | 68.4 | 62.9 | 61.0 | 55.6 | -22.4 | -15.2 |
| raw lzma2 (corpus; held-out = C(train+test) − C(train)) | 42.7 | 35.0 | 62.6 | 56.6 | 55.5 | 49.5 | -21.6 | -14.5 |
| gzip -9, each character alone | 339.8 | – | 342.3 | – | 341.8 | – | – | – |
| xz -9e, each character alone | 640.0 | – | 640.7 | – | 640.8 | – | – | – |
| token 1-gram (add-α, α by CV; total = prequential code) | 45.2 | 44.1 | 45.3 | 43.5 | 45.2 | 43.8 | 0.6 | 0.2 |
| token 2-gram (add-α, α by CV; total = prequential code) | 34.9 | 31.2 | 46.4 | 43.1 | 40.0 | 36.8 | -11.9 | -5.6 |
| token 3-gram (add-α, α by CV; total = prequential code) | 35.2 | 29.0 | 53.1 | 49.6 | 46.9 | 42.7 | -20.6 | -13.6 |
| GRU, hidden 8 (total = train NLL + 8-bit parameters) | 46.5 | 29.4 | 56.9 | 37.1 | 50.6 | 32.3 | -7.7 | -2.9 |
| GRU, hidden 16 (total = train NLL + 8-bit parameters) | 64.4 | 26.6 | 77.6 | 36.4 | 71.2 | 31.6 | -9.8 | -4.9 |
| GRU, hidden 32 (total = train NLL + 8-bit parameters) | 113.4 | 24.2 | 129.8 | 36.3 | 122.3 | 31.0 | -12.0 | -6.8 |
| GRU, hidden 64 (total = train NLL + 8-bit parameters) | 253.7 | 23.1 | 271.4 | 36.3 | 263.1 | 31.1 | -13.2 | -8.0 |
| GRU, hidden 128 (total = train NLL + 8-bit parameters) | 683.1 | 22.7 | 702.4 | 36.5 | 692.2 | 31.1 | -13.8 | -8.4 |
| TRELLIS, structures given (operators as relations) (total = model + data) | 34.5 | 30.3 | 38.4 | 35.8 | 35.1 | 31.2 | -5.5 | -0.9 |
| TRELLIS, structures given (operators as tokens) (total = model + data) | 31.7 | 24.9 | 39.5 | 36.6 | 36.9 | 34.0 | -11.7 | -9.1 |
| TRELLIS, sequences alone (total = model + data) | 32.0 | 25.6 | 42.2 | 39.4 | 39.4 | 35.2 | -13.8 | -9.5 |

Held-out gain over the token unigram of the same corpus (bits per character; the unigram codes all three corpora alike, so this separates what a method takes from structure from how badly it overfits a corpus without any):

| Method | real | shuffled | slot-shuffled | real − shuffled | real − slot-shuffled |
|---|---|---|---|---|---|
| xz -9e | 9.0 | -13.1 | -5.7 | 22.1 | 14.7 |
| gzip -9 | 3.6 | -19.4 | -11.8 | 23.0 | 15.4 |
| token 2-gram | 12.9 | 0.4 | 7.0 | 12.4 | 5.9 |
| token 3-gram | 15.0 | -6.1 | 1.2 | 21.1 | 13.9 |
| GRU, hidden 8 | 14.7 | 6.4 | 11.5 | 8.3 | 3.1 |
| GRU, hidden 16 | 17.4 | 7.1 | 12.2 | 10.3 | 5.2 |
| GRU, hidden 32 | 19.8 | 7.2 | 12.8 | 12.6 | 7.0 |
| GRU, hidden 64 | 21.0 | 7.2 | 12.7 | 13.8 | 8.3 |
| GRU, hidden 128 | 21.3 | 7.0 | 12.7 | 14.4 | 8.6 |
| TRELLIS, structures given (operators as relations) | 13.8 | 7.7 | 12.6 | 6.1 | 1.1 |
| TRELLIS, structures given (operators as tokens) | 19.1 | 6.8 | 9.8 | 12.3 | 9.3 |
| TRELLIS, sequences alone | 18.4 | 4.1 | 8.7 | 14.3 | 9.7 |

TRELLIS two-part codes (bits per training character: model + data = total):

| Model | real | shuffled | slot_shuffled |
|---|---|---|---|
| TRELLIS, structures given (operators as relations) | 4.3 + 30.2 = 34.5 (21 symbols, 280 chunk types) | 1.7 + 36.7 = 38.4 (4 symbols, 92 chunk types) | 3.6 + 31.5 = 35.1 (16 symbols, 124 chunk types) |
| TRELLIS, structures given (operators as tokens) | 12.3 + 19.4 = 31.7 (27 symbols, 89 chunk types) | 1.8 + 37.8 = 39.5 (12 symbols, 33 chunk types) | 9.4 + 27.5 = 36.9 (15 symbols, 44 chunk types) |
| TRELLIS, sequences alone | 12.8 + 19.2 = 32.0 (23 symbols, 50 chunk types) | 4.5 + 37.7 = 42.2 (14 symbols, 46 chunk types) | 14.2 + 25.3 = 39.4 (17 symbols, 38 chunk types) |

## 2. Chunk validity

Each chunk type (symbol, left symbol, relation, right symbol) is represented by its modal training yield.

| Model | cond | chunk types | modal yield a database component | operator + leading parts | complete, not listed | crosses components | mean purity | types with purity ≥ 0.5 (of which component) | specific reused unit: purity ≥ 0.5, a component used as a proper part | modal yield only ever a whole character | composite instances that are components |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRELLIS, structures given (operators as relations) | real | 280 | 87.9% | 0.0% | 12.1% | 0.0% | 0.73 | 216 (86.6%) | 60.0% | 14.6% | 91.3% |
| TRELLIS, structures given (operators as relations) | shuffled | 92 | 4.3% | 0.0% | 95.7% | 0.0% | 0.17 | 11 (0.0%) | 0.0% | 30.4% | 4.2% |
| TRELLIS, structures given (operators as relations) | slot_shuffled | 124 | 20.2% | 0.0% | 79.8% | 0.0% | 0.45 | 54 (22.2%) | 9.7% | 21.8% | 15.9% |
| TRELLIS, structures given (operators as tokens) | real | 89 | 30.3% | 64.0% | 5.6% | 0.0% | 0.48 | 39 (33.3%) | 14.6% | 2.2% | 46.1% |
| TRELLIS, structures given (operators as tokens) | shuffled | 33 | 12.1% | 72.7% | 15.2% | 0.0% | 0.23 | 6 (0.0%) | 0.0% | 3.0% | 2.0% |
| TRELLIS, structures given (operators as tokens) | slot_shuffled | 44 | 9.1% | 72.7% | 18.2% | 0.0% | 0.27 | 9 (0.0%) | 0.0% | 4.5% | 7.8% |
| TRELLIS, sequences alone | real | 50 | 44.0% | 24.0% | 0.0% | 32.0% | 0.43 | 21 (33.3%) | 14.0% | 6.0% | 39.0% |
| TRELLIS, sequences alone | shuffled | 46 | 8.7% | 2.2% | 26.1% | 63.0% | 0.17 | 6 (0.0%) | 0.0% | 6.5% | 0.8% |
| TRELLIS, sequences alone | slot_shuffled | 38 | 13.2% | 13.2% | 26.3% | 47.4% | 0.28 | 8 (0.0%) | 0.0% | 7.9% | 4.1% |

Recall of the K most frequent multi-part database components (proper parts of the training characters):

| Model | K | modal yield of a chunk type (any purity) | … with purity ≥ 0.5 | modal yield of a category (any purity) | … with purity ≥ 0.5 | occurrences analysed as one constituent |
|---|---|---|---|---|---|---|
| TRELLIS, structures given (operators as relations) | 50 (50) | 46.0% | 30.0% | 14.0% | 0.0% | 100.0% |
| TRELLIS, structures given (operators as relations) | 100 (100) | 41.0% | 27.0% | 9.0% | 1.0% | 100.0% |
| TRELLIS, structures given (operators as relations) | 200 (200) | 30.5% | 21.0% | 5.0% | 0.5% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 50 (50) | 20.0% | 8.0% | 8.0% | 2.0% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 100 (100) | 13.0% | 5.0% | 4.0% | 1.0% | 100.0% |
| TRELLIS, structures given (operators as tokens) | 200 (200) | 8.5% | 4.0% | 2.0% | 0.5% | 100.0% |
| TRELLIS, sequences alone | 50 (50) | 20.0% | 12.0% | 8.0% | 4.0% | 57.8% |
| TRELLIS, sequences alone | 100 (100) | 12.0% | 6.0% | 5.0% | 2.0% | 68.4% |
| TRELLIS, sequences alone | 200 (200) | 7.0% | 3.0% | 2.5% | 1.0% | 71.9% |

## 3. Completion and cued recall (held-out real characters)

Masked: one atomic component of each held-out character hidden, every learned component ranked by the probability of the filled character (246 candidates; chance top-1 0.4%). Cue: the operator and the first part. Cue bits = −log2 P(character | cue). Open: 100 draws given the cue, ranked by frequency. Closed: the real corpus characters sharing the cue, ranked by probability. Percentile rank: 0 = first, chance 0.5.

| Method | masked top-1 / top-5 | masked percentile rank | cue bits (mean / median) | open: top-1 / top-5 | open: draws real / real unlearned / well formed | closed (all): top-1 / top-5 (chance) | closed: percentile rank |
|---|---|---|---|---|---|---|---|
| token 2-gram | 36.8% / 66.4% | 0.053 | 20.2 / 19.0 | 0.8% / 2.2% | 4.4% / 2.4% / 18.2% | 11.6% / 23.0% (11.2% / 23.3%) | 0.482 |
| token 3-gram | 52.6% / 73.0% | 0.074 | 19.5 / 19.0 | 2.0% / 4.4% | 8.6% / 4.2% / 17.1% | 11.6% / 21.8% (11.2% / 23.3%) | 0.490 |
| GRU, hidden 128 | 54.4% / 72.8% | 0.041 | 13.7 / 12.5 | 2.0% / 4.8% | 22.8% / 10.9% / 83.7% | 10.4% / 21.8% (nan% / nan%) | 0.510 |
| TRELLIS, structures given (operators as relations) | 18.8% / 43.6% | 0.080 | 18.8 / 17.2 | 0.8% / 2.6% | 7.3% / 6.3% / 100.0% | 11.8% / 24.2% (11.2% / 23.3%) | 0.474 |
| TRELLIS, structures given (operators as tokens) | 50.0% / 68.8% | 0.064 | 15.3 / 13.5 | 1.4% / 4.4% | 14.7% / 8.5% / 97.0% | 11.2% / 22.8% (11.2% / 23.3%) | 0.497 |
| TRELLIS, sequences alone | 50.8% / 68.8% | 0.063 | 15.9 / 14.3 | 1.2% / 4.2% | 11.2% / 6.4% / 60.4% | 11.2% / 23.2% (11.2% / 23.3%) | 0.486 |

## Across seeds (13, 17, 23): key numbers

Full table: `summary_seeds.md`. Bits per character; shares in %. Cue bits on the exact subset: cues whose operator has no three-part counterpart (⿴…⿻) and characters whose relational form names them alone, where the relational number is like-for-like (42–48 queries per seed).

| Quantity | seed 13 | seed 17 | seed 23 | mean | min–max |
|---|---|---|---|---|---|
| total · TRELLIS supervised · real | 31.7 | 31.0 | 31.5 | 31.4 | 31.0–31.7 |
| total · TRELLIS unsupervised · real | 32.0 | 32.2 | 32.3 | 32.2 | 32.0–32.3 |
| total · TRELLIS relational · real | 34.5 | 34.5 | 34.4 | 34.5 | 34.4–34.5 |
| total · token 2-gram (adaptive code) · real | 34.9 | 34.4 | 34.9 | 34.7 | 34.4–34.9 |
| total · token 3-gram (adaptive code) · real | 35.2 | 34.5 | 35.1 | 34.9 | 34.5–35.2 |
| total · xz -9e · real | 42.9 | 42.4 | 43.0 | 42.8 | 42.4–43.0 |
| total · GRU 8 (8-bit) · real | 46.5 | 45.8 | 46.5 | 46.3 | 45.8–46.5 |
| total · GRU 8 (entropy-coded, best bit depth) · real | 39.0 | 40.0 | 38.4 | 39.1 | 38.4–40.0 |
| held-out · GRU 128 · real | 22.7 | 22.9 | 23.6 | 23.1 | 22.7–23.6 |
| held-out · TRELLIS supervised · real | 24.9 | 24.7 | 25.7 | 25.1 | 24.7–25.7 |
| held-out · TRELLIS unsupervised · real | 25.6 | 26.3 | 26.8 | 26.3 | 25.6–26.8 |
| held-out · token 3-gram · real | 29.0 | 29.5 | 30.5 | 29.7 | 29.0–30.5 |
| held-out · TRELLIS relational · real | 30.3 | 30.6 | 31.7 | 30.9 | 30.3–31.7 |
| gain lost when slot_shuffled · TRELLIS unsupervised | 9.7 | 8.0 | 9.1 | 8.9 | 8.0–9.7 |
| gain lost when slot_shuffled · TRELLIS supervised | 9.3 | 8.0 | 8.0 | 8.5 | 8.0–9.3 |
| gain lost when slot_shuffled · TRELLIS relational | 1.1 | 0.7 | 0.5 | 0.8 | 0.5–1.1 |
| gain lost when slot_shuffled · GRU 128 | 8.6 | 8.3 | 8.5 | 8.5 | 8.3–8.6 |
| gain lost when slot_shuffled · token 3-gram | 13.9 | 13.4 | 13.4 | 13.6 | 13.4–13.9 |
| validity: modal yield a component · unsupervised · real | 44.0 | 38.1 | 36.8 | 39.6 | 36.8–44.0 |
| validity: modal yield a component · unsupervised · shuffled | 8.7 | 6.8 | 6.8 | 7.4 | 6.8–8.7 |
| validity: modal yield a component · unsupervised · slot_shuffled | 13.2 | 17.9 | 13.7 | 14.9 | 13.2–17.9 |
| validity: specific reused unit · unsupervised · real | 14.0 | 9.5 | 7.0 | 10.2 | 7.0–14.0 |
| validity: specific reused unit · supervised · real | 14.6 | 6.9 | 13.3 | 11.6 | 6.9–14.6 |
| validity: specific reused unit · relational · real | 60.0 | 48.8 | 49.3 | 52.7 | 48.8–60.0 |
| recall@50, chunk type purity >= 0.5 · unsupervised | 12.0 | 4.0 | 4.0 | 6.7 | 4.0–12.0 |
| recall@50, chunk type purity >= 0.5 · supervised | 8.0 | 6.0 | 8.0 | 7.3 | 6.0–8.0 |
| recall@50, chunk type purity >= 0.5 · relational | 30.0 | 44.0 | 22.0 | 32.0 | 22.0–44.0 |
| masked top-1 · TRELLIS relational | 18.8 | 24.6 | 24.6 | 22.7 | 18.8–24.6 |
| masked top-1 · TRELLIS supervised | 50.0 | 52.8 | 49.2 | 50.7 | 49.2–52.8 |
| masked top-1 · TRELLIS unsupervised | 50.8 | 49.2 | 53.0 | 51.0 | 49.2–53.0 |
| masked top-1 · token 3-gram | 52.6 | 53.6 | 53.0 | 53.1 | 52.6–53.6 |
| masked top-1 · GRU (best validation size) | 54.4 | 57.2 | 57.6 | 56.4 | 54.4–57.6 |
| open draws real · TRELLIS supervised | 14.7 | 15.8 | 14.3 | 14.9 | 14.3–15.8 |
| open draws real · TRELLIS unsupervised | 11.2 | 11.4 | 10.5 | 11.0 | 10.5–11.4 |
| open draws real · TRELLIS relational | 7.3 | 8.3 | 6.6 | 7.4 | 6.6–8.3 |
| open draws real · GRU (best validation size) | 22.8 | 24.8 | 23.9 | 23.8 | 22.8–24.8 |
| cue bits, exact subset · relational | 16.0 | 16.4 | 16.1 | 16.2 | 16.0–16.4 |
| cue bits, exact subset · supervised | 13.2 | 12.2 | 15.0 | 13.5 | 12.2–15.0 |
| cue bits, exact subset · unsupervised | 14.1 | 13.3 | 14.9 | 14.1 | 13.3–14.9 |
| cue bits, exact subset · 3-gram | 17.0 | 15.8 | 19.3 | 17.4 | 15.8–19.3 |
| cue bits, exact subset · GRU (best validation size) | 12.6 | 11.8 | 13.8 | 12.8 | 11.8–13.8 |

## Caveats

- Tables 1–3 above are seed 13; the across-seeds table covers seeds 13, 17 and 23 (each seed is its own split and shuffle). Three seeds give a spread, not confidence intervals.
- With structures given, every composite is a gold IDS subtree, so a chunk type's modal yield is a database component almost by construction (relational: 91% of composite instances already are). The shuffled rows give the floor; the measurement is informative mainly for sequences alone, and through purity and recall for all modes.
- Recall "modal yield of a chunk type (any purity)" counts a component even when it is a small share of its type's instances (e.g. ⿱十一 at purity 0.01 with operators as tokens); the purity ≥ 0.5 column is the strict version.
- Closed-set cue ranking is at chance for every method, by construction: the candidates are all real characters with the same cue and the held-out truth is a random one of them. Read cue bits instead.
- Relational cue bits on all queries condition on a coarser cue (⿲ A is written ⿰ A …, so binary and ternary cues merge, and a canonical tree can stand for two sequences); only the exact-subset cue bits are like-for-like.
- GRU totals: entropy-coded weights (adaptive KT code over 2–8-bit levels, best depth per model) shorten the GRU's total a lot (hidden 8: 46 → 39 bits) but not below TRELLIS; hidden sizes below 8 and MDL-oriented training were not tried.
- The n-gram total is the prequential (adaptive add-α) code; the GRU total is training NLL at 8-bit weights plus 8 bits per parameter; TRELLIS's total is its own model + data code. gzip/xz include their headers; per-character compression is dominated by them.

