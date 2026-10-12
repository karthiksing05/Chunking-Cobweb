# Joint chunks as a night candidate (2026-10-11)

`src/trellis2` now searches every start with and without the joint chunk move, consolidates the best three of each
kind, takes the best of each kind through re-analysis without the read's context and sampled re-analysis, and lets the
full code at the end of the night choose (`UnsupervisedLearner(joint_chunks=True)`, the default; `joint_chunks=False`
gives the previous night). The first rule chose right after consolidation and lost the characters gain (s13: joint
68,309 vs plain 66,684 at that point; at the end of the night 62,414 vs 63,971). Results (`v2/`; baseline = `../*_baseline`):

| Domain | Run | Winner | Code (bits) | Held-out | Brackets / generation |
|---|---|---|---|---|---|
| characters, 2,000 | s13 | joint | 63,971 → 62,414 | 25.64 → 24.78 b/char | omission 27.5 → 11.9%; well formed 56.5 → 92.9%; attested 42.2 → 67.3% |
| characters, 2,000 | s17 | joint | 64,479 → 62,148 | 26.34 → 24.80 b/char | omission 43.9 → 6.3%; well formed 44.5 → 88.5%; attested 34.7 → 69.1% |
| six synthetic corpora | s13, s17 | plain in 11 of 12 | never longer; large s13 8,015 → 7,989 | – | large s13 commission 7.5 → 6.6% |
| TinyStories 2,500 | s13 | joint | 36,538 → 36,335 | 12.02 → 11.69 b/sent | real 63.1 → 71.2%; new 39.6 → 31.3% |
| deep, nightly to 1,000 | s13 | joint | 81,031 → 79,956 | 75.45 → 74.62 | omission 58.7 → 61.9%; grammatical 3.5 → 5.1% |
| deep, nightly to 1,000 | s17 | plain | 79,785 (equal) | – | – |

Chunk-type validity on characters does not improve (components 44.0 → 47.2% / 38.1 → 44.8%; strict top-50 recall
12 → 6% / 4 → 2%): the structure moves to gold, the categories stay generic.

Code: `mdl_search.CompositeState` (the joint chunk move as a third beam move, scored exactly from changed rows),
`chunk_and_merge(..., composites=False)`, `unsupervised.UnsupervisedLearner(joint_chunks=True)` with `_rest_of_night`;
tests `tests/trellis2/test_composites.py` (5). Full suite 124 passed.
