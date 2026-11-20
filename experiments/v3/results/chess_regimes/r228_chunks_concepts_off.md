# r228: chunks-off and concepts-off controls in chess

r225 found that removing chunks changes nothing measurable in language. This is the same control in chess, on the paper's chess recipe
(`confs/trellis-v2/paper/main.tex`, configuration C5: relational chunks of at most four pieces; compression recipe of `logs/v3_r103_submit.sh`, ab-control:
`chess_compress_gen.py`, config chess_flat + chess_join interact + exact_alone, env ACOPY=1 ROUTEW=2 COVER=2, learn 4,000 transfer positions, code 1,000 per test set).
Seeds 13, 17, 19. Code: `experiments/v3/diag/r228_submit.sh`, `r228_report.py`, `experiments/v3/r228_cs_leaf.py` (a copy of `chess_cs1973_exact.py` with a CS_LEAF switch).
Data: `experiments/v3/results/chess_compress/r228-{full,off}-<set>_s<seed>.json`, `experiments/v3/results/chess_cs1973_exact/r228-{csoff,csleaf}_s<seed>.json`; tables: `r228_tables.md`.
Jobs: smoke 4810900; compression and recall jobs 4810906-4810935; report 4811609.

## Arms

- **Full recipe**: max chunk size 4, all concept levels. Compression = the best (content, context) level row of the run; recall = the existing cs73x runs (same seeds, 4,000 positions).
- **Chunks off**: max chunk size 1 (no stored composite above a single piece; "stored chunks" = 0 in every run). Compression and recall both rerun.
- **Concepts off (leaf-only reads)**: compression: the (content 0, context 0) row of the full run, i.e. every code reads the leaf concept of the chunk, no ancestors.
  Recall (CS_LEAF=1): a unit counts as recognised only if the leaf its content (and context) sorts to holds at least `support` instances, and decoding uses that leaf, never an ancestor.

## (a) Bits per position after 4,000 positions (mean of 3 seeds; lower is better; model cost not charged)

| test set | full recipe | concepts off | chunks off | piece-square list | BPE over squares | BPE over shapes |
|---|---|---|---|---|---|---|
| control | 129.2 | 129.9 | 183.3 | 181.9 | 109.6 | 111.4 |
| other openings (family) | 137.5 | 138.2 | 191.9 | 190.4 | 123.7 | 120.4 |
| other wing | 153.4 | 153.8 | 200.7 | 198.3 | 146.5 | 142.7 |
| shifted | 151.5 | 152.1 | 210.7 | 208.9 | 200.4 | 149.7 |

Per seed (full / concepts off / chunks off), control: 130.8, 128.9, 127.8 / 131.7, 129.5, 128.5 / 183.2, 183.2, 183.5; shifted: 153.7, 150.1, 150.6 / 154.3, 151.0, 151.1 / 210.8, 211.1, 210.1;
other sets in `r228_tables.md`. The full recipe on seed 13 gives 130.8 on control against 130.6 in the paper table (rounding of the best-level row; same recipe, rerun).
- **Removing chunks costs 54 to 59 bits per position** (control +54.1, other openings +54.4, other wing +47.3, shifted +59.2), and lands within 1.4 to 1.8 bits of the piece-square list. Chunks off is exactly the piece-square list code (single tops only).
- **Removing concepts costs under one bit** (+0.7, +0.7, +0.4, +0.6): in the code, the chunk's own stored instances (copy routes) carry nearly all of the gain, and the ancestors add 0.4 to 0.7 bits.
- The seed spread of every difference is under 4 bits, far under the chunk effect.

## (b) Chase and Simon, first-trial correct pieces at 4,000 positions (mean of 3 seeds, 10 simulated subjects each)

| arm | K | middle | end | random middle | random end |
|---|---|---|---|---|---|
| full recipe (cs73x) | 7 | 12.91 | 7.38 | 5.93 | 6.12 |
| full recipe (cs73x) | 4 | 9.53 | 4.47 | 3.15 | 3.23 |
| chunks off | 7 | 6.98 | 6.89 | 6.37 | 6.27 |
| chunks off | 4 | 3.99 | 3.94 | 3.47 | 3.40 |
| concepts off (leaf-only reads) | 7 | 0.00 | 0.00 | 0.00 | 0.00 |
| concepts off (leaf-only reads) | 4 | 0.00 | 0.00 | 0.00 | 0.00 |

- **Chunks off removes the chunking effect on recall.** Middle-game recall at K=7 falls from 12.9 to 7.0, end-game from 7.4 to 6.9, and the game-versus-random gap disappears (7.0 and 6.9 against 6.4 and 6.3); at K=4, 9.5 falls to 4.0. With one piece per slot the recall is about the slot count (K=7: 6.3 to 7.0; K=4: 3.4 to 4.0), which is what the full model gets on random positions.
- **Concepts off (leaf-only reads) recalls nothing (0.00 in every cell, three seeds).** Recognition and decoding both read at a concept holding at least `support` stored instances. A leaf holds one instance, so no unit is recognised, every label is an omission, and the stored chunks (14k per seed) are never read. The control is degenerate by construction: it shows that recall needs ancestor concepts, not that the learner would recall with a smaller concept tree.

## Reading

1. **Chess differs from language here.** In language r225 found chunks removable at no measurable cost; in chess removing chunks takes the code back to the piece-square list (+47 to +59 bits per position, no better than 183 to 211 bits) and removes the recall advantage over random positions. Chunks carry the compression and the Chase and Simon pattern.
2. **Concepts matter for the two measures differently.** For compression, the leaf level is nearly as good as the best level (0.4 to 0.7 bits): the code can reuse stored instances exactly without generalising across them (the compression recipe's copy routes are instance-level). For recall, concepts decide everything: generalisation across instances is what lets a new position's chunks be recognised, so leaf-only recall is zero.
3. **Compared with BPE**: chunks off is far behind both BPE coders (183 against 110/111 on control); the full recipe sits between BPE and the piece-square list, as in the paper table.

## Caveats

Three seeds (compression 1,000 coded positions per set; recall 10 subjects per seed); "concepts off" is two different operationalisations (leaf level of the code's own levels table for bits; leaf-only recognition and decoding for recall), and they disagree by design; the full-recipe bits are the minimum over the run's (content, context) levels of the copy rows (seed 13 control 130.8 against 130.6 in the paper table); the recall control for concepts off recognises nothing, so it bounds rather than measures.
