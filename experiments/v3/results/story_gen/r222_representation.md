# Round 222: word-tree representation (endings, nearest function words) and the size-cut read-out. Verdict: NO-GO at (b); generation (c) not run

Diagnostics only (spaCy for oracle/purity; never in the model). Same oracle design that hit .50-.52 in r219: every NOUN/VERB/ADJ content word swapped for another type of the same TRELLIS word-tree class, nothing frequency-protected (K=0),
the R-sem sentences of the gated-leaf arm as input (n=4000: 697 sources seed 13, 679 seed 17), 3 draws, r214_judge2 unchanged. Reference: signature-exact oracle R-syn .50 (s13) / .52 (s17). Go for (c): tree-class R-syn >= .45.

## What changed (and where it sits in the original architecture)
- `WordTree219` (`src/trellis2/trellis3_r219.py`) adds two optional facets to the existing word-category tree: `+suf` the type's own ending (last 1-3 characters of the raw token) and `+nf` the nearest feature (frequent, function-like) word to the left and right of each occurrence, skipping content words. Both are raw-token derived (no tagger). Earlier: `+lr` (joint left/right pair), `+d2` (words two away).
  Same Fisher Cobweb tree, same level read, same class emission; only the facets that describe a type differ. This is a change of which features feed the word tree.
- Size cut (`r219_wordtree.py`, names `m<k>`): a word's class is the lowest ancestor of its leaf that holds >= k member types. It is a different read-out rule on the same tree (the tree's own `count`), not a new module; at emission it would replace "lowest ancestor with information level <= b" in `class_of` / `names`. Not wired into the generator because the gate failed.

## (b1) Judge-free purity (content-word classes, token-weighted share of class-mates with the token's morph signature; seeds 13 | 17; b-levels in `r222w/purity_*`)
Depth cuts, n=4000, b=6 (classes ~ 38-51): func100 .38 | .39; fn25+lr .44 | .51; fn25+suf .36 | .44; fn25+nf .41 | .43; fn25+suf+nf .29 | .43; fn15+suf+nf .39 | .46; fn25+lr+suf+nf .31 | .43; fn25+suf+nf+d2 .37 | .39.
b=7 (classes ~ 65-84): .39-.52 | .45-.58 (best fn25+lr .52 | .58, fn25+suf .39 | .55). Endings and nearest function words do not raise purity over +lr; combining them lowers it on seed 13.
Size cuts (content-type classes / purity), seed 13 | seed 17:
| tree | m=2 | m=3 | m=5 | m=8 | m=12 | m=20 |
|---|---|---|---|---|---|---|
| all | 76/.43 \| 74/.49 | 55/.39 \| 59/.46 | 38/.36 \| 37/.37 | 29/.35 \| 25/.36 | 20/.29 \| 17/.27 | 11/.27 \| 11/.21 |
| func:100 | 79/.44 \| 75/.45 | 57/.38 \| 58/.43 | 38/.37 \| 43/.37 | 26/.33 \| 29/.36 | 19/.33 \| 21/.32 | 13/.28 \| 10/.27 |
| fn25+lr | 84/.52 \| 78/.58 | 57/.47 \| 60/.54 | 40/.44 \| 44/.47 | 27/.35 \| 29/.43 | 16/.29 \| 20/.36 | 11/.25 \| 9/.26 |
| fn25+suf | 67/.39 \| 70/.55 | 48/.37 \| 54/.46 | 35/.35 \| 34/.44 | 21/.30 \| 25/.40 | 13/.29 \| 15/.35 | 8/.23 \| 9/.28 |
| fn25+nf | 80/.44 \| 74/.45 | 57/.41 \| 58/.44 | 40/.40 \| 40/.42 | 30/.36 \| 25/.40 | 17/.34 \| 16/.31 | 9/.28 \| 9/.27 |
| fn15+suf+nf | 70/.42 \| 70/.54 | 49/.40 \| 59/.49 | 38/.38 \| 42/.45 | 27/.33 \| 26/.38 | 17/.31 \| 15/.33 | 11/.30 \| 9/.27 |
Purity falls with m (fewer, bigger classes) and is no higher at a given class count than the depth cut (e.g. fn25+lr b=6: 45/.44 | 51/.51 vs m=3: 57/.47 | 60/.54 is a similar curve). The tree at ~50-80 classes has purity .4-.58 whichever feature set or cut.

## (b2) Tree-class oracle R-syn (judge), n=4000
Depth cuts b=5/6/7 (r222t, seed 13 | seed 17), new feature sets: fn25+suf .28/.28/.23 | .31/.34/.34; fn25+nf .28/.34/.34 | .31/.30/.37; fn25+suf+nf .20/.18/.21 | .25/.30/.33; fn25+lr+suf+nf .20/.22/.27 | .29/.34/.40; fn15+suf+nf .31/.33/.34 | .30/.35/.37;
fn25+suf+nf+d2 .30/.29/.29 | .27/.29/.38. Earlier (r219): all .29/.30/.32 | .30/.34/.34; func100 .24/.32/.32 | .27/.29/.32; fn25+lr .27/.32/.33 | .27/.34/.29.
Size cuts (r222m), R-syn, seed 13 | seed 17:
| tree | m=2 | m=3 | m=5 | m=8 | m=12 | m=20 |
|---|---|---|---|---|---|---|
| all | .32 \| .33 | .29 \| .36 | .30 \| .33 | .29 \| .29 | .25 \| .22 | .20 \| .15 |
| func:100 | .32 \| .33 | .33 \| .30 | .31 \| .27 | .25 \| .26 | .25 \| .27 | .19 \| .19 |
| fn25+lr | .33 \| .30 | .32 \| .35 | .29 \| .30 | .28 \| .29 | .24 \| .25 | .22 \| .23 |
| fn25+suf | .23 \| .35 | .26 \| .38 | .26 \| .33 | .28 \| .31 | .25 \| .25 | .20 \| .19 |
| fn25+nf | .35 \| .36 | .31 \| .30 | .31 \| .34 | .29 \| .31 | .24 \| .26 | .22 \| .16 |
| fn15+suf+nf | .33 \| .36 | .33 \| .35 | .31 \| .34 | .31 \| .32 | .34 \| .27 | .27 \| .22 |
Best cells: .349 (s13, fn25+nf m=2), .379 (s17, fn25+suf m=3, one cell); .395 (s17, fn25+lr+suf+nf b=7). Every other cell .15-.37. The m cut and the b cut give the same R-syn range (.30-.38 at the best level), peaking at the finest classes (m=2-3 / b=6-7) and falling for coarser cuts.

## Go / no-go
- (b) NO-GO: best tree-class oracle R-syn .395 < .45 (and .35 under the matched-seed average of the best cell), against the signature oracle .50-.52. Adding endings and nearest function words, and cutting by size instead of depth, leave purity at .4-.58 and R-syn at .30-.38.
- (c) not run: no setting has a class source that, even as an oracle swap, reaches R-syn .45; the b / W' grid and learning curves would act on classes already measured not to produce the regime. The existing generated ceiling stays .265 (r214 Addendum 7). Showcase examples per regime were not produced for the same reason; the earlier examples for R-sem and R-non are in r214_phase01.md.
- What the numbers say: a tree built on neighbour/ending features has ~50-80 cells at which its classes are about half morphosyntactically pure; the signature oracle has ~1.0 purity by construction. The residual impurity is mixed tags/frames inside cells, which none of the facets (neighbours, joint pairs, two-away, endings, nearest function words) or cuts (depth, size) removed. A feature that sees the verb's complement (the word after the object) or a supervised-free tagging pass would be a different mechanism, not a representation tweak.
Files: `r222w/` (purity per level), `r222t/` (depth-cut oracle), `r222m/s{13,17}/` (size-cut dumps, purity.md, o/report.md). Scripts: `experiments/v3/diag/r222_trees.sh`, `r222_oracle_tree.sh`, `r222_msize.sh`, `r219_wordtree.py`, `r219_oracle.py`.
Jobs: 4807136 (feature dumps, purity), 4807145 (depth oracle with judge), 4807772 / 4807773 (size cut seeds 13 / 17).
