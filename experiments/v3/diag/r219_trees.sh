#!/bin/bash
# Round 219 step 1c: feature sets for the word tree (dumps only, CPU) and their morph swap purity vs existing trees. n=4000 seed 13 and 17.
cd /coc/flash10/ksingara3/Chunking-Cobweb
R=experiments/v3/results/story_gen/r214p01; O=experiments/v3/results/story_gen/r219w; mkdir -p $O
SP=~/flash/miniconda3/envs/cobint/bin/python; TP=~/flash/miniconda3/envs/trellis/bin/python
export R219_CPOS=NOUN,VERB,ADJ
echo "host $(hostname) start $(date)"
for S in 13 17; do
  REF=$R/wordtree_all_s${S}_n4000.json
  SPECS="all=$REF,func100=$R/wordtree_func100_s${S}_n4000.json"
  for F in fn:15 fn:25 fn:25+d2 fn:25+lr fn:25+d2+lr fn:40+d2+lr fn:100+d2+lr; do
    N=$(echo $F | tr -d ":+"); $TP experiments/v3/diag/r219_wordtree.py $REF $O/wt_${N}_s$S.json $F $S
    SPECS="$SPECS,$N=$O/wt_${N}_s$S.json"
  done
  $SP experiments/v3/diag/r219_oracle.py purity $O/purity_s${S}_n4000.md $(echo $SPECS | tr ',' ' ')
done
echo "end $(date)"
