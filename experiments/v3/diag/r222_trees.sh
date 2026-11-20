#!/bin/bash
# Round 222 (a)+(b judge-free): word-tree feature sets with endings and nearest function words; purity per level.
cd /coc/flash10/ksingara3/Chunking-Cobweb
R=experiments/v3/results/story_gen/r214p01; O=experiments/v3/results/story_gen/r222w; mkdir -p $O
SP=~/flash/miniconda3/envs/cobint/bin/python; TP=~/flash/miniconda3/envs/trellis/bin/python
export R219_CPOS=NOUN,VERB,ADJ
echo "host $(hostname) start $(date)"
for S in 13 17; do
  REF=$R/wordtree_all_s${S}_n4000.json
  SPECS="func100=$R/wordtree_func100_s${S}_n4000.json,fn25lr=experiments/v3/results/story_gen/r219w/wt_fn25lr_s$S.json"
  for F in fn:25+suf fn:25+nf fn:25+suf+nf fn:25+lr+suf+nf fn:40+suf+nf fn:15+suf+nf fn:25+suf+nf+d2; do
    N=$(echo $F | tr -d ":+"); $TP experiments/v3/diag/r219_wordtree.py $REF $O/wt_${N}_s$S.json $F $S
    SPECS="$SPECS,$N=$O/wt_${N}_s$S.json"
  done
  $SP experiments/v3/diag/r219_oracle.py purity $O/purity_s${S}_n4000.md $(echo $SPECS | tr ',' ' ')
done
echo "end $(date)"
