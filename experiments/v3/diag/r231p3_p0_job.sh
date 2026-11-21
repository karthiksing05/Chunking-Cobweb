#!/bin/bash
# r231 P3: word-tree dumps at n=2000/4000 (allsuf; all at 4000) and purity.
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH OMP_NUM_THREADS=1
cd /coc/flash10/ksingara3/Chunking-Cobweb; export PYTHONPATH=src
R=experiments/v3/results/story_gen/r231
mkdir -p $R/wt3
CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg_w5.json)"
for S in 13 17; do for N in 2000 4000; do for F in allsuf all; do
  [ $F = all ] && [ $N = 2000 ] && continue
  R214_WORDTREE=$R/wt3/wordtree_${F}_s${S}_n${N}.json timeout 3000 python experiments/v3/diag/r231_story.py "${CFG%\}},\"wtree_feats\":\"$F\"}" $S $N story "[]" story | tail -1
done; done; done
python experiments/v3/diag/r230_grammar.py purity $R/purity_p3.json $R/wt3/wordtree_*.json
