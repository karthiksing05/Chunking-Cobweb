#!/bin/bash
# r231 step 0: word-tree dumps for the variants (n=1000, seeds 13/17) and purity per level (incl. the r230 'all' dumps).
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH OMP_NUM_THREADS=1
cd /coc/flash10/ksingara3/Chunking-Cobweb
export PYTHONPATH=src
R=experiments/v3/results/story_gen/r231
mkdir -p $R/wt
echo "host $(hostname) start $(date)"
CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg_w5.json)"
for S in 13 17; do for F in ${VARIANTS:-abs:4 pos allsuf}; do
  T=$(echo $F | tr -d ':')
  R214_WORDTREE=$R/wt/wordtree_${T}_s${S}_n1000.json timeout 1500 python experiments/v3/diag/r231_story.py "${CFG%\}},\"wtree_feats\":\"$F\"}" $S 1000 story "[]" story | tail -2
done; done
timeout 600 python experiments/v3/diag/r230_grammar.py purity $R/purity${PTAG:-}.json experiments/v3/results/story_gen/r230/wt/wordtree_all_s*_n1000.json $R/wt/wordtree_*_n1000.json
echo "end $(date)"
