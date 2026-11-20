#!/bin/bash
# r230 step 0 (one CPU job): judge controls, word-tree dumps (feats x seeds, n=1000), purity per level.
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH OMP_NUM_THREADS=1
cd /coc/flash10/ksingara3/Chunking-Cobweb
export PYTHONPATH=src
R=experiments/v3/results/story_gen/r230
echo "host $(hostname) start $(date)"
timeout 600 python experiments/v3/diag/r230_grammar.py controls $R/controls.json
CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg_w5.json)"
for S in 13 17; do for F in all func:5 func:8 func:15 func:30; do
  T=$(echo $F | tr -d ':')
  R214_WORDTREE=$R/wt/wordtree_${T}_s${S}_n1000.json timeout 900 python experiments/v3/diag/r230_story.py "${CFG%\}},\"wtree_feats\":\"$F\"}" $S 1000 story "[]" story | tail -2
done; done
timeout 600 python experiments/v3/diag/r230_grammar.py purity $R/purity.json $R/wt/wordtree_*_n1000.json
echo "end $(date)"
