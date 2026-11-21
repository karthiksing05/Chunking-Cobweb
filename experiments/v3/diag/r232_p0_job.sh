#!/bin/bash
# r232 step 0: func:5 word-tree dumps (seeds 13, 17, n=1000) and POS purity per level for func:5 and the func:100 dumps of r214.
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH OMP_NUM_THREADS=1
cd /coc/flash10/ksingara3/Chunking-Cobweb; export PYTHONPATH=src
R=experiments/v3/results/story_gen/r232; mkdir -p $R/wt
echo "host $(hostname) start $(date)"
CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg.json)"
for S in 13 17; do
  R214_WORDTREE=$R/wt/wordtree_func5_s${S}_n1000.json timeout 1500 python experiments/v3/diag/r214_story.py "${CFG%\}},\"wtree_feats\":\"func:5\"}" $S 1000 story "[]" story | tail -1
  for T in func5 func100; do
    F=$R/wt/wordtree_${T}_s${S}_n1000.json; [ $T = func100 ] && F=experiments/v3/results/story_gen/r214p01/wordtree_func100_s${S}_n1000.json
    ~/flash/miniconda3/envs/cobint/bin/python experiments/v3/diag/r214_wordmap.py $F $R/purity_${T}_s${S}.md $R/purity_${T}_s${S}.json
  done
done
echo "end $(date)"
