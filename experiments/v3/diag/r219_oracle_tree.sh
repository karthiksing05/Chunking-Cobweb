#!/bin/bash
# Round 219 step 1b: swap content words (NOUN/VERB/ADJ, K=0) inside the word-tree classes at b (all, func100 trees), vs the morph signature oracle.
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=experiments/v3/results/story_gen/r214p01
O=experiments/v3/results/story_gen/r219t
SP=~/flash/miniconda3/envs/cobint/bin/python; JP=~/flash/miniconda3/envs/chunks/bin/python
export R219_K=0 R219_ONLY=morph_c,t_ R219_CPOS=NOUN,VERB,ADJ R219_TB=3,4,5,6,7
echo "host $(hostname) start $(date)"
mkdir -p $O
for cfg in "n1000_s13 oracle_pos s13_n1000" "n4000_s13 oracle4000_pos s13_n4000" "n4000_s17 oracle4000_s17 s17_n4000"; do
  set -- $cfg
  R219_TREES=all=$R/wordtree_all_$3.json,func100=$R/wordtree_func100_$3.json $SP experiments/v3/diag/r219_oracle.py make $O/o_$1 $R/$2/variants.json $R/wordtree_all_$3.json
  $JP experiments/v3/diag/r214_judge2.py score $O/o_$1/sentences.txt $O/o_$1/scores.tsv
  $SP experiments/v3/diag/r219_oracle.py report $O/o_$1 $O/o_$1/scores.tsv $O/o_$1/report.md
done
echo "end $(date)"
