#!/bin/bash
# Round 219 step 1d: judge oracle for the new feature sets (fn25lr, fn25, fn40d2lr) at b 5-7, content-word swaps inside the tree classes (n=4000).
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=experiments/v3/results/story_gen/r214p01; W=experiments/v3/results/story_gen/r219w
O=experiments/v3/results/story_gen/r219t2
SP=~/flash/miniconda3/envs/cobint/bin/python; JP=~/flash/miniconda3/envs/chunks/bin/python
export R219_K=0 R219_ONLY=t_ R219_CPOS=NOUN,VERB,ADJ R219_TB=5,6,7
echo "host $(hostname) start $(date)"
mkdir -p $O
for cfg in "n4000_s13 oracle4000_pos s13" "n4000_s17 oracle4000_s17 s17"; do
  set -- $cfg
  R219_TREES=fn25lr=$W/wt_fn25lr_$3.json,fn25=$W/wt_fn25_$3.json,fn40d2lr=$W/wt_fn40d2lr_$3.json $SP experiments/v3/diag/r219_oracle.py make $O/o_$1 $R/$2/variants.json $R/wordtree_all_$3_n4000.json
  $JP experiments/v3/diag/r214_judge2.py score $O/o_$1/sentences.txt $O/o_$1/scores.tsv
  $SP experiments/v3/diag/r219_oracle.py report $O/o_$1 $O/o_$1/scores.tsv $O/o_$1/report.md
done
echo "end $(date)"
