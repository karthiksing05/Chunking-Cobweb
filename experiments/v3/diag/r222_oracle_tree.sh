#!/bin/bash
# Round 222 (b): tree-class oracle, all N/V/ADJ content words swapped inside the tree class (K=0), the r219 swap design that hit .50. n=4000.
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=experiments/v3/results/story_gen/r214p01; W=experiments/v3/results/story_gen/r222w; V=experiments/v3/results/story_gen/r219w
O=experiments/v3/results/story_gen/r222t
SP=~/flash/miniconda3/envs/cobint/bin/python; JP=~/flash/miniconda3/envs/chunks/bin/python
export R219_K=0 R219_ONLY=t_ R219_CPOS=NOUN,VERB,ADJ R219_TB=5,6,7
echo "host $(hostname) start $(date)"
mkdir -p $O
for cfg in "n4000_s13 oracle4000_pos s13" "n4000_s17 oracle4000_s17 s17"; do
  set -- $cfg
  T=""; for n in fn25suf fn25nf fn25sufnf fn25lrsufnf fn15sufnf fn25sufnfd2; do T="$T,$n=$W/wt_${n}_$3.json"; done
  R219_TREES=${T#,} $SP experiments/v3/diag/r219_oracle.py make $O/o_$1 $R/$2/variants.json $R/wordtree_all_$3_n4000.json
  $JP experiments/v3/diag/r214_judge2.py score $O/o_$1/sentences.txt $O/o_$1/scores.tsv
  $SP experiments/v3/diag/r219_oracle.py report $O/o_$1 $O/o_$1/scores.tsv $O/o_$1/report.md
done
echo "end $(date)"
