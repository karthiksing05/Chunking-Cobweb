#!/bin/bash
# Round 219 step 1: agreement-aware oracle + judge controls. One GPU job (spaCy env cobint, judge env chunks). Diagnostics only.
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=experiments/v3/results/story_gen/r214p01
O=experiments/v3/results/story_gen/r219
SP=~/flash/miniconda3/envs/cobint/bin/python; JP=~/flash/miniconda3/envs/chunks/bin/python
echo "host $(hostname) start $(date)"
mkdir -p $O
$SP experiments/v3/diag/r219_oracle.py controls $O/controls
$JP experiments/v3/diag/r214_judge2.py score $O/controls/sentences.txt $O/controls/scores.tsv
$SP experiments/v3/diag/r219_oracle.py creport $O/controls $O/controls/scores.tsv $O/controls/report.md
for cfg in "n1000_s13 oracle_pos wordtree_all_s13_n1000" "n4000_s13 oracle4000_pos wordtree_all_s13_n4000" "n4000_s17 oracle4000_s17 wordtree_all_s17_n4000"; do
  set -- $cfg
  $SP experiments/v3/diag/r219_oracle.py make $O/o_$1 $R/$2/variants.json $R/$3.json
  $JP experiments/v3/diag/r214_judge2.py score $O/o_$1/sentences.txt $O/o_$1/scores.tsv
  $SP experiments/v3/diag/r219_oracle.py report $O/o_$1 $O/o_$1/scores.tsv $O/o_$1/report.md
done
echo "end $(date)"
