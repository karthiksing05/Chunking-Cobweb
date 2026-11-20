#!/bin/bash
# Round 222 size-cut: class = lowest ancestor holding >= m member types, m in 2 3 5 8 12 20. Dumps + purity (CPU), then the tree-class oracle with the judge (GPU) for seed $1.
S=$1
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=experiments/v3/results/story_gen/r214p01; O=experiments/v3/results/story_gen/r222m; mkdir -p $O/s$S
SP=~/flash/miniconda3/envs/cobint/bin/python; TP=~/flash/miniconda3/envs/trellis/bin/python; JP=~/flash/miniconda3/envs/chunks/bin/python
REF=$R/wordtree_all_s${S}_n4000.json
echo "host $(hostname) start $(date)"
SPECS=""
for F in all func:100 fn:25+lr fn:25+suf fn:25+nf fn:15+suf+nf; do
  N=$(echo $F | tr -d ":+"); $TP experiments/v3/diag/r219_wordtree.py $REF $O/s$S/wt_$N.json $F $S
  SPECS="$SPECS,$N=$O/s$S/wt_$N.json"
done
export R219_CPOS=NOUN,VERB,ADJ R219_TB=m2,m3,m5,m8,m12,m20
$SP experiments/v3/diag/r219_oracle.py purity $O/s$S/purity.md $(echo ${SPECS#,} | tr ',' ' ')
export R219_K=0 R219_ONLY=t_
SRC=oracle4000_pos; [ $S = 17 ] && SRC=oracle4000_s17
R219_TREES=${SPECS#,} $SP experiments/v3/diag/r219_oracle.py make $O/s$S/o $R/$SRC/variants.json $REF
$JP experiments/v3/diag/r214_judge2.py score $O/s$S/o/sentences.txt $O/s$S/o/scores.tsv
$SP experiments/v3/diag/r219_oracle.py report $O/s$S/o $O/s$S/o/scores.tsv $O/s$S/o/report.md
echo "end $(date)"
