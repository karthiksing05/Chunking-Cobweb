#!/bin/bash
# usage: r223_judge.sh <out dir> <label=dump dir> ...   (one GPU job: collect, judge, report)
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=$1; shift; mkdir -p $O
PY=~/flash/miniconda3/envs/chunks/bin/python; TP=~/flash/miniconda3/envs/trellis/bin/python
echo "host $(hostname) start $(date)"
$TP experiments/v3/diag/r214_phase01.py collect $O "$@"
$PY experiments/v3/diag/r214_judge2.py score $O/sentences.txt $O/scores.tsv
$TP experiments/v3/diag/r214_phase01.py report $O $O/scores.tsv $O/report.md "$@"
echo "end $(date)"
