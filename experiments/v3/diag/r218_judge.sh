#!/bin/bash
# Round 218: prep sentences (env trellis), judge them (env chunks, GPU), report. Run as one GPU job.
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname) start $(date)"
~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r218_gen_report.py prep
~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/r214_judge2.py score experiments/v3/results/story_gen/r218/sentences.txt experiments/v3/results/story_gen/r218/scores.tsv
~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r218_gen_report.py report | tee experiments/v3/results/story_gen/r218/report.md
echo "end $(date)"
