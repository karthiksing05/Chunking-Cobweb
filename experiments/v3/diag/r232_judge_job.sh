#!/bin/bash
# one GPU job: collect, judge (Qwen3-8B + phi-4, r214_judge2), report.  usage: r232_judge_job.sh   (all four generation dirs)
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=experiments/v3/results/story_gen/r232
S="func5=$R/g_func5_s13 func5=$R/g_func5_s17 func100=$R/g_func100_s13 func100=$R/g_func100_s17"
echo "host $(hostname) start $(date)"
~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r214_phase01.py collect $R $S
~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/r214_judge2.py score $R/sentences.txt $R/scores.tsv
~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r214_phase01.py report $R $R/scores.tsv $R/report.md $S
echo "end $(date)"
