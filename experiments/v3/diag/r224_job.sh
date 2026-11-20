#!/bin/bash
# Round 224: prep sentences (env trellis), judge2 score (env chunks, GPU), report (env trellis). One GPU job.
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname) start $(date)"
D=experiments/v3/results/story_gen/r224
~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r224_rejudge.py prep || exit 1
~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/r214_judge2.py score $D/sentences.txt $D/scores.tsv || exit 1
~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r224_rejudge.py report
echo "end $(date)"
