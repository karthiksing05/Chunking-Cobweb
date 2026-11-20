#!/bin/bash
# Round 224 pass 2: n=4000 needs r200 frontier dumps too; judge only the sentences not yet scored (env trellis prep, env chunks judge2, report).
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname) start $(date)"
D=experiments/v3/results/story_gen/r224
~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r224_rejudge.py prep || exit 1
if [ -s $D/sentences_b.txt ] && [ "$(wc -l < $D/sentences_b.txt)" -gt 1 ]; then
  ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/r214_judge2.py score $D/sentences_b.txt $D/scores_b.tsv || exit 1
fi
~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r224_rejudge.py report
echo "end $(date)"
