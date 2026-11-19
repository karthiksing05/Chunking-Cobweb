#!/bin/bash
# usage: run.sh <cmd...>  (compute node; env chunks)
cd /coc/flash10/ksingara3/Chunking-Cobweb
export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
~/flash/miniconda3/envs/${PYENV:-chunks}/bin/python experiments/v3/diag/r214_judge2.py "$@"
