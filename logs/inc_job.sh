#!/bin/bash
# Generic runner for the incremental-learner experiments: all arguments are
# passed to python. Do not edit while jobs are running.
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /coc/flash10/ksingara3/Chunking-Cobweb
echo "host $(hostname) start $(date)"
python "$@"
echo "end $(date) status $?"
