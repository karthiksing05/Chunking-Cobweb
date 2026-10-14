#!/bin/bash
# Run an experiment from a snapshot of the sources (src/, experiments/v2/
# without results), so edits to the repository never reach a running job.
# Usage: inc_snapjob.sh TAG script.py args...   (script path relative to the repo)
# Results paths given as absolute or repo-relative are rewritten to the repo.
REPO=/coc/flash10/ksingara3/Chunking-Cobweb
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TAG=$1; shift
SNAP=$REPO/logs/inc_snap/$TAG
if [ ! -d "$SNAP" ]; then echo "missing snapshot $SNAP"; exit 1; fi
cd $REPO
echo "host $(hostname) start $(date) snapshot $SNAP"
python $SNAP/"$@"
echo "end $(date) status $?"
