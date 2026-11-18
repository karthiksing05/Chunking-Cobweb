#!/bin/bash
# Round 204: the full tests/trellis2 suite (compute node).
export PATH=$HOME/flash/miniconda3/envs/chunks/bin:$PATH
export PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:$PYTHONPATH
cd /coc/flash10/ksingara3/Chunking-Cobweb
echo host $(hostname)
PYTHONPATH=src:$PYTHONPATH timeout 3000 python -m pytest tests/trellis2 -x -q 2>&1 | tail -40
echo end status $?
