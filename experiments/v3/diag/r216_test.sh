#!/bin/bash
# Round 216: new tests and the full tests/trellis2 suite on a compute node
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
cd /coc/flash10/ksingara3/Chunking-Cobweb
echo "host $(hostname) start $(date)"
python -m pytest tests/trellis2/test_r216_parse.py -q 2>&1 | tail -30
python -m pytest tests/trellis2 -q 2>&1 | tail -15
echo "end $(date)"
