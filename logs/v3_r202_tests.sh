#!/bin/bash
# Round 202: tests (r202 + full tests/trellis2 suite) on a compute node.
cd /coc/flash10/ksingara3/Chunking-Cobweb
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
echo "host $(hostname) start $(date)"
PYTHONPATH=src:tests/trellis2 timeout 5400 python -m pytest -q -p no:cacheprovider tests/trellis2/test_r202_parse_gen.py 2>&1 | tail -30
if [ "$1" = all ]; then PYTHONPATH=src:tests/trellis2 timeout 7200 python -m pytest -q -p no:cacheprovider tests/trellis2 2>&1 | tail -15; fi
echo "end $(date)"
