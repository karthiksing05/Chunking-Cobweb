#!/bin/bash
cd /coc/flash10/ksingara3/Chunking-Cobweb
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH PYTHONPATH=src:tests/trellis2
python -m pytest -x -q tests/trellis2/test_r223_context.py tests/trellis2/test_r214_level.py 2>&1 | tail -25
