#!/bin/bash
cd /coc/flash10/ksingara3/Chunking-Cobweb && PYTHONPATH=src:tests/trellis2 timeout 1500 ~/flash/miniconda3/envs/trellis/bin/python -m pytest -q tests/trellis2/test_r194_conj.py tests/trellis2/test_r192_both_sides.py 2>&1 | tail -25
