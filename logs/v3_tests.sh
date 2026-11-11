#!/bin/bash
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
cd /coc/flash10/ksingara3/Chunking-Cobweb
python -m pytest tests -q 2>&1 | tail -5
