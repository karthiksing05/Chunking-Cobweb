#!/bin/bash
# usage: r230_judge_job.sh <out.json> <dump dir>...   (judge every arm dump, print the dial table)
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
cd /coc/flash10/ksingara3/Chunking-Cobweb; export PYTHONPATH=src
O=$1; shift
python experiments/v3/diag/r230_grammar.py judge $O "$@" > /dev/null
python experiments/v3/diag/r230_grammar.py table $O
