#!/bin/bash
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
cd /coc/flash10/ksingara3/Chunking-Cobweb; export PYTHONPATH=src
P=experiments/v3/results/story_gen/r231/p1/allsuf_s13_n1000_c6
for a in 0 1 4; do python experiments/v3/diag/r231_examples.py $P/story_s13_n1000_arm$a.jsonl 3; done > experiments/v3/results/story_gen/r231/examples.jsonl
