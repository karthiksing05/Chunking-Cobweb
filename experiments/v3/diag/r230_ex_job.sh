#!/bin/bash
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
cd /coc/flash10/ksingara3/Chunking-Cobweb; export PYTHONPATH=src
P=experiments/v3/results/story_gen/r230/p1
for s in 13 17; do
python experiments/v3/diag/r230_grammar.py examples $P/func5_s${s}_n1000_c6/story_s${s}_n1000_arm4.jsonl 5 1
python experiments/v3/diag/r230_grammar.py examples $P/func5_s${s}_n1000_c6_b6/story_s${s}_n1000_arm0.jsonl 5 1
python experiments/v3/diag/r230_grammar.py examples $P/func5_s${s}_n1000_c6_b0/story_s${s}_n1000_arm0.jsonl 5 1
done
