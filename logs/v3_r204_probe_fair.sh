#!/bin/bash
# Round 204 probe aggregation (rerun of the fair step of v3_r204_probe.sh, which lacked PYTHONPATH for cobweb_cu).
export PATH=$HOME/flash/miniconda3/envs/chunks/bin:$PATH
export PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:$PYTHONPATH
cd /coc/flash10/ksingara3/Chunking-Cobweb
R=experiments/v3/results/story_gen
n=1000
python experiments/v3/story_frontier.py $R/r204-probe-frontier-n$n.jsonl $R/r204-probe-n$n-w6 $R/r204-probe-n$n-w7 > /dev/null || exit 1
R204_SEEDS=13 python experiments/v3/diag/r204_fair.py $n $R/r204-probe-fair-n$n $R/r200-frontier-n$n.jsonl $R/r204-probe-frontier-n$n.jsonl
echo end status $?
