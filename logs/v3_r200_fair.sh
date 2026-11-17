#!/bin/bash
# Round 200 aggregation for one n: frontier rows of the r200 dirs, merge with r191, fair productivity, selection + paired.
# usage: v3_r200_fair.sh <n> <dirs...>
export PATH=$HOME/flash/miniconda3/envs/chunks/bin:$PATH
export PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:$PYTHONPATH
cd /coc/flash10/ksingara3/Chunking-Cobweb
R=experiments/v3/results/story_gen
n=$1; shift
python experiments/v3/story_frontier.py $R/r200-new-frontier-n$n.jsonl "$@" > /dev/null || exit 1
cat $R/r191-frontier-n$n.jsonl $R/r200-new-frontier-n$n.jsonl > $R/r200-frontier-n$n.jsonl
python experiments/v3/story_frontier_fair.py $R/r200-frontier-n$n.jsonl $n $R/r200-fair-n$n || exit 1
python experiments/v3/story_fair_select.py $R/r200-fair-n$n
echo end status $?
