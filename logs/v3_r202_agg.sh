#!/bin/bash
# Round 202 probe aggregation (seed 13, n 1000): frontier rows of the given tag dirs, fair / envelope vs r200 rows, classes.
# usage: sbatch ... logs/v3_r202_agg.sh <out tag> <dir> [<dir> ...]
export PATH=$HOME/flash/miniconda3/envs/chunks/bin:$PATH
export PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:$PYTHONPATH
cd /coc/flash10/ksingara3/Chunking-Cobweb
R=experiments/v3/results/story_gen
T=$1; shift
PL="$R/r184-n1000-w5 $R/r188-n1000-w5 $R/r190-n1000-w5 $R/r200-n1000-w5"
python experiments/v3/story_frontier.py $R/$T-frontier-n1000.jsonl "$@" > /dev/null || exit 1
R202_SEEDS=13 python experiments/v3/diag/r202_fair.py 1000 $R/$T-fair-n1000 $R/r200-frontier-n1000.jsonl $R/$T-frontier-n1000.jsonl > /dev/null || exit 1
cat $R/$T-fair-n1000.select.txt
R202_SEEDS=13 python experiments/v3/diag/r202_classes.py $R/$T-classes-n1000.tsv "$@" $PL
echo end status $?
