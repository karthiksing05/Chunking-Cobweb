#!/bin/bash
# Round 202 aggregation for one n: frontier rows of the r202 dirs; fair / selection / held-out envelopes against the
# round 200 frontier (r191 + r200 rows; at n 4000 the r191 file plus the r200 n4000 dirs' rows when judged); r192
# classes, length / repeat / premature-end rates for r202 and the plain w5 dirs. usage: v3_r202_final.sh <n>
export PATH=$HOME/flash/miniconda3/envs/chunks/bin:$PATH
export PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:$PYTHONPATH
cd /coc/flash10/ksingara3/Chunking-Cobweb
R=experiments/v3/results/story_gen
n=$1
D="$R/r202-n$n-w5"; [ $n = 1000 ] && D="$D $R/r202abl-n1000-w5"
python experiments/v3/story_frontier.py $R/r202-frontier-n$n.jsonl $D > /dev/null || exit 1
case $n in
  1000) PL="$R/r184-n1000-w5 $R/r188-n1000-w5 $R/r190-n1000-w5 $R/r200-n1000-w5" ;;
  4000) PL="$R/r200-n4000-w5" ;;
  *) PL="$R/r191-n$n-w5 $R/r200-n$n-w5" ;;
esac
if [ $n = 4000 ]; then
  if [ -f $R/r200-frontier-n4000.jsonl ]; then BASE=$R/r200-frontier-n4000.jsonl; else
    python experiments/v3/story_frontier.py $R/r202-base-frontier-n4000.jsonl $R/r200-n4000-w5 $R/r200-n4000-w6 $R/r200-n4000-w7 $R/r200-n4000-refs > /dev/null
    BASE="$R/r191-frontier-n4000.jsonl $R/r202-base-frontier-n4000.jsonl"; fi
else BASE=$R/r200-frontier-n$n.jsonl; fi
python experiments/v3/diag/r202_fair.py $n $R/r202-fair-n$n $BASE $R/r202-frontier-n$n.jsonl > /dev/null || exit 1
cat $R/r202-fair-n$n.select.txt
python experiments/v3/diag/r202_classes.py $R/r202-classes-n$n.tsv $D $PL
echo end status $?
