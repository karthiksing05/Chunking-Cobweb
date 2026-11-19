#!/bin/bash
# Round 214g: emission classes from a word tree whose features are the top-F frequent neighbours only (wtree_feats func:F), seed S, n=1000,
# edge exact, token input, gate on, tau 1, W' 7: emission K in 50 100 x b in 3 4 5 6 7, for F in 50 100; plus the all-feature tree at b 3.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
S=${1:-13}; FS=${2:-"func:50 func:100"}; KS=${3:-"50 100"}; BS=${4:-"3 4 5 6 7"}
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": ${5:-1.0}, \"edge\": \"exact\", \"wreach\": 7, \"wemit\": $1, \"wemitk\": $2, \"pair\": \"gate\"$3$4}"; }
sub() { mkdir -p $1; CFG="$(cat $R/cfg.json)"; CFG="${CFG%\}},\"wemit_tree\":true,\"wtree_feats\":\"$2\"}"; j=$(STORY_DUMP=$1 sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J v3-r214g --export=ALL -o logs/v3-r214g-$(basename $1).log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFG" $S 1000 story "$3" story); echo "$(basename $1) $j"; }
for F in $FS; do for K in $KS; do
  A=""; for b in $BS; do A="$A, $(arm $b $K)"; done
  sub $R/g${S}_${F/:/}_K$K $F "[${A:2}]"
done; done
if [ "$S" = 13 ]; then sub $R/g13_all_b3 all "[$(arm 3 50), $(arm 3 100)]"; fi
