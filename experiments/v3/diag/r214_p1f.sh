#!/bin/bash
# Round 214f: open-class emission (wemitk K: the K most frequent types stay singleton classes), seed S, n=1000, edge exact, token input, gate on,
# tau 1, W' 7: K in 50 100 200 x b in 4 5 6 7 (two jobs); plus the gated token leaf model at tau 0.7 and at W' 2 (tau 1, 0.7).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
S=${1:-13}; KS=${2:-"50 100 200"}; BS=${3:-"4 5 6 7"}
CFGI="$(cat $R/cfg.json)"; CFGI="${CFGI%\}},\"wemit_tree\":true}"
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": $4, \"edge\": \"exact\", \"wreach\": $3, \"wemit\": $1, \"wemitk\": $2, \"pair\": \"gate\"}"; }
sub() { mkdir -p $1; j=$(STORY_DUMP=$1 sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J v3-r214f --export=ALL -o logs/v3-r214f-$(basename $1).log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFGI" $S 1000 story "$2" story); echo "$(basename $1) $j"; }
i=0
for K in $KS; do
  A=""; for b in $BS; do A="$A, $(arm $b $K 7 1.0)"; done
  sub $R/f${S}_K$K "[${A:2}]"
done
if [ "$S" = 13 ]; then sub $R/f13_leaf "[$(arm '"leaf"' 0 7 0.7), $(arm '"leaf"' 0 2 1.0), $(arm '"leaf"' 0 2 0.7)]"; fi
