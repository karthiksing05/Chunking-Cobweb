#!/bin/bash
# Round 223 cheap check: context abstraction (cbits) x W' x K, n=1000 seed 13 (window 5, edge exact, token input, word pair gate, tau 1, exact emission).
# usage: r223_p1.sh <tree feats: func:100|all> ; one job per b_ctx (leaf: W' only).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps"
R=experiments/v3/results/story_gen/r223p1
S=${S:-13}; N=${N:-1000}; F=${1:-func:100}; TAG=$(echo $F | tr -d ':'); WS=${WS:-"1 2 3 5"}; KS=${KS:-"25 50"}; BS=${BS:-"leaf 7 6 4 0"}
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": ${T:-1.0}, \"edge\": \"exact\", \"wreach\": $1, \"wemitk\": $2, \"cbits\": $3, \"pair\": \"gate\"}"; }
for b in $BS; do
  A=""; ks=$KS; [ $b = leaf ] && ks="25"
  for k in $ks; do for w in $WS; do A="$A, $(arm $w $k \"$b\")"; done; done
  D=$R/${TAG}_s${S}_n${N}_b$b; mkdir -p $D
  CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg_w5.json)"; CFG="${CFG%\}},\"wemit_tree\":true,\"wtree_feats\":\"$F\"}"
  STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=64G -t 36:00:00 -J v3-r223-$b --export=ALL -o logs/v3-r223p1-${TAG}-s$S-n$N-b$b.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r223_story.py "$CFG" $S $N story "[${A:2}]" story
done
