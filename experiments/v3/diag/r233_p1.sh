#!/bin/bash
# Round 233 (PLAN_INTERWEAVE): expansive (tree) generation vs the causal word-window generator on the controlled grammar.
# usage: r233_p1.sh <seed> <n> [feats=func:5]   -> one job; arms listed in $R/<tag>/arms.json (dump file armI = arms[I])
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps"
S=${1:-13}; N=${2:-1000}; F=${3:-func:5}; TAG=$(echo $F | tr -d ':')
R=experiments/v3/results/story_gen/r233/p1
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg_w5.json)"; CFG="${CFG%\}},\"wemit_tree\":true,\"wtree_feats\":\"$F\"}"
A='{"causal": 1, "units": "words", "new": true, "temp": 1.0, "edge": "exact", "wreach": 5, "wemit": "leaf", "pair": "gate"}'
for k in ${LV-0 1 2 3 4 6}; do A="$A, {\"choose\": \"leaf\", \"level\": $k, \"fill_name\": false}"; done
A="$A, {\"choose\": \"best\", \"fill_name\": false}, {\"choose\": \"match\", \"fill_name\": false}, {\"choose\": \"backoff\", \"fill_name\": false}"
D=$R/${TAG}_s${S}_n${N}; mkdir -p $D; echo "[$A]" > $D/arms.json
STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=${MEM-32G} -t 24:00:00 -J r233s$S --export=ALL -o logs/v3-r233-$(basename $D).log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r230_story.py "$CFG" $S $N story "[$A]" story
