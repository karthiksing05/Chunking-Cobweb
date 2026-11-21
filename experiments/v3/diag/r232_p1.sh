#!/bin/bash
# Round 232 generation: r214_p1h.sh (C6a: class-level gate + emission at b, W' 7, K=50 exact types, tau 1, n=1000) with the word-tree features as a parameter.
# usage: r232_p1.sh <seed> <feats: func:5|func:100>   Prints the job id. Dumps to r232/g_<feats>_s<seed>.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps"
R=experiments/v3/results/story_gen/r232
S=$1; F=$2; T=$(echo $F | tr -d ':')
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": 7, \"wemit\": $1, \"wemitk\": 50, \"wgate\": $1, \"pair\": \"gate\"}"; }
leaf() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": 7, \"wemit\": \"leaf\", \"pair\": \"gate\"}"; }
D=$R/g_${T}_s${S}; mkdir -p $D
CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg.json)"; CFG="${CFG%\}},\"wemit_tree\":true,\"wtree_feats\":\"$F\"}"
STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J r232g --export=ALL -o logs/v3-r232-g_${T}_s${S}.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFG" $S 1000 story "[$(leaf), $(arm 7), $(arm 6), $(arm 5), $(arm 4), $(arm 0)]" story
