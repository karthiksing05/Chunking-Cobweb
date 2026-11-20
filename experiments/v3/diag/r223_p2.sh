#!/bin/bash
# Round 223 cheap check part 2 (prediction 3): function words abstracted too (cfbits) and W' = 0, n=1000 seed 13, func:100 tree, K=25, W' 3.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps"
R=experiments/v3/results/story_gen/r223p1
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": $1, \"wemitk\": 25, \"cbits\": \"$2\", \"cfbits\": \"$3\", \"pair\": \"gate\"}"; }
A="$(arm 0 leaf leaf), $(arm 3 leaf 7), $(arm 3 leaf 6), $(arm 3 leaf 4), $(arm 3 7 6), $(arm 3 6 4), $(arm 3 0 6), $(arm 3 0 4), $(arm 5 0 6)"
D=$R/p2_func100_s13_n1000; mkdir -p $D
CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg_w5.json)"; CFG="${CFG%\}},\"wemit_tree\":true,\"wtree_feats\":\"func:100\"}"
STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=64G -t 36:00:00 -J v3-r223-p2 --export=ALL -o logs/v3-r223p2.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r223_story.py "$CFG" 13 1000 story "[${A}]" story
