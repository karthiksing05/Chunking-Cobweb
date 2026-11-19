#!/bin/bash
# Round 214 class emission follow-up (seed 13, token input, gate on, edge exact): b 6.5 at tau 0.7 and 2, W' = 2 at b 6.5 and 7.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
CFGI="$(cat $R/cfg.json)"; CFGI="${CFGI%\}},\"wemit_tree\":true}"
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": $2, \"edge\": \"exact\", \"wreach\": $3, \"wemit\": $1, \"pair\": \"gate\"}"; }
D=$R/e13_i_follow; mkdir -p $D
STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J v3-r214e --export=ALL -o logs/v3-r214e-follow.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFGI" 13 1000 story "[$(arm 6.5 0.7 7), $(arm 6.5 2.0 7), $(arm 6.5 1.0 2), $(arm 7 1.0 2)]" story
