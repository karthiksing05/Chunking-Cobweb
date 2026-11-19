#!/bin/bash
# Round 214h part 2: class-level pair gate (wgate b) at the emission level. Seed S, n=1000, edge exact, token input, tau 1, W' 7, emission K 50.
# func:100 tree: emission b = gate b in 0 4 5 6 7; all-feature tree: b = 6. (b leaf with the word gate = the existing gated leaf arm.)
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
S=${1:-13}
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": 7, \"wemit\": $1, \"wemitk\": 50, \"wgate\": $1, \"pair\": \"gate\"}"; }
sub() { mkdir -p $1; CFG="$(cat $R/cfg.json)"; CFG="${CFG%\}},\"wemit_tree\":true,\"wtree_feats\":\"$2\"}"; j=$(STORY_DUMP=$1 sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J v3-r214h --export=ALL -o logs/v3-r214h-$(basename $1).log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFG" $S 1000 story "$3" story); echo "$(basename $1) $j"; }
sub $R/h${S}_func100 func:100 "[$(arm 7), $(arm 6), $(arm 5), $(arm 4), $(arm 0)]"
sub $R/h${S}_all all "[$(arm 6)]"
