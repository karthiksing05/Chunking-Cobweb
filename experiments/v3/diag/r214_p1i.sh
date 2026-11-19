#!/bin/bash
# Round 214i: n=4000 (window 5, W' 5), seed S: class-gate dial with func:100 emission (K 50, b = emission = gate in 7 6 4 0) and the gated token leaf
# at tau 0.7 / 1 (the word gate, no emission); plus the word-tree dumps for the oracle. Dumps r214p01/i<S>_n4000.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
S=${1:-13}; LEAF=${2:-"0.7 1.0"}
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": 5, \"wemit\": $1, \"wemitk\": 50, \"wgate\": $1, \"pair\": \"gate\"}"; }
leaf() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": $1, \"edge\": \"exact\", \"wreach\": 5, \"wemit\": \"leaf\", \"pair\": \"gate\"}"; }
A=""; for t in $LEAF; do A="$A, $(leaf $t)"; done; for b in 7 6 4 0; do A="$A, $(arm $b)"; done
D=$R/i${S}_n4000; mkdir -p $D
CFG="$(cat $R/cfg_w5.json)"; CFG="${CFG%\}},\"wemit_tree\":true,\"wtree_feats\":\"func:100\"}"
STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=200G -t 48:00:00 -J v3-r214i --export=ALL -o logs/v3-r214i-$S.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFG" $S 4000 story "[${A:2}]" story
for f in all func:100; do t=$(echo $f | tr -d ':'); C="$(cat $R/cfg_w5.json)"; C="${C%\}},\"wtree_feats\":\"$f\"}"
R214_WORDTREE=$R/wordtree_${t}_s${S}_n4000.json sbatch --parsable $P -c 1 --mem=64G -t 4:00:00 -J v3-r214iw --export=ALL -o logs/v3-r214i-wt-$t-$S.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$C" $S 4000 story "[]" story; done
