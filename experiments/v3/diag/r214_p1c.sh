#!/bin/bash
# Round 214 track 2 probe: window names from the word tree (wtree b in bits or "leaf"), seed 13, n=1000, gate off, tau 1, edge exact;
# per learner the read reach W' = 7, 3, 1. Dumps: experiments/v3/results/story_gen/r214p01/t_<b>. Login node: only sbatch.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
CFG0=$(cat $R/cfg.json)
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": $1}"; }
A="[$(arm 7), $(arm 3), $(arm 1)]"
for b in 0 4 5 6 7 '"leaf"'; do
  t=$(echo $b | tr -d '"'); D=$R/t_$t; mkdir -p $D
  CFG="${CFG0%\}},\"wtree\":$b}"
  j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J v3-r214t$t --export=ALL -o logs/v3-r214t_$t.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFG" 13 1000 story "$A" story)
  echo "t_$t $j"
done
