#!/bin/bash
# Round 214 track 1: knob B (read_reach W' = 1,2,3,5,7) and the temperature control (default read, tau 1,2,4,8) on the token learner,
# seed 13, n=1000, gate off, edge exact. Dumps in experiments/v3/results/story_gen/r214p01/b{1,2}. Login node: only sbatch.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
CFG=$(cat $R/cfg.json)
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": $2, \"edge\": \"exact\", \"wreach\": $1}"; }
J1="[$(arm 1 1.0), $(arm 2 1.0), $(arm 3 1.0), $(arm 5 1.0)]"
J2="[$(arm 7 1.0), $(arm 7 2.0), $(arm 7 4.0), $(arm 7 8.0)]"
k=0
for A in "$J1" "$J2"; do
  k=$((k+1)); D=$R/b$k; mkdir -p $D
  j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J v3-r214b$k --export=ALL -o logs/v3-r214b$k.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFG" 13 1000 story "$A" story)
  echo "b$k $j"
done
