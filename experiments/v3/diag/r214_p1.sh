#!/bin/bash
# Round 214 Phase 1: seed 13, n=1000, 15 arms in 3 learner jobs (128G). Recipe: Trellis3R204 edge exact, window 7, bands naming, cu tree,
# posterior store, words, new, gate OFF except the three gate arms; tau 1 except the temperature path (b = inf, tau 2 / 4 / 8).
# Dumps: experiments/v3/results/story_gen/r214p01/j{1,2,3}.  Usage: r214_p1.sh   (login node: only sbatch)
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
CFG=$(cat $R/cfg.json)
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS
arm() { # bits temp [gate]
  local g=""; [ "$3" = gate ] && g=', "pair": "gate"'
  echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": $2, \"edge\": \"exact\", \"bits\": $1$g}"
}
arr() { local out="" sep=""; for a in "$@"; do out="$out$sep$a"; sep=", "; done; echo "[$out]"; }
J1=$(arr "$(arm 0 1.0)" "$(arm 1 1.0)" "$(arm 2 1.0)" "$(arm 4 1.0)" "$(arm 6 1.0)")
J2=$(arr "$(arm 8 1.0)" "$(arm 12 1.0)" "$(arm '"inf"' 1.0)" "$(arm '"basic"' 1.0)" "$(arm '"inf"' 2.0)")
J3=$(arr "$(arm 2 1.0 gate)" "$(arm 4 1.0 gate)" "$(arm '"inf"' 1.0 gate)" "$(arm '"inf"' 4.0)" "$(arm '"inf"' 8.0)")
k=0
for A in "$J1" "$J2" "$J3"; do
  k=$((k+1)); D=$R/j$k; mkdir -p $D
  j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J v3-r214p1-j$k --export=ALL -o logs/v3-r214p1-j$k.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$CFG" 13 1000 story "$A" story)
  echo "j$k $j"
done
