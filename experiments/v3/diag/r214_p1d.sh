#!/bin/bash
# Round 214 output-side abstraction (class emission, wemit b): seed S (default 13), n=1000, edge exact, tau 1, W' 7.
#   (i) token input: one learner with wemit_tree (word tree built, window as tokens); arms wemit b in 0 4 5 6 6.5 7 (+ leaf), gate on / off
#   (ii) input named at the same b: learner wtree b (window_names all), arm wemit b, gate on / off
# Usage: r214_p1d.sh [seed] ; dumps r214p01/e<seed>_i_on, _i_off, _ii_<b>. Login node: only sbatch.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
R=experiments/v3/results/story_gen/r214p01
S=${1:-13}; BS=${2:-"0 4 5 6 6.5 7"}
CFG0=$(cat $R/cfg.json)
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
arm() { local g=""; [ "$3" = gate ] && g=', "pair": "gate"'; echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": ${4:-1.0}, \"edge\": \"exact\", \"wreach\": ${5:-7}, \"wemit\": $1$g}"; }
sub() { # dir cfg arms
  mkdir -p $1; j=$(STORY_DUMP=$1 sbatch --parsable $P -c 1 --mem=128G -t 24:00:00 -J v3-r214e --export=ALL -o logs/v3-r214e-$(basename $1).log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r214_story.py "$2" $S 1000 story "$3" story); echo "$(basename $1) $j"; }
CFGI="${CFG0%\}},\"wemit_tree\":true}"
ON=""; OFF=""
for b in $BS '"leaf"'; do ON="$ON, $(arm $b x gate)"; done
for b in $BS; do OFF="$OFF, $(arm $b x off)"; done
sub $R/e${S}_i_on "$CFGI" "[${ON:2}]"
sub $R/e${S}_i_off "$CFGI" "[${OFF:2}]"
for b in $BS; do
  sub $R/e${S}_ii_$b "${CFG0%\}},\"wtree\":$b}" "[$(arm $b x gate), $(arm $b x off)]"
done
