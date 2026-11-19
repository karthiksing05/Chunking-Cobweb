#!/bin/bash
# Round 218 generation: seed S (default 13), n=1000, the r212 gate arm (temp 2.0, edge exact) with a learner whose learn() uses the windowed parse.
# One job per cell; dumps experiments/v3/results/story_gen/r218/<cell>_s<S>. Usage (login node: only sbatch): r218_gen.sh <seed>
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps"
S=${1:-13}
R=experiments/v3/results/story_gen/r218
CFG='{"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0,"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content","story_facet":true,"window":7}'
ARM='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 2.0, "edge": "exact"}]'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY
declare -A KW=( [full]='{}' [W3L]='{"win_W":3,"win_learn":true}' [W4L]='{"win_W":4,"win_learn":true}' [D1L]='{"io_depth_D":1,"win_learn":true}'
                [D2L]='{"io_depth_D":2,"win_learn":true}' [D3L]='{"io_depth_D":3,"win_learn":true}' [W4D2L]='{"win_W":4,"io_depth_D":2,"win_learn":true}'
                [D1cL]='{"io_depth_D":1,"io_depth_mode":"chart","win_learn":true}' [D2cL]='{"io_depth_D":2,"io_depth_mode":"chart","win_learn":true}' )
CELLS=${2:-${!KW[@]}}                       # optional: space-separated cell names
for c in $CELLS; do
  D=$R/${c}_s$S; mkdir -p $D
  R218_KW="${KW[$c]}" STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=96G -t 24:00:00 -J v3-r218g-$c --export=ALL -o logs/v3-r218-gen-$c-s$S.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r218_story.py "$CFG" $S 1000 story "$ARM" story
done
