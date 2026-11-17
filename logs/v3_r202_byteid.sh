#!/bin/bash
# Round 202: t3_story_r202.py with the plain arm must reproduce t3_story.py's r200-n250-w5 seed 13 temp 1.3 dump byte for
# byte; the parse arm at the same n is the first timing probe. Usage: bash logs/v3_r202_byteid.sh
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen
D=$R/r202-byteid-n250-w5; mkdir -p $D
G='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 1.3}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 1.3, "gen": "parse"}]'
j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=24G -t 8:00:00 -J v3-r202b --export=ALL -o logs/v3-r202-byteid.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story_r202.py "{$S,$D0,\"story_facet\":true,\"window\":5}" 13 250 story "$G" story)
echo "byteid $j"
