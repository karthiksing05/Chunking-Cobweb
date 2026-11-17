#!/bin/bash
# Round 192 pre-test: both-sides check scored on the judged r188 n1000 dumps (no generation).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
R=experiments/v3/results/story_gen
for s in ${SEEDS:-13 17}; do for w in 3 4 5; do
  sbatch $P -c 1 --mem=12G -t 2:00:00 -J v3-r192t -o $R/r192-diag/pretest_w${w}_s$s.txt --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r192_pretest.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s 1000 $R/r188-n1000-w$w/story_s${s}_n1000_arm0.jsonl $R/r188-n1000-w$w/story_s${s}_n1000_arm1.jsonl
done; done
