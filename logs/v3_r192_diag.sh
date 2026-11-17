#!/bin/bash
# Round 192 diagnosis: trace the novel trigrams of the r188 n1000 causal words-gated dumps (seed 13, window 3/4/5).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
R=experiments/v3/results/story_gen
s=${SEED:-13}
for w in 3 4 5; do
  sbatch $P -c 1 --mem=12G -t 3:00:00 -J v3-r192d -o logs/v3-r192-diag-w$w-s$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r192_trace.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s 1000 $R/r192-diag/trace_w${w}_s$s.jsonl $R/r188-n1000-w$w/story_s${s}_n1000_arm0.jsonl $R/r188-n1000-w$w/story_s${s}_n1000_arm1.jsonl
done
