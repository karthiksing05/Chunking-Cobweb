#!/bin/bash
# Round 194: parse consistency with conjunctive context attributes (r185 recipe + "conj": true), seeds 13 17 19,
# checkpoints 250,1000; then the aggregate table (cross-seed F1 vs r185's .742 at 1k).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
tag=r194conj; Q=""
for s in 13 17 19; do
  j=$(sbatch --parsable $P -c 1 --mem=48G -t 12:00:00 -J v3-r194p -o logs/v3-r194-parse-$s.log --open-mode=truncate \
    logs/v3_job.sh experiments/v3/story_parse_eval.py "{$S,$D0,\"story_facet\":true,\"conj\":true}" $s 250,1000 $tag)
  Q="$Q:$j"
done
ja=$(sbatch --parsable $P -c 1 --mem=8G -t 0:30:00 -J v3-r194pa --dependency=afterany$Q -o logs/v3-r194-parse-agg.log --open-mode=truncate logs/v3_job.sh experiments/v3/story_parse_eval.py agg $tag)
echo "parse $Q agg $ja"
