#!/bin/bash
# Round 193 pre-test, variant 2 (targeted: only words inside a new trigram): Gibbs repair (Trellis3.gibbs_repair) on the judged r192 n1000 "both" dumps (temp 0.7 arm),
# seeds 13 17, windows 3 4 5; original vs repaired judged, then story_frontier.py over the pre-test dirs.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
R=experiments/v3/results/story_gen
Q=""; DIRS=""
for w in 3 4 5; do
  D=$R/r193-pre2-w$w; mkdir -p $D; DIRS="$DIRS $D"
  for s in 13 17; do
    j=$(sbatch --parsable $P -c 1 --mem=16G -t 3:00:00 -J v3-r193t -o $R/r192-diag/r193_pretest2_w${w}_s$s.txt --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r193_pretest.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s 1000 0.7 $R/r192-n1000-w$w/story_s${s}_n1000_arm1.jsonl $D targeted)
    Q="$Q:$j"
  done
done
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 2:00:00 -J v3-r193tj --dependency=afterok$Q -o logs/v3-r193-pre2-judge.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end")
jf=$(sbatch --parsable $P -c 1 --mem=24G -t 1:00:00 -J v3-r193tf --dependency=afterany:$jj -o logs/v3-r193-pre2-frontier.log --open-mode=truncate logs/v3_job.sh experiments/v3/story_frontier.py $R/r193-pre2-frontier-n1000.jsonl $DIRS)
echo "gen $Q judge $jj frontier $jf"
