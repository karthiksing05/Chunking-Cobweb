#!/bin/bash
# Round 194b (chunk-junction pre-test): the r192 "both" arm drawing committed chunks as well as words ("units": "both"),
# with each sentence's unit lengths recorded ("seg": true), temp 0.7, w3 w4, seeds 13 17, n 1000; judged; then
# experiments/v3/diag/r194_junction.py (gram|tri by where the new trigram falls) and story_frontier.py with the r192 dirs.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen; n=1000
G='[{"causal": 1, "units": "both", "pair": "gate", "new": true, "temp": 0.7, "both": true, "seg": true}]'
Q=""; DIRS=""
for w in 3 4; do
  D=$R/r194-n$n-junction-w$w; mkdir -p $D; DIRS="$DIRS $D"
  for s in 13 17; do
    j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=24G -t 12:00:00 -J v3-r194u --export=ALL -o logs/v3-r194-n$n-junction-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
    Q="$Q:$j"
  done
done
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 4:00:00 -J v3-r194uj --dependency=afterany$Q -o logs/v3-r194-junction-judge.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
echo "gen $Q judge $jj"
