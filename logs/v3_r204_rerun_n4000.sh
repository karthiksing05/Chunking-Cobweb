#!/bin/bash
# Round 204: rerun of the n=4000 w6 seed 17 job (4780734), which was Killed (node out-of-memory, as r200) at 160G on droid;
# then re-judge the r204 n=4000 dirs and redo the fair step. Same arms and settings as v3_r204_grid.sh.
# usage: v3_r204_rerun_n4000.sh <exclude nodes extra>
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,droid"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY
R=experiments/v3/results/story_gen
n=4000; w=6; s=17
G='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 1.3, "edge": "exact"}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 1.6, "edge": "exact"}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 2.0, "edge": "exact"}]'
D=$R/r204-n$n-w$w
j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=200G -t 48:00:00 -J v3-r204g --export=ALL -o logs/v3-r204-n$n-w$w-$s.rerun.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
echo "rerun gen $j"
