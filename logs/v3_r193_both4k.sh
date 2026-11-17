#!/bin/bash
# Round 193: the r192 both-sides arm ("both": true) at n 4000, seeds 13 17 (as r191), windows 3/4/5 x temp {0.5,0.7,1.0};
# judge, story_frontier.py with the r184 / r191 n4000 dirs, plot_n / envelope. (The r193 Gibbs repair failed its
# pre-test and is not run; see experiments/v3/results/story_gen/r192_diagnosis.md section 8.)
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen
n=4000; T=r193
gen() { echo "{\"causal\": 1, \"units\": \"words\", \"pair\": \"gate\", \"new\": true, \"temp\": $1, \"both\": true}"; }
G="[$(gen 0.5), $(gen 0.7), $(gen 1.0)]"
Q=""; DIRS=""
for w in 3 4 5; do
  D=$R/$T-n$n-w$w; mkdir -p $D; DIRS="$DIRS $D"
  for s in 13 17; do
    j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=48G -t 48:00:00 -J v3-$T --export=ALL -o logs/v3-$T-n$n-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
    Q="$Q:$j"
  done
done
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 8:00:00 -J v3-${T}j --dependency=afterany$Q -o logs/v3-$T-judge-n$n.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
OLD=""; for d in r184-n$n-w2 r184-n$n-w3 r184-n$n-w4 r184-n$n-w5 r184-n$n-refs r191-n$n-w3 r191-n$n-w4 r191-n$n-refs; do OLD="$OLD $R/$d"; done
jf=$(sbatch --parsable $P -c 1 --mem=32G -t 2:00:00 -J v3-${T}f --dependency=afterany:$jj -o logs/v3-$T-frontier-n$n.log --open-mode=truncate logs/v3_job.sh experiments/v3/story_frontier.py $R/$T-frontier-n$n.jsonl $DIRS $OLD)
jp=$(sbatch --parsable $P -c 1 --mem=8G -t 1:00:00 -J v3-${T}p --dependency=afterany:$jf -o logs/v3-$T-plot-n$n.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/story_frontier_plot_n.py $R/$T-frontier-n$n.jsonl $R/$T-frontier-n$n $R/${T}_frontier_n$n.png)
echo "$T n$n gen $Q judge $jj frontier $jf plot $jp"
