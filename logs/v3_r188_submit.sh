#!/bin/bash
# Round 188 (stories): sampling temperature for TRELLIS's causal generator and the n-gram samplers (the neural LMs
# already had it), on the round-184 grammaticality/novelty frontier. Causal words-gated forced-new at learner window
# 3, 4, 5 x temp {0.5, 0.7} (the two temps are two arms of one run per window: the learner is shared, each arm reseeds
# its sampler), ngram 3/4/5 forced-new at temp 0.7. n 1000, seeds 13 17 19 23. Then the judge, then
# experiments/v3/story_frontier.py over the r188 dirs AND the r184 n1000 dirs (one frontier run, all rows).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
G='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.5}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.7}]'
B='[{"ngram": 3, "new": true, "temp": 0.7}, {"ngram": 4, "new": true, "temp": 0.7}, {"ngram": 5, "new": true, "temp": 0.7}]'
R=experiments/v3/results/story_gen
n=1000; SEEDS="13 17 19 23"
Q=""; DIRS=""
for w in 3 4 5; do
  D=$R/r188-n$n-w$w; mkdir -p $D; DIRS="$DIRS $D"
  for s in $SEEDS; do
    j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=48G -t 12:00:00 -J v3-r188 --export=ALL -o logs/v3-r188-n$n-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
    Q="$Q:$j"
  done
done
D=$R/r188-n$n-refs; mkdir -p $D; DIRS="$DIRS $D"
for s in $SEEDS; do
  j=$(STORY_BASE="$B" STORY_DUMP=$D sbatch --parsable $P -c 4 --mem=24G -t 12:00:00 -J v3-r188r --export=ALL -o logs/v3-r188-n$n-refs-$s.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true}" $s $n story "$B" story)
  Q="$Q:$j"
done
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 8:00:00 -J v3-r188j --dependency=afterany$Q -o logs/v3-r188-judge-n$n.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
R184=""; for d in w2 w3 w4 w5 refs; do R184="$R184 $R/r184-n$n-$d"; done
jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r188f --dependency=afterany:$jj -o logs/v3-r188-frontier-n$n.log --open-mode=truncate logs/v3_job.sh experiments/v3/story_frontier.py $R/r188-frontier-n$n.jsonl $DIRS $R184)
echo "n$n gen $Q judge $jj frontier $jf"
