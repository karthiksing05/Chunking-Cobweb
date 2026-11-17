#!/bin/bash
# Round 190 (stories): complete the grammaticality/novelty curves at n 1000 (seeds 13 17 19 23, forced-new).
#  - GPT and LSTM at temp 0.3 0.4 0.5 0.6 (STORY_BASE; model selection untouched)
#  - TRELLIS causal words-gated, learner window 4 and 5 at temp 1.3. Window 6 is skipped: the causal filler in
#    t3_story.py pads the history with 5 BOS tokens (t_ = [T3BOS] * 5 + out), so window 6 reads t_[-6] and fails.
# Then the judge, story_frontier.py over the r190 dirs and every r184/r188 n1000 dir, and the plot / envelope.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
G='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 1.3}]'
B='[{"lm": "gpt", "new": true, "temp": 0.3}, {"lm": "gpt", "new": true, "temp": 0.4}, {"lm": "gpt", "new": true, "temp": 0.5}, {"lm": "gpt", "new": true, "temp": 0.6},
    {"lm": "lstm", "new": true, "temp": 0.3}, {"lm": "lstm", "new": true, "temp": 0.4}, {"lm": "lstm", "new": true, "temp": 0.5}, {"lm": "lstm", "new": true, "temp": 0.6}]'
R=experiments/v3/results/story_gen
n=1000; SEEDS="13 17 19 23"
Q=""; DIRS=""
for w in 4 5; do
  D=$R/r190-n$n-w$w; mkdir -p $D; DIRS="$DIRS $D"
  for s in $SEEDS; do
    j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=48G -t 12:00:00 -J v3-r190 --export=ALL -o logs/v3-r190-n$n-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
    Q="$Q:$j"
  done
done
D=$R/r190-n$n-refs; mkdir -p $D; DIRS="$DIRS $D"
for s in $SEEDS; do
  j=$(STORY_BASE="$B" STORY_DUMP=$D sbatch --parsable $P -c 4 --mem=24G -t 24:00:00 -J v3-r190r --export=ALL -o logs/v3-r190-n$n-refs-$s.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true}" $s $n story "$B" story)
  Q="$Q:$j"
done
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 8:00:00 -J v3-r190j --dependency=afterany$Q -o logs/v3-r190-judge-n$n.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
OLD=""; for d in r184-n$n-w2 r184-n$n-w3 r184-n$n-w4 r184-n$n-w5 r184-n$n-refs r188-n$n-w3 r188-n$n-w4 r188-n$n-w5 r188-n$n-refs; do OLD="$OLD $R/$d"; done
jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r190f --dependency=afterany:$jj -o logs/v3-r190-frontier-n$n.log --open-mode=truncate logs/v3_job.sh experiments/v3/story_frontier.py $R/r190-frontier-n$n.jsonl $DIRS $OLD)
jp=$(sbatch --parsable $P -c 1 --mem=8G -t 1:00:00 -J v3-r190p --dependency=afterany:$jf -o logs/v3-r190-plot.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/story_frontier_plot.py $R/r190-frontier-n$n.jsonl $R/r190-frontier-n$n $R/r190_frontier.png docs/site/story_frontier.png)
echo "n$n gen $Q judge $jj frontier $jf plot $jp"
