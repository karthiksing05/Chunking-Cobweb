#!/bin/bash
# Round 191 (stories): the grammaticality/novelty frontier at other training sizes.
#  n 250, 500 (seeds 13 17 19 23): TRELLIS causal words-gated window 3/4/5 x temp {0.5,0.7,1.0} (3 arms per run);
#     GPT, LSTM x temp {0.3,0.4,0.5,0.7,1.0}; ngram 3/4/5 x temp {0.7,1.0}; retrieved swap.
#  n 1000 (seeds 13 17 19 23): GPT, LSTM at temp 0.15, 0.2 (low-novelty end of the neural curves).
#  n 4000 (seeds 13 17): GPT, LSTM x temp {0.3,0.4,0.5,0.7}; ngram 3/4/5 x temp {0.7,1.0}; TRELLIS w3, w4 x temp {0.5,0.7}.
# Then the judge per n, story_frontier.py over that n's dirs, and the plot / envelope.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen
gen() { echo "{\"causal\": 1, \"units\": \"words\", \"pair\": \"gate\", \"new\": true, \"temp\": $1}"; }
lms() { # $1 = temps
  local out="" sep=""
  for m in gpt lstm; do for t in $1; do out="$out$sep{\"lm\": \"$m\", \"new\": true, \"temp\": $t}"; sep=", "; done; done
  echo "[$out]"; }
ngr() { local out="" sep=""; for g in 3 4 5; do for t in 0.7 1.0; do out="$out$sep{\"ngram\": $g, \"new\": true, \"temp\": $t}"; sep=", "; done; done; echo "$out"; }
finish() { # n tag dirs... ; uses $Q and $EXTRA_DEP (extra frontier dependency)
  local n=$1; shift; local DIRS="$*"
  local JC=""; for d in $DIRS; do case $d in *r191-*) JC="$JC $J $d;";; esac; done
  jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 8:00:00 -J v3-r191j --dependency=afterany$Q -o logs/v3-r191-judge-n$n.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
  jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r191f --dependency=afterany:$jj$EXTRA_DEP -o logs/v3-r191-frontier-n$n.log --open-mode=truncate logs/v3_job.sh experiments/v3/story_frontier.py $R/r191-frontier-n$n.jsonl $DIRS)
  jp=$(sbatch --parsable $P -c 1 --mem=8G -t 1:00:00 -J v3-r191p --dependency=afterany:$jf -o logs/v3-r191-plot-n$n.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/story_frontier_plot.py $R/r191-frontier-n$n.jsonl $R/r191-frontier-n$n $R/r191_frontier_n$n.png)
  echo "n$n gen $Q judge $jj frontier $jf plot $jp"; }

SEEDS="13 17 19 23"
# ---- n 250 / 500
for n in 250 500; do
  Q=""; DIRS=""
  G="[$(gen 0.5), $(gen 0.7), $(gen 1.0)]"
  for w in 3 4 5; do
    D=$R/r191-n$n-w$w; mkdir -p $D; DIRS="$DIRS $D"
    for s in $SEEDS; do
      j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=48G -t 12:00:00 -J v3-r191 --export=ALL -o logs/v3-r191-n$n-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
      Q="$Q:$j"
    done
  done
  B="$(lms "0.3 0.4 0.5 0.7 1.0" | sed 's/]$//'), $(ngr), {\"swap\": \"retrieve\"}]"
  D=$R/r191-n$n-refs; mkdir -p $D; DIRS="$DIRS $D"
  for s in $SEEDS; do
    j=$(STORY_BASE="$B" STORY_DUMP=$D sbatch --parsable $P -c 4 --mem=24G -t 24:00:00 -J v3-r191r --export=ALL -o logs/v3-r191-n$n-refs-$s.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true}" $s $n story "$B" story)
    Q="$Q:$j"
  done
  EXTRA_DEP=""; finish $n $DIRS
done
# ---- n 1000: low-temperature neural curves
n=1000; Q=""
B="$(lms "0.15 0.2")"
D=$R/r191-n$n-refs; mkdir -p $D
for s in $SEEDS; do
  j=$(STORY_BASE="$B" STORY_DUMP=$D sbatch --parsable $P -c 4 --mem=24G -t 24:00:00 -J v3-r191r --export=ALL -o logs/v3-r191-n$n-refs-$s.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true}" $s $n story "$B" story)
  Q="$Q:$j"
done
OLD=""; for d in r184-n$n-w2 r184-n$n-w3 r184-n$n-w4 r184-n$n-w5 r184-n$n-refs r188-n$n-w3 r188-n$n-w4 r188-n$n-w5 r188-n$n-refs r190-n$n-w4 r190-n$n-w5 r190-n$n-refs; do OLD="$OLD $R/$d"; done
EXTRA_DEP=""; finish $n $D $OLD
# ---- n 4000 (seeds 13 17)
n=4000; SEEDS="13 17"; Q=""; DIRS=""
G="[$(gen 0.5), $(gen 0.7)]"
for w in 3 4; do
  D=$R/r191-n$n-w$w; mkdir -p $D; DIRS="$DIRS $D"
  for s in $SEEDS; do
    j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=48G -t 48:00:00 -J v3-r191 --export=ALL -o logs/v3-r191-n$n-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
    Q="$Q:$j"
  done
done
B="$(lms "0.3 0.4 0.5 0.7" | sed 's/]$//'), $(ngr)]"
D=$R/r191-n$n-refs; mkdir -p $D; DIRS="$DIRS $D"
for s in $SEEDS; do
  j=$(STORY_BASE="$B" STORY_DUMP=$D sbatch --parsable $P -c 4 --mem=24G -t 48:00:00 -J v3-r191r --export=ALL -o logs/v3-r191-n$n-refs-$s.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true}" $s $n story "$B" story)
  Q="$Q:$j"
done
OLD=""; for d in w2 w3 w4 w5 refs; do OLD="$OLD $R/r184-n$n-$d"; done
EXTRA_DEP=":4757487"; finish $n $DIRS $OLD
