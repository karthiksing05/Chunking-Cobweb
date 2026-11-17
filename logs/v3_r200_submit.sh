#!/bin/bash
# Round 200: extend the fair productivity comparison. New points only (r190/r191 dirs untouched).
#  TRELLIS plain causal words-gated, window {5,6,7} x temp {1.3,1.6,2.0} (skip n1000 w5 t1.3 = r190), n 250/500/1000 seeds 13 17 19 23, n4000 seeds 13 17.
#  LSTM, GPT at temp {0.8,0.9,1.2}, same n and seeds.
#  t3_story.py: causal history padding is now max(5, window) BOS (was 5; windows 6+ failed in r190).
# Then judge the r200 dirs, story_frontier.py on them, merge with r191 frontier rows, story_frontier_fair.py, story_fair_select.py.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen
gen() { echo "{\"causal\": 1, \"units\": \"words\", \"pair\": \"gate\", \"new\": true, \"temp\": $1}"; }
garr() { local out="" sep=""; for t in "$@"; do out="$out$sep$(gen $t)"; sep=", "; done; echo "[$out]"; }
lms() { local out="" sep=""; for m in gpt lstm; do for t in 0.8 0.9 1.2; do out="$out$sep{\"lm\": \"$m\", \"new\": true, \"temp\": $t}"; sep=", "; done; done; echo "[$out]"; }
for n in 250 500 1000 4000; do
  if [ $n = 4000 ]; then SEEDS="13 17"; TL=48:00:00; M=64G; else SEEDS="13 17 19 23"; TL=24:00:00; M=48G; fi
  Q=""; DIRS=""
  for w in 5 6 7; do
    if [ $n = 1000 ] && [ $w = 5 ]; then G=$(garr 1.6 2.0); else G=$(garr 1.3 1.6 2.0); fi
    D=$R/r200-n$n-w$w; mkdir -p $D; DIRS="$DIRS $D"
    for s in $SEEDS; do
      j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=$M -t $TL -J v3-r200 --export=ALL -o logs/v3-r200-n$n-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
      Q="$Q:$j"
    done
  done
  B="$(lms)"
  D=$R/r200-n$n-refs; mkdir -p $D; DIRS="$DIRS $D"
  for s in $SEEDS; do
    j=$(STORY_BASE="$B" STORY_DUMP=$D sbatch --parsable $P -c 4 --mem=24G -t $TL -J v3-r200r --export=ALL -o logs/v3-r200-n$n-refs-$s.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true}" $s $n story "$B" story)
    Q="$Q:$j"
  done
  JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
  jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 8:00:00 -J v3-r200j --dependency=afterany$Q -o logs/v3-r200-judge-n$n.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
  jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r200f --dependency=afterany:$jj -o logs/v3-r200-fair-n$n.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && export PATH=\$HOME/flash/miniconda3/envs/chunks/bin:\$PATH && python experiments/v3/story_frontier.py $R/r200-new-frontier-n$n.jsonl $DIRS > /dev/null && cat $R/r191-frontier-n$n.jsonl $R/r200-new-frontier-n$n.jsonl > $R/r200-frontier-n$n.jsonl && python experiments/v3/story_frontier_fair.py $R/r200-frontier-n$n.jsonl $n $R/r200-fair-n$n && python experiments/v3/story_fair_select.py $R/r200-fair-n$n; echo end status \$?")
  echo "n$n gen $Q judge $jj fair $jf"
done
