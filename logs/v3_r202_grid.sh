#!/bin/bash
# Round 202 grid: parse-guided arm v3 ("gen": "parse3"), window 5, temps 1.0 1.3 1.6 2.0, n 250/500/1000 seeds 13 17 19 23,
# n 4000 seeds 13 17; ablation at n 1000 (4 seeds): "parse2" (open-chart Z + closure probability) and "close2" (closure
# probability alone), temps 1.0 1.3. Then the judge per n. Aggregation: logs/v3_r202_final.sh.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen
arm() { echo "{\"causal\": 1, \"units\": \"words\", \"pair\": \"gate\", \"new\": true, \"temp\": $2, \"gen\": \"$1\"}"; }
G="[$(arm parse3 1.0), $(arm parse3 1.3), $(arm parse3 1.6), $(arm parse3 2.0)]"
GA="[$(arm parse2 1.0), $(arm parse2 1.3), $(arm close2 1.0), $(arm close2 1.3)]"
for n in 250 500 1000 4000; do
  if [ $n = 4000 ]; then SEEDS="13 17"; TL=48:00:00; M=64G; else SEEDS="13 17 19 23"; TL=24:00:00; M=40G; fi
  Q=""; DIRS="$R/r202-n$n-w5"; mkdir -p $R/r202-n$n-w5
  for s in $SEEDS; do
    j=$(STORY_DUMP=$R/r202-n$n-w5 sbatch --parsable $P -c 1 --mem=$M -t $TL -J v3-r202 --export=ALL -o logs/v3-r202-n$n-w5-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story_r202.py "{$S,$D0,\"story_facet\":true,\"window\":5}" $s $n story "$G" story)
    Q="$Q:$j"
    if [ $n = 1000 ]; then
      mkdir -p $R/r202abl-n1000-w5
      j=$(STORY_DUMP=$R/r202abl-n1000-w5 sbatch --parsable $P -c 1 --mem=$M -t $TL -J v3-r202 --export=ALL -o logs/v3-r202abl-n1000-w5-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story_r202.py "{$S,$D0,\"story_facet\":true,\"window\":5}" $s $n story "$GA" story)
      Q="$Q:$j"; DIRS="$DIRS $R/r202abl-n1000-w5"
    fi
  done
  JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
  jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 8:00:00 -J v3-r202j --dependency=afterany$Q -o logs/v3-r202-judge-n$n.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
  echo "n$n gen $Q judge $jj"
done
