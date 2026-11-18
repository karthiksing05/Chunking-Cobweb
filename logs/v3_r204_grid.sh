#!/bin/bash
# Round 204 step 3 (only if the probe passed): edge_exact full grid. Seeds 13 17 19 23 at n 250/500/1000, seeds 13 17 at n 4000;
# TRELLIS plain-causal words-gated + "edge": "exact", windows {5,6,7} x temp {1.3,1.6,2.0}. n=1000 seed 13 w6/w7 reuse the probe dirs
# (same job, same arms: not rerun). n=4000 gets 160G (r200 w6/w7 hit out-of-memory kills at 64G). Then judge, frontier, r204_fair.py.
# usage: v3_r204_grid.sh [n ...]    (default: 250 500 1000 4000)
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY
R=experiments/v3/results/story_gen
gen() { echo "{\"causal\": 1, \"units\": \"words\", \"pair\": \"gate\", \"new\": true, \"temp\": $1, \"edge\": \"exact\"}"; }
garr() { local out="" sep=""; for t in "$@"; do out="$out$sep$(gen $t)"; sep=", "; done; echo "[$out]"; }
NS="${@:-250 500 1000 4000}"
for n in $NS; do
  if [ $n = 4000 ]; then SEEDS="13 17"; TL=48:00:00; M=160G; else SEEDS="13 17 19 23"; TL=24:00:00; M=48G; fi
  Q=""; DIRS=""
  for w in 5 6 7; do
    G=$(garr 1.3 1.6 2.0)
    D=$R/r204-n$n-w$w; mkdir -p $D; DIRS="$DIRS $D"
    for s in $SEEDS; do
      if [ $n = 1000 ] && [ $s = 13 ] && [ $w != 5 ]; then continue; fi   # the probe's dirs
      j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=$M -t $TL -J v3-r204g --export=ALL -o logs/v3-r204-n$n-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
      Q="$Q:$j"
    done
  done
  PF=""
  if [ $n = 1000 ]; then          # probe dumps are named r204-probe-n1000-w6/7 (the family's window is read from the dir name)
    for w in 6 7; do
      for f in $R/r204-probe-n1000-w$w/story_s13_n1000_arm*; do cp -n $f $R/r204-n1000-w$w/ 2>/dev/null; done
    done
  fi
  JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
  jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 8:00:00 -J v3-r204j --dependency=afterany$Q -o logs/v3-r204-judge-n$n.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
  jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r204f --dependency=afterany:$jj -o logs/v3-r204-fair-n$n.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && export PATH=\$HOME/flash/miniconda3/envs/chunks/bin:\$PATH PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:\$PYTHONPATH && python experiments/v3/story_frontier.py $R/r204-frontier-n$n.jsonl $DIRS > /dev/null && python experiments/v3/diag/r204_fair.py $n $R/r204-fair-n$n $R/r200-frontier-n$n.jsonl $R/r204-frontier-n$n.jsonl; echo end status \$?")
  echo "n$n gen $Q judge $jj fair $jf"
done
