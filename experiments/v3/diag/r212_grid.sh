#!/bin/bash
# Round 212: more seeds for the generation claim. New seeds 29 31 37 41 at n 250/500/1000:
#   TRELLIS edge (exact) at the r205-selected window with the selected temperature and one neighbour each side; LSTM and GPT at t .7/.8/.9.
#   Part B: n250 w7 edge t4.5 and t6.0 on seeds 13 17 19 23.
# Then judge (3B), frontier rows, and r212_analysis.py. Dirs r212-n$n-w$w, r212-n$n-refs.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY
R=experiments/v3/results/story_gen
gen() { echo "{\"causal\": 1, \"units\": \"words\", \"pair\": \"gate\", \"new\": true, \"temp\": $1, \"edge\": \"exact\"}"; }
garr() { local out="" sep=""; for t in "$@"; do out="$out$sep$(gen $t)"; sep=", "; done; echo "[$out]"; }
lms() { local out="" sep=""; for m in gpt lstm; do for t in 0.7 0.8 0.9; do out="$out$sep{\"lm\": \"$m\", \"new\": true, \"temp\": $t}"; sep=", "; done; done; echo "[$out]"; }
NEW="29 31 37 41"; OLD="13 17 19 23"; TL=48:00:00
QALL=""
for n in 250 500 1000; do
  case $n in 250) W=7; TS="3.0 3.5 4.0"; M=48G;; 500) W=6; TS="2.4 2.8 3.2"; M=48G;; 1000) W=7; TS="1.6 2.0 2.4"; M=128G;; esac
  D=$R/r212-n$n-w$W; mkdir -p $D; DIRS="$D"; Q=""
  G=$(garr $TS)
  for s in $NEW; do
    j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=$M -t $TL -J v3-r212g --export=ALL -o logs/v3-r212-n$n-w$W-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$W}" $s $n story "$G" story)
    Q="$Q:$j"
  done
  if [ $n = 250 ]; then
    G=$(garr 4.5 6.0)
    for s in $OLD; do
      j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=$M -t $TL -J v3-r212gB --export=ALL -o logs/v3-r212-n$n-w$W-B$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$W}" $s $n story "$G" story)
      Q="$Q:$j"
    done
  fi
  B="$(lms)"
  DR=$R/r212-n$n-refs; mkdir -p $DR; DIRS="$DIRS $DR"
  for s in $NEW; do
    j=$(STORY_BASE="$B" STORY_DUMP=$DR sbatch --parsable $P -c 4 --mem=24G -t $TL -J v3-r212r --export=ALL -o logs/v3-r212-n$n-refs-$s.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true}" $s $n story "$B" story)
    Q="$Q:$j"
  done
  JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
  jj=$(sbatch --parsable $P -c 4 --gres=gpu:a40:1 --mem=32G -t 8:00:00 -J v3-r212j --dependency=afterany$Q -o logs/v3-r212-judge-n$n.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
  echo "n$n gen $Q judge $jj"
  QALL="$QALL:$jj"; ALLDIRS="$ALLDIRS $DIRS"
done
ja=$(sbatch --parsable $P -c 1 --mem=32G -t 4:00:00 -J v3-r212a --dependency=afterany$QALL -o logs/v3-r212-analysis.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && export PATH=\$HOME/flash/miniconda3/envs/chunks/bin:\$PATH PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:\$PYTHONPATH && for n in 250 500 1000; do python experiments/v3/story_frontier.py $R/r212-frontier-n\$n.jsonl \$(ls -d $R/r212-n\$n-*) > /dev/null || exit 1; done && python experiments/v3/diag/r212_analysis.py; echo end status \$?")
echo "analysis $ja"
