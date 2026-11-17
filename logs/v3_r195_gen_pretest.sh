#!/bin/bash
# Round 195 pre-test 2: generation with the level-mixing read ("read": $1, Trellis3.hier_fills) + "both", temp 0.7,
# w3 w4 w5, seeds 13 17, n 1000 -> judge -> compare against the matched r192 "both" dumps (arm1 = temp 0.7).
# $2 = "byteid" also reruns the default r191-n250-w3 seed 13 for byte identity.
# Usage: bash logs/v3_r195_gen_pretest.sh <read> [byteid]
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
RD=$1
R=experiments/v3/results/story_gen; O=$R/r192-diag; n=1000
G='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.7, "both": true, "read": "'$RD'"}]'
Q=""; DIRS=""
for w in 3 4 5; do
  D=$R/r195-n$n-$RD-w$w; mkdir -p $D; DIRS="$DIRS $D"
  for s in 13 17; do
    j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=24G -t 16:00:00 -J v3-r195 --export=ALL -o logs/v3-r195-n$n-$RD-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n story "$G" story)
    Q="$Q:$j"
  done
done
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
CMP=""; for w in 3 4 5; do CMP="$CMP ~/flash/miniconda3/envs/trellis/bin/python experiments/v3/diag/r194_compare.py r192-both-w$w '$R/r192-n$n-w$w/story_s{s}_n${n}_arm1.jsonl' r195-$RD-w$w '$R/r195-n$n-$RD-w$w/story_s{s}_n${n}_arm0.jsonl' 13,17 $n;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 4:00:00 -J v3-r195j --dependency=afterany$Q -o logs/v3-r195-judge-$RD.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
jc=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r195c --dependency=afterany:$jj -o $O/r195_${RD}_compare.txt --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && export OMP_NUM_THREADS=1; $CMP echo end status \$?")
echo "gen $Q judge $jj compare $jc"
if [ "$2" = byteid ]; then
  B=$R/r195-byteid-n250-w3; mkdir -p $B
  GB='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.5}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.7}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 1.0}]'
  jb=$(STORY_DUMP=$B sbatch --parsable $P -c 1 --mem=24G -t 4:00:00 -J v3-r195b --export=ALL -o logs/v3-r195-byteid.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":3}" 13 250 story "$GB" story)
  echo "byteid $jb"
fi
