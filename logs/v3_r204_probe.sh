#!/bin/bash
# Round 204 probe (step 2): edge_exact generation, seed 13, n=1000, w6 and w7, temp 1.3 1.6 2.0; judge; fair count vs r200 (seed 13 only).
# Same stream / prompts / learner / judge as r200 (logs/v3_r200_submit.sh), plus the arm key "edge": "exact" (t3_story.py).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY
R=experiments/v3/results/story_gen
n=1000; seed=13
gen() { echo "{\"causal\": 1, \"units\": \"words\", \"pair\": \"gate\", \"new\": true, \"temp\": $1, \"edge\": \"exact\"}"; }
garr() { local out="" sep=""; for t in "$@"; do out="$out$sep$(gen $t)"; sep=", "; done; echo "[$out]"; }
Q=""; DIRS=""
for w in 6 7; do
  G=$(garr 1.3 1.6 2.0)
  D=$R/r204-probe-n$n-w$w; mkdir -p $D; DIRS="$DIRS $D"
  j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=48G -t 12:00:00 -J v3-r204g --export=ALL -o logs/v3-r204-probe-n$n-w$w-$seed.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $seed $n story "$G" story)
  Q="$Q:$j"
done
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 4:00:00 -J v3-r204j --dependency=afterany$Q -o logs/v3-r204-probe-judge.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r204f --dependency=afterany:$jj -o logs/v3-r204-probe-fair.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && export PATH=\$HOME/flash/miniconda3/envs/chunks/bin:\$PATH PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:\$PYTHONPATH && python experiments/v3/story_frontier.py $R/r204-probe-frontier-n$n.jsonl $DIRS > /dev/null && R204_SEEDS=13 python experiments/v3/diag/r204_fair.py $n $R/r204-probe-fair-n$n $R/r200-frontier-n$n.jsonl $R/r204-probe-frontier-n$n.jsonl; echo end status \$?")
echo "gen $Q judge $jj fair $jf"
