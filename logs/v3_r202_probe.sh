#!/bin/bash
# Round 202 probe: the parse-guided arm at n=1000, seed 13, window 5, temps 1.0 and 1.3 (t3_story_r202.py), judged,
# then fair / envelope / classes against the plain arm's seed-13 dumps (r184/r188/r190/r200 n1000 w5).
# Usage: bash logs/v3_r202_probe.sh <tag> '<extra arm json, e.g. "gen": "parse">' [temps, default "1.0 1.3"]
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen
T=$1; X=$2; TEMPS=${3:-"1.0 1.3"}
G=""; sep=""; for t in $TEMPS; do G="$G$sep{\"causal\": 1, \"units\": \"words\", \"pair\": \"gate\", \"new\": true, \"temp\": $t, $X}"; sep=", "; done; G="[$G]"
D=$R/$T-n1000-w5; mkdir -p $D
j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=32G -t 24:00:00 -J v3-$T --export=ALL -o logs/v3-$T-n1000-w5-13.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story_r202.py "{$S,$D0,\"story_facet\":true,\"window\":5}" 13 1000 story "$G" story)
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 4:00:00 -J v3-${T}j --dependency=afterok:$j -o logs/v3-$T-judge.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $J $D; echo end status \$?")
PL="$R/r184-n1000-w5 $R/r188-n1000-w5 $R/r190-n1000-w5 $R/r200-n1000-w5"
jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-${T}f --dependency=afterok:$jj -o logs/v3-$T-fair.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && export PATH=\$HOME/flash/miniconda3/envs/chunks/bin:\$PATH && python experiments/v3/story_frontier.py $R/$T-frontier-n1000.jsonl $D > /dev/null && R202_SEEDS=13 python experiments/v3/diag/r202_fair.py 1000 $R/$T-fair-n1000 $R/r200-frontier-n1000.jsonl $R/$T-frontier-n1000.jsonl && R202_SEEDS=13 python experiments/v3/diag/r202_classes.py $R/$T-classes-n1000.tsv $D $PL; echo end status \$?")
echo "$T gen $j judge $jj fair $jf"
