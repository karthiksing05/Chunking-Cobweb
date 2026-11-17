#!/bin/bash
# Round 194 pre-test: conjunctive context attributes ("conj": true in the Trellis3 cfg; lj2..ljW, rj2..rjW).
# (1) held-out log-loss of the causal fill read (experiments/v3/diag/r194_logloss.py), conj vs not, w3 w4, seeds 13 17, n 1000;
# (2) generation with conj, "both" on, temp 0.7, w3 w4, seeds 13 17, n 1000 -> judge -> story_frontier.py together with the
#     r192 n1000 dirs ("both" without conj; its temp-0.7 arm is arm1);
# (3) default byte-identity: r191-n250-w3 seed 13 rerun with the current code, compared to the stored dump;
# (4) the full tests/trellis2 suite.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen; O=$R/r192-diag; n=1000
mkdir -p $O
L=""
for w in 3 4; do for s in 13 17; do for c in false true; do
  j=$(sbatch --parsable $P -c 1 --mem=24G -t 6:00:00 -J v3-r194l -o $O/r194_logloss_w${w}_s${s}_conj$c.txt --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r194_logloss.py "{$S,$D0,\"story_facet\":true,\"window\":$w,\"conj\":$c}" $s $n)
  L="$L:$j"
done; done; done
G='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.7, "both": true}]'
Q=""; DIRS=""
for w in 3 4; do
  D=$R/r194-n$n-conj-w$w; mkdir -p $D; DIRS="$DIRS $D"
  for s in 13 17; do
    j=$(STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=24G -t 12:00:00 -J v3-r194 --export=ALL -o logs/v3-r194-n$n-conj-w$w-$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":$w,\"conj\":true}" $s $n story "$G" story)
    Q="$Q:$j"
  done
done
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 4:00:00 -J v3-r194j --dependency=afterany$Q -o logs/v3-r194-judge-n$n.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r194f --dependency=afterany:$jj -o logs/v3-r194-pre-frontier.log --open-mode=truncate logs/v3_job.sh experiments/v3/story_frontier.py $R/r194-pre-frontier-n$n.jsonl $DIRS $R/r192-n$n-w3 $R/r192-n$n-w4 $R/r191-n$n-refs)
# (3) byte identity of the default path
B=$R/r194-byteid-n250-w3; mkdir -p $B
GB='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.5}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 0.7}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 1.0}]'
jb=$(STORY_DUMP=$B sbatch --parsable $P -c 1 --mem=24G -t 4:00:00 -J v3-r194b --export=ALL -o logs/v3-r194-byteid.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":3}" 13 250 story "$GB" story)
jt=$(sbatch --parsable $P -c 2 --mem=16G -t 2:00:00 -J v3-r194t -o logs/v3-r194-tests-all.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && PYTHONPATH=src:tests/trellis2 timeout 6000 ~/flash/miniconda3/envs/trellis/bin/python -m pytest -q -x tests/trellis2 2>&1 | tail -15")
echo "logloss $L gen $Q judge $jj frontier $jf byteid $jb tests $jt"
