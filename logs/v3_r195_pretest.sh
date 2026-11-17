#!/bin/bash
# Round 195 pre-test 1: held-out log-loss of the level-mixing causal reads (Trellis3.hier_fills: kn, top, ngram, path,
# post) against the default product read, one learner per (w, seed), n 1000 (experiments/v3/diag/r195_logloss.py);
# plus the tests/trellis2 suite (incl. test_r195_hier.py).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
R=experiments/v3/results/story_gen; O=$R/r192-diag; n=1000
mkdir -p $O
L=""
for w in 3 4 5; do for s in 13 17; do
  j=$(sbatch --parsable $P -c 1 --mem=24G -t 8:00:00 -J v3-r195l -o $O/r195_logloss_w${w}_s${s}.txt --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r195_logloss.py "{$S,$D0,\"story_facet\":true,\"window\":$w}" $s $n)
  L="$L:$j"
done; done
jt=$(sbatch --parsable $P -c 2 --mem=16G -t 2:00:00 -J v3-r195t -o logs/v3-r195-tests.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && PYTHONPATH=src:tests/trellis2 timeout 6000 ~/flash/miniconda3/envs/trellis/bin/python -m pytest -q tests/trellis2/test_r195_hier.py 2>&1 | tail -30")
echo "logloss $L tests $jt"
