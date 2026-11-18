#!/bin/bash
# Round 204: n=4000 closing chain. After the 5 original gens' judge (4780737) and the w6 seed-17 rerun (4781938) finish:
# judge the r204 n=4000 dirs again (cached scores are kept; the rerun's dump is new), then the fair step. Overwrites
# experiments/v3/results/story_gen/r204-fair-n4000.{txt,tsv,perseed.tsv} and r204-frontier-n4000.jsonl.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon"
J='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/coc/flash10/ksingara3/data/hf_llm HF_HUB_OFFLINE=1 ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/t3_judge.py'
R=experiments/v3/results/story_gen
DIRS="$R/r204-n4000-w5 $R/r204-n4000-w6 $R/r204-n4000-w7"
JC=""; for d in $DIRS; do JC="$JC $J $d;"; done
jj=$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=32G -t 8:00:00 -J v3-r204j --dependency=afterany:4780737:4781938 -o logs/v3-r204-judge-n4000b.log --open-mode=truncate --wrap "echo host \$(hostname); cd /coc/flash10/ksingara3/Chunking-Cobweb && $JC echo end status \$?")
jf=$(sbatch --parsable $P -c 1 --mem=24G -t 2:00:00 -J v3-r204f --dependency=afterany:$jj -o logs/v3-r204-fair-n4000b.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && export PATH=\$HOME/flash/miniconda3/envs/chunks/bin:\$PATH PYTHONPATH=/coc/flash10/ksingara3/cobweb-private/build-cu:\$PYTHONPATH && python experiments/v3/story_frontier.py $R/r204-frontier-n4000.jsonl $DIRS > /dev/null && python experiments/v3/diag/r204_fair.py 4000 $R/r204-fair-n4000 $R/r200-frontier-n4000.jsonl $R/r204-frontier-n4000.jsonl; echo end status \$?")
echo "judge $jj fair $jf"
