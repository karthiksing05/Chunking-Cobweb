#!/usr/bin/env bash
# Rerun every TRELLIS v2 experiment with the current night, one after another,
# logging each to logs/. On a cluster, submit these lines as separate jobs and
# give the long English nights (stories_9000, stories_250_10000) their own nodes.
# Usage: WORKERS=8 bash experiments/v2/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${WORKERS:-8}
R=experiments/v2/results
mkdir -p logs
run() { local name=$1; shift; echo "$(date) start $name"; "$@" > "logs/$name.log" 2>&1 || echo "$(date) FAILED: $name"; }

run unsupervised python experiments/v2/run_unsupervised.py --seeds 13,17 --workers "$W"
run incremental bash -c "python experiments/v2/run_incremental.py --seeds 13,17 --workers $W && python experiments/v2/plot_incremental.py $R/incremental"
run prompts_synthetic python experiments/v2/run_prompts_synthetic.py --workers "$W"
run characters python experiments/v2/run_characters.py --out "$R/characters"
for n in 10 15 20; do
  run "wsj$n" python experiments/v2/run_treebank.py --train-max-len "$n" --out "$R/treebank/wsj$n"
done
run stories_2500 python experiments/v2/run_stories.py --train 2500 --workers "$W" --out "$R/stories_2500"
run stories_250 python experiments/v2/run_stories.py --train 2500 --vocab 250 --max-len 8 --workers "$W" --out "$R/stories_250"
run stories python experiments/v2/run_stories.py --train 5000 --workers "$W" --out "$R/stories"
run prompts python experiments/v2/run_prompts.py --grammar "$R/stories/grammar.pkl" --out "$R/prompts"
run stories_9000 python experiments/v2/run_stories.py --train 9000 --workers "$W" --out "$R/stories_9000"
run stories_250_10000 python experiments/v2/run_stories.py --train 10000 --vocab 250 --max-len 8 --workers "$W" --out "$R/stories_250_10000"
echo "$(date) done"
