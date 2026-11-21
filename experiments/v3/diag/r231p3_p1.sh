#!/bin/bash
# Round 230 grid on the controlled grammar. usage: r230_p1.sh <seed> <n> [feats=func:5] [K=5]
# C6 (class-level gate + emission at b, reach 5) and C9 (context window read at b, reach W', exact emission); one job per C9 level, one for C6.
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps"
S=${1:-13}; N=${2:-1000}; F=${3:-func:5}; K=${4:-5}; TAG=$(echo $F | tr -d ':')
R=experiments/v3/results/story_gen/r231/p3
BS=${BS-"leaf 7 6 5 4 3 0"}; WS=${WS:-"1 2 3 5"}; C6B=${C6B-"7 6 5 4 3 2 0"}
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY R214_LEVELS R214_WORDMAP R214_WORDTREE
CFG="$(cat experiments/v3/results/story_gen/r214p01/cfg_w5.json)"; CFG="${CFG%\}},\"wemit_tree\":true,\"wtree_feats\":\"$F\"}"
c9() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": $1, \"wemitk\": $K, \"cbits\": \"$2\", \"pair\": \"gate\"}"; }
c6() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": $1, \"wemit\": $2, \"wemitk\": $K, \"wgate\": $2, \"pair\": \"gate\"}"; }
leafarm() { echo "{\"causal\": 1, \"units\": \"words\", \"new\": true, \"temp\": 1.0, \"edge\": \"exact\", \"wreach\": $1, \"wemit\": \"leaf\", \"pair\": \"gate\"}"; }
sub() { mkdir -p $1; STORY_DUMP=$1 sbatch --parsable $P -c 1 --mem=${MEM-32G} -t 48:00:00 -J r231$4 --export=ALL -o logs/v3-r231-$(basename $1).log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r231_story.py "$CFG" $S $N story "$2" story; }
A=""; for w in ${LW-0 1 2 3 5}; do A="$A, $(leafarm $w)"; done; for b in $C6B; do A="$A, $(c6 5 $b)"; done
[ -n "${NOC6-}" ] || echo "c6 $(sub $R/${TAG}_s${S}_n${N}_c6${C6TAG} "[${A:2}]" x c6)"
for b in $BS; do A=""; for w in $WS; do A="$A, $(c9 $w $b)"; done
  echo "c9 b=$b $(sub $R/${TAG}_s${S}_n${N}_c9b$b "[${A:2}]" x c9$b)"; done
