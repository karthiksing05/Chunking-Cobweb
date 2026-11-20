#!/bin/bash
# Round 225 launcher (login node: only sbatch). Usage: r225_launch.sh trellis|lms|gen
#   trellis: TRELLIS arms (full, nochunk, noconcept, neither) x n 250..4000 x seeds 13 17, bits at every cell; parse metrics at n=1000
#   lms:     BPE + n-grams + LSTM + GPT held-out bits, n 250..4000 x seeds 13 17 (GPU)
#   gen:     generation at n=1000, 3 temperatures per arm, then judge2 (Qwen3-8B + phi-4) and the report (dependency chain)
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps"
cd /coc/flash10/ksingara3/Chunking-Cobweb
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP STORY_R204_REPLAY
mem() { case $1 in 250) echo 32G;; 500) echo 48G;; 1000) echo 64G;; 2000) echo 100G;; 4000) echo 200G;; esac; }
case $1 in
trellis)
  for n in 4000 2000 1000 500 250; do
    for s in 13 17; do
      for a in full nochunk noconcept neither noconceptp; do
        [ $a = noconceptp ] && [ $n != 1000 ] && continue
        mode=bits; [ $n = 1000 ] && mode=parse
        echo "$(sbatch --parsable $P -c 1 --mem=$(mem $n) -t 24:00:00 -J v3-r225-$a --export=ALL -o logs/v3-r225-$a-s$s-n$n.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r225_trellis.py $s $n $a $mode) $a s$s n$n $mode"
      done
    done
  done;;
lms)
  for n in 4000 2000 1000 500 250; do
    for s in 13 17; do
      m=24G; [ $n -ge 2000 ] && m=48G
      echo "$(sbatch --parsable $P -c 4 --gres=gpu:1 --mem=$m -t 24:00:00 -J v3-r225-lms --export=ALL -o logs/v3-r225-lms-s$s-n$n.log --open-mode=truncate logs/v3_job_lm.sh experiments/v3/diag/r225_lms.py $s $n) lms s$s n$n"
    done
  done;;
gen)
  R=experiments/v3/results/story_gen/r225
  BASE='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0,"restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content","story_facet":true,"window":7'
  ARM='[{"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 1.2, "edge": "exact"}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 2.0, "edge": "exact"}, {"causal": 1, "units": "words", "pair": "gate", "new": true, "temp": 3.0, "edge": "exact"}]'
  Q=""
  for s in 13 17; do
    for a in full nochunk noconcept neither noconceptp; do
      st=posterior; kw='{}'
      case $a in nochunk) st=none;; noconcept) kw='{"concepts_off": true}';; neither) st=none; kw='{"concepts_off": true}';; noconceptp) kw='{"concepts_off": true, "concepts_off_parse": true}';; esac
      D=$R/${a}_s$s; mkdir -p $D
      j=$(R225_KW="$kw" STORY_DUMP=$D sbatch --parsable $P -c 1 --mem=96G -t 24:00:00 -J v3-r225g-$a --export=ALL -o logs/v3-r225-gen-$a-s$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r225_story.py "{$BASE,\"store\":\"$st\"}" $s 1000 story "$ARM" story)
      echo "$j gen $a s$s"; Q="$Q:$j"
    done
  done
  jp=$(sbatch --parsable $P -c 2 --mem=16G -t 1:00:00 -J v3-r225-prep --dependency=afterok$Q -o logs/v3-r225-prep.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r225_gen_report.py prep)
  jj=$(sbatch --parsable $P -c 4 --gres=gpu:a40:1 --mem=48G -t 8:00:00 -J v3-r225-judge --dependency=afterok:$jp -o logs/v3-r225-judge.log --open-mode=truncate --wrap "cd /coc/flash10/ksingara3/Chunking-Cobweb && export HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ~/flash/miniconda3/envs/chunks/bin/python experiments/v3/diag/r214_judge2.py score $R/sentences.txt $R/scores.tsv; echo end status \$?")
  jr=$(sbatch --parsable $P -c 1 --mem=16G -t 1:00:00 -J v3-r225-rep --dependency=afterok:$jj -o logs/v3-r225-rep.log --open-mode=truncate logs/v3_job.sh experiments/v3/diag/r225_gen_report.py report)
  echo "prep $jp judge $jj report $jr";;
esac
