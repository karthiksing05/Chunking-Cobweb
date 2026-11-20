#!/bin/bash
# Round 228: chunks-off / concepts-off controls in chess. Login node: only sbatch.
# (a) bits per position, recipe of logs/v3_r103_submit.sh (ab-control), four test sets, seeds 13, 17, 19:
#     full (max_size 4) and chunks off (max_size 1); concepts off = the (content 0, context 0) row of the full run's "by levels".
# (b) Chase & Simon trial-1 recall at 4,000 positions (K=7, K=4): chunks off (max_size 1) and concepts off (CS_LEAF=1, leaf-only reads).
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon,chomps -c 1"
C='"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"interact","exact_alone":true,"chess_abs":false'
CC='"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"any"'
for s in 13 17 19; do
  T=data/chess/transfer; [ $s != 13 ] && T=data/chess/transfer_s$s
  for S in control family wing shift; do
    sbatch --parsable $P --mem=32G -t 24:00:00 -J r228b --export=ALL,ACOPY=1,ROUTEW=2,COVER=2,LEARN_FILE=$T/learn.fen,CODE_FILE=$T/$S.fen -o logs/r228-full-$S-s$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/chess_compress_gen.py "{$C}" $s r228-full-$S 4000 4000 1000
    sbatch --parsable $P --mem=16G -t 4:00:00 -J r228b --export=ALL,ACOPY=1,ROUTEW=2,COVER=2,LEARN_FILE=$T/learn.fen,CODE_FILE=$T/$S.fen -o logs/r228-off-$S-s$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/chess_compress_gen.py "{$C,\"max_size\":1}" $s r228-off-$S 4000 4000 1000
  done
  sbatch --parsable $P --mem=32G -t 36:00:00 -J r228c -o logs/r228-csoff-s$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/r228_cs_leaf.py "{$CC,\"max_size\":1}" $s r228-csoff 4000 10 16
  sbatch --parsable $P --mem=32G -t 36:00:00 -J r228c --export=ALL,CS_LEAF=1 -o logs/r228-csleaf-s$s.log --open-mode=truncate logs/v3_job.sh experiments/v3/r228_cs_leaf.py "{$CC}" $s r228-csleaf 4000 10 16
done
