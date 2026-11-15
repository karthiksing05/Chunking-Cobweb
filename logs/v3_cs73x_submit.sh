#!/bin/bash
# Chase & Simon (1973) exact replication (experiments/v3/chess_cs1973_exact.py): learning curve 0..4660 (all of
# train.fen), 10 simulated subjects per seed, K = 7 (K = 4 sensitivity), theta 16, trial learning in forked children
P="--account=overcap --partition=overcap --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,walle,jill,gideon -c 1"
C='"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"any"'
for s in 13 17 19; do
  sbatch --parsable $P --mem=32G -t 36:00:00 -J v3-cs73x -o logs/v3-cs73x-$s.log --open-mode=truncate \
    logs/v3_job.sh experiments/v3/chess_cs1973_exact.py "{$C}" $s cs73x 0,100,300,1000,2000,4000,4660 10 16
done
