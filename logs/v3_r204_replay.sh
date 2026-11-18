#!/bin/bash
# Round 204 pre-test: replay the boundary steps of the r200 w6 t1.6 seed-13 n=1000 dump under default vs edge_exact.
S='"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0='"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
unset STORY_DIAG STORY_PAIRS STORY_LICENCE FILL_GAMMA FILL_BACKOFF STORY_BOUND STORY_BASE STORY_DUMP
export STORY_R204_REPLAY=experiments/v3/results/story_gen/r200-n1000-w6
export R204_OUT=experiments/v3/results/story_gen/r204_replay_s${1}.json
cd /coc/flash10/ksingara3/Chunking-Cobweb
bash logs/v3_job.sh experiments/v3/diag/t3_story.py "{$S,$D0,\"story_facet\":true,\"window\":6}" $1 1000 r204replay "[]" story
