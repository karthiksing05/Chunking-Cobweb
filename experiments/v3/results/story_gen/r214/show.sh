#!/bin/bash
cd /coc/flash10/ksingara3/Chunking-Cobweb
~/flash/miniconda3/envs/chunks/bin/python - "$@" <<'P'
import json,sys
d='experiments/v3/results/story_gen/r214'
C=json.load(open(d+'/controls.json')); S=json.load(open(d+'/scores.json'))
key,cl=sys.argv[1],sys.argv[2]
for s in C[cl][:40]: print(f"{S[key][s]:.2f}  {s}")
P
