#!/bin/bash
export PATH=$HOME/flash/miniconda3/envs/trellis/bin:$PATH
cd /coc/flash10/ksingara3/Chunking-Cobweb
python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'experiments/v2')
import trellis2, trellis2.cobweb, trellis2.incremental_cobweb, trellis2.data, trellis2.memory, trellis2.evaluation
import trellis2.characters, trellis2.chess, trellis2.stories, trellis2.treebank, trellis2.mdl, complex_grammars
print('imports ok')"
python -m pytest tests -q 2>&1 | tail -15
