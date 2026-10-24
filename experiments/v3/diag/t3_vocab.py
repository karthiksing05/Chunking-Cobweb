"""Vocabulary size and word-frequency profile of each condition's training split."""
import os, sys
from collections import Counter
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
for cond in CONDITIONS:
    try:
        train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
    except Exception as ex:
        print(cond, "error", ex); continue
    c = Counter(t for e in train for t in e.tokens)
    print(cond, "sentences", len(train), "types", len(c), "seen<=3", sum(1 for v in c.values() if v <= 3), flush=True)
