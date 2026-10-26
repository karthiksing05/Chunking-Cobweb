"""Held-out TinyStories sentences for silver parsing: corpus[20000:20000+k] after the
seed's shuffle (beyond every training stream). Usage: silver_dump.py <seed> [k]"""
import json
import random
import sys

sys.path.insert(0, "src")
from trellis2.stories import heldout_path, load_corpus  # noqa: E402

seed = int(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 400
corpus = load_corpus(vocab_size=250)
random.Random(seed).shuffle(corpus)
out = heldout_path(seed)
json.dump([list(s) for s in corpus[20000:20000 + k]], open(out, "w", encoding="utf-8"))
print(out, len(corpus))
