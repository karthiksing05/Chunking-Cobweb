"""Why does word_cc decide copying? A generation slot's pool is the stored members of the
context concept just above a stored word's leaf. This counts the distinct words in that
concept, weighted by word occurrences. Prediction: with word_cc on, most occurrences sit
under a concept holding that word alone, so the slot can only give back the same word.

Usage: t3_slot_words.py <tag> <cfg json> <conds> <seed> <n>
"""
import json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.trellis3 import Trellis3

tag, cfg = sys.argv[1], json.loads(sys.argv[2])
seed, n = int(sys.argv[4]), int(sys.argv[5])
for cond in sys.argv[3].split(","):
    train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    L = Trellis3(seed=seed, **cfg)
    for e in train[:n]:
        L.learn(e.tokens)
    words = [e for e in L.records if e.token is not None and e.leaf is not None and e.weight > 0]
    out = {"tag": tag, "cond": cond, "seed": seed, "word occurrences": len(words)}
    for up in (1, 2, 3):
        pool = defaultdict(Counter)
        place = []
        for e in words:
            path = L.ctx.ancestors(e.leaf)[::-1]
            node = path[max(len(path) - 1 - up, 1 if len(path) > 1 else 0)]
            pool[node.id][e.token] += e.weight
            place.append((node.id, e.weight))
        tot = sum(w for _, w in place)
        out[f"up {up}"] = {
            "mean distinct words": round(sum(w * len(pool[i]) for i, w in place) / tot, 2),
            "share under a one-word concept": round(sum(w for i, w in place if len(pool[i]) == 1) / tot, 3),
            "concepts": len(pool)}
    print(json.dumps(out), flush=True)
