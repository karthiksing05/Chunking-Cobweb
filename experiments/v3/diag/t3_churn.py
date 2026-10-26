"""How many stored records change description per experience on TinyStories
(the work an event-driven repair would face): refresh every record every
step (relabel, no budget) and log the refreshed / reassigned counts.

Usage: t3_churn.py <cfg json> <seed> <n> [every]
"""
import json
import random
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from trellis2.stories import read_sentences, simple_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
every = int(sys.argv[4]) if len(sys.argv) > 4 else 50
corpus = simple_sentences(read_sentences(), vocab_size=250)
random.Random(seed).shuffle(corpus)
L = Trellis3(seed=seed, **cfg)
t0, last = time.time(), dict(L.stats)
for k, s in enumerate(corpus[:n], 1):
    L.learn(s)
    if k % every == 0:
        d = {key: L.stats.get(key, 0) - last.get(key, 0) for key in ("refreshed", "reassigned")}
        last = dict(L.stats)
        old = sum(1 for e in L.records if e.token is None)
        print(json.dumps({"step": k, "records": len(L.records), "chunk records": old,
                          "refreshed per step": round(d["refreshed"] / every, 1),
                          "reassigned per step": round(d["reassigned"] / every, 1),
                          "s": round(time.time() - t0)}), flush=True)
