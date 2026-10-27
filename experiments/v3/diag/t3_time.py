"""Learning time alone (docs/TEMPORAL_INTERDEP.md, round 47): TinyStories stream as in
t3_use.py, no measurement between checkpoints, so the clock counts only learn().

Usage: t3_time.py <cfg json> <seed> <checkpoints> <tag>
"""
import json
import random
import sys
import time

sys.path.insert(0, "src")
from trellis2.stories import heldout_path, load_corpus  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
corpus = load_corpus(vocab_size=250)
random.Random(seed).shuffle(corpus)
held_set = {tuple(s) for s in json.load(open(heldout_path(13), encoding="utf-8"))[:200]}
train = [s for s in corpus if tuple(s) not in held_set][:max(checks)]
L = Trellis3(seed=seed, **cfg)
t0, done = time.time(), 0
for c in checks:
    for s in train[done:c]:
        L.learn(s)
    done = c
    print(json.dumps({"tag": tag, "seed": seed, "checkpoint": c, "learn s": round(time.time() - t0),
                      "records": len(L.records), "use repaired": L.stats.get("use repaired", 0)}), flush=True)
