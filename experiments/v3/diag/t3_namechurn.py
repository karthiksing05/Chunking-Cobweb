"""Why word names change between experiences (TinyStories): for every word
type, its context name (bag of concept ids) after experience t vs t-1, and
each change classified as

  weights   same concept ids, different weights
  new node  an id that did not exist before this experience (fringe split,
            merge, new child)
  existing  an id that already existed (a reassignment of where the word's
            instances sit, a split above them, or a band cut moving)

Usage: t3_namechurn.py <cfg json> <seed> <n> [every]
"""
import json
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from trellis2.stories import read_sentences, simple_sentences  # noqa: E402
from trellis2.trellis3 import Elem, Trellis3  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
every = int(sys.argv[4]) if len(sys.argv) > 4 else 50
corpus = simple_sentences(read_sentences(), vocab_size=250)
random.Random(seed).shuffle(corpus)
L = Trellis3(seed=seed, **cfg)


def names(words):
    L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
    out = {}
    for w in words:
        v = L._ctx_name(Elem((0, 1), token=w)) or {}
        if isinstance(v, list):
            v = Counter(u for u in v if u and u != "-")
        out[w] = {u: float(p) for u, p in v.items()}
    return out


def max_id(tree):
    return max(int(nd.id) for nd in tree.nodes())


seen, prev, t0 = set(), None, time.time()
tot = Counter()
for k, s in enumerate(corpus[:n], 1):
    before = max_id(L.ctx)
    L.learn(s)
    seen.update(s)
    if prev is not None:              # the experience right after a checkpoint
        cur = names(seen)
        c = Counter()
        for w, nm in cur.items():
            old = prev.get(w, {})
            if nm == old:
                c["unchanged"] += 1
                continue
            ids_new, ids_old = set(nm), set(old)
            if ids_new == ids_old:
                c["weights only"] += 1
                continue
            added = ids_new - ids_old
            fresh = [u for u in added if str(u).lstrip("c").isdigit() and int(str(u).lstrip("c")) > before]
            c["ids changed"] += 1
            c["  with a new node id"] += bool(fresh)
            c["  only existing ids"] += not fresh
        tot.update(c)
        print(json.dumps({"step": k, "types": len(cur), **c, "s": round(time.time() - t0)}), flush=True)
        prev = None
    if k % every == 0:
        prev = names(seen)
print(json.dumps({"total": dict(tot)}))
