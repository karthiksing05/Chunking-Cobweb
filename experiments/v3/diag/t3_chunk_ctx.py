"""How much does chunk context tell generation about the next word?

Held-out story sentences are read left to right. At each position the real word is
scored under the distribution generation draws from, given the words before it and
the sentence's length (no previous sentence, no topic). Nats per word; higher is better.

  window        Cobweb's prediction from the surface window (the two words before it)
  flat          inside-outside, every composite one symbol: concepts of words, no chunk identity
  chunks+window inside-outside over the whole tree, but only the two words before it are given
  chunks        inside-outside over the whole tree, every word before it given
  window+chunks the product of window and chunks, renormalised (fill "both")

chunks minus flat is what chunk identity adds; chunks minus chunks+window is what the
chunks beyond the surface window add. Each distribution is mixed with 1% uniform, so an
unseen word costs a bounded amount.

Usage: t3_chunk_ctx.py <cfg json> <seed> <n> [sentences]
"""
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.chart_gen import ChartGen, HOLE  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
N_HELD = int(sys.argv[4]) if len(sys.argv) > 4 else 200
N_TEST_STORIES, EPS, WIN = 1500, 0.01, 2

with open(default_tinystories_path(), encoding="utf-8") as f:
    raw = f.read()
stories = []
for st in raw.split("<|endoftext|>"):
    st = re.sub(r"\s+", " ", st.replace("’", "'").replace("‘", "'"))
    sents = [w for w in (WORD.findall(m.group(0).lower()) for m in re.finditer(r"[^.!?]+[.!?]", st)) if w]
    if sents:
        stories.append(sents)
all_sents = read_sentences()
vocab = {w for w, _ in Counter(w for s in all_sents for w in s).most_common(250)}
keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
random.Random(seed).shuffle(stories)
test, train_st = stories[:N_TEST_STORIES], stories[N_TEST_STORIES:]
train = []
for st in train_st:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, i > 0 and keep(st[i - 1])))
    if len(train) >= n:
        break
train = train[:n]
seen = {tuple(s) for s, _ in train}
held = [s for st in test for s in st if keep(s) and tuple(s) not in seen]
random.Random(seed + 3).shuffle(held)
held = held[:N_HELD]

L = Trellis3(seed=seed, **cfg)
t0 = time.time()
for s, fol in train:
    L.learn(s, follows=fol)
L._quiet = True
print(json.dumps({"seed": seed, "n": n, "learn s": round(time.time() - t0), "held": len(held)}), flush=True)

BASE = {"ctx_cut": 12, "lam": 4, "word_sym": "context", "cut_by": "bits"}
FLAT, FULL = ChartGen(L, cut=0, **BASE), ChartGen(L, cut=64, **BASE)
print(json.dumps({"flat": FLAT.stats, "whole tree": FULL.stats}), flush=True)
V = len(vocab)


def word_dist(G, toks, i):
    """P(w at i) from the chart, the words in ``toks`` given and None elsewhere."""
    lifted = [False] * len(toks)
    mu = G._marginal(toks, lifted, None, G.start, i)
    if mu is None:
        return {}
    pc = defaultdict(float)
    for k in np.nonzero(mu > mu.max() * 1e-6)[0]:
        c = G.emit.get(int(k))
        if c:
            tot = sum(c.values())
            for w, x in c.items():
                pc[w] += mu[k] * x / tot
    return pc


def score(d, w):
    tot = sum(d.values())
    return math.log((1 - EPS) * d.get(w, 0.0) / tot + EPS / V) if tot > 0 else math.log(EPS / V + (1 - EPS) / V)


tot, by_pos = defaultdict(float), defaultdict(lambda: defaultdict(float))
count, count_pos = 0, Counter()
t0 = time.time()
for si, s in enumerate(held):
    m = len(s)
    for i in range(m):
        before = [w if k < i else None for k, w in enumerate(s)]
        near = [w if i - WIN <= k < i else None for k, w in enumerate(s)]
        win = FULL._cobweb_fills([w if w is not None else HOLE for w in before], i)
        chunks = word_dist(FULL, before, i)
        arms = {"window": win, "flat": word_dist(FLAT, before, i), "chunks+window": word_dist(FULL, near, i),
                "chunks": chunks, "window+chunks": {w: x * win[w] for w, x in chunks.items() if w in win} or chunks}
        for a, d in arms.items():
            v = score(d, s[i])
            tot[a] += v
            by_pos[min(i, 5)][a] += v
        count += 1
        count_pos[min(i, 5)] += 1
    if (si + 1) % 25 == 0:
        print(json.dumps({"sentences": si + 1, "s": round(time.time() - t0),
                          "nats per word": {a: round(v / count, 4) for a, v in tot.items()}}), flush=True)
print(json.dumps({"tag": "chunk context", "seed": seed, "n": n, "words": count,
                  "nats per word": {a: round(v / count, 4) for a, v in tot.items()},
                  "by position (5 = 5 or later)": {str(p): {a: round(v / count_pos[p], 4) for a, v in d.items()}
                                                   for p, d in sorted(by_pos.items())}}), flush=True)
