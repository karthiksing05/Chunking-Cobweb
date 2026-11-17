"""Round 194 pre-test (log-loss): held-out next-word bits of the causal context read, P(w | l1..lW) from
Trellis3._gap_fills (the words-only fill distribution, as the causal arm of t3_story.py reads it),
for a learner rebuilt as in t3_story.py / r193_pretest.py. Held-out sentences come from the first 1500 stories (never
trained). Reported overall and split by whether the (l2, l1, w) trigram is stored (a new trigram is where the
conjunction matters). As in the causal arm, each word carries whether it ends the sentence: P(w, end | left) =
fill(left + r1=EOS)[w], P(w, not end | left) = fill(left)[w] - that, normalized over the fill(left) mass.

Usage (compute node): r194_logloss.py <cfg json> <seed> <n> [held-out sentences, default 400]
"""
import json
import math
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import BOS, EOS, Trellis3  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
H = int(sys.argv[4]) if len(sys.argv) > 4 else 400
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
train = []
for st in stories[1500:]:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, any(keep(t) for t in st[:i])))
    if len(train) >= n:
        break
train = train[:n]
seen = {tuple(s) for s, _ in train}
held = [s for st in stories[:1500] for s in st if keep(s) and tuple(s) not in seen][:H]
t0 = time.time()
L = Trellis3(seed=seed, **cfg)
for s, fol in train:
    L.learn(s, follows=fol)
L._quiet = True
t1 = time.time()
tri = L.stored_trigrams()
W = L.window
words = lambda d: {v[1]: x for v, x in d.items() if v[0] == "w"}
bits = {"all": [], "stored": [], "new": []}
for s in held:
    t = [BOS] * W + list(s) + [EOS]
    for i in range(W, len(t) - 1):
        q = {f"l{k}": t[i - k] for k in range(1, W + 1)}
        m, e = words(L._gap_fills(q)), words(L._gap_fills({**q, "r1": EOS}))
        z, mw = sum(m.values()), m.get(t[i], 0.0)
        me = min(e.get(t[i], 0.0), mw)
        p = (me if t[i + 1] == EOS else mw - me) / z if z > 0 else 0.0
        b = -math.log2(max(p, 1e-12))
        bits["all"].append(b)
        bits["stored" if (t[i - 2], t[i - 1], t[i]) in tri else "new"].append(b)
mean = lambda v: round(sum(v) / len(v), 4) if v else None
print(json.dumps({"seed": seed, "n": n, "window": W, "conj": bool(cfg.get("conj")), "held": len(held),
                  "bits/word": mean(bits["all"]), "bits | stored trigram": mean(bits["stored"]),
                  "bits | new trigram": mean(bits["new"]), "tokens": len(bits["all"]), "new tokens": len(bits["new"]),
                  "zero-mass": sum(b >= 39 for b in bits["all"]), "train s": round(t1 - t0), "eval s": round(time.time() - t1)}),
      flush=True)
