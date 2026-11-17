"""Round 195 pre-test (log-loss): held-out next-word bits of the causal read, the default product read
(Trellis3._gap_fills, as r194_logloss.py) against the level-mixing reads of Trellis3.hier_fills ("kn", "top",
"ngram", "path", "post"), all on ONE learner rebuilt as in t3_story.py. Each word carries whether it ends the
sentence (P(w, end | left) and P(w, not end | left)). Split by whether (l2, l1, w) is stored; "common" = the
tokens every read gives mass > 0 (the default read gives none to words memory never committed as fillers).

Usage (compute node): r195_logloss.py <cfg json> <seed> <n> [held-out sentences, default 400] [reads, comma list]
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
READS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["default", "kn", "top", "ngram", "path", "post"]
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
toks = []          # (stored?, {read: p})
secs = Counter()
for s in held:
    t = [BOS] * W + list(s) + [EOS]
    for i in range(W, len(t) - 1):
        q = {f"l{k}": t[i - k] for k in range(1, W + 1)}
        fin = t[i + 1] == EOS
        ps = {}
        for r in READS:
            a = time.time()
            if r == "default":
                m, e = words(L._gap_fills(q)), words(L._gap_fills({**q, "r1": EOS}))
                z = sum(m.values())
            else:
                m, e = L.hier_fills(q, how=r, only={t[i]})
                m, e = words(m), words(e)
                z = 1.0
            mw = m.get(t[i], 0.0)
            me = min(e.get(t[i], 0.0), mw)
            ps[r] = (me if fin else mw - me) / z if z > 0 else 0.0
            secs[r] += time.time() - a
        toks.append(((t[i - 2], t[i - 1], t[i]) in tri, ps))
mean = lambda v: round(sum(v) / len(v), 4) if v else None
bits = lambda p: -math.log2(max(p, 1e-12))
common = [k for k, (_, ps) in enumerate(toks) if all(p > 0 for p in ps.values())]
out = {"seed": seed, "n": n, "window": W, "held": len(held), "tokens": len(toks),
       "new tokens": sum(not st for st, _ in toks), "common tokens": len(common), "train s": round(t1 - t0)}
for r in READS:
    out[r] = {"bits/word": mean([bits(ps[r]) for _, ps in toks]),
              "stored": mean([bits(ps[r]) for st, ps in toks if st]),
              "new": mean([bits(ps[r]) for st, ps in toks if not st]),
              "common": mean([bits(toks[k][1][r]) for k in common]),
              "common new": mean([bits(toks[k][1][r]) for k in common if not toks[k][0]]),
              "zero-mass": sum(ps[r] <= 0 for _, ps in toks), "s": round(secs[r])}
print(json.dumps(out), flush=True)
