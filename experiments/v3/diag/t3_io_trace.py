"""Traces of inside-outside generation, for the explorer: for each prompt (a held-out story
sentence), the probability of each length and, position by position, the top symbols and the
words' probabilities under the inside-outside marginal, the window prediction and their product.

Usage: t3_io_trace.py <cfg json> <seed> <n> <chart json> <out json>
"""
import json
import random
import re
import sys
from collections import Counter

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.chart_gen import ChartGen  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, chart, out_path = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4]), sys.argv[5]
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
test, train_st = stories[:1500], stories[1500:]
train = []
for st in train_st:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, i > 0 and keep(st[i - 1])))
    if len(train) >= n:
        break
train = train[:n]
seen = {tuple(s) for s, _ in train}
pairs = [(st[i - 1], st[i]) for st in test for i in range(1, len(st)) if keep(st[i - 1]) and keep(st[i])]
random.Random(seed + 1).shuffle(pairs)
L = Trellis3(seed=seed, **cfg)
for s, fol in train:
    L.learn(s, follows=fol)
L._quiet = True
CG = ChartGen(L, **chart)
rng = random.Random(seed + 7)
cases = []
for p, real in pairs[:24]:
    info = {}
    g = CG.generate(["*"], rng, prev=p, info=info)
    if g:
        cases.append({"prompt": " ".join(p), "real next": " ".join(real), "generated": " ".join(g), "new": tuple(g) not in seen,
                      "lengths": info["lengths"], "steps": info["steps"]})
json.dump({"seed": seed, "n": n, "chart": chart, "stats": CG.stats, "cases": cases}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", out_path, len(cases))
