"""Semantic generation across sentences (docs/SEMANTIC_GEN.md): given a story's
sentence, does the generated next sentence belong with it? TinyStories read story
by story, in order; a sentence that directly follows a kept sentence of its story
is learned with ``follows`` (NEXT, a same-level chunk context).

Cross coherence (gold for scoring only): content words (majority coarse NLTK tag
NN, VB or JJ), every pair of one prompt word and one generated word (different
words); PMI from within-sentence co-occurrence over the whole TinyStories file.

  real next   the story's actual next sentence (ceiling)
  random      a random held-out sentence (floor)
  share       (generated - random) / (real next - random)

The unconditioned generator ({} arm) gets the same prompts, so its share is the
floor up to sampling. Grammar: tag pattern attested (first 20000 corpus sentences),
novel, and within-sentence coherence share as in t3_sem.py.

Usage: t3_story.py <cfg json> <seed> <n> <tag> <arms>
  arms: JSON list of generate_tree keyword sets; "cond": true passes the prompt
  as ``after`` (with "after_level"), e.g. [{}, {"cond": true}, {"cond": true, "after_level": 2}]
"""
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

import nltk

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
arms = json.loads(sys.argv[5])
FILL, N_PROMPT, N_TEST_STORIES = "both+peer", 400, 1500

# stories as lists of sentences (as read_sentences splits them)
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

# training stream: kept sentences in story order, follows = the previous sentence of the story was kept
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
pairs = pairs[:N_PROMPT]
held = [s for st in test for s in st if keep(s)]

# gold tags (scoring only)
corpus = [s for s in all_sents if keep(s)]
gold_ct = defaultdict(Counter)
patterns = set()
for s in corpus[:20000]:
    tagged = nltk.pos_tag(s)
    patterns.add(tuple(t[:2] for _, t in tagged))
    for w, t in tagged:
        gold_ct[w][t[:2]] += 1
gold = {w: c.most_common(1)[0][0] for w, c in gold_ct.items()}
CONTENT = {"NN", "VB", "JJ"}
by_tag = defaultdict(Counter)
for s in corpus:
    for w in s:
        if gold.get(w) in CONTENT:
            by_tag[gold[w]][w] += 1

uni, pair, n_sent = Counter(), Counter(), 0
for s in all_sents:
    ws = sorted({w for w in s if gold.get(w) in CONTENT and w in vocab})
    n_sent += 1
    uni.update(ws)
    pair.update(combinations(ws, 2))


def pmi(a, b):
    k = (a, b) if a < b else (b, a)
    return math.log2((pair[k] + 0.5) * n_sent / ((uni[a] + 0.5) * (uni[b] + 0.5)))


content = lambda s: [w for w in s if gold.get(w) in CONTENT]


def cross(prompts, outs):
    vals = []
    for p, g in zip(prompts, outs):
        ps = [pmi(a, b) for a in content(p) for b in content(g) if a != b]
        if g is not None and ps:
            vals.append(sum(ps) / len(ps))
    return (sum(vals) / len(vals) if vals else None), len(vals)


def within(sents):
    vals = []
    for s in sents:
        ps = [pmi(s[i], s[j]) for i, j in combinations(range(len(s)), 2)
              if j - i >= 2 and s[i] != s[j] and gold.get(s[i]) in CONTENT and gold.get(s[j]) in CONTENT]
        if ps:
            vals.append(sum(ps) / len(ps))
    return sum(vals) / len(vals) if vals else None


def shuffled(sents, rng):
    return [[rng.choices(*zip(*by_tag[gold[w]].items()))[0] if gold.get(w) in CONTENT else w for w in s] for s in sents]


rng = random.Random(seed + 11)
P = [p for p, _ in pairs]
real_c, _ = cross(P, [q for _, q in pairs])
rand_c, _ = cross(P, [rng.choice(held) for _ in P])
w_real = within([q for _, q in pairs])
w_shuf = within(shuffled([q for _, q in pairs], rng))
print(json.dumps({"tag": tag, "seed": seed, "train": len(train), "follows": sum(f for _, f in train), "prompts": len(P),
                  "REAL next": round(real_c, 3), "RANDOM": round(rand_c, 3),
                  "within real": round(w_real, 3), "within shuffled": round(w_shuf, 3)}), flush=True)

L = Trellis3(seed=seed, **cfg)
t0 = time.time()
for s, fol in train:
    L.learn(s, follows=fol)
learn_s = time.time() - t0
L._quiet = True
ix = L.concept_index()
for arm in arms:
    arm = dict(arm)
    cond = arm.pop("cond", False)
    g_rng = random.Random(seed + 7)
    t1 = time.time()
    G = [L.generate_tree(g_rng, index=ix, fill_name=FILL, **({"after": p} if cond else {}), **{"choose": "leaf", **arm}) for p in P]
    ok = [(p, g) for p, g in zip(P, G) if g]
    c, cn = cross(*zip(*ok)) if ok else (None, 0)
    S = [g for _, g in ok]
    pat = [tuple(t[:2] for _, t in nltk.pos_tag(g)) in patterns for g in S]
    nov = [tuple(g) not in seen for g in S]
    w = within(S)
    ws = within(shuffled(S, random.Random(seed + 13)))
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    print(json.dumps({"tag": tag, "seed": seed, "n": n, "arm": {"cond": cond, **arm}, "learn s": round(learn_s),
                      "gen s": round(time.time() - t1), "generated": len(S),
                      "cross": None if c is None else round(c, 3), "cross scored": cn,
                      "cross share": None if c is None else round((c - rand_c) / (real_c - rand_c), 3),
                      "tag pattern attested": m(pat), "novel": m(nov),
                      "within share": None if w is None or ws is None else round((w - ws) / (w_real - w_shuf), 3),
                      "examples": [" ".join(p) + "  ->  " + " ".join(g) for p, g in ok[:12]]}), flush=True)
