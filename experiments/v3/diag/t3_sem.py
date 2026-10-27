"""Semantic generation (docs/SEMANTIC_GEN.md): do generated sentences hold
together in meaning, beyond grammar? TinyStories, one learner, several generators.

Coherence (gold for scoring only): content words (majority coarse NLTK tag NN, VB
or JJ) of a sentence, every pair two or more positions apart; a pair's PMI is read
from within-sentence co-occurrence over the whole TinyStories file (every sentence,
words of the 250-word vocabulary). A sentence scores the mean PMI of its pairs.

  shuffled    the same sentences with every content word replaced by a random word
              of its tag (by frequency): grammar kept, meaning destroyed
  share       (sentences - shuffled) / (real - real shuffled): 0 = no more
              coherent than chance words of the right category, 1 = as real text

Grammar as in t3_use.py: trigrams attested, novel, attested + novel; and, independent
of any word window, tag pattern attested (the coarse tag sequence of the sentence
occurs in the first 20000 corpus sentences).

Usage: t3_sem.py <cfg json> <seed> <n> <tag> <arms>
  arms: a JSON list of generate_tree keyword sets, e.g. [{}, {"sibling": 1},
  {"choose": "match"}, {"choose": "backoff", "beta": 2}]; choose defaults to "leaf".
  ({} is the current generator; "sibling": k conditions a right part on its frozen
  left part, "sibling_beta" sharpens that.)
"""
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

import nltk

sys.path.insert(0, "src")
from trellis2.stories import heldout_path, load_corpus, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
arms = json.loads(sys.argv[5])
FILL, N_GEN = "both+peer", 600

corpus = load_corpus(vocab_size=250)
vocab = {w for s in corpus for w in s}
random.Random(seed).shuffle(corpus)
held = json.load(open(heldout_path(13), encoding="utf-8"))
held_set = {tuple(s) for s in held}
train = [s for s in corpus if tuple(s) not in held_set][:n]
seen = {tuple(s) for s in train}
trigrams = set()
for s in corpus:
    p = ["<s>", "<s>"] + list(s) + ["</s>"]
    trigrams.update(zip(p, p[1:], p[2:]))

# gold tags (scoring only)
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

# PMI oracle: within-sentence co-occurrence over the whole file
uni, pair, n_sent = Counter(), Counter(), 0
for s in read_sentences():
    ws = sorted({w for w in s if gold.get(w) in CONTENT and w in vocab})
    n_sent += 1
    uni.update(ws)
    pair.update(combinations(ws, 2))


def pmi(a, b):
    k = (a, b) if a < b else (b, a)
    return math.log2((pair[k] + 0.5) * n_sent / ((uni[a] + 0.5) * (uni[b] + 0.5)))


def coherence(sents):
    vals = []
    for s in sents:
        ps = [pmi(s[i], s[j]) for i, j in combinations(range(len(s)), 2)
              if j - i >= 2 and s[i] != s[j] and gold.get(s[i]) in CONTENT and gold.get(s[j]) in CONTENT]
        if ps:
            vals.append(sum(ps) / len(ps))
    return (sum(vals) / len(vals) if vals else None), len(vals)


def shuffled(sents, rng):
    out = []
    for s in sents:
        t = []
        for w in s:
            g = gold.get(w)
            if g in CONTENT:
                ws, cs = zip(*by_tag[g].items())
                t.append(rng.choices(ws, cs)[0])
            else:
                t.append(w)
        out.append(t)
    return out


def grammar(S):
    tri = [all(t in trigrams for t in zip(p, p[1:], p[2:])) for p in (["<s>", "<s>"] + list(g) + ["</s>"] for g in S)]
    nov = [tuple(g) not in seen for g in S]
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    pat = [tuple(t[:2] for _, t in nltk.pos_tag(g)) in patterns for g in S]
    return {"trigrams attested": m(tri), "novel": m(nov), "attested + novel": m([a and b for a, b in zip(tri, nov)]),
            "tag pattern attested": m(pat), "tag pattern attested + novel": m([a and b for a, b in zip(pat, nov)])}


rng = random.Random(seed + 11)
real = [list(s) for s in held[:300]]
r_coh, r_n = coherence(real)
r_shuf, _ = coherence(shuffled(real, rng))
print(json.dumps({"tag": tag, "seed": seed, "REAL": {"coherence": round(r_coh, 3), "shuffled": round(r_shuf, 3), "sentences scored": r_n}}), flush=True)

L = Trellis3(seed=seed, **cfg)
t0 = time.time()
for s in train:
    L.learn(s)
learn_s = time.time() - t0
L._quiet = True
ix = L.concept_index()
for arm in arms:
    g_rng = random.Random(seed + 7)
    t1 = time.time()
    S = [g for g in (L.generate_tree(g_rng, index=ix, fill_name=FILL, **{"choose": "leaf", **arm}) for _ in range(N_GEN)) if g]
    coh, cn = coherence(S)
    nov_S = [g for g in S if tuple(g) not in seen]
    ncoh, ncn = coherence(nov_S)
    nshuf, _ = coherence(shuffled(nov_S, random.Random(seed + 13)))
    shuf, _ = coherence(shuffled(S, random.Random(seed + 11)))
    share = (coh - shuf) / (r_coh - r_shuf) if coh is not None and shuf is not None else None
    print(json.dumps({"tag": tag, "seed": seed, "n": n, "arm": arm, "learn s": round(learn_s), "gen s": round(time.time() - t1),
                      "generated": len(S), **grammar(S),
                      "coherence": None if coh is None else round(coh, 3), "shuffled": None if shuf is None else round(shuf, 3),
                      "share": None if share is None else round(share, 3), "sentences scored": cn,
                      "novel coherence": None if ncoh is None else round(ncoh, 3),
                      "novel share": None if ncoh is None or nshuf is None else round((ncoh - nshuf) / (r_coh - r_shuf), 3),
                      "novel scored": ncn,
                      "examples": [" ".join(g) for g in S[:10]]}), flush=True)
