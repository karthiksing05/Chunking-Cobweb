"""Representations on TinyStories (docs/TEMPORAL_INTERDEP.md): convergence,
currency and generation on real children's English.

Corpus: TinyStories sentences of 3-8 words that use only the ``vocab`` most
frequent words (trellis2.stories), shuffled by seed; the first ``n`` are the
stream. Gold for scoring only: each word's majority coarse part-of-speech tag
(NLTK's tagger on the sentences, first two letters of the Penn tag; PRP$ and
WP$ map to PR/WP, TO counts as IN).

CONVERGENCE at checkpoints: for each word type seen so far, is its nearest
other type by learned name (sum of min weights) of the same tag; by how often
the word was seen (1, 2-3, 4-7, 8-15, 16+).
TEMPORAL at the end: share of a sample of stored records whose band-level
concepts differ from where their current description sorts (context, content).
GENERATION at the end (300 samples, generate_tree): distinct, novel (not a
training sentence), trigrams attested (every trigram, padded with <s> </s>,
occurs in the whole filtered corpus), tag pattern attested (its coarse tag
sequence is one of the corpus's), and attested + novel for each.

Usage: t3_stories.py <cfg json> <seed> <n> [vocab] [tag] [fill]
"""
import json
import os as _os
LEVEL = int(_os.environ.get("GEN_LEVEL", "0"))   # generation draws fillers this many concepts above the leaf
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import nltk  # noqa: E402
from trellis2.stories import read_sentences, simple_sentences  # noqa: E402
from trellis2.trellis3 import Elem, Trellis3  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
vocab = int(sys.argv[4]) if len(sys.argv) > 4 else 250
tag = sys.argv[5] if len(sys.argv) > 5 else ""
fill = sys.argv[6] if len(sys.argv) > 6 else "both"
fill = {"none": False, "name": True}.get(fill, fill)

corpus = simple_sentences(read_sentences(), vocab_size=vocab)
random.Random(seed).shuffle(corpus)
train = corpus[:n]


def coarse(t):
    if t in ("PRP$", "WP$"):
        return t[:2]
    if t == "TO":
        return "IN"
    return t[:2]


tags_of = {}
gold_ct = defaultdict(Counter)
for s in corpus[:max(n, 20000)]:
    seq = tuple(coarse(t) for _, t in nltk.pos_tag(s))
    tags_of[tuple(s)] = seq
    for w, t in zip(s, seq):
        gold_ct[w][t] += 1
gold = {w: c.most_common(1)[0][0] for w, c in gold_ct.items()}
patterns = set(tags_of.values())
trigrams = set()
for s in corpus:
    p = ["<s>", "<s>"] + list(s) + ["</s>"]
    trigrams.update(zip(p, p[1:], p[2:]))


def as_bag(v):
    if isinstance(v, dict):
        t = sum(v.values()) or 1.0
        return {u: p / t for u, p in v.items()}
    if isinstance(v, list):
        real = [u for u in v if u and u != "-"]
        return {u: real.count(u) / len(real) for u in set(real)} if real else {}
    return {}


def overlap(a, b):
    return sum(min(p, b.get(u, 0.0)) for u, p in a.items())


def names(L, words):
    L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
    return {w: as_bag(L._ctx_name(Elem((0, 1), token=w)) or {}) for w in words}


def band_ids(L, path):
    total, out, k = max(path[0].count, 1e-9), [], 0
    for nd in path[1:]:
        b = -math.log2(max(nd.count, 1e-9) / total)
        while k < len(L.band_bits) and b >= L.band_bits[k]:
            out.append(nd.id)
            k += 1
    return out


def misplaced(L, tree, leaf, x):
    st = tree.ancestors(leaf)[::-1][:-1]
    now = tree.categorize(x)[:-1]
    a, b = band_ids(L, st) if st else [], band_ids(L, now) if now else []
    m = max(len(a), len(b))
    return 1 - (sum(p == q for p, q in zip(a, b)) / m if m else 1.0)


L = Trellis3(seed=seed, **cfg)
seen = Counter()
ctxv = defaultdict(Counter)        # baseline: raw counts of the words either side


def cosine(a, b):
    num = sum(v * b.get(u, 0) for u, v in a.items())
    den = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return num / den if den else 0.0
checks = sorted({50, 100, 200, 400, 800, 1600, n} & set(range(1, n + 1)))
curve = {}
t0 = time.time()
order = ["1", "2-3", "4-7", "8-15", "16+"]
for k, s in enumerate(train, 1):
    L.learn(s)
    seen.update(s)
    p = ["<s>"] + list(s) + ["</s>"]
    for i in range(1, len(p) - 1):
        ctxv[p[i]]["L:" + p[i - 1]] += 1
        ctxv[p[i]]["R:" + p[i + 1]] += 1
    if k in checks:
        words = [w for w in seen if w in gold]
        nm = names(L, words)
        hit, by = [], defaultdict(list)
        for w in words:
            best = max((x for x in words if x != w), key=lambda x: overlap(nm[w], nm[x]), default=None)
            ok = best is not None and gold[best] == gold[w] and overlap(nm[w], nm[best]) > 0
            hit.append(ok)
            c = seen[w]
            by["1" if c == 1 else "2-3" if c <= 3 else "4-7" if c <= 7 else "8-15" if c <= 15 else "16+"].append(ok)
        base = []
        for w in words:
            best = max((x for x in words if x != w), key=lambda x: cosine(ctxv[w], ctxv[x]), default=None)
            base.append(best is not None and gold[best] == gold[w])
        curve[k] = {"words": len(words), "category accuracy": round(sum(hit) / max(len(hit), 1), 3),
                    "baseline (raw window counts)": round(sum(base) / max(len(base), 1), 3),
                    "by count": {b: (round(sum(by[b]) / len(by[b]), 3), len(by[b])) for b in order if by[b]},
                    "s": round(time.time() - t0)}
        print(json.dumps({"checkpoint": k, **curve[k]}), flush=True)
learn_s = time.time() - t0

mx, mc = [], []
for r in random.Random(seed).sample(L.records, min(1500, len(L.records))):
    if r.leaf is not None:
        mx.append(misplaced(L, L.ctx, r.leaf, r.ctx))
    if r.token is None and r.cleaf is not None:
        mc.append(misplaced(L, L.cnt, r.cleaf, r.cnt))

rng = random.Random(seed)
ix = L.concept_index()
S = [g for g in (L.generate_tree(rng, index=ix, level=LEVEL, choose="leaf", fill_name=fill) for _ in range(300)) if g]
seen_s = {tuple(s) for s in train}
tri_ok = []
pat_ok = []
for g in S:
    p = ["<s>", "<s>"] + list(g) + ["</s>"]
    tri_ok.append(all(t in trigrams for t in zip(p, p[1:], p[2:])))
    pat_ok.append(tuple(coarse(t) for _, t in nltk.pos_tag(g)) in patterns)
nov = [tuple(g) not in seen_s for g in S]
m = lambda v: round(sum(v) / len(v), 3) if v else None
out = {"tag": tag, "seed": seed, "n": n, "vocab": vocab, "fill": fill, "cfg": cfg, "learn s": round(learn_s),
       "types": len(seen), "CONVERGENCE": curve,
       "TEMPORAL": {"misplaced at bands context / content": [m(mx), m(mc)]},
       "GENERATION": {"generated": len(S), "distinct": m([1] * len(set(map(tuple, S)))) and round(len(set(map(tuple, S))) / max(len(S), 1), 3),
                      "novel": m(nov), "trigrams attested": m(tri_ok), "tag pattern attested": m(pat_ok),
                      "trigrams attested + novel": m([a and b for a, b in zip(tri_ok, nov)]),
                      "tag pattern attested + novel": m([a and b for a, b in zip(pat_ok, nov)]),
                      "examples": [" ".join(g) for g in S[:8]]}}
print(json.dumps(out), flush=True)
