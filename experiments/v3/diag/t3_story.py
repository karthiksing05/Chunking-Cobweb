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

Each prompt gets DRAWS generations and DRAWS random sentences. The unconditioned
generator ({} arm) gets the same prompts, so its share is the
floor up to sampling. Grammar: tag pattern attested (first 20000 corpus sentences),
novel, and within-sentence coherence share as in t3_sem.py.

Usage: t3_story.py <cfg json> <seed> <n> <tag> <arms> [adjacent|story]
  arms: JSON list of generate_tree keyword sets; "cond": true passes the prompt
  as ``after`` (with "after_level"), e.g. [{}, {"cond": true}, {"cond": true, "after_level": 2}]
"""
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

import nltk

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.chart_gen import ChartGen  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
arms = json.loads(sys.argv[5])
FOLLOW = sys.argv[6] if len(sys.argv) > 6 else "adjacent"
FILL, N_PROMPT, N_TEST_STORIES, DRAWS = "both+peer", 400, 1500, 4

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
# training stream: kept sentences in story order; follows = the previous sentence of the
# story was kept (FOLLOW "adjacent") or any earlier sentence of the story was (FOLLOW "story":
# the nearest earlier kept sentence then counts as the previous experience)
train = []
for k, st in enumerate(train_st):
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, (i > 0 and keep(st[i - 1])) if FOLLOW == "adjacent" else any(keep(t) for t in st[:i])))
    if len(train) >= n:
        break
train = train[:n]
seen = {tuple(s) for s, _ in train}
adj = lambda sts: [(st[i - 1], st[i]) for st in sts for i in range(1, len(st)) if keep(st[i - 1]) and keep(st[i])]
val = adj(train_st[k + 1:k + 400])[:200]   # pairs of stories the stream never reached: for fitting the topic weight
pairs = adj(test)
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
rand_c, _ = cross([p for p in P for _ in range(DRAWS)], [rng.choice(held) for _ in P for _ in range(DRAWS)])
w_real = within([q for _, q in pairs])
w_shuf = within(shuffled([q for _, q in pairs], rng))
# references without the learner: what memory of the training stream could give at most
#   retrieve  the stored next sentence whose previous sentence shares the most idf-weighted
#             words with the prompt (no gold; ties broken at random)
#   oracle    the training sentence with the highest gold cross PMI with the prompt
stored = [(train[i - 1][0], s) for i, (s, fol) in enumerate(train) if fol and i > 0]
df = Counter(w for s, _ in train for w in set(s))
idf = {w: math.log(len(train) / c) for w, c in df.items()}
ret = []
for p in P:
    ps = set(p)
    sc = [sum(idf.get(w, 0.0) for w in ps & set(a)) + 1e-6 * rng.random() for a, _ in stored]
    ret.append(stored[max(range(len(stored)), key=sc.__getitem__)][1] if stored else [])
ret_c, _ = cross(P, ret)
cands = [s for s, _ in train]
orc = [max(cands, key=lambda g: cross([p], [g])[0] if cross([p], [g])[0] is not None else -1e9) for p in P[:100]]
orc_c, _ = cross(P[:100], orc)
orc_real, _ = cross(P[:100], [q for _, q in pairs[:100]])
orc_rand, _ = cross(P[:100], [rng.choice(held) for _ in P[:100]])
print(json.dumps({"tag": tag, "seed": seed, "train": len(train), "follows": sum(f for _, f in train), "prompts": len(P),
                  "REAL next": round(real_c, 3), "RANDOM": round(rand_c, 3),
                  "within real": round(w_real, 3), "within shuffled": round(w_shuf, 3),
                  "retrieve share": round((ret_c - rand_c) / (real_c - rand_c), 3),
                  "oracle share (100 prompts)": round((orc_c - orc_rand) / (orc_real - orc_rand), 3)}), flush=True)
if len(sys.argv) > 7 and sys.argv[7] == "refs":
    sys.exit()


# ---------------------------------------------------------------------------------------
# STORY_BASE=<JSON list of arms>: reference generators that never see the learner, on the
# same stream, prompts and measures (then exit).
#   {"ngram": k}                 maximum-likelihood k-gram sampler with sentence boundaries
#   {"ngram": k, "topic": b}     each word also weighted by exp(b * lift), lift = mean over the
#                                prompt's words u of log P(w in next | u in previous) / P(w in next)
#   {"swap": "random"}           a stored sentence with one word replaced by another word stored
#                                between the same two neighbours
#   {"swap": "retrieve"}         the same, on the stored sentence that followed the stored
#                                sentence most like the prompt (idf overlap)
#     "n": k                     k swaps in a row
#     "frame": "wide"            the replacement stood between the same two neighbours on each side
#     "frame": "bigram"          the replacement stood after the left neighbour and before the right
#                                one, not necessarily between them (both bigrams stored)
#     "unit": "span"             a span of 1-3 words replaced by a stored span of 1-3 words that
#                                stood between the same two neighbours
# STORY_DIAG=<dump dir>: for each judged dump of this seed there, how the judged-grammatical
# share splits by whether every bigram / trigram of the generation stands in the stream.
def grams(s, k):
    t = ["<s>"] + list(s) + ["</s>"]
    return [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]


have = {k: {g for s, _ in train for g in grams(s, k)} for k in (2, 3)}
attested = lambda s, k: all(g in have[k] for g in grams(s, k))
# ---------------------------------------------------------------------------------------
# STORY_LICENCE=1: the generalization step alone, with no generation and no judge. Every pair
# of stream words the stream never showed side by side is scored by each licence; a pair is
# right when the whole corpus has it (twice or more). Reported: average precision (the pairs
# ranked by the licence) and the precision reached at fixed recall. Licences without the
# learner here; the learner's own (Cobweb's prediction of a word from one neighbour) after
# learning, when a recipe is given.
LIC, LIC_SET = os.environ.get("STORY_LICENCE"), None
if LIC:
    _sents = [list(s) for s, _ in train]
    pc = Counter(g for s in _sents for g in zip(s, s[1:]))
    wc_ = Counter(w for s in _sents for w in s)
    Lf, Rt = defaultdict(set), defaultdict(set)
    for a_, b_ in pc:
        Lf[b_].add(a_)
        Rt[a_].add(b_)
    full = Counter(g for s in all_sents if keep(s) for g in zip(s, s[1:]))
    words_ = sorted(wc_)
    cand = [(a, b) for a in words_ for b in words_ if (a, b) not in pc]
    good = {x: full[x] >= 2 for x in cand}
    n_pos = sum(good.values())
    jit = random.Random(seed + 31)
    tie = {x: jit.random() for x in cand}

    def ap(score):
        order = sorted(cand, key=lambda x: (score(*x), tie[x]), reverse=True)
        hit, tot, at = 0, 0.0, {}
        for i, x in enumerate(order, 1):
            if good[x]:
                hit += 1
                tot += hit / i
                for r in (0.1, 0.25, 0.5):
                    if str(r) not in at and hit >= r * n_pos:
                        at[str(r)] = round(hit / i, 3)
        return {"average precision": round(tot / max(n_pos, 1), 3), "precision at recall": at}

    def cosv(u, v):
        d = sum(x * v.get(k, 0) for k, x in u.items())
        return d / math.sqrt(sum(x * x for x in u.values()) * sum(x * x for x in v.values())) if d else 0.0
    rvec = {a: {b: pc[(a, b)] for b in Rt[a]} for a in words_}
    lvec = {b: {a: pc[(a, b)] for a in Lf[b]} for b in words_}
    simR = {a: {a2: cosv(rvec[a], rvec[a2]) for a2 in words_ if a2 != a} for a in words_}
    simL = {b: {b2: cosv(lvec[b], lvec[b2]) for b2 in words_ if b2 != b} for b in words_}
    zR = {a: sum(simR[a].values()) for a in words_}
    zL = {b: sum(simL[b].values()) for b in words_}

    def lic_support(a, b):
        return sum(1 for a2 in Lf[b] for b2 in Rt[a] if (a2, b2) in pc)

    def lic_density(a, b):
        k = len(Lf[b]) * len(Rt[a])
        return lic_support(a, b) / k if k else 0.0

    def lic_similar(a, b):
        # words like a (by what follows them) vote with how often b followed them, and
        # words like b (by what precedes them) with how often they followed a
        pr = sum(v * pc[(a2, b)] / wc_[a2] for a2, v in simR[a].items() if v and (a2, b) in pc) / zR[a] if zR[a] else 0.0
        pl = sum(v * pc[(a, b2)] / wc_[b2] for b2, v in simL[b].items() if v and (a, b2) in pc) / zL[b] if zL[b] else 0.0
        return math.sqrt(pr * pl)
    print(json.dumps({"licence": "setup", "seed": seed, "n": n, "stream words": len(words_), "unstored pairs": len(cand),
                      "in the corpus": n_pos, "base rate": round(n_pos / len(cand), 4)}), flush=True)
    for name, fn in (("frequency of the two words", lambda a, b: wc_[a] * wc_[b]), ("analogy, count", lic_support),
                     ("analogy, density", lic_density), ("similar words", lic_similar)):
        print(json.dumps({"licence": name, "seed": seed, "n": n, **ap(fn)}), flush=True)
    # a Cobweb tree over the stream's adjacent word pairs, each an instance (left word, right
    # word), learned in stream order. A concept holding (a2, b2), (a, b2) and (a2, b) gives the
    # unstored (a, b) the product of its two shares: the analogy, read off a concept.
    if LIC.startswith("words"):
        # a Cobweb tree over word types, each described by two bags: the shares of the words
        # stored just before it and just after it. The licence for an unstored (a, b) reads b's
        # share among what followed the words grouped with a, and a's share among what preceded
        # the words grouped with b. "tok": a type weighs as its count; "exact": exact_alone on.
        try:
            from trellis2.cu_cobweb import CUCobwebTree as _Tree
        except Exception:
            from trellis2.incremental_cobweb import IncrementalCobwebTree as _Tree
        t2 = time.time()
        WT = _Tree(("Lb", "Rb"), seed=seed)
        if "exact" in LIC:
            WT.exact_alone = True
        first, leaf_of = [], {}
        for s_ in _sents:
            for w in s_:
                if w not in leaf_of:
                    leaf_of[w] = None
                    first.append(w)
        if "set" in LIC:                                      # each distinct neighbour counts once
            share = lambda d: {k: 1.0 / len(d) for k in d} if d else {"<none>": 1.0}
        else:
            share = lambda d: {k: v / sum(d.values()) for k, v in d.items()} if d else {"<none>": 1.0}
        for w in first:
            leaf_of[w] = WT.ifit({"Lb": share(lvec[w]), "Rb": share(rvec[w])}, float(wc_[w]) if "tok" in LIC else 1.0)
        paths = {}
        for w, lf in leaf_of.items():
            path, nd = [], lf
            while nd is not None:
                path.append((dict(nd.av.get("Lb", {}).items()), dict(nd.av.get("Rb", {}).items()), nd.count))
                nd = nd.parent
            paths[w] = path                                   # leaf ... root
        # the same paths by types: a concept's members are the words under it; it has seen v
        # on a side as many times as it has members with v stored on that side
        ids, members = {}, defaultdict(set)
        for w, lf in leaf_of.items():
            nd, chain = lf, []
            while nd is not None:
                chain.append(nd.id)
                members[nd.id].add(w)
                nd = nd.parent
            ids[w] = chain
        tcount = {}

        def types(nid, attr, v):
            k = (nid, attr, v)
            if k not in tcount:
                tcount[k] = sum(1 for m_ in members[nid] if ((m_, v) if attr == "R" else (v, m_)) in pc)
            return tcount[k]

        def read_types(w, attr, v, how):
            chain = ids[w]
            if how == "sum":                                  # every concept around w votes with its members' share
                return sum(types(c, attr, v) / len(members[c]) for c in chain[1:])
            p_ = None
            for c in reversed(chain):
                x, m_ = types(c, attr, v), len(members[c])
                p_ = (x + 1.0) / (m_ + len(words_)) if p_ is None else (x + p_) / (m_ + 1.0)
            return p_
        depth_ = sum(len(v) for v in paths.values()) / len(paths)

        def read(w, attr, v, how):
            path = paths[w]
            if how == "nearest":                              # the smallest concept around w that has seen v there
                for lv, rv, cnt in path[1:-1]:
                    x = (lv if attr == "L" else rv).get(v, 0.0)
                    if x > 0:
                        return x / cnt
                return 0.0
            p_ = None
            for lv, rv, cnt in reversed(path):                 # root down: each level backed off to the one above
                x = (lv if attr == "L" else rv).get(v, 0.0)
                p_ = (x + 1.0) / (cnt + len(words_)) if p_ is None else (x + p_) / (cnt + 1.0)
            return p_
        print(json.dumps({"licence": "word tree", "mode": LIC, "seed": seed, "n": n, "words": len(first),
                          "mean path": round(depth_, 1), "learn s": round(time.time() - t2, 1)}), flush=True)
        for how in ("nearest", "back-off"):
            print(json.dumps({"licence": f"word tree, {how}", "seed": seed, "n": n,
                              **ap(lambda a, b: math.sqrt(read(a, "R", b, how) * read(b, "L", a, how)))}), flush=True)
            print(json.dumps({"licence": f"word tree, {how}, left word's group only", "seed": seed, "n": n,
                              **ap(lambda a, b: read(a, "R", b, how))}), flush=True)
        if "sides" in LIC:
            # two trees: words by what follows them (asked what may follow a) and words by what
            # precedes them (asked what may precede b); each read by types, summed over a path
            side = {}
            for attr, vec in (("R", rvec), ("L", lvec)):
                T_ = _Tree((attr + "b",), seed=seed)
                T_.exact_alone = "exact" in LIC
                lf_ = {w: T_.ifit({attr + "b": share(vec[w])}, float(wc_[w]) if "tok" in LIC else 1.0) for w in first}
                ch_, mem_ = {}, defaultdict(set)
                for w, nd in lf_.items():
                    chain = []
                    while nd is not None:
                        chain.append(nd.id)
                        mem_[nd.id].add(w)
                        nd = nd.parent
                    ch_[w] = chain
                side[attr] = (ch_, mem_, {})

            def read_side(w, attr, v):
                ch_, mem_, memo = side[attr]
                tot = 0.0
                for c in ch_[w][1:]:
                    k = (c, v)
                    if k not in memo:
                        memo[k] = sum(1 for m_ in mem_[c] if ((m_, v) if attr == "R" else (v, m_)) in pc) / len(mem_[c])
                    tot += memo[k]
                return tot
            print(json.dumps({"licence": "word trees by side", "mode": LIC, "seed": seed, "n": n,
                              "mean path": {a_: round(sum(len(v) for v in side[a_][0].values()) / len(first), 1) for a_ in side}}), flush=True)
            print(json.dumps({"licence": "word trees by side, by types, sum, added", "seed": seed, "n": n,
                              **ap(lambda a, b: read_side(a, "R", b) + read_side(b, "L", a))}), flush=True)
            print(json.dumps({"licence": "word trees by side, by types, sum", "seed": seed, "n": n,
                              **ap(lambda a, b: math.sqrt(read_side(a, "R", b) * read_side(b, "L", a)))}), flush=True)
        for how in ("sum", "back-off"):
            print(json.dumps({"licence": f"word tree by types, {how}", "seed": seed, "n": n,
                              **ap(lambda a, b: math.sqrt(read_types(a, "R", b, how) * read_types(b, "L", a, how)))}), flush=True)
            print(json.dumps({"licence": f"word tree by types, {how}, added", "seed": seed, "n": n,
                              **ap(lambda a, b: read_types(a, "R", b, how) + read_types(b, "L", a, how))}), flush=True)
    elif LIC != "1":
        try:
            from trellis2.cu_cobweb import CUCobwebTree as _Tree
        except Exception:
            from trellis2.incremental_cobweb import IncrementalCobwebTree as _Tree
        t2 = time.time()
        PT = _Tree(("L", "R"), seed=seed)
        if LIC == "tree-exact":
            PT.exact_alone = True
        for s_ in _sents:
            for a_, b_ in zip(s_, s_[1:]):
                PT.ifit({"L": a_, "R": b_}, 1.0)
        acc, deep, stack, n_nodes = defaultdict(lambda: defaultdict(float)), 0, [(PT.root, 0)], 0
        while stack:
            nd, d = stack.pop()
            ch = list(nd.children)
            n_nodes += 1
            deep = max(deep, d)
            stack.extend((c, d + 1) for c in ch)
            if d == 0:
                continue
            lv, rv, cnt = dict(nd.av.get("L", {}).items()), dict(nd.av.get("R", {}).items()), nd.count
            keys = ["all levels"] + [f"depth {k}" for k in (1, 2, 3, 4, 6, 8) if d == k or (not ch and d < k)]
            if len(lv) * len(rv) == 1:
                continue                                   # one stored pair: nothing unstored in it
            for a_, x in lv.items():
                for b_, y in rv.items():
                    if (a_, b_) not in pc:
                        v = x * y / cnt
                        for k in keys:
                            acc[k][(a_, b_)] += v
        print(json.dumps({"licence": "pair tree", "seed": seed, "n": n, "pairs learned": int(PT.root.count), "concepts": n_nodes,
                          "deepest": deep, "exact_alone": LIC == "tree-exact", "learn s": round(time.time() - t2, 1)}), flush=True)
        for k in sorted(acc):
            d_ = acc[k]
            print(json.dumps({"licence": f"pair tree, {k}", "seed": seed, "n": n, **ap(lambda a, b: d_.get((a, b), 0.0))}), flush=True)
    LIC_SET = (cand, ap, words_)
    if cfg == {}:
        sys.exit()
BASE, DIAG = os.environ.get("STORY_BASE"), os.environ.get("STORY_DIAG")
if BASE or DIAG:
    sents = [list(s) for s, _ in train]

    def table(k):
        tab = defaultdict(Counter)
        for s in sents:
            t = ["<s>"] * (k - 1) + s + ["</s>"]
            for i in range(k - 1, len(t)):
                tab[tuple(t[i - k + 1:i])][t[i]] += 1
        return tab

    def grams(s, k):
        t = ["<s>"] + list(s) + ["</s>"]
        return [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]
    have = {k: {g for s in sents for g in grams(s, k)} for k in (2, 3)}
    attested = lambda s, k: all(g in have[k] for g in grams(s, k))
    # licence by analogy: an unstored pair (a, b) is supported by each stored pair (a2, b2)
    # with (a, b2) and (a2, b) stored too: a and a2 share a right neighbour, b and b2 a left
    # neighbour, and a2 b2 stood together (a rectangle of stored pairs missing one corner)
    pairc = Counter(g for s in sents for g in grams(s, 2))
    lefts, rights = defaultdict(set), defaultdict(set)
    for a_, b_ in pairc:
        lefts[b_].add(a_)
        rights[a_].add(b_)

    def support(a, b):
        return sum(1 for a2 in lefts[b] for b2 in rights[a] if (a2, b2) in pairc)

    if DIAG and os.environ.get("STORY_PAIRS"):
        # judged by the whole corpus: of the pairs the stream never showed, which stand in the
        # corpus (twice or more), by their analogy support
        full = Counter(g for s in all_sents if keep(s) for g in grams(s, 2))
        words_ = sorted({w for s in sents for w in s})
        sup, good, tok = {}, {}, {}
        for a in words_ + ["<s>"]:
            for b in words_ + ["</s>"]:
                if (a, b) in pairc or (a, b) == ("<s>", "</s>"):
                    continue
                sup[(a, b)] = support(a, b)
                good[(a, b)] = full[(a, b)] >= 2
                tok[(a, b)] = full[(a, b)]
        n_good, n_tok = sum(good.values()), sum(tok.values())
        out = {"diag": "pair licence", "seed": seed, "n": n, "unstored pairs": len(sup),
               "in the corpus": round(n_good / len(sup), 4), "by support at least": {}}
        for k in (1, 2, 4, 8, 16, 32, 64):
            sel = [x for x, v in sup.items() if v >= k]
            out["by support at least"][str(k)] = {
                "pairs": len(sel), "in the corpus": round(sum(good[x] for x in sel) / max(len(sel), 1), 3),
                "share of the corpus's unstored pairs": round(sum(good[x] for x in sel) / max(n_good, 1), 3),
                "share of their occurrences": round(sum(tok[x] for x in sel) / max(n_tok, 1), 3)}
        print(json.dumps(out), flush=True)
        ref = random.Random(seed + 21).sample(held, min(300, len(held)))
        cov = {}
        for k in (1, 2, 4, 8):
            cov[str(k)] = round(sum(all(g in pairc or support(*g) >= k for g in grams(s, 2)) for s in ref) / len(ref), 3)
        print(json.dumps({"diag": "held-out real sentences, every pair stored or supported at least", "seed": seed, **cov}), flush=True)

    if DIAG:
        import glob
        for f in sorted(glob.glob(os.path.join(DIAG, f"*_s{seed}_n{n}_arm*.jsonl"))):
            sc = f[:-len(".jsonl")] + ".scores.json"
            if not os.path.exists(sc):
                continue
            rows = [json.loads(l) for l in open(f, encoding="utf-8")][1:]
            ps = json.load(open(sc, encoding="utf-8"))
            out = {"diag": os.path.basename(f), "seed": seed, "generated": len(rows)}
            for k in (2, 3):
                for nov in (True, False):
                    for att in (True, False):
                        sel = [p > 0.5 for r, p in zip(rows, ps) if r["novel"] == nov and attested(r["gen"].split(), k) == att]
                        out[f"{k}-grams {'all' if att else 'not all'} stored, {'new' if nov else 'stored sentence'}"] = \
                            {"share of generations": round(len(sel) / max(len(rows), 1), 3), "judged grammatical": round(sum(sel) / max(len(sel), 1), 3)}
            print(json.dumps(out), flush=True)
        ref = random.Random(seed + 21).sample(held, min(300, len(held)))
        print(json.dumps({"diag": "held-out real sentences", "seed": seed,
                          "every bigram stored": round(sum(attested(s, 2) for s in ref) / len(ref), 3),
                          "every trigram stored": round(sum(attested(s, 3) for s in ref) / len(ref), 3),
                          "stored whole": round(sum(tuple(s) in seen for s in ref) / len(ref), 3)}), flush=True)
    if not BASE:
        sys.exit()

    DUMP = os.environ.get("STORY_DUMP")
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
        ref = random.Random(seed + 21).sample(held, min(300, len(held)))
        with open(os.path.join(DUMP, f"{tag}_s{seed}_n{n}_refs.jsonl"), "w", encoding="utf-8") as fh:
            prng = random.Random(seed + 23)
            for s, t in zip(ref, shuffled(ref, random.Random(seed + 22))):
                fh.write(json.dumps({"real": " ".join(s), "word order scrambled": " ".join(prng.sample(s, len(s))),
                                     "content swapped": " ".join(t)}) + "\n")
    # topic lift from stored (previous, next) pairs
    n_pair = max(len(stored), 1)
    c_u, c_w, c_uw = Counter(), Counter(), Counter()
    for a, b in stored:
        c_u.update(set(a)); c_w.update(set(b))
        c_uw.update((u, w) for u in set(a) for w in set(b))

    def lift(prev, w):
        pw = (c_w[w] + 0.5) / (n_pair + 1.0)
        vals = [math.log(((c_uw[(u, w)] + pw) / (c_u[u] + 1.0)) / pw) for u in set(prev) if c_u[u]]
        return sum(vals) / len(vals) if vals else 0.0
    frame = defaultdict(Counter)
    for s in sents:
        t = ["<s>"] + s + ["</s>"]
        for i in range(1, len(t) - 1):
            frame[(t[i - 1], t[i + 1])][t[i]] += 1
    spanf, widef, big, unic = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    for s in sents:
        t = ["<s>", "<s>"] + s + ["</s>", "</s>"]
        unic.update(s)
        big.update(zip(t[1:-1], t[2:-1]))
        for i in range(2, len(t) - 2):
            widef[(t[i - 2], t[i - 1], t[i + 1], t[i + 2])][t[i]] += 1
            for m_ in (1, 2, 3):
                if i + m_ <= len(t) - 2:
                    spanf[(t[i - 1], t[i + m_])][tuple(t[i:i + m_])] += 1

    _side = {}

    def side_licence(a, b):
        """Two Cobweb trees over the stream's word types, one by the set of words stored just
        after each word and one by the set stored just before, a word weighing as its count,
        exact_alone on. An unstored (a, b) is licensed by the share of the words grouped with a
        that b followed, summed over the concepts on a's path, and likewise for b's side."""
        if not _side:
            from trellis2.cu_cobweb import CUCobwebTree as _Tree
            order = []
            for s_ in sents:
                for w in s_:
                    if w not in _side:
                        _side[w] = 1
                        order.append(w)
            _side.clear()
            for attr in ("R", "L"):
                nb = defaultdict(set)
                for (x, y) in pairc:
                    if x != "<s>" and y != "</s>":
                        (nb[x].add(y) if attr == "R" else nb[y].add(x))
                T_ = _Tree((attr,), seed=seed)
                T_.exact_alone = True
                ch_, mem_ = {}, defaultdict(set)
                for w in order:
                    d = nb[w]
                    nd = T_.ifit({attr: {k: 1.0 / len(d) for k in d} if d else {"<none>": 1.0}}, float(unic[w]))
                    ch_[w] = nd
                for w, nd in list(ch_.items()):
                    chain = []
                    while nd is not None:
                        chain.append(nd.id)
                        mem_[nd.id].add(w)
                        nd = nd.parent
                    ch_[w] = chain
                _side[attr] = (ch_, mem_, {})

        def rd(w, attr, v):
            ch_, mem_, memo = _side[attr]
            tot = 0.0
            for c in ch_.get(w, ())[1:]:
                k = (c, v)
                if k not in memo:
                    memo[k] = sum(1 for m_ in mem_[c] if ((m_, v) if attr == "R" else (v, m_)) in pairc) / len(mem_[c])
                tot += memo[k]
            return tot
        return math.sqrt(rd(a, "R", b) * rd(b, "L", a))

    def swap_once(s, rng, frame, unit, mn=1, pick="draw"):
        t = ["<s>", "<s>"] + s + ["</s>", "</s>"]
        m_ = rng.choice([k for k in (1, 2, 3) if k < len(s)]) if unit == "span" else 1
        i = rng.randrange(2, len(t) - 2 - m_ + 1)
        own, l, r = tuple(t[i:i + m_]), t[i - 1], t[i + m_]
        if frame == "wide":
            c = {(w,): k for w, k in widef[(t[i - 2], l, r, t[i + 2])].items()}
        elif frame == "rect":      # at least one new pair, each new pair supported by analogy
            c = {}
            for w in unic:
                sc_, new_ = 1.0, 0
                for x, y in ((l, w), (w, r)):
                    if pairc[(x, y)]:
                        sc_ *= pairc[(x, y)]
                    else:
                        sp = support(x, y)
                        sc_ *= sp if sp >= mn else 0.0
                        new_ += 1
                if new_ and sc_ > 0:
                    c[(w,)] = sc_
        elif frame in ("tree", "count"):   # at least one new pair; a stored pair counts 1, a new one by its licence
            c = {}
            lic_ = side_licence if frame == "tree" else support
            for w in unic:
                sc_, new_ = 1.0, 0
                for x, y in ((l, w), (w, r)):
                    if not pairc[(x, y)]:
                        sc_ *= lic_(x, y) if x != "<s>" and y != "</s>" else 0.0
                        new_ += 1
                if new_ and sc_ > 0:
                    c[(w,)] = sc_
        elif frame == "bigram":
            c = {(w,): big[(l, w)] * big[(w, r)] / unic[w] for w in unic if big[(l, w)] and big[(w, r)]}
        else:
            c = {y: k for y, k in spanf[(l, r)].items() if unit == "span" or len(y) == 1}
        c.pop(own, None)
        c = {y: k for y, k in c.items() if 3 <= len(s) - m_ + len(y) <= 8}
        if not c:
            return None
        if pick == "best":                 # the best-licensed replacement (ties at random)
            top = max(c.values())
            y = rng.choice(sorted(k for k, v in c.items() if v >= top * (1 - 1e-12)))
        else:
            y = rng.choices(list(c), weights=list(c.values()))[0]
        return s[:i - 2] + list(y) + s[i - 2 + m_:]
    for ai, arm in enumerate(json.loads(BASE)):
        g_rng = random.Random(seed + 7)
        t1 = time.time()
        PP = [p for p in P for _ in range(DRAWS)]
        if "ngram" in arm:
            k, tab, beta = arm["ngram"], table(arm["ngram"]), arm.get("topic", 0)

            def gen(p):
                for _ in range(200):
                    h, out = ["<s>"] * (k - 1), []
                    while len(out) <= 8:
                        c = tab[tuple(h[len(h) - k + 1:])] if k > 1 else tab[()]
                        ws = list(c)
                        wt = [c[w] * (math.exp(beta * lift(p, w)) if beta and w != "</s>" else 1.0) for w in ws]
                        w = g_rng.choices(ws, weights=wt)[0]
                        if w == "</s>":
                            break
                        out.append(w); h.append(w)
                    if 3 <= len(out) <= 8:
                        return out
                return None
        else:
            def gen(p):
                for _ in range(200):
                    if arm["swap"] == "retrieve" and stored:
                        ps_ = set(p)
                        sc = [sum(idf.get(w, 0.0) for w in ps_ & set(a)) + 1e-6 * g_rng.random() for a, _ in stored]
                        s = list(stored[max(range(len(stored)), key=sc.__getitem__)][1])
                    else:
                        s = list(g_rng.choice(sents))
                    for _k in range(arm.get("n", 1)):
                        s = swap_once(s, g_rng, arm.get("frame", "lr"), arm.get("unit", "word"), arm.get("min", 1), arm.get("pick", "draw")) if s else None
                    if s:
                        return s
                return None
        G = [gen(p) for p in PP]
        ok = [(p, g) for p, g in zip(PP, G) if g]
        c, cn = cross(*zip(*ok)) if ok else (None, 0)
        S = [g for _, g in ok]
        pat = [tuple(t[:2] for _, t in nltk.pos_tag(g)) in patterns for g in S]
        nov = [tuple(g) not in seen for g in S]
        m = lambda v: round(sum(v) / max(len(v), 1), 3)
        print(json.dumps({"tag": tag, "seed": seed, "n": n, "arm": arm, "gen s": round(time.time() - t1), "generated": len(S),
                          "cross share": None if c is None else round((c - rand_c) / (real_c - rand_c), 3),
                          "tag pattern attested": m(pat), "novel": m(nov),
                          "every bigram stored": m([attested(g, 2) for g in S]), "every trigram stored": m([attested(g, 3) for g in S]),
                          "examples": [" ".join(p) + "  ->  " + " ".join(g) for p, g in ok[:12]]}), flush=True)
        if DUMP:
            with open(os.path.join(DUMP, f"{tag}_s{seed}_n{n}_arm{ai}.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"arm": arm}) + "\n")
                for (p, g), a_, b_ in zip(ok, pat, nov):
                    fh.write(json.dumps({"prompt": " ".join(p), "gen": " ".join(g), "pattern": a_, "novel": b_}) + "\n")
    sys.exit()

L = Trellis3(seed=seed, **cfg)
t0 = time.time()
EXP = []                                       # each experience's tokens and its records' range in memory
for s, fol in train:
    a0 = len(L.records)
    L.learn(s, follows=fol)
    EXP.append((list(s), a0, len(L.records)))
learn_s = time.time() - t0


if LIC_SET:                                    # the learner's licence: Cobweb's prediction from one neighbour
    cand, ap, words_ = LIC_SET

    def dist(q):
        f = {k[1]: v for k, v in L._gap_fills(q).items() if k[0] == "w"}
        z = sum(f.values()) or 1.0
        return {w: v / z for w, v in f.items()}
    PL = {a: dist({"l1": a}) for a in words_}      # P(word | the word before it is a)
    PR = {b: dist({"r1": b}) for b in words_}      # P(word | the word after it is b)
    for name, fn in (("Cobweb, word given the one before", lambda a, b: PL[a].get(b, 0.0)),
                     ("Cobweb, word given the one after", lambda a, b: PR[b].get(a, 0.0)),
                     ("Cobweb, both", lambda a, b: math.sqrt(PL[a].get(b, 0.0) * PR[b].get(a, 0.0)))):
        print(json.dumps({"licence": name, "seed": seed, "n": n, "learn s": round(learn_s), **ap(fn)}), flush=True)
    sys.exit()


def yield_of(e):
    return [e.token] if e.token is not None else yield_of(e.left) + yield_of(e.right)


_mem = {}


def t_swap(p, unit, up, by, rng):
    """One stored unit of a stored sentence replaced by another stored unit. The sentence is the
    one that followed the stored sentence most like the prompt (as the "retrieve" reference).
    unit "word": a word for a word; "unit": a word or a chunk of the committed analysis for a word
    or a committed chunk. by "concept": the replacement is another member of the unit's context
    concept ``up`` levels above its leaf, drawn by stored weight; by "fills": drawn from Cobweb's
    prediction for the unit's surface window (_gap_fills)."""
    if not _mem:
        _mem["m"] = L._members(L.ctx, "ctx")
        _mem["next"] = [i for i, (_, fol) in enumerate(train) if fol and i > 0]
    ps_ = set(p)
    for _ in range(50):
        if _mem["next"]:
            sc = [sum(idf.get(w, 0.0) for w in ps_ & set(train[i - 1][0])) + 1e-6 * rng.random() for i in _mem["next"]]
            t = _mem["next"][max(range(len(sc)), key=sc.__getitem__)]
        else:
            t = rng.randrange(len(EXP))
        s, a0, b0 = EXP[t]
        cands = [e for e in L.records[a0:b0] if e.leaf is not None and
                 (e.token is not None or (unit == "unit" and e.used and e.span[1] - e.span[0] < len(s)))]
        if not cands:
            continue
        e = rng.choice(cands)
        i, j = e.span
        pool = {}
        if by == "fills":
            for f, w in L._gap_fills(dict(L._surface(s, i, j))).items():
                if f[0] == "w":
                    y = (f[1],)
                elif unit == "unit":
                    y = tuple(yield_of(L.records[f[1]]))
                else:
                    continue
                pool[y] = pool.get(y, 0.0) + w
        else:
            path = L.ctx.ancestors(e.leaf)[::-1]
            node = path[max(len(path) - 1 - up, 1 if len(path) > 1 else 0)]
            for r in _mem["m"][f"c{node.id}"]:
                if r.weight <= 0 or (r.token is None and (unit != "unit" or not r.used)):
                    continue
                y = tuple(yield_of(r))
                pool[y] = pool.get(y, 0.0) + r.weight
        pool.pop(tuple(s[i:j]), None)
        pool = {y: w for y, w in pool.items() if w > 0 and 3 <= len(s) - (j - i) + len(y) <= 8}
        if not pool:
            continue
        y = rng.choices(list(pool), weights=list(pool.values()))[0]
        return s[:i] + list(y) + s[j:]
    return None
L._quiet = True
ix = L.concept_index()
DUMP = os.environ.get("STORY_DUMP")            # a directory: every generation of every arm, as JSON lines
if DUMP:                                         # judge calibration: held-out sentences, their words in random order, content words swapped by tag
    os.makedirs(DUMP, exist_ok=True)
    ref = random.Random(seed + 21).sample(held, min(300, len(held)))
    with open(os.path.join(DUMP, f"{tag}_s{seed}_n{n}_refs.jsonl"), "w", encoding="utf-8") as fh:
        prng = random.Random(seed + 23)
        for s, t in zip(ref, shuffled(ref, random.Random(seed + 22))):
            fh.write(json.dumps({"real": " ".join(s), "word order scrambled": " ".join(prng.sample(s, len(s))),
                                 "content swapped": " ".join(t)}) + "\n")
for ai, arm in enumerate(arms):
    arm = dict(arm)
    cond = arm.pop("cond", False)
    critic = arm.pop("critic", 0)          # draw this many and keep the one memory parses best per word
    critic_mode = arm.pop("critic_mode", "mean")   # "mean": parsed best per word; "weakest": least-attested join best; "chart": chart probability
    weakest = critic_mode == "weakest"
    g_rng = random.Random(seed + 7)
    t1 = time.time()
    PP = [p for p in P for _ in range(DRAWS)]
    chart = arm.pop("chart", None)         # chart generation (chart_gen.py) with these settings
    tsw = arm.pop("tswap", None)           # one stored unit swapped (t_swap)
    if tsw is not None:
        up_, by_ = arm.pop("up", 1), arm.pop("by", "concept")
        gen = lambda p: t_swap(p, tsw, up_, by_, g_rng)
        arm.update({"tswap": tsw, "up": up_, "by": by_})
    elif chart is not None:
        CG = ChartGen(L, **chart)
        print(json.dumps({"tag": tag, "seed": seed, "chart": chart, **CG.stats}), flush=True)
        if chart.get("topic") == "fit":    # beta by likelihood on held-out pairs (stored pairs are recalled whole)
            fit = CG.fit_topic(val)
            print(json.dumps({"tag": tag, "seed": seed, "fit pairs": len(val), "fit topic": {str(b): round(v, 4) for b, v in fit.items()},
                              "beta": CG.topic}), flush=True)
            if os.environ.get("FIT_ONLY"):     # a diagnostic: the fit alone, no generation
                continue
        print(json.dumps(CG.describe()), flush=True)
        gen = lambda p: CG.generate(["*"], g_rng, prev=p if cond else None)
        arm["chart"] = chart
    else:
        gen = lambda p: L.generate_tree(g_rng, index=ix, fill_name=FILL, **({"after": p} if cond else {}), **{"choose": "leaf", **arm})

    def score(g):
        if critic_mode == "chart":          # the chart's own probability of the filled sentence, per word (as the coder)
            return CG.logZ(CG.inside(list(g)), len(g)) / len(g)
        r = L.parse(g)
        mean = r["logZ"] / max(len(g) - 1, 1)
        if not weakest:
            return mean
        joins = [r["comps"][c].gain for c in r["chosen"] if c in r["comps"]]
        return (min(joins) if joins else mean, mean)

    def best(p):
        cands = [g for g in (gen(p) for _ in range(critic)) if g]
        return max(cands, key=score) if cands else None
    G = [best(p) if critic else gen(p) for p in PP]
    ok = [(p, g) for p, g in zip(PP, G) if g]
    c, cn = cross(*zip(*ok)) if ok else (None, 0)
    S = [g for _, g in ok]
    pat = [tuple(t[:2] for _, t in nltk.pos_tag(g)) in patterns for g in S]
    nov = [tuple(g) not in seen for g in S]
    w = within(S)
    ws = within(shuffled(S, random.Random(seed + 13)))
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    print(json.dumps({"tag": tag, "seed": seed, "n": n, "arm": {"cond": cond, **({"critic": critic} if critic else {}), **({"critic_mode": critic_mode} if critic_mode != "mean" else {}), **arm}, "learn s": round(learn_s),
                      "gen s": round(time.time() - t1), "generated": len(S),
                      "cross": None if c is None else round(c, 3), "cross scored": cn,
                      "cross share": None if c is None else round((c - rand_c) / (real_c - rand_c), 3),
                      "tag pattern attested": m(pat), "novel": m(nov),
                      "every bigram stored": m([attested(g, 2) for g in S]), "every trigram stored": m([attested(g, 3) for g in S]),
                      "novel tag pattern attested": m([a for a, b in zip(pat, nov) if b]),
                      "within share": None if w is None or ws is None else round((w - ws) / (w_real - w_shuf), 3),
                      "examples": [" ".join(p) + "  ->  " + " ".join(g) for p, g in ok[:12]],
                      "novel examples": [(" ".join(g), a) for (_, g), a, b in zip(ok, pat, nov) if b][:16]}), flush=True)
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
        with open(os.path.join(DUMP, f"{tag}_s{seed}_n{n}_arm{ai}.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"arm": {"cond": cond, **({"critic": critic} if critic else {}), **({"critic_mode": critic_mode} if critic_mode != "mean" else {}), **arm}}) + "\n")
            for (p, g), a, b in zip(ok, pat, nov):
                fh.write(json.dumps({"prompt": " ".join(p), "gen": " ".join(g), "pattern": a, "novel": b}) + "\n")

# infill (INFILL=<n>): mask 1-3 consecutive words of held-out sentences; the gap is filled
# by expansion (generate_tree's gap mode: one stored instance standing there) and by the
# chart (chart_gen.py: parts of several constituents may cover it)
N_INF = int(os.environ.get("INFILL", 0))
if N_INF:
    irng, g_rng = random.Random(seed + 31), random.Random(seed + 37)
    CG = ChartGen(L, **next((a["chart"] for a in arms if "chart" in a), {}))   # the first chart arm's settings
    items = []
    for s in held:
        if len(s) >= 4 and len(items) < N_INF:
            m_ = irng.randint(1, 3)
            i = irng.randint(0, len(s) - m_)
            items.append((s, i, m_))
    methods = {"expand gap": lambda s, i, m_: L.generate_tree(g_rng, index=ix, fill_name=FILL, choose="leaf",
                                                              gap=(s[:i], s[i + m_:])),
               "chart": lambda s, i, m_: CG.generate(s[:i] + ["*"] + s[i + m_:], g_rng)}
    for mi, (name, f) in enumerate(methods.items()):
        t1, outs = time.time(), []
        for s, i, m_ in items:
            try:
                g = f(s, i, m_)
            except Exception as ex:                    # a failed fill counts as no output
                g = None
            outs.append(g)
        ok = [(it, g) for it, g in zip(items, outs) if g]
        S = [g for _, g in ok]
        pat = [tuple(t[:2] for _, t in nltk.pos_tag(g)) in patterns for g in S]
        nov = [tuple(g) not in seen for g in S]
        exact = [g == it[0] for it, g in ok]
        m = lambda v: round(sum(v) / max(len(v), 1), 3)
        print(json.dumps({"tag": tag, "seed": seed, "infill": name, "items": len(items), "filled": m([g is not None for g in outs]),
                          "gen s": round(time.time() - t1), "exact original": m(exact), "novel": m(nov),
                          "tag pattern attested": m(pat), "novel tag pattern attested": m([a for a, b in zip(pat, nov) if b]),
                          "examples": [" ".join(it[0][:it[1]] + ["[" + " ".join(g[it[1]:len(g) - (len(it[0]) - it[1] - it[2])]) + "]"] +
                                                it[0][it[1] + it[2]:]) for it, g in ok[:16]]}), flush=True)
        if DUMP:
            with open(os.path.join(DUMP, f"{tag}_s{seed}_n{n}_arm{90 + mi}.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"arm": {"infill": name}}) + "\n")
                for (s, i, m_), g in ok:
                    fh.write(json.dumps({"prompt": " ".join(s[:i] + ["*"] + s[i + m_:]), "gen": " ".join(g),
                                         "pattern": tuple(t[:2] for _, t in nltk.pos_tag(g)) in patterns,
                                         "novel": tuple(g) not in seen}) + "\n")
