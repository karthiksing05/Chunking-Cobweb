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
lm_held = [s for st in train_st[k + 1:k + 400] for s in st if keep(s)]   # for sizing the neural LM baselines: stories the stream never reached
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
#   {"lm": "gpt"} / {"lm": "lstm"}  word-level neural LM trained only on the n sentences (t3_lm.py), size
#                                chosen by held-out loss on lm_held, sampled unconditioned like the ngram arm
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
                for r in (0.01, 0.05, 0.1, 0.25, 0.5):
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
    if LIC != "1":
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
            if os.environ.get("STORY_EXAMPLES"):      # the new sentences with an unstored pair, the pair marked, and the score
                ex = []
                for r, p in zip(rows, ps):
                    g = r["gen"].split()
                    if r["novel"] and not attested(g, 2):
                        t_ = ["<s>"] + g + ["</s>"]
                        bad = [f"{a}_{b}" for a, b in zip(t_, t_[1:]) if (a, b) not in have[2]]
                        ex.append(f"{p:.2f} | {' '.join(g)} | {' '.join(bad)}")
                out["unstored examples"] = ex[:60]
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

    def swap_once(s, rng, frame, unit, mn=1, pick="draw"):
        t = ["<s>", "<s>"] + s + ["</s>", "</s>"]
        m_ = rng.choice([k for k in (1, 2, 3) if k < len(s)]) if unit == "span" else 1
        if pick.startswith("slot"):
            # the slot is chosen too: each position offers its best replacement, weighed by its
            # score relative to the score of the word that stood there ("slot": drawn by that
            # ratio; "slot-best": the highest ratio)
            opts = [x for x in (_swap_at(s, t, i, 1, rng, frame, unit, mn, "best") for i in range(2, len(t) - 2)) if x and x[1]]
            if not opts:
                return None
            if pick == "slot-best":
                top = max(v for _, v in opts)
                return rng.choice([o for o, v in opts if v >= top * (1 - 1e-12)])
            return rng.choices([o for o, _ in opts], weights=[v for _, v in opts])[0]
        x = _swap_at(s, t, rng.randrange(2, len(t) - 2 - m_ + 1), m_, rng, frame, unit, mn, pick)
        return x[0] if x else None

    def _swap_at(s, t, i, m_, rng, frame, unit, mn, pick):
        own, l, r = tuple(t[i:i + m_]), t[i - 1], t[i + m_]
        own_score = None
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
        elif frame == "count":             # at least one new pair; a stored pair counts 1, a new one by its analogy count
            c = {}
            lic_ = support
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
        own_v = c.pop(own, None)
        own_v = own_score if own_score is not None else own_v
        c = {y: k for y, k in c.items() if 3 <= len(s) - m_ + len(y) <= 8}
        if not c:
            return None
        if pick == "best":                 # the best-licensed replacement (ties at random)
            top = max(c.values())
            y = rng.choice(sorted(k for k, v in c.items() if v >= top * (1 - 1e-12)))
        else:
            y = rng.choices(list(c), weights=list(c.values()))[0]
        # the offer's weight: its score relative to the standing word's; a pair licence has no such
        # score for the standing word (its pairs are stored), so its own value stands
        return s[:i - 2] + list(y) + s[i - 2 + m_:], (c[y] / own_v if own_v else (c[y] if frame == "count" else None))
    for ai, arm in enumerate(json.loads(BASE)):
        g_rng = random.Random(seed + 7)
        t1 = time.time()
        PP = [p for p in P for _ in range(DRAWS)]
        lm_info = None
        if "lm" in arm:
            from t3_lm import train_lm
            sample, lm_info = train_lm(arm["lm"], sents, lm_held, vocab, seed, temp=float(arm.get("temp", 1.0)))
            print(json.dumps({"tag": tag, "seed": seed, "n": n, "arm": arm, **lm_info}), flush=True)
            pool = []
            while len(pool) < len(PP):          # unconditioned: draws until enough sentences of 3-8 words (the ngram arm's retry, in bulk)
                pool += [s for s in sample(2048) if s and 3 <= len(s) <= 8 and not (arm.get("new") and tuple(s) in seen)]
            it_ = iter(pool)
            gen = lambda p: next(it_)
        elif "ngram" in arm:
            k, tab, beta = arm["ngram"], table(arm["ngram"]), arm.get("topic", 0)
            temp_ = float(arm.get("temp", 1.0))

            def gen(p):
                for _ in range(200):
                    h, out = ["<s>"] * (k - 1), []
                    while len(out) <= 8:
                        c = tab[tuple(h[len(h) - k + 1:])] if k > 1 else tab[()]
                        ws = list(c)
                        wt = [c[w] * (math.exp(beta * lift(p, w)) if beta and w != "</s>" else 1.0) for w in ws]
                        if temp_ != 1.0:    # r188: sampling temperature, counts ** (1 / temp); absent = untouched
                            wt = [x ** (1.0 / temp_) for x in wt]
                        w = g_rng.choices(ws, weights=wt)[0]
                        if w == "</s>":
                            break
                        out.append(w); h.append(w)
                    if 3 <= len(out) <= 8 and not (arm.get("new") and tuple(out) in seen):   # "new": never a stored sentence
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
                    if s and not (arm.get("new") and tuple(s) in seen):
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
if os.environ.get("FILL_GAMMA") or os.environ.get("FILL_BACKOFF"):   # test-time reads of Cobweb's fills
    import functools                           # r177 FILL_GAMMA: fillers by types (0) .. tokens (1); r178 FILL_BACKOFF=types
    _fk = {**({"gamma": float(os.environ["FILL_GAMMA"])} if os.environ.get("FILL_GAMMA") else {}),
           **({"backoff": os.environ["FILL_BACKOFF"]} if os.environ.get("FILL_BACKOFF") else {})}
    L._gap_fills = functools.partial(L._gap_fills, **_fk)
    print(json.dumps({"fill read": _fk}), flush=True)


if os.environ.get("STORY_R204_REPLAY"):       # r204 pre-test: replay judged dumps' boundary steps under the default read and edge_exact, then exit
    sys.path.insert(0, os.path.join("experiments", "v3", "diag"))
    import r204_replay
    r204_replay.run(L, train, seed, n, os.environ["STORY_R204_REPLAY"], os.environ.get("R204_OUT", "r204_replay.json"))
    sys.exit()

if os.environ.get("STORY_R207_M2"):           # r207 mechanism-2 probe (no generation): pivot_support "left" AUC + tree statistics, then exit
    sys.path.insert(0, os.path.join("experiments", "v3", "diag"))
    import r207_m2
    r207_m2.run(L, train, seed, n, os.environ["STORY_R207_M2"], os.environ.get("R207_OUT", "r207_m2.tsv"))
    sys.exit()

if os.environ.get("STORY_CRITIC_AUC"):    # r182 pre-test: does the learner's parse tell judged-good new sentences from bad?
    # On judged dumps of this seed and n, each new generation is scored by the parse (log Z per join,
    # and the weakest chosen join) and by a stream baseline (the rarest trigram's count); reported:
    # AUC against the judge, and the judged-grammatical share kept when only the top half/third pass.
    import glob
    L._quiet = True
    tri = Counter(g for s, _ in train for g in grams(s, 3))

    def auc(xs, ys):
        pos = [x for x, y in zip(xs, ys) if y]
        neg = [x for x, y in zip(xs, ys) if not y]
        if not pos or not neg:
            return None
        w = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
        return round(w / (len(pos) * len(neg)), 3)
    for f in sorted(glob.glob(os.path.join(os.environ["STORY_CRITIC_AUC"], f"*_s{seed}_n{n}_arm*.jsonl"))):
        sc = f[:-len(".jsonl")] + ".scores.json"
        if not os.path.exists(sc):
            continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8")][1:]
        ps = json.load(open(sc, encoding="utf-8"))
        items = [(r["gen"].split(), p > 0.5) for r, p in zip(rows, ps) if r["novel"] and len(r["gen"].split()) >= 2]
        feats = defaultdict(list)
        for g, _ in items:
            r_ = L.parse(g)
            feats["logZ per join"].append(r_["logZ"] / max(len(g) - 1, 1))
            joins = [r_["comps"][c].gain for c in r_["chosen"] if c in r_["comps"]]
            feats["weakest join"].append(min(joins) if joins else -1e9)
            feats["rarest trigram count"].append(min(tri.get(t_, 0) for t_ in grams(g, 3)))
            feats["minus length"].append(-len(g))
        ys = [y for _, y in items]
        out = {"critic auc": os.path.basename(f), "seed": seed, "n": n, "new": len(items), "judged good": round(sum(ys) / max(len(ys), 1), 3)}
        for k, xs in feats.items():
            order = sorted(range(len(xs)), key=lambda i: -xs[i])
            keep = {fr: round(sum(ys[i] for i in order[:int(len(order) * fr)]) / max(int(len(order) * fr), 1), 3) for fr in (0.5, 0.33)}
            out[k] = {"auc": auc(xs, ys), "good among top half": keep[0.5], "good among top third": keep[0.33]}
        print(json.dumps(out), flush=True)
    sys.exit()

if os.environ.get("STORY_BOUND"):             # r177 pre-test: where held-out unstored bigrams fall in the learner's parse
    # Each adjacent pair (t_i, t_i+1) of a held-out sentence is joined by the one split of the parse at i+1;
    # its children's sizes say whether the pair is word+word (inside a phrase) or joins larger units
    # (a phrase junction). Phrase-boundary recombination can reach an unstored pair only at a junction.
    L._quiet = True
    ref = random.Random(seed + 21).sample(held, min(300, len(held)))
    tab = {True: Counter(), False: Counter()}
    for s in ref:
        if len(s) < 2:
            continue
        r = L.parse(list(s))
        for (i_, k_, j_) in r["chosen"]:
            pair = (s[k_ - 1], s[k_])
            cls = "word+word" if (k_ - i_ == 1 and j_ - k_ == 1) else ("unit+unit" if (k_ - i_ >= 2 and j_ - k_ >= 2) else "word+unit")
            tab[pair in have[2]][cls] += 1
    for st, c in tab.items():
        z = sum(c.values()) or 1
        print(json.dumps({"bound": "stored" if st else "unstored", "seed": seed, "pairs": z,
                          **{k: round(c[k] / z, 3) for k in ("word+word", "word+unit", "unit+unit")}}), flush=True)
    sys.exit()

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
    # The same context tree read by types: a context concept's members are the distinct words
    # that filled the occurrences under it. An unstored (a, b) is licensed on b's side by the
    # share of the members of the concepts above b's occurrences that a was stored before, and
    # on a's side by the share of the members above a's occurrences that b was stored after.
    # "union": each concept above any occurrence counts once; "mean": the path sums averaged
    # over the word's occurrences; "leafless": the occurrence's own leaf left out.
    from trellis2.trellis3 import FILL as T3FILL
    t3_ = time.time()
    members, occ_paths = defaultdict(set), defaultdict(list)
    for e in L.records:
        if e.leaf is None:
            continue
        for which, attr, value, w in e.pas:
            if which == "ctx" and attr == T3FILL and isinstance(value, tuple) and value[0] == "w":
                path, nd = [], e.leaf
                while nd is not None:
                    path.append(nd.id)
                    members[nd.id].add(value[1])
                    nd = nd.parent
                occ_paths[value[1]].append(path)
    pc_set = set(pc)
    sh_memo = {}

    def share(cid, x, side):                  # side "R": members m with (x, m) stored; "L": with (m, x)
        k = (cid, x, side)
        if k not in sh_memo:
            mem = members[cid]
            sh_memo[k] = len(mem & (Rt[x] if side == "R" else Lf[x])) / len(mem)
        return sh_memo[k]
    cpaths = {}

    def concepts(w, how):
        k = (w, how)
        if k not in cpaths:
            ps = occ_paths.get(w, [])
            if how == "union":
                cpaths[k] = [sorted({c for p in ps for c in p[1:]})]
            elif how == "union-all":
                cpaths[k] = [sorted({c for p in ps for c in p})]
            else:                                 # mean
                cpaths[k] = [p[1:] for p in ps]
        return cpaths[k]

    def side_read(w, x, side, how):
        ps = concepts(w, how)
        if not ps:
            return 0.0
        return sum(sum(share(c, x, side) for c in p) for p in ps) / len(ps)
    stats_ = {"filler words": len(occ_paths), "concepts with members": len(members),
              "mean occurrences per word": round(sum(len(v) for v in occ_paths.values()) / max(len(occ_paths), 1), 1),
              "mean path": round(sum(len(p) for v in occ_paths.values() for p in v) / max(sum(len(v) for v in occ_paths.values()), 1), 1)}
    print(json.dumps({"licence": "context tree by types, setup", "seed": seed, "n": n, **stats_}), flush=True)
    for how in ("union", "union-all", "mean"):
        print(json.dumps({"licence": f"context tree by types, {how}", "seed": seed, "n": n,
                          **ap(lambda a, b: math.sqrt(side_read(b, a, "R", how) * side_read(a, b, "L", how)))}), flush=True)
        print(json.dumps({"licence": f"context tree by types, {how}, b's side only", "seed": seed, "n": n,
                          **ap(lambda a, b: side_read(b, a, "R", how))}), flush=True)
    print(json.dumps({"licence": "context tree by types, done", "seed": seed, "n": n, "s": round(time.time() - t3_)}), flush=True)
    t4_ = time.time()
    _w, _ix, _M = L.type_licence()                 # the learner's own method: should equal "mean" above
    print(json.dumps({"licence": "learner type_licence", "seed": seed, "n": n, "s": round(time.time() - t4_, 1),
                      **ap(lambda a, b: float(_M[_ix[a], _ix[b]]) if a in _ix and b in _ix else 0.0)}), flush=True)
    # How the path is weighed (same tree, same members): every concept alike (as above), the root
    # left out, each concept by its information (log2 of the root's count over its own), or only the
    # k concepts nearest the occurrence.
    import numpy as np
    cnt_of, depth_from_leaf = {}, defaultdict(list)
    for e in L.records:
        if e.leaf is None:
            continue
        nd = e.leaf
        while nd is not None:
            cnt_of[nd.id] = nd.count
            nd = nd.parent
    root_id = L.ctx.root.id
    wl = sorted(occ_paths)
    wi = {w: i for i, w in enumerate(wl)}
    cl = sorted(members)
    ci = {c: k for k, c in enumerate(cl)}
    Mem = np.zeros((len(cl), len(wl)))
    for c, ms in members.items():
        for m_ in ms:
            Mem[ci[c], wi[m_]] = 1.0 / len(ms)
    aft, bef = np.zeros((len(wl), len(wl))), np.zeros((len(wl), len(wl)))
    for (x_, y_) in pc:
        if x_ in wi and y_ in wi:
            aft[wi[x_], wi[y_]] = 1.0
            bef[wi[y_], wi[x_]] = 1.0
    sh_a, sh_b = Mem @ aft.T, Mem @ bef.T
    n_root = max(cnt_of.get(root_id, 1.0), 1e-9)

    def weigh(mode):
        Wm = np.zeros((len(wl), len(cl)))
        for w, ps in occ_paths.items():
            for p in ps:
                above = p[1:]
                if mode.startswith("near"):
                    above = above[:int(mode[4:])]
                for c in above:
                    if mode == "noroot" and c == root_id:
                        continue
                    f = math.log2(n_root / max(cnt_of.get(c, n_root), 1e-9)) if mode == "bits" else 1.0
                    Wm[wi[w], ci[c]] += f / len(ps)
        return Wm
    for mode in ("all", "noroot", "bits", "near2", "near4", "near6"):
        Wm = weigh(mode)
        B_, A_ = (Wm @ sh_a).T, (Wm @ sh_b)            # b's side [a, b], a's side [a, b]
        for comb, Mx in (("geo", np.sqrt(B_ * A_)), ("min", np.minimum(B_ / max(B_.mean(), 1e-12), A_ / max(A_.mean(), 1e-12)))):
            print(json.dumps({"licence": f"context tree by types, path {mode}, {comb}", "seed": seed, "n": n,
                              **ap(lambda a, b: float(Mx[wi[a], wi[b]]) if a in wi and b in wi else 0.0)}), flush=True)
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


_us = {}
_us_log = []                                   # (a chunk was involved, ratio of the chosen offer), read and cleared by the arm loop


def _us_build(units):
    """Unit occurrences of the whole stream (words, and committed chunks when units == "chunks"), their place fillers,
    and two Cobweb trees over the unit types (words stored after / before each type), with chains and members."""
    from trellis2.cu_cobweb import CUCobwebTree as _Tree
    U = {"units": units, "count": defaultdict(int), "after": defaultdict(set), "before": defaultdict(set), "order": [],
         "F": {k: defaultdict(set) for k in ("lr", "wide", "after", "before")}, "slots": [], "votes": {}}
    for t, (s, a0, b0) in enumerate(EXP):
        spans = {(i, i + 1) for i in range(len(s))}
        if units == "chunks":
            spans |= {e.span for e in L.records[a0:b0]
                      if e.token is None and e.used and e.leaf is not None and e.span[1] - e.span[0] < len(s)}
        occ = []
        for (i, j) in sorted(spans):
            y = tuple(s[i:j])
            l = s[i - 1] if i > 0 else "<s>"
            r = s[j] if j < len(s) else "</s>"
            l2 = s[i - 2] if i > 1 else "<s>"
            r2 = s[j + 1] if j + 1 < len(s) else "</s>"
            if y not in U["count"]:
                U["order"].append(y)
            U["count"][y] += 1
            U["after"][y].add(r)
            U["before"][y].add(l)
            U["F"]["lr"][(l, r)].add(y)
            U["F"]["wide"][(l2, l, r, r2)].add(y)
            U["F"]["after"][l].add(y)
            U["F"]["before"][r].add(y)
            occ.append((i, j, y, l, r, l2, r2))
        U["slots"].append(occ)
    for attr, side in (("R", "after"), ("L", "before")):
        T = _Tree((attr,), seed=seed)
        T.exact_alone = True
        leaf = {}
        for y in U["order"]:
            d = U[side][y]
            leaf[y] = T.ifit({attr: {v: 1.0 / len(d) for v in d}}, float(U["count"][y]))
        chain, mem = {}, defaultdict(set)
        for y, nd in leaf.items():
            ch = []
            while nd is not None:
                ch.append(nd.id)
                mem[nd.id].add(y)
                nd = nd.parent
            chain[y] = ch
        U[attr] = (chain, mem)
    return U


def _us_votes(U, kind, key):
    """Vote of every unit type in the place (kind, key): over both trees and every concept on its path (leaf excluded),
    the share of the concept's members among the place's fillers. Sparse dict, cached per place."""
    ck = (kind, key)
    if ck in U["votes"]:
        return U["votes"][ck]
    F = U["F"][kind].get(key)
    out = {}
    if F:
        for attr in ("R", "L"):
            chain, mem = U[attr]
            cnt = defaultdict(int)
            for f in F:
                for c in chain[f]:
                    cnt[c] += 1
            nodes = set(cnt)
            for y in U["order"]:
                tot = 0.0
                for c in chain[y][1:]:
                    k = cnt.get(c)
                    if k:
                        tot += k / len(mem[c])
                if tot:
                    out[y] = out.get(y, 0.0) + tot
    U["votes"][ck] = out
    return out


def u_swap(p, units, places, pick, rng):
    """Slot-and-swap over stored units: each unit occurrence (a word, or a committed chunk) of the retrieved stored sentence
    offers its best replacement, weighed by its votes relative to the votes of the unit standing there; one offer is taken
    ("slot": drawn by that ratio; "slot-best": the highest)."""
    if units not in _us:
        _us[units] = _us_build(units)
    U = _us[units]
    if not _mem:
        _mem["m"] = L._members(L.ctx, "ctx")
        _mem["next"] = [i for i, (_, fol) in enumerate(train) if fol and i > 0]
    ps_ = set(p)
    if _mem["next"]:
        sc = [sum(idf.get(w, 0.0) for w in ps_ & set(train[i - 1][0])) + 1e-6 * rng.random() for i in _mem["next"]]
        t = _mem["next"][max(range(len(sc)), key=sc.__getitem__)]
    else:
        t = rng.randrange(len(EXP))
    s = EXP[t][0]
    offers = []
    for (i, j, own, l, r, l2, r2) in U["slots"][t]:
        pv = [_us_votes(U, "lr", (l, r))]
        if places in ("wide", "all"):
            pv.append(_us_votes(U, "wide", (l2, l, r, r2)))
        if places == "all":
            pv += [_us_votes(U, "after", l), _us_votes(U, "before", r)]
        pv = [v for v in pv if v]
        sown = sum(v.get(own, 0.0) for v in pv)
        if sown <= 0:
            continue
        score = defaultdict(float)
        for v in pv:
            for y, x in v.items():
                score[y] += x
        base = len(s) - (j - i)
        cand = {y: x for y, x in score.items() if y != own and 3 <= base + len(y) <= 8}
        if not cand:
            continue
        top = max(cand.values())
        y = rng.choice(sorted(y for y, x in cand.items() if x == top))
        offers.append((top / sown, i, j, own, y))
    if not offers:
        return None
    if pick == "slot-best":
        top = max(o[0] for o in offers)
        o = rng.choice([o for o in offers if o[0] == top])
    else:
        o = rng.choices(offers, weights=[o[0] for o in offers])[0]
    ratio, i, j, own, y = o
    _us_log.append((len(own) > 1 or len(y) > 1, ratio))
    return s[:i] + list(y) + s[j:]
L._quiet = True
ix = L.concept_index()
DUMP = os.environ.get("STORY_DUMP")            # a directory: every generation of every arm, as JSON lines
_SEG_ = {}                                       # r194 ("seg" arms): generated sentence -> its unit lengths
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
    usw = arm.pop("uswap", None)           # one stored unit (word or chunk) swapped by Cobweb-tree votes (u_swap)
    causal = arm.pop("causal", None)       # left to right from the context tree (see below)
    extra = {}
    _us_log.clear()
    if tsw is not None:
        up_, by_ = arm.pop("up", 1), arm.pop("by", "concept")
        gen = lambda p: t_swap(p, tsw, up_, by_, g_rng)
        arm.update({"tswap": tsw, "up": up_, "by": by_})
    elif usw is not None:
        pl_, pk_ = arm.pop("places", "wide"), arm.pop("pick", "slot")
        gen = lambda p: u_swap(p, usw, pl_, pk_, g_rng)
        arm.update({"uswap": usw, "places": pl_, "pick": pk_})
    elif causal is not None:
        # Causal generation, the story counterpart of chess's causal context: left to right, each filler
        # (a word, or with units "both" also a committed chunk, which places its words) is drawn from the
        # context tree given what is already placed (l2, l1), jointly with whether the sentence ends right
        # after it: Cobweb's filler prediction with r1 = end marker (end) and without it minus that (go on).
        # Lengths outside 3..8 are redrawn.
        from trellis2.trellis3 import BOS as T3BOS, EOS as T3EOS
        units_ = arm.pop("units", "both")
        new_ = arm.pop("new", False)       # never return a stored sentence (redraw)
        q_ = float(arm.pop("q", 0.5))
        near_ = arm.pop("near", None)      # the licence read at the k concepts nearest each occurrence (r158)
        # ("spec", r160, swapped a drawn word for a member of its context concept: no better than trigram; deleted r171)
        sweep_ = int(arm.pop("sweep", 0))  # r171: k Gibbs passes over the draft, each word redrawn from the context tree
                                           # given both sides (l2, l1, r1, r2), the gate on both of its pairs
        temp_ = float(arm.pop("temp", 1.0))  # r188: sampling temperature; the drawn weights go to w ** (1 / temp), 1.0 = unchanged
        repair_ = bool(arm.pop("repair", False))   # r193: after drawing, redraw each word given both windows until a sweep changes nothing (max length sweeps)
        reach_ = int(arm.pop("reach", 1))  # r192 v2 (with "both"): 2 also checks the word two back as a pivot (r1 = l1, r2 = new word)
        seg_ = bool(arm.pop("seg", False))  # r194: record each generated sentence's unit lengths in the dump ("units")
        read_ = arm.pop("read", None)       # r195: the fill read that mixes levels (Trellis3.hier_fills how = "kn" | "top" |
                                           # "ngram" | "path" | "post"); P(w) and P(w, end) replace the product read's masses
        pool_ = arm.pop("pool", None)      # r197: "geo" = the fill read weights each occurrence by the GEOMETRIC MEAN of its per-attribute
                                           # back-off factors (product ** (1 / A)) in _gap_fills, fill_words and pivot_support
        L.pool = pool_
        edge_ = arm.pop("edge", None)      # r204 (M1): "exact" = the start / end marker slots are read from the occurrences that really
                                           # stood at that boundary (trellis3_r204.Trellis3R204, edge_exact); default None = unchanged
        if edge_ == "exact":
            from trellis2.trellis3_r204 import Trellis3R204
            L.__class__ = Trellis3R204
            L.edge_exact = True
        escape_ = arm.pop("escape", None)  # r207 (M1 of r206): "n1" = a novel continuation gets only its history's missing mass
                                           # N1 / (c + N1) at (l2, l1) and, for the fillers the pair gate admits, at l1
                                           # (trellis3_r207.Trellis3R207.escape_split); default None = unchanged
        if escape_ == "n1":
            from trellis2.trellis3_r207 import Trellis3R207
            _ee_ = bool(getattr(L, "edge_exact", False))
            L.__class__ = Trellis3R207
            L.edge_exact = _ee_
            L.escape_mass = True
        both_ = arm.pop("both", None)    # r192: a filler that makes a trigram memory never stored is kept only if the
                                           # context tree, re-predicting the previous word (the pivot) from ITS left
                                           # context, finds the filler supports it: log P(pivot | left, r1 = w) /
                                           # P(pivot | left) >= 0 (Trellis3.pivot_support kind "pivot"; "left" / "both"
                                           # are the other kinds); a new end ((l1, w, end) never stored) likewise with w
                                           # as the pivot and the end marker as r1, and only after a word memory saw
                                           # end a sentence (the end marker is a right neighbour without a type licence;
                                           # likewise the first word: only a word memory saw start a sentence)
        esc_ = arm.pop("esc", None)        # r179 "licence": unstored next words drawn FROM the type licence, with the
                                           # Witten-Bell escape mass d/(d+c) of l1 (d distinct stored successors, c tokens);
                                           # the context tree draws the stored successors with the rest
        pair_ = arm.pop("pair", None)      # "gate": a filler whose first word was never stored after l1 is kept
        _cz = {}                           # only if the context tree read by types licenses the pair at or above
        if pair_:                          # the median over stored pairs (as chart_gen's pair_lic gate)
            _w_, _ix_, _M_ = L.type_licence(near=near_)
            _st_ = set(getattr(L, "_pair_counts", {}))
            _vals_ = sorted(float(_M_[_ix_[a], _ix_[b]]) for a, b in _st_ if a in _ix_ and b in _ix_)
            _med_ = _vals_[min(len(_vals_) - 1, int(len(_vals_) * q_))] if _vals_ else 0.0   # q_: the quantile of stored pairs (median by default)
        if esc_ == "gated2":               # r180: the same type licence over pairs two apart (a _ b), read off the same tree
            _pc0_ = L._pair_counts
            _sk_ = {}
            for _s_, _ in train:
                for ab in zip(_s_, _s_[2:]):
                    _sk_[ab] = _sk_.get(ab, 0) + 1
            L._pair_counts = _sk_
            L.__dict__.pop("_type_licence", None)
            _w2_, _ix2_, _M2_ = L.type_licence(near=near_)
            L._pair_counts = _pc0_
            L.__dict__.pop("_type_licence", None)
            L.type_licence(near=near_)
            _st2_ = set(_sk_)
            _v2_ = sorted(float(_M2_[_ix2_[a], _ix2_[b]]) for a, b in _st2_ if a in _ix2_ and b in _ix2_)
            _med2_ = _v2_[min(len(_v2_) - 1, int(len(_v2_) * q_))] if _v2_ else 0.0

        def _ok2(l2, w):
            if esc_ != "gated2" or l2 == T3BOS or (l2, w) in _st2_:
                return True
            return l2 in _ix2_ and w in _ix2_ and float(_M2_[_ix2_[l2], _ix2_[w]]) >= _med2_ > 0

        def _ok(l1, w):
            if not pair_ or l1 == T3BOS or (l1, w) in _st_:
                return True
            return l1 in _ix_ and w in _ix_ and float(_M_[_ix_[l1], _ix_[w]]) >= _med_ > 0

        def _fills(l2, l1, *far):          # far: l3, l4, ... when the learner's window is wider (r161: 3; r184: up to 5)
            k = (far, l2, l1)
            if k not in _cz:
                q = {"l2": l2, "l1": l1, **{f"l{3 + i}": w for i, w in enumerate(far)}}
                a_, e_ = L.hier_fills(q, how=read_) if read_ else (L._gap_fills(q), L._gap_fills({**q, "r1": T3EOS}))
                vs, ws = [], []
                for v, m_ in a_.items():
                    if units_ == "words" and v[0] != "w":
                        continue
                    if not _ok(l1, _words(v)[0]):
                        continue
                    me = e_.get(v, 0.0) if not pool_ else min(e_.get(v, 0.0), m_)   # r197: with "pool" the end read has A + 1 factors, not nested in the A of m_
                    vs += [(v, True), (v, False)]
                    ws += [me, max(m_ - me, 0.0)]
                if escape_ == "n1" and units_ == "words" and not read_ and vs:   # r207: rescale to the history's missing mass
                    _wm_ = {}
                    for j_ in range(0, len(vs), 2):
                        _wm_[vs[j_][0][1]] = ws[j_] + ws[j_ + 1]
                    _fx_ = L.escape_split(q, _wm_, (lambda w: (not pair_) or l1 == T3BOS or (l1, w) in _st_))
                    ws = [x * _fx_.get(vs[j_][0][1], 1.0) for j_, x in enumerate(ws)]
                if esc_ and l1 != T3BOS and l1 in _ix_:
                    keep = [j for j, (v, _) in enumerate(vs) if (l1, _words(v)[0]) in _st_]
                    vs, ws = [vs[j] for j in keep], [ws[j] for j in keep]
                    succ = {b: c for (a, b), c in L._pair_counts.items() if a == l1}
                    d_, c_ = len(succ), sum(succ.values())
                    row = _M_[_ix_[l1]]
                    nov = [(w, float(row[_ix_[w]])) for w in _w_ if w not in succ and w not in (T3BOS, T3EOS)
                           and float(row[_ix_[w]]) > 0 and (esc_ == "licence" or (_ok(l1, w) and _ok2(l2, w)))]
                    zs, zn = sum(ws), sum(m for _, m in nov)
                    if zs > 0 and zn > 0 and d_ > 0:
                        e_mass = d_ / (d_ + c_)
                        ws = [x * (1 - e_mass) / zs for x in ws]
                        vs += [(("w", w), False) for w, _ in nov]
                        ws += [e_mass * m / zn for _, m in nov]
                _cz[k] = (vs, ws)
            return _cz[k]
        _yc = {}

        def _words(v):
            if v[0] == "w":
                return [v[1]]
            if v[1] not in _yc:
                _yc[v[1]] = yield_of(L.records[v[1]])
            return _yc[v[1]]
        if both_:
            _bk_ = "pivot" if both_ is True else both_
            _tri_ = L.stored_trigrams()
            _ends_ = {t[1] for t in _tri_ if t[2] == T3EOS}   # words memory saw end a sentence
            _W_ = max(getattr(L, "window", 2), 2)
            _bz = {}
            _f0 = _fills

            def _fills(l2, l1, *far, _x=T3BOS, _x2=T3BOS):   # the same fillers, novel trigrams kept only when both sides agree
                k = (far, l2, l1, _x, _x2)         # _x, _x2: the words before the window (l_W+1, l_W+2)
                if k not in _bz:
                    vs, ws = _f0(l2, l1, *far)
                    h = [l1, l2, *far]     # l1 .. lW of the filler
                    cand = sorted({_words(v)[0] for v, _ in vs})
                    nov = [w for w in cand if l1 != T3BOS and (l2, l1, w) not in _tri_]
                    piv_left = {f"l{i + 1}": x for i, x in enumerate(h[1:] + [_x])}   # the pivot l1 stands after l2 .. l_W+1
                    if _bk_ == "pairs":    # ablation: a new trigram only over a stored pair (the strict pair gate)
                        go = {(l1, w): (0.0 if (l1, w) in L._pair_counts else -1.0) for w in nov}
                    else:
                        go = L.pivot_support(piv_left, [l1], nov, kind=_bk_) if nov else {}
                    if reach_ >= 2 and nov and l2 != T3BOS:   # r192 v2: also l2 as the pivot, l1 its r1, the new word its r2
                        q2 = {**{f"l{i + 1}": x for i, x in enumerate(h[2:] + [_x, _x2])}, "r1": l1}
                        go2 = L.pivot_support(q2, [l2], nov, kind=_bk_, slot="r2")
                        go = {(l1, w): min(go.get((l1, w), 0.0), go2.get((l2, w), 0.0)) for w in nov}
                    endc = [w for w in cand if (l1, w, T3EOS) not in _tri_ and w in _ends_]
                    if _bk_ == "pairs":    # ablation: no right-side read; only stored pairs and the boundary words
                        end = {(w, T3EOS): 0.0 for w in endc}
                    else:
                        end = L.pivot_support({f"l{i + 1}": x for i, x in enumerate(h)}, endc, [T3EOS], kind=_bk_) if endc else {}
                    keep = []
                    for j, (v, d) in enumerate(vs):
                        w = _words(v)[0]
                        if go.get((l1, w), 0.0) < 0 or (l1 == T3BOS and (T3BOS, T3BOS, w) not in _tri_):
                            continue   # (a new start: only a word memory saw start a sentence, as the end)
                        if d and (l1, w, T3EOS) not in _tri_ and (w not in _ends_ or end[(w, T3EOS)] < 0):
                            continue
                        keep.append(j)
                    _bz[k] = ([vs[j] for j in keep], [ws[j] for j in keep])
                return _bz[k]

        def gen(p, _rng=g_rng):
            for _ in range(50):
                out, done, lens = [], False, []
                while not done and len(out) <= 8:
                    t_ = [T3BOS] * max(5, getattr(L, "window", 2)) + out
                    vs, ws = _fills(t_[-2], t_[-1], *[t_[-3 - i] for i in range(max(getattr(L, "window", 2), 2) - 2)],
                                    **({"_x": t_[-_W_ - 1] if len(t_) > _W_ else T3BOS,
                                        "_x2": t_[-_W_ - 2] if len(t_) > _W_ + 1 else T3BOS} if both_ else {}))
                    if not vs or sum(ws) <= 0:
                        break
                    if temp_ != 1.0:
                        ws = [x ** (1.0 / temp_) for x in ws]
                    (v, done) = _rng.choices(vs, weights=ws)[0]
                    out += _words(v)
                    lens.append(len(_words(v)))
                if done and 3 <= len(out) <= 8 and not (new_ and tuple(out) in seen):
                    if seg_:      # r194: the unit lengths of the sentence, for the chunk-junction pre-test
                        _SEG_[tuple(out)] = lens
                    return out
            return None

        blend_ = arm.pop("blend", None)    # r170: each word from w * chart + (1 - w) * context tree (as characters, r169)
        if blend_ is not None:
            # The chart's P(w at i | placed words, length n) (its posterior at i with the rest free, as _fill_chart)
            # carries the grammar state; the context tree's causal prediction (go on, or end at the last place)
            # the local context. n is drawn from the stored sentences' lengths (3..8). The gate applies to the blend.
            import numpy as _np
            CGb = ChartGen(L, **(chart or {}))
            _lens = [len(s) for s in seen if 3 <= len(s) <= 8]

            def _norm(d):
                t_ = sum(d.values())
                return {k: x / t_ for k, x in d.items()} if t_ > 0 else {}

            def _pchart(out, n, i):
                toks = list(out) + [None] * (n - len(out))
                mu = CGb._marginal(toks, [t is None for t in toks], None, CGb.start, i)
                if mu is None:
                    return {}
                pc = defaultdict(float)
                for k in _np.nonzero(mu > mu.max() * 1e-6)[0]:
                    c = CGb.emit.get(int(k))
                    if c:
                        tot = sum(c.values())
                        for w, x in c.items():
                            pc[w] += mu[k] * x / tot
                return _norm(pc)

            def gen(p, _rng=g_rng):
                for _ in range(50):
                    n = _rng.choice(_lens)
                    out = []
                    for i in range(n):
                        t_ = [T3BOS, T3BOS] + out
                        vs, ws = _fills(t_[-2], t_[-1])
                        last = i == n - 1
                        pk = _norm({v[1]: x for (v, d), x in zip(vs, ws) if v[0] == "w" and d == last})
                        pc = _pchart(out, n, i)
                        cand = {w: blend_ * pc.get(w, 0.0) + (1 - blend_) * pk.get(w, 0.0) for w in set(pc) | set(pk)}
                        cand = {w: x for w, x in cand.items() if x > 0 and _ok(t_[-1], w)}
                        if not cand:
                            break
                        words, wts = zip(*cand.items())
                        out.append(_rng.choices(words, weights=wts)[0])
                    if len(out) == n and not (new_ and tuple(out) in seen):
                        return out
                return None
            arm["blend"] = blend_
            if chart:
                arm["chart"] = chart
        arm.update({"causal": causal, "units": units_, **({"pair": pair_} if pair_ else {}), **({"esc": esc_} if esc_ else {}), **({"new": True} if new_ else {}), **({"q": q_} if q_ != 0.5 else {}), **({"near": near_} if near_ else {}), **({"sweep": sweep_} if sweep_ else {}), **({"temp": temp_} if temp_ != 1.0 else {}), **({"both": both_} if both_ else {}), **({"reach": reach_} if reach_ != 1 else {}), **({"repair": True} if repair_ else {}), **({"seg": True} if seg_ else {}), **({"read": read_} if read_ else {}), **({"pool": pool_} if pool_ else {})})
        if sweep_:
            _tw = {}

            def _two(l2, l1, r1, r2):
                k = (l2, l1, r1, r2)
                if k not in _tw:
                    f_ = L._gap_fills({"l2": l2, "l1": l1, "r1": r1, "r2": r2})
                    d_ = {v[1]: x for v, x in f_.items() if v[0] == "w" and x > 0 and _ok(l1, v[1]) and (r1 == T3EOS or _ok(v[1], r1))}
                    _tw[k] = (list(d_), list(d_.values()))
                return _tw[k]

            _draft = gen

            def gen(p, _rng=g_rng):
                for _ in range(20):
                    out = _draft(p, _rng)
                    if out is None:
                        return None
                    out = list(out)
                    for _s in range(sweep_):
                        for i in range(len(out)):
                            t_ = [T3BOS, T3BOS] + out + [T3EOS, T3EOS]
                            ws_, xs_ = _two(t_[i], t_[i + 1], t_[i + 3], t_[i + 4])
                            if ws_:
                                out[i] = _rng.choices(ws_, weights=xs_)[0]
                    if not (new_ and tuple(out) in seen):
                        return out
                return None
        if repair_:                        # r193: two-sided Gibbs repair of each drawn sentence (Trellis3.gibbs_repair)
            _draft3 = gen

            def gen(p, _rng=g_rng):
                out = _draft3(p, _rng)
                if out is None:
                    return None
                o2, _ = L.gibbs_repair(out, _rng, temp=temp_)
                return out if (new_ and tuple(o2) in seen) else o2
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
    if causal is not None and edge_ == "exact":     # r204: later arms of this run read the plain boundary again
        L.edge_exact = False
        arm["edge"] = "exact"
    if causal is not None and escape_ == "n1":      # r207: later arms of this run read without the escape again
        L.escape_mass = False
        arm["escape"] = "n1"
    ok = [(p, g) for p, g in zip(PP, G) if g]
    if usw is not None:
        extra = {"chunk for chunk": round(sum(a for a, _ in _us_log) / max(len(_us_log), 1), 3),
                 "unit types": len(_us[usw]["order"]),
                 "mean ratio": round(sum(b for _, b in _us_log) / max(len(_us_log), 1), 3)}
        _us_log.clear()
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
                      **extra,
                      "novel tag pattern attested": m([a for a, b in zip(pat, nov) if b]),
                      "within share": None if w is None or ws is None else round((w - ws) / (w_real - w_shuf), 3),
                      "examples": [" ".join(p) + "  ->  " + " ".join(g) for p, g in ok[:12]],
                      "novel examples": [(" ".join(g), a) for (_, g), a, b in zip(ok, pat, nov) if b][:16]}), flush=True)
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
        with open(os.path.join(DUMP, f"{tag}_s{seed}_n{n}_arm{ai}.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"arm": {"cond": cond, **({"critic": critic} if critic else {}), **({"critic_mode": critic_mode} if critic_mode != "mean" else {}), **arm}}) + "\n")
            for (p, g), a, b in zip(ok, pat, nov):
                fh.write(json.dumps({"prompt": " ".join(p), "gen": " ".join(g), "pattern": a, "novel": b,
                                     **({"units": _SEG_.get(tuple(g))} if _SEG_ and arm.get("seg") else {})}) + "\n")

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
