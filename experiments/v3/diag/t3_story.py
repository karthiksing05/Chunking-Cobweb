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
for st in train_st:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, (i > 0 and keep(st[i - 1])) if FOLLOW == "adjacent" else any(keep(t) for t in st[:i])))
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

L = Trellis3(seed=seed, **cfg)
t0 = time.time()
for s, fol in train:
    L.learn(s, follows=fol)
learn_s = time.time() - t0
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
    if chart is not None:
        CG = ChartGen(L, **chart)
        print(json.dumps({"tag": tag, "seed": seed, "chart": chart, **CG.stats}), flush=True)
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
