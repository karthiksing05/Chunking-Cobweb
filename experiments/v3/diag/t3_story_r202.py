"""Round 202: story generation through the parse (docs: experiments/v3/results/story_gen/r202_parse_gen.md).

A trimmed copy of t3_story.py (left untouched: other rounds' jobs read it): the same stream, prompts, references,
learner and dump format, and ONLY the causal words arm. Lines up to the learner are copied verbatim, so a plain arm
reproduces t3_story.py's dump byte for byte (checked against r200-n250-w5, seed 13). New, opt-in arm key:

  "gen": "parse"   each step: M = 16 proposals from the plain arm's window read (pair gate, temperature as usual),
                   reweighted by the learner's inside score of the extended prefix (parse_gen.PrefixChart: right
                   edge open to go on, closed to end), one drawn by sampling-importance-resampling
                   (parse_gen.parse_step). "M": k changes the proposal count.

Usage: t3_story_r202.py <cfg json> <seed> <n> <tag> <arms> [adjacent|story]   (as t3_story.py)
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



def grams(s, k):
    t = ["<s>"] + list(s) + ["</s>"]
    return [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]


have = {k: {g for s, _ in train for g in grams(s, k)} for k in (2, 3)}
attested = lambda s, k: all(g in have[k] for g in grams(s, k))

L = Trellis3(seed=seed, **cfg)
t0 = time.time()
EXP = []                                       # each experience's tokens and its records' range in memory
for s, fol in train:
    a0 = len(L.records)
    L.learn(s, follows=fol)
    EXP.append((list(s), a0, len(L.records)))
learn_s = time.time() - t0


def yield_of(e):
    return [e.token] if e.token is not None else yield_of(e.left) + yield_of(e.right)


from trellis2.parse_gen import PrefixChart, parse_step  # noqa: E402
L._quiet = True
DUMP = os.environ.get("STORY_DUMP")            # a directory: every generation of every arm, as JSON lines
if DUMP:                                         # judge calibration: held-out sentences, their words in random order, content words swapped by tag
    os.makedirs(DUMP, exist_ok=True)
    ref = random.Random(seed + 21).sample(held, min(300, len(held)))
    with open(os.path.join(DUMP, f"{tag}_s{seed}_n{n}_refs.jsonl"), "w", encoding="utf-8") as fh:
        prng = random.Random(seed + 23)
        for s, t in zip(ref, shuffled(ref, random.Random(seed + 22))):
            fh.write(json.dumps({"real": " ".join(s), "word order scrambled": " ".join(prng.sample(s, len(s))),
                                 "content swapped": " ".join(t)}) + "\n")
PC = None
for ai, arm in enumerate(arms):
    arm = dict(arm)
    cond = arm.pop("cond", False)
    g_rng = random.Random(seed + 7)
    t1 = time.time()
    PP = [p for p in P for _ in range(DRAWS)]
    causal = arm.pop("causal", None)
    if causal is None:
        raise SystemExit("t3_story_r202.py runs the causal arm only")
    extra = {}
    from trellis2.trellis3 import BOS as T3BOS, EOS as T3EOS
    units_ = arm.pop("units", "both")
    new_ = arm.pop("new", False)       # never return a stored sentence (redraw)
    q_ = float(arm.pop("q", 0.5))
    near_ = arm.pop("near", None)
    temp_ = float(arm.pop("temp", 1.0))
    gen_ = arm.pop("gen", None)        # r202: "parse" = through the parse (parse_gen.parse_step)
    M_ = int(arm.pop("M", 16))
    pair_ = arm.pop("pair", None)      # "gate": as t3_story.py (type licence at or above the median of stored pairs)
    if arm:
        raise SystemExit(f"t3_story_r202.py: unsupported arm keys {sorted(arm)}")
    if gen_ not in (None, "parse", "close"):
        raise SystemExit(f"t3_story_r202.py: unknown gen {gen_}")
    _cz = {}
    if pair_:
        _w_, _ix_, _M_ = L.type_licence(near=near_)
        _st_ = set(getattr(L, "_pair_counts", {}))
        _vals_ = sorted(float(_M_[_ix_[a], _ix_[b]]) for a, b in _st_ if a in _ix_ and b in _ix_)
        _med_ = _vals_[min(len(_vals_) - 1, int(len(_vals_) * q_))] if _vals_ else 0.0

    def _ok(l1, w):
        if not pair_ or l1 == T3BOS or (l1, w) in _st_:
            return True
        return l1 in _ix_ and w in _ix_ and float(_M_[_ix_[l1], _ix_[w]]) >= _med_ > 0

    _yc = {}

    def _words(v):
        if v[0] == "w":
            return [v[1]]
        if v[1] not in _yc:
            _yc[v[1]] = yield_of(L.records[v[1]])
        return _yc[v[1]]

    def _fills(l2, l1, *far):          # as t3_story.py (no esc / read / pool / both)
        k = (far, l2, l1)
        if k not in _cz:
            q = {"l2": l2, "l1": l1, **{f"l{3 + i}": w for i, w in enumerate(far)}}
            a_, e_ = (L._gap_fills(q), L._gap_fills({**q, "r1": T3EOS}))
            vs, ws = [], []
            for v, m_ in a_.items():
                if units_ == "words" and v[0] != "w":
                    continue
                if not _ok(l1, _words(v)[0]):
                    continue
                me = e_.get(v, 0.0)
                vs += [(v, True), (v, False)]
                ws += [me, max(m_ - me, 0.0)]
            _cz[k] = (vs, ws)
        return _cz[k]

    pstats = {}
    if gen_ in ("parse", "close") and PC is None:
        PC = PrefixChart(L)

    def gen(p, _rng=g_rng):
        for _ in range(50):
            out, done = [], False
            while not done and len(out) <= 8:
                t_ = [T3BOS] * max(5, getattr(L, "window", 2)) + out
                vs, ws = _fills(t_[-2], t_[-1], *[t_[-3 - i] for i in range(max(getattr(L, "window", 2), 2) - 2)])
                if not vs or sum(ws) <= 0:
                    break
                if temp_ != 1.0:
                    ws = [x ** (1.0 / temp_) for x in ws]
                if gen_ in ("parse", "close"):
                    (v, done) = parse_step(PC, out, vs, ws, _words, _rng, temp=temp_, M=M_, stats=pstats, mode=gen_)
                else:
                    (v, done) = _rng.choices(vs, weights=ws)[0]
                out += _words(v)
            if done and 3 <= len(out) <= 8 and not (new_ and tuple(out) in seen):
                return out
        return None

    arm.update({"causal": causal, "units": units_, **({"pair": pair_} if pair_ else {}), **({"new": True} if new_ else {}), **({"q": q_} if q_ != 0.5 else {}), **({"near": near_} if near_ else {}), **({"temp": temp_} if temp_ != 1.0 else {}), **({"gen": gen_} if gen_ else {}), **({"M": M_} if gen_ and M_ != 16 else {})})
    G = [gen(p) for p in PP]
    ok = [(p, g) for p, g in zip(PP, G) if g]
    if gen_ in ("parse", "close"):
        st_ = max(pstats.get("steps", 0), 1)
        extra = {"parse steps": pstats.get("steps", 0), "distinct candidates per step": round(pstats.get("distinct", 0) / st_, 2),
                 "parse moved the draw": round(pstats.get("moved", 0) / st_, 3),
                 "mean logZ spread": round(pstats.get("spread", 0.0) / st_, 2),
                 "charts": PC.calls, "composites built": PC.built, "memo": len(PC.memo)}
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
                      "every bigram stored": m([attested(g, 2) for g in S]), "every trigram stored": m([attested(g, 3) for g in S]),
                      **extra,
                      "novel tag pattern attested": m([a for a, b in zip(pat, nov) if b]),
                      "within share": None if w is None or ws is None else round((w - ws) / (w_real - w_shuf), 3),
                      "examples": [" ".join(p) + "  ->  " + " ".join(g) for p, g in ok[:12]],
                      "novel examples": [(" ".join(g), a) for (_, g), a, b in zip(ok, pat, nov) if b][:16]}), flush=True)
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
        with open(os.path.join(DUMP, f"{tag}_s{seed}_n{n}_arm{ai}.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"arm": {"cond": cond, **arm}}) + "\n")
            for (p, g), a, b in zip(ok, pat, nov):
                fh.write(json.dumps({"prompt": " ".join(p), "gen": " ".join(g), "pattern": a, "novel": b}) + "\n")
