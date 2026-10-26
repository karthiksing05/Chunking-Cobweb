"""Generation from partial prompts on TinyStories (docs/PROMPTING.md).

Prompts from held-out sentences (heldout_s13.json): prefix-1 [w1 *], prefix-2
[w1 w2 *], infill [w1 * w_{n-1} w_n]. Methods:

  reject     unprompted generate_tree, the first of up to 300 samples matching the template
  pattern    generate_tree(pattern=...): fillers drawn without replacement, backtracking
  pattern+ctx  as pattern, a slot's fillers from the context concept its generated
             left neighbours (and stored right neighbours) sort to (ctx_fill)
  gap        the missing piece only: a committed instance drawn by its posterior of
             standing in the gap's surroundings (Trellis3._gap_fills), expanded
  gap+critic 8 gap completions, the one the parser recognises best (logZ per composite)
  sibling+critic  (prefixes) the continuation generated as the right part of a committed
             composite whose left part matches a chunk on the prompt's right edge
  and unprompted samples with and without ctx_fill (grammar reference)

Measures: success (a sentence matching the template), trigrams attested (every padded
trigram occurs in the corpus), tag pattern attested (NLTK coarse tags; diagnostic),
novel (not a training sentence), distinct, and coherence: mean PMI (sentence
co-occurrence over the corpus) between the prompt's words and the generated words,
minus the same with prompts and outputs paired at random.

Usage: t3_prompt.py <cfg json> <seed> <n train> <tag> [fill] [k prompts]
"""
import json
import math
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
import nltk  # noqa: E402
from trellis2.stories import heldout_path, load_corpus  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
fill = sys.argv[5] if len(sys.argv) > 5 else "both+peer"
fill = {"none": False, "name": True}.get(fill, fill)
K = int(sys.argv[6]) if len(sys.argv) > 6 else 100
corpus = load_corpus(vocab_size=250)
random.Random(seed).shuffle(corpus)
held = json.load(open(heldout_path(13), encoding="utf-8"))
held_set = {tuple(s) for s in held}
train = [s for s in corpus[:n] if tuple(s) not in held_set]
train_set = {tuple(s) for s in train}


def coarse(t):
    return "PR" if t == "PRP$" else "WP" if t == "WP$" else "IN" if t == "TO" else t[:2]


trigrams, patterns = set(), set()
docf, pairf = Counter(), Counter()
for s in corpus:
    p = ["<s>", "<s>"] + list(s) + ["</s>"]
    trigrams.update(zip(p, p[1:], p[2:]))
    u = sorted(set(s))
    docf.update(u)
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            pairf[(u[i], u[j])] += 1
for s in corpus[:20000]:
    patterns.add(tuple(coarse(t) for _, t in nltk.pos_tag(s)))
N = len(corpus)


def pmi(a, b):
    if a == b:
        return None
    k = pairf.get((a, b) if a < b else (b, a), 0)
    return math.log((k + 0.1) * N / (docf[a] * docf[b] + 1e-9))


def coherence(prompt_words, out_words):
    vals = [pmi(a, b) for a in set(prompt_words) for b in set(out_words) if a in docf and b in docf]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


L = Trellis3(seed=seed, **cfg)
t0 = time.time()
for s in train:
    L.learn(s)
learn_s = time.time() - t0
ix = L.concept_index()
rng = random.Random(seed)


def templates(s):
    out = {"prefix-1": [s[0], "*"], "prefix-2": s[:2] + ["*"]}
    if len(s) >= 5:
        out["infill"] = [s[0], "*"] + s[-2:]
    return out


def matches(pat, toks):
    st = {0}
    for t in toks:
        nx = set()
        for q in st:
            if q < len(pat):
                if pat[q] == "*":
                    nx.update((q, q + 1))
                elif pat[q] == "?" or pat[q] == t:
                    nx.add(q + 1)
        st = nx
    return len(pat) in st


def gen(pat, method):
    if method == "reject":
        for _ in range(300):
            g = L.generate_tree(rng, index=ix, choose="leaf", fill_name=fill)
            if g and matches(pat, g):
                return g
        return None
    if method in ("gap", "gap+critic", "sibling+critic"):
        i = pat.index("*")
        k = 1 if method == "gap" else 8
        mode = "sibling" if method.startswith("sibling") else "window"
        if mode == "sibling" and pat[i + 1:]:
            return None
        cands = [g for g in (L.generate_tree(rng, index=ix, choose="leaf", fill_name=fill, gap=(pat[:i], pat[i + 1:]),
                                             gap_mode=mode) for _ in range(k)) if g]
        if len(cands) <= 1:
            return cands[0] if cands else None
        # memory as critic: the completion the parser recognises best (log total weight
        # over analyses per composite, so length does not decide)
        return max(cands, key=lambda g: L.parse(g)["logZ"] / max(len(g) - 1, 1))
    return L.generate_tree(rng, index=ix, choose="leaf", fill_name=fill, pattern=pat,
                           ctx_fill=(method == "pattern+ctx"))


def score(outs):
    ok = [g for g in outs if g]
    m = lambda v: round(sum(v) / len(v), 3) if v else None
    tri = []
    for g in ok:
        p = ["<s>", "<s>"] + g + ["</s>"]
        tri.append(all(t in trigrams for t in zip(p, p[1:], p[2:])))
    pat = [tuple(coarse(t) for _, t in nltk.pos_tag(g)) in patterns for g in ok]
    nov = [tuple(g) not in train_set for g in ok]
    return {"success": m([g is not None for g in outs]), "trigrams attested": m(tri), "tag pattern attested": m(pat),
            "novel": m(nov), "attested + novel": m([a and b for a, b in zip(tri, nov)]),
            "distinct": round(len({tuple(g) for g in ok}) / max(len(ok), 1), 3)}


res = {"tag": tag, "seed": seed, "n": n, "cfg": cfg, "fill": fill, "learn s": round(learn_s)}
prompts = [s for s in held if len(s) >= 4][:K]
for kind in ("prefix-1", "prefix-2", "infill"):
    rows = [(s, templates(s)[kind]) for s in prompts if kind in templates(s)]
    for method in ("reject", "pattern", "pattern+ctx", "gap", "gap+critic", "sibling+critic"):
        if method == "sibling+critic" and kind == "infill":
            continue
        t1 = time.time()
        outs = [gen(p, method) for _, p in rows]
        sc = score(outs)
        fixed = [[w for w in p if w not in ("*", "?")] for _, p in rows]
        newp = [(f, [w for w in g if w not in f]) for f, g in zip(fixed, outs) if g]
        coh = [coherence(f, o) for f, o in newp]
        sh = [o for _, o in newp]
        random.Random(1).shuffle(sh)
        cohr = [coherence(f, o) for (f, _), o in zip(newp, sh)]
        c1 = [v for v in coh if v is not None]
        c0 = [v for v in cohr if v is not None]
        sc["coherence (PMI)"] = round(sum(c1) / max(len(c1), 1), 3)
        sc["coherence gain over shuffled"] = round(sum(c1) / max(len(c1), 1) - sum(c0) / max(len(c0), 1), 3)
        sc["s per prompt"] = round((time.time() - t1) / max(len(rows), 1), 2)
        sc["examples"] = [" ".join(p) + "  ->  " + (" ".join(g) if g else "FAIL") for (_, p), g in list(zip(rows, outs))[:8]]
        res[f"{kind} / {method}"] = sc
        print(json.dumps({f"{kind} / {method}": sc}), flush=True)
for method, cf in (("unprompted", False), ("unprompted+ctx", True)):
    outs = [L.generate_tree(rng, index=ix, choose="leaf", fill_name=fill, ctx_fill=cf)
            for _ in range(200)]
    sc = score(outs)
    sc["examples"] = [" ".join(g) for g in outs[:8] if g]
    res[method] = sc
    print(json.dumps({method: sc}), flush=True)
print(json.dumps(res))
