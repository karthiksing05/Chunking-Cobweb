"""How TRELLIS v2 evolves as it takes in more data (evaluation only; the learner
sees raw tokens, gold labels only score).

Corpora: a synthetic grammar condition (small, med, large, c_agree, ...; scored
by the generating CFG and its gold trees) or ``stories`` (TinyStories, 3-8 words
from the ``vocab`` most frequent words; scored by NLTK coarse POS tags).

At each checkpoint:
  GENERATION   300 samples (generate_tree): grammatical (stories: tag sequence
               attested and all trigrams attested), novel against what has been
               seen, distinct, mean length, 8 examples.
  STRUCTURE    super-structure reuse. A parse's skeleton is its analysis with words
               abstracted and each chunk named by its content concept at one
               information band (coarse: first concept >= 2 bits; fine: >= 6 bits).
               Over the last <= 300 training sentences (parsed now) and up to 200
               held-out ones: distinct skeletons, share covered by the 10 most
               used, held-out sentences whose skeleton occurs in training, and
               consistency: within each skeleton, the share of its sentences with
               its majority tag sequence (weighted). Generated sentences: share
               whose skeleton (parsed back) occurs in training.
  PROBES       five fixed held-out sentences, bracketed, each chunk labelled with
               its coarse concept.
  CATEGORIES   word-category accuracy (nearest word by learned name has the same
               gold tag).

Usage: t3_evolve.py <cond|stories> <cfg json> <seed> <checkpoints> [tag] [fill] [vocab]
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
from trellis2.trellis3 import Elem, Trellis3  # noqa: E402

cond, cfg, seed = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
checks = [int(x) for x in sys.argv[4].split(",")]
tag = sys.argv[5] if len(sys.argv) > 5 else ""
fill = sys.argv[6] if len(sys.argv) > 6 else "both"
fill = {"none": False, "name": True}.get(fill, fill)
vocab = int(sys.argv[7]) if len(sys.argv) > 7 else 250

if cond == "stories":
    import nltk
    from trellis2.stories import read_sentences, simple_sentences
    corpus = simple_sentences(read_sentences(), vocab_size=vocab)
    random.Random(seed).shuffle(corpus)
    train, test = corpus[:max(checks)], corpus[max(checks):max(checks) + 200]
    coarse = lambda t: "IN" if t == "TO" else t[:2]
    tag_cache = {}

    def tags(s):
        k = tuple(s)
        if k not in tag_cache:
            tag_cache[k] = tuple(coarse(t) for _, t in nltk.pos_tag(list(s)))
        return tag_cache[k]
    patterns = {tags(s) for s in corpus[:20000]}
    trigrams = set()
    for s in corpus:
        p = ["<s>", "<s>"] + list(s) + ["</s>"]
        trigrams.update(zip(p, p[1:], p[2:]))

    def grammatical(s):
        p = ["<s>", "<s>"] + list(s) + ["</s>"]
        return tags(s) in patterns and all(t in trigrams for t in zip(p, p[1:], p[2:]))
    gc = defaultdict(Counter)
    for s in corpus[:20000]:
        for w, t in zip(s, tags(s)):
            gc[w][t] += 1
    gold = {w: c.most_common(1)[0][0] for w, c in gc.items()}
else:
    from trellis2 import load_corpus, v1_split
    from trellis2.data import CONDITIONS, default_data_root, target_grammar
    from trellis2.evaluation import CFG
    tr, te = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    g = CFG(target_grammar(cond))
    pre = {}
    gc = defaultdict(Counter)
    for e in tr + te:
        lab = g.gold_labels(e.tokens, e.tree)
        seq = tuple(lab.get((i, i + 1), "?") for i in range(len(e.tokens)))
        pre[tuple(e.tokens)] = seq
        for w, t in zip(e.tokens, seq):
            gc[w][t] += 1
    gold = {w: c.most_common(1)[0][0] for w, c in gc.items()}
    train, test = [e.tokens for e in tr], [e.tokens for e in te][:200]

    def tags(s):
        k = tuple(s)
        if k not in pre:
            r = g.parse_labels(s) if hasattr(g, "parse_labels") else None
            pre[k] = tuple(r) if r else tuple("?" for _ in s)
        return pre[k]

    def grammatical(s):
        return g.recognizes(list(s))

L = Trellis3(seed=seed, **cfg)


def band_node(path, bits):
    total = max(path[0].count, 1e-9)
    for nd in path[1:]:
        if -math.log2(max(nd.count, 1e-9) / total) >= bits:
            return nd
    return path[-1]


def skeleton(tokens, bits, rules=None, spans=None):
    """The analysis with words abstracted, chunks named at one band; optionally
    collects its rules (concept -> left right) and each concept's span tags."""
    r = L.parse(tokens)
    n = len(tokens)
    if n < 2:
        return "w"
    split = {(i, j): k for (i, k, j) in r["chosen"]}
    comps = r["comps"]
    tg = tags(tokens) if spans is not None else None

    def label(i, j):
        if j - i == 1:
            return "w"
        k = split[(i, j)]
        return f"c{band_node(L.cnt.categorize(comps[(i, k, j)].cnt), bits).id}"

    def walk(i, j):
        if j - i == 1:
            return "w"
        k = split[(i, j)]
        c = label(i, j)
        if rules is not None:
            rules.append(f"{c} -> {label(i, k)} {label(k, j)}")
        if spans is not None:
            spans[c][tg[i:j]] += 1
        return f"{c}({walk(i, k)} {walk(k, j)})"
    return walk(0, n)


def bracket(tokens):
    r = L.parse(tokens)
    n = len(tokens)
    if n < 2:
        return " ".join(tokens)
    split = {(i, j): k for (i, k, j) in r["chosen"]}
    comps = r["comps"]

    def walk(i, j):
        if j - i == 1:
            return tokens[i]
        k = split[(i, j)]
        nd = band_node(L.cnt.categorize(comps[(i, k, j)].cnt), 2)
        return f"[c{nd.id} {walk(i, k)} {walk(k, j)}]"
    return walk(0, n)


def as_bag(v):
    if isinstance(v, dict):
        t = sum(v.values()) or 1.0
        return {u: p / t for u, p in v.items()}
    if isinstance(v, list):
        real = [u for u in v if u and u != "-"]
        return {u: real.count(u) / len(real) for u in set(real)} if real else {}
    return {}


def category_accuracy(words):
    L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
    nm = {w: as_bag(L._ctx_name(Elem((0, 1), token=w)) or {}) for w in words}
    ov = lambda a, b: sum(min(p, b.get(u, 0.0)) for u, p in a.items())
    ok = []
    for w in words:
        best = max((x for x in words if x != w), key=lambda x: ov(nm[w], nm[x]), default=None)
        ok.append(best is not None and gold.get(best) == gold.get(w) and ov(nm[w], nm[best]) > 0)
    return round(sum(ok) / max(len(ok), 1), 3)


probes = test[:5]
done, t0 = 0, time.time()
seen_words = Counter()
for n in checks:
    for s in train[done:n]:
        L.learn(s)
        seen_words.update(s)
    done = n
    learn_s = time.time() - t0
    seen = {tuple(s) for s in train[:n]}
    rng = random.Random(seed)
    ix = L.concept_index()
    S = [s for s in (L.generate_tree(rng, index=ix, level=LEVEL, choose="leaf", fill_name=fill) for _ in range(300)) if s]
    gram = [grammatical(s) for s in S]
    nov = [tuple(s) not in seen for s in S]
    row = {"tag": tag, "cond": cond, "seed": seed, "experiences": n, "learn s": round(learn_s),
           "GENERATION": {"generated": len(S),
                          "grammatical": round(sum(gram) / max(len(S), 1), 3),
                          "novel": round(sum(nov) / max(len(S), 1), 3),
                          "grammatical and novel": round(sum(a and b for a, b in zip(gram, nov)) / max(len(S), 1), 3),
                          "distinct": round(len(set(map(tuple, S))) / max(len(S), 1), 3),
                          "mean length": round(sum(map(len, S)) / max(len(S), 1), 2),
                          "examples": [" ".join(s) + (" ✓" if a else " ✗") + ("" if b else " (seen)")
                                       for s, a, b in zip(S[:8], gram, nov)]},
           "STRUCTURE": {}}
    recent = train[max(0, n - 300):n]
    for name, bits in (("coarse", 2), ("fine", 6)):
        rules_tr, spans = [], defaultdict(Counter)
        sk_tr = [skeleton(s, bits, rules_tr, spans) for s in recent]
        cnt, rc = Counter(sk_tr), Counter(rules_tr)
        rules_te, rules_gen = [], []
        sk_te = [skeleton(s, bits, rules_te) for s in test]
        sk_gen = [skeleton(s, bits, rules_gen) for s in S[:100]]
        occ = sum(sum(c.values()) for c in spans.values())
        row["STRUCTURE"][name] = {
            "sentence skeletons: distinct / sentences": [len(cnt), len(recent)],
            "held-out skeleton seen in training": round(sum(k in cnt for k in sk_te) / max(len(sk_te), 1), 3),
            "rules: distinct / uses": [len(rc), len(rules_tr)],
            "rules: top-20 share of uses": round(sum(c for _, c in rc.most_common(20)) / max(len(rules_tr), 1), 3),
            "held-out rules seen in training": round(sum(x in rc for x in rules_te) / max(len(rules_te), 1), 3),
            "generated rules seen in training": round(sum(x in rc for x in rules_gen) / max(len(rules_gen), 1), 3),
            "concepts used": len(spans),
            "concept consistency (majority tag sequence of its spans)": round(
                sum(c.most_common(1)[0][1] for c in spans.values()) / max(occ, 1), 3),
            "top rules": rc.most_common(5)}
    row["PROBES"] = [bracket(s) for s in probes]
    row["CATEGORIES"] = {"word category accuracy": category_accuracy([w for w in seen_words if w in gold]),
                         "word types": len(seen_words)}
    print(json.dumps(row), flush=True)
