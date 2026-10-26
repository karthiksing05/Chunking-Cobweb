"""Is TinyStories parsing consistent, and does it reflect what was learned? No gold.

On a fixed held-out set (experiments/v3/results/silver/heldout_s13.json, shared by
every seed so parses can be compared across seeds), at checkpoints of the stream:

  phrase consistency   a word string (2-4 words) occurring 3+ times in the held-out
                       set is bracketed the same way everywhere: mean over strings of
                       the majority share (constituent or not), weighted by occurrences
  substitution         one word swapped for its nearest other word by learned name
                       (same category) vs for a word far from it by learned name
                       (control): bracket F1 between the two parses
  stability            bracket F1 with the parses at the previous checkpoint
  chunk reuse          share of held-out brackets whose word string memory chose as a
                       chunk in training; and of held-out occurrences of such strings,
                       the share bracketed
Parses are saved (experiments/v3/results/consistency/<tag>_s<seed>.json) for
cross-seed agreement (t3_parse_agree.py).

Usage: t3_parse_consistency.py <cfg json> <seed> <checkpoints> <tag>
"""
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from trellis2.stories import heldout_path, load_corpus  # noqa: E402
from trellis2.trellis3 import Elem, Trellis3  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
corpus = load_corpus(vocab_size=250)
random.Random(seed).shuffle(corpus)
held = json.load(open(heldout_path(13), encoding="utf-8"))
held_set = {tuple(s) for s in held}
L = Trellis3(seed=seed, **cfg)
rng = random.Random(seed + 99)


def inner(n, spans):
    return {tuple(s) for s in spans if s[1] - s[0] >= 2 and tuple(s) != (0, n)}


def f1(a, b):
    if not a and not b:
        return 1.0
    m = len(a & b)
    return 2 * m / max(len(a) + len(b), 1)


def brackets(toks):
    return inner(len(toks), L.parse(toks)["tree"].brackets())


def names(words):
    L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
    out = {}
    for w in words:
        v = L._ctx_name(Elem((0, 1), token=w)) or {}
        if isinstance(v, list):
            real = [u for u in v if u and u != "-"]
            v = {u: real.count(u) / len(real) for u in set(real)} if real else {}
        t = sum(v.values()) or 1.0
        out[w] = {u: p / t for u, p in v.items()}
    return out


def overlap(a, b):
    return sum(min(p, b.get(u, 0.0)) for u, p in a.items())


chunks = Counter()          # word strings memory chose as chunks in training
seen = Counter()
prev, saved, t0, done = None, {}, time.time(), 0
for c in checks:
    for s in corpus[done:c]:
        if tuple(s) in held_set:
            continue
        tree = L.learn(s)
        seen.update(s)
        for (i, j) in tree.brackets():
            if j - i >= 2:
                chunks[tuple(s[i:j])] += 1
    done = c
    learn_s = time.time() - t0
    parses = [brackets(s) for s in held]
    # phrase consistency
    occ = defaultdict(list)
    for s, br in zip(held, parses):
        n = len(s)
        for k in (2, 3, 4):
            for i in range(0, n - k + 1):
                if (i, i + k) == (0, n):
                    continue
                occ[tuple(s[i:i + k])].append((i, i + k) in br)
    rep = {g: v for g, v in occ.items() if len(v) >= 3}
    tot = sum(len(v) for v in rep.values())
    cons = sum(max(sum(v), len(v) - sum(v)) for v in rep.values()) / max(tot, 1)
    const_rate = sum(sum(v) for v in rep.values()) / max(tot, 1)
    # substitution invariance
    vocab = [w for w, m in seen.items() if m >= 3]
    nm = names(vocab)
    near, far = {}, {}
    for w in vocab:
        others = sorted((overlap(nm[w], nm[x]), x) for x in vocab if x != w)
        if others and others[-1][0] > 0:
            near[w] = others[-1][1]
            far[w] = rng.choice([x for _, x in others[:max(1, len(others) // 4)]])
    sub_near, sub_far = [], []
    for s, br in zip(held[:200], parses[:200]):
        pos = [i for i, w in enumerate(s) if w in near]
        if not pos:
            continue
        i = rng.choice(pos)
        for sub, out in ((near, sub_near), (far, sub_far)):
            t = list(s)
            t[i] = sub[s[i]]
            out.append(f1(br, brackets(t)))
    # stability and chunk reuse
    stab = sum(f1(a, b) for a, b in zip(parses, prev)) / len(parses) if prev else None
    allb = [(s, b) for s, br in zip(held, parses) for b in br]
    reuse = sum(chunks[tuple(s[i:j])] > 0 for s, (i, j) in allb) / max(len(allb), 1)
    hit, cov = 0, 0
    for s, br in zip(held, parses):
        n = len(s)
        for i in range(n):
            for j in range(i + 2, n + 1):
                if (i, j) != (0, n) and chunks[tuple(s[i:j])] >= 3:
                    cov += 1
                    hit += (i, j) in br
    ex = [" ".join(s) + "  ->  " + L.parse(s)["tree"].to_string(s) for s in held[:6]]
    row = {"tag": tag, "seed": seed, "checkpoint": c, "learn s": round(learn_s),
           "phrase consistency": round(cons, 3), "repeated strings": len(rep), "constituent rate": round(const_rate, 3),
           "substitution F1 near / far": [round(sum(sub_near) / max(len(sub_near), 1), 3), round(sum(sub_far) / max(len(sub_far), 1), 3)],
           "stability vs previous checkpoint": stab and round(stab, 3),
           "brackets that are training chunks": round(reuse, 3),
           "training chunks (3+) bracketed": round(hit / max(cov, 1), 3),
           "examples": ex}
    print(json.dumps(row), flush=True)
    saved[c] = [sorted(b) for b in parses]
    prev = parses
os.makedirs("experiments/v3/results/consistency", exist_ok=True)
json.dump(saved, open(f"experiments/v3/results/consistency/{tag}_s{seed}.json", "w", encoding="utf-8"))
