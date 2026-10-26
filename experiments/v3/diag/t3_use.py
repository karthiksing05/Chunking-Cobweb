"""Repair on use (docs/TEMPORAL_INTERDEP.md, round 43): is the memory that TRELLIS
actually uses current? TinyStories, at checkpoints of the stream:

  use phase    parse 200 held-out sentences, 100 unprompted samples and 100 gap
               completions (prefix-2 prompts from held-out sentences). Reached
               records: those under the concept every sort ends at (a leaf: its
               records; else its leaf children's), and every record generation draws.
  reached      among reached records (each counted once): stale (its stored
               description differs from its current one) and misplaced (the
               band-level concepts above its leaf differ from where its CURRENT
               description sorts, at weight 1), per hierarchy; also weighted by how
               often the use phase reached it
  whole tree   misplaced at bands (stored description) on 1500 random records, as
               in t3_stories.py
At the end: learning time and generation (300 samples: trigrams attested, novel).

Usage: t3_use.py <cfg json> <seed> <checkpoints> <tag>
"""
import json
import math
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
from trellis2.stories import heldout_path, load_corpus  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
corpus = load_corpus(vocab_size=250)
random.Random(seed).shuffle(corpus)
held = json.load(open(heldout_path(13), encoding="utf-8"))[:200]
held_set = {tuple(s) for s in held}
train = [s for s in corpus if tuple(s) not in held_set][:max(checks)]
trigrams = set()
for s in corpus:
    p = ["<s>", "<s>"] + list(s) + ["</s>"]
    trigrams.update(zip(p, p[1:], p[2:]))
L = Trellis3(seed=seed, **cfg)
FILL = "both+peer"


def band_ids(path):
    total, out, k = max(path[0].count, 1e-9), [], 0
    for nd in path[1:]:
        b = -math.log2(max(nd.count, 1e-9) / total)
        while k < len(L.band_bits) and b >= L.band_bits[k]:
            out.append(nd.id)
            k += 1
    return out


def misplaced(tree, leaf, x):
    st = tree.ancestors(leaf)[::-1][:-1]
    now = tree.categorize(x, 1.0)[:-1]
    a, b = band_ids(st) if st else [], band_ids(now) if now else []
    m = max(len(a), len(b))
    return 1 - (sum(p == q for p, q in zip(a, b)) / m if m else 1.0)


def current(e):
    """The record's description now (content re-read; the context's own-content facet re-read)."""
    cnt = L._content(e) if e.token is None else None
    ctx = dict(e.ctx)
    if L.donate_ctx:
        L._facet(ctx, "cc", L._cc_ids(e))
    return cnt, ctx


def leaf_index():
    at = {"ctx": {}, "cnt": {}}
    for e in L.records:
        if e.leaf is not None:
            at["ctx"].setdefault(id(e.leaf), []).append(e)
        if e.cleaf is not None and e.token is None:
            at["cnt"].setdefault(id(e.cleaf), []).append(e)
    return at


def use_phase(rng):
    """Run the use phase; return Counter of reached records (by id) and the records."""
    at = leaf_index()
    hits, recs = Counter(), {}
    origs = {}

    def hook(which, tree):
        orig = tree.categorize
        origs[which] = orig

        def cat(x, w=1.0):
            path = orig(x, w)
            if path:
                node = path[-1]
                for c in ([node] if not node.children else node.children):
                    if not c.children:
                        for e in at[which].get(id(c), ()):
                            hits[id(e)] += 1
                            recs[id(e)] = e
            return path
        tree.categorize = cat

    class Drawn(list):
        def __getitem__(self, i):
            e = list.__getitem__(self, i)
            if isinstance(i, int):
                hits[id(e)] += 1
                recs[id(e)] = e
            return e

    hook("ctx", L.ctx)
    hook("cnt", L.cnt)
    saved = L.records
    try:
        for s in held:
            L.parse(s)
        L.records = Drawn(saved)
        if hasattr(saved, "owner"):
            saved.marking = True
        ix = L.concept_index()
        for _ in range(100):
            L.generate_tree(rng, index=ix, choose="leaf", fill_name=FILL)
        for s in held[:100]:
            if len(s) >= 3:
                L.generate_tree(rng, index=ix, choose="leaf", fill_name=FILL, gap=(s[:2], []))
    finally:
        if hasattr(saved, "owner"):
            saved.marking = False
        L.records = saved
        for which in ("ctx", "cnt"):
            getattr(L, which).categorize = origs[which]
    return hits, recs


t0, done, rows = time.time(), 0, []
rng = random.Random(seed + 7)
for c in checks:
    for s in train[done:c]:
        L.learn(s)
    done = c
    learn_s = time.time() - t0
    quiet = L._quiet
    hits, recs = use_phase(rng)
    L._quiet = True                  # measuring is not use
    st, mp, wst, wmp = {"ctx": [], "cnt": []}, {"ctx": [], "cnt": []}, {"ctx": 0.0, "cnt": 0.0}, {"ctx": 0.0, "cnt": 0.0}
    tot = {"ctx": 0.0, "cnt": 0.0}
    for k, e in recs.items():
        cnt, ctx = current(e)
        for which, tree, leaf, old, new in (("ctx", L.ctx, e.leaf, e.ctx, ctx), ("cnt", L.cnt, e.cleaf, e.cnt, cnt)):
            if leaf is None or new is None:
                continue
            s_ = float(new != old)
            m_ = misplaced(tree, leaf, new)
            st[which].append(s_)
            mp[which].append(m_)
            wst[which] += hits[k] * s_
            wmp[which] += hits[k] * m_
            tot[which] += hits[k]
    whole = {"ctx": [], "cnt": []}
    for r in random.Random(seed).sample(list(L.records), min(1500, len(L.records))):
        if r.leaf is not None:
            whole["ctx"].append(misplaced(L.ctx, r.leaf, r.ctx))
        if r.token is None and r.cleaf is not None:
            whole["cnt"].append(misplaced(L.cnt, r.cleaf, r.cnt))
    L._quiet = quiet
    m = lambda v: round(sum(v) / len(v), 3) if v else None
    row = {"tag": tag, "seed": seed, "checkpoint": c, "learn s": round(learn_s), "records": len(L.records),
           "reached records ctx / cnt": [len(st["ctx"]), len(st["cnt"])],
           "reached stale ctx / cnt": [m(st["ctx"]), m(st["cnt"])],
           "reached misplaced ctx / cnt": [m(mp["ctx"]), m(mp["cnt"])],
           "reached misplaced, by use ctx / cnt": [round(wmp[w] / max(tot[w], 1e-9), 3) for w in ("ctx", "cnt")],
           "whole tree misplaced ctx / cnt": [m(whole["ctx"]), m(whole["cnt"])],
           "use repaired so far": L.stats.get("use repaired", 0)}
    print(json.dumps(row), flush=True)
    rows.append(row)
t1 = time.time()
ix = L.concept_index()
S = [g for g in (L.generate_tree(rng, index=ix, choose="leaf", fill_name=FILL) for _ in range(300)) if g]
seen = {tuple(s) for s in train}
tri = [all(t in trigrams for t in zip(*(lambda p: (p, p[1:], p[2:]))(["<s>", "<s>"] + list(g) + ["</s>"]))) for g in S]
nov = [tuple(g) not in seen for g in S]
print(json.dumps({"tag": tag, "seed": seed, "GENERATION": {"generated": len(S), "trigrams attested": round(sum(tri) / max(len(S), 1), 3),
                  "novel": round(sum(nov) / max(len(S), 1), 3),
                  "attested + novel": round(sum(a and b for a, b in zip(tri, nov)) / max(len(S), 1), 3),
                  "examples": [" ".join(g) for g in S[:8]]}, "learn s": rows[-1]["learn s"]}), flush=True)
