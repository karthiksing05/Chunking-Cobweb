"""Are representations temporal and interdependent? (evaluation only)

TEMPORAL -- stored representations follow the current trees:
  stale             share of stored records whose stored description differs from
                    the one computed now (content; context)
  drift             mean total-variation distance between stored and current
                    descriptions, averaged over attributes (0 = current, 1 = disjoint)
  misplaced         share of stored leaves no longer under the concept their
                    current description sorts to one level above the leaf
  name churn        mean TV distance of the frequent words' names between
                    consecutive checkpoints (a stable representation settles)

INTERDEPENDENCE / CONVERGENCE -- a word's representation is useful early:
  word category accuracy at checkpoints: for each word type seen so far, its
  nearest other word type by name overlap (sum of min weights) has the same
  gold preterminal (gold used only to score), overall and by how often the word
  was seen (1, 2-3, 4-7, 8+).

Usage: t3_props.py <cond> <cfg json> [seed] [tag]
"""
import os, sys, json, time, math
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3, Elem

cond, cfg = sys.argv[1], json.loads(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
tag = sys.argv[4] if len(sys.argv) > 4 else ""
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
g = CFG(target_grammar(cond))
gold = defaultdict(Counter)                      # word -> its gold preterminal (scoring only)
for e in train:
    for (i, j), a in g.gold_labels(e.tokens, e.tree).items():
        if j - i == 1:
            gold[e.tokens[i]][a] += 1
gold = {w: c.most_common(1)[0][0] for w, c in gold.items()}


def as_bag(v):
    if isinstance(v, dict):
        t = sum(v.values()) or 1.0
        return {u: p / t for u, p in v.items()}
    if isinstance(v, list):
        real = [u for u in v if u]
        return {u: real.count(u) / len(real) for u in set(real)} if real else {}
    return {v: 1.0}


def tv(a, b):
    a, b = as_bag(a), as_bag(b)
    return 0.5 * sum(abs(a.get(u, 0.0) - b.get(u, 0.0)) for u in set(a) | set(b))


def overlap(a, b):
    a, b = as_bag(a), as_bag(b)
    return sum(min(p, b.get(u, 0.0)) for u, p in a.items())


def desc_drift(old, new):
    keys = set(old) | set(new)
    return sum(tv(old.get(k, {}), new.get(k, {})) for k in keys) / max(len(keys), 1)


def names(L, words):
    L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
    return {w: as_bag(L._ctx_name(Elem((0, 1), token=w)) or {}) for w in words}


L = Trellis3(seed=seed, **cfg)
seen = Counter()
checks = {20, 40, 80, 160, 320, 640, len(train)}
curve, churn, prev = {}, [], None
t0 = time.time()
for n, e in enumerate(train, 1):
    L.learn(e.tokens)
    seen.update(e.tokens)
    if n in checks:
        words = [w for w in seen if w in gold]
        nm = names(L, words)
        hit, by = [], defaultdict(list)
        for w in words:
            best = max((x for x in words if x != w), key=lambda x: overlap(nm[w], nm[x]), default=None)
            ok = best is not None and gold[best] == gold[w] and overlap(nm[w], nm[best]) > 0
            hit.append(ok)
            k = seen[w]
            by["1" if k == 1 else "2-3" if k <= 3 else "4-7" if k <= 7 else "8+"].append(ok)
        curve[n] = {"words": len(words), "category accuracy": round(sum(hit) / max(len(hit), 1), 3),
                    "by count": {b: (round(sum(v) / len(v), 3), len(v)) for b, v in sorted(by.items())}}
        top = [w for w, _ in seen.most_common(30)]
        if prev is not None:
            churn.append((n, round(sum(tv(prev[w], nm[w]) for w in top if w in prev) / len(top), 3)))
        prev = nm
learn_s = time.time() - t0

stale_c, stale_x, drift_c, drift_x, mis_x, mis_c = [], [], [], [], [], []
band_x, band_c = [], []


def band_ids(path):
    """The concepts at the information bands of a root-first path (as naming="bands")."""
    total, out, k = max(path[0].count, 1e-9), [], 0
    for nd in path[1:]:
        b = -math.log2(max(nd.count, 1e-9) / total)
        while k < len(L.band_bits) and b >= L.band_bits[k]:
            out.append(nd.id)
            k += 1
    return out


def band_agree(stored, now):
    """Share of band levels at which the stored leaf's concept is the one the current sort reaches."""
    a, b = band_ids(stored), band_ids(now)
    n = max(len(a), len(b))
    return sum(x == y for x, y in zip(a, b)) / n if n else 1.0
import random
for r in random.Random(seed).sample(L.records, min(1500, len(L.records))):
    if r.token is None and r.cnt is not None:
        new = L._content(r)
        stale_c.append(new != r.cnt)
        drift_c.append(desc_drift(r.cnt, new))
    newx = dict(r.ctx)
    if L.donate_ctx:
        L._facet(newx, "cc", L._cc_ids(r))
    if L.neighbour_types and r.nbt is not None:
        L._nb_facets(newx, r)
    stale_x.append(newx != r.ctx)
    drift_x.append(desc_drift(r.ctx, newx))
    now = L.ctx.categorize(r.ctx)
    st = L.ctx.ancestors(r.leaf)[::-1]
    mis_x.append(st[-2].id not in {nd.id for nd in now} if len(st) > 1 else False)
    band_x.append(1 - band_agree(st[:-1], now[:-1] if len(now) > 1 else now))
    if r.token is None and r.cleaf is not None:
        now = L.cnt.categorize(r.cnt)
        st = L.cnt.ancestors(r.cleaf)[::-1]
        mis_c.append(st[-2].id not in {nd.id for nd in now} if len(st) > 1 else False)
        band_c.append(1 - band_agree(st[:-1], now[:-1] if len(now) > 1 else now))
m = lambda v: round(sum(v) / len(v), 3) if v else None
print(json.dumps({"tag": tag, "cond": cond, "seed": seed, "learn s": round(learn_s),
                  "TEMPORAL": {"stale content / context": [m(stale_c), m(stale_x)],
                               "drift content / context": [m(drift_c), m(drift_x)],
                               "misplaced context / content": [m(mis_x), m(mis_c)],
                               "misplaced at bands context / content": [m(band_x), m(band_c)],
                               "name churn (checkpoint, TV)": churn},
                  "CONVERGENCE": curve}), flush=True)
