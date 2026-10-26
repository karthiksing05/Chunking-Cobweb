"""Where TinyStories misplacement comes from (round 31: content stays at ~0.21 at
every reassignment budget). The TEMPORAL measure compares a record's stored band
path with where its STORED description sorts now, by categorize at weight 1,
while ifit placed it with its posterior weight. Measured here, by record weight:

  fresh   the sentence's new records, right after the sentence is learned
  end     a sample of all records after the stream
  w=1     categorize at weight 1 (the TEMPORAL measure)
  own w   categorize at the record's own weight (what ifit used)

Usage: t3_misplace_why.py <cfg json> <seed> <n>
"""
import json
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "experiments/v3/diag")
from trellis2.stories import read_sentences, simple_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3, _drift  # noqa: E402
import math  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
corpus = simple_sentences(read_sentences(), vocab_size=250)
random.Random(seed).shuffle(corpus)
L = Trellis3(seed=seed, **cfg)


def band_ids(path):
    total, out, k = max(path[0].count, 1e-9), [], 0
    for nd in path[1:]:
        b = -math.log2(max(nd.count, 1e-9) / total)
        while k < len(L.band_bits) and b >= L.band_bits[k]:
            out.append(nd.id)
            k += 1
    return out


def mis(tree, leaf, x, w):
    st = tree.ancestors(leaf)[::-1][:-1]
    now = tree.categorize(x, w)[:-1]
    a, b = band_ids(st) if st else [], band_ids(now) if now else []
    m = max(len(a), len(b))
    return 1 - (sum(p == q for p, q in zip(a, b)) / m if m else 1.0)


def wbin(w):
    return "w>=0.5" if w >= 0.5 else "0.1-0.5" if w >= 0.1 else "<0.1"


def measure(recs, acc):
    for r in recs:
        if r.leaf is not None:
            acc[("ctx", wbin(r.weight), "w=1")].append(mis(L.ctx, r.leaf, r.ctx, 1.0))
            acc[("ctx", wbin(r.weight), "own w")].append(mis(L.ctx, r.leaf, r.ctx, r.weight))
        if r.token is None and r.cleaf is not None:
            acc[("cnt", wbin(r.weight), "w=1")].append(mis(L.cnt, r.cleaf, r.cnt, 1.0))
            acc[("cnt", wbin(r.weight), "own w")].append(mis(L.cnt, r.cleaf, r.cnt, r.weight))


def report(tag, acc):
    out = {}
    for which in ("ctx", "cnt"):
        for how in ("w=1", "own w"):
            allv = [v for (a, b, c), vs in acc.items() if a == which and c == how for v in vs]
            out[f"{which} {how}"] = [round(sum(allv) / max(len(allv), 1), 3), len(allv)]
            for b in ("w>=0.5", "0.1-0.5", "<0.1"):
                vs = acc.get((which, b, how), [])
                if vs:
                    out[f"{which} {how} {b}"] = [round(sum(vs) / len(vs), 3), len(vs)]
    print(json.dumps({tag: out}), flush=True)


fresh = defaultdict(list)
t0 = time.time()
for k, s in enumerate(corpus[:n], 1):
    first = len(L.records)
    L.learn(s)
    if k > n // 2:                     # fresh records once the tree has some size
        measure(L.records[first:], fresh)
report("fresh", fresh)
end = defaultdict(list)
sample = random.Random(seed).sample(L.records, min(1500, len(L.records)))
measure(sample, end)
report("end", end)
# the CURRENT description (names read now, as a refresh would write them):
# how often it differs from the stored one, its drift, and where it sorts
L._wc_cache, L._rel_cache, L._nb_cache = {}, {}, {}
cur = defaultdict(list)
for r in sample:
    if r.token is None and r.cleaf is not None:
        r.memo.pop("own", None)
        new = L._content(r)
        st = L.cnt.translate(r.cnt) if hasattr(L.cnt, "translate") else r.cnt
        cur["cnt changed"].append(float(_drift(st, new) > 1e-9))
        cur["cnt drift"].append(_drift(st, new))
        cur["cnt misplaced, current desc, own w"].append(mis(L.cnt, r.cleaf, new, r.weight))
    if r.leaf is not None:
        st = L.ctx.translate(r.ctx) if hasattr(L.ctx, "translate") else r.ctx
        new = dict(st)
        if L.donate_ctx:
            L._facet(new, "cc", L._cc_ids(r))
        cur["ctx changed"].append(float(_drift(st, new) > 1e-9))
        cur["ctx drift"].append(_drift(st, new))
        cur["ctx misplaced, current desc, own w"].append(mis(L.ctx, r.leaf, new, r.weight))
print(json.dumps({"current": {k: round(sum(v) / max(len(v), 1), 3) for k, v in cur.items()}}), flush=True)
print(json.dumps({"records": len(L.records), "s": round(time.time() - t0), "stats": L.stats,
                  "synced": getattr(L.cnt, "n_synced", 0) + getattr(L.ctx, "n_synced", 0)}))
