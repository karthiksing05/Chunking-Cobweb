"""Dump real instances and concepts from a learned TRELLIS (the Representation page).

Learns the S7 story stream (config as in rounds 68-74), then re-parses one training
sentence and prints, for a word and for chunks of memory's analysis: the stored
context and content instances, their leaves' ancestry with each concept's information
(-log2 of its share of the tree), and the concept just above each leaf: count, its
most common values per attribute, and its passive (generation) statistics.

Usage: t3_dump_repr.py <config json> <seed> <n> <sentence words...>
"""
import json
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3, FILL, LPART, RPART, SENT, NEXT  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
want = sys.argv[4:]
import random  # noqa: E402

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
train = []
for st in stories[1500:]:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, i > 0 and keep(st[i - 1])))
    if len(train) >= n:
        break
train = train[:n]

L = Trellis3(seed=seed, **cfg)
for s, fol in train:
    L.learn(s, follows=fol)
if not want:
    want = next(s for s, fol in train[n // 2:] if fol and 5 <= len(s) <= 6 and "the" in s)
idx = next((i for i, (s, _) in enumerate(train) if s == want), None)
L._pv = list(train[idx - 1][0]) if idx and train[idx][1] else None    # the facet as when it was learned
print("sentence", " ".join(want), "trained" if idx is not None else "not trained", "after", L._pv)


def info(tree, node):
    return round(-math.log2(max(node.count, 1e-9) / max(tree.root.count, 1e-9)), 2)


def top(d, k=6):
    return [(str(v), round(w, 3)) for v, w in sorted(d.items(), key=lambda kv: -kv[1])[:k]]


def concept(tree, node, k=6):
    out = {"id": f"c{node.id}", "count": round(node.count, 2), "info bits": info(tree, node),
           "children": len(node.children), "values": {a: top(d, k) for a, d in node.av.items()}}
    pas = {}
    for a, d in getattr(node, "pav", {}).items():
        named = Counter()
        for v, w in d.items():
            try:
                if a == FILL:
                    named[v[1] if v[0] == "w" else "[" + " ".join(_yield(L.records[v[1]])) + "]"] += w
                elif a in (LPART, RPART):
                    named[" ".join(_yield(L.records[v[0]]))] += w
                elif a in (SENT, NEXT):
                    named[" ".join(_yield(L.records[v]))] += w
                else:
                    named[str(v)] += w
            except Exception:
                named[str(v)] += w
        pas[a] = top(named, k)
    out["passives"] = pas
    return out


def _yield(e):
    out, st = [], [e]
    while st:
        x = st.pop()
        if x.token is not None:
            out.append(x.token)
        else:
            st += [x.right, x.left]
    return out


def path(tree, leaf):
    p = tree.ancestors(leaf)[::-1]
    return [(f"c{x.id}", info(tree, x), round(x.count, 1)) for x in p]


r = L.parse(list(want))
print("analysis", r["tree"].to_string(list(want)) if hasattr(r["tree"], "to_string") else r["tree"])
words, comps, chosen = r["words"], r["comps"], r["chosen"]
dump = {"config": cfg, "seed": seed, "n": n, "sentence": want, "records": len(L.records)}
items = []
for e in words[:3]:
    ctx = L._full(L._context(list(want), e, analysed=True), L.ctx_attrs)
    leaf = L.ctx.categorize(ctx)[-1]
    items.append({"kind": "word", "yield": e.token, "context instance": ctx,
                  "context path": path(L.ctx, leaf),
                  "context concept above": concept(L.ctx, leaf.parent or leaf),
                  "type name (raw)": L._res_bag((e.token,)), "mixed name": L._mixed_name(e.token)})
for key in sorted(chosen, key=lambda t: (t[2] - t[0], t)):
    e = comps[key]
    cnt = L._content(e)
    cleaf = L.cnt.categorize(cnt)[-1]
    ctx = L._full(L._context(list(want), e, analysed=True), L.ctx_attrs)
    leaf = L.ctx.categorize(ctx)[-1]
    items.append({"kind": "chunk", "yield": " ".join(_yield(e)), "span": key, "posterior": r["post"].get(key),
                  "content instance": cnt, "content path": path(L.cnt, cleaf),
                  "content concept above": concept(L.cnt, cleaf.parent or cleaf),
                  "context instance": ctx, "context path": path(L.ctx, leaf),
                  "context concept above": concept(L.ctx, leaf.parent or leaf)})
dump["items"] = items
print(json.dumps(dump, ensure_ascii=False, default=str, indent=1))

# how many sampled type leaves no longer hang in the context tree (their name reads blank)
dead = tot = 0
for k, res in L.type_res.items():
    for _, leaf in res:
        tot += 1
        dead += leaf is not None and leaf.parent is None and leaf is not L.ctx.root
wide = sorted(((len(n.children), n.count, f"c{n.id}") for n in [L.cnt.root] + list(L.cnt.root.children)), reverse=True)[:5]
print(json.dumps({"sampled leaves": tot, "detached": dead, "detached share": dead / max(tot, 1),
                  "widest content concepts (children, count, id)": wide}))
