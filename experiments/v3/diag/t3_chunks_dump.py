"""Chunks and concepts from learned story memory, for reading: the content concepts that
hold the most stored weight at a few information levels, each with its commonest chunks,
and the context concepts of words with their commonest words; a few parses of held-out
sentences with each chunk's posterior.

Usage: t3_chunks_dump.py <cfg json> <seed> <n>
"""
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
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
test, train_st = stories[:1500], stories[1500:]
train = []
for st in train_st:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, i > 0 and keep(st[i - 1])))
    if len(train) >= n:
        break
train = train[:n]
L = Trellis3(seed=seed, **cfg)
for s, fol in train:
    L.learn(s, follows=fol)
L._quiet = True


def words(e):
    if e.token is not None:
        return [e.token]
    return words(e.left) + words(e.right)


def path(node):
    out = [node]
    while out[-1].parent is not None:
        out.append(out[-1].parent)
    return out


# every composite's weight under each concept on its content path
tot = max(L.cnt.root.count, 1e-9)
under, nodes = defaultdict(Counter), {}
for e in L.records:
    if e.token is None and e.cleaf is not None and e.weight > 0:
        s = " ".join(words(e))
        for node in path(e.cleaf):
            under[node.id][s] += e.weight
            nodes[node.id] = node
print(json.dumps({"records": len(L.records), "content concepts": len(nodes), "experiences": n}), flush=True)
for lo, hi in ((3, 5), (5, 7), (7, 9), (9, 12)):
    cands = [(i, -math.log2(max(nodes[i].count, 1e-9) / tot)) for i in under]
    cands = [(i, b) for i, b in cands if lo <= b < hi]
    cands.sort(key=lambda t: t[1])
    for i, b in cands[:14]:
        print(json.dumps({"tree": "content", "bits": round(b, 1), "concept": f"c{i}", "weight": round(nodes[i].count, 1),
                          "chunks": [(s, round(w, 2)) for s, w in under[i].most_common(10)]}), flush=True)
# context concepts: the words that stand there
ctot = max(L.ctx.root.count, 1e-9)
wunder, cn = defaultdict(Counter), {}
for e in L.records:
    if e.token is not None and e.leaf is not None:
        for node in path(e.leaf):
            wunder[node.id][e.token] += 1
            cn[node.id] = node
cands = [(i, -math.log2(max(cn[i].count, 1e-9) / ctot)) for i in wunder]
for lo, hi in ((3, 5), (5, 7), (7, 9)):
    sel = sorted([(i, b) for i, b in cands if lo <= b < hi], key=lambda t: t[1])[:12]
    for i, b in sel:
        print(json.dumps({"tree": "context", "bits": round(b, 1), "concept": f"c{i}", "words": wunder[i].most_common(10)}), flush=True)
held = [s for st in test for s in st if keep(s)]
random.Random(seed + 3).shuffle(held)
for s in held[:30]:
    r = L.parse(s)
    chunks = sorted(((" ".join(s[i:j]), round(sum(p for (a, k, b), p in r["post"].items() if (a, b) == (i, j)), 2))
                     for i, _, j in r["chosen"]), key=lambda t: len(t[0]))
    print(json.dumps({"parse": " ".join(s), "chunks": chunks}), flush=True)
