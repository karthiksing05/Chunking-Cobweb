"""Tree shape after learning n sentences, with and without exact_alone (the singleton scored by its own EC).

Usage: t3_alone_check.py <config json> <n> <0|1>
"""
import json, sys, re, random
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
from collections import Counter
from trellis2.stories import read_sentences
from trellis2.trellis3 import Trellis3
cfg = json.loads(sys.argv[1]); n = int(sys.argv[2]); alone = sys.argv[3] == "1"
sents = read_sentences()
vocab = {w for w, _ in Counter(w for s in sents for w in s).most_common(250)}
train = [s for s in sents if 3 <= len(s) <= 8 and all(w in vocab for w in s)][:n]
L = Trellis3(seed=13, exact_alone=alone, **cfg)
for s in train:
    L.learn(s)
def stats(t):
    st, widths, depths = [(t.root, 0)], [], []
    while st:
        x, d = st.pop()
        if x.children:
            widths.append(len(x.children)); st += [(c, d + 1) for c in x.children]
        else:
            depths.append(d)
    return {"internal": len(widths), "max children": max(widths), "mean depth": round(sum(depths) / len(depths), 2), "leaves": len(depths)}
print(json.dumps({"alone": alone, "records": len(L.records), "cnt": stats(L.cnt), "ctx": stats(L.ctx)}))
