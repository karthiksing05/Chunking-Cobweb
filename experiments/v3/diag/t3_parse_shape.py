"""Tree shape of held-out parses against gold, on the toy grammars. For each recipe and
grammar: omission and commission against gold; of the parse's internal brackets the share
that end at the sentence end (right), start at its beginning (left) or neither; the share
of sentences parsed purely right- or left-branching; and the same scores for a purely
right-branching and a purely left-branching parser. Gold scores only.

Usage: t3_parse_shape.py <tag> <cfg json> <conds> <seed> <n>
"""
import json, os, sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, Tree, default_data_root
from trellis2.evaluation import BracketTally
from trellis2.trellis3 import Trellis3

tag, cfg = sys.argv[1], json.loads(sys.argv[2])
seed, n = int(sys.argv[4]), int(sys.argv[5])


def fixed(m, right):
    split = {}
    for x in range(m - 1):
        if right:
            split[(x, m)] = x + 1
        else:
            split[(0, m - x)] = m - x - 1
    return Tree(m, split, {}, [(0, m)])


def share(trees):
    R = L = M = pr = pl = big = 0
    for t in trees:
        m = t.n
        inner = [(i, j) for i, j in t.brackets() if j - i >= 2 and j - i < m]
        R += sum(j == m for _, j in inner); L += sum(i == 0 for i, _ in inner)
        M += sum(i != 0 and j != m for i, j in inner)
        if m >= 4 and inner:
            big += 1; pr += all(j == m for _, j in inner); pl += all(i == 0 for i, _ in inner)
    tot = max(R + L + M, 1)
    return {"right": round(R / tot, 3), "left": round(L / tot, 3), "neither": round(M / tot, 3),
            "pure right": round(pr / max(big, 1), 3), "pure left": round(pl / max(big, 1), 3)}


for cond in sys.argv[3].split(","):
    train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    L = Trellis3(seed=seed, **cfg)
    for e in train[:n]:
        L.learn(e.tokens)
    test = test[:100]
    pred = [L.parse(e.tokens)["tree"] for e in test]
    a, r, l = BracketTally(), BracketTally(), BracketTally()
    for e, p in zip(test, pred):
        a.add(e.tree, p); r.add(e.tree, fixed(e.tree.n, True)); l.add(e.tree, fixed(e.tree.n, False))
    ex = []
    for e, p in list(zip(test, pred))[:4]:
        ex.append({"tokens": " ".join(e.tokens), "gold": sorted(e.tree.brackets()), "parse": sorted(p.brackets())})
    print(json.dumps({"tag": tag, "cond": cond, "seed": seed, "omission": round(a.omission, 3), "commission": round(a.commission, 3),
                      "exact": round(a.exact / max(a.sentences, 1), 3), "shape": share(pred), "gold shape": share([e.tree for e in test]),
                      "right-branching omission": round(r.omission, 3), "left-branching omission": round(l.omission, 3), "examples": ex}), flush=True)
