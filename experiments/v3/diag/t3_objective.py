"""Does TRELLIS v3's chunk factor prefer the gold analysis? Sum of gains over a
tree's composites (the log factor inside-outside uses), for gold, the MBR parse
and right-branching, on held-out sentences; self-taught or taught with gold
analyses (store='gold': the gold tree's composites stored, weight 1)."""
import os, sys, json
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, Tree
from trellis2.trellis3 import Trellis3
cond, mode = sys.argv[1], sys.argv[2]
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"store": "posterior", "children": False, "spine": False}
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = Trellis3(seed=13, **cfg)
if mode == "gold":
    orig = L.parse
    def forced(tokens, gold=None):
        r = orig(tokens)
        if gold is not None:
            r["chosen"] = [(i, gold.split[(i, j)], j) for (i, j) in gold.composite_spans()]
            r["post"] = {k: (1.0 if k in set(r["chosen"]) else 0.0) for k in r["comps"]}
        return r
    for e in train:
        L.parse = lambda toks, g=e.tree: forced(toks, g)
        L.learn(e.tokens)
    L.parse = orig
else:
    for e in train:
        L.learn(e.tokens)
def score(r, tree):
    return sum(r["comps"][(i, tree.split[(i, j)], j)].gain for (i, j) in tree.composite_spans())
wins = {"gold>ours": 0, "ours>gold": 0, "tie": 0}; tot = {"gold": 0.0, "ours": 0.0, "rb": 0.0}
for e in test:
    n = len(e.tokens)
    if n < 3: continue
    r = L.parse(e.tokens)
    rb = Tree(n, {(i, n): i + 1 for i in range(n - 1)}, {}, [(0, n)])
    g, o, b = score(r, e.tree), score(r, r["tree"]), score(r, rb)
    tot["gold"] += g; tot["ours"] += o; tot["rb"] += b
    wins["gold>ours" if g > o + 1e-6 else "ours>gold" if o > g + 1e-6 else "tie"] += 1
print(cond, mode, {k: round(v) for k, v in tot.items()}, wins)
