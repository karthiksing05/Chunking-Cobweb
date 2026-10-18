"""Where the chunk factor disagrees with gold: train (gold-taught or self-taught),
then for held-out sentences whose own analysis outscores gold print both trees
with the gain of every composite (content + context parts), and tally which
gold spans lose (by label and length) and which non-gold spans win."""
import os, sys, json
from collections import Counter
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3
cond, mode = sys.argv[1], sys.argv[2]
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"store": "posterior", "children": False, "spine": False}
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = Trellis3(seed=13, **cfg); g = CFG(target_grammar(cond))
if mode == "gold":
    orig = L.parse
    def forced(tokens, gold):
        r = orig(tokens)
        r["chosen"] = [(i, gold.split[(i, j)], j) for (i, j) in gold.composite_spans()]
        r["post"] = {k: (1.0 if k in set(r["chosen"]) else 0.0) for k in r["comps"]}
        return r
    for e in train:
        L.parse = lambda toks, gt=e.tree: forced(toks, gt); L.learn(e.tokens)
    L.parse = orig
else:
    for e in train: L.learn(e.tokens)
def parts(e):
    f = L._climb if L.factor == "climb" else L._gain
    return f(L.cnt, e.cnt), f(L.ctx, e.ctx)
lost, won, shown = Counter(), Counter(), 0
for e in test:
    n = len(e.tokens)
    if n < 3: continue
    r = L.parse(e.tokens); lab = g.gold_labels(e.tokens, e.tree)
    gs = {(i, e.tree.split[(i, j)], j) for (i, j) in e.tree.composite_spans()}
    os_ = set(r["chosen"])
    G = sum(r["comps"][k].gain for k in gs); O = sum(r["comps"][k].gain for k in os_)
    gold_spans = {(i, j) for (i, _, j) in gs}
    for k in gs - os_:
        if (k[0], k[2]) not in {(a, c) for a, _, c in os_}: lost[(lab.get((k[0], k[2]), "?"), k[2] - k[0])] += 1
    for k in os_ - gs:
        if (k[0], k[2]) not in gold_spans: won[k[2] - k[0]] += 1
    if O > G + 1e-6 and shown < 8:
        shown += 1
        print("gold %.1f ours %.1f" % (G, O))
        print("  gold", e.tree.to_string(e.tokens)); print("  ours", r["tree"].to_string(e.tokens))
        for tag, ks in (("gold-only", gs - os_), ("ours-only", os_ - gs)):
            for k in sorted(ks):
                c, x = parts(r["comps"][k])
                print("   %-9s %-32s cnt %5.1f ctx %5.1f" % (tag, " ".join(e.tokens[k[0]:k[1]]) + " | " + " ".join(e.tokens[k[1]:k[2]]), c, x))
print("gold spans missed (label, length):", lost.most_common(12))
print("non-gold spans chosen (length):", sorted(won.items()))
