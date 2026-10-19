"""What the content tree's concepts hold. Train (one pass, records tagged with
whether their span is a gold constituent and its label, for diagnosis only),
then for each held-out gold constituent take the content concept one level
above where its instance sorts and report: the share of that concept's stored
weight that is gold constituents, the label purity among those, and its size."""
import os, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3
cond, cfg = sys.argv[1], json.loads(sys.argv[2])
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = Trellis3(seed=13, **cfg); g = CFG(target_grammar(cond))
for e in train:
    n0 = len(L.records); L.learn(e.tokens)
    lab = g.gold_labels(e.tokens, e.tree); gold = e.tree.brackets()
    for r in L.records[n0:]:
        r.memo["gold"] = lab.get(r.span, "?") if (r.token is None and r.span in gold) else None
# members of every content concept
at = defaultdict(list)
for r in L.records:
    if r.cleaf is not None: at[id(r.cleaf)].append(r)
def members(node):
    out, st = [], [node]
    while st:
        m = st.pop()
        if m.children: st.extend(m.children)
        else: out.extend(at.get(id(m), []))
    return out
rows = defaultdict(list)
for e in test:
    r = L.parse(e.tokens); lab = g.gold_labels(e.tokens, e.tree)
    for (i, j) in e.tree.composite_spans():
        k = e.tree.split[(i, j)]; el = r["comps"][(i, k, j)]
        path = L.cnt.categorize(el.cnt)
        for up in (1, 2, 3):
            node = path[max(len(path) - 1 - up, 1)] if len(path) > 1 else path[0]
            ms = members(node); W = sum(m.weight for m in ms) or 1e-9
            gw = sum(m.weight for m in ms if m.memo.get("gold"))
            labs = Counter()
            for m in ms:
                if m.memo.get("gold"): labs[m.memo["gold"]] += m.weight
            same = labs.get(lab.get((i, j), "?"), 0.0) / (sum(labs.values()) or 1e-9)
            rows[up].append((gw / W, same, len(ms)))
for up in (1, 2, 3):
    xs = rows[up]
    print(cond, "up%d" % up, "gold-constituent share of concept weight %.2f | same label among constituents %.2f | members %.0f" % (
        sum(a for a, _, _ in xs) / len(xs), sum(b for _, b, _ in xs) / len(xs), sum(c for _, _, c in xs) / len(xs)), flush=True)
tot = sum(r.weight for r in L.records if r.token is None); gtot = sum(r.weight for r in L.records if r.token is None and r.memo.get("gold"))
print(cond, "whole content tree: gold-constituent share of weight %.2f" % (gtot / tot))
