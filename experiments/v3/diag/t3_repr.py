"""Representation scorecard for TRELLIS v2's two trees, independent of parsing
scores. Trains one pass (records tagged with gold constituency for diagnosis
only), then reports:
  shape     per tree: internal nodes' mean children, mean leaf depth, mean size
            (members) of the concept one level above each leaf
  content   for held-out gold constituents, at 1 and 2 levels above where the
            instance sorts: share of the concept's weight that is gold
            constituents, and same-label share among those
  context   word part-of-speech purity one level above each held-out word's leaf
  generate  expansive generation: grammatical / novel (500 samples)"""
import os, sys, json, random, time
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3
cond, tag, cfg = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = Trellis3(seed=13, **cfg); g = CFG(target_grammar(cond)); t0 = time.time()
for e in train:
    n0 = len(L.records); L.learn(e.tokens)
    lab = g.gold_labels(e.tokens, e.tree); gold = e.tree.brackets()
    for r in L.records[n0:]:
        r.memo["gold"] = lab.get(r.span, "?") if (r.token is None and r.span in gold) else None
learn = time.time() - t0
def shape(tree, which):
    kids, depths, stack = [], [], [(tree.root, 0)]
    while stack:
        n, d = stack.pop()
        if n.children:
            kids.append(len(n.children)); stack.extend((c, d + 1) for c in n.children)
        else:
            depths.append(d)
    at = defaultdict(list)
    for r in L.records:
        leaf = r.leaf if which == "ctx" else r.cleaf
        if leaf is not None: at[id(leaf)].append(r)
    return kids, depths, at
def members(node, at):
    out, st = [], [node]
    while st:
        m = st.pop()
        if m.children: st.extend(m.children)
        else: out.extend(at.get(id(m), []))
    return out
out = {"cond": cond, "tag": tag, "learn s": round(learn)}
for name, tree, which in (("ctx", L.ctx, "ctx"), ("cnt", L.cnt, "cnt")):
    kids, depths, at = shape(tree, which)
    sizes = [len(members(r.leaf.parent if which == "ctx" else r.cleaf.parent, at))
             for r in L.records[::25] if (r.leaf if which == "ctx" else r.cleaf) is not None
             and (r.leaf if which == "ctx" else r.cleaf).parent is not None]
    out[name + " children"] = round(sum(kids) / len(kids), 1)
    out[name + " depth"] = round(sum(depths) / len(depths), 1)
    out[name + " u1 size"] = round(sum(sizes) / max(len(sizes), 1))
    if name == "cnt": cat = at
    else: ctxat = at
rows, pos = defaultdict(list), []
for e in test:
    r = L.parse(e.tokens); lab = g.gold_labels(e.tokens, e.tree)
    for (i, j) in e.tree.composite_spans():
        el = r["comps"][(i, e.tree.split[(i, j)], j)]; path = L.cnt.categorize(el.cnt)
        for up in (1, 2):
            node = path[max(len(path) - 1 - up, 1)] if len(path) > 1 else path[0]
            ms = members(node, cat); W = sum(m.weight for m in ms) or 1e-9
            labs = Counter()
            for m in ms:
                if m.memo.get("gold"): labs[m.memo["gold"]] += m.weight
            rows[up].append((sum(labs.values()) / W, labs.get(lab.get((i, j), "?"), 0.0) / (sum(labs.values()) or 1e-9)))
    for w in r["words"]:
        path = L.ctx.categorize(w.ctx)
        pos.append((path[max(len(path) - 2, 0)].id, lab.get(w.span, "?")))
for up in (1, 2):
    out[f"cnt u{up} phrase share"] = round(sum(a for a, _ in rows[up]) / len(rows[up]), 2)
    out[f"cnt u{up} same type"] = round(sum(b for _, b in rows[up]) / len(rows[up]), 2)
grp = defaultdict(Counter)
for c, l in pos: grp[c][l] += 1
out["ctx u1 POS purity"] = round(sum(c.most_common(1)[0][1] for c in grp.values()) / len(pos), 2)
mem = L._members(L.ctx, "ctx"); rng = random.Random(0); seen = {tuple(e.tokens) for e in train}
S = [s for s in (L.generate_expand(rng, members=mem) for _ in range(500)) if s]
gram = [s for s in S if g.recognizes(s)]
out["generate grammatical"] = round(len(gram) / max(len(S), 1), 3)
out["generate gram+novel"] = round(sum(tuple(s) not in seen for s in gram) / max(len(S), 1), 3)
print(json.dumps(out), flush=True)
