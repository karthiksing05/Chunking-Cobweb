"""Audit of the hierarchy ablation: what do one or two trees hold, on the same stream?

For hierarchy "two", "shared" (one tree, a composite's two descriptions as two records)
and "joint" (one tree, one record per element):

  size        learning time, records, nodes, leaf depth
  cu          category utility of the tree's top split and of a cut near 6 bits, by
              attribute family (window, own/neighbour content, previous sentence, parts):
              which description the concepts organise around
  mix         share of the weight under concepts near 3 / 6 / 9 bits that is words, and
              how pure each concept is in word-or-composite (1 = never mixed)
  words       gold-tag purity of word occurrences under context concepts within 6 / 9 / 12
              bits, against the same concepts with tags shuffled
  chunks      gold tag-pattern purity of two-word composites under their concepts
  same place  of pairs of chunks with the same window and different parts, the share under
              one concept within 6 bits; of pairs with the same parts and different windows,
              the same: does a concept group by where, by what, or by both?
  parses      bracket agreement with the two-tree parse is computed from these dumps by the
              caller; here the parses of 200 held-out sentences and their mean top posterior

Evaluation only. Gold tags score, they never teach.

Usage: t3_hier_audit.py <cfg json> <seed> <n> <out json>
"""
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict

import nltk

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, out_path = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
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
gold_ct = defaultdict(Counter)
for s in [s for s in all_sents if keep(s)][:20000]:
    for w, t in nltk.pos_tag(s):
        gold_ct[w][t[:2]] += 1
gold = {w: c.most_common(1)[0][0] for w, c in gold_ct.items()}
held = [s for st in test for s in st if keep(s)]
random.Random(seed + 3).shuffle(held)
held = held[:200]
FAMILY = {"l1": "window", "l2": "window", "r1": "window", "r2": "window", "cc": "own content", "lc": "neighbour chunks",
          "rc": "neighbour chunks", "pv": "previous sentence", "L": "part names", "R": "part names", "LC": "part content", "RC": "part content"}


def words_of(e):
    return [e.token] if e.token is not None else words_of(e.left) + words_of(e.right)


def path(node):
    out = [node]
    while out[-1].parent is not None:
        out.append(out[-1].parent)
    return out[::-1]


def cut(node, tree, bits):
    tot = max(tree.root.count, 1e-9)
    keep_ = [c for c in path(node) if -math.log2(max(c.count, 1e-9) / tot) <= bits]
    return keep_[-1] if keep_ else tree.root


def tree_nodes(tree):
    out, stack = [], [tree.root]
    while stack:
        x = stack.pop()
        out.append(x)
        stack.extend(x.children)
    return out


def sq(node, attr):
    d = node.av.get(attr, {})
    c = max(node.count, 1e-9)
    return sum((v / c) ** 2 for v in d.values())


def cu_by_family(tree, nodes):
    """Sum over the concepts of P(C) [sum_v P(v|C)^2 - sum_v P(v|root)^2], per attribute family."""
    root, fam = tree.root, defaultdict(float)
    for a in tree.attrs:
        base = sq(root, a)
        v = sum(max(c.count, 0.0) / max(root.count, 1e-9) * (sq(c, a) - base) for c in nodes)
        fam[FAMILY.get(a, a)] += v
    return {k: round(v, 4) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])}


def purity(groups):
    """Weighted majority share over groups of labels."""
    tot = sum(sum(c.values()) for c in groups.values())
    return sum(max(c.values()) for c in groups.values()) / max(tot, 1e-9)


res = {"seed": seed, "n": n, "arms": {}}
rows = {}
for mode in ("two", "shared", "joint"):
    t0 = time.time()
    L = Trellis3(seed=seed, hierarchy=mode, **cfg)
    for s, fol in train:
        L.learn(s, follows=fol)
    L._quiet = True
    a = {"learn s": round(time.time() - t0), "records": len(L.records), "composites": sum(e.token is None for e in L.records),
         "moved": L.stats.get("reassigned", 0)}
    for name, tree in ({"context tree": L.ctx, "content tree": L.cnt} if mode == "two" else {"tree": L.ctx}).items():
        ns = tree_nodes(tree)
        leaves = [x for x in ns if not x.children]
        depth = [len(path(x)) - 1 for x in leaves]
        tot = max(tree.root.count, 1e-9)
        band = [x for x in ns if -math.log2(max(x.count, 1e-9) / tot) >= 6 and
                (x.parent is None or -math.log2(max(x.parent.count, 1e-9) / tot) < 6)]
        a[name] = {"nodes": len(ns), "leaves": len(leaves), "mean leaf depth": round(sum(depth) / max(len(depth), 1), 1),
                   "cu top split": cu_by_family(tree, list(tree.root.children)), "cu at 6 bits": cu_by_family(tree, band)}
    # the weight of words and composites under concepts of the (context) tree
    mix = {}
    for bits in (3, 6, 9):
        g = defaultdict(Counter)
        for e in L.records:
            if e.leaf is not None and e.weight > 0:
                g[cut(e.leaf, L.ctx, bits).id]["word" if e.token is not None else "composite"] += e.weight
        mix[str(bits)] = {"concepts": len(g), "word-or-composite purity": round(purity(g), 3)}
    a["mix"] = mix
    # word categories: gold tags under the concepts a word's occurrences sort to
    wp = {}
    rng = random.Random(seed + 5)
    occ = [e for e in L.records if e.token is not None and e.leaf is not None and e.token in gold]
    for bits in (6, 9, 12):
        g, gs = defaultdict(Counter), defaultdict(Counter)
        shuf = [gold[e.token] for e in occ]
        rng.shuffle(shuf)
        for e, t in zip(occ, shuf):
            k = cut(e.leaf, L.ctx, bits).id
            g[k][gold[e.token]] += 1
            gs[k][t] += 1
        wp[str(bits)] = {"concepts": len(g), "tag purity": round(purity(g), 3), "shuffled": round(purity(gs), 3)}
    a["word purity"] = wp
    # chunk types: tag patterns of two-word composites under their concepts
    cp = {}
    two = [e for e in L.records if e.token is None and e.cleaf is not None and e.weight > 0 and len(words_of(e)) == 2
           and all(w in gold for w in words_of(e))]
    for bits in (6, 9, 12):
        g, gs = defaultdict(Counter), defaultdict(Counter)
        pat = [tuple(gold[w] for w in words_of(e)) for e in two]
        shuf = list(pat)
        rng.shuffle(shuf)
        for e, t, u in zip(two, pat, shuf):
            k = cut(e.cleaf, L.cnt, bits).id
            g[k][t] += e.weight
            gs[k][u] += e.weight
        cp[str(bits)] = {"concepts": len(g), "tag-pattern purity": round(purity(g), 3), "shuffled": round(purity(gs), 3)}
    a["chunk purity"] = cp
    # does a concept group by where, by what, or both? pairs of stored composites, within 6 bits
    comps = [e for e in L.records if e.token is None and e.leaf is not None and e.weight >= 0.5 and len(words_of(e)) >= 2]
    win = lambda e: (e.ctx.get("l1"), e.ctx.get("r1"))
    what = lambda e: tuple(words_of(e))
    by_win, by_what = defaultdict(list), defaultdict(list)
    for e in comps:
        by_win[win(e)].append(e)
        by_what[what(e)].append(e)
    prng = random.Random(seed + 9)

    def pair_share(groups, differ, leaf):
        same = tot = 0
        gs = [g for g in groups.values() if len(g) >= 2]
        for _ in range(4000 if gs else 0):
            g = prng.choice(gs)
            x, y = prng.sample(g, 2)
            if differ(x) == differ(y):
                continue
            tot += 1
            same += cut(leaf(x), L.ctx, 6) is cut(leaf(y), L.ctx, 6) or cut(leaf(x), L.ctx, 6).id == cut(leaf(y), L.ctx, 6).id
        return round(same / max(tot, 1), 3), tot

    a["same concept within 6 bits"] = {
        "context place, same window, different chunk": pair_share(by_win, what, lambda e: e.leaf),
        "context place, same chunk, different window": pair_share(by_what, win, lambda e: e.leaf),
        "content place, same window, different chunk": pair_share(by_win, what, lambda e: e.cleaf),
        "content place, same chunk, different window": pair_share(by_what, win, lambda e: e.cleaf)}
    # parses of held-out sentences, and how sure they are
    ps, top = [], []
    for s in held:
        r = L.parse(s)
        ps.append(sorted((i, j) for i, _, j in r["chosen"]))
        top.append(sum(r["post"].get(k, 0.0) for k in r["chosen"]) / max(len(r["chosen"]), 1))
    a["mean chosen posterior"] = round(sum(top) / max(len(top), 1), 3)
    rows[mode] = ps
    res["arms"][mode] = a
    print(json.dumps({"mode": mode, **a}), flush=True)


def f1(a, b):
    tp = fa = fb = 0
    for x, y in zip(a, b):
        sx, sy = set(map(tuple, x)), set(map(tuple, y))
        tp += len(sx & sy)
        fa += len(sx)
        fb += len(sy)
    return round(2 * tp / max(fa + fb, 1), 3)


res["bracket F1 with two trees"] = {m: f1(rows["two"], rows[m]) for m in ("shared", "joint")}
res["parses"] = {m: rows[m] for m in rows}
print(json.dumps({"bracket F1 with two trees": res["bracket F1 with two trees"]}), flush=True)
json.dump(res, open(out_path, "w", encoding="utf-8"))
