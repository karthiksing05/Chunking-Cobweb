"""Failure-mode check for TRELLIS v2 representations (evaluation only).

Memory is an AND/OR structure. A content concept is an AND node: a chunk type,
its parts generated together. A context concept is an OR node: a category,
any of its members can fill the slot. The learner sees raw tokens only; the
generating grammar (gold labels of gold spans) is used here only to score.
See docs/FAILURE_MODES.md for what each number means and how to read it.

  A-or   context concepts as substitution classes, at levels above the leaf:
         precision (a member swapped in for another stays grammatical) and
         recall (members of the same gold category share the concept);
         also cross-length (e.g. "the dog" for "the big red dog").
  A-and  content concepts as chunk types: swapping a whole member in, and
         recombining one member's left part with another member's right part
         (does the chunk type generate its constituents independently?).
  B      comparability across granularity: AUC of path-overlap similarity
         (same gold category vs different), for same-length and
         cross-length pairs, in each tree.
  F      Fodor & Pylyshyn: systematicity (swap two same-category constituents
         of a sentence; does memory analyse the result as well?) and
         productivity (grammatical samples longer than any training sentence).
  C      co-evolution: how stale stored representations are (descriptions not
         yet relabelled; stored leaf no longer where the description sorts).

Usage: t3_failure.py <cond> <cfg json> [seed] [tag]
"""
import os, sys, json, math, random, time
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3

cond, cfg = sys.argv[1], json.loads(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
tag = sys.argv[4] if len(sys.argv) > 4 else ""
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
g = CFG(target_grammar(cond))
rng = random.Random(seed)
t0 = time.time()
L = Trellis3(seed=seed, **cfg)
owner = []
for si, e in enumerate(train):
    n0 = len(L.records)
    L.learn(e.tokens)
    owner += [si] * (len(L.records) - n0)
learn_s = time.time() - t0

# gold instances: every gold span of every training sentence that memory stored
by_sent = defaultdict(dict)
for r, si in zip(L.records, owner):
    key = (r.span, None if r.token is not None else r.span[0] + (r.left.span[1] - r.left.span[0]))
    by_sent[si][key] = r
inst = []          # dict(si, span, split, label, rec, ctx path, cnt path)
gold_total = 0
for si, e in enumerate(train):
    lab = g.gold_labels(e.tokens, e.tree)
    for (i, j), a in lab.items():
        gold_total += 1
        k = None if j - i == 1 else e.tree.split[(i, j)]
        r = by_sent[si].get(((i, j), k))
        if r is None or r.leaf is None:
            continue
        inst.append({"si": si, "span": (i, j), "split": k, "label": a, "rec": r,
                     "cp": L.ctx.ancestors(r.leaf)[::-1],
                     "np": L.cnt.ancestors(r.cleaf)[::-1] if r.cleaf is not None else None})


def toks(x):
    i, j = x["span"]
    return train[x["si"]].tokens[i:j]


def substitute(a, new):
    t = train[a["si"]].tokens
    i, j = a["span"]
    return t[:i] + list(new) + t[j:]


def concept(path, level):
    """level ('u', n): n above the leaf; ('d', n): depth n below the root."""
    kind, n = level
    if kind == "u":
        return path[max(len(path) - 1 - n, 0)].id
    return path[min(n, len(path) - 1)].id


LEVELS = [("u", 1), ("u", 2), ("u", 4), ("d", 3), ("d", 2), ("d", 1)]


def substitution(key, level, pool, recombine=False, n=1500):
    """precision / recall of concepts at ``level`` as substitution classes."""
    index = defaultdict(list)
    for x in pool:
        for node in x[key]:
            index[node.id].append(x)
    by_label = defaultdict(list)
    for x in pool:
        by_label[x["label"]].append(x)
    prec, prec_x, rec, rec_x, rc = [], [], [], [], []
    for a in rng.sample(pool, min(n, len(pool))):
        c = concept(a[key], level)
        cands = [b for b in index[c] if b["si"] != a["si"]]
        if cands:
            b = rng.choice(cands)
            ok = g.recognizes(substitute(a, toks(b)))
            prec.append(ok)
            if len(toks(b)) != len(toks(a)):
                prec_x.append(ok)
            if recombine and a["split"] is not None and b["split"] is not None:
                ta, tb = toks(a), toks(b)
                ka, kb = a["split"] - a["span"][0], b["split"] - b["span"][0]
                rc.append(g.recognizes(substitute(a, ta[:ka] + tb[kb:])))
        same = [b for b in by_label[a["label"]] if b["si"] != a["si"]]
        if same:
            b = rng.choice(same)
            hit = c in {nd.id for nd in b[key]}
            rec.append(hit)
            if len(toks(b)) != len(toks(a)):
                rec_x.append(hit)
    m = lambda v: round(sum(v) / len(v), 3) if v else None
    out = {"precision": m(prec), "recall": m(rec), "precision cross-length": m(prec_x),
           "recall cross-length": m(rec_x)}
    if recombine:
        out["recombination"] = m(rc)
    return out


def ipath_bag(path):
    return Counter(L._path_ids(path, "ipath"))


def auc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, r1 = 1, 0, 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (rank + rank + (j - i) - 1) / 2
        r1 += avg * sum(1 for t in allv[i:j] if t[1])
        rank += j - i; i = j
    return round((r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 3)


def comparability(key, pool, n=20000):
    bags = {id(x): ipath_bag(x[key]) for x in pool}
    res = defaultdict(lambda: ([], []))
    for _ in range(n):
        a, b = rng.choice(pool), rng.choice(pool)
        if a is b:
            continue
        A, Bg = bags[id(a)], bags[id(b)]
        sim = sum((A & Bg).values()) / max(sum((A | Bg).values()), 1)
        la, lb = len(toks(a)), len(toks(b))
        group = "same length" if la == lb else ("word vs chunk" if min(la, lb) == 1 else "chunk vs chunk, lengths differ")
        for gname in (group, "all"):
            res[gname][0 if a["label"] == b["label"] else 1].append(sim)
    return {k: auc(*v) for k, v in res.items()}


report = {"cond": cond, "seed": seed, "tag": tag, "cfg": cfg, "learn seconds": round(learn_s),
          "gold spans stored": round(len(inst) / max(gold_total, 1), 3)}
chunks = [x for x in inst if x["split"] is not None and x["np"] is not None]
report["A-or (context concepts)"] = {f"{k}{n}": substitution("cp", (k, n), inst) for k, n in LEVELS}
report["A-and (content concepts)"] = {f"{k}{n}": substitution("np", (k, n), chunks, recombine=True)
                                       for k, n in LEVELS}
report["B comparability AUC"] = {"context": comparability("cp", inst), "content": comparability("np", chunks)}

# F: systematicity -- swap two disjoint same-category gold constituents
sys_orig, sys_swap, n_sw = [], [], 0
for si, e in enumerate(train[:200]):
    lab = g.gold_labels(e.tokens, e.tree)
    spans = [(s, a) for s, a in lab.items() if s[1] - s[0] > 1 and s != (0, len(e.tokens))]
    pairs = [(p, q) for p in spans for q in spans if p[1] == q[1] and p[0][1] <= q[0][0]
             and e.tokens[p[0][0]:p[0][1]] != e.tokens[q[0][0]:q[0][1]]]
    if not pairs:
        continue
    (p, _), (q, _) = rng.choice(pairs)
    t = e.tokens
    sw = t[:p[0]] + t[q[0]:q[1]] + t[p[1]:q[0]] + t[p[0]:p[1]] + t[q[1]:]
    if not g.recognizes(sw):
        continue
    d = len(t[q[0]:q[1]]) - len(t[p[0]:p[1]])
    orig_spans = {p, q}
    new_spans = {(p[0], p[0] + (q[1] - q[0])), (q[0] + d, q[1])}
    for tokens_, want, acc in ((t, orig_spans, sys_orig), (sw, new_spans, sys_swap)):
        r = L.parse(tokens_)
        got = {(i, j) for (i, k, j) in r["chosen"]}
        acc.append(len(want & got) / 2)
    n_sw += 1
m = lambda v: round(sum(v) / len(v), 3) if v else None
report["F systematicity"] = {"pairs": n_sw, "swapped constituents found, original": m(sys_orig),
                             "swapped constituents found, swapped": m(sys_swap)}

# F: productivity -- grammatical samples by length, against the training maximum
max_len = max(len(e.tokens) for e in train)
ix = L.concept_index()
S = [s for s in (L.generate_tree(rng, index=ix) for _ in range(500)) if s]
longer = [s for s in S if len(s) > max_len]
report["F productivity"] = {"samples": len(S), "grammatical": m([g.recognizes(s) for s in S]),
                            "training max length": max_len,
                            "longer than any training sentence": m([len(s) > max_len for s in S]),
                            "of those, grammatical": m([g.recognizes(s) for s in longer])}

# C: co-evolution -- staleness of stored representations
stale_desc, stale_pos_ctx, stale_pos_cnt = [], [], []
for r in rng.sample(L.records, min(1500, len(L.records))):
    if r.token is None:
        stale_desc.append(L._content(r) != r.cnt)
    now = L.ctx.categorize(r.ctx)
    stored = L.ctx.ancestors(r.leaf)[::-1]
    stale_pos_ctx.append(stored[-2].id not in {nd.id for nd in now} if len(stored) > 1 else False)
    if r.token is None and r.cleaf is not None:
        now = L.cnt.categorize(r.cnt)
        stored = L.cnt.ancestors(r.cleaf)[::-1]
        stale_pos_cnt.append(stored[-2].id not in {nd.id for nd in now} if len(stored) > 1 else False)
report["C co-evolution"] = {"content description not yet relabelled": m(stale_desc),
                            "context leaf no longer under where its description sorts": m(stale_pos_ctx),
                            "content leaf no longer under where its description sorts": m(stale_pos_cnt)}
report["seconds"] = round(time.time() - t0)
print(json.dumps(report), flush=True)
