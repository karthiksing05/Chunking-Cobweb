"""The compact grammar in TRELLIS v2's memory, found by description length
(evaluation only; the learner keeps no grammar).

The grammar memory uses is read off its own analyses of the training
experiences (its readout). Its categories are a cut through the context tree,
chosen as in Li & Abe's (1998) tree-cut model: start from the root's children
and repeatedly split the category whose split most shortens the total
description, the analyses' code under the projected grammar plus the grammar's
own size, until no split shortens it. Each branch so stops at the granularity
its evidence pays for.

Reports per corpus: categories, binary and lexical rules, data and grammar
bits, and samples from the cut grammar (grammatical, grammatical and new),
with a null of the same number of categories assigned at random."""
import os, sys, json, math, random, time
from collections import defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3

cond, cfg = sys.argv[1], json.loads(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
t0 = time.time()
L = Trellis3(seed=seed, **cfg)
for e in train:
    L.learn(e.tokens)
g = CFG(target_grammar(cond)); seen = {tuple(e.tokens) for e in train}
tg = target_grammar(cond)

# memory's analyses of the training experiences; every analysis node keeps its
# path of context concepts (root first)
nodes = []          # (path ids, word or None, left index, right index)
starts = []
for e in train:
    r = L.parse(e.tokens); t = r["tree"]; n = len(e.tokens)
    el = {(w.span[0], w.span[1]): w for w in r["words"]}
    for (i, k, j) in r["chosen"]:
        el[(i, j)] = r["comps"][(i, k, j)]
    index = {}
    for span, x in el.items():
        path = L.ctx.ancestors(x.leaf)[::-1] if x.leaf is not None else L.ctx.categorize(x.ctx)
        index[span] = len(nodes)
        nodes.append([[p.id for p in path], x.token, None, None])
    for (i, j), x in el.items():
        if j - i > 1:
            k = t.split[(i, j)]
            nodes[index[(i, j)]][2] = index[(i, k)]
            nodes[index[(i, j)]][3] = index[(k, j)]
    if n > 1:
        starts.append(index[(0, n)])
vocab = len({x[1] for x in nodes if x[1] is not None})
children = defaultdict(list)
stack = [L.ctx.root]
while stack:
    m = stack.pop()
    children[m.id] = [c.id for c in m.children]
    stack.extend(m.children)


def categorise(frontier):
    out = []
    for path, _, _, _ in nodes:
        a = path[0]
        for p in path:
            a = p
            if p in frontier:
                break
        out.append(a)
    return out


def counts(cat):
    rules = defaultdict(lambda: defaultdict(int)); start = defaultdict(int)
    for idx, (path, tok, l, r) in enumerate(nodes):
        rhs = ("w", tok) if tok is not None else ("r", cat[l], cat[r])
        rules[cat[idx]][rhs] += 1
    for s in starts:
        start[cat[s]] += 1
    return rules, start


def code_entropy(d):
    n = sum(d.values())
    return -sum(c * math.log2(c / n) for c in d.values()) if n else 0.0


def description(frontier):
    cat = categorise(frontier)
    rules, start = counts(cat)
    K = max(len(rules), 2)
    data = code_entropy(start) + sum(code_entropy(d) for d in rules.values())
    gram = 0.0
    for a, d in rules.items():
        n = sum(d.values())
        for rhs in d:
            gram += (3 * math.log2(K) if rhs[0] == "r" else math.log2(K) + math.log2(max(vocab, 2)))
            gram += 0.5 * math.log2(max(n, 2))            # its parameter
    return data + gram, data, gram, rules, start


frontier = set(children[L.ctx.root.id]) or {L.ctx.root.id}
best, data, gram, rules, start = description(frontier)
while True:
    trials = []
    for f in frontier:
        if not children[f]:
            continue
        cand = (frontier - {f}) | set(children[f])
        dl = description(cand)[0]
        if dl < best:
            trials.append((dl, f))
    if not trials:
        break
    trials.sort()
    applied = 0
    for dl, f in trials[:8]:                          # several non-overlapping splits per round
        cand = (frontier - {f}) | set(children[f])
        d = description(cand)
        if d[0] < best:
            frontier, (best, data, gram, rules, start) = cand, d
            applied += 1
    if not applied:
        break


def draw(rng, d):
    t = rng.random() * sum(d.values())
    for k, v in d.items():
        t -= v
        if t <= 0:
            return k
    return k


def sample(rules, start, rng, n=500):
    S = []
    for _ in range(n):
        out, stack, ok = [], [(draw(rng, start), 0)], True
        while stack and ok:
            a, dep = stack.pop()
            if dep > 25 or a not in rules:
                ok = False
                break
            k = draw(rng, rules[a])
            if k[0] == "w":
                out.append(k[1])
            else:
                stack.append((k[2], dep + 1)); stack.append((k[1], dep + 1))
        if ok and out and len(out) <= 40:
            S.append(out)
    return S


S = sample(rules, start, random.Random(seed))
gram_ok = [s for s in S if g.recognizes(s)]
K = len(rules)
salt = random.Random(seed + 1).random()
null_cat = [hash((path[-1], salt)) % K for path, _, _, _ in nodes]
nr, ns = counts(null_cat)
NS = sample(nr, ns, random.Random(seed))
print(json.dumps({
    "cond": cond, "seed": seed, "target": {"nonterminals": len(tg), "rules": sum(len(v) for v in tg.values())},
    "categories": K,
    "binary rules": sum(1 for d in rules.values() for r in d if r[0] == "r"),
    "lexical rules": sum(1 for d in rules.values() for r in d if r[0] == "w"),
    "data bits per sentence": round(data / len(train), 1), "grammar bits": round(gram),
    "grammatical": round(len(gram_ok) / max(len(S), 1), 3),
    "grammatical and new": round(sum(tuple(s) not in seen for s in gram_ok) / max(len(S), 1), 3),
    "null grammatical": round(sum(1 for s in NS if g.recognizes(s)) / max(len(NS), 1), 3),
    "examples": [" ".join(s) + (" ✓" if g.recognizes(s) else " ✗") for s in S[:5]],
    "seconds": round(time.time() - t0)}), flush=True)
