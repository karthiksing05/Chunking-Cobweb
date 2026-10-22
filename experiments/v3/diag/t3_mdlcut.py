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
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
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


direction = sys.argv[4] if len(sys.argv) > 4 else "down"


def top_down():
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
        for dl, f in trials[:8]:                      # several non-overlapping splits per round
            cand = (frontier - {f}) | set(children[f])
            d = description(cand)
            if d[0] < best:
                frontier, (best, data, gram, rules, start) = cand, d
                applied += 1
        if not applied:
            break
    return frontier


def bottom_up():
    """Li & Abe's direction: start at full detail (every analysis node at its
    own leaf) and merge a set of sibling categories into their parent while the
    description shortens. The description is updated incrementally."""
    used = set(p for path, _, _, _ in nodes for p in path)
    parent, kids = {}, defaultdict(list)
    for path, _, _, _ in nodes:
        for a, b in zip(path, path[1:]):
            if b not in parent:
                parent[b] = a; kids[a].append(b)
    cat = [path[-1] for path, _, _, _ in nodes]
    members = defaultdict(set)
    for i, c in enumerate(cat):
        members[c].add(i)
    up = [None] * len(nodes)
    for i, (_, _, l, r) in enumerate(nodes):
        if l is not None:
            up[l] = i; up[r] = i
    is_start = [False] * len(nodes)
    for s0 in starts:
        is_start[s0] = True
    R = defaultdict(lambda: defaultdict(int)); S = defaultdict(int)
    st = {"nb": 0, "nl": 0}
    cache = {}                                       # lhs -> (entropy bits, parameter bits)
    lv = math.log2(max(vocab, 2))

    def rhs(i):
        _, tok, l, r = nodes[i]
        return ("w", tok) if tok is not None else ("r", cat[l], cat[r])

    def add(i, sgn, dirty):
        a, k = cat[i], rhs(i)
        d = R[a]; before = d.get(k, 0); d[k] = before + sgn
        if d[k] == 0:
            del d[k]
            if not d:
                del R[a]
        if (before == 0) != (before + sgn == 0):
            st["nb" if k[0] == "r" else "nl"] += sgn
        dirty.add(a)
        if is_start[i]:
            S[a] += sgn
            if not S[a]:
                del S[a]

    def refresh(dirty):
        for a in dirty:
            if a in R:
                d = R[a]; n = sum(d.values())
                cache[a] = (code_entropy(d), len(d) * 0.5 * math.log2(max(n, 2)))
            else:
                cache.pop(a, None)

    def total():
        K = max(len(R), 2); lk = math.log2(K)
        return (code_entropy(S) + sum(h for h, _ in cache.values()) + sum(q for _, q in cache.values())
                + st["nb"] * 3 * lk + st["nl"] * (lk + lv))

    def relabel(change):                             # {node index: new category}
        aff = set(change)
        for i in change:
            if up[i] is not None:
                aff.add(up[i])
        dirty = set()
        for i in aff:
            add(i, -1, dirty)
        old = {}
        for i, c in change.items():
            old[i] = cat[i]; members[cat[i]].discard(i); cat[i] = c; members[c].add(i)
        for i in aff:
            add(i, +1, dirty)
        refresh(dirty)
        return old

    dirty = set()
    for i in range(len(nodes)):
        add(i, +1, dirty)
    refresh(dirty)
    best = total()
    frontier = set(cat)

    def candidates():
        return [p for p in kids if p not in frontier and all(c in frontier for c in kids[p])]

    def merge(p):
        change = {i: p for c in kids[p] for i in members.get(c, ())}
        return relabel(change)

    while True:
        trials = []
        for p in candidates():
            old = merge(p); dl = total(); relabel(old)
            if dl <= best + 1e-9:                   # ties pass unary chains
                trials.append((dl, p))
        if not trials:
            break
        trials.sort()
        applied = 0
        for _, p in trials[:64]:
            if not all(c in frontier for c in kids[p]):
                continue
            old = merge(p); dl = total()
            if dl <= best + 1e-9:
                best = dl; frontier -= set(kids[p]); frontier.add(p); applied += 1
            else:
                relabel(old)
        if not applied:
            break
    return frontier


frontier = bottom_up() if direction == "up" else top_down()
best, data, gram, rules, start = description(frontier)


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


def held_out_bits(rules, start):
    """-log2 P(sentence) under the cut grammar as a PCFG (inside algorithm,
    summing over all its analyses), per held-out sentence; None if it has
    probability zero."""
    lex = defaultdict(dict); binr = []
    for a, d in rules.items():
        n = sum(d.values())
        for k, v in d.items():
            if k[0] == "w":
                lex[k[1]][a] = v / n
            else:
                binr.append((a, k[1], k[2], v / n))
    ns = sum(start.values()); ps = {a: v / ns for a, v in start.items()}
    out = []
    for e in test:
        x = e.tokens; n = len(x)
        if n < 2 or any(w not in lex for w in x):
            out.append(None); continue
        I = {}
        for i, w in enumerate(x):
            I[(i, i + 1)] = dict(lex[w])
        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span; cell = defaultdict(float)
                for k in range(i + 1, j):
                    A, B = I[(i, k)], I[(k, j)]
                    if not A or not B:
                        continue
                    for a, b, c, q in binr:
                        if b in A and c in B:
                            cell[a] += q * A[b] * B[c]
                I[(i, j)] = cell
        pr = sum(ps.get(a, 0.0) * v for a, v in I[(0, n)].items())
        out.append(-math.log2(pr) if pr > 0 else None)
    return out


def word_models():
    uni = defaultdict(int); bi = defaultdict(lambda: defaultdict(int))
    for e in train:
        x = ["<s>"] + list(e.tokens) + ["</s>"]
        for a, b in zip(x, x[1:]):
            uni[b] += 1; bi[a][b] += 1
    N = sum(uni.values()); V = len(uni)
    def uni_bits(t):
        return -sum(math.log2(uni[w] / N) for w in list(t) + ["</s>"])
    def bi_bits(t):                                   # add-0.1 smoothing
        x = ["<s>"] + list(t) + ["</s>"]
        return -sum(math.log2((bi[a][b] + 0.1) / (sum(bi[a].values()) + 0.1 * V)) for a, b in zip(x, x[1:]))
    return uni_bits, bi_bits


uni_bits, bi_bits = word_models()
HB = held_out_bits(rules, start)
cov = [(h, e.tokens) for h, e in zip(HB, test) if h is not None]
held = {"held-out sentences": len(test), "covered": round(len(cov) / max(len(test), 1), 3),
        "grammar bits/sentence": round(sum(h for h, _ in cov) / max(len(cov), 1), 1),
        "unigram bits/sentence": round(sum(uni_bits(t) for _, t in cov) / max(len(cov), 1), 1),
        "bigram bits/sentence": round(sum(bi_bits(t) for _, t in cov) / max(len(cov), 1), 1)}
S = sample(rules, start, random.Random(seed))
gram_ok = [s for s in S if g.recognizes(s)]
K = len(rules)
salt = random.Random(seed + 1).random()
null_cat = [hash((path[-1], salt)) % K for path, _, _, _ in nodes]
nr, ns = counts(null_cat)
NS = sample(nr, ns, random.Random(seed))
print(json.dumps({
    "cond": cond, "seed": seed, "direction": direction, "target": {"nonterminals": len(tg), "rules": sum(len(v) for v in tg.values())},
    "categories": K,
    "binary rules": sum(1 for d in rules.values() for r in d if r[0] == "r"),
    "lexical rules": sum(1 for d in rules.values() for r in d if r[0] == "w"),
    "data bits per sentence": round(data / len(train), 1), "grammar bits": round(gram),
    "grammatical": round(len(gram_ok) / max(len(S), 1), 3),
    "grammatical and new": round(sum(tuple(s) not in seen for s in gram_ok) / max(len(S), 1), 3),
    "null grammatical": round(sum(1 for s in NS if g.recognizes(s)) / max(len(NS), 1), 3),
    "held out": held, "examples": [" ".join(s) + (" ✓" if g.recognizes(s) else " ✗") for s in S[:5]],
    "seconds": round(time.time() - t0)}), flush=True)
