"""How compressible is the grammar TRELLIS v2 learns? (evaluation only; the
learner keeps no grammar). Project memory onto a cut of the context tree: an
element's category is its deepest ancestor holding at least m instances. Every
stored chunk adds weight to the rule  cat(chunk) -> cat(left) cat(right),
every word to  cat(word) -> word, every sentence-level chunk to  START -> cat.
Generate from the projected grammar and check the samples against the target
grammar. Larger m = fewer, coarser categories = a smaller grammar. Reports, per
cut: categories, rules and lexical entries carrying at least one unit of
weight, the grammar's size in bits (each rule names three categories, each
lexical entry a category and a word), and grammatical / new samples. A null
grammar with the same number of categories, each element assigned to one at
random (by its leaf), is projected and sampled the same way."""
import os, sys, json, math, random
from collections import defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3, BOS, EOS
cond, cfg = sys.argv[1], json.loads(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
L = Trellis3(seed=seed, **cfg)
for e in train: L.learn(e.tokens)
g = CFG(target_grammar(cond)); seen = {tuple(e.tokens) for e in train}
tg = target_grammar(cond)
print(json.dumps({"cond": cond, "target": {"nonterminals": len(tg), "rules": sum(len(v) for v in tg.values())}}), flush=True)
root = L.ctx.root

def path_of(e):
    return L.ctx.ancestors(e.leaf)[::-1] if e.leaf is not None else L.ctx.categorize(e.ctx)

def cat(e, m):
    node = root
    for n in path_of(e)[1:]:
        if n.count >= m:
            node = n
    return node.id

def draw(rng, d):
    t = rng.random() * sum(d.values())
    for k, v in d.items():
        t -= v
        if t <= 0:
            return k
    return k

def project(catf):
    rules = defaultdict(lambda: defaultdict(float)); start = defaultdict(float)
    for e in L.records:
        if e.weight <= 0:
            continue
        a = catf(e)
        if e.token is not None:
            rules[a][("w", e.token)] += 1.0
        else:
            rules[a][("r", catf(e.left), catf(e.right))] += e.weight
            if L.is_sentence(e):
                start[a] += e.weight
    return rules, start

def sample(rules, start, rng):
    S = []
    for _ in range(500):
        out, stack, ok = [], [(draw(rng, start), 0)], True
        while stack and ok:
            a, d = stack.pop()
            if d > 20 or a not in rules:
                ok = False
                break
            k = draw(rng, rules[a])
            if k[0] == "w":
                out.append(k[1])
            else:
                stack.append((k[2], d + 1)); stack.append((k[1], d + 1))
        if ok and out and len(out) <= 40:
            S.append(out)
    return S

base = root.sstot / max(root.count, 1e-9) ** 2

def cu(n):
    """Corter & Gluck (1992): the category utility of one concept."""
    return (n.count / max(root.count, 1e-9)) * (n.sstot / max(n.count, 1e-9) ** 2 - base)

def basic(e):
    p = path_of(e)[1:]
    return max(p, key=cu).id if p else root.id

cuts = [("basic", None)] + [(f, f) for f in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0)]
for frac, _ in cuts:
    m = max(frac * root.count, 1.0) if frac != "basic" else None
    catf = basic if frac == "basic" else (lambda e, m=m: cat(e, m))
    rules = defaultdict(lambda: defaultdict(float)); start = defaultdict(float)
    for e in L.records:
        if e.weight <= 0:
            continue
        a = catf(e)
        if e.token is not None:
            rules[a][("w", e.token)] += 1.0
        else:
            rules[a][("r", catf(e.left), catf(e.right))] += e.weight
            if L.is_sentence(e):
                start[a] += e.weight
    cats = {a for a, d in rules.items() if sum(d.values()) >= 1.0}
    nrule = sum(1 for d in rules.values() for k, v in d.items() if k[0] == "r" and v >= 1.0)
    nlex = sum(1 for d in rules.values() for k, v in d.items() if k[0] == "w" and v >= 1.0)
    vocab = len({k[1] for d in rules.values() for k in d if k[0] == "w"})
    bits = nrule * 3 * math.log2(max(len(cats), 2)) + nlex * (math.log2(max(len(cats), 2)) + math.log2(max(vocab, 2)))
    rng = random.Random(seed); S = []
    for _ in range(500):
        out, stack, ok = [], [(draw(rng, start), 0)], True
        while stack and ok:
            a, d = stack.pop()
            if d > 20 or a not in rules:
                ok = False
                break
            k = draw(rng, rules[a])
            if k[0] == "w":
                out.append(k[1])
            else:
                stack.append((k[2], d + 1)); stack.append((k[1], d + 1))
        if ok and out and len(out) <= 40:
            S.append(out)
    gram = [s for s in S if g.recognizes(s)]
    k = max(len(cats), 1)
    salt = random.Random(seed + 1).random()
    nr, ns = project(lambda e: hash((path_of(e)[-1].id, salt)) % k)
    NS = sample(nr, ns, random.Random(seed))
    null_gram = sum(1 for s in NS if g.recognizes(s)) / max(len(NS), 1)
    print(json.dumps({"cond": cond, "cut share": frac, "null grammatical": round(null_gram, 3), "categories": len(cats), "rules": nrule, "lexical": nlex,
                      "grammar bits": round(bits), "generated": round(len(S) / 500, 2),
                      "grammatical": round(len(gram) / max(len(S), 1), 3),
                      "grammatical and new": round(sum(tuple(s) not in seen for s in gram) / max(len(S), 1), 3),
                      "examples": [" ".join(s) + (" ✓" if g.recognizes(s) else " ✗") for s in S[:3]]}), flush=True)
