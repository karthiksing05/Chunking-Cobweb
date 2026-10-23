"""Export real TRELLIS v2 data for the interactive report (evaluation only).

For one grammar: learn the training experiences, then write a JSON with

* parses of held-out experiences: every span's posterior mass, the composites
  inside-outside weighs most with their gain curves (the upward traversal of
  each sorting path in both hierarchies: log2 share, log2 P(x|C) - log2 P(x|root),
  their sum), the committed analysis and the gold tree (scoring only);
* generation traces under several concept-choice rules: for every part, where
  it could stand (the composite's LPART/RPART), the leaf drawn, and every concept
  on the path above that leaf with what each rule reads there (instances of the
  part's kind, the back-off stay probability, the surroundings score, the kind
  check) and how many of its instances are grammatical in place;
* type names: the sampled context leaves of frequent types and the bag of
  concepts they give.

Concepts are shown by unfolding them from the trees (the commonest place each
part stood, the commonest instance there); records are never read.

Usage: t3_export.py <cond> <cfg json> <seed> <out.json>
"""
import os, sys, json, math, random
from collections import Counter
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3, FILL, LPART, RPART, SENT, log_prob, LN2

cond, cfg, seed, out_path = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
g = CFG(target_grammar(cond))
L = Trellis3(seed=seed, **cfg)
for e in train:
    L.learn(e.tokens)
seen = {tuple(e.tokens) for e in train}
CTX, CNT = L.concept_index()
WINDOW = [a for a in L.ctx_attrs if a[0] in "lr" and a[1:].isdigit()]


_R = {}


def render(f, depth=0):
    if depth == 0 and f in _R:
        return _R[f]
    if depth == 0:
        _R[f] = render(f, 1)
        return _R[f]
    if f[0] == "w":
        return f[1]
    x = CNT.get(f[1])
    if x is None or depth > 10:
        return "…"
    words = []
    for a in (LPART, RPART):
        where = L.cnt.passive(x, a)
        if not where:
            return "…"
        leaf_id, primitive = max(where.items(), key=lambda kv: kv[1])[0]
        leaf = CTX.get(leaf_id)
        fills = {k: c for k, c in L.ctx.passive(leaf, FILL).items() if (k[0] == "w") == primitive} if leaf else {}
        if not fills:
            return "…"
        words.append(render(max(fills.items(), key=lambda kv: kv[1])[0], depth + 1))
    return " ".join(words)


def surroundings(node):
    """The commonest value of l1 and r1 at a context concept, with its share."""
    out = {}
    for a in ("l1", "r1"):
        d = node.av.get(a, {})
        if isinstance(d, dict) and d:
            v, c = max(d.items(), key=lambda kv: kv[1])
            out[a] = [v, round(c / max(node.count, 1e-9), 3)]
    return out


def bracket(t, i, j):
    if j - i == 1:
        return i
    k = t.split[(i, j)]
    return [bracket(t, i, k), bracket(t, k, j)]


# ---------------------------------------------------------------- parsing --
def gain_curve(tree, x):
    root = tree.root
    base = log_prob(root, x, root, L.alpha)
    curve = []
    for depth, node in enumerate(tree.categorize(x)):
        share = math.log2(max(node.count, 1e-9) / root.count)
        fit = (log_prob(node, x, root, L.alpha) - base) / LN2
        curve.append({"depth": depth, "concept": node.id, "count": round(node.count, 1),
                      "log2 share": round(share, 2), "fit": round(fit, 2), "gain": round(share + fit, 2)})
    return curve


def parse_case(tokens, tree_gold):
    r = L.parse(tokens)
    n = len(tokens)
    span_post = Counter()
    for (i, k, j), p in r["post"].items():
        span_post[(i, j)] += p
    comps = sorted(r["comps"].items(), key=lambda kv: -r["post"].get(kv[0], 0.0))[:30]
    rows = []
    chosen = set(r["chosen"])
    for (i, k, j), e in comps:
        cc, nc = gain_curve(L.ctx, e.ctx), gain_curve(L.cnt, e.cnt)
        rows.append({"i": i, "k": k, "j": j, "text": " ".join(tokens[i:j]),
                     "left": " ".join(tokens[i:k]), "right": " ".join(tokens[k:j]),
                     "posterior": round(r["post"].get((i, k, j), 0.0), 4),
                     "gain": round(e.gain, 2), "committed": (i, k, j) in chosen,
                     "context path": cc, "content path": nc})
    return {"tokens": tokens,
            "spans": [{"i": i, "j": j, "mass": round(p, 4)} for (i, j), p in span_post.items()],
            "composites": rows,
            "committed": bracket(r["tree"], 0, n), "gold": bracket(tree_gold, 0, n)}


cases = [e for e in test if len(e.tokens) >= 4]
cases = sorted(cases, key=lambda e: len(e.tokens))
pick = [cases[len(cases) // 5], cases[len(cases) // 2], cases[-1]] if len(cases) >= 3 else cases
parses = [parse_case(e.tokens, e.tree) for e in pick]

# ------------------------------------------------------------- generation --
RULES = [{"name": "leaf", "args": {"choose": "leaf"}},
         {"name": "back-off", "args": {"choose": "backoff", "beta": 1.0}},
         {"name": "back-off + kind", "args": {"choose": "backoff", "beta": 1.0, "kind": 1, "primitive_leaf": True}},
         {"name": "three levels up + kind", "args": {"choose": "leaf", "level": 3, "kind": 1, "primitive_leaf": True}}]
root_n = max(L.ctx.root.count, 1e-9)
sizes = {a: len(L.ctx.root.av.get(a, {})) + 1 for a in WINDOW}


def surround_score(leaf, node):
    x = {a: max(leaf.av[a].items(), key=lambda kv: kv[1])[0] for a in WINDOW
         if isinstance(leaf.av.get(a), dict) and leaf.av[a]}
    s = math.log2(max(node.count, 1e-9) / root_n)
    for a, v in x.items():
        d = node.av.get(a, {})
        s += math.log2((d.get(v, 0.0) + L.alpha) / (node.count + L.alpha * sizes[a]))
    return s


def kinds_of(leaf, k=1):
    out = set()
    for f in L.ctx.passive(leaf, FILL):
        if f[0] == "c" and f[1] in CNT:
            y = CNT[f[1]]
            for _ in range(k):
                y = y.parent or y
            out.add(y.id)
    return out


def under(cid, kinds):
    y = CNT.get(cid)
    while y is not None:
        if y.id in kinds:
            return True
        y = y.parent
    return False


def path_table(step, sentence, beta=1.0, max_levels=9):
    leaf = CTX[step["leaf"]]
    prim = step["primitive"]
    kinds = kinds_of(leaf)
    s0, s1 = step["start"], step.get("end", step["start"])
    rows, node, level = [], leaf, 0
    while node is not None and level < max_levels:
        fills = {f: c for f, c in L.ctx.passive(node, FILL).items() if (f[0] == "w") == prim}
        n = sum(fills.values())
        kind_fills = fills if prim else {f: c for f, c in fills.items() if under(f[1], kinds)}
        top = sorted(fills.items(), key=lambda kv: -kv[1])[:8]
        texts = [(render(f), c) for f, c in top]
        ok, checked = 0.0, 0.0
        for f, c in sorted(fills.items(), key=lambda kv: -kv[1])[:150]:
            checked += c
            t = render(f).split()
            if "…" not in t and g.recognizes(sentence[:s0] + t + sentence[s1:]):
                ok += c
        rows.append({"level": level, "concept": node.id, "count": round(node.count, 1),
                     "instances": len(fills), "mass": round(n, 2),
                     "stay probability": round(n / (n + beta), 3) if n > 0 else 0.0,
                     "surroundings score": round(surround_score(leaf, node), 2),
                     "kind-matching mass": round(sum(kind_fills.values()), 2),
                     "grammatical in place": round(ok / checked, 3) if checked else None,
                     "checked mass share": round(checked / n, 3) if n else None,
                     "surroundings": surroundings(node),
                     "top instances": texts})
        node, level = node.parent, level + 1
    return rows


gens = []
for rule in RULES:
    rng = random.Random(seed)
    samples, tries = [], 0
    while len(samples) < 5 and tries < 200:
        tries += 1
        tr = []
        s = L.generate_tree(rng, index=(CTX, CNT), trace=tr, **rule["args"])
        if not s:
            continue
        steps = []
        for t in tr:
            leaf = CTX[t["leaf"]]
            # where this part could stand: the composite's LPART / RPART options
            x = CNT[t["composite"]]
            where = L.cnt.passive(x, LPART if t["part"] == "left" else RPART)
            opts = sorted(where.items(), key=lambda kv: -kv[1])[:6]
            chosen_level = 0
            y = leaf
            while y is not None and y.id != t["concept"]:
                y, chosen_level = y.parent, chosen_level + 1
            steps.append({"depth": t["depth"], "part": t["part"], "primitive": t["primitive"],
                          "composite": render(("c", t["composite"])),
                          "start": t["start"], "end": t.get("end", t["start"]),
                          "where options": [{"leaf": o[0][0], "primitive": o[0][1], "count": round(o[1], 2),
                                             "surroundings": surroundings(CTX[o[0][0]]) if o[0][0] in CTX else {}}
                                            for o in opts],
                          "leaf": t["leaf"], "chosen level": chosen_level, "concept": t["concept"],
                          "drawn": render(t["drawn"]) if t["drawn"] else None,
                          "path": path_table(t, s, beta=rule["args"].get("beta", 1.0))})
        samples.append({"sentence": s, "grammatical": g.recognizes(s), "new": tuple(s) not in seen,
                        "steps": steps})
    gens.append({"rule": rule["name"], "args": rule["args"], "samples": samples})

# --------------------------------------------------------------- type names --
def concept_info(key):
    """Depth below the root and size of a named context concept ("c<id>")."""
    try:
        node = CTX[int(str(key).lstrip("c"))]
    except (ValueError, KeyError):
        return None
    return {"depth": len(L.ctx.ancestors(node)) - 1, "count": round(node.count, 1), "surroundings": surroundings(node)}


types = []
if L.type_names == "sample":
    words = Counter(t for e in train for t in e.tokens)
    for w, _ in words.most_common(40)[::5][:6]:
        res = L.type_res.get((w,), [])
        bag = Counter()
        sampled = []
        for _, leaf in res:
            ids = L._path_ids(L.ctx.ancestors(leaf)[::-1])
            for v in ids:
                bag[v] += 1.0 / (len(ids) * len(res))
            sampled.append({"leaf": leaf.id, "surroundings": surroundings(leaf), "path length": len(ids),
                            "path": sorted(set(ids))})
        types.append({"type": w, "occurrences": words[w], "sampled leaves": sampled,
                      "bag top": [[k, round(v, 4)] for k, v in bag.most_common(12)],
                      "bag concepts": {k: concept_info(k) for k, _ in bag.most_common(12)}})

summary = {}
for rule in RULES:
    rng = random.Random(seed + 1)
    S = [s for s in (L.generate_tree(rng, index=(CTX, CNT), **rule["args"]) for _ in range(300)) if s]
    summary[rule["name"]] = {"grammatical": round(sum(g.recognizes(s) for s in S) / max(len(S), 1), 3),
                             "grammatical and new": round(sum(g.recognizes(s) and tuple(s) not in seen for s in S) / max(len(S), 1), 3),
                             "distinct": round(len(set(map(tuple, S))) / max(len(S), 1), 3)}

json.dump({"grammar": cond, "seed": seed, "config": cfg, "train size": len(train),
           "trees": {"context concepts": len(CTX), "content concepts": len(CNT)},
           "parses": parses, "generation": gens, "rule summary": summary, "types": types},
          open(out_path, "w", encoding="utf-8"), ensure_ascii=False, default=str)
print("wrote", out_path)
