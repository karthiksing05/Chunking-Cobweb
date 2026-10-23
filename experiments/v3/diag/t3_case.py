"""Case studies for the TRELLIS v2 report (evaluation only).

1. Learning one experience: the composites inside-outside weighs most, and the
   analysis memory commits to.
2. One part of a committed composite: the context concepts on its path, what
   committed instances each holds, and the concept-choice score.
3. Generation traces from the two hierarchies alone: each part, the context
   concept chosen for it, and the instance drawn.

Concepts are rendered as text through the stored records, for display only;
generation itself never reads them.

Usage: t3_case.py <cond> <cfg json> <seed> [n learned before the case]
"""
import os, sys, json, math, random
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3, FILL, LPART, RPART

cond, cfg, seed = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
n_before = int(sys.argv[4]) if len(sys.argv) > 4 else 300
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
g = CFG(target_grammar(cond))
L = Trellis3(seed=seed, **cfg)
for e in train[:n_before]:
    L.learn(e.tokens)


def text(e):
    return " ".join(L._type_key(e))


def bracket(t, toks, i, j):
    if j - i == 1:
        return toks[i]
    k = t.split[(i, j)]
    return f"[{bracket(t, toks, i, k)} {bracket(t, toks, k, j)}]"


# 1. learning one experience
longest = max(len(x.tokens) for x in train)
case = next(e for e in train[n_before:] if len(e.tokens) >= min(6, longest))
r = L.parse(case.tokens)
span_post = defaultdict(float)
for (i, k, j), p in r["post"].items():
    span_post[(i, j)] += p
top = sorted(((p, s) for s, p in span_post.items() if 1 < s[1] - s[0] < len(case.tokens)), reverse=True)[:8]
learn = {"sentence": " ".join(case.tokens),
         "top spans by posterior": [(" ".join(case.tokens[i:j]), round(p, 3)) for p, (i, j) in top],
         "committed analysis": bracket(r["tree"], case.tokens, 0, len(case.tokens)),
         "gold (scoring only)": bracket(case.tree, case.tokens, 0, len(case.tokens))}
L.learn(case.tokens)

# rendering from the trees alone: a content concept unfolds into its most
# frequent parts (the commonest place each part stood, the commonest committed
# instance there), down to primitives
CTX_IX, CNT_IX = L.concept_index()


def render(f, depth=0):
    if f[0] == "w":
        return f[1]
    x = CNT_IX.get(f[1])
    if x is None or depth > 12:
        return "?"
    words = []
    for a in (LPART, RPART):
        where = L.cnt.passive(x, a)
        if not where:
            return "?"
        leaf_id, primitive = max(where.items(), key=lambda kv: kv[1])[0]
        leaf = CTX_IX.get(leaf_id)
        fills = {k: c for k, c in L.ctx.passive(leaf, FILL).items() if (k[0] == "w") == primitive} if leaf else {}
        if not fills:
            return "?"
        words.append(render(max(fills.items(), key=lambda kv: kv[1])[0], depth + 1))
    return " ".join(words)


# 2. one part of a committed composite: the context concepts on its path, what
# committed instances each holds, how many are grammatical in its place, and the
# concept-choice score generate_tree(choose="best") maximises
window = [a for a in L.ctx_attrs if a[0] in "lr" and a[1:].isdigit()]
root_n = L.ctx.root.count
sizes = {a: len(L.ctx.root.av.get(a, {})) + 1 for a in window}


def choice_score(leaf, node):
    x = {a: max(leaf.av[a].items(), key=lambda kv: kv[1])[0] for a in window
         if isinstance(leaf.av.get(a), dict) and leaf.av[a]}
    s = math.log2(max(node.count, 1e-9) / root_n)
    for a, v in x.items():
        d = node.av.get(a, {})
        s += math.log2((d.get(v, 0.0) + L.alpha) / (node.count + L.alpha * sizes[a]))
    return s


def part_study(leaf, sentence_tokens, span, levels=6):
    out, node = [], leaf
    for level in range(levels):
        fills = L.ctx.passive(node, FILL)
        kinds = Counter({render(f): c for f, c in fills.items() if f[0] == "c"})
        ok = sum(c for s, c in kinds.items()
                 if g.recognizes(sentence_tokens[:span[0]] + s.split() + sentence_tokens[span[1]:]))
        out.append({"levels above leaf": level, "concept": node.id,
                    "choice score (bits)": round(choice_score(leaf, node), 2),
                    "composite instances": len(kinds), "mass": round(sum(kinds.values()), 1),
                    "grammatical in place (mass share)": round(ok / max(sum(kinds.values()), 1e-9), 3),
                    "top instances": kinds.most_common(8)})
        if node.parent is None:
            break
        node = node.parent
    return out


# the composite parts of the case's committed analysis, re-read after learning
r2 = L.parse(case.tokens)
best_part = None
for (i, k, j) in r2["chosen"]:
    if (i, j) == (0, len(case.tokens)):
        continue
    leaf = L.ctx.categorize(r2["comps"][(i, k, j)].ctx)[-1]
    n_inst = sum(1 for f in L.ctx.passive(leaf, FILL) if f[0] == "c")
    if best_part is None or n_inst > best_part[0]:
        best_part = (n_inst, (i, j), leaf)
study = None
if best_part:
    _, (i, j), leaf = best_part
    study = {"sentence": " ".join(case.tokens), "part": " ".join(case.tokens[i:j]),
             "levels": part_study(leaf, case.tokens, (i, j))}

# 3. generation traces from the two hierarchies alone
rng = random.Random(seed)
ix = L.concept_index()
seen = {tuple(e.tokens) for e in train}
traces = {}
for choose in ("leaf", "best"):
    traces[choose] = []
    while len(traces[choose]) < 4:
        tr = []
        s = L.generate_tree(rng, index=ix, choose=choose, trace=tr)
        if not s:
            continue
        traces[choose].append({
            "sentence": " ".join(s), "grammatical": g.recognizes(s), "new": tuple(s) not in seen,
            "steps": [{"depth": t["depth"], "part": t["part"], "primitive": t["primitive"],
                       "leaf": t["leaf"], "concept": t["concept"], "instances": len(t["instances"]),
                       "drawn": render(t["drawn"]) if t["drawn"] else None} for t in tr]})

print(json.dumps({"cond": cond, "seed": seed, "learning": learn, "part": study, "traces": traces},
                 ensure_ascii=False, indent=1))
