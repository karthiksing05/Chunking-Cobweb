"""Case studies for the TRELLIS v2 report (evaluation only).

1. Learning one experience: the composites inside-outside weighs most, and the
   analysis memory commits to.
2. One slot (a context concept) and what it holds, at its leaf and the two
   concepts above: the fillers, and how many are grammatical substitutes.
3. Generation traces from the two hierarchies alone: each slot drawn, its
   number of options, the filler drawn.

Concepts are rendered as text through the stored records, for display only;
generation itself never reads them.

Usage: t3_case.py <cond> <cfg json> <seed> [n learned before the case]
"""
import os, sys, json, random
from collections import Counter, defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3, FILL

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
case = next(e for e in train[n_before:] if len(e.tokens) >= 6)
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

# rendering: content leaf id -> its committed chunks' text
by_cleaf = defaultdict(Counter)
for rec in L.records:
    if rec.token is None and rec.used and rec.cleaf is not None:
        by_cleaf[rec.cleaf.id][text(rec)] += 1


def render(f):
    if f[0] == "w":
        return f[1]
    c = by_cleaf.get(f[1])
    return c.most_common(1)[0][0] if c else f"<c{f[1]}>"


# 2. one slot: an object-position chunk slot with several fillers
def slot_study(node, sentence_tokens, span):
    out = []
    for level in range(3):
        fills = L.ctx.passive(node, FILL)
        chunk_fills = Counter({render(f): c for f, c in fills.items()})
        ok = sum(c for s, c in chunk_fills.items()
                 if g.recognizes(sentence_tokens[:span[0]] + s.split() + sentence_tokens[span[1]:]))
        out.append({"level above leaf": level, "concept": node.id, "fillers": len(chunk_fills),
                    "mass": round(sum(chunk_fills.values()), 1),
                    "grammatical substitutes (mass share)": round(ok / max(sum(chunk_fills.values()), 1e-9), 3),
                    "top fillers": chunk_fills.most_common(8)})
        node = node.parent or node
    return out


study = None
for rec in reversed(L.records):
    if rec.token is None and rec.used and rec.leaf is not None and len(L.ctx.passive(rec.leaf, FILL)) >= 3:
        # the experience this record came from: find a training sentence containing it
        s = text(rec).split()
        host = next((e for e in reversed(train[:n_before + 1])
                     if any(e.tokens[i:i + len(s)] == s for i in range(len(e.tokens) - len(s) + 1))), None)
        if host is None:
            continue
        i = next(i for i in range(len(host.tokens) - len(s) + 1) if host.tokens[i:i + len(s)] == s)
        study = {"sentence": " ".join(host.tokens), "slot holds": " ".join(s),
                 "levels": slot_study(rec.leaf, host.tokens, (i, i + len(s)))}
        break

# 3. generation traces
rng = random.Random(seed)
ix = L.concept_index()
traces = []
while len(traces) < 4:
    tr = []
    s = L.generate_tree(rng, index=ix, trace=tr)
    if not s:
        continue
    traces.append({"sentence": " ".join(s), "grammatical": g.recognizes(s),
                   "new": tuple(s) not in {tuple(e.tokens) for e in train},
                   "steps": [{"depth": t["depth"], "side": "left" if t["side"] == "lslot" else "right",
                              "slot": t["slot"], "slot options": t["slot options"],
                              "fillers": len(t["fillers"]), "drawn": render(t["drawn"]) if t["drawn"] else None}
                             for t in tr]})

print(json.dumps({"cond": cond, "seed": seed, "learning": learn, "slot": study, "traces": traces},
                 ensure_ascii=False, indent=1))
