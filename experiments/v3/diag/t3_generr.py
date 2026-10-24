"""What goes wrong in ungrammatical generations? (evaluation only)

Generates with the leaf rule and a trace; for each ungrammatical sentence prints
every step: the slot (the stored part the composite's statistics pointed to: its
words in its source sentence and its gold label) and what was drawn into it (a
word, or a stored composite with its words and gold label), plus the produced
substring's gold categories. Gold only scores. Also counts, over all steps of
ungrammatical sentences, how often the drawn item's gold label differs from the
slot's.

Usage: t3_generr.py <cond> <cfg json> [seed] [n] [tag]
"""
import os, sys, json, random
from collections import Counter
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3

cond, cfg = sys.argv[1], json.loads(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
tag = sys.argv[5] if len(sys.argv) > 5 else ""
fill = sys.argv[6] if len(sys.argv) > 6 else False
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
g = CFG(target_grammar(cond))
L = Trellis3(seed=seed, **cfg)
src, label = {}, {}
for e in train:
    n0 = len(L.records)
    L.learn(e.tokens)
    gl = g.gold_labels(e.tokens, e.tree)
    for r in L.records[n0:]:
        src[r.rid] = e.tokens
        label[r.rid] = gl.get(r.span, "-")


def words(rid):
    r = L.records[rid]
    i, j = r.span
    return " ".join(src[rid][i:j])


def cats(toks):
    ch = g.chart(toks)
    return sorted(ch.get((0, len(toks)), []))


ix = L.concept_index()
rng = random.Random(seed)
bad, steps, mism, kinds = 0, 0, 0, Counter()
shown = 0
for _ in range(n):
    tr = []
    s = L.generate_tree(rng, index=ix, choose="leaf", trace=tr, fill_name=fill)
    if not s or g.recognizes(s):
        continue
    bad += 1
    lines = []
    for t in tr:
        slot = t["slot"]
        f = t["drawn"]
        got = f[1] if f[0] == "w" else words(f[1])
        lab_d = "word" if f[0] == "w" else label[f[1]]
        lab_s = label[slot]
        sub = s[t["start"]:t.get("end", len(s))]
        steps += 1
        if f[0] == "c" and lab_d != lab_s:
            mism += 1
            kinds[(lab_s, lab_d)] += 1
        lines.append(f"  d{t['depth']} {t['part']:5s} slot=[{words(slot)}]:{lab_s}  drawn=[{got}]:{lab_d}  -> '{' '.join(sub)}' {cats(sub)}")
    if shown < 12:
        print("✗", " ".join(s))
        print("\n".join(lines))
        shown += 1
print(json.dumps({"tag": tag, "cond": cond, "seed": seed, "ungrammatical": bad, "steps": steps,
                  "composite drawn with a label differing from the slot's": mism,
                  "top (slot, drawn) label pairs": [[list(k), v] for k, v in kinds.most_common(10)]}), flush=True)
