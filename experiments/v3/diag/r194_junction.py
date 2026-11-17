"""Round 194b chunk-junction pre-test: in judged dumps of a "seg" arm (each row carries the unit lengths it was drawn
in), where do the new (unstored) trigrams fall, and how grammatical are the sentences by that place?

A new trigram (a, b, c) at positions i-2..i is placed by its units:
  inside     all three words in one drawn unit
  word       it touches a single-word unit (the word-by-word step the r192 arm takes)
  chunk      it spans units that are all multi-word chunks (a pure chunk junction)
  checked / unchecked: whether c starts a unit (the r192 check reads (l2, l1, first word)) or not (the trigram
  (l1, first word, second word) of a chunk is never checked)
"edge" = a trigram with the start or end pad (as story_frontier.grams). A sentence's class: "chunk" if all its new trigrams are chunk-junction ones, "word" otherwise.
Also: gram|tri overall, the units' share of chunks, and for reference the same measures on a words-only dump.

Usage (compute node): r194_junction.py <n> <dir> [...]
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "experiments/v3")
from story_frontier import Train, grams, train_sentences  # noqa: E402

n = int(sys.argv[1])
m = lambda v: round(sum(v) / len(v), 3) if v else None
for d in sys.argv[2:]:
    agg = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, f"*_n{n}_arm*.jsonl"))):
        seed = int(re.search(r"_s(\d+)_n", f).group(1))
        T = Train(train_sentences(seed, n))
        rows = [json.loads(l) for l in list(open(f, encoding="utf-8"))[1:]]
        sc = json.load(open(f[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
        per = defaultdict(list)
        kinds = Counter()
        for r, j in zip(rows, sc):
            s = tuple(r["gen"].split())
            if s in T.whole:
                continue
            ok = j > 0.5
            lens = r.get("units") or [1] * len(s)
            unit = [-1] + [k for k, L in enumerate(lens) for _ in range(L)] + [len(lens)]   # padded, as grams()
            start = {1 + sum(lens[:k]) for k in range(len(lens) + 1)}                        # the end pad starts a unit
            per["new"].append(ok)
            per["chunk share of units"].append(m([L > 1 for L in lens]))
            t = ["<s>", *s, "</s>"]
            news = []
            for i in range(2, len(t)):
                if (t[i - 2], t[i - 1], t[i]) in T.tri:
                    continue
                us = {unit[i - 2], unit[i - 1], unit[i]}
                kind = ("inside" if len(us) == 1 else "edge" if (-1 in us or len(lens) in us) else
                        "word" if any(lens[u] == 1 for u in us) else "chunk")
                news.append(kind)
                kinds[kind] += 1
                kinds["checked" if i in start else "unchecked"] += 1
            tri = any(g not in T.tri for g in grams(s, 3))
            if tri:
                per["gram|tri"].append(ok)
                if news:
                    per["gram | new tri all chunk-junction" if all(k == "chunk" for k in news) else
                        "gram | some new tri word-junction" if "word" in news else
                        "gram | new tri inside" if "inside" in news else "gram | new tri edge+chunk only"].append(ok)
                    per["gram | new tri all checked"].append(ok) if all(
                        (i in start) for i in range(2, len(t)) if (t[i - 2], t[i - 1], t[i]) not in T.tri) else \
                        per["gram | some new tri unchecked"].append(ok)
            per["tri"].append(tri)
        out = {k: (m(v), len(v)) for k, v in per.items()}
        out["new trigram places"] = dict(kinds)
        print(f"{os.path.basename(d)} s{seed}: " + json.dumps(out), flush=True)
        for k, v in per.items():
            agg[k] += v
    print(f"{os.path.basename(d)} ALL: " + json.dumps({k: (m(v), len(v)) for k, v in agg.items()}), flush=True)
