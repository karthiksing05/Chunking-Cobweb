"""Round 192 pre-test (no generation): on judged causal dumps, would a both-sides check have kept the judged-grammatical
novel-trigram sentences and dropped the rest? For each sentence with a trigram memory never stored (Trellis3.stored_trigrams)
every such step is scored with Trellis3.pivot_support (pivot = the word before the new word, x = the new word; at a new
end, pivot = the last word, x = the end marker), kinds "pivot", "left", "both". Reported: the judged-grammatical share
and kept share of the new-trigram sentences under each rule (every step >= 0), by class of sentence.

Usage (compute node): r192_pretest.py <cfg json> <seed> <n> <dump.jsonl> [...]
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3, BOS, EOS  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dumps = sys.argv[4:]
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
train = []
for st in stories[1500:]:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, any(keep(t) for t in st[:i])))
    if len(train) >= n:
        break
train = train[:n]
L = Trellis3(seed=seed, **cfg)
for s, fol in train:
    L.learn(s, follows=fol)
L._quiet = True
W = L.window
tri = L.stored_trigrams()
pairs = set(L._pair_counts)
ends = {s[-1] for s, _ in train}
# cross-check against the stream's own trigrams
tri_s = set()
for s, _ in train:
    t = [BOS] + list(s) + [EOS]
    tri_s.update(zip(t, t[1:], t[2:]))
print(json.dumps({"seed": seed, "window": W, "stored trigrams (memory)": len(tri), "stream": len(tri_s),
                  "memory minus stream": len(tri - tri_s), "stream minus memory": len(tri_s - tri)}), flush=True)

KINDS = ("pivot", "left", "both")
res = []
for d in dumps:
    lines = [json.loads(l) for l in open(d, encoding="utf-8")]
    rows = lines[1:]
    ps = json.load(open(d[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
    for r, pj in zip(rows, ps):
        s = r["gen"].split()
        full = [BOS] * (W + 1) + s + [EOS]
        o = W + 1
        steps = []
        for i in range(len(s) + 1):            # i: the new word's index in s (len(s): the end)
            l2, l1, w = full[o + i - 2], full[o + i - 1], full[o + i]
            if l1 == BOS or (l2, l1, w) in tri:
                continue
            j = o + i - 1                      # the pivot
            left = {f"l{k}": full[j - k] for k in range(1, W + 1)}
            sc = {kd: L.pivot_support(left, [l1], [w], kind=kd)[(l1, w)] for kd in KINDS}
            steps.append({"end": w == EOS, "licensed": w != EOS and (l1, w) not in pairs,
                          "end pair stored": w != EOS or l1 in ends, **sc})
        if steps:
            res.append((pj > 0.5, steps, r["gen"]))
print(f"sentences with a new trigram: {len(res)}, judged grammatical {sum(g for g, _, _ in res) / len(res):.3f}")
rules = {"all": lambda st: True}
for kd in KINDS:
    rules[f"{kd} >= 0"] = (lambda kd: lambda st: st[kd] >= 0)(kd)
rules["no unstored bigram"] = lambda st: not st["licensed"]
rules["end only after a stored last word"] = lambda st: st["end pair stored"]
for kd in KINDS:
    rules[f"{kd} >= 0, end after stored last word"] = (lambda kd: lambda st: st[kd] >= 0 and st["end pair stored"])(kd)
cls = lambda steps: ("A unstored bigram" if any(s["licensed"] for s in steps) else
                     "B new end" if any(s["end"] for s in steps) else "C middle only")
for name, rule in rules.items():
    kept = [(g, c) for g, st, _ in res for c in [cls(st)] if all(rule(x) for x in st)]
    by = defaultdict(list)
    for g, c in kept:
        by[c].append(g)
    tail = "; ".join(f"{c} {len(v) / len(res):.3f} gram {sum(v) / len(v):.3f}" for c, v in sorted(by.items()))
    print(f"{name:42s} kept {len(kept) / len(res):.3f} gram {sum(g for g, _ in kept) / max(len(kept), 1):.3f} | {tail}")
print("\nkept by 'both >= 0, end after stored last word' and judged ungrammatical (random 30):")
bad = [(gen, st) for g, st, gen in res if not g and all(x["both"] >= 0 and x["end pair stored"] for x in st)]
random.Random(0).shuffle(bad)
for gen, st in bad[:30]:
    print("  " + gen + " | " + "; ".join(f"p{x['pivot']:+.2f} l{x['left']:+.2f}{' END' if x['end'] else ''}" for x in st))
