"""Round 193 pre-test: two-sided Gibbs repair (Trellis3.gibbs_repair) applied to already generated, judged sentences.
For each sentence of a judged r192 dump (the "both" arm), the learner (rebuilt as in t3_story.py) repairs it; the
original and the repaired sentences are written as two arms of a new dump dir (same order, refs copied), to be judged
by t3_judge.py and scored by story_frontier.py. Also prints how often the repair changes a sentence, split by the judge's
verdict on the original.

Usage (compute node): r193_pretest.py <cfg json> <seed> <n> <temp> <in dump.jsonl> <out dir> [targeted]
"""
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, temp, src, out = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
TARGETED = len(sys.argv) > 7 and sys.argv[7] == "targeted"
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
seen = {tuple(s) for s, _ in train}
L = Trellis3(seed=seed, **cfg)
for s, fol in train:
    L.learn(s, follows=fol)
L._quiet = True

lines = [json.loads(l) for l in open(src, encoding="utf-8")]
arm, rows = lines[0]["arm"], lines[1:]
ps = json.load(open(src[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
rng = random.Random(seed + 41)
t0 = time.time()
rep, sweeps = [], []
for r in rows:
    o, k = L.gibbs_repair(r["gen"].split(), rng, temp=temp, targeted=TARGETED)
    rep.append(o)
    sweeps.append(k)
os.makedirs(out, exist_ok=True)
base = os.path.basename(src).split("_arm")[0]
shutil.copy(os.path.join(os.path.dirname(src), base + "_refs.jsonl"), os.path.join(out, base + "_refs.jsonl"))
for ai, (tag, sents) in enumerate((("original", [r["gen"].split() for r in rows]), ("repaired", rep))):
    with open(os.path.join(out, f"{base}_arm{ai}.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"arm": {**arm, "pre": tag}}) + "\n")
        for r, g in zip(rows, sents):
            fh.write(json.dumps({"prompt": r["prompt"], "gen": " ".join(g), "pattern": False, "novel": tuple(g) not in seen}) + "\n")
ch = [" ".join(o) != r["gen"] for o, r in zip(rep, rows)]
good = [p > 0.5 for p in ps]
m = lambda v: round(sum(v) / max(len(v), 1), 3)
print(json.dumps({"seed": seed, "window": L.window, "rows": len(rows), "repair s": round(time.time() - t0),
                  "changed": m(ch), "changed | judged grammatical": m([c for c, g in zip(ch, good) if g]),
                  "changed | judged ungrammatical": m([c for c, g in zip(ch, good) if not g]),
                  "became stored": m([tuple(o) in seen for o in rep]), "mean sweeps": m(sweeps)}), flush=True)
k = 0
for o, r, p in zip(rep, rows, ps):
    if " ".join(o) != r["gen"] and k < 40:
        print(f"  {p:.3f} | {r['gen']}  ->  {' '.join(o)}")
        k += 1
