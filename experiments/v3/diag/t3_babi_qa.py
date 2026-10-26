"""bAbI answers as gap filling (docs/BABI.md). After learning the bAbI stream
(statements and "<question> ? <answer>" lines, shuffled), each test question is the
prompt q + ["?"] and the answer is the generated gap (gap+critic: 8 completions, the
one the parser recognises best). Per task:

  type accuracy    the answer is one of the answers that task has in training (a
                   location for "where", a person for "who", yes/no, a number...)
  exact accuracy   the answer equals the gold answer (no story: chance-level expected)
  sibling          gap_mode="sibling": the answer generated as the right part of a stored
                   experience whose left part matches the parsed question chunk
  story            the same with fillers restricted to the story's words (and the
                   task's closed answers: yes / no / maybe / none / numbers)
Baseline: an answer drawn from all training answers by frequency (type and exact).

Usage: t3_babi_qa.py <cfg json> <seed> <n train> <tag> [per task]
"""
import json
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from trellis2.babi import babi_rows, babi_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
per = int(sys.argv[5]) if len(sys.argv) > 5 else 25
stream = [s for s in babi_sentences("train") if 3 <= len(s) <= 12]
random.Random(seed).shuffle(stream)
train_rows = babi_rows("train")
answers = defaultdict(Counter)
for r in train_rows:
    answers[r["task"]][tuple(r["answer"])] += 1
all_ans = Counter()
for t, c in answers.items():
    all_ans.update(c)
closed = {"yes", "no", "maybe", "none", "nothing"} | {str(i) for i in range(10)} | \
    {"one", "two", "three", "four", "five", "six", "zero"}
L = Trellis3(seed=seed, **cfg)
t0 = time.time()
for s in stream[:n]:
    L.learn(s)
learn_s = time.time() - t0
rng = random.Random(seed)
test = defaultdict(list)
for r in babi_rows("test"):
    if len(test[r["task"]]) < per:
        test[r["task"]].append(r)


def answer(q, story=None, k=8, mode="window"):
    left = [w for w in q if w != "?"] + ["?"]
    cands = []
    for _ in range(k * (4 if story else 1)):
        g = L.generate_tree(rng, choose="leaf", fill_name="both+peer", gap=(left, []), gap_mode=mode)
        if not g:
            continue
        a = g[len(left):]
        if story is not None and not all(w in story or w in closed for w in a):
            continue
        cands.append(g)
        if len(cands) >= k:
            break
    if not cands:
        return None
    best = max(cands, key=lambda g: L.parse(g)["logZ"] / max(len(g) - 1, 1))
    return tuple(best[len(left):])


res = {"tag": tag, "seed": seed, "n": n, "learn s": round(learn_s), "tasks": {}}
tot = Counter()
for task in sorted(test):
    c = Counter()
    ex = []
    for r in test[task]:
        gold = tuple(r["answer"])
        a = answer(r["question"])
        words = {w for s in r["story"] for w in s}
        b = answer(r["question"], story=words)
        sib = answer(r["question"], mode="sibling")
        sb = answer(r["question"], story=words, mode="sibling")
        base = rng.choices(list(all_ans), weights=list(all_ans.values()))[0]
        c["n"] += 1
        c["type"] += a in answers[task]
        c["exact"] += a == gold
        c["story type"] += b in answers[task]
        c["story exact"] += b == gold
        c["sibling type"] += sib in answers[task]
        c["sibling exact"] += sib == gold
        c["sibling story type"] += sb in answers[task]
        c["sibling story exact"] += sb == gold
        c["base type"] += base in answers[task]
        c["base exact"] += base == gold
        if len(ex) < 3:
            ex.append(f"{' '.join(r['question'])} -> {' '.join(a) if a else 'FAIL'} / story: {' '.join(b) if b else 'FAIL'} / sibling: {' '.join(sib) if sib else 'FAIL'} (gold {' '.join(gold)})")
    tot.update(c)
    row = {k: round(v / c["n"], 3) for k, v in c.items() if k != "n"}
    row["examples"] = ex
    res["tasks"][task] = row
    print(json.dumps({"task": task, **row}), flush=True)
res["all"] = {k: round(v / tot["n"], 3) for k, v in tot.items() if k != "n"}
print(json.dumps(res))
