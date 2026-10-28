"""Case study for S4 (bAbI questions as gap filling): the story, the question,
memory's answers and the question line's parse, for the first test questions of a
few tasks. Answering as t3_babi_qa.py: the question plus "?" is the prompt, eight
gap+critic completions, the one the parser recognises best; "story" restricts
fillers to the story's words and the closed answers.

Usage: CORPUS=babi t3_babi_cases.py <cfg json> <seed> <n train> <tag> [tasks] [per task]
"""
import json
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from trellis2.babi import babi_rows, babi_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tasks = [int(t) for t in (sys.argv[5] if len(sys.argv) > 5 else "1,3,6,15").split(",")]
per = int(sys.argv[6]) if len(sys.argv) > 6 else 3
stream = [s for s in babi_sentences("train") if 3 <= len(s) <= 12]
random.Random(seed).shuffle(stream)
answers = defaultdict(Counter)
for r in babi_rows("train"):
    answers[r["task"]][tuple(r["answer"])] += 1
closed = {"yes", "no", "maybe", "none", "nothing"} | {str(i) for i in range(10)} | \
    {"one", "two", "three", "four", "five", "six", "zero"}
L = Trellis3(seed=seed, **cfg)
for s in stream[:n]:
    L.learn(s)
L._quiet = True
rng = random.Random(seed)


def answer(q, story=None, k=8):
    left = [w for w in q if w != "?"] + ["?"]
    cands = []
    for _ in range(k * (4 if story else 1)):
        g = L.generate_tree(rng, choose="leaf", fill_name="both+peer", gap=(left, []))
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
    return " ".join(best[len(left):])


def render(toks, spans):
    spans = sorted({tuple(s) for s in spans if s[1] - s[0] >= 2} | {(0, len(toks))}, key=lambda s: (s[0], -s[1]))
    opens, closes = Counter(i for i, _ in spans), Counter(j for _, j in spans)
    return " ".join("[" * opens[k] + w + "]" * closes[k + 1] for k, w in enumerate(toks))


test = defaultdict(list)
for r in babi_rows("test"):
    if r["task"] in tasks and len(test[r["task"]]) < per:
        test[r["task"]].append(r)
for task in tasks:
    for r in test[task]:
        gold = " ".join(r["answer"])
        words = {w for s in r["story"] for w in s}
        a, b = answer(r["question"]), answer(r["question"], story=words)
        a_t = tuple(a.split()) if a else None
        line = [w for w in r["question"] if w != "?"] + ["?"] + r["answer"]
        print(json.dumps({"case": "S4", "tag": tag, "seed": seed, "n": n, "task": task,
                          "story": [" ".join(s) for s in r["story"]], "question": " ".join(r["question"]),
                          "answer": a, "story-restricted": b, "gold": gold,
                          "right type": a_t in answers[task], "exact": a == gold, "story exact": b == gold,
                          "parse of the question line": render(line, L.parse(line)["tree"].brackets())}), flush=True)
