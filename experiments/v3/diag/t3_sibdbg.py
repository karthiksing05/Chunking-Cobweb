"""Debug sibling-mode prompting on bAbI: how question-answer lines parse, and which
stored experiences the sibling posterior prefers for a question prompt."""
import json
import os
import random
import sys
sys.path.insert(0, "src")
os.environ["CORPUS"] = "babi"
from trellis2.stories import load_corpus  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
corpus = load_corpus()
random.Random(seed).shuffle(corpus)
L = Trellis3(seed=seed, **cfg)
for s in corpus[:n]:
    L.learn(s)
for s in (["where", "is", "mary", "?", "bathroom"], ["is", "john", "in", "the", "kitchen", "?", "no"],
          ["mary", "moved", "to", "the", "bathroom"]):
    print(" ".join(s), "->", L.parse(s)["tree"].to_string(s))
for q in (["where", "is", "mary", "?"], ["is", "john", "in", "the", "kitchen", "?"]):
    tr = []
    g = L.generate_tree(random.Random(1), choose="leaf", fill_name="both+peer", gap=(q, []), gap_mode="sibling", trace=tr)
    print(" ".join(q), "=>", g, json.dumps(tr[0])[:600] if tr else "")
    tr = []
    g = L.generate_tree(random.Random(1), choose="leaf", fill_name="both+peer", gap=(q, []), trace=tr)
    print("   window =>", g, json.dumps(tr[0])[:400] if tr else "")
