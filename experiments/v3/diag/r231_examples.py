"""r231: first k generations (file order) of each regime judged in one arm dump. usage: r231_examples.py <arm.jsonl> <k>"""
import json
import sys

sys.path.insert(0, "experiments/v3/diag")
import r230_grammar as G  # noqa: E402

arm, rows = G.read_arm(sys.argv[1])
k = int(sys.argv[2])
out = {"sem": [], "syn": [], "non": []}
for r in rows:
    j = G.judge(r["gen"].split())
    if len(out[j[0]]) < k:
        out[j[0]].append([r["gen"], j[1], r["novel"]])
print(json.dumps({"file": sys.argv[1], "arm": arm, "first": out}))
