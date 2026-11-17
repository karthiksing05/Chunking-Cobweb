"""Round 192: compact case lists from r192_trace.py outputs (class, judge, novel trigrams, both-sides ratio per step).
Usage: r192_cases.py <k per class> <trace.jsonl> [...]"""
import json
import random
import sys
from collections import defaultdict

k = int(sys.argv[1])
rows = [json.loads(l) for f in sys.argv[2:] for l in open(f, encoding="utf-8")]
rows = [r for r in rows if "steps" in r]


def cls(r):
    if any(s["gate"] == "licensed" and not s["end"] for s in r["steps"]):
        return "A licensed unstored bigram"
    if any(s["end"] for s in r["steps"]):
        return "B novel end"
    return "C middle trigram(s) only"


by = defaultdict(list)
for r in rows:
    ok = min(s["both log ratio"] for s in r["steps"]) >= 0
    by[(cls(r), ok)].append(r)
for key in sorted(by):
    rs = by[key]
    g = sum(r["judge"] > 0.5 for r in rs) / len(rs)
    print(f"\n=== {key[0]} | both-sides passes every step: {key[1]} | {len(rs)} sentences ({len(rs) / len(rows):.3f}), judged grammatical {g:.3f}")
    bad = [r for r in rs if r["judge"] <= 0.5]
    seen = set()
    random.Random(1).shuffle(bad)
    for r in bad:
        if r["gen"] in seen:
            continue
        seen.add(r["gen"])
        st = "; ".join(f"{s['l2']} {s['l1']} [{s['w']}] r={s['both log ratio']:+.2f}" for s in r["steps"])
        print(f"  {r['judge']:.3f} | {r['gen']} | {st}")
        if len(seen) >= k:
            break
