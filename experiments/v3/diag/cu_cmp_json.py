"""Compare two JSON-lines outputs ignoring timing fields ("learn s", "s").
Usage: cu_cmp_json.py a.json b.json"""
import json
import sys

DROP = {"learn s", "s"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in DROP and k != "cfg"}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


a = [strip(json.loads(x)) for x in open(sys.argv[1]) if x.strip()]
b = [strip(json.loads(x)) for x in open(sys.argv[2]) if x.strip()]
print(sys.argv[1], "vs", sys.argv[2], "IDENTICAL (timings ignored)" if a == b else "DIFFERENT")
if a != b:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            for k in x:
                if x.get(k) != y.get(k):
                    print(f"  line {i} {k}:\n    {json.dumps(x.get(k))[:600]}\n    {json.dumps(y.get(k))[:600]}")
