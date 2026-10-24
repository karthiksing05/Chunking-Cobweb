"""T1 of the representation brief: audit bag mass (evaluation only).

Per attribute of each tree, over the stored records' current descriptions:
mean, spread (sd / mean), min and max of the mass per instance (sum of a bag's
weights, 1 for a single value), and at the root: mean mass per instance
(sum of value counts / count), observed share (na / count), live values, and
values whose count is below 1e-6 (dead values still counted in K).

Usage: t3_mass.py <cond> <cfg json> [seed] [tag]
"""
import os, sys, json, math
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.trellis3 import Trellis3

cond, cfg = sys.argv[1], json.loads(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
tag = sys.argv[4] if len(sys.argv) > 4 else ""
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
L = Trellis3(seed=seed, **cfg)
for e in train:
    L.learn(e.tokens)


def audit(tree, field):
    out = {}
    for a in tree.attrs:
        m = []
        for r in L.records:
            x = getattr(r, field)
            if x and a in x:
                v = x[a]
                m.append(sum(v.values()) if isinstance(v, dict) else 1.0)
        root = tree.root
        d = root.av.get(a, {})
        mu = sum(m) / len(m) if m else 0.0
        sd = math.sqrt(sum((u - mu) ** 2 for u in m) / len(m)) if m else 0.0
        out[a] = {"instances": len(m), "mass mean": round(mu, 4), "mass sd/mean": round(sd / mu, 4) if mu else None,
                  "mass min": round(min(m), 4) if m else None, "mass max": round(max(m), 4) if m else None,
                  "root mass per instance": round(sum(d.values()) / max(root.count, 1e-9), 4),
                  "root observed share": round(root.na.get(a, 0.0) / max(root.count, 1e-9), 4),
                  "root live values": sum(1 for c in d.values() if c > 1e-6),
                  "root values below 1e-6": sum(1 for c in d.values() if c <= 1e-6)}
    return out


print(json.dumps({"tag": tag, "cond": cond, "seed": seed, "ctx": audit(L.ctx, "ctx"), "cnt": audit(L.cnt, "cnt")}), flush=True)
