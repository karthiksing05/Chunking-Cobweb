"""Round 211 inventory of n=1000 dumps by (family, temp, seed). Usage (repo root, compute node): r211_inventory.py"""
import glob, json, os, sys
from collections import defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, "experiments/v3"); sys.path.insert(0, "experiments/v3/diag")
import r204_fair as rf
GD = "experiments/v3/results/story_gen"
inv = defaultdict(lambda: defaultdict(set))
seenf = set()
for jl in sorted(glob.glob(f"{GD}/*frontier-n1000.jsonl")):
    for l in open(jl, encoding="utf-8"):
        r = json.loads(l)
        if r["n"] != 1000: continue
        k = (r["dir"], r["file"])
        if k in seenf: continue
        seenf.add(k)
        try: fam, t = rf.family_of(r)
        except Exception as e: fam, t = "?", 0
        arm = json.loads(r["arm"])
        extra = ",".join(f"{a}={b}" for a, b in sorted(arm.items()) if a not in ("temp", "causal", "units", "pair", "new", "cond"))
        inv[(fam, extra)][round(t, 2)].add((r["seed"], r["dir"], os.path.basename(jl)[:4]))
for (fam, extra), d in sorted(inv.items()):
    print(fam, "|", extra)
    for t, v in sorted(d.items()):
        print("   t%.2f seeds %s dirs %s src %s" % (t, sorted({s for s, _, _ in v}), sorted({x for _, x, _ in v}), sorted({x for _, _, x in v})))
