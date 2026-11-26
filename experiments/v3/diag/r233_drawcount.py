"""Round 233: draws actually written per arm in the dialgen dumps (failed draws are dropped by the driver).
usage: r233_drawcount.py <dump dir> ..."""
import glob, json, os, re, sys
for d in sys.argv[1:]:
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "story_*_arm*.jsonl"))):
        rows = [json.loads(x) for x in open(f, encoding="utf-8")][1:]
        out[re.search(r"arm(\d+)", f).group(1)] = sum(1 for r in rows if r.get("gen"))
    print(json.dumps({"dir": os.path.basename(d.rstrip("/")), "written": out}))
