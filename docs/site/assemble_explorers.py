"""Collect the explorers' regenerated runs into docs/site/trellis_data.json (see logs/v3_x_submit.sh)."""
import glob
import json
import os

R = "experiments/v3/results/explorers"


def rows(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


curves = [r for r in rows("logs/v3-xc-*.log") if "experiences" in r]
rules = []
for r in rows("logs/v3-xr-*.log"):
    if "k1 gram+new" in r:
        rules.append({"run": r["tag"], "cond": r["cond"], "grammatical": r["k1 grammatical"], "grammatical and new": r["k1 gram+new"],
                      "distinct": r["k1 distinct"], "examples": r["k1 examples"]})
failures = [r for r in rows("logs/v3-xf-*.log") if "A-or (context concepts)" in r]
chars = []
for f in sorted(glob.glob(R + "/chars/*/results.json")):
    for r in json.load(open(f, encoding="utf-8")):
        chars.append({"run": "CURRENT", "seed": r["seed"], "experiences": r["train"], "omission": r["held-out omission"],
                      "well formed": r["well formed"], "attested": r["positions attested"], "real": r["real"], "novel": r["novel"]})
examples = {c: json.load(open(f"{R}/ex_{c}.json", encoding="utf-8")) for c in ("small", "med", "large")}
json.dump({"curves": curves, "rules": rules, "failures": failures, "characters": chars, "examples": examples},
          open("docs/site/trellis_data.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print({k: len(v) for k, v in (("curves", curves), ("rules", rules), ("failures", failures), ("characters", chars))}, "bytes", os.path.getsize("docs/site/trellis_data.json"))
