"""r196: are the factor parts of a repeated string stable across seeds? Pearson r over strings of the per-string mean."""
import json, sys, math
tag, ci = sys.argv[1], int(sys.argv[2])
R = {s: json.load(open(f"experiments/v3/results/story_parse_consist/{tag}_s{s}.json"))["checkpoints"][ci]["diag"] for s in (13, 17, 19)}
def corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb) if va and vb else float("nan")
keys = sorted(R[13])
for c in ("coh", "ctx", "post", "in"):
    m = {s: [sum(float(r[c]) for r in R[s][k]) / len(R[s][k]) for k in keys] for s in R}
    print(tag, ci, c, "r13-17 %.3f r13-19 %.3f r17-19 %.3f" % (corr(m[13], m[17]), corr(m[13], m[19]), corr(m[17], m[19])))
# per occurrence: same decision across seeds?
agree = tot = 0
for k in keys:
    for a, b, c in zip(R[13][k], R[17][k], R[19][k]):
        tot += 1; agree += a["in"] == b["in"] == c["in"]
print(tag, ci, "occurrence decisions identical in all 3 seeds %.3f" % (agree / tot))
