"""Round 225 generation report (the r218 pipeline: r212 gate arm, Qwen3-8B + phi-4 judge2).
Dumps: experiments/v3/results/story_gen/r225/<arm>_s<seed>/story_s<seed>_n1000_arm<i>.jsonl, arm i = temperature TEMPS[i].
  prep    collect every generated sentence into r225/sentences.txt (judge input, unique)
  report  read r225/scores.tsv (r214_judge2.py score) and write r225/gen_rows.tsv (one row per arm, temp, seed) and r225/gen_table.md
Per row: attempted draws (ATT = 1600 prompts x draws; a draw that gave no sentence still counts), distinct new sentences (novel flag, distinct
strings), judge syntax pass, syntax and semantics pass (among distinct new), and the fair counts: distinct new syntax-pass per attempted draw,
distinct new syntax+semantics-pass per attempted draw.
Usage (repo root, compute node, env trellis):  r225_gen_report.py prep | report
"""
import glob
import json
import os
import re
import sys

D = "experiments/v3/results/story_gen/r225"
TEMPS = [1.2, 2.0, 3.0]
ATT = 1600
ARMS = ["full", "nochunk", "noconcept", "neither", "noconceptp"]


def rows():
    out = {}
    for d in sorted(glob.glob(D + "/*_s*")):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        if not os.path.isdir(d) or not m:
            continue
        for f in glob.glob(d + "/story_s*_n1000_arm*.jsonl"):
            ai = int(re.search(r"_arm(\d+)\.jsonl$", f).group(1))
            rs = [json.loads(l) for l in open(f, encoding="utf-8")]
            out[(m.group(1), TEMPS[ai], int(m.group(2)))] = [r for r in rs if "gen" in r]
    return out


def prep():
    R = rows()
    sents = sorted({r["gen"] for v in R.values() for r in v})
    open(os.path.join(D, "sentences.txt"), "w", encoding="utf-8").write("\n".join(sents) + "\n")
    print({"%s_t%g_s%d" % k: len(v) for k, v in R.items()}, "unique", len(sents))


def report():
    R = rows()
    J = {}
    with open(os.path.join(D, "scores.tsv"), encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        for l in fh:
            v = l.rstrip("\n").split("\t")
            J[v[0]] = dict(zip(head[1:], v[1:]))
    cols = ["arm", "temp", "seed", "rows", "distinct_new", "syn_pass", "syn_sem_pass", "fair_syn", "fair_syn_sem", "mean_words"]
    recs = []
    for (a, t, s), rs in sorted(R.items()):
        seen, new = set(), []
        for r in rs:
            if r.get("novel") and r["gen"] not in seen:
                seen.add(r["gen"])
                new.append(r["gen"])
        syn = sum(J[x]["syn_pass"] == "1" for x in new if x in J)
        both = sum(J[x]["syn_pass"] == "1" and J[x]["sem_pass"] == "1" for x in new if x in J)
        recs.append(dict(arm=a, temp=t, seed=s, rows=len(rs), distinct_new=len(new), syn_pass=syn / max(len(new), 1), syn_sem_pass=both / max(len(new), 1),
                         fair_syn=syn / ATT, fair_syn_sem=both / ATT, mean_words=sum(len(x.split()) for x in new) / max(len(new), 1)))
    with open(os.path.join(D, "gen_rows.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in recs:
            fh.write("\t".join(("%.4f" % r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    out = ["| arm | temp | seed | sentences returned | distinct new | syntax pass (of distinct new) | syntax+semantics pass | fair syntax (per %d draws) | fair syntax+semantics | mean words |" % ATT,
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        out.append("| %s | %g | %d | %d | %d | %.3f | %.3f | %.4f | %.4f | %.2f |" % (r["arm"], r["temp"], r["seed"], r["rows"], r["distinct_new"], r["syn_pass"],
                                                                                   r["syn_sem_pass"], r["fair_syn"], r["fair_syn_sem"], r["mean_words"]))
    out.append("\nMean over seeds:\n")
    out.append("| arm | temp | seeds | returned | distinct new share of draws | syntax pass | syntax+semantics pass | fair syntax | fair syntax+semantics |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for t in TEMPS:
            v = [r for r in recs if r["arm"] == a and r["temp"] == t]
            if not v:
                continue
            m = lambda k: sum(r[k] for r in v) / len(v)
            out.append("| %s | %g | %d | %.0f | %.3f | %.3f | %.3f | %.4f | %.4f |" % (a, t, len(v), m("rows"), m("distinct_new") / ATT, m("syn_pass"), m("syn_sem_pass"), m("fair_syn"), m("fair_syn_sem")))
    open(os.path.join(D, "gen_table.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"prep": prep, "report": report}[sys.argv[1]]()
