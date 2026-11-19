"""Round 218 generation report. prep: collect every generated sentence of results/story_gen/r218/<cell>_s<seed>/story_s*_n1000_arm0.jsonl into
sentences.txt (judge input, one per line, unique); report: read the judge TSV (r214_judge2.py score, ensemble Qwen3-8B + phi-4) and print per
(cell, seed): attempted draws, distinct new (novel flag, distinct strings), judge syntax pass, syntax+semantics pass among distinct new,
and distinct-new-grammatical sentences per attempted draw.   Usage (repo root, compute node): r218_gen_report.py prep | report"""
import glob
import json
import os
import re
import sys

D = "experiments/v3/results/story_gen/r218"


def rows():
    out = {}
    for d in sorted(glob.glob(D + "/*_s*")):
        if not os.path.isdir(d):
            continue
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        for f in glob.glob(d + "/story_s*_n1000_arm0.jsonl"):
            rs = [json.loads(l) for l in open(f, encoding="utf-8")]
            out[(m.group(1), int(m.group(2)))] = [r for r in rs if "gen" in r]
    return out


def prep():
    R = rows()
    sents = sorted({r["gen"] for v in R.values() for r in v})
    open(os.path.join(D, "sentences.txt"), "w", encoding="utf-8").write("\n".join(sents) + "\n")
    print({"%s_s%d" % k: len(v) for k, v in R.items()}, "unique", len(sents))


def report():
    R = rows()
    J = {}
    with open(os.path.join(D, "scores.tsv"), encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        for l in fh:
            v = l.rstrip("\n").split("\t")
            J[v[0]] = dict(zip(head[1:], v[1:]))
    print("| cell | seed | attempted | distinct new | syntax pass (distinct new) | syntax+semantics pass | distinct new syntax-pass per attempted draw | mean words |")
    print("|---|---|---|---|---|---|---|---|")
    tot = {}
    for (c, s), rs in sorted(R.items()):
        seen, new = set(), []
        for r in rs:
            if r.get("novel") and r["gen"] not in seen:
                seen.add(r["gen"])
                new.append(r["gen"])
        syn = sum(J[x]["syn_pass"] == "1" for x in new if x in J)
        both = sum(J[x]["syn_pass"] == "1" and J[x]["sem_pass"] == "1" for x in new if x in J)
        words = sum(len(x.split()) for x in new) / max(len(new), 1)
        print("| %s | %d | %d | %d | %.3f | %.3f | %.4f | %.2f |" % (c, s, len(rs), len(new), syn / max(len(new), 1), both / max(len(new), 1), syn / max(len(rs), 1), words))
        tot.setdefault(c, []).append((len(new) / max(len(rs), 1), syn / max(len(new), 1), both / max(len(new), 1), syn / max(len(rs), 1)))
    print("\nmean over seeds: cell | distinct-new share | syntax pass | syntax+semantics pass | grammatical distinct new per draw")
    for c, v in tot.items():
        print("%s | %s" % (c, " | ".join("%.3f" % (sum(x[i] for x in v) / len(v)) for i in range(4))))


if __name__ == "__main__":
    {"prep": prep, "report": report}[sys.argv[1]]()
