"""Round 198: gold ceiling for the parse 'consistency' / 'decisive' statistics of story_parse_eval.py.
Uses the same held-out sentences and the same strings (2-4 word proper sub-spans with >= 3 occurrences) as story_parse_eval,
reading each per-seed JSON's "held" and "gold_phrase" (the gold shallow-phrase spans) and each method's stored brackets.
The method statistics are recomputed here and checked against the stored metrics (max abs diff reported).
Usage (repo root, compute node): story_parse_gold_ceiling.py <out.md>
"""
import json
import sys
from collections import defaultdict

D = "experiments/v3/results/story_parse_eval/"
RUNS = [("TRELLIS default (r185)", "r185", ["trellis_chosen", "trellis_post"]),
        ("TRELLIS parts (r196parts)", "r196parts", ["trellis_chosen", "trellis_post"]),
        ("TRELLIS parts+edge_once (r196pe)", "r196pe", ["trellis_chosen", "trellis_post"]),
        ("BPE x1 (r185)", "r185", ["bpe_x1"]), ("right-branching", "r185", ["right_branching"])]
SEEDS = [13, 17, 19]
CPS = [250, 1000, 4000]


def strings_of(held):
    occ = defaultdict(list)
    for h, s in enumerate(held):
        for i in range(len(s)):
            for j in range(i + 2, min(len(s) - 1, i + 4) + 1):
                occ[tuple(s[i:j])].append((h, i, j))
    return {k: v for k, v in occ.items() if len(v) >= 3}


def stat(brs, strings):
    tot = hit = dec = 0
    for v in strings.values():
        b = sum((i, j) in brs[h] for h, i, j in v)
        tot += len(v)
        hit += b
        dec += max(b, len(v) - b)
    return hit / tot, dec / tot


def mean(v):
    return sum(v) / len(v)


def main(out):
    res = defaultdict(dict)       # (label, cp, method) -> seed -> (cons, dec)
    gold = defaultdict(dict)      # cp -> seed -> (cons, dec)
    chk = 0.0
    for label, tag, methods in RUNS:
        for s in SEEDS:
            J = json.load(open("%s%s_s%d.json" % (D, tag, s), encoding="utf-8"))
            held = [tuple(h) for h in J["held"]]
            st = strings_of(held)
            assert len(st) == J["strings"], (tag, s, len(st), J["strings"])
            g = [set(map(tuple, x)) for x in J["gold_phrase"]]
            gc = stat(g, st)
            for ck in J["checkpoints"]:
                cp = ck["learned"]
                gold[cp][s] = gc
                for m in methods:
                    brs = [set(map(tuple, b)) for b in ck["methods"][m]["brackets"]]
                    c, d = stat(brs, st)
                    mt = ck["methods"][m]["metrics"]
                    chk = max(chk, abs(c - mt["consistency"]), abs(d - mt["decisive"]))
                    res[(label, cp, m)][s] = (c, d)
    L = ["Round 198B: gold ceiling for parse consistency / decisive (mean over seeds 13/17/19; [per-seed]).",
         "Strings: 2-4 word proper sub-spans of the 300 held-out sentences with >= 3 occurrences (story_parse_eval's rule); "
         "consistency = share of occurrences bracketed; decisive = mean of max(bracketed, unbracketed) share.",
         "Gold = the NLTK shallow-phrase spans (NP/PP/VP/AP/VG, >= 2 words, not the whole sentence) used as the F1 reference.",
         "Recomputed method statistics match the stored metrics (max abs diff %.1e).\n" % chk]
    for cp in CPS:
        if cp not in gold:
            continue
        gc = mean([gold[cp][s][0] for s in SEEDS]); gd = mean([gold[cp][s][1] for s in SEEDS])
        L.append("### %d sentences learned\n" % cp)
        L.append("| method | consistency | decisive | abs(consistency - gold) | abs(decisive - gold) |")
        L.append("|---|---|---|---|---|")
        L.append("| GOLD shallow phrases | %.3f [%s] | %.3f [%s] | 0 | 0 |" % (
            gc, " / ".join("%.3f" % gold[cp][s][0] for s in SEEDS), gd, " / ".join("%.3f" % gold[cp][s][1] for s in SEEDS)))
        for label, tag, methods in RUNS:
            for m in methods:
                r = res.get((label, cp, m))
                if not r:
                    continue
                c = [r[s][0] for s in SEEDS]; d = [r[s][1] for s in SEEDS]
                dc = mean([abs(r[s][0] - gold[cp][s][0]) for s in SEEDS]); dd = mean([abs(r[s][1] - gold[cp][s][1]) for s in SEEDS])
                L.append("| %s / %s | %.3f [%s] | %.3f [%s] | %.3f | %.3f |" % (
                    label, m, mean(c), " / ".join("%.3f" % x for x in c), mean(d), " / ".join("%.3f" % x for x in d), dc, dd))
        L.append("")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
