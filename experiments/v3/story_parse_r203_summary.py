"""Round 203 summary: one compact table per checkpoint for the r196/r198/r203 parse arms against BPE and the baselines.
Columns: brackets/sent, F1, F1+sv, crossing, xseed F1, abs(consistency - gold), abs(decisive - gold) (gold = the NLTK
shallow phrases, as r198), each mean over seeds 13/17/19. Reads results/story_parse_eval/<tag>_s<seed>.json and
<tag>_table.json (written by `story_parse_r203.py agg <tag>`).
Usage (compute node, repo root): story_parse_r203_summary.py <out.md>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import story_parse_gold_ceiling as G  # noqa: E402

D = G.D
SEEDS, CPS = G.SEEDS, G.CPS
# (label, tag, method)
ROWS = [
    ("GOLD (shallow phrases)", None, None),
    ("default r185 / post", "r185", "trellis_post"),
    ("default r185 / chosen", "r185", "trellis_chosen"),
    ("parts r196 / post", "r196parts", "trellis_post"),
    ("parts r196 / chosen", "r196parts", "trellis_chosen"),
    ("parts+cohesion r196 / post", "r196coh", "trellis_post"),
    ("parts+edge_once r196 / post", "r196pe", "trellis_post"),
    ("r203 default (rerun) / post", "r203def", "trellis_post"),
    ("r203 default (rerun) / post_bud", "r203def", "post_bud"),
    ("r203 default (rerun) / gm_bud", "r203def", "gm_bud"),
    ("r203 default (rerun) / gm_p50", "r203def", "gm_p50"),
    ("r203 parts (rerun) / post", "r203base", "trellis_post"),
    ("r203 parts (rerun) / chosen", "r203base", "trellis_chosen"),
    ("r203 parts (rerun) / post_bud", "r203base", "post_bud"),
    ("r203 parts (rerun) / gm_bud", "r203base", "gm_bud"),
    ("r203 parts (rerun) / gm_p50", "r203base", "gm_p50"),
    ("r203a edge_w=1 / post", "r203a", "trellis_post"),
    ("r203a edge_w=1 / chosen", "r203a", "trellis_chosen"),
    ("r203a edge_w=1 / post_bud", "r203a", "post_bud"),
    ("r203a edge_w=1 / gm_bud", "r203a", "gm_bud"),
    ("r203a edge_w=1 / gm_p50", "r203a", "gm_p50"),
    ("r203b goodman tree / post", "r203b", "trellis_post"),
    ("r203b goodman tree / chosen", "r203b", "trellis_chosen"),
    ("r203b goodman tree / post_bud", "r203b", "post_bud"),
    ("r203b goodman tree / gm_bud", "r203b", "gm_bud"),
    ("r203b goodman tree / gm_p50", "r203b", "gm_p50"),
    ("r203c edge_w=1+goodman / post", "r203c", "trellis_post"),
    ("r203c edge_w=1+goodman / chosen", "r203c", "trellis_chosen"),
    ("r203c edge_w=1+goodman / post_bud", "r203c", "post_bud"),
    ("r203c edge_w=1+goodman / gm_bud", "r203c", "gm_bud"),
    ("r203c edge_w=1+goodman / gm_p50", "r203c", "gm_p50"),
    ("BPE x1 / bpe_x1", "r203base", "bpe_x1"),
    ("PMI (post budget) / pmi_post", "r203base", "pmi_post"),
    ("right-branching", "r185", "right_branching"),
    ("random binary", "r185", "random_binary"),
]


def mean(v):
    return sum(v) / len(v)


def main(out):
    runs, tabs, gold = {}, {}, {}
    tags = sorted({t for _, t, _ in ROWS if t})
    for t in tags:
        for s in SEEDS:
            p = "%s%s_s%d.json" % (D, t, s)
            if os.path.exists(p):
                runs[(t, s)] = json.load(open(p, encoding="utf-8"))
        p = "%s%s_table.json" % (D, t)
        if os.path.exists(p):
            tabs[t] = json.load(open(p, encoding="utf-8"))
    L = ["Round 203 summary: means over seeds 13/17/19 [per-seed in the per-tag tables]. |d| = absolute difference from the gold "
         "shallow-phrase span set's consistency (%.3f) / decisive (%.3f)." % (0.116, 0.982),
         "post_bud / gm_bud = posterior threshold / Goodman lam bisected so brackets per held-out sentence equal BPE x1's (a matched budget).",
         "gm_p50 = n-ary Goodman, lam .5. Missing rows = arm not run.", ""]
    for cp in CPS:
        L.append("### %d sentences learned\n" % cp)
        L.append("| method | brackets/sent | F1 | F1+sv | crossing | xseed F1 | abs(cons-gold) | abs(dec-gold) | consistency | decisive |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for label, tag, m in ROWS:
            if tag is None:
                L.append("| %s | - | 1 | 1 | 0 | 1 | 0 | 0 | 0.116 | 0.982 |" % label)
                continue
            vals = {k: [] for k in ("b", "F", "Fsv", "cr", "xs", "dc", "dd", "c", "d")}
            for s in SEEDS:
                J = runs.get((tag, s))
                if J is None:
                    break
                ck = [c for c in J["checkpoints"] if c["learned"] == cp]
                if not ck or m not in ck[0]["methods"]:
                    break
                mt = ck[0]["methods"][m]["metrics"]
                held = [tuple(h) for h in J["held"]]
                st = G.strings_of(held)
                gc, gd = G.stat([set(map(tuple, x)) for x in J["gold_phrase"]], st)
                vals["b"].append(mt["brackets/sent"]); vals["F"].append(mt["F1"]); vals["Fsv"].append(mt["F1+sv"])
                vals["cr"].append(mt["crossing"]); vals["c"].append(mt["consistency"]); vals["d"].append(mt["decisive"])
                vals["dc"].append(abs(mt["consistency"] - gc)); vals["dd"].append(abs(mt["decisive"] - gd))
                try:
                    vals["xs"].append(tabs[tag]["%d|%s" % (cp, m)]["per_seed"][str(s)]["xseed F1"])
                except Exception:
                    pass
            else:
                f = lambda k, nd=3: ("%.3f" % mean(vals[k])) if vals[k] and all(x is not None for x in vals[k]) else "n/a"
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    label, f("b"), f("F"), f("Fsv"), f("cr"), f("xs"), f("dc"), f("dd"), f("c"), f("d")))
        L.append("")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
