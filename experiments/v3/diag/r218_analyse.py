"""Round 218 analysis: parse quality, tree statistics and cost of every cell of the windowed / depth-limited IO grid (r218_run.py).
Reads results/story_parse_eval/r218_raw/<cell>_s<seed>_n1000.json (+ r215 sets, r216_gold.json). Writes r218_tables.md, r218_spectrum.tsv,
r218_cases.md next to r218_raw.  Usage (repo root, compute node):  r218_analyse.py
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402
import r215_analyse as A  # noqa: E402
import r215_parse as P  # noqa: E402
from r218_cells import CELLS  # noqa: E402

RAW = os.path.join(C.OUT, "r218_raw")
N = 1000
SETS = {s: C.load_sets(s) for s in C.SEEDS}
GOLD = {l: {tuple(x) for x in v} for l, v in json.load(open(os.path.join(C.OUT, "r216_gold.json"), encoding="utf-8")).items()}


def load():
    R = {}
    for name in CELLS:
        for s in C.SEEDS:
            f = os.path.join(RAW, "%s_s%d_n%d.json" % (name, s, N))
            if os.path.exists(f):
                R[(name, s)] = json.load(open(f, encoding="utf-8"))
    return R


def inner(chosen, n):
    return {(i, j) for i, k, j in chosen if (i, j) != (0, n)}


def prf(pred, gold):
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    npred, ngold = sum(len(p) for p in pred), sum(len(g) for g in gold)
    Pp, Rr = tp / npred if npred else 0.0, tp / ngold if ngold else 0.0
    return 2 * Pp * Rr / (Pp + Rr) if Pp + Rr else 0.0


def rb(n):
    return {(i, n) for i in range(1, n - 1)}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def cell_stats(R, name, s):
    d = R[(name, s)]
    res = d["res"]
    S = SETS[s]
    out = {}
    for sc, fn in (("logZ_pw", lambda r: None if r["logZ"] is None else r["logZ"] / r["len"]), ("logw_pw", lambda r: r["logw"] / r["len"])):
        vals = {l: fn(r) for l, r in res.items()}
        if any(v is None for l, v in vals.items() if l in S["C1"]):
            continue
        out[sc] = A.metrics(vals, S)
    c1 = S["C1"]
    trees = [inner(res[l]["chosen"], len(l.split())) for l in c1]
    gold = [GOLD[l] for l in c1]
    out["f1"] = prf(trees, gold)
    out["f1_rb"] = prf([rb(len(l.split())) for l in c1], gold)
    out["nonrb"] = sum(inner(res[l]["chosen"], len(l.split())) != rb(len(l.split())) for l in c1) / len(c1)
    out["chunk_len"] = mean([j - i for l in c1 for i, j in inner(res[l]["chosen"], len(l.split()))])
    ch = sum(res[l]["chunks"] for l in c1)
    out["reuse_any"] = sum(res[l]["reuse_any"] for l in c1) / max(ch, 1)
    out["reuse_w50"] = sum(res[l]["reuse_w50"] for l in c1) / max(ch, 1)
    out["factors"] = mean([res[l]["factors"] for l in c1])
    out["triples"] = mean([res[l]["triples"] for l in c1])
    out["seconds"] = mean([res[l]["seconds"] for l in c1])
    out["windows"] = mean([res[l]["windows"] for l in c1])
    out["passes"] = mean([res[l]["passes"] for l in c1])
    out["learn_s"] = d["learn_s"]
    out["records"] = d["records"]
    if (("full", s) in R):
        fr = R[("full", s)]["res"]
        out["ident_full"] = sum(inner(res[l]["chosen"], len(l.split())) == inner(fr[l]["chosen"], len(l.split())) for l in c1) / len(c1)
        out["f1_full"] = prf(trees, [inner(fr[l]["chosen"], len(l.split())) for l in c1])
    return out


def xseed(R, name):
    seeds = [s for s in C.SEEDS if (name, s) in R]
    com = SETS[13]["C1"]
    br = {s: [inner(R[(name, s)]["res"][l]["chosen"], len(l.split())) for l in com] for s in seeds}
    v = [prf(br[a], br[b]) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    return mean(v), len(seeds)


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return "%.3f±%.3f" % (m, sd)


def main():
    R = load()
    names = [c for c in CELLS if any((c, s) in R for s in C.SEEDS)]
    ST = {(c, s): cell_stats(R, c, s) for c in names for s in C.SEEDS if (c, s) in R}
    out = ["# r218 tables (n=1000, seeds %s)\n" % ",".join(str(s) for s in C.SEEDS)]

    def seeds_of(c):
        return [s for s in C.SEEDS if (c, s) in ST]

    UN = ("C3scr", "C3adj", "C4v")

    def table(title, score):
        out.append("\n### %s\n" % title)
        out.append("| cell | seeds | AUC scramble | AUC adjacent swap | AUC agreement | mean ungram AUC | AUC C2full | AUC C2one | P(accept C1) | P(accept C2full) | P(accept C2one) | P(accept C3adj) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in names:
            ss = seeds_of(c)
            if any(score not in ST[(c, s)] for s in ss):
                continue
            g = lambda f: ms([f(ST[(c, s)][score]) for s in ss])
            ung = ms([mean([ST[(c, s)][score][k]["auc"] for k in UN]) for s in ss])
            out.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                c, len(ss), g(lambda m: m["C3scr"]["auc"]), g(lambda m: m["C3adj"]["auc"]), g(lambda m: m["C4v"]["auc"]), ung,
                g(lambda m: m["C2full"]["auc"]), g(lambda m: m["C2one"]["auc"]), g(lambda m: m["p_C1"]),
                g(lambda m: m["C2full"]["p_set"]), g(lambda m: m["C2one"]["p_set"]), g(lambda m: m["C3adj"]["p_set"])))

    table("Acceptance by score logZ per word (sum of window logZ; = r215's score for `full`)", "logZ_pw")
    table("Acceptance by score logw per word (log weight of the chosen tree; comparable across cells)", "logw_pw")
    # paired differences against full
    out.append("\n### Paired difference of mean ungrammatical AUC (scr, adj, agr), logw_pw, cell minus full, per seed\n")
    out.append("| cell | " + " | ".join("s%d" % s for s in C.SEEDS) + " | mean | seeds positive |")
    out.append("|---|" + "---|" * (len(C.SEEDS) + 2))
    for c in names:
        if c == "full":
            continue
        d = []
        for s in C.SEEDS:
            if (c, s) in ST and ("full", s) in ST and "logw_pw" in ST[(c, s)]:
                d.append(mean([ST[(c, s)]["logw_pw"][k]["auc"] - ST[("full", s)]["logw_pw"][k]["auc"] for k in UN]))
            else:
                d.append(None)
        v = [x for x in d if x is not None]
        out.append("| %s | %s | %s | %d/%d |" % (c, " | ".join("%+.3f" % x if x is not None else "-" for x in d),
                                                 "%+.3f" % mean(v) if v else "-", sum(x > 0 for x in v), len(v)))
    # trees
    out.append("\n### Trees on the 300 C1 sentences per seed (F1 vs NLTK shallow-phrase gold, right-branching F1 on the same sentences, share of non-right-branching trees, mean inner chunk length, chunk reuse = share of inner chunks whose word string is a stored composite (any weight / weight >= .5), tree identical to full IO's, F1 vs full IO's tree, cross-seed agreement = mean pairwise F1 of the 300 seed-13 C1 trees over the seeds)\n")
    out.append("| cell | F1 gold | RB F1 | non-RB trees | inner chunk len | reuse any | reuse w>=.5 | identical to full | F1 vs full | cross-seed agr. (seeds) | records |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    xs = {}
    for c in names:
        ss = seeds_of(c)
        g = lambda k: ms([ST[(c, s)].get(k) for s in ss])
        xs[c] = xseed(R, c)
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %.3f (%d) | %s |" % (c, g("f1"), g("f1_rb"), g("nonrb"), g("chunk_len"), g("reuse_any"),
                   g("reuse_w50"), g("ident_full"), g("f1_full"), xs[c][0] or 0, xs[c][1], "%d" % mean([ST[(c, s)]["records"] for s in ss])))
    # cost
    out.append("\n### Cost per C1 sentence (mean over C1 sentences, 3-8 tokens, then over seeds): composite evaluations (_factor calls: 2 Cobweb queries each), IO triples, windows, passes, parse seconds; learner training seconds (learn cells)\n")
    out.append("| cell | factors | factors / full | IO triples | windows | passes | parse s | learn s |")
    out.append("|---|---|---|---|---|---|---|---|")
    ff = mean([ST[("full", s)]["factors"] for s in seeds_of("full")]) if "full" in names else None
    for c in names:
        ss = seeds_of(c)
        f = mean([ST[(c, s)]["factors"] for s in ss])
        out.append("| %s | %.1f | %.2f | %.1f | %.1f | %.1f | %.4f | %s |" % (c, f, f / ff if ff else float("nan"), mean([ST[(c, s)]["triples"] for s in ss]),
                   mean([ST[(c, s)]["windows"] for s in ss]), mean([ST[(c, s)]["passes"] for s in ss]), mean([ST[(c, s)]["seconds"] for s in ss]),
                   "%.0f" % mean([ST[(c, s)]["learn_s"] for s in ss]) if ST[(c, ss[0])]["learn_s"] else "-"))
    # by length for full and a few
    out.append("\n### Composite evaluations by sentence length (seed mean, all lines)\n")
    lens = list(range(3, 9))
    out.append("| cell | " + " | ".join("len %d" % k for k in lens) + " |")
    out.append("|---|" + "---|" * len(lens))
    for c in names:
        row = []
        for k in lens:
            v = []
            for s in seeds_of(c):
                v += [r["factors"] for l, r in R[(c, s)]["res"].items() if r["len"] == k]
            row.append("%.1f" % mean(v) if v else "-")
        out.append("| %s | %s |" % (c, " | ".join(row)))
    open(os.path.join(C.OUT, "r218_tables.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    # spectrum tsv
    with open(os.path.join(C.OUT, "r218_spectrum.tsv"), "w", encoding="utf-8") as fh:
        fh.write("cell\tlearn\tseeds\tfactors_per_sent\tfactors_rel_full\tparse_s_per_sent\tungram_auc_logw\tadj_auc_logw\tungram_auc_logZ\tp_accept_C2full_logw\t"
                 "f1_gold\tnonrb\tchunk_len\tf1_vs_full\tcross_seed_f1\n")
        for c in names:
            ss = seeds_of(c)
            f = mean([ST[(c, s)]["factors"] for s in ss])
            gm = lambda sc, fn: mean([fn(ST[(c, s)][sc]) for s in ss if sc in ST[(c, s)]])
            un = lambda m: mean([m[k]["auc"] for k in UN])
            fh.write("\t".join(str(x) for x in (c, int(CELLS[c]["kw"].get("win_learn", False)), len(ss), "%.3f" % f, "%.3f" % (f / ff if ff else 0),
                     "%.5f" % mean([ST[(c, s)]["seconds"] for s in ss]), "%.4f" % gm("logw_pw", un), "%.4f" % gm("logw_pw", lambda m: m["C3adj"]["auc"]),
                     "%.4f" % (gm("logZ_pw", un) or float("nan")), "%.4f" % gm("logw_pw", lambda m: m["C2full"]["p_set"]),
                     "%.4f" % mean([ST[(c, s)]["f1"] for s in ss]), "%.4f" % mean([ST[(c, s)]["nonrb"] for s in ss]),
                     "%.3f" % mean([ST[(c, s)]["chunk_len"] for s in ss]), "%.4f" % mean([ST[(c, s)].get("f1_full") for s in ss]),
                     "%.4f" % (xs[c][0] or 0))) + "\n")
    # case studies: seed 13 C1 sentences, longest, where the compared cells give the most distinct trees
    show = [c for c in ("greedy", "W2", "W4", "W4s2", "D1", "D3", "D1c", "D2c", "W4D2", "full", "W4L", "D3L", "D2cL") if (c, 13) in R]
    cases = []
    if ("full", 13) in R:
        cand = []
        for l in SETS[13]["C1"]:
            w = l.split()
            if len(w) < 7:
                continue
            tr = {c: inner(R[(c, 13)]["res"][l]["chosen"], len(w)) for c in show}
            cand.append((len({frozenset(t) for t in tr.values()}), len(w), l))
        cand.sort(key=lambda t: (-t[0], -t[1], t[2]))
        picked = [l for _, _, l in cand[:4]]
        cases.append("# r218 case studies (seed 13, n=1000, C1 test sentences with the most distinct trees across the shown cells)\n")
        for l in picked:
            w = l.split()
            cases.append("\n**%s**\n" % l)
            cases.append("| cell | tree | composites evaluated |")
            cases.append("|---|---|---|")
            for c in show:
                r = R[(c, 13)]["res"][l]
                cases.append("| %s | `%s` | %d |" % (c, A.unbracket(P.bracket(r["chosen"], len(w)), w), r["factors"]))
        open(os.path.join(C.OUT, "r218_cases.md"), "w", encoding="utf-8").write("\n".join(cases) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
