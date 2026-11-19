"""Round 216 analysis: the parse score with the generator in it, against logZ, gen_lp_gate and the baselines, plus the tree comparison.
Reads r216_raw/run_s*_n*.json (r216_run.py) and the r215 raw files (baselines, r215 logZ); writes r216_parse_curve.tsv, r216_figure.tsv,
r216_tables.md, r216_gold.json (NLTK shallow-phrase gold, cached). Does NOT touch r215_parse_curve.tsv (rerunning r215_analyse.py would drop the 215b rows).
Scores (per word):
  logZ_pw        r215 parse score, logZ / len
  gen_lp_gate    r215b generator score, sum of log P over steps / (len + 1)
  a              logZ_pw + lam_a * gen_lp_gate
  b              logZb_pw + lam_b * gen_lp_gate, logZb = log inside weight with r216="split" (analysis-dependent)
  b_lit          the same with r216="chunk" (the literal chunk-sum form)
  b_parse, b_lit_parse   logZb_pw, logZbl_pw alone
lam = sd on Gcal of the parse term / sd on Gcal of gen_lp_gate (standardised sum; Gcal only, grammatical sentences only, no corrupted sentence is used).
Usage (repo root, compute node):  r216_analyse.py
"""
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402
import r215_analyse as A  # noqa: E402
import r215b_analyse as B  # noqa: E402
import r215_parse as P  # noqa: E402

RAW = os.path.join(C.OUT, "r216_raw")
SETS = {s: C.load_sets(s) for s in C.SEEDS}
NS = C.NS
HEAD = [("TRELLIS", "logZ_pw"), ("TRELLIS", "gen_lp_gate"), ("TRELLIS", "a"), ("TRELLIS", "b"), ("TRELLIS", "b_lit"),
        ("TRELLIS", "b_parse"), ("5-gram", "pw"), ("LSTM", "pw"), ("GPT", "pw")]
LAMX = (0.25, 0.5, 2.0, 4.0)           # sensitivity of a and b to lam (multiples of the Gcal-fitted value); not a fit


def sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def runs():
    out = {}
    for s in C.SEEDS:
        for n in NS:
            f = os.path.join(RAW, "run_s%d_n%d.json" % (s, n))
            if os.path.exists(f):
                out[(s, n)] = json.load(open(f, encoding="utf-8"))
    return out


def scores(d, seed):
    """name -> {sentence: value} for one run; also the fitted lams."""
    K = 2 * d["V"]
    res = {l: r for l, r in d["res"].items() if "gate" in r}
    gen = {l: B.step_scores(r, "gate", B.EPS[0], K) / (r["len"] + 1) for l, r in res.items()}
    zp = {l: r["logZ"] / r["len"] for l, r in res.items()}
    zb = {l: r["split"]["logZ"] / r["len"] for l, r in res.items()}
    zc = {l: r["chunk"]["logZ"] / r["len"] for l, r in res.items()}
    cal = SETS[seed]["Gcal"]
    out, lam = {"logZ_pw": zp, "gen_lp_gate": gen, "b_parse": zb, "b_lit_parse": zc}, {}
    for nm, z in (("a", zp), ("b", zb), ("b_lit", zc)):
        lam[nm] = sd([z[l] for l in cal]) / sd([gen[l] for l in cal])
        out[nm] = {l: z[l] + lam[nm] * gen[l] for l in res}
        for x in LAMX:
            out["%s_x%g" % (nm, x)] = {l: z[l] + x * lam[nm] * gen[l] for l in res}
    return out, lam


def main():
    R = runs()
    tab = A.collect()
    new = {}
    lams = {}
    for (seed, n), d in R.items():
        o, lam = scores(d, seed)
        lams[(seed, n)] = lam
        for k, v in o.items():
            new[("TRELLIS", k, seed, n)] = v
    M = {}
    for key, vals in {**{k: v for k, v in tab.items() if k[0] in ("5-gram", "LSTM", "GPT")}, **new}.items():
        M[key] = A.metrics(vals, SETS[key[2]])
    # ---- TSVs
    names = sorted({k[1] for k in new})
    with open(os.path.join(C.OUT, "r216_parse_curve.tsv"), "w", encoding="utf-8") as fh:
        fh.write("model\tscore\tseed\tn\tset\tauc\tauc_pair\tp_succ_gram\tp_succ_set\tgap\tthreshold\tp_succ_cal\n")
        for (model, sc, seed, n), m in sorted(M.items()):
            fh.write("\t".join(map(str, (model, sc, seed, n, "C1", "", "", m["p_C1"], "", "", m["thr"], m["p_cal"]))) + "\n")
            for k in A.REFS:
                r = m[k]
                fh.write("\t".join(map(str, (model, sc, seed, n, k, r["auc"], r["auc_pair"], r["p_ref"], r["p_set"], r["gap"], m["thr"], m["p_cal"]))) + "\n")

    def seeds_of(model, sc, n):
        return [s for s in C.SEEDS if (model, sc, s, n) in M]
    ns = [n for n in NS if any(k[3] == n for k in new)]
    FIG = [("TRELLIS", "logZ_pw", "logZ"), ("TRELLIS", "gen_lp_gate", "gen_lp_gate"), ("TRELLIS", "a", "a: logZ + lam gen"), ("TRELLIS", "b", "b: split-conditional parse + lam gen"),
           ("TRELLIS", "b_lit", "b_lit: chunk-sum parse + lam gen"), ("TRELLIS", "b_parse", "b parse alone"), ("5-gram", "pw", "5-gram"), ("LSTM", "pw", "LSTM"), ("GPT", "pw", "GPT")]
    with open(os.path.join(C.OUT, "r216_figure.tsv"), "w", encoding="utf-8") as fh:
        fh.write("n\tmodel\tscore\tset\tauc_mean\tauc_sd\tgap_mean\tgap_sd\tauc_pair_mean\tauc_pair_sd\tp_succ_set_mean\tp_succ_gram_mean\tseeds\n")
        for n in ns:
            for model, sc, label in FIG:
                for k in A.REFS:
                    ss = seeds_of(model, sc, n)
                    if not ss:
                        continue
                    f = lambda g: A.ms([g(M[(model, sc, s, n)]) for s in ss])
                    am, asd = f(lambda m: m[k]["auc"])
                    gm, gsd = f(lambda m: m[k]["gap"])
                    pm, psd = f(lambda m: m[k]["auc_pair"])
                    sm, _ = f(lambda m: m[k]["p_set"])
                    cm, _ = f(lambda m: m[k]["p_ref"])
                    fh.write("\t".join(map(str, (n, model if model != "TRELLIS" else "TRELLIS", label, k, "%.4f" % am, "%.4f" % asd, "%.4f" % gm, "%.4f" % gsd,
                                                   "%.4f" % pm, "%.4f" % psd, "%.4f" % sm, "%.4f" % cm, len(ss)))) + "\n")
    out = []

    def table(title, rows, f, sets=list(A.REFS)):
        out.append("\n### %s\n" % title)
        for k in sets:
            out.append("\n**%s** (%s)\n" % (k, C.NAMES[k]))
            out.append("| model / score | " + " | ".join("n=%d" % n for n in ns) + " |")
            out.append("|---|" + "---|" * len(ns))
            for model, sc in rows:
                cells = []
                for n in ns:
                    v = [f(M[(model, sc, s, n)], k) for s in seeds_of(model, sc, n)]
                    cells.append(A.fmt(v) + (" (%d)" % len(v) if v and len(v) < 4 else ""))
                out.append("| %s %s | " % (model, sc) + " | ".join(cells) + " |")
    table("Unpaired AUC (mean ± sd over seeds)", HEAD, lambda m, k: m[k]["auc"])
    table("Paired AUC", HEAD, lambda m, k: m[k]["auc_pair"])
    table("Operating-point gap P(success | gram) - P(success | corrupted), threshold P = .9 on Gcal", HEAD, lambda m, k: m[k]["gap"])
    # P(success), content swaps
    out.append("\n### P(success) at the operating point (threshold fixed on Gcal): grammatical test sentences and the two content-swap classes (grammatical, not meaningful)\n")
    out.append("A pure-syntax scorer accepts C2full / C2one as often as C1.\n")
    for lab, g in (("C1 (real test sentences)", lambda m, k: m["p_C1"]), ("C2full (all content words swapped)", lambda m, k: m["C2full"]["p_set"]),
                   ("C2one (one content word swapped)", lambda m, k: m["C2one"]["p_set"]),
                   ("C3scr (scramble, for scale)", lambda m, k: m["C3scr"]["p_set"])):
        out.append("\n**%s**\n" % lab)
        out.append("| model / score | " + " | ".join("n=%d" % n for n in ns) + " |")
        out.append("|---|" + "---|" * len(ns))
        for model, sc in HEAD:
            out.append("| %s %s | " % (model, sc) + " | ".join(A.fmt([g(M[(model, sc, s, n)], None) for s in seeds_of(model, sc, n)], sd=False) for n in ns) + " |")
    # differences to gen_lp_gate (per seed)
    out.append("\n### Does the parse add anything? Per-seed difference to gen_lp_gate (mean over 4 seeds; `+` marks all four seeds > 0, `-` all four < 0)\n")
    cells_pos = defaultdict(int)
    for metric, f in (("unpaired AUC", lambda m, k: m[k]["auc"]), ("paired AUC", lambda m, k: m[k]["auc_pair"]), ("gap", lambda m, k: m[k]["gap"])):
        for cand in ("a", "b", "b_lit"):
            out.append("\n**%s: %s minus gen_lp_gate**\n" % (metric, cand))
            out.append("| class | " + " | ".join("n=%d" % n for n in ns) + " |")
            out.append("|---|" + "---|" * len(ns))
            for k in A.REFS:
                cells = []
                for n in ns:
                    ss = [s for s in C.SEEDS if ("TRELLIS", cand, s, n) in M and ("TRELLIS", "gen_lp_gate", s, n) in M]
                    dv = [f(M[("TRELLIS", cand, s, n)], k) - f(M[("TRELLIS", "gen_lp_gate", s, n)], k) for s in ss]
                    mark = "+" if dv and all(x > 0 for x in dv) else "-" if dv and all(x < 0 for x in dv) else ""
                    cells.append("%+.3f%s" % (sum(dv) / len(dv), mark) if dv else "-")
                    cells_pos[(metric, cand, "+")] += mark == "+"
                    cells_pos[(metric, cand, "-")] += mark == "-"
                    cells_pos[(metric, cand, "N")] += 1
                out.append("| %s | " % k + " | ".join(cells) + " |")
    out.append("\nCells (n x class) where all four seeds agree on the sign of (candidate - gen_lp_gate):\n")
    out.append("| metric | candidate | all seeds > 0 | all seeds < 0 | cells |")
    out.append("|---|---|---|---|---|")
    for (metric, cand, t), v in sorted(cells_pos.items()):
        if t == "+":
            out.append("| %s | %s | %d | %d | %d |" % (metric, cand, v, cells_pos[(metric, cand, "-")], cells_pos[(metric, cand, "N")]))
    # lam sensitivity
    out.append("\n### Sensitivity to lam (multiples of the Gcal-fitted lam; x0 = gen_lp_gate alone is the limit lam -> infinity; NOT a fit, test sets are used): unpaired AUC, mean over classes and seeds\n")
    out.append("| score | " + " | ".join("n=%d" % n for n in ns) + " |")
    out.append("|---|" + "---|" * len(ns))
    for sc in ["logZ_pw"] + ["%s_x%g" % (c, x) for c in ("a", "b") for x in LAMX[:2]] + ["a", "b"] + ["%s_x%g" % (c, x) for c in ("a", "b") for x in LAMX[2:]] + ["gen_lp_gate"]:
        cells = []
        for n in ns:
            v = []
            for s in seeds_of("TRELLIS", sc, n):
                m = M[("TRELLIS", sc, s, n)]
                v.append(sum(m[k]["auc"] for k in A.REFS) / len(A.REFS))
            cells.append(A.fmt(v, sd=False))
        out.append("| %s | " % sc + " | ".join(cells) + " |")
    out.append("\nFitted lam (parse term sd / gen sd on Gcal), mean over seeds:\n")
    out.append("| n | a | b | b_lit |")
    out.append("|---|---|---|---|")
    for n in ns:
        out.append("| %d | " % n + " | ".join(A.fmt([lams[(s, n)][c] for s in C.SEEDS if (s, n) in lams], sd=False) for c in ("a", "b", "b_lit")) + " |")
    # ---- check: gen_lp_gate equals r215b's where it exists
    out.append("\n### Check: this round's gen_lp_gate equals r215b's (max abs sentence-score difference over the run's sentences)\n")
    for (seed, n), d in sorted(R.items()):
        f = os.path.join(C.RAW, "gen_s%d_n%d.json" % (seed, n))
        if os.path.exists(f):
            old = B.build(seed, n)[0]["gen_lp_gate"]
            new_ = new[("TRELLIS", "gen_lp_gate", seed, n)]
            out.append("- seed %d n=%d: %.2e" % (seed, n, max(abs(old[l] - new_[l]) for l in new_)))
    # ---- trees
    trees(R, out, M)
    open(os.path.join(C.OUT, "r216_tables.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


# ------------------------------------------------------------------------------------------------ trees
def gold_spans(lines):
    f = os.path.join(C.OUT, "r216_gold.json")
    cache = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    todo = [l for l in lines if l not in cache]
    if todo:
        import nltk
        CH = nltk.RegexpParser(r"""
          NP: {<DT|PRP\$|POS|CD|PDT>*<JJ.*|RBS|RBR>*<NN.*>+}
              {<PRP|EX>}
          AP: {<RB.*>*<JJ.*>+}
          VG: {<MD|TO>?<RB.*>*<VB.*>+(<RB.*>*<VB.*>+)*<RP>?}
          PP: {<IN|TO><NP>}
          VP: {<VG><NP|PP|AP|RB.*>+}
        """)
        for l in todo:
            w = l.split()
            tree = CH.parse(nltk.pos_tag(["I" if x == "i" else x for x in w]))
            spans = set()

            def walk(t, i):
                if isinstance(t, nltk.Tree):
                    j = i
                    for c in t:
                        j = walk(c, j)
                    if t.label() in ("NP", "PP", "VP", "AP", "VG") and j - i >= 2 and j - i < len(w):
                        spans.add((i, j))
                    return j
                return i + 1
            walk(tree, 0)
            cache[l] = sorted(spans)
        json.dump(cache, open(f, "w", encoding="utf-8"))
    return {l: {tuple(x) for x in cache[l]} for l in lines}


def inner(chosen, n):
    return {(i, j) for i, k, j in chosen if (i, j) != (0, n)}


def prf(pred, gold):
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    npred, ngold = sum(len(p) for p in pred), sum(len(g) for g in gold)
    P, R = tp / npred if npred else 0.0, tp / ngold if ngold else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0


def trees(R, out, M):
    out.append("\n## Do the trees change? (chosen binary tree = MBR; brackets = inner spans of 2+ words, whole sentence excluded)\n")
    MODES = (("default", lambda r: r["chosen"]), ("b split", lambda r: r["split"]["chosen"]), ("b_lit chunk", lambda r: r["chunk"]["chosen"]))
    allc1 = sorted({l for s in C.SEEDS for l in SETS[s]["C1"]})
    gold = gold_spans(allc1)
    rb = lambda n: {(i, n) for i in range(1, n - 1)}
    ns = sorted({n for _, n in R})
    out.append("F1 against the NLTK shallow-phrase gold (micro over each seed's 300 C1 test sentences; mean ± sd over seeds), right-branching on the same sentences, identical-tree share and F1 between the trees of the two parsers:\n")
    out.append("| n | default F1 | b split F1 | b_lit chunk F1 | right-branching F1 | trees identical default vs b | F1(default, b) | trees identical default vs b_lit |")
    out.append("|---|---|---|---|---|---|---|---|")
    for n in ns:
        cols = {m: [] for m, _ in MODES}
        rbs, ident, fb, ident_l = [], [], [], []
        for s in C.SEEDS:
            if (s, n) not in R:
                continue
            ls, res = SETS[s]["C1"], R[(s, n)]["res"]
            g = [gold[l] for l in ls]
            for m, f in MODES:
                cols[m].append(prf([inner(f(res[l]), len(l.split())) for l in ls], g))
            rbs.append(prf([rb(len(l.split())) for l in ls], g))
            d = [inner(res[l]["chosen"], len(l.split())) for l in ls]
            b = [inner(res[l]["split"]["chosen"], len(l.split())) for l in ls]
            c = [inner(res[l]["chunk"]["chosen"], len(l.split())) for l in ls]
            ident.append(sum(x == y for x, y in zip(d, b)) / len(ls))
            ident_l.append(sum(x == y for x, y in zip(d, c)) / len(ls))
            fb.append(prf(b, d))
        out.append("| %d | " % n + " | ".join(A.fmt(cols[m]) for m, _ in MODES) + " | %s | %s | %s | %s |" % (A.fmt(rbs), A.fmt(ident, sd=False), A.fmt(fb, sd=False), A.fmt(ident_l, sd=False)))
    out.append("\nCross-seed agreement: the 300 C1 sentences of seed 13 parsed by the learners of all four seeds; mean over the 6 seed pairs of F1 between their bracket sets (micro over sentences):\n")
    out.append("| n | default | b split | b_lit chunk |")
    out.append("|---|---|---|---|")
    com = SETS[13]["C1"]
    for n in ns:
        if not all((s, n) in R for s in C.SEEDS):
            continue
        row = []
        for m, f in MODES:
            br = {s: [inner(f(R[(s, n)]["res"][l]), len(l.split())) for l in com] for s in C.SEEDS}
            v = [prf(br[a], br[b]) for i, a in enumerate(C.SEEDS) for b in C.SEEDS[i + 1:]]
            row.append(A.fmt(v))
        out.append("| %d | " % n + " | ".join(row) + " |")
    out.append("\nBrackets per sentence (default / b split / b_lit chunk; chosen binary tree has len-2 inner brackets, so this only shows the sentence length mix) are not informative; the sets above are what changes.\n")
    # also: the distribution of the split-word's context start in chosen trees (how much left context the tree gives each word)
    out.append("Mean length of the left sibling (words of context the tree gives the right child's first word) in the chosen trees, seed mean:\n")
    out.append("| n | default | b split | b_lit chunk |")
    out.append("|---|---|---|---|")
    for n in ns:
        row = []
        for m, f in MODES:
            v = []
            for s in C.SEEDS:
                if (s, n) not in R:
                    continue
                res = R[(s, n)]["res"]
                xs = [k - i for l in SETS[s]["C1"] for i, k, j in f(res[l])]
                v.append(sum(xs) / len(xs))
            row.append(A.fmt(v, sd=False))
        out.append("| %d | " % n + " | ".join(row) + " |")
    # ---- examples
    for n in (250, 4000):
        if (13, n) not in R:
            continue
        res, S = R[(13, n)]["res"], SETS[13]
        sc, lam = scores(R[(13, n)], 13)
        thr = {k: C.threshold([sc[k][l] for l in S["Gcal"]]) for k in ("logZ_pw", "gen_lp_gate", "a", "b")}
        base4 = scores(R[(13, 4000)], 13)[0] if (13, 4000) in R else sc
        t4 = C.threshold([base4["logZ_pw"][l] for l in S["Gcal"]])
        ok = lambda l: base4["logZ_pw"][l] >= t4
        N_ = len(S["C1"])
        ia = next((i for i in range(N_) if ok(S["C1"][i]) and not ok(S["C3scr"][i]) and not ok(S["C3adj"][i])), 0)
        ib = next((i for i in range(N_) if ok(S["C1"][i]) and ok(S["C3scr"][i])), 1)
        picks = [("grammatical", S["C1"][ia]), ("adjacent swap", S["C3adj"][ia]), ("scramble", S["C3scr"][ia]), ("content swap (C2full)", S["C2full"][ia]),
                 ("grammatical", S["C1"][ib]), ("scramble (passes logZ at n=4000)", S["C3scr"][ib]), ("grammatical (C4 source)", S["C4v_src"][0]), ("agreement error", S["C4v"][0])]
        out.append("\n### Example parses, seed 13, n=%d (scores per word: logZ, gen_lp_gate, a, b; PASS = at or above the Gcal threshold)\n" % n)
        out.append("| class | sentence | logZ | gen | a | b | default parse | (b) parse |")
        out.append("|---|---|---|---|---|---|---|---|")
        for cls, l in picks:
            r, w = res[l], l.split()
            cell = lambda k: "%.2f %s" % (sc[k][l], "PASS" if sc[k][l] >= thr[k] else "fail")
            out.append("| %s | %s | %s | %s | %s | %s | `%s` | `%s` |" % (cls, l, cell("logZ_pw"), cell("gen_lp_gate"), cell("a"), cell("b"),
                       A.unbracket(P.bracket(r["chosen"], len(w)), w), A.unbracket(P.bracket(r["split"]["chosen"], len(w)), w)))


if __name__ == "__main__":
    main()
