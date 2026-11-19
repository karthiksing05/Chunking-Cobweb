"""Round 215b analysis: gen_lp / gen_lp_gate (and rank / top-1 / parse_norm variants) against the r215 rows, same sets, thresholds and metrics.
Reads r215_raw/gen_s*_n*.json (r215b_gen.py) and the r215 raw files; appends TRELLIS rows to r215_parse_curve.tsv (only once: existing 215b rows
are dropped first) and writes r215b_tables.md (tables for r215b_genscore.md).
Score per word: sum over steps of log((p + eps) / (1 + eps K)) / (len + 1)  (len + 1 as the LMs' words + </s>); K = 2 |vocab| outcomes (word, ends here).
Usage (repo root, compute node):  r215b_analyse.py
"""
import json
import math
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402
import r215_analyse as A  # noqa: E402

EPS = (1e-6, 1e-4)          # primary, sensitivity
NEW = {"gen_lp": "nogate", "gen_lp_gate": "gate"}


def step_scores(rec, variant, eps, K):
    steps = rec[variant]
    lp = 0.0
    for p, _, _, fb in steps:
        p = 1.0 / K if p is None else p
        lp += math.log((p + eps) / (1 + eps * K))
    return lp


def build(seed, n):
    """name -> {sentence: value} for this (seed, n)."""
    d = json.load(open(os.path.join(C.RAW, "gen_s%d_n%d.json" % (seed, n)), encoding="utf-8"))
    K, out = 2 * d["V"], {}
    pf = os.path.join(C.RAW, "pn_s%d_n%d.json" % (seed, n))
    if os.path.exists(pf):
        pn = json.load(open(pf, encoding="utf-8"))["perm_lse"]
        old = json.load(open(os.path.join(C.RAW, "trellis_s%d_n%d.json" % (seed, n)), encoding="utf-8"))["res"]
        for l, r in d["res"].items():
            r["perm_lse"] = pn[l]
            r["logZ"] = old[l]["logZ"]
    for r in d["res"].values():
        r["_L"] = r["len"] + 1
    for nm, var in NEW.items():
        for eps in EPS:
            tag = nm if eps == EPS[0] else "%s_e%g" % (nm, eps)
            out[tag] = {l: step_scores(r, var, eps, K) / r["_L"] for l, r in d["res"].items()}
        out[nm + "_raw"] = {l: step_scores(r, var, EPS[0], K) for l, r in d["res"].items()}
        # rank / top-1 of the true outcome (no eps involved): mean of -log rank per step, share of steps where the true outcome is the argmax
        out[nm + "_rank"] = {l: -sum(math.log(x[1]) if x[1] else math.log(2 * d["V"]) for x in r[var]) / len(r[var]) for l, r in d["res"].items()}
        out[nm + "_top1"] = {l: sum(1 for x in r[var] if x[1] == 1) / len(r[var]) for l, r in d["res"].items()}
        out[nm + "_unsup"] = {l: -sum(1 for x in r[var] if (x[0] or 0) == 0) / len(r[var]) for l, r in d["res"].items()}   # -(share of steps with no mass on the true outcome)
    if all("perm_lse" in r for r in d["res"].values()):
        out["parse_norm"] = {l: (r["logZ"] - r["perm_lse"]) / r["len"] for l, r in d["res"].items()}
    return out, d


def main():
    tab = A.collect()
    SETS = {s: C.load_sets(s) for s in C.SEEDS}
    new, diag = {}, {}
    for seed in C.SEEDS:
        for n in C.NS:
            if os.path.exists(os.path.join(C.RAW, "gen_s%d_n%d.json" % (seed, n))):
                o, d = build(seed, n)
                diag[(seed, n)] = d
                for sc, vals in o.items():
                    new[("TRELLIS", sc, seed, n)] = vals
    M = {}
    for key, vals in {**tab, **new}.items():
        if key[2] in SETS and (key in new or key[0] in ("TRELLIS", "5-gram", "3-gram", "LSTM", "GPT", "2-gram")):
            M[key] = A.metrics(vals, SETS[key[2]])
    # append rows
    f = os.path.join(C.OUT, "r215_parse_curve.tsv")
    names = {k[1] for k in new}
    keep = [l for l in open(f, encoding="utf-8").read().splitlines() if l.split("\t")[0] != "TRELLIS" or l.split("\t")[1] not in names]
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("\n".join(keep) + "\n")
        for (model, sc, seed, n), vals in sorted(new.items()):
            m = M[(model, sc, seed, n)]
            fh.write("\t".join(map(str, (model, sc, seed, n, "C1", "", "", m["p_C1"], "", "", m["thr"], m["p_cal"]))) + "\n")
            for k in A.REFS:
                r = m[k]
                fh.write("\t".join(map(str, (model, sc, seed, n, k, r["auc"], r["auc_pair"], r["p_ref"], r["p_set"], r["gap"], m["thr"], m["p_cal"]))) + "\n")
    ns = sorted({k[3] for k in new})
    out = []
    rows = [("TRELLIS", "logZ_pw"), ("TRELLIS", "parse_norm"), ("TRELLIS", "gen_lp_gate"), ("TRELLIS", "gen_lp"), ("TRELLIS", "gen_lp_gate_e0.0001"),
            ("TRELLIS", "gen_lp_e0.0001"), ("5-gram", "pw"), ("LSTM", "pw"), ("GPT", "pw")]
    secondary = [("TRELLIS", "gen_lp_gate_raw"), ("TRELLIS", "gen_lp_gate_rank"), ("TRELLIS", "gen_lp_gate_top1"), ("TRELLIS", "gen_lp_gate_unsup"),
                 ("TRELLIS", "gen_lp_rank"), ("TRELLIS", "gen_lp_top1"), ("TRELLIS", "gen_lp_unsup")]

    def seeds_of(model, sc, n):
        return [s for s in C.SEEDS if (model, sc, s, n) in M]

    def table(title, models, f):
        out.append("\n### %s\n" % title)
        for k in A.REFS:
            out.append("\n**%s** (%s)\n" % (k, C.NAMES[k]))
            out.append("| model / score | " + " | ".join("n=%d" % n for n in ns) + " |")
            out.append("|---|" + "---|" * len(ns))
            for model, sc in models:
                cells = []
                for n in ns:
                    v = [f(M[(model, sc, s, n)], k) for s in seeds_of(model, sc, n)]
                    cells.append(A.fmt(v) + (" (%d)" % len(v) if v and len(v) < 4 else ""))
                out.append("| %s %s | " % (model, sc) + " | ".join(cells) + " |")
    table("Unpaired AUC (mean ± sd over seeds)", rows, lambda m, k: m[k]["auc"])
    table("Paired AUC", rows, lambda m, k: m[k]["auc_pair"])
    table("Operating-point gap, P(success | gram) - P(success | corrupted), threshold P = .9 on Gcal", rows, lambda m, k: m[k]["gap"])
    table("Secondary TRELLIS generator scores, unpaired AUC (raw = not per word; rank = mean -log rank of the true outcome; top1 = share of steps where the true outcome is the argmax; unsup = -share of steps with zero mass on the true outcome)",
          secondary, lambda m, k: m[k]["auc"])
    table("Secondary TRELLIS generator scores, paired AUC", secondary, lambda m, k: m[k]["auc_pair"])
    # diagnostics: floor hits, support, p of true, mean max p
    out.append("\n### Generator diagnostics per set (gated variant; mean over seeds)\n")
    out.append("| n | set | share of steps with zero mass on the true outcome | mean P(true outcome) | share of steps where true = argmax | steps with Z = 0 |")
    out.append("|---|---|---|---|---|---|")
    for n in ns:
        for sname in ["C1", "C3scr", "C3adj", "C2full", "C2one", "C4v"]:
            z, p, t1, zz = [], [], [], []
            for s in C.SEEDS:
                if (s, n) not in diag:
                    continue
                recs = [diag[(s, n)]["res"][l]["gate"] for l in SETS[s][sname]]
                st = [x for r in recs for x in r]
                z.append(sum(1 for x in st if (x[0] or 0) == 0) / len(st))
                p.append(sum((x[0] or 0) for x in st) / len(st))
                t1.append(sum(1 for x in st if x[1] == 1) / len(st))
                zz.append(sum(x[3] for x in st) / len(st))
            out.append("| %d | %s | %s | %s | %s | %s |" % (n, sname, A.fmt(z, sd=False), A.fmt(p, sd=False), A.fmt(t1, sd=False), A.fmt(zz, sd=False)))
    out.append("\n### P(success | grammatical) at the Gcal threshold, test C1 (mean over seeds)\n")
    out.append("| score | " + " | ".join("n=%d" % n for n in ns) + " |")
    out.append("|---|" + "---|" * len(ns))
    for model, sc in rows:
        out.append("| %s %s | " % (model, sc) + " | ".join(A.fmt([M[(model, sc, s, n)]["p_C1"] for s in seeds_of(model, sc, n)], sd=False) for n in ns) + " |")
    open(os.path.join(C.OUT, "r215b_tables.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
