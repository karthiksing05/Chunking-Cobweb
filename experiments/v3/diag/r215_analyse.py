"""Round 215 analysis: AUC and operating points from results/story_parse_eval/r215_raw (r215_parse.py, r215_lms.py).

Per (seed, n, model, score): grammatical reference C1 (C4's own sources for C4), five corrupted sets, calibration set Gcal.
  threshold: the 10th percentile of the score on Gcal (400 held-out sentences not in C1/C4_src), so P(success | grammatical) = .9 there;
             success = score >= threshold.
  auc        P(score of a reference sentence > score of a corrupted one), reference and corrupted sets of 300 sentences of the same lengths
  auc_pair   the same over aligned pairs (a sentence and its own corruption)
  gap        P(success | C1 test) - P(success | corrupted set) at the threshold
Writes r215_parse_curve.tsv and r215_tables.md (tables, examples) next to the raw dir's parent.
Usage (repo root, compute node):  r215_analyse.py
"""
import json
import math
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402

TR_SCORES = {"logZ_pw": lambda r: r["logZ"] / r["len"], "logZ_raw": lambda r: r["logZ"], "logZ_pj": lambda r: r["logZ"] / (r["len"] - 1),
             "vit_pw": lambda r: r["vit"] / r["len"], "vit_raw": lambda r: r["vit"], "chosen_mean": lambda r: r["chosen_mean"],
             "chosen_min": lambda r: r["chosen_min"], "chosen_sum_pw": lambda r: r["chosen_sum"] / r["len"]}
REFS = {"C3scr": "C1", "C3adj": "C3adj_src", "C2full": "C1", "C2one": "C1", "C4v": "C4v_src"}
HEAD_TR, HEAD_LM = "logZ_pw", "pw"


def collect():
    """rows: (model, score, seed, n) -> {sentence: value}"""
    tab = {}
    for seed in C.SEEDS:
        for n in C.NS:
            f = os.path.join(C.RAW, "trellis_s%d_n%d.json" % (seed, n))
            if os.path.exists(f):
                d = json.load(open(f, encoding="utf-8"))
                for sc, fn in TR_SCORES.items():
                    tab[("TRELLIS", sc, seed, n)] = {l: fn(r) for l, r in d["res"].items()}
            f = os.path.join(C.RAW, "lms_s%d_n%d.json" % (seed, n))
            if os.path.exists(f):
                d = json.load(open(f, encoding="utf-8"))
                for m, v in d["scores"].items():
                    if m in ("random", "neglen"):
                        tab[(m, "raw", seed, n)] = v
                        continue
                    name = {"ng2": "2-gram", "ng3": "3-gram", "ng5": "5-gram", "lstm": "LSTM", "gpt": "GPT"}[m]
                    tab[(name, "pw", seed, n)] = {l: x / (len(l.split()) + 1) for l, x in v.items()}
                    tab[(name, "raw", seed, n)] = v
    return tab


def metrics(vals, S):
    cal = [vals[l] for l in S["Gcal"]]
    thr = C.threshold(cal)
    out = {"thr": thr, "p_cal": C.succ(cal, thr), "p_C1": C.succ([vals[l] for l in S["C1"]], thr),
           "p_C4_src": C.succ([vals[l] for l in S["C4_src"]], thr)}
    for k, ref in REFS.items():
        x, r = [vals[l] for l in S[k]], [vals[l] for l in S[ref]]
        out[k] = {"auc": C.auc(r, x), "auc_pair": C.paired_auc(r, x), "p_ref": C.succ(r, thr), "p_set": C.succ(x, thr)}
        out[k]["gap"] = out[k]["p_ref"] - out[k]["p_set"]
    return out


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, sd


def fmt(v, sd=True):
    m, s = ms(v)
    if m is None:
        return "-"
    return ("%.3f" % m) + (("±%.3f" % s) if sd and len(v) > 1 else "")


def unbracket(tree, words):
    return "".join(words[int(t)] if t.isdigit() else t for t in __import__("re").findall(r"\d+|[() ]", tree)).replace("  ", " ")


def main():
    tab = collect()
    SETS = {s: C.load_sets(s) for s in C.SEEDS}
    M = {}
    with open(os.path.join(C.OUT, "r215_parse_curve.tsv"), "w", encoding="utf-8") as fh:
        fh.write("model\tscore\tseed\tn\tset\tauc\tauc_pair\tp_succ_gram\tp_succ_set\tgap\tthreshold\tp_succ_cal\n")
        for (model, sc, seed, n), vals in sorted(tab.items()):
            m = metrics(vals, SETS[seed])
            M[(model, sc, seed, n)] = m
            fh.write("\t".join(map(str, (model, sc, seed, n, "C1", "", "", m["p_C1"], "", "", m["thr"], m["p_cal"]))) + "\n")
            for k in REFS:
                r = m[k]
                fh.write("\t".join(map(str, (model, sc, seed, n, k, r["auc"], r["auc_pair"], r["p_ref"], r["p_set"], r["gap"], m["thr"], m["p_cal"]))) + "\n")
    out = []
    ns = [n for n in C.NS if any(k[3] == n for k in M)]
    models = [("TRELLIS", HEAD_TR), ("5-gram", HEAD_LM), ("3-gram", HEAD_LM), ("2-gram", HEAD_LM), ("LSTM", HEAD_LM), ("GPT", HEAD_LM), ("random", "raw"), ("neglen", "raw")]

    def seeds_of(model, sc, n):
        return [s for s in C.SEEDS if (model, sc, s, n) in M]

    def table(title, key, models, sets, f=lambda m, k: m[k]["auc"]):
        out.append("\n### %s\n" % title)
        for k in sets:
            out.append("\n**%s** (%s)\n" % (k, C.NAMES[k]))
            out.append("| model | " + " | ".join("n=%d" % n for n in ns) + " |")
            out.append("|---|" + "---|" * len(ns))
            for model, sc in models:
                cells = []
                for n in ns:
                    v = [f(M[(model, sc, s, n)], k) for s in seeds_of(model, sc, n)]
                    cells.append(fmt(v) + (" (%d)" % len(v) if v and len(v) < 4 else ""))
                out.append("| %s | " % model + " | ".join(cells) + " |")
    table("AUC by n, mean ± sd over seeds 13/17/19/23 (per-word score; TRELLIS = logZ per word; (k) = k seeds present)", "auc", models, list(REFS))
    table("Paired AUC (a sentence against its own corruption)", "auc_pair", models, list(REFS), f=lambda m, k: m[k]["auc_pair"])
    table("Operating point: P(success | gram) - P(success | corrupted) at the threshold giving P(success | gram) = .9 on Gcal", "gap", models, list(REFS), f=lambda m, k: m[k]["gap"])
    out.append("\n### P(success) at the operating point (threshold fixed on Gcal)\n")
    for k in ["C1"] + list(REFS):
        out.append("\n**%s**\n" % ("C1 (grammatical test)" if k == "C1" else "%s (%s)" % (k, C.NAMES[k])))
        out.append("| model | " + " | ".join("n=%d" % n for n in ns) + " |")
        out.append("|---|" + "---|" * len(ns))
        for model, sc in models[:6]:
            cells = []
            for n in ns:
                v = [(M[(model, sc, s, n)]["p_C1"] if k == "C1" else M[(model, sc, s, n)][k]["p_set"]) for s in seeds_of(model, sc, n)]
                cells.append(fmt(v))
            out.append("| %s | " % model + " | ".join(cells) + " |")
    out.append("\n### Raw (not length-normalised) scores, AUC mean over seeds\n")
    rawm = [("TRELLIS", "logZ_raw"), ("TRELLIS", "vit_raw"), ("5-gram", "raw"), ("LSTM", "raw"), ("GPT", "raw")]
    table("raw", "auc", rawm, list(REFS))
    out.append("\n### TRELLIS score variants, AUC mean over seeds\n")
    for n in ns:
        out.append("\n**n=%d**\n" % n)
        out.append("| score | " + " | ".join(REFS) + " |")
        out.append("|---|" + "---|" * len(REFS))
        for sc in TR_SCORES:
            out.append("| %s | " % sc + " | ".join(fmt([M[("TRELLIS", sc, s, n)][k]["auc"] for s in seeds_of("TRELLIS", sc, n)], sd=False) for k in REFS) + " |")
    # stored share and thresholds
    out.append("\n### Share of C1 test sentences that are stored training sentences (seed 13..23 mean)\n")
    st = []
    for n in ns:
        v = []
        for s in C.SEEDS:
            f = os.path.join(C.RAW, "trellis_s%d_n%d.json" % (s, n))
            if os.path.exists(f):
                d = json.load(open(f, encoding="utf-8"))["res"]
                v.append(sum(d[l]["stored"] for l in SETS[s]["C1"]) / len(SETS[s]["C1"]))
        st.append("n=%d: %s" % (n, fmt(v)))
    out.append("; ".join(st))
    # examples
    for n in (250, 4000):
        f = os.path.join(C.RAW, "trellis_s13_n%d.json" % n)
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))["res"]
        S = SETS[13]
        thr = M[("TRELLIS", HEAD_TR, 13, n)]["thr"]
        out.append("\n### Example parses, seed 13, n=%d (score = logZ per word; threshold %.3f; PASS = success)\n" % (n, thr))
        out.append("| class | sentence | score | parse |")
        out.append("|---|---|---|---|")
        # the same base items at both n, chosen at n=4000: ia = grammatical passes and its scramble and adjacent swap fail;
        # ib = a scramble that passes; ic = a grammatical sentence that fails
        d4 = json.load(open(os.path.join(C.RAW, "trellis_s13_n4000.json"), encoding="utf-8"))["res"]
        t4 = M[("TRELLIS", HEAD_TR, 13, 4000)]["thr"]
        ok = lambda l: d4[l]["logZ"] / d4[l]["len"] >= t4
        N_ = len(S["C1"])
        ia = next((i for i in range(N_) if ok(S["C1"][i]) and not ok(S["C3scr"][i]) and not ok(S["C3adj"][i])), 0)
        ib = next((i for i in range(N_) if ok(S["C1"][i]) and ok(S["C3scr"][i])), 1)
        ic = next((i for i in range(N_) if not ok(S["C1"][i])), 2)
        picks = [("grammatical", S["C1"][ia]), ("adjacent swap", S["C3adj"][ia]), ("scramble", S["C3scr"][ia]), ("content swap (C2full)", S["C2full"][ia]),
                 ("grammatical", S["C1"][ib]), ("scramble (passes at n=4000)", S["C3scr"][ib]), ("grammatical (fails at n=4000)", S["C1"][ic]),
                 ("grammatical (C4 source)", S["C4v_src"][0]), ("agreement error", S["C4v"][0])]
        for cls, l in picks:
            r = d[l]
            sc = r["logZ"] / r["len"]
            out.append("| %s | %s | %.3f %s | `%s` |" % (cls, l, sc, "PASS" if sc >= thr else "fail", unbracket(r["tree"], l.split())))
    open(os.path.join(C.OUT, "r215_tables.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
