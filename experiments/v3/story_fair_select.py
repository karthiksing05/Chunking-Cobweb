"""Round 200: best-setting selection and paired comparison on top of story_frontier_fair.py's perseed.tsv.

Reads <prefix>.perseed.tsv (fam, temp, seed, A, ..., gramnovel). Setting = TRELLIS plain (window, temp) | LSTM temp | GPT temp.
Value per seed = gramnovel / attempted. A setting counts only if it has every seed of the run.
Reports (a) per-family max over settings (mean over all seeds, per-seed range), (b) selection on SEL seeds (13, 17), the
chosen setting reported on HELD seeds (19, 23), (c) per-seed paired TRELLIS-best vs LSTM-best and GPT-best (same seeds)
with the number of seeds TRELLIS wins, for (a) and for (b). At n=4000 only seeds 13, 17 exist: both are reported.
Usage: story_fair_select.py <prefix>   (writes <prefix>.select.txt)
"""
import re
import sys
from collections import defaultdict

SEL, HELD = (13, 17), (19, 23)


def mean(v):
    return sum(v) / len(v)


def main(prefix):
    val = defaultdict(dict)   # (family, setting) -> {seed: rate}
    for ln in open(prefix + ".perseed.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = ln.split("\t")
        fam, temp, seed, A, gn = c[0], float(c[1]), int(c[2]), int(c[3]), int(c[8])
        if re.fullmatch(r"TRELLIS w\d+", fam):
            key = ("TRELLIS", "%s t%.2f" % (fam[len("TRELLIS "):], temp))
        elif fam in ("LSTM", "GPT"):
            key = (fam, "t%.2f" % temp)
        else:
            continue
        val[key][seed] = gn / A
    seeds = sorted({s for d in val.values() for s in d})
    full = {k: d for k, d in val.items() if all(s in d for s in seeds)}
    out = ["seeds present: %s; settings with all seeds: %d of %d" % (seeds, len(full), len(val))]
    fams = ["TRELLIS", "LSTM", "GPT"]
    best_all, best_sel = {}, {}
    out.append("\n(a) per-family max over settings, all seeds (distinct gram-novel per attempted draw)")
    for f in fams:
        ks = [k for k in full if k[0] == f]
        if not ks:
            out.append("  %-8s none" % f)
            continue
        b = max(ks, key=lambda k: mean(list(full[k].values())))
        best_all[f] = b
        v = [full[b][s] for s in seeds]
        out.append("  %-8s %.3f  [%.3f, %.3f]  at %s  (seeds %s)" % (f, mean(v), min(v), max(v), b[1], seeds))
    out.append("\n    per-setting means (all seeds)")
    for f in fams:
        for k in sorted((k for k in full if k[0] == f), key=lambda k: k[1]):
            out.append("    %-8s %-10s %.3f" % (f, k[1], mean([full[k][s] for s in seeds])))
    sel = [s for s in SEL if s in seeds]
    held = [s for s in HELD if s in seeds]
    out.append("\n(b) setting chosen on seeds %s, reported on seeds %s%s" % (sel, held or "-", "" if held else " (n=4000: both seeds reported separately)"))
    for f in fams:
        ks = [k for k in full if k[0] == f]
        if not ks or not sel:
            continue
        b = max(ks, key=lambda k: mean([full[k][s] for s in sel]))
        best_sel[f] = b
        out.append("  %-8s chose %-10s  sel %.3f  held %s  per-seed %s" % (
            f, b[1], mean([full[b][s] for s in sel]),
            ("%.3f" % mean([full[b][s] for s in held])) if held else "-",
            ", ".join("%d:%.3f" % (s, full[b][s]) for s in seeds)))
    for title, best, use in (("(c1) paired, best-over-all-seeds settings", best_all, seeds),
                             ("(c2) paired, selected on %s, compared on %s" % (sel, held or seeds), best_sel, held or seeds)):
        out.append("\n" + title)
        if "TRELLIS" not in best:
            continue
        for o in ("LSTM", "GPT"):
            if o not in best:
                continue
            d = [(s, full[best["TRELLIS"]][s], full[best[o]][s]) for s in use]
            w = sum(t > x for _, t, x in d)
            out.append("  TRELLIS %s vs %s %s: wins %d of %d seeds; mean diff %+.3f; %s" % (
                best["TRELLIS"][1], o, best[o][1], w, len(d), mean([t - x for _, t, x in d]),
                ", ".join("%d:%.3f/%.3f" % z for z in d)))
    open(prefix + ".select.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
