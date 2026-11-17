"""Round 190: the grammaticality-vs-novelty frontier from a story_frontier.py jsonl (all n=1000 rows).

x = tri (share of new sentences with an unstored trigram), y = gram (new judged grammatical) and, second panel,
gram | tri. One line per generator family along its temperature curve (labels = temperature), error bars = seed
range, dashed line = the judge's real-sentence ceiling (mean over seeds). Also writes the means / per-seed tables
and the upper-envelope comparison at tri = .1 .. .7.

Usage: story_frontier_plot.py <frontier.jsonl> <out prefix (results dir/name)> <png> [<png> ...]
  (writes <prefix>.means.tsv, <prefix>.perseed.tsv, <prefix>.envelope.txt; matplotlib; run on a compute node)
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
COLS = [("gram", "novel judged grammatical"), ("tri", "unstored trigram"), ("gram|tri", "gram | unstored trigram"),
        ("combined", "gram new tri")]


def family(r):
    arm = json.loads(r["arm"])
    if "lm" in arm:
        return arm["lm"].upper() if arm["lm"] == "gpt" else "LSTM", arm.get("temp", 1.0)
    if "ngram" in arm:
        return "ngram %d" % arm["ngram"], arm.get("temp", 1.0)
    if "swap" in arm:
        return "retrieved swap", 1.0
    m = re.search(r"-w(\d+)$", r["dir"])
    return "TRELLIS w%s" % m.group(1), arm.get("temp", 1.0)


def mean(v):
    return sum(v) / len(v)


def interp(pts, x):
    """Upper envelope of the piecewise-linear curve through pts [(x, y)] at x; None outside the measured x range."""
    pts = sorted(pts)
    if not pts or x < pts[0][0] - 1e-12 or x > pts[-1][0] + 1e-12:
        return None
    best = None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 - 1e-12 <= x <= x1 + 1e-12:
            y = y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            best = y if best is None else max(best, y)
    return best if best is not None else (pts[0][1] if len(pts) == 1 and abs(pts[0][0] - x) < 1e-12 else None)


def main(jl, prefix, pngs):
    rows = [json.loads(l) for l in open(jl, encoding="utf-8")]
    rows = [r for r in rows if r["n"] == 1000]
    pt = defaultdict(lambda: defaultdict(list))        # (family, temp) -> seed -> row
    for r in rows:
        f, t = family(r)
        pt[(f, t)][r["seed"]] = r
    stats = {}                                         # (family, temp) -> {col: (mean, lo, hi, n)}
    for k, bs in pt.items():
        d = {}
        for c, key in COLS:
            v = [b[key] for b in bs.values() if b[key] is not None]
            d[c] = (mean(v), min(v), max(v), len(v)) if v else None
        d["seeds"] = len(bs)
        stats[k] = d
    cal = {}
    for r in rows:
        if r["calibration real"] is not None:
            cal[r["seed"]] = r["calibration real"]
    ceiling = mean(list(cal.values()))
    fams = sorted({k[0] for k in stats}, key=lambda f: (not f.startswith("TRELLIS"), f))

    def f6(x):
        return "-" if x is None else "%.3f" % x
    with open(prefix + ".means.tsv", "w", encoding="utf-8") as fh:
        fh.write("family\ttemp\tseeds\t" + "\t".join("%s mean [lo,hi]" % c for c, _ in COLS) + "\n")
        for k in sorted(stats, key=lambda k: (fams.index(k[0]), k[1])):
            d = stats[k]
            fh.write("%s\t%.1f\t%d\t" % (k[0], k[1], d["seeds"]) + "\t".join(
                "-" if d[c] is None else "%.3f [%.3f,%.3f]" % d[c][:3] for c, _ in COLS) + "\n")
    with open(prefix + ".perseed.tsv", "w", encoding="utf-8") as fh:
        fh.write("family\ttemp\tseed\t" + "\t".join(c for c, _ in COLS) + "\n")
        for k in sorted(pt, key=lambda k: (fams.index(k[0]), k[1])):
            for s, b in sorted(pt[k].items()):
                fh.write("%s\t%.1f\t%d\t" % (k[0], k[1], s) + "\t".join(f6(b[key]) for _, key in COLS) + "\n")

    # upper envelope per family at LEVELS (mean curve; seed range from each seed's own curve)
    curves = {}
    for f in fams:
        ks = [k for k in stats if k[0] == f]
        cm = [(stats[k]["tri"][0], stats[k]["gram"][0]) for k in ks]
        seeds = sorted({s for k in ks for s in pt[k]})
        cs = {s: [(pt[k][s]["unstored trigram"], pt[k][s]["novel judged grammatical"]) for k in ks if s in pt[k]]
              for s in seeds}
        curves[f] = (cm, cs)
    out = []
    out.append("judge real-sentence ceiling (mean over seeds): %.3f  (per seed %s)" % (
        ceiling, ", ".join("%d:%.3f" % kv for kv in sorted(cal.items()))))
    out.append("upper envelope of gram along each family's temperature curve (linear interpolation in tri, no extrapolation)")
    out.append("level | " + " | ".join(fams))
    best = {}
    for x in LEVELS:
        vals = {}
        for f in fams:
            cm, cs = curves[f]
            m = interp(cm, x)
            if m is None:
                continue
            sv = [v for v in (interp(c, x) for c in cs.values()) if v is not None]
            vals[f] = (m, min(sv) if sv else None, max(sv) if sv else None, len(sv))
        best[x] = vals
        out.append("tri=%.1f | " % x + " | ".join(
            "%s" % ("-" if f not in vals else "%.3f [%s,%s]%s" % (vals[f][0], f6(vals[f][1]), f6(vals[f][2]),
                                                                   "" if vals[f][3] == len(curves[f][1]) else " (%d seeds)" % vals[f][3]))
            for f in fams))
    out.append("")
    for x in LEVELS:
        v = sorted(best[x].items(), key=lambda kv: -kv[1][0])
        if len(v) >= 2:
            out.append("tri=%.1f: highest %s %.3f [%s,%s]; next %s %.3f [%s,%s]; margin %+.3f" % (
                x, v[0][0], v[0][1][0], f6(v[0][1][1]), f6(v[0][1][2]), v[1][0], v[1][1][0], f6(v[1][1][1]),
                f6(v[1][1][2]), v[0][1][0] - v[1][1][0]))
        else:
            out.append("tri=%.1f: %s" % (x, v))
    txt = "\n".join(out)
    open(prefix + ".envelope.txt", "w", encoding="utf-8").write(txt + "\n")
    print(txt)

    # plot
    style = {"TRELLIS w3": ("#1b4f8a", "o"), "TRELLIS w4": ("#2f8fd1", "s"), "TRELLIS w5": ("#7fc4ee", "^"),
             "TRELLIS w6": ("#0a2a4d", "D"), "TRELLIS w2": ("#9aa5b1", "o"), "GPT": ("#d95f02", "o"),
             "LSTM": ("#1b9e77", "s"), "retrieved swap": ("#c2185b", "*")}
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True)
    for ax, (c, ylab) in zip(axs, [("gram", "new judged grammatical (gram)"),
                                  ("gram|tri", "grammatical among new sentences with an unstored trigram (gram | tri)")]):
        for f in fams:
            if f == "TRELLIS w2":
                continue
            ks = sorted((k for k in stats if k[0] == f and stats[k][c] is not None), key=lambda k: stats[k]["tri"][0])
            if f.startswith("ngram"):
                col = "#555555"
                if c != "gram":
                    continue
                for k in ks:
                    ax.errorbar(0, stats[k]["gram"][0], yerr=[[stats[k]["gram"][0] - stats[k]["gram"][1]],
                                [stats[k]["gram"][2] - stats[k]["gram"][0]]], fmt="x", color=col, ms=6, lw=1)
                    ax.annotate(f.split()[1] + "g t%.1f" % k[1], (0, stats[k]["gram"][0]), fontsize=6.5, xytext=(4, 2),
                                textcoords="offset points", color=col)
                continue
            col, mk = style.get(f, ("#000", "o"))
            xs = [stats[k]["tri"][0] for k in ks]
            ys = [stats[k][c][0] for k in ks]
            xe = [[stats[k]["tri"][0] - stats[k]["tri"][1] for k in ks], [stats[k]["tri"][2] - stats[k]["tri"][0] for k in ks]]
            ye = [[stats[k][c][0] - stats[k][c][1] for k in ks], [stats[k][c][2] - stats[k][c][0] for k in ks]]
            if f == "retrieved swap":
                ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt=mk, color=col, ms=12, lw=1, label=f, zorder=5)
                continue
            ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt=mk + "-", color=col, ms=5, lw=1.4, elinewidth=0.8, capsize=1.5,
                        label=f, alpha=0.95)
            for k, x_, y_ in zip(ks, xs, ys):
                ax.annotate("%g" % k[1], (x_, y_), fontsize=6.5, xytext=(3, 4), textcoords="offset points", color=col)
        ax.axhline(ceiling, ls="--", color="k", lw=1)
        ax.text(0.99, ceiling + 0.004, "judge ceiling on real sentences %.3f" % ceiling, ha="right", va="bottom",
                fontsize=7.5, transform=ax.get_yaxis_transform())
        ax.set_xlabel("novelty: share of new sentences containing an unstored trigram (tri)")
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlim(-0.03, 1.0)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_title("grammaticality vs novelty (n=1000, 4 seeds; labels = temperature, bars = seed range)", fontsize=9)
    axs[1].set_title("grammaticality of the sentences that are novel at the trigram level", fontsize=9)
    axs[0].legend(fontsize=7.5, loc="lower left", frameon=False)
    fig.tight_layout()
    for p in pngs:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        fig.savefig(p, dpi=170)
        print("saved", p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
