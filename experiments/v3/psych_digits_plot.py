"""Learning curve figure for the digit-span replication (P3f; docs/PLAN_PSYCH.md).

Span per session = mean list length over the 5-session block ending at that session (RSS95 Fig. 4 style), mean of 3 seeds,
with the seed range as a faint band.  DD's points are only those in docs/CE1981_PROTOCOL.md: start 7-9, about 80 near
session 500, 104 at about 865.  Usage: python experiments/v3/psych_digits_plot.py [tag=P3f]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join("experiments", "v3", "results", "psych_digits")
ARMS = [("full", "TRELLIS (EPAM procedure)", "#2a78d6", "-"),
        ("linkoff", "links off", "#eb6834", "--"),
        ("conceptsoff", "concepts off", "#1baf7a", "-."),
        ("prioroff", "prior knowledge off", "#eda100", ":")]
G_ARMS = [("full", "TRELLIS, EPAM storage (1 chunk/list)", "#2a78d6", "-"),
          ("store8", "8 s/chunk storage", "#eb6834", "--"),
          ("randlabel", "random labels", "#1baf7a", "-."),
          ("prioroff", "prior knowledge off", "#eda100", ":")]
HELD = (23, 29, 31)                               # P3g verdicts are on held-out seeds only
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def block_curve(d):
    by = {}
    for r in d["rows"]:
        by.setdefault(r["s"], []).append(r["len"])
    xs, ys = [], []
    for s in range(5, d["sessions"] + 1):
        v = [x for t in range(s - 4, s + 1) for x in by.get(t, [])]
        if v:
            xs.append(s)
            ys.append(sum(v) / len(v))
    return xs, ys


def main(tag):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ends = []
    p3g = tag.startswith("P3g")
    for arm, name, col, ls in (G_ARMS if p3g else ARMS):
        fns = sorted(glob.glob(os.path.join(OUT, "%s_%s_s*.json" % (tag, arm))))
        if p3g:
            fns = [fn for fn in fns if int(fn.rsplit("_s", 1)[1][:-5]) in HELD]
        runs = [json.load(open(fn, encoding="utf-8")) for fn in fns]
        if not runs:
            continue
        curves = [block_curve(d) for d in runs]
        n = min(len(c[0]) for c in curves)
        xs = curves[0][0][:n]
        mean = [sum(c[1][i] for c in curves) / len(curves) for i in range(n)]
        lo = [min(c[1][i] for c in curves) for i in range(n)]
        hi = [max(c[1][i] for c in curves) for i in range(n)]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.plot(xs, mean, color=col, linestyle=ls, linewidth=2, label=name)
        ends.append([mean[-1], xs[-1], name])
    ends.sort()
    for i in range(1, len(ends)):                 # keep end labels at least 7 span units apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 7)
    for y, x, name in ends:
        ax.annotate(name, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    dd = [(1, 8), (500, 80), (865, 104)]
    ax.scatter([p[0] for p in dd], [p[1] for p in dd], marker="D", s=46, color=INK, zorder=5, label="DD (RSS95)")
    ax.annotate("DD", (865, 104), xytext=(-22, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlim(0, 1080)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Practice session", color=INK2)
    ax.set_ylabel("Digit span (mean list length, 5-session block)", color=INK2)
    ax.set_title("Digit span with practice: TRELLIS in EPAM IV's procedure vs. DD" + (" (held-out seeds)" if p3g else ""),
                 color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT, "%s_curve.png" % tag)
    fig.savefig(path, facecolor=fig.get_facecolor())
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "P3f")
