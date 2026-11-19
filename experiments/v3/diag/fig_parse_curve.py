"""Parsing evaluation curve figures for the TinyStories section (inline SVG, no libraries).

P1: P(parse success) against training sentences n (log scale, 125 to 4000), one panel per TRELLIS score
    (logZ = parse only, gen_lp_gate, (b) split), one line per sentence class, band = sd over seeds.
    Source: r216_parse_curve.tsv (per-seed rows; success = score >= the 10th percentile of the
    grammatical calibration half).
P2: unpaired AUC (real sentences against each corrupted class) against n, one panel per class,
    TRELLIS gen_lp_gate and (b) against the 5-gram, LSTM and GPT. Source: r216_figure.tsv
    (mean and sd over seeds 13/17/19/23).

Usage:
    python3 experiments/v3/diag/fig_parse_curve.py [--outdir docs/site]
writes frag_parse_p1.html and frag_parse_p2.html (the SVG only; captions live in skeleton.html).
"""
import argparse
import csv
import html
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(ROOT, "experiments/v3/results/story_parse_eval")
CURVE = os.path.join(RES, "r216_parse_curve.tsv")
FIG = os.path.join(RES, "r216_figure.tsv")
NS = [125, 250, 500, 1000, 2000, 4000]

CLASSES = [  # key, label, colour, marker
    ("C1", "real sentences (C1)", "var(--ink)", "circle"),
    ("C3scr", "scramble", "var(--s2)", "square"),
    ("C3adj", "adjacent swap", "var(--s1)", "diamond"),
    ("C4v", "agreement error", "var(--cand)", "circle"),
    ("C2full", "content swap (C2full)", "var(--s3)", "square"),
]
SCORES = [  # (model, score in curve tsv, panel title)
    ("TRELLIS", "logZ_pw", "Parse only (logZ per word)"),
    ("TRELLIS", "gen_lp_gate", "Generator (gen_lp_gate)"),
    ("TRELLIS", "b", "Parse with the generator inside, (b) split"),
]
FIG_SCORE_B = "b: split-conditional parse + lam gen"
LINES2 = [  # label, model, score in figure tsv, colour, marker
    ("5-gram", "5-gram", "5-gram", "var(--cand)", "diamond"),
    ("LSTM", "LSTM", "LSTM", "var(--s1)", "circle"),
    ("GPT", "GPT", "GPT", "var(--s2)", "square"),
    ("TRELLIS gen_lp_gate", "TRELLIS", "gen_lp_gate", "var(--s3)", "circle"),
    ("TRELLIS (b) split", "TRELLIS", FIG_SCORE_B, "var(--foc)", "square"),
]
PANEL2 = [("C3scr", "Full scramble"), ("C3adj", "Adjacent swap"), ("C4v", "Agreement error"), ("C2full", "Content swap (C2full)")]


def mean_sd(v):
    m = sum(v) / len(v)
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0)


def marker(kind, x, y, colour, title, r=3.6):
    if kind == "square":
        s = f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}"'
        e = "</rect>"
    elif kind == "diamond":
        s = f'<polygon points="{x:.1f},{y - r - 1:.1f} {x + r + 1:.1f},{y:.1f} {x:.1f},{y + r + 1:.1f} {x - r - 1:.1f},{y:.1f}"'
        e = "</polygon>"
    else:
        s = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"'
        e = "</circle>"
    return f'{s} fill="{colour}" stroke="var(--surface)" stroke-width="1.2"><title>{html.escape(title)}</title>{e}'


class Panel:
    def __init__(self, x0, y0, w, h, ymin, ymax, yticks, ylab, title):
        self.x0, self.y0, self.w, self.h, self.ymin, self.ymax = x0, y0, w, h, ymin, ymax
        self.yticks, self.ylab, self.title = yticks, ylab, title

    def fx(self, n):
        return self.x0 + self.w * (math.log2(n) - math.log2(125)) / (math.log2(4000) - math.log2(125))

    def fy(self, v):
        return self.y0 + self.h - self.h * (v - self.ymin) / (self.ymax - self.ymin)

    def frame(self, a, xlab=True):
        for t in self.yticks:
            a(f'<line x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.fy(t):.1f}" y2="{self.fy(t):.1f}" stroke="var(--gridc)" stroke-width="1"/>')
            a(f'<text x="{self.x0 - 6}" y="{self.fy(t) + 4:.1f}" text-anchor="end" style="font-size:11px;fill:var(--muted)">{t:g}</text>')
        yb = self.y0 + self.h
        a(f'<line x1="{self.x0}" x2="{self.x0 + self.w}" y1="{yb}" y2="{yb}" stroke="var(--muted)" stroke-width="1"/>')
        for n in NS:
            a(f'<line x1="{self.fx(n):.1f}" x2="{self.fx(n):.1f}" y1="{yb}" y2="{yb + 4}" stroke="var(--muted)" stroke-width="1"/>')
            a(f'<text x="{self.fx(n):.1f}" y="{yb + 16}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">{n}</text>')
        if xlab:
            a(f'<text x="{self.x0 + self.w / 2:.0f}" y="{yb + 32}" text-anchor="middle" style="font-size:12px;fill:var(--ink)">training sentences n</text>')
        a(f'<text x="{self.x0}" y="{self.y0 - 10}" style="font-size:12.5px;font-weight:600;fill:var(--ink)">{html.escape(self.title)}</text>')
        a(f'<text transform="translate({self.x0 - 36} {self.y0 + self.h / 2:.0f}) rotate(-90)" text-anchor="middle" style="font-size:11.5px;fill:var(--ink2)">{html.escape(self.ylab)}</text>')

    def series(self, a, pts, colour, kind, label, fmt, dash=None):
        """pts: [(n, mean, sd)]"""
        up = [(self.fx(n), self.fy(min(self.ymax, m + s))) for n, m, s in pts]
        dn = [(self.fx(n), self.fy(max(self.ymin, m - s))) for n, m, s in pts]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in up + dn[::-1])
        a(f'<polygon points="{poly}" fill="{colour}" fill-opacity="0.14" stroke="none"/>')
        d = f' stroke-dasharray="{dash}"' if dash else ""
        a(f'<polyline points="{" ".join(f"{self.fx(n):.1f},{self.fy(m):.1f}" for n, m, s in pts)}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"{d}/>')
        for n, m, s in pts:
            a(marker(kind, self.fx(n), self.fy(m), colour, f"{label}, n={n}: {fmt} {m:.3f} (sd {s:.3f})"))


def read_curve():
    """(score, set, n) -> [value per seed]; C1 uses p_succ_gram."""
    d = defaultdict(list)
    with open(CURVE, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["model"] != "TRELLIS" or r["score"] not in {s for _, s, _ in SCORES}:
                continue
            n = int(r["n"])
            if r["set"] == "C1":
                d[(r["score"], "C1", n)].append(float(r["p_succ_gram"]))
            elif r["set"] in {c[0] for c in CLASSES}:
                d[(r["score"], r["set"], n)].append(float(r["p_succ_set"]))
    return d


def build_p1():
    d = read_curve()
    W, H = 800, 770
    out = []
    a = out.append
    a(f'<svg class="dg" viewBox="0 0 {W} {H}" role="img" aria-label="Parse success rate against training sentences, three panels. '
      f'Parse only: real sentences stay near .93; content swaps are accepted nearly as often (.905 at 4,000 sentences), adjacent swaps at .87 and scrambles at .77. '
      f'Generator score and the (b) split score: success on scrambles falls from .39 and .49 at 125 sentences to .05 at 4,000, while content swaps stay at .58 and .63.">')
    pos = [(70, 40), (450, 40), (70, 400)]
    for (model, sc, title), (x0, y0) in zip(SCORES, pos):
        p = Panel(x0, y0, 310, 240, 0, 1, [0, 0.2, 0.4, 0.6, 0.8, 1.0], "P(parse success)", title)
        p.frame(a)
        for key, label, col, mk in CLASSES:
            pts = []
            for n in NS:
                v = d.get((sc, key, n))
                if v:
                    m, s = mean_sd(v)
                    pts.append((n, m, s))
            p.series(a, pts, col, mk, f"{title}: {label}", "P(success)", dash="5 3" if key == "C1" else None)
    # legend in the fourth cell
    lx, ly = 450, 440
    a(f'<text x="{lx}" y="{ly}" style="font-size:12.5px;font-weight:600;fill:var(--ink)">Sentence class</text>')
    for i, (key, label, col, mk) in enumerate(CLASSES):
        y = ly + 24 + 22 * i
        dash = ' stroke-dasharray="5 3"' if key == "C1" else ""
        a(f'<line x1="{lx}" x2="{lx + 24}" y1="{y - 4}" y2="{y - 4}" stroke="{col}" stroke-width="2"{dash}/>')
        a(marker(mk, lx + 12, y - 4, col, label))
        a(f'<text x="{lx + 32}" y="{y}" style="font-size:12px;fill:var(--ink2)">{html.escape(label)}</text>')
    notes = [
        "Success = score at or above the 10th percentile of 400 held-out",
        "real sentences (one threshold per seed and n).",
        "Lines: mean of four seeds; band: plus or minus one sd.",
        "A syntax-only parser should accept content swaps as often",
        "as real sentences (green on the dashed line).",
    ]
    for i, t in enumerate(notes):
        a(f'<text x="{lx}" y="{ly + 150 + 15 * i}" style="font-size:11px;fill:var(--muted)">{html.escape(t)}</text>')
    a("</svg>")
    return "\n".join(out)


def read_fig():
    d = {}
    with open(FIG, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["auc_mean"] == "":
                continue
            d[(r["model"], r["score"], r["set"], int(r["n"]))] = (float(r["auc_mean"]), float(r["auc_sd"]))
    return d


def build_p2():
    d = read_fig()
    W, H = 800, 770
    out = []
    a = out.append
    a(f'<svg class="dg" viewBox="0 0 {W} {H}" role="img" aria-label="Unpaired AUC of real sentences against each corrupted class, along training sentences, four panels. '
      f'On scrambles TRELLIS is within the baselines\' range at 125 to 250 sentences and trails by .01 at 4,000. On adjacent swaps it is level through 250 sentences and falls behind from 500. On agreement it stays within .02 of the baselines through 500 sentences and falls below after; on content swaps it is below at every n.">')
    pos = [(70, 40), (450, 40), (70, 400), (450, 400)]
    for (key, title), (x0, y0) in zip(PANEL2, pos):
        p = Panel(x0, y0, 310, 240, 0.5, 1.0, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "AUC", title)
        p.frame(a, xlab=True)
        a(f'<line x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.fy(0.5):.1f}" y2="{p.fy(0.5):.1f}" stroke="var(--ink2)" stroke-width="1" stroke-dasharray="4 3"/>')
        for label, model, sc, col, mk in LINES2:
            pts = [(n,) + d[(model, sc, key, n)] for n in NS if (model, sc, key, n) in d]
            p.series(a, pts, col, mk, f"{title}: {label}", "AUC")
    lx, ly = 70, 700
    for label, model, sc, col, mk in LINES2:
        if lx + 30 + 7 * len(label) > W - 8:
            lx, ly = 70, ly + 20
        a(f'<line x1="{lx}" x2="{lx + 18}" y1="{ly - 4}" y2="{ly - 4}" stroke="{col}" stroke-width="2"/>')
        a(marker(mk, lx + 9, ly - 4, col, label))
        a(f'<text x="{lx + 24}" y="{ly}" style="font-size:12px;fill:var(--ink2)">{html.escape(label)}</text>')
        lx += 34 + 7 * len(label)
    a(f'<text x="70" y="{ly + 24}" style="font-size:11px;fill:var(--muted)">Mean of four seeds, band plus or minus one sd; dashed line = chance (.5). Baselines score log-probability per word.</text>')
    a("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "docs/site"))
    args = ap.parse_args()
    for name, svg in (("frag_parse_p1.html", build_p1()), ("frag_parse_p2.html", build_p2())):
        with open(os.path.join(args.outdir, name), "w", encoding="utf-8") as f:
            f.write(svg + "\n")
        print("wrote", name, len(svg))


if __name__ == "__main__":
    main()
