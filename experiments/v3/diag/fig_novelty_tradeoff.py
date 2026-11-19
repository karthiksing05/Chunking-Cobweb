"""Novelty-vs-grammaticality trade-off figure for the TinyStories section (inline SVG, no libraries).

x = novelty: share of distinct new sentences with an unstored trigram (tri|d).
y = grammaticality of the distinct new sentences under a strict judge.
One line per generator family; the points along a line are that family's dial
(sampling temperature for the neural models and TRELLIS, n-gram order for the samplers).

Default data: experiments/v3/results/story_gen/r211/r211_curves.tsv (n = 1000, four seeds,
Qwen3-8B strict judge, real-sentence ceiling .958). Settings whose mean distinct new sentences
per seed fall under 20 are left out, the same rule round 211 used for its interpolation.

Extra series (for example round 214's abstraction-knob points) come from a TSV with the columns
    family  dial  value  novelty  gram
and are drawn as extra lines, points ordered by value. Example row:
    TRELLIS r214 knob<TAB>abstraction<TAB>2<TAB>0.55<TAB>0.80

Usage:
    python3 experiments/v3/diag/fig_novelty_tradeoff.py [--extra extra.tsv] [--out frag.html]
"""
import argparse
import csv
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CURVES = os.path.join(ROOT, "experiments/v3/results/story_gen/r211/r211_curves.tsv")
JUDGE = "Qwen3-8B/strict"
CEILING = 0.958
MIN_DISTINCT = 20  # mean distinct new sentences per seed

# (family key in the TSV, legend label, colour token, marker, dial name)
BASE = [
    ("TRELLIS w3", "TRELLIS, window 3", "var(--s3)", "circle", "temperature"),
    ("TRELLIS w4", "TRELLIS, window 4", "var(--s3)", "square", "temperature"),
    ("TRELLIS w5", "TRELLIS, window 5", "var(--s3)", "diamond", "temperature"),
    ("TRELLIS edge w7", "TRELLIS edge, window 7", "var(--foc)", "circle", "temperature"),
    ("LSTM", "LSTM", "var(--s1)", "circle", "temperature"),
    ("GPT", "GPT", "var(--s2)", "square", "temperature"),
    ("ngram", "n-gram sampler", "var(--cand)", "diamond", "order (3, 4, 5)"),
]
EXTRA_COLOURS = ["var(--words)", "var(--ctx)", "var(--accent)", "var(--cnt)"]

W, H = 760, 470
X0, X1, Y0, Y1 = 64, 600, 40, 380  # plot box
YMIN, YMAX = 0.2, 1.0


def fx(v):
    return X0 + (X1 - X0) * v


def fy(v):
    return Y1 - (Y1 - Y0) * (v - YMIN) / (YMAX - YMIN)


def read_base(path):
    series = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["judge"] != JUDGE:
                continue
            fam = r["family"]
            if fam.startswith("ngram"):
                if abs(float(r["temp"]) - 1.0) > 1e-9:
                    continue
                series.setdefault("ngram", []).append((int(fam.split()[1]), 0.0, float(r["gram"])))
                continue
            if float(r["distinct_per_seed"]) < MIN_DISTINCT:
                continue
            series.setdefault(fam, []).append((float(r["temp"]), float(r["tri_d"]), float(r["gram"])))
    for k in series:
        series[k].sort()
    return series


def read_extra(path):
    series = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            series.setdefault((r["family"], r["dial"]), []).append((float(r["value"]), float(r["novelty"]), float(r["gram"])))
    for k in series:
        series[k].sort()
    return series


def marker(kind, x, y, colour, title):
    r = 4.5  # 9 px across
    if kind == "square":
        shape = f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}"'
    elif kind == "diamond":
        shape = f'<polygon points="{x:.1f},{y - r - 1:.1f} {x + r + 1:.1f},{y:.1f} {x:.1f},{y + r + 1:.1f} {x - r - 1:.1f},{y:.1f}"'
    else:
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"'
    return f'{shape} fill="{colour}" stroke="var(--surface)" stroke-width="1.5"><title>{html.escape(title)}</title>{"</rect>" if kind == "square" else "</polygon>" if kind == "diamond" else "</circle>"}'


def build(base, extra):
    out = []
    a = out.append
    a(f'<svg class="dg" viewBox="0 0 {W} {H}" role="img" aria-label="Grammaticality against novelty at 1,000 sentences learned, one line per generator family. '
      f'The LSTM and GPT lines lie above the TRELLIS lines at every novelty level from .3 up; n-gram samplers sit at zero novelty.">')
    # grid and axes
    for i in range(0, 6):
        v = YMIN + 0.8 * i / 5
        a(f'<line x1="{X0}" x2="{X1}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" stroke="var(--gridc)" stroke-width="1"/>')
        a(f'<text x="{X0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end" style="font-size:12px;fill:var(--muted)">{v:.1f}</text>')
    for i in range(0, 6):
        v = i / 5
        a(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{Y1}" y2="{Y1 + 5}" stroke="var(--muted)" stroke-width="1"/>')
        a(f'<text x="{fx(v):.1f}" y="{Y1 + 20}" text-anchor="middle" style="font-size:12px;fill:var(--muted)">{v:.1f}</text>')
    a(f'<line x1="{X0}" x2="{X1}" y1="{Y1}" y2="{Y1}" stroke="var(--muted)" stroke-width="1"/>')
    a(f'<text x="{(X0 + X1) / 2:.0f}" y="{Y1 + 42}" text-anchor="middle" style="font-size:13px;fill:var(--ink)">Novelty: share of new sentences with an unstored trigram</text>')
    a(f'<text transform="translate(16 {(Y0 + Y1) / 2:.0f}) rotate(-90)" text-anchor="middle" style="font-size:13px;fill:var(--ink)">Grammatical (strict judge)</text>')
    # ceiling
    a(f'<line x1="{X0}" x2="{X1}" y1="{fy(CEILING):.1f}" y2="{fy(CEILING):.1f}" stroke="var(--ink2)" stroke-width="1.5" stroke-dasharray="6 4"/>')
    a(f'<text x="{X1 - 4}" y="{fy(CEILING) - 6:.1f}" text-anchor="end" style="font-size:12px;fill:var(--ink2)">real held-out sentences: {CEILING:.3f}</text>')

    entries = []  # (label, colour, marker, pts, dial)
    for key, label, colour, mk, dial in BASE:
        if key in base:
            entries.append((label, colour, mk, base[key], dial))
    for i, ((fam, dial), pts) in enumerate(sorted(extra.items())):
        entries.append((fam, EXTRA_COLOURS[i % len(EXTRA_COLOURS)], ["circle", "square", "diamond"][i % 3], pts, dial))

    labels = []
    for label, colour, mk, pts, dial in entries:
        xy = [(fx(n), fy(g)) for (_, n, g) in pts]
        if len(xy) > 1:
            a(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in xy)}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>')
        for (v, n, g), (x, y) in zip(pts, xy):
            vs = f"{int(v)}" if label == "n-gram sampler" else f"{v:g}"
            a(marker(mk, x, y, colour, f"{label}, {dial} {vs}: novelty {n:.3f}, grammatical {g:.3f}"))
        # label at the end of the line with the highest grammaticality-to-the-right anchor
        end = max(xy, key=lambda p: p[0]) if len(xy) > 1 else xy[0]
        labels.append([end[0] + 11, end[1] + 4, label, colour])
    # nudge labels apart where they would overlap
    labels.sort(key=lambda l: l[1])
    for i in range(1, len(labels)):
        for j in range(i):
            if abs(labels[i][0] - labels[j][0]) < 130 and labels[i][1] - labels[j][1] < 14:
                labels[i][1] = labels[j][1] + 14
    for x, y, label, colour in labels:
        a(f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:12.5px;font-weight:600;fill:{colour}">{html.escape(label)}</text>')
    # legend
    lx, ly = X0, H - 14
    a(f'<text x="{lx}" y="{ly - 20}" style="font-size:12px;fill:var(--muted)">Along each line: sampling temperature rises from left to right (n-gram samplers: order 3, 4, 5, from low to high grammaticality).</text>')
    for label, colour, mk, pts, dial in entries:
        a(f'<line x1="{lx}" x2="{lx + 18}" y1="{ly - 4}" y2="{ly - 4}" stroke="{colour}" stroke-width="2"/>')
        a(marker(mk, lx + 9, ly - 4, colour, label))
        a(f'<text x="{lx + 24}" y="{ly}" style="font-size:12px;fill:var(--ink2)">{html.escape(label)}</text>')
        lx += 30 + 7 * len(label)
    a("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=CURVES)
    ap.add_argument("--extra", default=None, help="TSV with columns family, dial, value, novelty, gram")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = read_base(args.curves)
    extra = read_extra(args.extra) if args.extra else {}
    svg = build(base, extra)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(svg + "\n")
    else:
        print(svg)


if __name__ == "__main__":
    main()
