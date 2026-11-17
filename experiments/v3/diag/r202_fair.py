"""Round 202: fair productivity (story_frontier_fair.py, unchanged rules) with the parse-guided arm as its own family,
then selection on seeds 13 / 17 and held-out report on 19 / 23 (n = 4000: seeds 13 / 17 only, both reported).

Family: an arm with "gen": "parse" is "TRELLIS parse w<W>"; everything else as story_frontier_plot_n.family.
Outputs <prefix>.tsv / .perseed.tsv / .txt (story_frontier_fair.main) and <prefix>.select.txt:
  (a) per setting (family, window, temp): mean over SEL and over HELD seeds of the fair count (distinct new,
      grammatical, unstored-trigram sentences per ATTEMPTED draw), tri, gram, gram|tri (shares among distinct new);
  (b) per family, the setting with the best fair count on SEL, reported on HELD, paired per seed vs the others;
  (c) held-out envelopes at tri = .1 .. .7 (linear interpolation along temperature within a family-window curve, no
      extrapolation): gram vs tri, and gram|tri vs tri.

Usage (repo root, compute node): r202_fair.py <n> <out prefix> <frontier.jsonl> [<frontier.jsonl> ...]
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
import story_frontier_fair as sff  # noqa: E402
import story_frontier_plot_n as spn  # noqa: E402

SEL, HELD = (13, 17), (19, 23)


def family_of(r):
    arm = json.loads(r["arm"])
    if arm.get("gen") in ("parse", "close"):
        m = re.search(r"-w(\d+)$", r["dir"])
        return "TRELLIS %s w%s%s" % (arm["gen"], m.group(1), "" if arm.get("M", 16) == 16 else " M%d" % arm["M"]), arm.get("temp", 1.0)
    return spn.family(r)


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def f3(x):
    return "-" if x is None else "%.3f" % x


def main(n, prefix, jls):
    rows = []
    for jl in jls:
        rows += [json.loads(l) for l in open(jl, encoding="utf-8")]
    rows = [r for r in rows if r["n"] == n]
    seen, uniq = set(), []
    for r in rows:                                   # a dump listed in two frontier files counts once
        k = (r["dir"], r["file"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    merged = prefix + ".frontier.jsonl"
    with open(merged, "w", encoding="utf-8") as fh:
        for r in uniq:
            fh.write(json.dumps(r) + "\n")
    sff.family = family_of
    sff.main(merged, n, prefix)
    per = defaultdict(dict)                          # (family, temp) -> seed -> metrics
    for ln in open(prefix + ".perseed.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = ln.split("\t")
        fam, temp, seed, A, ret, d, g, tr, gn = c[0], float(c[1]), int(c[2]), *map(int, c[3:9])
        per[(fam, temp)][seed] = dict(fair=gn / A, tri=tr / d if d else None, gram=g / d if d else None,
                                      gt=gn / tr if tr else None)
    seeds = sorted({s for d in per.values() for s in d})
    only = os.environ.get("R202_SEEDS")              # probe: restrict to these seeds (selection = report)
    if only:
        seeds = [int(s) for s in only.split(",")]
        per = {k: {s: v for s, v in d.items() if s in seeds} for k, d in per.items()}
    sel = [s for s in SEL if s in seeds] or seeds
    held = [s for s in HELD if s in seeds] or sel
    out = [f"n={n}; seeds {seeds}; selection seeds {sel}; reported on {held}"]

    def fam_group(f):
        if f.startswith("TRELLIS parse"):
            return "TRELLIS parse"
        if f.startswith("TRELLIS close"):
            return "TRELLIS close"
        if re.fullmatch(r"TRELLIS w\d+", f):
            return "TRELLIS"
        return f if f in ("LSTM", "GPT") else None
    full = {k: d for k, d in per.items() if fam_group(k[0]) and all(s in d for s in seeds)}
    out.append("\n(a) per setting: fair count, tri, gram, gram|tri -- mean over selection seeds | held-out seeds")
    for k in sorted(full, key=lambda k: (fam_group(k[0]) != "TRELLIS parse", k)):
        d = full[k]
        out.append("  %-22s t%.2f  sel %s %s %s %s | held %s %s %s %s" % (
            k[0], k[1], *[f3(mean([d[s][m] for s in sel])) for m in ("fair", "tri", "gram", "gt")],
            *[f3(mean([d[s][m] for s in held])) for m in ("fair", "tri", "gram", "gt")]))
    best = {}
    out.append("\n(b) setting with the best fair count on %s, reported on %s" % (sel, held))
    for g in ("TRELLIS parse", "TRELLIS close", "TRELLIS", "LSTM", "GPT"):
        ks = [k for k in full if fam_group(k[0]) == g]
        if not ks:
            continue
        b = max(ks, key=lambda k: mean([full[k][s]["fair"] for s in sel]))
        best[g] = b
        d = full[b]
        out.append("  %-14s %s t%.2f: sel fair %s | held fair %s, tri %s, gram %s, gram|tri %s; per seed fair %s" % (
            g, b[0], b[1], f3(mean([d[s]["fair"] for s in sel])),
            *[f3(mean([d[s][m] for s in held])) for m in ("fair", "tri", "gram", "gt")],
            ", ".join("%d:%.3f" % (s, d[s]["fair"]) for s in seeds)))
    if "TRELLIS parse" in best:
        p = full[best["TRELLIS parse"]]
        for g in ("TRELLIS close", "TRELLIS", "LSTM", "GPT"):
            if g in best:
                o = full[best[g]]
                diffs = [(s, p[s]["fair"], o[s]["fair"]) for s in held]
                out.append("  paired fair count, parse vs %s on %s: wins %d of %d; mean diff %+.3f; %s" % (
                    g, held, sum(a > b for _, a, b in diffs), len(diffs), mean([a - b for _, a, b in diffs]),
                    ", ".join("%d:%.3f/%.3f" % z for z in diffs)))
    curves = defaultdict(list)
    for (f, t), d in per.items():
        if all(s in d for s in held) and (fam_group(f) or f.startswith("TRELLIS")):
            curves[f].append((mean([d[s]["tri"] for s in held]), mean([d[s]["gram"] for s in held]),
                              mean([d[s]["gt"] for s in held])))
    fams = sorted(curves, key=lambda f: (not f.startswith("TRELLIS parse"), not f.startswith("TRELLIS"), f))
    for title, yi in (("gram (distinct new) vs tri", 1), ("gram|tri vs tri", 2)):
        out += ["", "(c) held-out envelope on %s: %s (interpolated along temperature, no extrapolation)" % (held, title),
                "tri | " + " | ".join(fams)]
        for x in spn.LEVELS:
            cells = []
            for f in fams:
                pts = [(c[0], c[yi]) for c in curves[f] if c[0] is not None and c[yi] is not None]
                y = spn.interp(pts, x)
                cells.append(f3(y))
            out.append("%.1f | %s" % (x, " | ".join(cells)))
    open(prefix + ".select.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3:])
