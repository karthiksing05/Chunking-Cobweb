"""Round 211 stage 3: operating points at gram >= .95 / .90 / .95 x ceiling for every generator at n=1000, strict judges.
Usage (repo root, compute node): r211_analyse.py <r211 dir> <out md>
Rules: curves are piecewise linear in temperature through the measured settings (mean over the seeds used); no extrapolation;
candidate operating points = measured settings with gram >= thr plus the interpolated crossings of thr; settings need all four seeds.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

SEEDS, SEL, HELD = (13, 17, 19, 23), (13, 17), (19, 23)
D, OUTMD = sys.argv[1], sys.argv[2]
CM = sys.argv[3] if len(sys.argv) > 3 else "gram"   # constraint metric: gram (all distinct new) or gt (gram among distinct new with an unstored trigram)
GD = "experiments/v3/results/story_gen"
JUDGES = ["Qwen3-8B/strict", "phi-4/strict"]
M = ["gram", "tri_d", "dist", "fair", "gt"]
MNAME = {"gram": "gram", "tri_d": "tri|d", "dist": "distinct new/draw", "fair": "fair (new, gram, unstored trigram)/draw"}
recs = json.load(open(os.path.join(D, "r211_sets.json"), encoding="utf-8"))
scores = json.load(open(os.path.join(D, "r211_scores.json"), encoding="utf-8"))
mean = lambda v: sum(v) / len(v)


def cell(r, j):
    sc = scores[j]
    g = [sc[s] > .5 for s, t, u in r["sents"]]
    tri = [t for s, t, u in r["sents"]]
    d = len(g)
    return dict(gt=(sum(a and t for a, t in zip(g, tri)) / sum(tri)) if sum(tri) else 0.0, gram=sum(g) / d if d else 0.0, tri_d=sum(tri) / d if d else 0.0, dist=d / r["A"],
                fair=sum(a and t for a, t in zip(g, tri)) / r["A"], d=d)


def famgroup(f):
    return f


# series[j][fam][temp][seed] = cell
series = {j: defaultdict(lambda: defaultdict(dict)) for j in JUDGES}
for j in JUDGES:
    for r in recs:
        series[j][r["fam"]][r["temp"]][r["seed"]] = cell(r, j)
    for f in list(series[j]):                      # keep only settings with all four seeds
        for t in list(series[j][f]):
            if any(s not in series[j][f][t] for s in SEEDS):
                del series[j][f][t]
        if not series[j][f]:
            del series[j][f]

ceil = {}
for j in JUDGES:
    refs = []
    for f in sorted(glob.glob(f"{GD}/r200-n1000-refs/story_s*_n1000_refs.jsonl")):
        for ln in open(f, encoding="utf-8"):
            refs.append(json.loads(ln))
    c = {k: sum(scores[j][x[k]] > .5 for x in refs) / len(refs) for k in ("real", "word order scrambled", "content swapped")}
    ceil[j] = c


MIN_D = 20          # settings with fewer than 20 distinct new sentences per seed (mean) have no usable gram estimate and are left out of the interpolation
def valid_ts(j, fam):
    return [t for t in sorted(series[j][fam]) if mean([series[j][fam][t][s]["d"] for s in SEEDS]) >= MIN_D]


def curve(j, fam, seeds, metric):
    return [(t, mean([series[j][fam][t][s][metric] for s in seeds])) for t in valid_ts(j, fam)]


def interp(pts, x):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 - 1e-12 <= x <= x1 + 1e-12:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[0][1] if len(pts) == 1 and abs(pts[0][0] - x) < 1e-12 else None


def candidates(gpts, thr):
    """temperatures at which interpolated gram >= thr that bound the feasible set (measured feasible points + crossings)."""
    c = [t for t, g in gpts if g >= thr]
    for (t0, g0), (t1, g1) in zip(gpts, gpts[1:]):
        if (g0 - thr) * (g1 - thr) < 0:
            c.append(t0 + (t1 - t0) * (thr - g0) / (g1 - g0))
    return sorted(set(c))


def best_point(j, fam, seeds, metric, thr):
    g = curve(j, fam, seeds, CM)
    cs = candidates(g, thr)
    if not cs:
        return None
    mp = curve(j, fam, seeds, metric)
    t = max(cs, key=lambda x: interp(mp, x))
    return t, interp(mp, t), interp(g, t)


def at(j, fam, seed, metric, t):
    return interp(curve(j, fam, [seed], metric), t)


THR = lambda j: [(".95", .95), (".90", .90), (".85", .85), (".80", .80), (".75", .75), (".70", .70), ("0.95 x ceiling (%.3f)" % (.95 * ceil[j]["real"]), .95 * ceil[j]["real"])]
L = ["# Round 211: the %s >= .95 operating point at n=1000 (strict judges)" % CM, ""]
L.append("Code: `experiments/v3/diag/r211_collect_judge.py` (collect + judge), `r211_analyse.py` (this file); grid `logs/v3_r211_grid.sh`; per-sentence flags `r211/r211_sets.json`, scores `r211/r211_scores.json`, curves `r211/r211_curves.tsv`.")
L.append("")
for j in JUDGES:
    c = ceil[j]
    L.append(f"* {j}: real-sentence ceiling {c['real']:.3f} (1200 held-out refs), scrambled false-pass {c['word order scrambled']:.3f}, content-swapped {c['content swapped']:.3f}.")
L.append("")
tsv = ["judge\tfamily\ttemp\tgram\ttri_d\tdist\tfair\tgram_given_tri\tdistinct_per_seed"]
for j in JUDGES:
    L.append(f"\n## Curves under {j} (mean of seeds 13/17/19/23; gram = share of distinct new judged grammatical; tri|d = share of distinct new with an unstored trigram; dist = distinct new per attempted draw; fair = distinct new, grammatical, unstored trigram per attempted draw)\n")
    L.append("| family | temp | gram | tri\\|d | dist/draw | fair | gram\\|tri | distinct/seed |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in sorted(series[j], key=lambda f: (not f.startswith("TRELLIS edge"), not f.startswith("TRELLIS"), f)):
        for t in sorted(series[j][f]):
            v = {m: mean([series[j][f][t][s][m] for s in SEEDS]) for m in M}
            gt = v["fair"] / (v["tri_d"] * v["dist"]) if v["tri_d"] * v["dist"] > 0 else float("nan")
            L.append(f"| {f} | {t:.2f} | {v['gram']:.3f} | {v['tri_d']:.3f} | {v['dist']:.3f} | {v['fair']:.3f} | {gt:.3f} | {mean([series[j][f][t][s]['d'] for s in SEEDS]):.0f}{"" if t in valid_ts(j, f) else " (below MIN_D, not used)"} |")
            tsv.append("\t".join([j, f, "%.2f" % t] + ["%.4f" % v[m] for m in M] + [str(mean([series[j][f][t][s]["d"] for s in SEEDS]))]))
open(os.path.join(D, "r211_curves.tsv"), "w", encoding="utf-8").write("\n".join(tsv) + "\n")

L.append("\n## Maximum tri|d and maximum fair count per family at a gram constraint (mean of 4 seeds; temperature in brackets; '-' = gram constraint not met by any measured or interpolated point of the curve, no extrapolation)\n")
for j in JUDGES:
    for name, thr in THR(j):
        L.append(f"\n### {j}, {CM} >= {name}\n")
        L.append("| family | lowest-temp gram (T) | max tri\\|d (T, gram there) | max fair (T, gram there) | dist/draw at max-fair T |")
        L.append("|---|---|---|---|---|")
        for f in sorted(series[j], key=lambda f: (not f.startswith("TRELLIS edge"), not f.startswith("TRELLIS"), f)):
            g = curve(j, f, SEEDS, CM)
            if not g:
                continue
            lo = f"{g[0][1]:.3f} (t{g[0][0]:.2f})"
            a = best_point(j, f, SEEDS, "tri_d", thr)
            b = best_point(j, f, SEEDS, "fair", thr)
            if a is None:
                L.append(f"| {f} | {lo} | - | - | - |")
            else:
                dd = interp(curve(j, f, SEEDS, "dist"), b[0])
                L.append(f"| {f} | {lo} | {a[1]:.3f} (t{a[0]:.2f}, {a[2]:.3f}) | {b[1]:.3f} (t{b[0]:.2f}, {b[2]:.3f}) | {dd:.3f} |")

# ---- paired comparison: choose family+temperature on seeds 13/17, report on 19/23
TR = lambda j: [f for f in series[j] if f.startswith("TRELLIS")]
LM = ["LSTM", "GPT"]
L.append("\n## Paired comparison at the operating point (family and temperature chosen on seeds 13/17; per-seed values on 19/23 are the held-out report; each seed's own curve is interpolated at the chosen temperature)\n")
L.append("TRELLIS candidates = every plain and edge window; the one with the best selection-seed value is carried forward. Held-out gram is the value on seeds 19/23 at the chosen temperature (the constraint was only enforced on 13/17).\n")
summary = {}
per_seed_summary = {}
for j in JUDGES:
    for name, thr in THR(j):
        for metric in ("fair", "tri_d"):
            L.append(f"\n### {j}, {CM} >= {name}, metric {MNAME[metric]}\n")
            L.append("| arm | chosen setting | sel (13/17) value | sel gram | held-out (19/23) value | held-out gram | per-seed value 13/17/19/23 |")
            L.append("|---|---|---|---|---|---|---|---|")
            ch = {}
            cand = {"TRELLIS": TR(j), "LSTM": ["LSTM"], "GPT": ["GPT"]}
            for arm, fams in cand.items():
                best = None
                for f in fams:
                    if f not in series[j]:
                        continue
                    p = best_point(j, f, SEL, metric, thr)
                    if p and (best is None or p[1] > best[1][1]):
                        best = (f, p)
                if best is None:
                    L.append(f"| {arm} | none reaches the constraint | - | - | - | - | - |")
                    continue
                f, (t, val, gs) = best
                ps = {s: at(j, f, s, metric, t) for s in SEEDS}
                pg = {s: at(j, f, s, CM, t) for s in SEEDS}
                ch[arm] = (f, t, ps, pg)
                L.append(f"| {arm} | {f} t{t:.2f} | {val:.3f} | {gs:.3f} | {mean([ps[s] for s in HELD]):.3f} | {mean([pg[s] for s in HELD]):.3f} | " + " / ".join(f"{ps[s]:.3f}" for s in SEEDS) + " |")
            for o in LM:
                if "TRELLIS" in ch and o in ch:
                    a, b = ch["TRELLIS"][2], ch[o][2]
                    sel = [a[s] - b[s] for s in SEL]
                    held = [a[s] - b[s] for s in HELD]
                    L.append(f"\nTRELLIS ({ch['TRELLIS'][0]} t{ch['TRELLIS'][1]:.2f}) minus {o} (t{ch[o][1]:.2f}): selection seeds {mean(sel):+.3f} (wins {sum(x > 0 for x in sel)}/2); held-out seeds {mean(held):+.3f} (wins {sum(x > 0 for x in held)}/2; per seed " + ", ".join(f"{s}:{a[s] - b[s]:+.3f}" for s in HELD) + ")")
                    summary[(j, name, metric, o)] = (ch["TRELLIS"][0], mean(sel), mean(held), sum(x > 0 for x in held))
                    ps2 = {}
                    for arm2, fam2 in (("TRELLIS", ch["TRELLIS"][0]), (o, o)):
                        ps2[arm2] = {s: best_point(j, fam2, [s], metric, thr) for s in SEEDS}
                    ok = [s for s in SEEDS if ps2["TRELLIS"][s] and ps2[o][s]]
                    dd = {s: ps2["TRELLIS"][s][1] - ps2[o][s][1] for s in ok}
                    per_seed_summary[(j, name, metric, o)] = (ch["TRELLIS"][0], dd, {s: (ps2["TRELLIS"][s][2], ps2[o][s][2]) for s in ok})
L.append("\n## Compact verdict table: TRELLIS minus opponent, held-out seeds 19/23\n")
L.append("| judge | gram constraint | metric | vs | TRELLIS pick | sel diff | held-out diff | held-out wins |")
L.append("|---|---|---|---|---|---|---|---|")
for (j, name, metric, o), (f, sd, hd, w) in sorted(summary.items()):
    L.append(f"| {j} | {name} | {MNAME[metric]} | {o} | {f} | {sd:+.3f} | {hd:+.3f} | {w}/2 |")

L.append("\n## Same comparison with the constraint enforced on each seed's own curve (family chosen on 13/17 as above)\n")
L.append("Removes seed-to-seed differences in grammaticality from the comparison: every seed is evaluated at its own interpolated point where its gram first reaches the constraint (or its best feasible point); seeds where either arm never reaches it are dropped (listed as n).\n")
L.append("| judge | gram constraint | metric | vs | TRELLIS pick | mean diff all seeds (n) | mean diff 13/17 | mean diff 19/23 | wins on 19/23 | per-seed diffs |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for (j, name, metric, o), (f, dd, gg) in sorted(per_seed_summary.items()):
    if not dd:
        L.append(f"| {j} | {name} | {MNAME[metric]} | {o} | {f} | no seed reaches the constraint with both arms | - | - | - | - |")
        continue
    sl = [dd[s] for s in SEL if s in dd]
    hl = [dd[s] for s in HELD if s in dd]
    L.append(f"| {j} | {name} | {MNAME[metric]} | {o} | {f} | {mean(list(dd.values())):+.3f} ({len(dd)}) | {('%+.3f' % mean(sl)) if sl else '-'} | {('%+.3f' % mean(hl)) if hl else '-'} | {sum(x > 0 for x in hl)}/{len(hl)} | " + ", ".join(f"{s}:{dd[s]:+.3f}" for s in dd) + " |")

# ---- copy / splice share among grammatical distinct new sentences at the settings nearest the constraint
L.append("\n## Copy and splice share (distinct new sentences judged grammatical)\n")
L.append("For each family and judge: the measured setting with the lowest temperature, and the measured setting with the highest gram that is closest to .95 from above if any (else the highest-gram setting). `tri=0` = distinct new sentence all of whose trigrams are stored (a splice of stored pieces or a stored span, no new trigram). `stored span` = it is a proper contiguous span of one stored sentence (a headless or truncated cut of a stored sentence). Exact stored copies: the dumps already exclude them (generation draws are 'new' only, so the share is 0 by construction in this column); they never enter any count here.\n")
L.append("| judge | family | T | gram | grammatical distinct new per draw | of those: tri=0 | of those: stored span | exact stored copies, share of draws | grammatical AND unstored trigram per draw |")
L.append("|---|---|---|---|---|---|---|---|---|")
bykey = {(r["fam"], r["temp"], r["seed"]): r for r in recs}
for j in JUDGES:
    for f in sorted(series[j], key=lambda f: (not f.startswith("TRELLIS edge"), not f.startswith("TRELLIS"), f)):
        ts = valid_ts(j, f)
        gm = {t: mean([series[j][f][t][s]["gram"] for s in SEEDS]) for t in ts}
        above = [t for t in ts if gm[t] >= .95]
        pick = {ts[0]}
        pick.add(min(above, key=lambda t: gm[t]) if above else max(ts, key=lambda t: gm[t]))
        for t in sorted(pick):
            n_g = n_t0 = n_sub = 0
            A = cp = 0
            for s in SEEDS:
                r = bykey[(f, t, s)]
                for sent, tri, sub in r["sents"]:
                    if scores[j][sent] > .5:
                        n_g += 1
                        n_t0 += (not tri)
                        n_sub += bool(sub)
                A += r["A"]
                cp += r["copyrows"]
            Fa = mean([series[j][f][t][s]["fair"] for s in SEEDS])
            L.append(f"| {j} | {f} | {t:.2f} | {gm[t]:.3f} | {n_g / A:.3f} | {n_t0 / n_g if n_g else 0:.3f} | {n_sub / n_g if n_g else 0:.3f} | {cp / A:.3f} | {Fa:.3f} |")
open(OUTMD, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
