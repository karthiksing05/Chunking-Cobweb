"""Tables and figures for the chess storage test (results/chess_memory).

    python experiments/v2/summarize_chess_memory.py [results dir]
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
NS = (1000, 4000, 16000, 32000)
KINDS = ("real", "random")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e2dc"
FAMILY = {"ar": ("#2a78d6", "transformer (AR)"), "ae": ("#eb6834", "autoencoder")}
UNIFORM = 64 * math.log2(13)

# Written by hand from the first complete pass; regenerate the tables, then re-read this.
FINDINGS = """## Findings (first complete pass, 2026-10-08; seed-13 split, one training seed per net)

**Confirmed (measured).**

- Real against random, held out: every model of positions shows a real-random gap of 51-68 bits/position (per-square categorical 53-54, TRELLIS read alone 51-52, TRELLIS grammar 52, autoencoder 57-61, transformer 59-68). Most of it is already in the per-square marginals. TRELLIS's chunks add 0.1-0.4 held-out bits/position on real boards (grammar against read alone: 0.38, 0.09, 0.11, 0.13 at N = 1000, 4000, 16000, 32000) and exactly 0 on random boards, where the search finds no chunk type at any N. On real boards it finds 0, 0, 1 and 4 chunk types (the first is always `[wP N1 bP]`, a blocked pawn pair).
- Only the transformer's gap grows with N (58.7, 62.6, 66.3, 68.4), so it keeps learning structure of real positions beyond the marginals and the counts. On random boards it reaches about 0.75 bits of the generator's own code (125.38 against 124.63 at N = 32000). TRELLIS stays about 2 bits above that floor.
- Held out, the transformer beats TRELLIS at every N, by 6.3, 10.9, 15.4 and 17.3 bits/position (71.1 / 65.2 / 60.0 / 57.0 against 77.4 / 76.1 / 75.4 / 74.3). The autoencoder stays above TRELLIS at every N (76.8 at best, N = 32000).
- Total code with parameters paid for. TRELLIS's total is a Bayesian-mixture (adaptive) code, so the like-for-like net code is the prequential one. Prequential: a 1,629-parameter transformer already beats TRELLIS at N = 1000 (84.6 against 86.7), and the best sizes (28k-156k parameters) beat it by 4.5, 6.8, 11.0 and 13.0 bits/position at N = 1000, 4000, 16000, 32000. Two-part with 8-bit weights: no net beats TRELLIS at N = 1000 (best 87.3 against 86.7); the crossover lies between N = 1000 and 4000 (at 4000: 1,629 parameters, 76.7 against 79.1). With 16-bit weights it lies between 4000 and 16000. The two-part optimum grows with N (about 1.6k parameters at N = 1000-4000, 8k at 16000, 28k at 32000); above it b*P/N dominates, e.g. 805k parameters cost 6,450 bits/position at N = 1000.
- Random positions: nets beat TRELLIS's total code only by 1-10 bits/position prequentially (most at N = 1000) and two-part only at N = 1000 (b = 8) and N = 32000 (b = 8, by 0.1).
- Quantization: at the sizes near the two-part optimum, 8-bit weights cost 0.1-1.5 bits/position of training NLL (3.0 at worst). Models that memorise small N lose up to 11 bits/position, but they are far from optimal anyway.
- Byte compressors lose to every model: xz -9e on the whole corpus reaches 116.5 bits/position on real boards at N = 32000, against 85 for the per-square model. Compressing each position alone costs more than raw FEN (headers).
- Reproduction: TRELLIS on the first 16,000 real positions reproduces the committed chess_1500_16000 run exactly (held out 75.37, read alone 75.48, total 76.28, 18,679 model bits, one chunk type).

**Suspected / caveats (not established).**

- The nets are lightly tuned: one learning rate per size, no dropout, one seed. A 20k-step sweep is included, and the best-size rows take the better of the two budgets. Better tuning would more likely widen the nets' lead than close it. Prequential totals are often identical across budgets, because early stopping triggered before 5k steps on the small prefixes.
- The autoencoder's rate gradient reaches the encoder through a soft-assignment surrogate. Its weak showing may partly reflect that estimator, not the model class. Its encoder is not charged in the two-part code (the receiver does not need it).
- The random floor uses a plug-in material entropy (4,804 material sets over 57,884 positions, 2,354 of them singletons), so it is probably optimistic by up to about 1 bit.
- Parameter bits are plain b*P (+32 per tensor). Entropy-coding the quantized weights would lower them, so the two-part crossovers are conservative for the nets.
"""


def load(out):
    tr = {}
    for p in glob.glob(os.path.join(out, "trellis_*.json")):
        r = json.load(open(p))
        tr[(r["kind"], r["N"])] = r
    base = {}
    if os.path.exists(os.path.join(out, "baselines.json")):
        for r in json.load(open(os.path.join(out, "baselines.json"))):
            base[(r["kind"], r["N"])] = r
    nets = {}
    # Nets at every training budget (out/nets_*.json: 5,000 steps;
    # out/steps20k/nets_*.json: 20,000); a model's name carries its budget.
    for p in glob.glob(os.path.join(out, "nets_*.json")) + glob.glob(os.path.join(out, "steps*", "nets_*.json")):
        r = json.load(open(p))
        r["model"] = f"{r['model']}@{r['steps'] // 1000}k"
        nets[(r["model"], r["kind"])] = r
    return tr, base, nets


def f(x, d=2):
    return "–" if x is None else (f"{x:,.{d}f}" if abs(x) < 1e5 else f"{x:,.0f}")


def best_compressor(b, which="corpus / N"):
    keys = [k for k in b if k.endswith(which)]
    k = min(keys, key=lambda k: b[k])
    return k.split(",")[0], b[k]


def two_part(r, N, b):
    e = r["two-part"].get(str(N))
    return None if e is None else e[f"b={b}"]["total bits/pos"]


def summarize(out):
    tr, base, nets = load(out)
    L = ["# Chess positions as a storage test (Track 2)", "",
         "Positions: Lichess, both players 1500+, after ply 30 (data/chess/positions_1500_ply30.fen), the split of "
         "run_chess.py (seed-13 permutation; first N train, next 500 held out). Random: each position's own pieces "
         "placed uniformly on distinct squares, pawns on ranks 2-7. All numbers are bits per position.", "",
         FINDINGS, "Figures: tradeoff_total.png (total code against transmitted parameters), tradeoff_heldout.png.", ""]

    # 1. Storage of the N training positions (total code).
    L += ["## Total code for the N training positions (bits/position; parameters paid for)", "",
          "TRELLIS and the read alone: adaptive (Dirichlet-mixture) codes, i.e. the grammar's model bits + data bits. "
          "Nets: best size per N; two-part with b-bit parameters, and prequential (no parameter convention).", ""]
    hdr = "| Method | " + " | ".join(f"{k} {n}" for n in NS for k in KINDS) + " |"
    L += [hdr, "|---|" + "---|" * (2 * len(NS))]

    def row(name, fn):
        vals = []
        for n in NS:
            for k in KINDS:
                try:
                    vals.append(fn(k, n))
                except (KeyError, TypeError, ValueError):
                    vals.append(None)
        return "| " + name + " | " + " | ".join(f(v) for v in vals) + " |"

    L.append(row("uniform, 64 log2 13", lambda k, n: UNIFORM))
    for c in ("gzip -9", "xz -9e", "zstd -19"):
        L.append(row(f"{c}, each position alone", lambda k, n, c=c: base[(k, n)][f"{c}, per position"]))
        L.append(row(f"{c}, corpus / N", lambda k, n, c=c: base[(k, n)][f"{c}, corpus / N"]))
    L.append(row("per-square categorical (KT mixture)",
                 lambda k, n: base[(k, n)]["per-square categorical"]["train bits/pos, adaptive code (KT, a=0.5)"]))
    L.append(row("TRELLIS read, squares on their own",
                 lambda k, n: tr[(k, n)]["train bits/pos, squares on their own (adaptive code)"]))
    L.append(row("TRELLIS read alone (counts, no chunks)",
                 lambda k, n: tr[(k, n)]["train bits/pos, the read alone (adaptive code)"]))
    L.append(row("**TRELLIS v2 (model + data bits)**",
                 lambda k, n: tr[(k, n)]["train bits/pos, TRELLIS v2 (two-part: model + data)"]))
    for fam in ("ar", "ae"):
        for b in (8, 16):
            L.append(row(f"{FAMILY[fam][1]}, two-part b={b}, best size", lambda k, n, fam=fam, b=b: min(
                two_part(r, n, b) for (m, kk), r in nets.items() if kk == k and m.startswith(fam)
                and two_part(r, n, b) is not None)))
        L.append(row(f"{FAMILY[fam][1]}, prequential, best size", lambda k, n, fam=fam: min(
            r["prequential bits/pos"][str(n)] for (m, kk), r in nets.items() if kk == k and m.startswith(fam)
            and str(n) in r["prequential bits/pos"])))
    L.append("")

    # 2. Held out.
    L += ["## Held-out bits/position (500 positions after the first N)", "",
          "Nets: the early-stopped model trained on the N-prefix (the prequential stage), best size; "
          "compressors: what 500 positions add to the compressed corpus.", ""]
    L += [hdr, "|---|" + "---|" * (2 * len(NS))]
    for c in ("gzip -9", "xz -9e", "zstd -19"):
        L.append(row(c, lambda k, n, c=c: base[(k, n)][f"{c}, held out (corpus+test minus corpus) / 500"]))
    L.append(row("per-square categorical", lambda k, n: base[(k, n)]["per-square categorical"]["held-out bits/pos"]))
    L.append(row("TRELLIS read, squares on their own", lambda k, n: tr[(k, n)]["held-out bits/pos, squares on their own"]))
    L.append(row("TRELLIS read alone", lambda k, n: tr[(k, n)]["held-out bits/pos, the read alone"]))
    L.append(row("**TRELLIS v2 (the grammar)**", lambda k, n: tr[(k, n)]["held-out bits/pos, TRELLIS v2 (the grammar)"]))
    op = os.path.join(out, "oracle_random.json")
    if os.path.exists(op):
        orc = json.load(open(op))["held-out bits/pos, the generator's own code (random positions)"]
        L.append(row("random positions' own generator (floor; plug-in material entropy)",
                     lambda k, n: orc[str(n)] if k == "random" else None))
    for fam in ("ar", "ae"):
        L.append(row(f"{FAMILY[fam][1]}, best size", lambda k, n, fam=fam: min(
            r["held-out bits/pos (early-stopped prefix model)"][str(n)] for (m, kk), r in nets.items()
            if kk == k and m.startswith(fam) and str(n) in r["held-out bits/pos (early-stopped prefix model)"])))
    L.append("")

    # Crossovers: the smallest net (and budget) whose total code beats TRELLIS's.
    L += ["## Crossovers: smallest net whose total code for the N training positions is below TRELLIS v2's", "",
          "| kind | N | TRELLIS v2 total | code | smallest net below it (params: bits/pos) | best net (params: bits/pos) |",
          "|---|---|---|---|---|---|"]
    for k in KINDS:
        for n in NS:
            t = tr.get((k, n))
            if not t:
                continue
            T = t["train bits/pos, TRELLIS v2 (two-part: model + data)"]
            for code, fn in (("two-part b=8", lambda r: two_part(r, n, 8)), ("two-part b=16", lambda r: two_part(r, n, 16)),
                             ("prequential", lambda r: r["prequential bits/pos"].get(str(n)))):
                cands = sorted([(r["params (transmitted)"], fn(r), m) for (m, kk), r in nets.items()
                                if kk == k and fn(r) is not None])
                below = [c for c in cands if c[1] < T]
                best = min(cands, key=lambda c: c[1]) if cands else None
                L.append(f"| {k} | {n} | {T:.2f} | {code} | "
                         + (f"{below[0][2]} ({below[0][0]:,}: {below[0][1]:.2f})" if below else "none") + " | "
                         + (f"{best[2]} ({best[0]:,}: {best[1]:.2f})" if best else "–") + " |")
    L.append("")

    # 3. TRELLIS details.
    L += ["## TRELLIS v2 runs", "", "| kind | N | model bits | data bits | chunk types | symbols | first moves | seconds |",
          "|---|---|---|---|---|---|---|---|"]
    for k in KINDS:
        for n in NS:
            r = tr.get((k, n))
            if r:
                L.append(f"| {k} | {n} | {r['model bits']:,.0f} | {r['data bits']:,.0f} | {r['chunk types']} | "
                         f"{r['symbols']} | {'; '.join(r['search moves'][:3])} | {r['seconds']:.0f} |")
    L.append("")

    # 4. Per-size nets.
    L += ["## Neural models by size", "",
          "Name@budget: the maximum number of training steps (batch 256; early stopping may stop sooner). "
          "Best-size rows above take the minimum over sizes and budgets. params = transmitted parameters (AE: decoder + latent prior; the encoder stays with the sender). "
          "Two-part = quantized model's training NLL + b * params + 32 bits per tensor.", ""]
    for k in KINDS:
        L += [f"### {k}", "", "| model | params | N | two-part b=8 | two-part b=16 | prequential | "
              "train NLL (b=16) | held-out (fit, b=16) | held-out (early-stopped) |", "|---|---|---|---|---|---|---|---|---|"]
        for (m, kk), r in sorted(nets.items(), key=lambda x: (x[0][0][:2], x[1]["params (transmitted)"])):
            if kk != k:
                continue
            for n in NS:
                e = r["two-part"].get(str(n))
                if not e:
                    continue
                L.append(f"| {m} | {r['params (transmitted)']:,} | {n} | {f(e['b=8']['total bits/pos'])} | "
                         f"{f(e['b=16']['total bits/pos'])} | {f(r['prequential bits/pos'].get(str(n)))} | "
                         f"{f(e['b=16']['train data bits/pos'])} | {f(e['b=16']['held-out bits/pos'])} | "
                         f"{f(r['held-out bits/pos (early-stopped prefix model)'].get(str(n)))} |")
        L.append("")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"trellis": list(tr.values()), "baselines": list(base.values()), "nets": list(nets.values())}, fh, indent=1)
    figures(out, tr, base, nets)


def figures(out, tr, base, nets):
    for what in ("total", "heldout"):
        fig, axes = plt.subplots(2, len(NS), figsize=(4.0 * len(NS), 7.2), facecolor=SURFACE, sharex=True)
        for row, k in enumerate(KINDS):
            for col, n in enumerate(NS):
                ax = axes[row, col]
                ax.set_facecolor(SURFACE)
                for fam, (color, label) in FAMILY.items():
                    rs = sorted([r for (m, kk), r in nets.items() if kk == k and m.startswith(fam)],
                                key=lambda r: r["params (transmitted)"])
                    if what == "total":
                        series = [(f"{label}, two-part b=8", "-", "o",
                                   lambda r: two_part(r, n, 8)),
                                  (f"{label}, two-part b=16", "--", "s", lambda r: two_part(r, n, 16)),
                                  (f"{label}, prequential", ":", "^",
                                   lambda r: r["prequential bits/pos"].get(str(n)))]
                    else:
                        series = [(f"{label}, early-stopped", "-", "o",
                                   lambda r: r["held-out bits/pos (early-stopped prefix model)"].get(str(n))),
                                  (f"{label}, fit to N, b=16", "--", "s",
                                   lambda r: (r["two-part"].get(str(n)) or {}).get("b=16", {}).get("held-out bits/pos"))]
                    by_p = {}
                    for r in rs:
                        by_p.setdefault(r["params (transmitted)"], []).append(r)
                    for lab, ls, mk, fn in series:
                        # The best training budget at each size.
                        pts = [(pp, min(v)) for pp, group in sorted(by_p.items())
                               if (v := [fn(r) for r in group if fn(r) is not None])]
                        if pts:
                            ax.plot(*zip(*pts), ls=ls, marker=mk, ms=4, lw=1.6, color=color, label=lab)
                t, b = tr.get((k, n)), base.get((k, n))
                lines = []
                if t:
                    if what == "total":
                        lines += [("TRELLIS v2 (model + data)", t["train bits/pos, TRELLIS v2 (two-part: model + data)"], INK, "-"),
                                  ("read alone, no chunks", t["train bits/pos, the read alone (adaptive code)"], MUTED, "-.")]
                    else:
                        lines += [("TRELLIS v2 (the grammar)", t["held-out bits/pos, TRELLIS v2 (the grammar)"], INK, "-"),
                                  ("read alone, no chunks", t["held-out bits/pos, the read alone"], MUTED, "-.")]
                if b:
                    ps = b["per-square categorical"]
                    lines.append(("per-square categorical",
                                  ps["train bits/pos, adaptive code (KT, a=0.5)"] if what == "total" else ps["held-out bits/pos"],
                                  MUTED, "--"))
                    name, v = best_compressor(b, "corpus / N" if what == "total" else "/ 500")
                    lines.append(("best byte compressor (whole corpus)", v, MUTED, ":"))
                orc_p = os.path.join(out, "oracle_random.json")
                if what == "heldout" and k == "random" and os.path.exists(orc_p):
                    v = json.load(open(orc_p))["held-out bits/pos, the generator's own code (random positions)"][str(n)]
                    lines.append(("random positions' generator (floor)", v, "#1baf7a", "-"))
                for lab, v, c, ls in lines:
                    ax.axhline(v, color=c, ls=ls, lw=1.2, label=lab)
                ax.set_xscale("log")
                ref = [v for _, v, _, _ in lines] or [UNIFORM / 2]
                lo = min([v for l in ax.get_lines() for v in l.get_ydata()] + ref)
                ax.set_ylim(lo * 0.95, max(ref) * (1.25 if what == "total" else 1.1))
                ax.grid(True, color=GRID, lw=0.6, which="both")
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                ax.tick_params(colors=INK2, labelsize=8)
                ax.set_title(f"{k} positions, N = {n:,}", fontsize=9, color=INK)
                if col == 0:
                    ax.set_ylabel("total bits / position" if what == "total" else "held-out bits / position",
                                  color=INK2, fontsize=9)
                if row == 1:
                    ax.set_xlabel("transmitted parameters", color=INK2, fontsize=9)
        h, lab = axes[0, 0].get_legend_handles_labels()
        h2, lab2 = axes[1, 0].get_legend_handles_labels()
        h += [x for x, l in zip(h2, lab2) if l not in lab]
        lab += [l for l in lab2 if l not in lab]
        fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle("Storing chess positions: " + ("total code of the N training positions (parameters paid for)"
                                                   if what == "total" else "held-out code"), color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0.1, 1, 0.96))
        fig.savefig(os.path.join(out, f"tradeoff_{what}.png"), dpi=150, facecolor=SURFACE)
        plt.close(fig)


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "chess_memory"))
