"""
Shared multi-seed orchestration for ACS-26 confs.

Every conf in this folder uses the same pattern: train+eval the TRELLIS
pipeline on a corpus with multiple seeds, then aggregate the
per-seed CSVs into mean ± 1 std (for learning curves) or
mean ± std summary statistics (for final headline metrics).

This file factors out:

    SEEDS                            — canonical 5-seed list
    load_learning_curve_csv(path)    — read a learning_curves.csv
    aggregate_seeds(seed_curves)     — mean/std across seeds
    write_aggregated_csv(path, agg)
    plot_learning_curves_with_band(  — single-variant chart
        out_path, agg, title, ...)
    plot_overlay_with_bands(         — multi-variant overlay
        out_path, agg_by_variant, ...)
    run_multi_seed_learning_curves(  — orchestrator
        corpus_dir, out_base, grammar, corpus, ...)
    compute_hollow_metrics(out_dir)  — F1/EM/step-pick from CSVs
    run_multi_seed_hollow_learn(     — orchestrator
        corpus_dir, out_base, grammar, corpus, ...)
"""
import os, sys, csv, shutil
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unittests.hollow_learn_test_mh import run_hollow_learn
from unittests.learning_curves_test import run_learning_curves


# Canonical seed list for variance estimation in ACS-26.
# Five seeds is the sweet spot: enough to get a stable mean/std
# (standard error ≈ std/√5 ≈ std/2.2) without 10×-ing run time.
SEEDS = [13, 17, 23, 42, 100]


# ───────────────────────────── Plot palette ──────────────────────────
# All plotters in this module reference the BLUE_* constants. Two
# palettes are defined below; the ``ACTIVE_PALETTE_NAME`` switch at
# the bottom rebinds the BLUE_* names to whichever palette is active
# so plotters don't need to know about it. To re-theme the whole
# acs-26 figure set, just flip the switch and re-run regenerate_viz.

# Palette A — monochromatic blue (one hue, six shades).
# Shades from a perceptually-uniform blue ramp; ``BLUE_LIGHTEST`` is
# light enough to use as a fill / overlay without losing legibility
# against white grid lines.
_BLUE_DARKEST  = "#08306b"   # navy — primary lead metric
_BLUE_DARK     = "#2171b5"   # standard blue — secondary metric
_BLUE_MID      = "#4292c6"   # medium — tertiary metric
_BLUE_LIGHT    = "#6baed6"   # light — quaternary / overlay
_BLUE_LIGHTER  = "#9ecae1"   # very light — error / chance
_BLUE_LIGHTEST = "#deebf7"   # near-white — background fills
_BLUE_TRIAD    = [_BLUE_LIGHTER, _BLUE_DARK, _BLUE_DARKEST]

# Palette B — Red / Orange / Green (three distinct hues + light fills).
# Maps slots so panel metrics get DISTINGUISHABLE HUES (not just
# shades): primary metric = red, secondary = orange, tertiary = green.
# Sweep variants get a traffic-light triad (green=low/simple,
# orange=mid, red=high/complex).
_RED_DARK     = "#c0392b"   # primary lead metric
_ORANGE_DARK  = "#e67e22"   # secondary
_GREEN_DARK   = "#27ae60"   # tertiary
_RED_LIGHT    = "#f5b7b1"   # primary fill / error / chance
_ORANGE_LIGHT = "#fad7a0"   # secondary fill
_GREEN_LIGHT  = "#a9dfbf"   # tertiary fill
_RYG_TRIAD    = [_GREEN_DARK, _ORANGE_DARK, _RED_DARK]   # low → high

# ↓ flip to "blue" to revert ↓
ACTIVE_PALETTE_NAME = "ryg"

if ACTIVE_PALETTE_NAME == "blue":
    BLUE_DARKEST, BLUE_DARK, BLUE_MID    = _BLUE_DARKEST, _BLUE_DARK, _BLUE_MID
    BLUE_LIGHT, BLUE_LIGHTER, BLUE_LIGHTEST = _BLUE_LIGHT, _BLUE_LIGHTER, _BLUE_LIGHTEST
    PALETTE_TRIAD = _BLUE_TRIAD
elif ACTIVE_PALETTE_NAME == "ryg":
    # Slot mapping for RYG: lead=RED, support=ORANGE, third=GREEN.
    # The "LIGHT/LIGHTER/LIGHTEST" slots become the light tints of
    # red/orange/green so per-panel hierarchy still works visually.
    BLUE_DARKEST, BLUE_DARK, BLUE_MID    = _RED_DARK, _ORANGE_DARK, _GREEN_DARK
    BLUE_LIGHT, BLUE_LIGHTER, BLUE_LIGHTEST = _RED_LIGHT, _ORANGE_LIGHT, _GREEN_LIGHT
    PALETTE_TRIAD = _RYG_TRIAD
else:
    raise ValueError(f"Unknown ACTIVE_PALETTE_NAME={ACTIVE_PALETTE_NAME!r}")

PALETTE_BLUE = [BLUE_DARKEST, BLUE_DARK, BLUE_MID, BLUE_LIGHT,
                BLUE_LIGHTER, BLUE_LIGHTEST]


# ───────────────────────────── CSV helpers ────────────────────────────

def load_learning_curve_csv(csv_path):
    """Read a learning_curves.csv into parallel arrays. Returns a tuple
    ``(xs, F1, P, R, EM, gen_gram, gen_novel, gen_gram_novel,
        p_parse_legal, prim_active)``.
    Older CSVs without the new columns get graceful fallbacks
    (P=R=F1, p_parse_legal=0, prim_active=1)."""
    xs = []; F1 = []; P = []; R = []; EM = []
    gen_gram = []; gen_novel = []; gen_gram_novel = []
    p_parse_legal = []; prim_active = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            xs.append(int(row["n_trained"]))
            f1 = float(row["parse_F1"])
            F1.append(f1)
            P.append(float(row.get("parse_P", f1)))
            R.append(float(row.get("parse_R", f1)))
            EM.append(float(row["parse_EM"]))
            gen_gram.append(float(row["gen_gram"]))
            gn = float(row["gen_novel"])
            gen_novel.append(gn)
            gen_gram_novel.append(float(row.get("gen_gram_novel", gn)))
            p_parse_legal.append(float(row.get("p_parse_legal", 0.0)))
            prim_active.append(float(row.get("prim_active_frac", 1.0)))
    return (xs, F1, P, R, EM,
            gen_gram, gen_novel, gen_gram_novel,
            p_parse_legal, prim_active)


def aggregate_seeds(seed_curves):
    """Given a list of tuples from ``load_learning_curve_csv``, return
    a dict of (mean, std) per metric. Trims to the shortest curve."""
    xs = seed_curves[0][0]
    n = min(len(sc[1]) for sc in seed_curves)
    out = {"xs": xs[:n]}
    names = ["F1", "P", "R", "EM",
             "gen_gram", "gen_novel", "gen_gram_novel",
             "p_parse_legal", "prim_active"]
    for i, name in enumerate(names, start=1):
        arr = np.array([sc[i][:n] for sc in seed_curves])
        out[name] = (arr.mean(axis=0), arr.std(axis=0))
    return out


def write_aggregated_csv(path, agg):
    """Write aggregated learning-curve mean/std to a CSV."""
    cols = ["F1", "P", "R", "EM",
            "gen_gram", "gen_novel", "gen_gram_novel",
            "p_parse_legal", "prim_active"]
    with open(path, "w") as f:
        header = ["n_trained"]
        for k in cols:
            header += [f"{k}_mean", f"{k}_std"]
        f.write(",".join(header) + "\n")
        for i, x in enumerate(agg["xs"]):
            row = [str(x)]
            for k in cols:
                m, s = agg[k]
                row += [f"{m[i]:.4f}", f"{s[i]:.4f}"]
            f.write(",".join(row) + "\n")


# ───────────────────── Multi-seed learning curves ─────────────────────

def run_multi_seed_learning_curves(corpus_dir, out_base, grammar, corpus,
                                    seeds=SEEDS, eval_every=10,
                                    n_gen_per_eval=50,
                                    primitives_first=None,
                                    maturity_gate: tuple = None,
                                    gate_mode:    str   = "skip"):
    """Orchestrate: run ``run_learning_curves`` once per seed into
    ``out_base/seed_{N}/``, then aggregate and write the unified
    ``out_base/aggregated.csv``. Returns the aggregated dict."""
    if os.path.exists(out_base):
        shutil.rmtree(out_base)
    os.makedirs(out_base, exist_ok=True)

    seed_curves = []
    for sd in seeds:
        seed_dir = os.path.join(out_base, f"seed_{sd}")
        print(f"\n--- learning-curves seed={sd} ---")
        run_learning_curves(
            corpus_dir=corpus_dir,
            out_dir=seed_dir,
            grammar=grammar,
            corpus=corpus,
            seed=sd,
            eval_every=eval_every,
            n_gen_per_eval=n_gen_per_eval,
            primitives_first=primitives_first,
            maturity_gate=maturity_gate,
            gate_mode=gate_mode,
        )
        seed_curves.append(load_learning_curve_csv(
            os.path.join(seed_dir, "learning_curves.csv")))

    agg = aggregate_seeds(seed_curves)
    agg_path = os.path.join(out_base, "aggregated.csv")
    write_aggregated_csv(agg_path, agg)
    print(f"\n  Aggregated CSV → {agg_path}")
    return agg


def plot_learning_curves_with_band(out_path, agg, title, n_seeds=None):
    """Single-variant 4-panel chart with ±1σ bands across seeds:
        A: Parse Precision + Recall + Exact-match  (P/R split, not F1)
        B: Generation Grammaticality + In-lexicon
        C: Generation Novelty (grammatical-only)
        D: Gate activation (% sentences with mature primitives)
    Also writes a separate ``grids_curves.png`` next to ``out_path`` with
    the GRIDS-style omission / co-mission plots."""
    xs = agg["xs"]
    suffix = f"  (mean ± std across {n_seeds} seeds)" if n_seeds else ""

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    panel_specs = [
        ("Parse accuracy — precision / recall (split)",
            [("P",  "Precision",   BLUE_DARKEST),
             ("R",  "Recall",      BLUE_DARK),
             ("EM", "Exact-match", BLUE_MID)]),
        ("Generation grammaticality",
            [("gen_gram", "Grammatical", BLUE_DARK)]),
        ("Generation novelty (grammatical-only)",
            [("gen_gram_novel", "Grammatical & Novel", BLUE_DARK)]),
        ("Gate activation",
            [("prim_active", "% train sents w/ mature primitives", BLUE_DARK)]),
    ]
    for ax, (sub_title, keys) in zip(axes, panel_specs):
        for k, lbl, c in keys:
            mean, std = agg[k]
            ax.plot(xs, 100*mean, "o-", color=c, linewidth=2, label=lbl)
            ax.fill_between(xs, 100*(mean - std), 100*(mean + std),
                            color=c, alpha=0.2)
        ax.set_xlabel("# training sentences"); ax.set_ylabel("%")
        ax.set_title(sub_title); ax.set_ylim(0, 105)
        ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(title + suffix, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Chart → {out_path}")

    # GRIDS-style omission/co-mission alongside the main chart.
    grids_path = os.path.join(os.path.dirname(out_path), "grids_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pp_m, pp_s = agg["p_parse_legal"]
    axes[0].plot(xs, pp_m, "o-", color=BLUE_DARKEST, linewidth=2)
    axes[0].fill_between(xs, pp_m - pp_s, pp_m + pp_s, color=BLUE_DARKEST, alpha=0.2)
    axes[0].set_xlabel("# training sentences")
    axes[0].set_ylabel("Probability of parsing a legal sentence")
    axes[0].set_title("(a) Parsing — errors of omission")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    gg_m, gg_s = agg["gen_gram"]
    axes[1].plot(xs, gg_m, "o-", color=BLUE_DARK, linewidth=2)
    axes[1].fill_between(xs, gg_m - gg_s, gg_m + gg_s, color=BLUE_DARK, alpha=0.2)
    axes[1].set_xlabel("# training sentences")
    axes[1].set_ylabel("Probability of generating a legal sentence")
    axes[1].set_title("(b) Generation — errors of co-mission")
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    fig.suptitle(f"GRIDS-style omission / co-mission — {title}{suffix}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(grids_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  GRIDS → {grids_path}")


def plot_overlay_with_bands(out_path, agg_by_variant, colors, labels,
                              suptitle, n_seeds=None):
    """Multi-variant overlay (4-panel): parse Precision, parse Recall,
    gen grammaticality, gen novelty (grammatical). One line per variant
    with ±1σ band. Also writes a GRIDS-style overlay next to ``out_path``."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    panel_specs = [
        ("Parse Precision (held-out test)",                "P",              "% bracket precision"),
        ("Parse Recall (held-out test)",                   "R",              "% bracket recall"),
        ("Generation grammaticality",                      "gen_gram",       "% of outputs"),
        ("Generation novelty (grammatical-only, not in train)",
                                                           "gen_gram_novel", "% of outputs"),
    ]
    for ax, (title, key, ylabel) in zip(axes, panel_specs):
        for v, agg in agg_by_variant.items():
            xs = agg["xs"]
            mean, std = agg[key]
            mean_pct = 100 * mean
            std_pct  = 100 * std
            c = colors.get(v, "#666666")
            ax.plot(xs, mean_pct, "o-", color=c, linewidth=2,
                    label=labels.get(v, v))
            ax.fill_between(xs, mean_pct - std_pct, mean_pct + std_pct,
                            color=c, alpha=0.18)
        ax.set_xlabel("# training sentences")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    suffix = f"  (mean ± std across {n_seeds} seeds)" if n_seeds else ""
    fig.suptitle(suptitle + suffix, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Overlay → {out_path}")

    # GRIDS-style overlay: P(parse legal) + P(generate legal), variants overlaid
    grids_path = os.path.join(os.path.dirname(out_path), "grids_overlay.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for v, agg in agg_by_variant.items():
        xs = agg["xs"]
        c = colors.get(v, "#666666"); lbl = labels.get(v, v)
        pp_m, pp_s = agg["p_parse_legal"]
        axes[0].plot(xs, pp_m, "o-", color=c, linewidth=2, label=lbl)
        axes[0].fill_between(xs, pp_m - pp_s, pp_m + pp_s, color=c, alpha=0.18)
        gg_m, gg_s = agg["gen_gram"]
        axes[1].plot(xs, gg_m, "o-", color=c, linewidth=2, label=lbl)
        axes[1].fill_between(xs, gg_m - gg_s, gg_m + gg_s, color=c, alpha=0.18)
    axes[0].set_xlabel("# training sentences")
    axes[0].set_ylabel("Probability of parsing a legal sentence")
    axes[0].set_title("(a) Parsing — errors of omission")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set_xlabel("# training sentences")
    axes[1].set_ylabel("Probability of generating a legal sentence")
    axes[1].set_title("(b) Generation — errors of co-mission")
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("GRIDS-style omission / co-mission overlay" + suffix,
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(grids_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  GRIDS overlay → {grids_path}")


# ─────────────────────── Multi-seed hollow_learn ──────────────────────

def compute_hollow_metrics(test_dir):
    """Read parse_accuracy.csv + step_pick_accuracy.csv from a
    hollow_learn output directory and return (F1, EM, step_pick) as
    fractions (0..1)."""
    pa = os.path.join(test_dir, "parse_accuracy.csv")
    sp = os.path.join(test_dir, "step_pick_accuracy.csv")
    tp = fp = fn = em = n = 0
    with open(pa) as f:
        r = csv.reader(f); next(r)
        for row in r:
            t, p, e = int(row[-3]), int(row[-2]), int(row[-1])
            tp += t; fp += p; fn += e
            if p == 0 and e == 0 and t > 0: em += 1
            n += 1
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-9)
    EM = em / max(n, 1)
    sp_ok = sp_n = 0
    with open(sp) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("is_gold") == "1":
                sp_ok += 1
            sp_n += 1
    step_pick = sp_ok / max(sp_n, 1)
    return F, EM, step_pick


def run_multi_seed_hollow_learn(corpus_dir, out_base, grammar, corpus,
                                  seeds=SEEDS, primitives_first=200,
                                  viz_dir=None,
                                  maturity_gate: tuple = None,
                                  gate_mode: str = "skip",
                                  viz_intermediates: bool = False):
    """Orchestrate: run ``run_hollow_learn`` once per seed into
    ``out_base/seed_{N}/``, then aggregate F1/EM/step-pick into
    ``out_base/hollow_learn_summary.csv`` and a printable table.

    After per-seed runs, also builds two aggregate visuals at
    ``out_base/``:
      * ``step_pick_histograms.png`` — gold vs non-gold heuristic
        histograms with raw candidates POOLED across all seeds.
      * ``performance_summary.png`` — six-panel scorecard with each
        bar showing mean ± 1 std across seeds (error bars).

    Returns a dict ``{F1_mean, F1_std, EM_mean, EM_std,
                        step_pick_mean, step_pick_std, per_seed}``."""
    if os.path.exists(out_base):
        shutil.rmtree(out_base)
    os.makedirs(out_base, exist_ok=True)

    per_seed = []
    for sd in seeds:
        seed_dir = os.path.join(out_base, f"seed_{sd}")
        print(f"\n=== hollow_learn seed={sd} ===")
        run_hollow_learn(
            corpus_dir=corpus_dir,
            out_dir=seed_dir,
            grammar=grammar,
            corpus=corpus,
            seed=sd,
            primitives_first=primitives_first,
            maturity_gate=maturity_gate,
            gate_mode=gate_mode,
            viz_intermediates=viz_intermediates,
        )
        f1, em, sp = compute_hollow_metrics(seed_dir)
        per_seed.append({"seed": sd, "F1": f1, "EM": em, "step_pick": sp})

    f1_arr = np.array([p["F1"] for p in per_seed])
    em_arr = np.array([p["EM"] for p in per_seed])
    sp_arr = np.array([p["step_pick"] for p in per_seed])
    agg = {
        "per_seed":       per_seed,
        "F1_mean":        float(f1_arr.mean()),
        "F1_std":         float(f1_arr.std()),
        "EM_mean":        float(em_arr.mean()),
        "EM_std":         float(em_arr.std()),
        "step_pick_mean": float(sp_arr.mean()),
        "step_pick_std":  float(sp_arr.std()),
    }
    # Write summary CSV.
    out_csv = os.path.join(out_base, "hollow_learn_summary.csv")
    with open(out_csv, "w") as f:
        f.write("seed,F1,EM,step_pick\n")
        for p in per_seed:
            f.write(f"{p['seed']},{p['F1']:.4f},{p['EM']:.4f},"
                    f"{p['step_pick']:.4f}\n")
        f.write(f"MEAN,{agg['F1_mean']:.4f},{agg['EM_mean']:.4f},"
                f"{agg['step_pick_mean']:.4f}\n")
        f.write(f"STD,{agg['F1_std']:.4f},{agg['EM_std']:.4f},"
                f"{agg['step_pick_std']:.4f}\n")
    print(f"\n  Hollow-learn summary CSV → {out_csv}")
    print(f"  F1        : {100*agg['F1_mean']:.1f} ± {100*agg['F1_std']:.1f}%")
    print(f"  EM        : {100*agg['EM_mean']:.1f} ± {100*agg['EM_std']:.1f}%")
    print(f"  Step-pick : {100*agg['step_pick_mean']:.1f} ± {100*agg['step_pick_std']:.1f}%")

    # Build aggregate visuals. Default location is out_base; pass
    # viz_dir to put them somewhere else (e.g. the grammar dir's top
    # level so they sit alongside hollow_learn/ + learning_curves/).
    target = viz_dir if viz_dir is not None else out_base
    print(f"\n=== Building aggregate visuals (across {len(seeds)} seeds) → {target} ===")
    build_aggregate_hollow_visuals(out_base, seeds=seeds, viz_dir=target)
    return agg


# ───────────────────── Aggregate hollow_learn visuals ─────────────────

_HIST_HEURS = ["cnt_root_lp", "sum_root_lp", "sum_leaf_lp",
               "sum_bl_lp", "cur_score",
               "chunk_pool_match", "L_trans", "R_trans"]


def _pool_cand_heur_logs(out_base, seeds):
    """Read per-seed cand_heur_log.csv files into a single dict mapping
    heuristic → (gold_values, nongold_values). Skips seeds whose csv
    is missing (older runs that pre-date the dump)."""
    gold = {h: [] for h in _HIST_HEURS}
    nong = {h: [] for h in _HIST_HEURS}
    n_seeds_used = 0
    for sd in seeds:
        path = os.path.join(out_base, f"seed_{sd}", "cand_heur_log.csv")
        if not os.path.exists(path):
            continue
        n_seeds_used += 1
        with open(path) as f:
            for row in csv.DictReader(f):
                bucket = gold if int(row["is_gold"]) else nong
                for h in _HIST_HEURS:
                    try:
                        bucket[h].append(float(row[h]))
                    except (ValueError, KeyError):
                        pass
    return gold, nong, n_seeds_used


def _plot_pooled_histograms(gold, nong, out_path, n_seeds):
    """Pooled gold-vs-non-gold heuristic histograms across all seeds."""
    n_h = len(_HIST_HEURS)
    n_cols = 4
    n_rows = (n_h + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i, h in enumerate(_HIST_HEURS):
        ax = axes[i]
        g = np.array([v for v in gold[h] if np.isfinite(v)])
        n = np.array([v for v in nong[h] if np.isfinite(v)])
        if len(g) == 0 and len(n) == 0:
            ax.set_title(f"{h}\n(no data)")
            continue
        all_v = np.concatenate([g, n]) if len(n) > 0 else g
        lo, hi = float(np.min(all_v)), float(np.max(all_v))
        if lo == hi:
            lo -= 0.5; hi += 0.5
        bins = np.linspace(lo, hi, 30)
        ax.hist(n, bins=bins, alpha=0.5, color=BLUE_LIGHTER,
                label=f"non-gold (n={len(n)})")
        ax.hist(g, bins=bins, alpha=0.7, color=BLUE_DARKEST,
                label=f"gold (n={len(g)})")
        if len(g) > 1 and len(n) > 1:
            pooled_std = float(np.sqrt(
                (np.var(g, ddof=1) + np.var(n, ddof=1)) / 2))
            d = ((float(np.mean(g)) - float(np.mean(n)))
                 / max(pooled_std, 1e-9))
            ax.set_title(f"{h}  (d={d:+.2f})", fontsize=10)
        else:
            ax.set_title(h, fontsize=10)
        ax.set_ylabel("# candidates", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.3)
    for j in range(n_h, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(
        f"Parse-time heuristic distributions — gold vs non-gold candidates  "
        f"(pooled across {n_seeds} seeds; d = Cohen's effect size)",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Pooled histograms → {out_path}")


def _seed_summary_metrics(seed_dir):
    """Extract scalar metrics from one seed's hollow_learn output dir.

    Returns dict with: precision, recall, F1, EM, step_pick,
    chunk_acc, in_lex, gram, novel, unique, gram_novel,
    mask_exact, mask_pos, chunk_per_class (list of (cls, P, R, F1, n)).
    """
    out = {}

    # parse_accuracy.csv → P/R/F1/EM
    tp = fp = fn = em = n = 0
    pa = os.path.join(seed_dir, "parse_accuracy.csv")
    with open(pa) as f:
        r = csv.reader(f); next(r)
        for row in r:
            t, p_, e = int(row[-3]), int(row[-2]), int(row[-1])
            tp += t; fp += p_; fn += e
            if p_ == 0 and e == 0 and t > 0: em += 1
            n += 1
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-9)
    out["precision"], out["recall"], out["F1"], out["EM"] = P, R, F, em / max(n, 1)
    out["n_parse"] = n

    # step_pick_accuracy.csv → step_pick + gate_pass_rate
    sp_ok = sp_gated = sp_total = 0
    avg_pairs = []
    sp = os.path.join(seed_dir, "step_pick_accuracy.csv")
    with open(sp) as f:
        for row in csv.DictReader(f):
            sp_total += 1
            if int(row.get("n_admitted", 0)) == 0:
                sp_gated += 1
            if row.get("is_gold") == "1":
                sp_ok += 1
            try:
                avg_pairs.append(int(row["n_pairs"]))
            except (KeyError, ValueError):
                pass
    out["step_pick"]      = sp_ok / max(sp_total, 1)
    out["gate_pass"]      = (sp_total - sp_gated) / max(sp_total, 1)
    out["step_chance"]    = 1 / max(np.mean(avg_pairs) if avg_pairs else 1, 1)
    out["n_steps"]        = sp_total

    # chunk_class_accuracy.csv → per-class + overall acc
    per_class = []
    cc_acc = 0.0
    cc_n   = 0
    cc = os.path.join(seed_dir, "chunk_class_accuracy.csv")
    if os.path.exists(cc):
        with open(cc) as f:
            for row in csv.DictReader(f):
                if row["class"] == "OVERALL":
                    continue
                n_g = int(row["n_gold"]); tp_ = int(row["tp"])
                def _f(s, d=0.0):
                    try: return float(s)
                    except (ValueError, TypeError): return d
                per_class.append((row["class"], _f(row["precision"]),
                                  _f(row["recall"]), _f(row["f1"]), n_g))
                cc_acc += tp_; cc_n += n_g
    out["chunk_acc"]     = cc_acc / max(cc_n, 1)
    out["chunk_n"]       = cc_n
    out["chunk_per_class"] = per_class

    # generation_from_scratch.csv → in_lex / gram / novel / unique / gram_novel
    gen_path = os.path.join(seed_dir, "generation_from_scratch.csv")
    n_gen = n_lex = n_gram = n_nov = n_gn = 0
    texts = []
    with open(gen_path) as f:
        for row in csv.DictReader(f):
            n_gen += 1
            il = int(row["in_lexicon"]); gr = int(row["grammatical"])
            nv = int(row["novel"])
            n_lex += il; n_gram += gr; n_nov += nv
            if gr and nv: n_gn += 1
            texts.append(row["text"])
    out["in_lex"]      = n_lex / max(n_gen, 1)
    out["gram"]        = n_gram / max(n_gen, 1)
    out["novel"]       = n_nov / max(n_gen, 1)
    out["unique"]      = len(set(texts)) / max(n_gen, 1)
    out["gram_novel"]  = n_gn / max(n_gen, 1)
    out["n_gen"]       = n_gen

    # generation_masked.csv → exact / pos_ok
    mk = os.path.join(seed_dir, "generation_masked.csv")
    m_total = m_exact = m_pos = 0
    with open(mk) as f:
        for row in csv.DictReader(f):
            m_total += 1
            m_exact += int(row["exact"])
            m_pos   += int(row["pos_ok"])
    out["mask_exact"] = m_exact / max(m_total, 1)
    out["mask_pos"]   = m_pos   / max(m_total, 1)
    out["n_mask"]     = m_total

    return out


def _grouped_bar(ax, labels, means, stds, colors, ymax=1.15):
    """Bar chart with ±1 std error bars, percent labels above each bar."""
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5,
                   yerr=stds, capsize=4, ecolor="#333")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.025,
                f"{100*m:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3)


def _plot_aggregate_summary(per_seed_metrics, out_path, n_seeds, title_suffix=""):
    """Aggregate six-panel scorecard with ±1 std error bars on every bar."""
    def ms(key):
        arr = np.array([m[key] for m in per_seed_metrics])
        return float(arr.mean()), float(arr.std())

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "TRELLIS — Hollow Learning Test Performance Summary  "
        f"(mean ± std across {n_seeds} seeds){title_suffix}",
        fontsize=12, fontweight="bold")

    # Panel A — Parse P/R/F1/EM.
    axA = fig.add_subplot(2, 3, 1)
    pa_keys = [("precision", "Precision", BLUE_DARKEST),
               ("recall",    "Recall",    BLUE_DARK),
               ("F1",        "F1",        BLUE_MID),
               ("EM",        "Exact-match", BLUE_LIGHT)]
    means = [ms(k)[0] for k, _, _ in pa_keys]
    stds  = [ms(k)[1] for k, _, _ in pa_keys]
    _grouped_bar(axA, [l for _, l, _ in pa_keys], means, stds,
                 [c for _, _, c in pa_keys])
    axA.set_ylabel("Score")
    n_parse_mean = int(np.mean([m["n_parse"] for m in per_seed_metrics]))
    axA.set_title(f"(A) End-to-end Parse Accuracy  (n≈{n_parse_mean})", fontsize=11)

    # Panel B — Step-pick + gate pass rate + chance.
    axB = fig.add_subplot(2, 3, 2)
    sp_keys = [("step_pick", "Step-pick\n(of admitted)", BLUE_DARKEST),
               ("gate_pass", "Gate pass-rate\n(climb cleared)", BLUE_DARK),
               ("step_chance", "chance",                  BLUE_LIGHTER)]
    means = [ms(k)[0] for k, _, _ in sp_keys]
    stds  = [ms(k)[1] for k, _, _ in sp_keys]
    _grouped_bar(axB, [l for _, l, _ in sp_keys], means, stds,
                 [c for _, _, c in sp_keys])
    axB.set_ylabel("Accuracy")
    n_steps_mean = int(np.mean([m["n_steps"] for m in per_seed_metrics]))
    axB.set_title(f"(B) Step-pick — Climbing-Ancestor  (n≈{n_steps_mean} steps)",
                   fontsize=11)
    axB.tick_params(axis="x", labelsize=8)

    # Panel C — Per-chunk-class P / R / F1 (mean ± std over seeds that
    # contain each class).
    axC = fig.add_subplot(2, 3, 3)
    class_data = {}  # name → list of (P,R,F1,n) across seeds
    for m in per_seed_metrics:
        for cls, P, R, F, n_g in m["chunk_per_class"]:
            class_data.setdefault(cls, []).append((P, R, F, n_g))
    if class_data:
        # Stable order: by mean n_gold (desc).
        ordering = sorted(class_data,
                          key=lambda c: -np.mean([d[3] for d in class_data[c]]))
        x = np.arange(len(ordering)); w = 0.27
        Pm, Ps = [], []; Rm, Rs = [], []; Fm, Fs = [], []; n_mean = []
        for cls in ordering:
            arr = np.array(class_data[cls])
            Pm.append(arr[:, 0].mean()); Ps.append(arr[:, 0].std())
            Rm.append(arr[:, 1].mean()); Rs.append(arr[:, 1].std())
            Fm.append(arr[:, 2].mean()); Fs.append(arr[:, 2].std())
            n_mean.append(int(arr[:, 3].mean()))
        axC.bar(x - w, Pm, w, yerr=Ps, capsize=3, ecolor="#333",
                label="Precision", color=BLUE_DARKEST)
        axC.bar(x,     Rm, w, yerr=Rs, capsize=3, ecolor="#333",
                label="Recall",    color=BLUE_DARK)
        axC.bar(x + w, Fm, w, yerr=Fs, capsize=3, ecolor="#333",
                label="F1",        color=BLUE_MID)
        axC.set_xticks(x)
        axC.set_xticklabels([f"{c}\n(n̄={n})" for c, n in zip(ordering, n_mean)],
                            fontsize=9)
    axC.set_ylim(0, 1.15)
    axC.set_ylabel("Score")
    cc_m, cc_s = ms("chunk_acc")
    axC.set_title(f"(C) Chunk-Class Probe   overall {100*cc_m:.1f} ± {100*cc_s:.1f}%",
                   fontsize=11)
    axC.legend(loc="lower right", fontsize=8)
    axC.grid(axis="y", alpha=0.3)

    # Panel D — From-scratch generation rates.
    axD = fig.add_subplot(2, 3, 4)
    gen_keys = [("in_lex",     "In-lexicon",   BLUE_DARKEST),
                ("gram",       "Grammatical",  BLUE_DARK),
                ("novel",      "Novel",        BLUE_MID),
                ("unique",     "Unique",       BLUE_LIGHT),
                ("gram_novel", "Gram & Novel", BLUE_LIGHTER)]
    means = [ms(k)[0] for k, _, _ in gen_keys]
    stds  = [ms(k)[1] for k, _, _ in gen_keys]
    _grouped_bar(axD, [l for _, l, _ in gen_keys], means, stds,
                 [c for _, _, c in gen_keys])
    axD.set_ylabel("Rate")
    n_gen_mean = int(np.mean([m["n_gen"] for m in per_seed_metrics]))
    axD.set_title(f"(D) From-scratch Generation  (n≈{n_gen_mean})", fontsize=11)
    axD.tick_params(axis="x", labelsize=8, rotation=15)

    # Panel E — Mask completion exact / POS.
    axE = fig.add_subplot(2, 3, 5)
    mk_keys = [("mask_exact", "Exact-token", BLUE_DARKEST),
               ("mask_pos",   "POS-class",   BLUE_DARK)]
    means = [ms(k)[0] for k, _, _ in mk_keys]
    stds  = [ms(k)[1] for k, _, _ in mk_keys]
    _grouped_bar(axE, [l for _, l, _ in mk_keys], means, stds,
                 [c for _, _, c in mk_keys])
    axE.set_ylabel("Recovery rate")
    n_mask_mean = int(np.mean([m["n_mask"] for m in per_seed_metrics]))
    axE.set_title(f"(E) Single-token Masked Completion  (n≈{n_mask_mean})",
                   fontsize=11)

    # Panel F — Scorecard.
    axF = fig.add_subplot(2, 3, 6)
    axF.axis("off")
    rows = [
        ("Parse F1",             *ms("F1")),
        ("Step-pick (climbing)", *ms("step_pick")),
        ("Chunk-class acc",      *ms("chunk_acc")),
        ("Gen grammatical",      *ms("gram")),
        ("Gen novelty",          *ms("novel")),
        ("Gen gram&novel",       *ms("gram_novel")),
        ("Mask POS recovery",    *ms("mask_pos")),
    ]
    axF.set_title("(F) Scorecard  (mean ± std)", fontsize=11)
    y_top = 0.95
    for i, (label, m, s) in enumerate(rows):
        y = y_top - i * 0.13
        axF.text(0.05, y, label, fontsize=12, va="center",
                 transform=axF.transAxes)
        axF.text(0.95, y, f"{100*m:.1f} ± {100*s:.1f}%",
                 fontsize=13, va="center", ha="right",
                 fontweight="bold", color=BLUE_DARKEST,
                 transform=axF.transAxes)
        axF.plot([0.04, 0.96], [y - 0.065, y - 0.065],
                 color="#ddd", linewidth=0.6, transform=axF.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Aggregate summary → {out_path}")


def build_aggregate_hollow_visuals(out_base, seeds=SEEDS, viz_dir=None):
    """Build both aggregate visuals derived from per-seed dirs under
    ``out_base/seed_{N}/``:
      * ``step_pick_histograms.png`` (pooled candidates)
      * ``performance_summary.png``  (mean ± std bars)
    Writes them to ``viz_dir`` (defaults to ``out_base``)."""
    target = viz_dir if viz_dir is not None else out_base
    os.makedirs(target, exist_ok=True)
    seed_dirs_present = [sd for sd in seeds
                         if os.path.isdir(os.path.join(out_base, f"seed_{sd}"))]
    if not seed_dirs_present:
        print(f"  (no seed dirs found under {out_base}, skipping)")
        return

    # 1. Pooled histograms (skipped silently if cand_heur_log.csv missing).
    gold, nong, n_used = _pool_cand_heur_logs(out_base, seed_dirs_present)
    if n_used > 0:
        _plot_pooled_histograms(
            gold, nong,
            os.path.join(target, "step_pick_histograms.png"),
            n_seeds=n_used)
    else:
        print(f"  (no cand_heur_log.csv files found, skipping pooled histograms)")

    # 2. Aggregate performance summary.
    per_seed_metrics = []
    for sd in seed_dirs_present:
        try:
            per_seed_metrics.append(_seed_summary_metrics(
                os.path.join(out_base, f"seed_{sd}")))
        except FileNotFoundError as e:
            print(f"  (skipping seed_{sd}: {e})")
    if per_seed_metrics:
        _plot_aggregate_summary(
            per_seed_metrics,
            os.path.join(target, "performance_summary.png"),
            n_seeds=len(per_seed_metrics))
