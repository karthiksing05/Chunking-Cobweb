"""
regenerate_viz.py — re-render all acs-26 visualisations from saved CSVs.

Walks every experiment directory under ``confs/acs-26/`` and regenerates:

  * **Per-grammar tests** (grammar_small / grammar_med / grammar_large):
      hollow_learn/  →  performance_summary.png  +  step_pick_histograms.png
                        (pooled across 5 seeds, written to top-level)
      learning_curves/seed_*/  →  per-seed learning_curves.png + grids_curves.png
      learning_curves/        →  aggregated.csv  +  learning_curves.png
                                 +  grids_curves.png  (mean ± 1σ bands)

  * **Experiment tests** (grammar_experiment / terminal_experiment / terminal_experiment_long):
      <variant>/seed_*/  →  per-seed learning_curves.png + grids_curves.png
      <variant>/         →  aggregated.csv + learning_curves.png + grids_curves.png
      <top-level>/       →  comparison.png  +  grids_overlay.png
                            (multi-variant overlay with ±1σ bands)

The point: **no model training happens**. All data is read from per-seed
CSVs (and per-seed cand_heur_log files for the hollow_learn histograms),
so this is fast and safe to run repeatedly. Tweak the plot code in
``experiment_harness.py`` (or inline in this file) to apply new paper-style
themes — colour palette, panel layout, axis ranges, etc. — without
re-running the heavy training pipeline.

Usage:
    python confs/acs-26/regenerate_viz.py                 # all experiments
    python confs/acs-26/regenerate_viz.py grammar_large   # one experiment
    python confs/acs-26/regenerate_viz.py grammar_small grammar_experiment
"""
import os, sys, glob, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_harness import (
    SEEDS,
    load_learning_curve_csv, aggregate_seeds, write_aggregated_csv,
    plot_learning_curves_with_band, plot_overlay_with_bands,
    build_aggregate_hollow_visuals,
    # Monochromatic blue palette — shared across this script and
    # the aggregate plotters in experiment_harness.py. Re-theme by editing
    # the palette block in experiment_harness.py.
    BLUE_DARKEST, BLUE_DARK, BLUE_MID, BLUE_LIGHT,
    BLUE_LIGHTER, BLUE_LIGHTEST, PALETTE_TRIAD,
)

# ───────────────────────────── Targets ───────────────────────────────

# Grammar tests: hollow_learn aggregate + learning_curves aggregate.
GRAMMAR_TESTS = ["grammar_small", "grammar_med", "grammar_large"]

# Experiment tests: per-variant learning_curves aggregate + multi-variant overlay.
# Variant colors use the monochromatic blue triad (PALETTE_TRIAD)
# so "increasing complexity / lexicon size" reads as "deeper blue".
_TRIAD3 = {0: PALETTE_TRIAD[0], 1: PALETTE_TRIAD[1], 2: PALETTE_TRIAD[2]}

EXPERIMENT_TESTS = {
    "grammar_experiment": {
        "variants": ["small", "med", "large"],
        "colors":   {"small": _TRIAD3[0], "med": _TRIAD3[1], "large": _TRIAD3[2]},
        "labels":   {"small": "small (S→NP VP; VP=V (NP))",
                     "med":   "med (+AdjP, PP, V NP PP)",
                     "large": "large (+RelClause)"},
        "title":    "TEST_GRAMMAR experiment — learning curves vs grammar complexity",
    },
    "terminal_experiment": {
        "variants": ["low", "med", "high"],
        "colors":   {"low": _TRIAD3[0], "med": _TRIAD3[1], "high": _TRIAD3[2]},
        "labels":   {"low":  "low  (11 terminals)",
                     "med":  "med  (22 terminals)",
                     "high": "high (39 terminals)"},
        "title":    "TEST_GRAMMAR_MED — learning curves vs lexicon size (N=200)",
    },
    "terminal_experiment_long": {
        "variants": ["low", "med", "high"],
        "colors":   {"low": _TRIAD3[0], "med": _TRIAD3[1], "high": _TRIAD3[2]},
        "labels":   {"low":  "low  (11 terminals)",
                     "med":  "med  (22 terminals)",
                     "high": "high (39 terminals)"},
        "title":    "TEST_GRAMMAR_MED — learning curves vs lexicon size (N=500)",
    },
}

# ───────────────────────────── Per-seed plotter ──────────────────────

def _plot_single_seed_learning_curves(csv_path, out_dir, label):
    """Render per-seed learning_curves.png + grids_curves.png from a
    single learning_curves.csv. Mirrors the inline plotting at the end
    of ``unittests/learning_curves_test.py::run_learning_curves``."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) if k != "n_trained" else int(v)
                          for k, v in r.items()})
    if not rows:
        return
    xs = [r["n_trained"] for r in rows]

    # 4-panel chart — same layout as the live test, monochromatic blue palette.
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes[0].plot(xs, [100*r["parse_P"]  for r in rows], "o-", color=BLUE_DARKEST, label="Precision")
    axes[0].plot(xs, [100*r["parse_R"]  for r in rows], "s-", color=BLUE_DARK,    label="Recall")
    axes[0].plot(xs, [100*r["parse_EM"] for r in rows], "d-", color=BLUE_MID,     label="Exact-match")
    axes[0].plot(xs, [100*r.get("prim_active_frac", 1.0) for r in rows],
                  "^--", color=BLUE_LIGHTER, alpha=0.8, label="gate active")
    axes[0].set_xlabel("# training sentences"); axes[0].set_ylabel("%")
    axes[0].set_title("Parse accuracy — precision / recall (split)")
    axes[0].set_ylim(0, 105); axes[0].grid(alpha=0.3); axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(xs, [100*r["gen_gram"] for r in rows], "o-", color=BLUE_DARKEST, label="Grammatical")
    axes[1].plot(xs, [100*r["gen_lex"]  for r in rows], "s-", color=BLUE_MID,     label="In-lexicon")
    axes[1].set_xlabel("# training sentences"); axes[1].set_ylabel("%")
    axes[1].set_title("Generation grammaticality")
    axes[1].set_ylim(0, 105); axes[1].grid(alpha=0.3); axes[1].legend(loc="lower right", fontsize=9)

    axes[2].plot(xs, [100*r.get("gen_gram_novel", 0.0) for r in rows],
                  "o-", color=BLUE_DARK, label="Grammatical & novel")
    axes[2].set_xlabel("# training sentences"); axes[2].set_ylabel("% of generated outputs")
    axes[2].set_title("Generation novelty (grammatical-only)")
    axes[2].set_ylim(0, 105); axes[2].grid(alpha=0.3); axes[2].legend()

    axes[3].plot(xs, [100*r.get("prim_active_frac", 1.0) for r in rows],
                  "^-", color=BLUE_DARK, label="% train sents w/ mature primitives")
    axes[3].set_xlabel("# training sentences"); axes[3].set_ylabel("%")
    axes[3].set_title("Gate activation")
    axes[3].set_ylim(0, 105); axes[3].grid(alpha=0.3); axes[3].legend(loc="lower right", fontsize=9)

    fig.suptitle(f"Learning curves — {label}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out_dir, "learning_curves.png"),
                 dpi=140, bbox_inches="tight")
    plt.close()

    # GRIDS-style chart (single seed) — monochromatic blue.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(xs, [r.get("p_parse_legal", 0.0) for r in rows],
                  "o-", color=BLUE_DARKEST, linewidth=2)
    axes[0].set_xlabel("# training sentences")
    axes[0].set_ylabel("Probability of parsing a legal sentence")
    axes[0].set_title("(a) Parsing — errors of omission")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    axes[1].plot(xs, [r["gen_gram"] for r in rows],
                  "o-", color=BLUE_DARK, linewidth=2)
    axes[1].set_xlabel("# training sentences")
    axes[1].set_ylabel("Probability of generating a legal sentence")
    axes[1].set_title("(b) Generation — errors of co-mission")
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    fig.suptitle(f"GRIDS-style omission / co-mission — {label}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out_dir, "grids_curves.png"),
                 dpi=140, bbox_inches="tight")
    plt.close()


# ───────────────────────────── Aggregator wrappers ───────────────────

def _regen_learning_curves_aggregate(lc_dir, title):
    """Re-aggregate per-seed learning_curves.csv files in ``lc_dir`` and
    write ``lc_dir/aggregated.csv`` + ``lc_dir/learning_curves.png`` +
    ``lc_dir/grids_curves.png``. Returns the aggregated dict (or None
    if no per-seed data found)."""
    seed_csvs = []
    n_used = 0
    for sd in SEEDS:
        p = os.path.join(lc_dir, f"seed_{sd}", "learning_curves.csv")
        if os.path.exists(p):
            seed_csvs.append(load_learning_curve_csv(p))
            n_used += 1
            # Also re-render per-seed visuals.
            _plot_single_seed_learning_curves(
                p, os.path.join(lc_dir, f"seed_{sd}"),
                label=f"{title} (seed={sd})")
    if not seed_csvs:
        print(f"    (no per-seed learning_curves.csv under {lc_dir})")
        return None
    agg = aggregate_seeds(seed_csvs)
    write_aggregated_csv(os.path.join(lc_dir, "aggregated.csv"), agg)
    plot_learning_curves_with_band(
        out_path=os.path.join(lc_dir, "learning_curves.png"),
        agg=agg, title=title, n_seeds=n_used,
    )
    return agg


# ───────────────────────────── Per-experiment handlers ───────────────

def regen_grammar_test(name):
    """Regen visuals for grammar_{small,med,large}: hollow_learn
    aggregate visuals (performance_summary + step_pick_histograms at
    top level) and learning_curves aggregate."""
    print(f"\n=== {name} ===")
    base = os.path.join(_HERE, name)
    if not os.path.isdir(base):
        print(f"  (missing dir, skipping)")
        return
    hl_dir = os.path.join(base, "hollow_learn")
    lc_dir = os.path.join(base, "learning_curves")

    # 1. hollow_learn aggregate visuals → top level of grammar dir
    if os.path.isdir(hl_dir):
        print(f"  [hollow_learn] regenerating performance_summary + step_pick_histograms")
        build_aggregate_hollow_visuals(hl_dir, viz_dir=base)
    else:
        print(f"  (no hollow_learn dir)")

    # 2. learning_curves: re-aggregate per-seed CSVs + re-render
    if os.path.isdir(lc_dir):
        print(f"  [learning_curves] re-aggregating + re-plotting")
        _regen_learning_curves_aggregate(lc_dir, title=name)
    else:
        print(f"  (no learning_curves dir)")


def regen_experiment(name, spec):
    """Regen per-variant aggregates + overlay (comparison.png +
    grids_overlay.png) for a experiment test."""
    print(f"\n=== {name} ===")
    base = os.path.join(_HERE, name)
    if not os.path.isdir(base):
        print(f"  (missing dir, skipping)")
        return

    agg_by_variant = {}
    for v in spec["variants"]:
        vdir = os.path.join(base, v)
        if not os.path.isdir(vdir):
            print(f"  [{v}] (variant dir missing)")
            continue
        print(f"  [{v}] re-aggregating + per-seed re-rendering")
        agg = _regen_learning_curves_aggregate(vdir, title=f"{name} — {v}")
        if agg is not None:
            agg_by_variant[v] = agg

    if not agg_by_variant:
        print(f"  (no variants had data; skipping overlay)")
        return

    print(f"  [overlay] writing comparison.png + grids_overlay.png")
    plot_overlay_with_bands(
        out_path=os.path.join(base, "comparison.png"),
        agg_by_variant=agg_by_variant,
        colors=spec["colors"],
        labels=spec["labels"],
        suptitle=spec["title"],
        n_seeds=len(SEEDS),
    )


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = list(GRAMMAR_TESTS) + list(EXPERIMENT_TESTS)

    for t in targets:
        if t in GRAMMAR_TESTS:
            regen_grammar_test(t)
        elif t in EXPERIMENT_TESTS:
            regen_experiment(t, EXPERIMENT_TESTS[t])
        else:
            print(f"[WARN] unknown experiment: {t}")
            print(f"       valid: {' '.join(list(GRAMMAR_TESTS) + list(EXPERIMENT_TESTS))}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
