"""
Shared multi-seed orchestration for ACS-26 confs.

Every conf in this folder uses the same pattern: train+eval the WEBSTER
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


# ───────────────────────────── CSV helpers ────────────────────────────

def load_learning_curve_csv(csv_path):
    """Read a learning_curves.csv into parallel arrays:
    (xs, F1, EM, gen_gram, gen_novel)."""
    xs, F1, EM, gen_gram, gen_novel = [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            xs.append(int(row["n_trained"]))
            F1.append(float(row["parse_F1"]))
            EM.append(float(row["parse_EM"]))
            gen_gram.append(float(row["gen_gram"]))
            gen_novel.append(float(row["gen_novel"]))
    return xs, F1, EM, gen_gram, gen_novel


def aggregate_seeds(seed_curves):
    """Given a list of (xs, F1, EM, gen_gram, gen_novel) tuples (one
    per seed, all with the same ``xs`` axis), return a dict of
    ``{"xs": x_array, "F1": (mean, std), "EM": (mean, std),
        "gen_gram": (mean, std), "gen_novel": (mean, std)}``.

    Trims to the shortest curve so means are always well-defined
    even if some seeds have one fewer eval point.
    """
    xs = seed_curves[0][0]
    n = min(len(sc[1]) for sc in seed_curves)
    out = {"xs": xs[:n]}
    for i, name in enumerate(["F1", "EM", "gen_gram", "gen_novel"], start=1):
        arr = np.array([sc[i][:n] for sc in seed_curves])
        out[name] = (arr.mean(axis=0), arr.std(axis=0))
    return out


def write_aggregated_csv(path, agg):
    """Write aggregated learning-curve mean/std to a CSV."""
    with open(path, "w") as f:
        f.write("n_trained,F1_mean,F1_std,EM_mean,EM_std,"
                "gen_gram_mean,gen_gram_std,gen_novel_mean,gen_novel_std\n")
        for i, x in enumerate(agg["xs"]):
            row = [str(x)]
            for k in ["F1", "EM", "gen_gram", "gen_novel"]:
                m, s = agg[k]
                row += [f"{m[i]:.4f}", f"{s[i]:.4f}"]
            f.write(",".join(row) + "\n")


# ───────────────────── Multi-seed learning curves ─────────────────────

def run_multi_seed_learning_curves(corpus_dir, out_base, grammar, corpus,
                                    seeds=SEEDS, eval_every=10,
                                    n_gen_per_eval=30):
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
        )
        seed_curves.append(load_learning_curve_csv(
            os.path.join(seed_dir, "learning_curves.csv")))

    agg = aggregate_seeds(seed_curves)
    agg_path = os.path.join(out_base, "aggregated.csv")
    write_aggregated_csv(agg_path, agg)
    print(f"\n  Aggregated CSV → {agg_path}")
    return agg


def plot_learning_curves_with_band(out_path, agg, title, n_seeds=None):
    """Single-variant 3-panel chart: parse acc, gen gram, gen novelty —
    with mean line + ±1 std shaded band for each metric."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panel_specs = [
        ("Parse accuracy (held-out test)",
            [("F1", "Parse F1", "#1f77b4"),
             ("EM", "Exact-match", "#9467bd")],
            "% on held-out test"),
        ("Generation grammaticality",
            [("gen_gram", "Grammatical", "#bcbd22")],
            "% of outputs"),
        ("Generation novelty (not in train)",
            [("gen_novel", "Novel", "#d62728")],
            "% of outputs"),
    ]
    xs = agg["xs"]
    for ax, (sub_title, keys, ylabel) in zip(axes, panel_specs):
        for k, lbl, c in keys:
            mean, std = agg[k]
            mean_pct = 100 * mean
            std_pct  = 100 * std
            ax.plot(xs, mean_pct, "o-", color=c, linewidth=2, label=lbl)
            ax.fill_between(xs, mean_pct - std_pct, mean_pct + std_pct,
                            color=c, alpha=0.2)
        ax.set_xlabel("# training sentences")
        ax.set_ylabel(ylabel)
        ax.set_title(sub_title)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    suffix = f"  (mean ± std across {n_seeds} seeds)" if n_seeds else ""
    fig.suptitle(title + suffix, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Chart → {out_path}")


def plot_overlay_with_bands(out_path, agg_by_variant, colors, labels,
                              suptitle, n_seeds=None):
    """Three-panel overlay: parse F1 / gen gram / gen novelty, one
    line per variant. Each variant gets a mean line + ±1 std band."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panel_specs = [
        ("Parse F1 (held-out test)",     "F1",        "% bracket F1"),
        ("Generation grammaticality",    "gen_gram",  "% of outputs"),
        ("Generation novelty (not in train)", "gen_novel", "% of outputs"),
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
                                  seeds=SEEDS, primitives_first=200):
    """Orchestrate: run ``run_hollow_learn`` once per seed into
    ``out_base/seed_{N}/``, then aggregate F1/EM/step-pick into
    ``out_base/hollow_learn_summary.csv`` and a printable table.

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
    return agg
