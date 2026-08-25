"""
Multi-seed orchestration for the paper's two experiments (Sections 5.2 & 5.3).

Both `run_grammar_experiment.py` and `run_terminal_experiment.py` run the
same pattern: train + evaluate TRELLIS on a corpus with five seeds, aggregate
the per-seed learning-curve CSVs into mean ± 1σ, and overlay the three
variants of the axis being varied.

Public entry points:

    SEEDS                            canonical 5-seed list
    run_multi_seed_learning_curves   train + eval per seed → aggregated.csv
    plot_overlay_with_bands          multi-variant overlay used by both runs
"""
import os, sys, csv, shutil
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learning_curves import run_learning_curves


# Canonical seed list for variance estimation in ACS-26.
# Five seeds is the sweet spot: enough to get a stable mean/std
# (standard error ≈ std/√5 ≈ std/2.2) without 10×-ing run time.
SEEDS = [13, 17, 7, 42, 100]


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
# Experiment variants get a traffic-light triad (green=low/simple,
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
                                    gate_mode:    str   = "skip",
                                    # Hyperparameters plumbed through to
                                    # run_learning_curves (defaults match
                                    # prior behaviour).
                                    context_length:        int   = None,
                                    threshold:             int   = None,
                                    content_alpha:         float = 1e-4,
                                    context_alpha:         float = 1e-4,
                                    content_bl_alpha:      float = 10,
                                    context_bl_alpha:      float = 10,
                                    climb_count_threshold: int   = None,
                                    chunk_pool_weight:     float = 0.0,
                                    sum_leaf_lp_coef:      float = 0.3,
                                    rank_mode:             str   = "context_forward",
                                    rank_w_ctx:            float = 1.0,
                                    rank_w_climb:          float = 3.0,
                                    rank_w_cnt:            float = 0.05,
                                    parse_beam_width:      int   = 1,
                                    content_boundary_feat: str   = None,
                                    content_seam_feat:     str   = None,
                                    content_child_class:   bool  = False,
                                    content_child_class_depth: int = 4,
                                    content_top_k:         int   = 7,
                                    content_pool_depth:    int   = 4,
                                    content_drop_cplx:     bool  = False,
                                    gen_anchor_mode:       str   = "basic",
                                    gen_anchor_tau:        float = 20,
                                    gen_pool_mode:         str   = "leaf",
                                    gen_pool_tau:          float = 50,
                                    parse_mode:            str   = "greedy"):
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
            context_length=context_length,
            threshold=threshold,
            content_alpha=content_alpha,
            context_alpha=context_alpha,
            content_bl_alpha=content_bl_alpha,
            context_bl_alpha=context_bl_alpha,
            climb_count_threshold=climb_count_threshold,
            chunk_pool_weight=chunk_pool_weight,
            sum_leaf_lp_coef=sum_leaf_lp_coef,
            rank_mode=rank_mode,
            rank_w_ctx=rank_w_ctx,
            rank_w_climb=rank_w_climb,
            rank_w_cnt=rank_w_cnt,
            parse_beam_width=parse_beam_width,
            content_boundary_feat=content_boundary_feat,
            content_seam_feat=content_seam_feat,
            content_child_class=content_child_class,
            content_child_class_depth=content_child_class_depth,
            content_top_k=content_top_k,
            content_pool_depth=content_pool_depth,
            content_drop_cplx=content_drop_cplx,
            gen_anchor_mode=gen_anchor_mode,
            gen_anchor_tau=gen_anchor_tau,
            gen_pool_mode=gen_pool_mode,
            gen_pool_tau=gen_pool_tau,
            parse_mode=parse_mode,
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
    the GRIDS-style omission / commission plots."""
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
        ax.set_xlabel("# training sentences", fontsize=12)
        ax.set_ylabel("%", fontsize=12)
        ax.set_ylim(0, 105)  # title removed (info in caption)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Chart → {out_path}")

    # GRIDS-style omission/commission alongside the main chart.
    grids_path = os.path.join(os.path.dirname(out_path), "grids_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pp_m, pp_s = agg["p_parse_legal"]
    axes[0].plot(xs, pp_m, "o-", color=BLUE_DARKEST, linewidth=2)
    axes[0].fill_between(xs, pp_m - pp_s, pp_m + pp_s, color=BLUE_DARKEST, alpha=0.2)
    axes[0].set_xlabel("# training sentences", fontsize=13)
    axes[0].set_ylabel("Probability of parsing a legal sentence", fontsize=13)
    # title removed (info in caption)
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis="both", labelsize=12)
    gg_m, gg_s = agg["gen_gram"]
    axes[1].plot(xs, gg_m, "o-", color=BLUE_DARK, linewidth=2)
    axes[1].fill_between(xs, gg_m - gg_s, gg_m + gg_s, color=BLUE_DARK, alpha=0.2)
    axes[1].set_xlabel("# training sentences", fontsize=13)
    axes[1].set_ylabel("Probability of generating a legal sentence", fontsize=13)
    # title removed (info in caption)
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.savefig(grids_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  GRIDS → {grids_path}")


def plot_overlay_with_bands(out_path, agg_by_variant, colors, labels,
                              suptitle, n_seeds=None, x_cap=None, compact=False):
    """Multi-variant overlay (4-panel): parse Precision, parse Recall,
    gen grammaticality, gen novelty (grammatical). One line per variant
    with ±1σ band. Also writes a GRIDS-style overlay next to ``out_path``.

    ``x_cap`` truncates every variant's curve at that many training sentences
    (``None`` shows the full range). ``compact`` switches the GRIDS overlay to
    a low-whitespace layout for the paper figure. The paper render sets
    ``compact=True`` (and leaves ``x_cap=None`` so the full curve — including
    the later grammaticality convergence — is visible); the experiment scripts
    leave both at their defaults for their live diagnostics."""
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
        ax.set_xlabel("# training sentences", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        # title removed (info in caption)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Overlay → {out_path}")

    # GRIDS-style overlay: parse F1 (left) + generation grammaticality (right).
    # This is the figure the paper embeds (grids_{grammar,terminal}_experiment).
    grids_path = os.path.join(os.path.dirname(out_path), "grids_overlay.png")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3) if compact else (10, 4),
                             gridspec_kw={"wspace": 0.35 if compact else 0.35})
    for v, agg in agg_by_variant.items():
        # Coerce to arrays: the run-script path passes plain lists, the paper
        # render passes numpy arrays; boolean-mask indexing needs arrays.
        xs = np.asarray(agg["xs"])
        m = xs <= x_cap if x_cap is not None else np.ones_like(xs, dtype=bool)
        xs_p = xs[m]
        c = colors.get(v, "#666666"); lbl = labels.get(v, v)
        pp_m, pp_s = np.asarray(agg["F1"][0]), np.asarray(agg["F1"][1])
        axes[0].plot(xs_p, pp_m[m], "o-", color=c, linewidth=2, markersize=4, label=lbl)
        axes[0].fill_between(xs_p, (pp_m - pp_s)[m], (pp_m + pp_s)[m], color=c, alpha=0.18)
        gg_m, gg_s = np.asarray(agg["gen_gram"][0]), np.asarray(agg["gen_gram"][1])
        axes[1].plot(xs_p, gg_m[m], "o-", color=c, linewidth=2, markersize=4, label=lbl)
        axes[1].fill_between(xs_p, (gg_m - gg_s)[m], (gg_m + gg_s)[m], color=c, alpha=0.18)
    _lab_fs = 15 if compact else 16
    _tick_fs = 13 if compact else 14
    _leg_fs = 12 if compact else 13
    axes[0].set_xlabel("# training sentences", fontsize=_lab_fs)
    axes[0].set_ylabel("Parse accuracy", fontsize=_lab_fs)
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis="both", labelsize=_tick_fs)
    axes[0].legend(loc="lower right", fontsize=_leg_fs)
    axes[1].set_xlabel("# training sentences", fontsize=_lab_fs)
    axes[1].set_ylabel("Generation grammaticality", fontsize=_lab_fs)
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="both", labelsize=_tick_fs)
    # Legend appears once on the left panel only (see axes[0].legend above)
    # so the two-panel overlay isn't duplicated.
    if x_cap is not None:
        axes[0].set_xlim(0, x_cap); axes[1].set_xlim(0, x_cap)
    if compact:
        plt.tight_layout(pad=0.3, w_pad=1.2)
    else:
        plt.tight_layout(w_pad=1.5)
    # Vertical divider positioned in the actual visible whitespace
    # midway between the two panels' rendered content (not just their
    # axes bboxes — the right panel's y-label sits to the LEFT of its
    # bbox, so we need the tight-bbox extent to find the true gap).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    left_x1  = inv.transform(
        axes[0].get_tightbbox(renderer).corners()[3])[0]  # right edge
    right_x0 = inv.transform(
        axes[1].get_tightbbox(renderer).corners()[0])[0]  # left edge
    mid_x = (left_x1 + right_x0) / 2
    left_bbox = axes[0].get_position(); right_bbox = axes[1].get_position()
    y_top = max(left_bbox.y1, right_bbox.y1)
    y_bot = min(left_bbox.y0, right_bbox.y0)
    from matplotlib.lines import Line2D
    divider = Line2D([mid_x, mid_x], [y_bot, y_top],
                     transform=fig.transFigure,
                     color="#888888", linestyle=":", linewidth=1.2)
    fig.add_artist(divider)
    plt.savefig(grids_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  GRIDS overlay → {grids_path}")


