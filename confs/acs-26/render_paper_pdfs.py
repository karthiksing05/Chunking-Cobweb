"""render_paper_pdfs.py — render every paper figure as a vector PDF
alongside its existing PNG, so the paper can use the PDF version for
zoom-friendly inclusion in LaTeX (``\\includegraphics`` accepts PDF
natively and matplotlib's PDF backend writes true vector output).

Strategy: monkey-patch ``plt.savefig`` so every PNG write also produces
a sibling PDF with the same basename. The figure-rendering functions
defined in ``make_paper_figures.py``, ``make_hierarchy_bars.py``, and
``experiment_harness.plot_overlay_with_bands`` all funnel through ``plt.savefig``,
so a single patch covers them all without touching the source files.

The script is graceful about missing prerequisites:
    * If ``TRAINED_MODEL`` (seed_42 MED final_ltm_data) is not on disk
      yet, the trellis-dependent figure (``instances.png``) is skipped
      with a warning instead of crashing.
    * If experiment ``aggregated.csv`` files are absent (experiment jobs still
      running), the experiment grids are skipped.

Outputs land in ``confs/acs-26/paper/graphics/`` next to the PNGs:
    hierarchies.{png,pdf}                instances.{png,pdf}
    parse_infographic.{png,pdf}          generation_infographic.{png,pdf}
    sample_parses_{small,med,large}.{png,pdf}
    hierarchy_bars_{content,context}.{png,pdf}
    grids_grammar_experiment.{png,pdf}        grids_terminal_experiment.{png,pdf}

Usage::

    python confs/acs-26/render_paper_pdfs.py
"""
from __future__ import annotations
import os
import sys
import shutil
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

# Headless backend MUST be set before any matplotlib import that triggers
# figure creation.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Monkey-patch plt.savefig: every PNG write also produces a sibling PDF.
# ----------------------------------------------------------------------
_orig_savefig = plt.savefig


def _savefig_with_pdf(path, *args, **kwargs):
    """Save the figure normally, then write a sibling .pdf if the path
    ended in .png. ``dpi`` is dropped for the PDF write (vector format)."""
    result = _orig_savefig(path, *args, **kwargs)
    if isinstance(path, str) and path.lower().endswith(".png"):
        pdf_path = path[:-4] + ".pdf"
        pdf_kwargs = dict(kwargs)
        pdf_kwargs.pop("dpi", None)
        try:
            _orig_savefig(pdf_path, *args, **pdf_kwargs)
            print(f"  + PDF → {pdf_path}")
        except Exception as e:
            print(f"  ! PDF write failed for {pdf_path}: {e}")
    return result


plt.savefig = _savefig_with_pdf


# ----------------------------------------------------------------------
# Pull in figure-rendering modules AFTER the patch is installed.
# ----------------------------------------------------------------------
import make_paper_figures as mpf
from make_paper_figures import (
    make_hierarchies_figure,
    make_parse_infographic_figure,
    make_generation_infographic_figure,
    make_instances_figure,
    make_sample_parses_figure,
    TRAINED_MODEL,
    OUT_DIR,
)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _safe_call(label: str, fn, *args, **kwargs):
    """Run ``fn`` with kwargs and swallow exceptions so one broken
    figure doesn't kill the rest."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover
        print(f"  ! {label} failed: {exc}")
        traceback.print_exc()


# ----------------------------------------------------------------------
# 1. No-training figures (always renderable).
# ----------------------------------------------------------------------
def render_no_training_figures() -> None:
    _section("no-training figures (hierarchies, infographics, sample parses)")
    _safe_call("hierarchies", make_hierarchies_figure,
               os.path.join(OUT_DIR, "hierarchies.png"))
    _safe_call("parse_infographic", make_parse_infographic_figure,
               os.path.join(OUT_DIR, "parse_infographic.png"))
    _safe_call("generation_infographic", make_generation_infographic_figure,
               os.path.join(OUT_DIR, "generation_infographic.png"))
    _safe_call("sample_parses", make_sample_parses_figure, OUT_DIR)


# ----------------------------------------------------------------------
# 2. Trellis-dependent figures (need a trained TRELLIS).
# ----------------------------------------------------------------------
def render_trellis_figures() -> None:
    _section("trellis-dependent figures (instances)")
    if not os.path.exists(TRAINED_MODEL):
        print(f"  SKIP: TRAINED_MODEL not found at {TRAINED_MODEL}")
        print("       run run_grammar_med.py to seed_42 first")
        return
    from parse_mh import TRELLIS  # noqa: E402
    print(f"  loading TRELLIS from {TRAINED_MODEL}")
    trellis = TRELLIS.load_state(TRAINED_MODEL)
    _safe_call("instances", make_instances_figure,
               os.path.join(OUT_DIR, "instances.png"), trellis)


# ----------------------------------------------------------------------
# 3. Hierarchy bars (need a freshly trained TRELLIS — runs its own
#    ~5–10 min training internally).
# ----------------------------------------------------------------------
def render_hierarchy_bars() -> None:
    _section("hierarchy bars (re-trains a TRELLIS from cfg_grammar_med)")
    try:
        import make_hierarchy_bars as mhb
    except Exception as exc:
        print(f"  SKIP: cannot import make_hierarchy_bars: {exc}")
        return
    _safe_call("hierarchy_bars", mhb.main)


# ----------------------------------------------------------------------
# 4. Experiment grids (re-render from aggregated.csv produced by the experiment
#    scripts; copies the resulting overlay+grids into paper/graphics/
#    under the names the paper references).
# ----------------------------------------------------------------------
def _render_experiment(name: str, experiment_dir: str, variants: list[str],
                  colors: dict, labels: dict, suptitle: str) -> None:
    """Re-render ``grids_<name>.{png,pdf}`` from the per-variant
    aggregated CSVs sitting under ``experiment_dir``. The overlay PNG +
    GRIDS overlay PNG land next to the CSVs, then the GRIDS overlay
    is copied (both PNG and PDF) into the paper graphics folder."""
    import csv as _csv
    import numpy as _np
    from experiment_harness import plot_overlay_with_bands

    missing = [v for v in variants
               if not os.path.exists(
                   os.path.join(experiment_dir, v, "aggregated.csv"))]
    if missing:
        print(f"  SKIP {name}: missing aggregated.csv for {missing}")
        return

    def _load_aggregated_csv(path: str) -> dict:
        """Inverse of ``experiment_harness.write_aggregated_csv``: read an
        aggregated CSV and rebuild the per-metric (mean, std) dict that
        ``plot_overlay_with_bands`` expects."""
        cols = ["F1", "P", "R", "EM",
                "gen_gram", "gen_novel", "gen_gram_novel",
                "p_parse_legal", "prim_active"]
        xs = []
        accum = {k: ([], []) for k in cols}
        with open(path) as f:
            for row in _csv.DictReader(f):
                xs.append(int(row["n_trained"]))
                for k in cols:
                    accum[k][0].append(float(row[f"{k}_mean"]))
                    accum[k][1].append(float(row[f"{k}_std"]))
        out = {"xs": _np.array(xs)}
        for k in cols:
            out[k] = (_np.array(accum[k][0]), _np.array(accum[k][1]))
        return out

    agg_by_variant = {}
    for v in variants:
        agg_path = os.path.join(experiment_dir, v, "aggregated.csv")
        agg_by_variant[v] = _load_aggregated_csv(agg_path)

    out_path = os.path.join(experiment_dir, "comparison.png")
    plot_overlay_with_bands(
        out_path=out_path,
        agg_by_variant=agg_by_variant,
        colors=colors,
        labels=labels,
        suptitle=suptitle,
        n_seeds=None,
        # Paper figure: compact, low-whitespace GRIDS layout over the FULL
        # training range (x_cap=None) so the later grammaticality convergence
        # (~n=300) is visible, not cut off at 200.
        x_cap=None,
        compact=True,
    )

    grids_src_png = os.path.join(experiment_dir, "grids_overlay.png")
    grids_src_pdf = os.path.join(experiment_dir, "grids_overlay.pdf")
    grids_dst_png = os.path.join(OUT_DIR, f"grids_{name}.png")
    grids_dst_pdf = os.path.join(OUT_DIR, f"grids_{name}.pdf")
    for src, dst in ((grids_src_png, grids_dst_png),
                     (grids_src_pdf, grids_dst_pdf)):
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"  copied → {dst}")
        else:
            print(f"  ! missing {src}, not copied")


def render_experiment_grids() -> None:
    _section("experiment grids (grids_grammar_experiment, grids_terminal_experiment)")

    grammar_experiment_dir  = os.path.join(_HERE, "grammar_experiment_20seed")
    terminal_experiment_dir = os.path.join(_HERE, "terminal_experiment_20seed")

    _render_experiment(
        name="grammar_experiment",
        experiment_dir=grammar_experiment_dir,
        variants=["small", "med", "large"],
        colors={"small": "#2ca02c", "med": "#1f77b4", "large": "#d62728"},
        labels={
            "small": "small — S→NP VP; VP=V NP",
            "med":   "med — +AdjP, +PP (S→S PP)",
            "large": "large — +RelClause",
        },
        suptitle="TEST_GRAMMAR experiment — learning curves vs grammar complexity",
    )
    _render_experiment(
        name="terminal_experiment",
        experiment_dir=terminal_experiment_dir,
        variants=["low", "med", "high"],
        colors={"low": "#2ca02c", "med": "#1f77b4", "high": "#d62728"},
        labels={
            "low":  "low  (11 terminals)",
            "med":  "med  (22 terminals)",
            "high": "high (39 terminals)",
        },
        suptitle="TEST_GRAMMAR_MED — learning curves vs lexicon size",
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"PDF render → {OUT_DIR}")

    render_no_training_figures()
    render_trellis_figures()
    render_hierarchy_bars()
    render_experiment_grids()

    print("\nDone. PDF files written next to each PNG.")


if __name__ == "__main__":
    main()
