"""
Long terminal-sweep experiment: same three variants as
``run_terminal_sweep.py`` (low / med / high lexicon size) but with
**500 sentences per variant** (vs. 200) to test whether the high-
terminal decline recovers once each surface form has been seen enough
times for the content tree to consolidate.

Hypothesis: high (39 terminals) plateaus around ~50% F1 in the 200-
sentence run because each lexical type only appears ~5x. Pushing to
500 sentences gives ~13x per type — if that's enough for Cobweb-
Discrete leaves to form stable basic levels, F1 should recover toward
the med (22-terminal) curve.

Output:
    data/cfg_terminal_long_{low,med,high}/   # 500 sentences each
    confs/acs-26/terminal_sweep_long/
        {low,med,high}/
            seed_{13,17,23,42,100}/learning_curves.{csv,png}
            aggregated.csv
        comparison.png                       # 3-variant overlay, ±1 std

Usage:
    python confs/acs-26/run_terminal_sweep_long.py
"""
import os, sys, json, random, hashlib, shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import (make_grammar_variant, LEXICON_VARIANTS,
                      generate_with_merges)
from _helpers import (run_multi_seed_learning_curves,
                      plot_overlay_with_bands, SEEDS)

N_SENTENCES = 500
EVAL_EVERY  = 25     # 20 eval points, same chart density as the 200-sentence run
DATASET_SEED = 13
DATA_BASE = os.path.join(_ROOT, "data")
OUT_BASE  = os.path.join(_HERE, "terminal_sweep_long")

COLORS = {"low": "#2ca02c", "med": "#1f77b4", "high": "#d62728"}


def make_dataset(variant: str, out_dir: str, seed: int = DATASET_SEED):
    """Generate ``N_SENTENCES`` unique sentences for ``variant``."""
    sizes = LEXICON_VARIANTS[variant]
    grammar, corpus = make_grammar_variant(sizes)
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    seen: set[str] = set()
    n_written = 0
    max_attempts = N_SENTENCES * 60
    attempts = 0
    while n_written < N_SENTENCES and attempts < max_attempts:
        attempts += 1
        text, merges = generate_with_merges(
            "S", grammar, flatten_at_parent=("VP",))
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        with open(os.path.join(out_dir, f"cfg_{h}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"sentence": text, "merges": merges},
                      f, indent=2)
        n_written += 1
    print(f"  {variant}: wrote {n_written} sentences "
          f"({attempts} attempts) into {out_dir}")
    return grammar, corpus


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"=== Terminal sweep (LONG, N={N_SENTENCES}) — generating datasets ===")
    grammars = {}
    corpora  = {}
    n_terminals = {}
    for v in ["low", "med", "high"]:
        ds_dir = os.path.join(DATA_BASE, f"cfg_terminal_long_{v}")
        if os.path.exists(ds_dir):
            shutil.rmtree(ds_dir)
        g, c = make_dataset(v, ds_dir)
        grammars[v] = g
        corpora[v]  = c
        n_terminals[v] = len(c)

    print(f"\n=== Terminal sweep (LONG) — {len(SEEDS)} seeds × {len(grammars)} variants "
          f"= {len(SEEDS)*len(grammars)} learning-curve runs ===")
    agg_by_variant = {}
    for v in ["low", "med", "high"]:
        ds_dir  = os.path.join(DATA_BASE, f"cfg_terminal_long_{v}")
        out_dir = os.path.join(OUT_BASE, v)
        print(f"\n--- variant: {v} ({n_terminals[v]} terminals) ---")
        agg_by_variant[v] = run_multi_seed_learning_curves(
            corpus_dir=ds_dir,
            out_base=out_dir,
            grammar=grammars[v],
            corpus=corpora[v],
            seeds=SEEDS,
            eval_every=EVAL_EVERY,
        )

    print(f"\n=== Building overlay comparison chart (mean ± std) ===")
    labels = {v: f"{v}  ({n_terminals[v]} terminals)" for v in agg_by_variant}
    plot_overlay_with_bands(
        out_path=os.path.join(OUT_BASE, "comparison.png"),
        agg_by_variant=agg_by_variant,
        colors=COLORS,
        labels=labels,
        suptitle=f"TEST_GRAMMAR_MED — learning curves vs lexicon size  (N={N_SENTENCES})",
        n_seeds=len(SEEDS),
    )


if __name__ == "__main__":
    main()
