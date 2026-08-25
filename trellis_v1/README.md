# TRELLIS

Reference implementation for **"A Unified Framework of Concepts and Chunks"**
(K. Singaravadivelan and P. Langley). The paper's PDF lives in `paper/`.

TRELLIS gives every element two descriptions — the parts that *compose* it
(content) and the neighbors that *surround* it (context) — and sorts each
description through its own Cobweb hierarchy. One elaboration loop drives
parsing, generation, and incremental learning.

## Layout

```
trellis_v1/
├── src/                 core system (parse_mh.py + grammar utilities)
├── experiments/         the two reproducers + shared harness
├── data/                paper corpora (400 sentences per condition)
├── paper/               paper source + compiled PDF + graphics
├── concept_formation/   Cobweb Python package (dependency)
└── cobweb-private/      C++ Cobweb build (dependency)
```

## Install

```bash
pip install -r requirements.txt
pip install -e cobweb-private
```

## Reproduce the paper's two experiments

Both scripts run five seeds against the corresponding paper condition,
aggregate the learning curves into mean ± 1σ, and write comparison plots.

```bash
# Section 5.2 — vary grammar complexity (SMALL ⊂ MED ⊂ LARGE).
# Produces Figure 5 and the left half of Appendix B.
python experiments/run_grammar_experiment.py

# Section 5.3 — vary lexicon size on MED (LOW ⊂ MED ⊂ HIGH).
# Produces Figure 6 and the right half of Appendix B.
python experiments/run_terminal_experiment.py
```

Each run takes about 15 minutes on a laptop (three variants × five seeds).
Output lands under `experiments/{grammar,terminal}_experiment/`:

```
<variant>/seed_<n>/learning_curves.{csv,png}
<variant>/aggregated.csv
comparison.png            multi-variant overlay, ±1σ bands
grids_overlay.png         Langley & Stromsten omission / commission view
```

The paper's Table 1 is the final-checkpoint row of each `aggregated.csv`
at `n_trained = 320`.

## What TRELLIS reports

Following \citet{grids-langley-stromsten}, both metrics decompose errors
into two complementary categories:

- **Parse accuracy** — bracket-level omission (gold bracket missed) plus
  commission (parser-produced bracket not in gold) on a held-out test fold.
- **Generation grammaticality** — commission-avoidance rate: the fraction
  of freely sampled sentences the source CFG accepts.

## Citing

```
@inproceedings{singaravadivelan-langley-trellis,
  title     = {A Unified Framework of Concepts and Chunks},
  author    = {Singaravadivelan, Karthik and Langley, Pat},
  booktitle = {Advances in Cognitive Systems},
  year      = {2026}
}
```
