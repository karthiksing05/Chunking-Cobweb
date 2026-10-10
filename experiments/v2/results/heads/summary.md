# Track 1: grammatical number, heads and split moves (2026-10-08 to 10-10)

Testbed: `complex_grammars.py` (agree, relative, deep; number agreement, centre embedding, coordination; 2,000
training + 500 held-out sentences each). Measures per 1,000 samples: grammatical (target CFG), number-blind
grammatical (the CFG with number erased: accepted there but not by the CFG = an agreement error only), novel; held-out
bits; bracket omission and commission. Number is used only to evaluate, never by a learner. All code is new
(`experiments/v2/{number_oracle,heads,splits,run_heads_probe,run_heads_characters,run_splits_paper,run_complex_final,
joint_seed_diag,number_split_singles}.py`, `tests/trellis2/test_heads.py`); `src/trellis2` is unchanged and the
experiments substitute names in-process (listed in each result's `substituted`).

## Findings (confirmed unless marked)

1. **Why number merges (Phase A).** No cut through the representation hierarchy separates singular from plural
   (only its leaves, i.e. token identity, do). Splitting every number-mixing category (oracle) shortens the code on
   agree (500 and 2,000; night and gold trees) and on relative from gold trees at 2,000, but lengthens it on relative
   at 500 and on the night's relative analyses at 2,000 (+395 bits).
2. **Heads (both hierarchies, domain-neutral, exact; 12 tests pass) do not make number learnable.** Description
   length never chooses head-conditioned rules (the categories reaching the composition side carry no number);
   representation-side heads help relative's structure (−4.3k bits against a control with blank head attributes)
   but not agreement, and cost characters 2 points of well-formedness.
3. **Greedy barrier.** The oracle number split pays only jointly: on agree 2,000 every noun/determiner/NP category
   split alone costs +190 to +2,550 bits while all together save 1.6–2.2k. A one-symbol split move therefore never
   finds number (it finds role splits instead: good for code and structure, number purity unchanged).
4. **Joint split move** (a seed split propagated through the composition graph by 2-means on neighbours' labels,
   pruned by the exact plain code, accepted by the exact full code) finds number by itself on the night's agree
   analyses: number purity 94–98% / 92–100% / 92% (nominal / verbal / determiner), agreement-only errors 50–54% →
   10–14%, grammatical 46–50% → 86–89%, code −1.5k to −2.2k bits, held-out −1.2 bits per sentence (both seeds,
   full nights with final-round refinement). The remaining errors match the non-constituent bottleneck: only 46%
   (agree) / 32% (relative) of the night's analyses join the subject with its VP, and the oracle split leaves
   0.3% agreement errors on gold trees but 19–33% on the night's analyses (supported, not proven).
5. **Relative**: on the night's analyses even the oracle number split costs bits (+395), and propagated number
   splits cost +128 to +1,900 (head-conditioned rescoring gives identical numbers), so the night's relative
   grammar gains little (seed 17: 29.6% → 34.1% grammatical; seed 13: no change). From gold trees the coordinated
   split pays (oracle −2,190; propagation −1,672 at best; the fixed move accepts 7 joint splits: grammatical 31.7% →
   51.4%, number purity 83% / 80% / 92%). What would make it pay on the night's analyses: analyses that join subject
   and VP (the structure search), and more data (oracle gain grows with size: gold +581 at 500 → −2,190 at 2,000;
   night +767 → +395) (suspicion from the trend, two sizes).
6. **Final-round refinement is the configuration**: splitting in every consolidation round lets the rounds diverge
   (term_low +0.7% code; joint splits every round cost characters 9.5 points of well-formedness at a shorter code);
   refining only the final grammar keeps the agree gain, is never worse than baseline on the paper's corpora, and
   leaves characters at 96.7% well formed.
7. **Description length vs generation on characters**: a shorter code went with worse generation twice
   (joint splits every round: −1.7% code, well formed 97.1% → 87.6%; splits + heads: −1.7% code, 94.8%).
8. **Deep (10–30 words)** stays at forests (0% whole trees) with or without splits: category expressiveness does not
   make whole trees pay there.

Speed: a final-round refinement takes 2–20 minutes at 2,000 sentences (agree), up to 53 minutes (relative gold,
fixed selection: 6 exact refits per step); `splits.CONFIG['joint_max_seeds']` and `['candidates']` cap it.

## Phase A: why number merges

Seed 13. Night = the night's grammar refitted from its stored analyses (reproduces the recorded code exactly); gold = the gold-tree grammar. A2 re-runs compile_grammar's stage B on fixed symbols (reproduces each chosen grammar's code exactly) and splits every category that mixes number (oracle, evaluation only).

| Run | Unsplit bits | Oracle split bits | Δ | Grammatical (number-blind), unsplit → split |
|---|---|---|---|---|
| agree 500 night | 16,497 | 16,251 | -246 | 21.3% (50.6%) → 39.6% (55.7%) |
| agree 500 gold | 19,005 | 18,657 | -348 | 18.9% (51.7%) → 52.3% (53.9%) |
| agree 2000 night | 59,038 | 57,461 | -1,577 | 46.0% (99.6%) → 78.6% (99.3%) |
| agree 2000 gold | 65,018 | 62,821 | -2,197 | 32.5% (67.6%) → 71.5% (71.8%) |
| relative 500 night | 24,842 | 25,609 | +767 | 10.6% (29.2%) → 19.6% (32.8%) |
| relative 500 gold | 29,324 | 29,905 | +581 | 17.6% (38.0%) → 36.7% (38.2%) |
| relative 2000 night | 87,441 | 87,836 | +395 | 34.7% (92.2%) → 56.2% (89.5%) |
| relative 2000 gold | 96,052 | 93,862 | -2,190 | 31.7% (57.5%) → 58.2% (60.5%) |

A1: no cut through the representation hierarchy separates number: the best cut with as many nodes as the chosen categories reaches 58–80% purity by number (base rate 54–68%), and 95% purity needs more than 4× the nodes; only the leaves are pure (token identity).

Greedy barrier (agree 2,000, gold): the oracle split of all categories at once -2,197 bits; each category alone -297 to +2,135 bits (16 of 19 cost bits).
Greedy barrier (agree 2,000, night): the oracle split of all categories at once -1,577 bits; each category alone -41 to +2,552 bits (12 of 14 cost bits).

## Heads and splits on gold trees (2,000 sentences, seed 13)

| Grammar | Variant | Code | K | Held-out bits | Omission | Grammatical | Number-blind | Agreement-only errors |
|---|---|---|---|---|---|---|---|---|
| agree | baseline | 65,018 | 24 | 29.67 | 5.7% | 32.5% | 67.6% | 35.1% |
| agree | composition heads | 65,020 | 24 | 29.67 | 5.7% | 34.0% | 68.9% | 34.9% |
| agree | control (head attributes, no sides) | 61,507 | 22 | 28.82 | 6.2% | 50.9% | 95.0% | 44.1% |
| agree | heads, both sides (learned) | 61,708 | 18 | 29.18 | 5.6% | 50.5% | 95.2% | 44.7% |
| agree | oracle heads (upper bound) | 61,829 | 20 | 29.19 | 6.3% | 42.0% | 93.3% | 51.3% |
| agree | splits every round | 60,871 | 36 | 28.59 | 4.3% | 52.3% | 99.4% | 47.1% |
| agree | splits every round (v2 proposals) | 60,354 | 36 | 28.49 | 5.2% | 49.0% | 99.5% | 50.5% |
| agree | splits + heads | 61,033 | 35 | 28.74 | 5.5% | 57.0% | 99.5% | 42.5% |
| agree | joint splits every round | 59,860 | 50 | 28.20 | 4.9% | 67.7% | 99.5% | 31.8% |
| agree | final-round single | 63,073 | 37 | 28.83 | 5.7% | 52.6% | 95.2% | 42.6% |
| agree | final-round joint | 63,898 | 55 | 28.73 | 5.3% | 49.5% | 97.8% | 48.3% |
| agree | final-round joint (pruned, fixed selection) | 62,256 | 49 | 28.06 | 5.2% | 67.5% | 96.4% | 28.9% |
| relative | baseline | 96,052 | 27 | 43.48 | 9.4% | 31.7% | 57.5% | 25.8% |
| relative | composition heads | 96,054 | 27 | 43.48 | 9.4% | 30.6% | 59.7% | 29.1% |
| relative | control (head attributes, no sides) | 95,850 | 31 | 43.09 | 11.5% | 27.7% | 75.1% | 47.4% |
| relative | heads, both sides (learned) | 91,556 | 20 | 42.65 | 9.9% | 33.8% | 93.1% | 59.3% |
| relative | oracle heads (upper bound) | 93,161 | 26 | 42.65 | 10.6% | 35.2% | 94.1% | 58.9% |
| relative | splits every round | 90,865 | 33 | 42.35 | 10.0% | 42.8% | 98.0% | 55.2% |
| relative | splits every round (v2 proposals) | 91,102 | 54 | 42.35 | 10.9% | 34.5% | 99.2% | 64.7% |
| relative | splits + heads | 88,715 | 26 | 42.01 | 9.7% | 38.9% | 97.8% | 58.9% |
| relative | joint splits every round | 92,147 | 71 | 42.22 | 9.1% | 40.2% | 99.1% | 58.9% |
| relative | final-round single | 93,003 | 50 | 42.25 | 9.6% | 42.7% | 94.4% | 51.7% |
| relative | final-round joint | 93,003 | 50 | 42.25 | 9.6% | 42.7% | 94.4% | 51.7% |
| relative | final-round joint (pruned, fixed selection) | 92,492 | 59 | 41.54 | 9.2% | 51.4% | 93.2% | 41.8% |

## Splits on the night's own analyses (2,000 sentences)

| Grammar | Seed | Variant | Code | Held-out bits | Grammatical | Agreement-only errors | Number purity (nominal / verbal / det) |
|---|---|---|---|---|---|---|---|
| agree | 13 | baseline | 59,038 | 28.79 | 46.0% | 53.6% | 67% / 57% / 68% |
| agree | 13 | splits | 58,919 | 28.73 | 46.5% | 53.3% | 67% / 57% / 68% |
| agree | 13 | splits_joint | 57,638 | 27.71 | 79.0% | 20.7% | 98% / 92% / 92% |
| agree | 13 | final_single | 58,919 | 28.73 | 46.5% | 53.3% | 67% / 57% / 68% |
| agree | 13 | final_joint | 57,638 | 27.71 | 79.0% | 20.7% | 98% / 92% / 92% |
| agree | 13 | final_joint (pruned, fixed selection) | 56,873 | 27.48 | 89.3% | 10.4% | 94% / 100% / 92% |
| agree | 17 | final_joint (pruned, fixed selection) | 57,421 | 27.63 | 86.0% | 13.5% | 89% / 100% / 92% |
| relative | 13 | baseline | 87,441 | 42.07 | 34.7% | 57.5% | 68% / 63% / 70% |
| relative | 13 | splits | 86,782 | 41.95 | 34.5% | 63.3% | 68% / 63% / 70% |
| relative | 13 | splits_joint | 86,782 | 41.95 | 34.5% | 63.3% | 68% / 63% / 70% |
| relative | 13 | final_single | 86,782 | 41.95 | 34.5% | 63.3% | 68% / 63% / 70% |
| relative | 13 | final_joint | 86,782 | 41.95 | 34.5% | 63.3% | 68% / 63% / 70% |
| relative | 13 | final_joint (pruned, fixed selection) | 86,768 | 41.96 | 32.7% | 64.9% | 68% / 63% / 70% |
| relative | 17 | final_joint (pruned, fixed selection) | 85,548 | 41.88 | 34.1% | 65.7% | 65% / 68% / 67% |

## Full unsupervised nights, final-round refinement (single + joint, heads off) against the current night

| Grammar | Seed | Code (night / +refine) | K | Held-out bits/sent (night / +refine / 2-gram / 3-gram) | Brackets om. = com. | Whole trees (train) | Grammatical (night / +refine) | Number-blind | Agreement-only errors (night / +refine) | Novel |
|---|---|---|---|---|---|---|---|---|---|---|
| agree | 13 | 59,038 / 56,873 | 20 / 42 | 28.79 / 27.48 / 31.2 / 33.9 | 22.0% / 22.0% | 100.0% / 100.0% | 46.0% / 89.3% | 99.7% | 53.6% / 10.4% | 99.5% |
| agree | 17 | 58,924 / 57,297 | 20 / 47 | 28.78 / 27.55 / 31.2 / 33.9 | 22.1% / 22.1% | 100.0% / 100.0% | 49.6% / 87.9% | 99.6% | 49.7% / 11.7% | 99.7% |
| relative | 13 | 87,441 / 87,441 | 31 / 31 | 42.07 / 42.07 / 46.7 / 49.9 | 42.7% / 42.7% | 84.8% / 84.8% | 34.7% / 34.7% | 92.2% | 57.5% / 57.5% | 100.0% |
| relative | 17 | 85,900 / 85,548 | 22 / 32 | 42.03 / 41.88 / 46.7 / 49.9 | 42.4% / 42.2% | 100.0% / 100.0% | 29.6% / 34.1% | 99.8% | 69.9% / 65.7% | 100.0% |
| deep | 13 | 163,567 / 162,694 | 27 / 57 | 76.23 / 75.88 / 77.3 / 88.3 | 53.2% / 53.0% | 0.0% / 0.0% | 5.6% / 5.9% | 7.9% | 2.6% / 2.0% | 100.0% |
| deep | 17 | 164,455 / (running) | | | | | | | | | |

## Characters (2,000, seed 13)

| Mode | Variant | Code | K | Held-out bits/char | Omission | Well formed | Attested | Real / rediscovered | Novel |
|---|---|---|---|---|---|---|---|---|---|
| unsupervised | baseline (results/characters) | 63,971 | 23 | 25.64 | 27.5% | 56.5% | 42.2% | 9.1% / 4.9% | 33.1% |
| supervised | baseline (results/characters) | 63,405 | 27 | 24.90 | 0.0% | 97.1% | 70.6% | 14.0% / 7.3% | 56.6% |
| supervised | both | 63,657 | 34 | 24.91 | 0.0% | 95.1% | 68.9% | 13.2% / 8.1% | 55.7% |
| supervised | composition | 63,407 | 27 | 24.90 | 0.0% | 98.2% | 67.5% | 13.9% / 8.3% | 53.6% |
| supervised | control | 62,241 | 27 | 25.03 | 0.1% | 98.2% | 65.4% | 11.2% / 7.1% | 54.2% |
| supervised | final_joint | 63,062 | 29 | 24.45 | 0.0% | 96.7% | 68.9% | 9.9% / 5.0% | 59.0% |
| supervised | final_single | 63,062 | 29 | 24.45 | 0.0% | 96.7% | 68.9% | 9.9% / 5.0% | 59.0% |
| supervised | splits_heads | 62,313 | 33 | 24.30 | 0.0% | 94.8% | 73.5% | 10.0% / 5.1% | 63.5% |
| supervised | splits_joint | 62,352 | 15 | 25.20 | 0.2% | 87.6% | 62.3% | 9.7% / 5.4% | 52.6% |
| supervised | splits | 62,836 | 33 | 24.14 | 0.0% | 97.8% | 74.1% | 12.8% / 7.6% | 61.3% |

## The paper's six corpora, gold trees (code / generation commission / held-out bits)

| Condition | Seed | Baseline | Splits every round | Final-round single | Final-round joint |
|---|---|---|---|---|---|
| small | 13 | 3,254 / 0.1% / 9.52 | 3,254 / 0.1% / 9.52 | 3,254 / 0.1% / 9.52 | 3,254 / 0.1% / 9.52 |
| small | 17 | 3,255 / 0.0% / 9.50 | 3,255 / 0.0% / 9.50 | 3,255 / 0.0% / 9.50 | 3,255 / 0.0% / 9.50 |
| med | 13 | 6,493 / 0.1% / 19.48 | 6,493 / 0.1% / 19.48 | 6,493 / 0.1% / 19.48 | 6,493 / 0.1% / 19.48 |
| med | 17 | 6,571 / 1.0% / 17.47 | 6,502 / 0.3% / 17.44 | 6,541 / 0.1% / 17.45 | 6,541 / 0.1% / 17.45 |
| large | 13 | 8,347 / 14.8% / 24.41 | 8,053 / 3.9% / 23.12 | 8,242 / 10.0% / 23.63 | 8,242 / 10.0% / 23.63 |
| large | 17 | 8,372 / 14.2% / 23.62 | 8,013 / 7.3% / 23.42 | 8,169 / 7.4% / 23.42 | 8,169 / 7.4% / 23.42 |
| term_low | 13 | 5,280 / 0.0% / 15.54 | 5,324 / 0.0% / 15.47 | 5,280 / 0.0% / 15.54 | 5,280 / 0.0% / 15.54 |
| term_low | 17 | 5,295 / 0.2% / 15.03 | 5,329 / 0.1% / 15.03 | 5,295 / 0.2% / 15.03 | 5,295 / 0.2% / 15.03 |
| term_med | 13 | 7,722 / 2.1% / 22.39 | 7,606 / 0.3% / 22.35 | 7,661 / 1.1% / 22.35 | 7,661 / 1.1% / 22.35 |
| term_med | 17 | 7,709 / 3.6% / 21.33 | 7,596 / 0.1% / 21.20 | 7,668 / 0.5% / 21.29 | 7,668 / 0.5% / 21.29 |
| term_high | 13 | 10,484 / 0.2% / 30.43 | 10,438 / 0.6% / 30.42 | 10,484 / 0.2% / 30.43 | 10,484 / 0.2% / 30.43 |
| term_high | 17 | 10,599 / 4.1% / 31.09 | 10,526 / 0.8% / 31.04 | 10,533 / 1.1% / 31.04 | 10,533 / 1.1% / 31.04 |

