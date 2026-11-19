# Round 215: TinyStories parsing-evaluation curve (does TRELLIS's parse detect grammaticality?)

Question: along the learning curve, how often does TRELLIS's parser succeed on grammatical sentences compared with ungrammatical ones, and does the gap widen with n?
Answer in one line: the gap widens with n, but it stays small. On full scrambles TRELLIS reaches paired AUC .66 -> .88 (n=125 -> 4000) and a gap of .04 -> .17 at the operating point; the n-gram, LSTM and GPT baselines reach .96 -> .998 and .25-.50 -> .90 on the same sentences. On the other three classes (adjacent swap, agreement, content swaps) TRELLIS's parse score is at or near chance, and the baselines are well above it.

## Definitions

- **Parse score.** Read from the code (`src/trellis2/trellis3.py`: `_chart`, `_inside_outside`, `parse`, `generate_best`). The parser builds a chart of every binary analysis; each join has a `gain` (bits, `_factor`); the analyses are weighted 2^(gain/tau) and `_inside_outside` sums them, giving `logZ` (the log total weight over all analyses). `generate_best` (the existing parse critic, r182) already ranks sentences by `logZ / (len - 1)`.
  - **Primary score: `logZ` per word** (`logZ_pw` = logZ / number of words). Higher = more recognised.
  - Secondary, all in the TSV: `logZ_raw`, `logZ_pj` (per join, as `generate_best`), `vit_pw` / `vit_raw` (the Viterbi analysis' total gain, `_chart` best[(0, n)]), `chosen_mean`, `chosen_min` (weakest join of the chosen MBR tree), `chosen_sum_pw`. They are compared in the "TRELLIS score variants" table: none changes the picture (best: `chosen_min`, C3scr AUC .695 at n=4000 against .641 for the primary).
- **Parse success.** score >= threshold. The threshold is fixed per (seed, n, model) at the 10th percentile of the score over a calibration half, `Gcal` (400 held-out grammatical sentences that are in no test set), so that P(success | grammatical) = .9 on the calibration half. The test grammatical set (`C1`, 300 sentences) then shows P(success | grammatical) of .91-.94 for every model (table below), so the operating point carries over.
- **Reported:** AUC (grammatical reference vs corrupted set; "unpaired") and paired AUC (each sentence against its own corruption, so the sentence's own difficulty cancels); P(success | gram) - P(success | corrupted) at the operating point ("gap"); P(success | each set).
- **Grammatical set.** Held-out real TinyStories sentences, 3-8 words, the 250-word vocabulary, from each seed's own held-out stories (the first 1500 stories of the seed's `Random(seed)` shuffle, which the seed's training stream never reads: the story_gen held-out). `C1` = 300 of them (the base), `Gcal` = 400 others. Seed 13's sets are exactly the r214a controls (`story_gen/r214/controls.json`; checked equal on C1, C2full, C2one, C3scr, C4, C4_src by `r215_sets.py`).
- **Ungrammatical sets (each separate; all 300 sentences, built from C1 by the r214a recipe):**
  - `C3scr` full scramble (random shuffle of all words).
  - `C3adj` one adjacent swap across POS (NLTK tags differ).
  - `C4v` agreement errors: was/were, is/are, has/have flipped, subject not i/you/and (289, 290, 287, 286 sentences for seeds 13, 17, 19, 23; the 10-14 dropped have the error word `has`, outside the 250-word vocabulary, which the neural LMs cannot score; reference = the unflipped sentences). A first run with all 300 crashed the LM scorer on `has`.
  - **Key class, syntactic-but-not-semantic:** `C2full` (every content word swapped for another of the same NLTK tag, verbs inside the same argument frame) and `C2one` (one content word swapped). A syntactic parser should treat them like grammatical sentences (AUC .5, gap 0). A semantic detector rejects them.
- **Length matching.** Every corrupted sentence has the same length as its source (the swap, scramble and flips keep the word count), and the reference set is the same sentences, so lengths match exactly; the paired AUC is length-free by construction. Per-word (`pw`) and raw scores are both reported (raw table below; per-word is primary).
- **Baselines**, trained on the same n sentences as the learner, scored on the same sentences, score = sentence log-prob per word (words + `</s>`):
  - n-gram: 2-, 3- and 5-gram, Witten-Bell interpolation (no scorer existed; `story_gen`'s `table(k)` is a sampler).
  - LSTM and GPT: `diag/t3_lm.py` grids and early stopping on held-out stories, as in r205/r212 (models are not saved there; retrained, with the same `_fit`).
  - `random`: a random score (what a right-branching or random parser gives: its output carries no information about the words; AUC .5). `neglen`: -length (constant on matched sets, AUC exactly .5).
- **Recipe and learner.** The paper recipe of r204/r205/r212: `children/spine/heights` off, naming bands, tree cu, refresh verify, store posterior, restructure sample, type_names and word_cc off, repair_path, repair_use 700, exact_alone content, story_facet on, FOLLOW=story, window per n as selected in r205 (n=250 w7, 500 w6, 1000 w7, 4000 w5). **n=125 (w7) and n=2000 (w6) have no r205 selection; these windows are my assumption.** The learner is trained on the first n kept stream sentences of each seed (seeds 13, 17, 19, 23) and `parse()` is called on every test sentence (parsing does not change the learner). `edge_exact` only changes `_gap_fills` (generation), which `parse` does not call, so it does not enter. No saved learners existed (r212/r205/r206/consistency runs dump only generations), so all 24 learners were retrained (jobs `r215_parse.py`; n=4000 takes about 25 min learning).
- **Prior work.** r182's parse-critic pre-test scored generated sentences against the judge (AUC .51-.57 for the parse score as a draw-and-rerank critic) and r192 (`story_gen/r192_diagnosis.md`) found the parse score reaches AUC .67 overall as a sentence filter but .55 (class C) / .58 (class B) inside a class. This round asks a different question (real sentences vs controlled corruptions along n) but the numbers agree: the score separates gross damage and little else.

## Results (means over seeds 13/17/19/23, ± sd across seeds)

Headline numbers, per-word score:

| n | TRELLIS C3scr AUC | paired | gap | GPT C3scr AUC | paired | gap | 5-gram C3scr AUC | gap |
|---|---|---|---|---|---|---|---|---|
| 125 | .537 | .663 | .042 | .888 | .953 | .504 | .880 | .432 |
| 250 | .561 | .737 | .071 | .941 | .980 | .698 | .931 | .647 |
| 500 | .580 | .796 | .123 | .967 | .989 | .818 | .957 | .758 |
| 1000 | .599 | .852 | .151 | .985 | .992 | .878 | .977 | .856 |
| 2000 | .614 | .882 | .162 | .991 | .997 | .887 | .987 | .892 |
| 4000 | .641 | .879 | .165 | .995 | .998 | .899 | .993 | .902 |

1. **Does the gap widen with n?** Yes for TRELLIS on scrambles (paired AUC .66 -> .88, gap .04 -> .17, saturating from n=1000-2000), and a little on adjacent swaps (paired .61 -> .78, gap .03 -> .07). It does not widen on agreement (C4v unpaired AUC .50 at every n; paired .46-.61, noisy; gap about 0) or on the content swaps (gap .00-.03). The baselines' gaps widen on every class (5-gram C4v gap .01 -> .27; GPT C3adj gap .13 -> .71).
2. **TRELLIS is not a competitive grammaticality detector.** At n=4000 TRELLIS passes 77% of scrambles at the operating point where it passes 93% of real sentences (gap .17); the baselines pass about 1% of scrambles (gap .90). Unpaired AUC is low (.64 at n=4000) even where the paired AUC is .88: the parse score varies more between different sentences (word frequency, length) than between a sentence and its scramble. The right-branching/random floor is .50; TRELLIS is above it on scrambles (paired and unpaired) and slightly on adjacent swaps, and at it on the rest.
3. **Syntactic-not-semantic class (C2full, C2one): TRELLIS parses them like grammatical sentences.** At n=4000 P(success) is .905 for C2full and .927 for C2one against .932 for C1 (gap .028 and .005; unpaired AUC .517 and .507). The baselines reject them: gap .50 (C2full) and .14-.18 (C2one) at n=4000, P(success | C2full) about .4. In the paired comparison TRELLIS scores the swapped sentence higher than the original at small n (paired AUC .35-.43 at n<=1000, .50 at n=2000, .58 at n=4000): swapped words are often more frequent ones. Read with care: the baselines' rejection is lexical plausibility (a k-gram or small LM penalises unseen collocations), not a verdict on syntax, and TRELLIS also passes the agreement errors (C4v) and nearly all adjacent swaps, so passing C2 is not evidence that TRELLIS's score encodes syntax. It is evidence that the score is insensitive to everything except gross word-order destruction.
4. **Operating point.** The threshold set on Gcal transfers (P(success | C1) .92-.94 for TRELLIS, .89-.93 for the baselines). TRELLIS's P(success | C3scr) falls from .88 to .77 over the curve; the baselines' falls from .40-.65 to .01.
5. **Score choice.** The other TRELLIS scores (table below) behave alike: unpaired C3scr AUC at n=4000 .59-.70; `chosen_min` (weakest join) is the best (.695 C3scr, .581 C3adj) and is still far below the n-gram models.
6. **Stored sentences.** Held-out C1 sentences that are also training sentences: 1% (n=125) to 12% (n=4000), the same for every model.

The examples (seed 13) are at the end. At n=250 most scrambles and adjacent swaps pass (e.g. `sad bird the little looked` fails, `not anymore was happy the fish` passes); at n=4000 more of them fail but so do some real sentences (`together they opened the box` fails with a score of 1.96 against the threshold 2.108), and the chosen trees for grammatical and agreement-error sentences are nearly identical (`they are not friends anymore` vs `they is not friends anymore`: 3.04 vs 3.05).

Files: `experiments/v3/diag/r215_sets.py`, `r215_common.py`, `r215_parse.py`, `r215_lms.py`, `r215_analyse.py`; sets `story_parse_eval/r215_sets/sets_s{seed}.json`; raw per-sentence scores and trees `r215_raw/`; TSV `r215_parse_curve.tsv` (columns model, score, seed, n, set, auc, auc_pair, p_succ_gram, p_succ_set, gap, threshold, p_succ_cal; the `C1` rows give P(success | grammatical test)).

# Tables (generated by r215_analyse.py)


### AUC by n, mean ± sd over seeds 13/17/19/23 (per-word score; TRELLIS = logZ per word; (k) = k seeds present)


**C3scr** (full scramble)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.537±0.013 | 0.561±0.019 | 0.580±0.007 | 0.599±0.013 | 0.614±0.012 | 0.641±0.018 |
| 5-gram | 0.880±0.016 | 0.931±0.017 | 0.957±0.012 | 0.977±0.007 | 0.987±0.003 | 0.993±0.003 |
| 3-gram | 0.867±0.018 | 0.920±0.015 | 0.950±0.011 | 0.972±0.007 | 0.985±0.003 | 0.992±0.002 |
| 2-gram | 0.852±0.018 | 0.905±0.014 | 0.937±0.012 | 0.962±0.007 | 0.977±0.003 | 0.987±0.003 |
| LSTM | 0.807±0.012 | 0.903±0.012 | 0.944±0.016 | 0.972±0.003 | 0.989±0.003 | 0.993±0.003 |
| GPT | 0.888±0.010 | 0.941±0.009 | 0.967±0.007 | 0.985±0.001 | 0.991±0.001 | 0.995±0.002 |
| random | 0.502±0.019 | 0.491±0.023 | 0.522±0.019 | 0.492±0.009 | 0.506±0.010 | 0.512±0.003 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

**C3adj** (one adjacent swap across POS)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.509±0.003 | 0.517±0.007 | 0.524±0.007 | 0.534±0.004 | 0.542±0.007 | 0.553±0.009 |
| 5-gram | 0.664±0.014 | 0.722±0.014 | 0.769±0.017 | 0.824±0.009 | 0.872±0.010 | 0.908±0.011 |
| 3-gram | 0.662±0.012 | 0.719±0.014 | 0.768±0.016 | 0.824±0.011 | 0.874±0.011 | 0.911±0.011 |
| 2-gram | 0.664±0.012 | 0.720±0.013 | 0.769±0.015 | 0.825±0.013 | 0.869±0.012 | 0.905±0.013 |
| LSTM | 0.602±0.012 | 0.676±0.015 | 0.738±0.021 | 0.813±0.012 | 0.885±0.013 | 0.914±0.017 |
| GPT | 0.681±0.019 | 0.749±0.013 | 0.807±0.008 | 0.869±0.011 | 0.911±0.004 | 0.938±0.007 |
| random | 0.494±0.018 | 0.506±0.033 | 0.513±0.024 | 0.494±0.021 | 0.498±0.013 | 0.512±0.016 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

**C2full** (all content words swapped within POS (syntactic, not semantic))

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.475±0.011 | 0.476±0.010 | 0.482±0.010 | 0.492±0.013 | 0.506±0.009 | 0.517±0.005 |
| 5-gram | 0.702±0.018 | 0.760±0.019 | 0.793±0.017 | 0.829±0.015 | 0.862±0.010 | 0.887±0.016 |
| 3-gram | 0.709±0.018 | 0.765±0.019 | 0.797±0.017 | 0.833±0.015 | 0.865±0.011 | 0.889±0.015 |
| 2-gram | 0.713±0.016 | 0.764±0.020 | 0.794±0.020 | 0.826±0.020 | 0.851±0.016 | 0.870±0.020 |
| LSTM | 0.694±0.022 | 0.747±0.023 | 0.780±0.025 | 0.824±0.021 | 0.858±0.009 | 0.886±0.012 |
| GPT | 0.682±0.023 | 0.733±0.019 | 0.772±0.023 | 0.814±0.026 | 0.843±0.015 | 0.880±0.014 |
| random | 0.490±0.021 | 0.514±0.052 | 0.504±0.015 | 0.492±0.013 | 0.486±0.006 | 0.499±0.010 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

**C2one** (one content word swapped within POS (syntactic, not semantic))

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.489±0.006 | 0.489±0.008 | 0.491±0.006 | 0.495±0.007 | 0.502±0.007 | 0.507±0.004 |
| 5-gram | 0.585±0.015 | 0.632±0.015 | 0.658±0.015 | 0.693±0.018 | 0.731±0.013 | 0.756±0.016 |
| 3-gram | 0.588±0.014 | 0.633±0.014 | 0.660±0.013 | 0.696±0.017 | 0.734±0.014 | 0.758±0.015 |
| 2-gram | 0.589±0.012 | 0.631±0.014 | 0.655±0.014 | 0.686±0.021 | 0.716±0.021 | 0.731±0.020 |
| LSTM | 0.587±0.014 | 0.625±0.018 | 0.644±0.019 | 0.679±0.021 | 0.713±0.012 | 0.739±0.009 |
| GPT | 0.578±0.013 | 0.608±0.012 | 0.633±0.018 | 0.665±0.024 | 0.695±0.013 | 0.735±0.007 |
| random | 0.505±0.016 | 0.499±0.037 | 0.495±0.020 | 0.478±0.018 | 0.505±0.006 | 0.520±0.018 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

**C4v** (agreement error (was/were, is/are, has/have), error word in the 250-word vocabulary)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.504±0.005 | 0.498±0.002 | 0.501±0.003 | 0.502±0.005 | 0.503±0.003 | 0.505±0.002 |
| 5-gram | 0.606±0.030 | 0.653±0.022 | 0.697±0.029 | 0.748±0.021 | 0.800±0.008 | 0.850±0.007 |
| 3-gram | 0.599±0.028 | 0.640±0.019 | 0.677±0.025 | 0.722±0.019 | 0.766±0.010 | 0.811±0.007 |
| 2-gram | 0.589±0.031 | 0.618±0.018 | 0.644±0.022 | 0.679±0.022 | 0.705±0.016 | 0.733±0.012 |
| LSTM | 0.577±0.025 | 0.642±0.020 | 0.693±0.021 | 0.748±0.013 | 0.800±0.018 | 0.831±0.016 |
| GPT | 0.597±0.022 | 0.634±0.024 | 0.695±0.019 | 0.730±0.023 | 0.793±0.024 | 0.816±0.014 |
| random | 0.480±0.024 | 0.509±0.024 | 0.481±0.013 | 0.518±0.026 | 0.491±0.012 | 0.494±0.021 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

### Paired AUC (a sentence against its own corruption)


**C3scr** (full scramble)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.663±0.043 | 0.737±0.053 | 0.796±0.032 | 0.852±0.032 | 0.882±0.017 | 0.879±0.023 |
| 5-gram | 0.956±0.012 | 0.972±0.011 | 0.988±0.006 | 0.993±0.006 | 0.996±0.003 | 0.998±0.002 |
| 3-gram | 0.955±0.016 | 0.971±0.011 | 0.986±0.009 | 0.994±0.007 | 0.997±0.003 | 0.998±0.002 |
| 2-gram | 0.955±0.018 | 0.971±0.012 | 0.984±0.009 | 0.992±0.002 | 0.996±0.003 | 0.996±0.004 |
| LSTM | 0.956±0.007 | 0.978±0.011 | 0.990±0.005 | 0.994±0.004 | 0.998±0.003 | 0.998±0.002 |
| GPT | 0.953±0.015 | 0.980±0.012 | 0.989±0.011 | 0.992±0.003 | 0.997±0.003 | 0.998±0.002 |
| random | 0.500±0.037 | 0.492±0.023 | 0.523±0.040 | 0.489±0.010 | 0.508±0.029 | 0.510±0.007 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

**C3adj** (one adjacent swap across POS)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.608±0.048 | 0.654±0.045 | 0.688±0.048 | 0.734±0.041 | 0.761±0.034 | 0.778±0.035 |
| 5-gram | 0.833±0.015 | 0.902±0.010 | 0.940±0.012 | 0.953±0.016 | 0.972±0.004 | 0.984±0.007 |
| 3-gram | 0.836±0.008 | 0.892±0.007 | 0.938±0.014 | 0.954±0.019 | 0.973±0.008 | 0.987±0.009 |
| 2-gram | 0.826±0.012 | 0.891±0.010 | 0.933±0.017 | 0.956±0.013 | 0.968±0.004 | 0.984±0.009 |
| LSTM | 0.877±0.025 | 0.930±0.018 | 0.955±0.019 | 0.975±0.011 | 0.992±0.002 | 0.992±0.006 |
| GPT | 0.854±0.021 | 0.916±0.010 | 0.944±0.017 | 0.977±0.009 | 0.986±0.006 | 0.991±0.008 |
| random | 0.500±0.036 | 0.504±0.022 | 0.504±0.038 | 0.508±0.023 | 0.506±0.019 | 0.508±0.018 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

**C2full** (all content words swapped within POS (syntactic, not semantic))

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.352±0.036 | 0.393±0.035 | 0.386±0.040 | 0.427±0.032 | 0.504±0.050 | 0.584±0.032 |
| 5-gram | 0.811±0.037 | 0.868±0.022 | 0.892±0.006 | 0.921±0.014 | 0.941±0.005 | 0.951±0.011 |
| 3-gram | 0.818±0.037 | 0.871±0.021 | 0.892±0.016 | 0.922±0.017 | 0.943±0.011 | 0.953±0.006 |
| 2-gram | 0.817±0.034 | 0.867±0.024 | 0.892±0.010 | 0.912±0.023 | 0.930±0.016 | 0.947±0.012 |
| LSTM | 0.797±0.021 | 0.853±0.023 | 0.887±0.022 | 0.913±0.023 | 0.948±0.014 | 0.956±0.012 |
| GPT | 0.803±0.030 | 0.835±0.015 | 0.875±0.006 | 0.911±0.023 | 0.943±0.016 | 0.953±0.006 |
| random | 0.492±0.026 | 0.528±0.048 | 0.496±0.021 | 0.482±0.015 | 0.488±0.006 | 0.509±0.020 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

**C2one** (one content word swapped within POS (syntactic, not semantic))

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.387±0.035 | 0.390±0.047 | 0.406±0.041 | 0.441±0.051 | 0.503±0.060 | 0.533±0.042 |
| 5-gram | 0.706±0.022 | 0.797±0.024 | 0.820±0.019 | 0.865±0.011 | 0.891±0.004 | 0.906±0.004 |
| 3-gram | 0.707±0.023 | 0.797±0.022 | 0.824±0.014 | 0.869±0.018 | 0.892±0.014 | 0.914±0.010 |
| 2-gram | 0.708±0.018 | 0.790±0.020 | 0.818±0.014 | 0.851±0.011 | 0.871±0.011 | 0.887±0.008 |
| LSTM | 0.733±0.034 | 0.810±0.021 | 0.828±0.022 | 0.866±0.017 | 0.893±0.008 | 0.898±0.019 |
| GPT | 0.715±0.022 | 0.765±0.039 | 0.811±0.023 | 0.843±0.022 | 0.863±0.006 | 0.895±0.010 |
| random | 0.500±0.014 | 0.498±0.037 | 0.497±0.033 | 0.487±0.016 | 0.507±0.008 | 0.511±0.030 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

**C4v** (agreement error (was/were, is/are, has/have), error word in the 250-word vocabulary)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.527±0.102 | 0.459±0.089 | 0.572±0.061 | 0.555±0.074 | 0.538±0.042 | 0.608±0.030 |
| 5-gram | 0.792±0.121 | 0.869±0.046 | 0.878±0.023 | 0.920±0.017 | 0.947±0.018 | 0.968±0.015 |
| 3-gram | 0.783±0.118 | 0.860±0.046 | 0.870±0.029 | 0.911±0.025 | 0.937±0.021 | 0.961±0.014 |
| 2-gram | 0.765±0.123 | 0.855±0.054 | 0.864±0.035 | 0.902±0.030 | 0.935±0.029 | 0.952±0.027 |
| LSTM | 0.782±0.081 | 0.889±0.038 | 0.927±0.015 | 0.953±0.017 | 0.974±0.011 | 0.983±0.008 |
| GPT | 0.781±0.058 | 0.876±0.039 | 0.931±0.023 | 0.957±0.007 | 0.971±0.017 | 0.974±0.006 |
| random | 0.471±0.022 | 0.518±0.039 | 0.467±0.029 | 0.529±0.030 | 0.483±0.017 | 0.491±0.025 |
| neglen | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |

### Operating point: P(success | gram) - P(success | corrupted) at the threshold giving P(success | gram) = .9 on Gcal


**C3scr** (full scramble)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.042±0.037 | 0.071±0.039 | 0.123±0.011 | 0.151±0.019 | 0.162±0.018 | 0.165±0.016 |
| 5-gram | 0.432±0.023 | 0.647±0.056 | 0.758±0.023 | 0.856±0.009 | 0.892±0.015 | 0.902±0.013 |
| 3-gram | 0.421±0.042 | 0.588±0.071 | 0.740±0.039 | 0.847±0.010 | 0.888±0.017 | 0.890±0.005 |
| 2-gram | 0.393±0.066 | 0.557±0.068 | 0.677±0.051 | 0.808±0.021 | 0.864±0.016 | 0.890±0.013 |
| LSTM | 0.250±0.044 | 0.548±0.044 | 0.737±0.020 | 0.833±0.014 | 0.889±0.013 | 0.903±0.013 |
| GPT | 0.504±0.052 | 0.698±0.042 | 0.818±0.020 | 0.878±0.014 | 0.887±0.020 | 0.899±0.023 |
| random | -0.006±0.032 | -0.020±0.032 | 0.013±0.026 | 0.009±0.020 | -0.010±0.013 | 0.002±0.029 |
| neglen | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 |

**C3adj** (one adjacent swap across POS)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.028±0.026 | 0.036±0.028 | 0.053±0.009 | 0.060±0.012 | 0.063±0.006 | 0.067±0.017 |
| 5-gram | 0.088±0.023 | 0.132±0.023 | 0.180±0.018 | 0.260±0.022 | 0.389±0.026 | 0.506±0.052 |
| 3-gram | 0.095±0.026 | 0.114±0.027 | 0.162±0.021 | 0.233±0.026 | 0.368±0.027 | 0.534±0.058 |
| 2-gram | 0.096±0.034 | 0.122±0.023 | 0.164±0.023 | 0.238±0.037 | 0.378±0.039 | 0.552±0.020 |
| LSTM | 0.042±0.016 | 0.082±0.014 | 0.137±0.027 | 0.223±0.059 | 0.421±0.068 | 0.558±0.071 |
| GPT | 0.127±0.028 | 0.177±0.023 | 0.247±0.044 | 0.367±0.037 | 0.555±0.022 | 0.705±0.025 |
| random | -0.006±0.034 | -0.005±0.032 | 0.013±0.017 | 0.015±0.002 | -0.010±0.014 | 0.025±0.041 |
| neglen | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 |

**C2full** (all content words swapped within POS (syntactic, not semantic))

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | -0.013±0.018 | -0.008±0.023 | 0.003±0.016 | 0.011±0.020 | 0.018±0.011 | 0.028±0.014 |
| 5-gram | 0.088±0.010 | 0.181±0.045 | 0.232±0.026 | 0.328±0.006 | 0.438±0.063 | 0.520±0.020 |
| 3-gram | 0.108±0.012 | 0.184±0.050 | 0.250±0.032 | 0.336±0.022 | 0.458±0.059 | 0.538±0.009 |
| 2-gram | 0.128±0.012 | 0.209±0.042 | 0.253±0.032 | 0.352±0.050 | 0.453±0.051 | 0.508±0.038 |
| LSTM | 0.105±0.016 | 0.166±0.013 | 0.233±0.049 | 0.288±0.047 | 0.401±0.023 | 0.499±0.036 |
| GPT | 0.107±0.014 | 0.157±0.022 | 0.209±0.052 | 0.278±0.058 | 0.375±0.059 | 0.503±0.009 |
| random | -0.010±0.038 | -0.001±0.048 | 0.012±0.031 | 0.001±0.023 | -0.010±0.012 | -0.002±0.011 |
| neglen | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 |

**C2one** (one content word swapped within POS (syntactic, not semantic))

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | -0.004±0.015 | -0.013±0.023 | -0.005±0.016 | 0.002±0.020 | 0.010±0.010 | 0.005±0.019 |
| 5-gram | 0.025±0.002 | 0.061±0.025 | 0.057±0.015 | 0.087±0.005 | 0.139±0.038 | 0.183±0.014 |
| 3-gram | 0.027±0.005 | 0.055±0.026 | 0.065±0.016 | 0.092±0.008 | 0.134±0.033 | 0.183±0.018 |
| 2-gram | 0.033±0.008 | 0.061±0.029 | 0.072±0.017 | 0.093±0.024 | 0.128±0.031 | 0.180±0.018 |
| LSTM | 0.030±0.009 | 0.048±0.019 | 0.063±0.029 | 0.076±0.017 | 0.107±0.021 | 0.142±0.030 |
| GPT | 0.031±0.005 | 0.052±0.008 | 0.057±0.017 | 0.072±0.013 | 0.096±0.019 | 0.170±0.017 |
| random | -0.008±0.016 | -0.008±0.031 | -0.003±0.029 | -0.004±0.018 | -0.004±0.011 | 0.011±0.030 |
| neglen | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 |

**C4v** (agreement error (was/were, is/are, has/have), error word in the 250-word vocabulary)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.005±0.012 | -0.009±0.013 | -0.010±0.021 | -0.004±0.014 | 0.005±0.002 | 0.003±0.006 |
| 5-gram | 0.007±0.016 | 0.056±0.028 | 0.069±0.018 | 0.139±0.038 | 0.194±0.039 | 0.271±0.025 |
| 3-gram | 0.002±0.020 | 0.052±0.012 | 0.064±0.012 | 0.117±0.024 | 0.156±0.033 | 0.197±0.030 |
| 2-gram | 0.011±0.027 | 0.046±0.008 | 0.060±0.014 | 0.095±0.007 | 0.132±0.030 | 0.159±0.024 |
| LSTM | -0.009±0.030 | 0.045±0.012 | 0.070±0.018 | 0.097±0.026 | 0.152±0.020 | 0.186±0.046 |
| GPT | 0.018±0.017 | 0.050±0.019 | 0.064±0.023 | 0.102±0.038 | 0.141±0.057 | 0.180±0.041 |
| random | -0.008±0.026 | 0.002±0.027 | -0.009±0.024 | 0.010±0.015 | 0.006±0.010 | 0.017±0.018 |
| neglen | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 |

### P(success) at the operating point (threshold fixed on Gcal)


**C1 (grammatical test)**

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.922±0.012 | 0.928±0.010 | 0.930±0.020 | 0.926±0.021 | 0.936±0.016 | 0.932±0.010 |
| 5-gram | 0.901±0.037 | 0.921±0.023 | 0.922±0.022 | 0.913±0.023 | 0.920±0.019 | 0.911±0.009 |
| 3-gram | 0.902±0.032 | 0.914±0.022 | 0.923±0.013 | 0.914±0.017 | 0.915±0.024 | 0.904±0.009 |
| 2-gram | 0.906±0.033 | 0.912±0.014 | 0.925±0.019 | 0.913±0.012 | 0.910±0.021 | 0.913±0.021 |
| LSTM | 0.901±0.036 | 0.907±0.034 | 0.893±0.022 | 0.911±0.013 | 0.912±0.002 | 0.913±0.008 |
| GPT | 0.908±0.033 | 0.906±0.035 | 0.908±0.032 | 0.917±0.013 | 0.903±0.025 | 0.906±0.023 |

**C3scr (full scramble)**

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.881±0.040 | 0.857±0.039 | 0.807±0.021 | 0.775±0.021 | 0.774±0.007 | 0.767±0.007 |
| 5-gram | 0.468±0.025 | 0.274±0.064 | 0.163±0.041 | 0.058±0.020 | 0.028±0.009 | 0.009±0.005 |
| 3-gram | 0.481±0.040 | 0.327±0.084 | 0.183±0.046 | 0.068±0.023 | 0.027±0.008 | 0.014±0.007 |
| 2-gram | 0.513±0.072 | 0.356±0.080 | 0.247±0.055 | 0.105±0.018 | 0.046±0.007 | 0.023±0.014 |
| LSTM | 0.651±0.078 | 0.358±0.073 | 0.156±0.028 | 0.078±0.026 | 0.022±0.013 | 0.010±0.005 |
| GPT | 0.404±0.064 | 0.208±0.037 | 0.090±0.040 | 0.038±0.016 | 0.016±0.008 | 0.007±0.006 |

**C3adj (one adjacent swap across POS)**

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.894±0.024 | 0.892±0.022 | 0.877±0.013 | 0.866±0.014 | 0.873±0.013 | 0.866±0.012 |
| 5-gram | 0.812±0.019 | 0.789±0.026 | 0.742±0.024 | 0.653±0.040 | 0.531±0.026 | 0.405±0.055 |
| 3-gram | 0.807±0.009 | 0.800±0.028 | 0.762±0.029 | 0.681±0.033 | 0.547±0.040 | 0.370±0.054 |
| 2-gram | 0.810±0.025 | 0.791±0.033 | 0.761±0.037 | 0.676±0.041 | 0.532±0.043 | 0.361±0.031 |
| LSTM | 0.859±0.048 | 0.825±0.042 | 0.757±0.038 | 0.688±0.071 | 0.491±0.069 | 0.355±0.066 |
| GPT | 0.782±0.054 | 0.729±0.020 | 0.661±0.075 | 0.550±0.047 | 0.348±0.041 | 0.201±0.029 |

**C2full (all content words swapped within POS (syntactic, not semantic))**

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.935±0.016 | 0.937±0.014 | 0.927±0.016 | 0.915±0.012 | 0.917±0.012 | 0.905±0.004 |
| 5-gram | 0.813±0.042 | 0.740±0.060 | 0.689±0.028 | 0.586±0.026 | 0.483±0.068 | 0.391±0.019 |
| 3-gram | 0.793±0.036 | 0.730±0.066 | 0.673±0.038 | 0.578±0.010 | 0.457±0.068 | 0.366±0.005 |
| 2-gram | 0.777±0.026 | 0.703±0.056 | 0.672±0.044 | 0.562±0.043 | 0.457±0.048 | 0.405±0.040 |
| LSTM | 0.796±0.051 | 0.741±0.036 | 0.660±0.043 | 0.623±0.056 | 0.511±0.023 | 0.414±0.033 |
| GPT | 0.802±0.043 | 0.748±0.046 | 0.699±0.078 | 0.638±0.053 | 0.528±0.041 | 0.402±0.025 |

**C2one (one content word swapped within POS (syntactic, not semantic))**

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.927±0.007 | 0.941±0.015 | 0.935±0.010 | 0.924±0.010 | 0.926±0.008 | 0.927±0.010 |
| 5-gram | 0.876±0.036 | 0.860±0.041 | 0.864±0.027 | 0.826±0.028 | 0.781±0.045 | 0.728±0.022 |
| 3-gram | 0.875±0.033 | 0.859±0.044 | 0.858±0.020 | 0.823±0.011 | 0.781±0.043 | 0.722±0.016 |
| 2-gram | 0.873±0.026 | 0.852±0.042 | 0.853±0.028 | 0.821±0.017 | 0.782±0.035 | 0.733±0.029 |
| LSTM | 0.871±0.037 | 0.858±0.034 | 0.830±0.031 | 0.835±0.024 | 0.805±0.019 | 0.772±0.024 |
| GPT | 0.877±0.034 | 0.853±0.041 | 0.851±0.049 | 0.844±0.013 | 0.807±0.012 | 0.736±0.034 |

**C4v (agreement error (was/were, is/are, has/have), error word in the 250-word vocabulary)**

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.707±0.067 | 0.793±0.046 | 0.844±0.052 | 0.842±0.046 | 0.832±0.049 | 0.845±0.040 |
| 5-gram | 0.892±0.012 | 0.879±0.040 | 0.870±0.014 | 0.799±0.045 | 0.749±0.039 | 0.654±0.020 |
| 3-gram | 0.906±0.011 | 0.892±0.027 | 0.881±0.021 | 0.822±0.031 | 0.780±0.028 | 0.733±0.031 |
| 2-gram | 0.906±0.018 | 0.888±0.020 | 0.885±0.017 | 0.846±0.018 | 0.809±0.019 | 0.781±0.014 |
| LSTM | 0.923±0.021 | 0.892±0.039 | 0.868±0.035 | 0.834±0.046 | 0.782±0.026 | 0.760±0.050 |
| GPT | 0.903±0.011 | 0.876±0.029 | 0.873±0.022 | 0.830±0.049 | 0.790±0.055 | 0.753±0.055 |

### Raw (not length-normalised) scores, AUC mean over seeds


### raw


**C3scr** (full scramble)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.531±0.009 | 0.548±0.010 | 0.560±0.005 | 0.573±0.007 | 0.582±0.007 | 0.594±0.010 |
| TRELLIS | 0.531±0.014 | 0.547±0.013 | 0.560±0.005 | 0.573±0.007 | 0.583±0.008 | 0.595±0.011 |
| 5-gram | 0.782±0.008 | 0.848±0.008 | 0.893±0.009 | 0.934±0.003 | 0.959±0.001 | 0.974±0.002 |
| LSTM | 0.726±0.012 | 0.831±0.014 | 0.884±0.014 | 0.928±0.006 | 0.960±0.004 | 0.972±0.006 |
| GPT | 0.779±0.013 | 0.868±0.012 | 0.917±0.008 | 0.952±0.009 | 0.969±0.004 | 0.980±0.003 |

**C3adj** (one adjacent swap across POS)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.508±0.003 | 0.513±0.005 | 0.518±0.005 | 0.526±0.002 | 0.532±0.005 | 0.536±0.006 |
| TRELLIS | 0.507±0.004 | 0.514±0.006 | 0.517±0.005 | 0.526±0.002 | 0.531±0.006 | 0.536±0.006 |
| 5-gram | 0.623±0.007 | 0.675±0.006 | 0.722±0.013 | 0.778±0.011 | 0.833±0.007 | 0.876±0.008 |
| LSTM | 0.576±0.008 | 0.643±0.013 | 0.699±0.018 | 0.769±0.017 | 0.843±0.011 | 0.882±0.025 |
| GPT | 0.627±0.014 | 0.698±0.014 | 0.762±0.013 | 0.827±0.022 | 0.878±0.011 | 0.914±0.005 |

**C2full** (all content words swapped within POS (syntactic, not semantic))

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.481±0.009 | 0.481±0.009 | 0.485±0.008 | 0.493±0.010 | 0.504±0.007 | 0.511±0.003 |
| TRELLIS | 0.479±0.008 | 0.482±0.009 | 0.487±0.008 | 0.494±0.011 | 0.504±0.007 | 0.511±0.003 |
| 5-gram | 0.652±0.009 | 0.705±0.012 | 0.740±0.016 | 0.781±0.018 | 0.818±0.013 | 0.849±0.014 |
| LSTM | 0.641±0.013 | 0.700±0.024 | 0.733±0.024 | 0.778±0.028 | 0.813±0.016 | 0.845±0.016 |
| GPT | 0.629±0.015 | 0.683±0.017 | 0.726±0.027 | 0.771±0.033 | 0.802±0.020 | 0.844±0.011 |

**C2one** (one content word swapped within POS (syntactic, not semantic))

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.491±0.004 | 0.491±0.006 | 0.493±0.004 | 0.496±0.005 | 0.502±0.005 | 0.505±0.003 |
| TRELLIS | 0.491±0.005 | 0.492±0.007 | 0.494±0.004 | 0.497±0.006 | 0.502±0.004 | 0.505±0.003 |
| 5-gram | 0.567±0.007 | 0.604±0.008 | 0.627±0.009 | 0.661±0.016 | 0.699±0.014 | 0.724±0.017 |
| LSTM | 0.565±0.010 | 0.602±0.018 | 0.618±0.015 | 0.650±0.020 | 0.681±0.016 | 0.708±0.014 |
| GPT | 0.558±0.008 | 0.586±0.011 | 0.609±0.017 | 0.638±0.024 | 0.666±0.017 | 0.707±0.011 |

**C4v** (agreement error (was/were, is/are, has/have), error word in the 250-word vocabulary)

| model | n=125 | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|---|
| TRELLIS | 0.503±0.004 | 0.499±0.002 | 0.501±0.002 | 0.502±0.002 | 0.502±0.002 | 0.505±0.002 |
| TRELLIS | 0.502±0.004 | 0.498±0.002 | 0.499±0.002 | 0.500±0.004 | 0.500±0.001 | 0.505±0.005 |
| 5-gram | 0.575±0.021 | 0.615±0.015 | 0.652±0.020 | 0.702±0.015 | 0.750±0.010 | 0.799±0.011 |
| LSTM | 0.553±0.018 | 0.606±0.016 | 0.648±0.017 | 0.698±0.005 | 0.741±0.018 | 0.768±0.018 |
| GPT | 0.566±0.015 | 0.597±0.016 | 0.647±0.014 | 0.682±0.025 | 0.731±0.022 | 0.758±0.016 |

### TRELLIS score variants, AUC mean over seeds


**n=125**

| score | C3scr | C3adj | C2full | C2one | C4v |
|---|---|---|---|---|---|
| logZ_pw | 0.537 | 0.509 | 0.475 | 0.489 | 0.504 |
| logZ_raw | 0.531 | 0.508 | 0.481 | 0.491 | 0.503 |
| logZ_pj | 0.540 | 0.510 | 0.473 | 0.489 | 0.504 |
| vit_pw | 0.536 | 0.508 | 0.474 | 0.490 | 0.503 |
| vit_raw | 0.531 | 0.507 | 0.479 | 0.491 | 0.502 |
| chosen_mean | 0.538 | 0.508 | 0.473 | 0.489 | 0.502 |
| chosen_min | 0.555 | 0.530 | 0.470 | 0.484 | 0.509 |
| chosen_sum_pw | 0.536 | 0.508 | 0.474 | 0.490 | 0.502 |

**n=250**

| score | C3scr | C3adj | C2full | C2one | C4v |
|---|---|---|---|---|---|
| logZ_pw | 0.561 | 0.517 | 0.476 | 0.489 | 0.498 |
| logZ_raw | 0.548 | 0.513 | 0.481 | 0.491 | 0.499 |
| logZ_pj | 0.566 | 0.518 | 0.474 | 0.488 | 0.498 |
| vit_pw | 0.558 | 0.517 | 0.478 | 0.490 | 0.497 |
| vit_raw | 0.547 | 0.514 | 0.482 | 0.492 | 0.498 |
| chosen_mean | 0.561 | 0.518 | 0.477 | 0.490 | 0.497 |
| chosen_min | 0.576 | 0.536 | 0.472 | 0.489 | 0.499 |
| chosen_sum_pw | 0.557 | 0.517 | 0.478 | 0.491 | 0.497 |

**n=500**

| score | C3scr | C3adj | C2full | C2one | C4v |
|---|---|---|---|---|---|
| logZ_pw | 0.580 | 0.524 | 0.482 | 0.491 | 0.501 |
| logZ_raw | 0.560 | 0.518 | 0.485 | 0.493 | 0.501 |
| logZ_pj | 0.588 | 0.526 | 0.480 | 0.490 | 0.502 |
| vit_pw | 0.575 | 0.521 | 0.483 | 0.493 | 0.497 |
| vit_raw | 0.560 | 0.517 | 0.487 | 0.494 | 0.499 |
| chosen_mean | 0.580 | 0.523 | 0.482 | 0.492 | 0.498 |
| chosen_min | 0.617 | 0.548 | 0.480 | 0.489 | 0.497 |
| chosen_sum_pw | 0.574 | 0.521 | 0.483 | 0.493 | 0.497 |

**n=1000**

| score | C3scr | C3adj | C2full | C2one | C4v |
|---|---|---|---|---|---|
| logZ_pw | 0.599 | 0.534 | 0.492 | 0.495 | 0.502 |
| logZ_raw | 0.573 | 0.526 | 0.493 | 0.496 | 0.502 |
| logZ_pj | 0.609 | 0.539 | 0.491 | 0.495 | 0.503 |
| vit_pw | 0.593 | 0.533 | 0.494 | 0.497 | 0.499 |
| vit_raw | 0.573 | 0.526 | 0.494 | 0.497 | 0.500 |
| chosen_mean | 0.600 | 0.536 | 0.494 | 0.497 | 0.501 |
| chosen_min | 0.630 | 0.556 | 0.497 | 0.501 | 0.495 |
| chosen_sum_pw | 0.593 | 0.533 | 0.494 | 0.497 | 0.499 |

**n=2000**

| score | C3scr | C3adj | C2full | C2one | C4v |
|---|---|---|---|---|---|
| logZ_pw | 0.614 | 0.542 | 0.506 | 0.502 | 0.503 |
| logZ_raw | 0.582 | 0.532 | 0.504 | 0.502 | 0.502 |
| logZ_pj | 0.626 | 0.547 | 0.507 | 0.503 | 0.504 |
| vit_pw | 0.608 | 0.539 | 0.507 | 0.503 | 0.500 |
| vit_raw | 0.583 | 0.531 | 0.504 | 0.502 | 0.500 |
| chosen_mean | 0.617 | 0.543 | 0.508 | 0.503 | 0.501 |
| chosen_min | 0.667 | 0.564 | 0.518 | 0.509 | 0.500 |
| chosen_sum_pw | 0.608 | 0.539 | 0.507 | 0.503 | 0.500 |

**n=4000**

| score | C3scr | C3adj | C2full | C2one | C4v |
|---|---|---|---|---|---|
| logZ_pw | 0.641 | 0.553 | 0.517 | 0.507 | 0.505 |
| logZ_raw | 0.594 | 0.536 | 0.511 | 0.505 | 0.505 |
| logZ_pj | 0.656 | 0.560 | 0.519 | 0.508 | 0.510 |
| vit_pw | 0.637 | 0.552 | 0.516 | 0.508 | 0.506 |
| vit_raw | 0.595 | 0.536 | 0.511 | 0.505 | 0.505 |
| chosen_mean | 0.650 | 0.558 | 0.518 | 0.509 | 0.511 |
| chosen_min | 0.695 | 0.581 | 0.523 | 0.513 | 0.507 |
| chosen_sum_pw | 0.637 | 0.552 | 0.517 | 0.508 | 0.507 |

### Share of C1 test sentences that are stored training sentences (seed 13..23 mean)

n=125: 0.009±0.003; n=250: 0.016±0.007; n=500: 0.025±0.007; n=1000: 0.049±0.009; n=2000: 0.077±0.014; n=4000: 0.118±0.009

### Example parses, seed 13, n=250 (score = logZ per word; threshold 1.159; PASS = success)

| class | sentence | score | parse |
|---|---|---|---|
| grammatical | the little bird looked sad | 1.221 PASS | `((((the little) bird) looked) sad)` |
| adjacent swap | little the bird looked sad | 1.179 PASS | `((((little the) bird) looked) sad)` |
| scramble | sad bird the little looked | 1.083 fail | `(sad ((bird the) (little looked)))` |
| content swap (C2full) | the hard door looked hard | 1.463 PASS | `((the (hard (door looked))) hard)` |
| grammatical | the fish was not happy anymore | 1.945 PASS | `(((the fish) (was not)) (happy anymore))` |
| scramble (passes at n=4000) | not anymore was happy the fish | 1.600 PASS | `((((not anymore) was) happy) (the fish))` |
| grammatical (fails at n=4000) | together they opened the box | 1.209 PASS | `((((together they) opened) the) box)` |
| grammatical (C4 source) | they are not friends anymore | 1.798 PASS | `((they ((are not) friends)) anymore)` |
| agreement error | they is not friends anymore | 1.797 PASS | `((they ((is not) friends)) anymore)` |

### Example parses, seed 13, n=4000 (score = logZ per word; threshold 2.108; PASS = success)

| class | sentence | score | parse |
|---|---|---|---|
| grammatical | the little bird looked sad | 2.265 PASS | `(((the little) (bird looked)) sad)` |
| adjacent swap | little the bird looked sad | 2.106 fail | `(((little the) (bird looked)) sad)` |
| scramble | sad bird the little looked | 1.730 fail | `(((sad bird) the) (little looked))` |
| content swap (C2full) | the hard door looked hard | 2.460 PASS | `(((the hard) (door looked)) hard)` |
| grammatical | the fish was not happy anymore | 3.595 PASS | `(((the fish) was) (not (happy anymore)))` |
| scramble (passes at n=4000) | not anymore was happy the fish | 2.800 PASS | `(((not anymore) was) (happy (the fish)))` |
| grammatical (fails at n=4000) | together they opened the box | 1.960 fail | `((((together they) opened) the) box)` |
| grammatical (C4 source) | they are not friends anymore | 3.039 PASS | `(((they are) not) (friends anymore))` |
| agreement error | they is not friends anymore | 3.052 PASS | `(((they is) not) (friends anymore))` |
