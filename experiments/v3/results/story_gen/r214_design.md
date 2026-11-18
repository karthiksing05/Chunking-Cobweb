# Round 214 design: one level-of-abstraction dial moves TRELLIS between meaningful, syntax-only and nonsense output

Design only: nothing here has been run. Every code location was read on branch `inside-outside` on 2026-11-18. The judge comes
from round 214a (`experiments/v3/diag/r214_judge2.py`, built separately). It takes sentences and returns `syn_p` and `sem_p`.

## 0. Goal and regime definitions

The claim to test: changing one representational variable, **the information level of the concept at which TRELLIS reads a
description**, moves its generation and its parsing through three regimes. A second variable, context reach, gives part of
the same axis. The regimes, scored on distinct new outputs (not stored sentences, as `new: true`), are:

| regime | judge rule (r214a) | example |
|---|---|---|
| R-sem | `syn_p > .5` and `sem_p > .5` | *the girl ate the red apple* |
| R-syn | `syn_p > .5` and `sem_p <= .5` | *the cup ate the happy sky* |
| R-non | `syn_p <= .5` (`sem_p` is reported, not used) | *happy the ate sky cup* |

`syn_p <= .5` with `sem_p > .5` (word salad on one topic) is part of R-non and is also reported as its own share.
**Purity** of a regime at a setting is the share of judged distinct new outputs that fall in it. Rates are reported per
attempted draw as well, so a setting that fails to produce output cannot hide behind purity (as with the fair count).

## 1. Where generation chooses, read from the code

The current paper recipe is the causal words arm of `experiments/v3/diag/t3_story.py` (branch `elif causal is not None`,
L999 on) on `Trellis3R204` (`edge_exact`). Its config (from `r212_grid.sh`): `naming: bands`, `tree: cu`, `store: posterior`,
`story_facet: true`, `window` 5 to 7. The arm is `{"causal":1, "units":"words", "pair":"gate", "new":true, "temp":τ, "edge":"exact"}`.

One step of generation goes as follows:

1. `gen` (t3_story.py L1171-1192) pads the placed words with `<s>` and calls `_fills(l2, l1, l3..lW)` (L1176). The query is
   `q = {l1..lW}`, the left window only.
2. `_fills` (L1087-1123) calls `L._gap_fills(q)` and `L._gap_fills({**q, "r1": </s>})` (L1091). That gives each filler's mass
   for "word, then go on" and for "word, then end". The pair gate `_ok` (L1082, L1096) drops a filler whose pair `(l1, w)` was
   never stored and that the type licence does not admit.
3. The temperature acts on the drawn weights (L1181-1182, `w ** (1/τ)`), and `_rng.choices` draws (word, end).

**There is no separate "choose a concept, then a word" step in the causal arm.** The concept enters inside the read.
`Trellis3._gap_fills` (`src/trellis2/trellis3.py` L2827-2925) gives a filler w the mass

  m(w | q) = Σ over committed occurrences e of w  of  weight(e) · Π over a in q of P(q_a | C(e)),

with C(e) = `e.leaf`, the occurrence's **context leaf** (L2908: `probs(e.leaf)`). P(v | C) is the hierarchical Dirichlet back-off
along the leaf's path, (n_C(v) + β P(v | parent)) / (n_C + β), with a Laplace root (L2848-2859). So the concept that conditions
the next word is fixed at the leaf, the most specific level, smoothed by its ancestors. The older concept-first generators
(`generate_sub` `up`, `generate_expand` `up`, `_generate_tree` `level` / `after_level`, `_next_pool` `level`, `_pv_pool`
`level`) count levels as edges above the leaf. None of them is the paper recipe.

**Information levels already exist.** `naming="bands"` names a path by the first concept reaching each of `band_bits =
(1,2,4,6,8,12)` bits, where a concept's level is -log2(count(C) / count(root)) (trellis3.py L752-758). `t3_chunks_dump.py`
dumps content concepts at the 3-5, 5-7, 7-9 and 9-12 bit bands. `naming="basic"` (L743-750) picks the concept on a path that
maximises Corter and Gluck's category utility, the basic level. Parsing scores each candidate chunk by `_gain`
(L1047-1060): the best concept on its sorting path of log2[share(C) P(x|C) / P(x|root)], over every level up to the leaf.

## 2. The two knobs

### Knob A (primary, representational): the read level b

**Definition.** Each occurrence is read at its context leaf lifted to level b: C_b(e) is the deepest ancestor of `e.leaf`
(the leaf included) whose information level is ≤ b bits. b = ∞ is the leaf, which is the default and is byte-identical.
b = 0 is the root. The grid is b ∈ {0, 1, 2, 4, 6, 8, 12, ∞}, the existing band edges, plus **b = basic**, the concept on
the leaf's path with the highest category utility (as in `naming="basic"`). The basic setting is TRELLIS choosing its own
level of abstraction, and the test of the user's hypothesis is where it lands.

**Generation location.** `trellis3.py` L2908, `probs(e.leaf)` → `probs(C_b(e))`. It also applies to the end read
(`r1 = </s>`, the same function).

**Parsing location.** `_gain` (L1047-1060) and `_code` (L983-993) take the maximum over `tree.categorize(x)[1:]`. Under the
knob, the maximum runs only over path concepts at level ≤ b. b = 0 gives gain 0 for every chunk. b = ∞ is the default.

**Mechanism.** At level b every occurrence under one concept gets the same factor, so

  m_b(w | q) = Σ_C P(q | C) · n(w, C),

a class-based read in which the classes are context concepts (positions), not word types. As b falls, C holds more
occurrences, P(q | C) grows less specific, and the fillers under C compete by frequency.

- b = ∞ (leaf, back-off; the default): an occurrence counts in proportion to how well its own window matches. The word two
  back (*ate* in *ate the _*) still selects among the nouns. Prediction: **R-sem**, from recombination of attested windows.
- b ≈ 4-6 bits: a concept is a slot (for example "after a determiner", about 1/16 to 1/64 of the tree). P(l1 = the | C) is
  high for every noun occurrence under it, but P(l2 = ate | C) is averaged over every verb that ever preceded that slot.
  So the read licenses any noun after *the*, and selection by *ate* is lost. Prediction: **R-syn**.
- b = 1-2 bits: a concept mixes slots (noun slots with verb slots), and the category is lost along with the selection.
  Prediction: R-syn turning into R-non.
- b = 0 (root): every occurrence has the same factor, so the read is the unigram distribution. With `edge_exact`, first and
  last words are still real starts and ends. Prediction: **R-non** (≥ .9).
- b = basic: predicted to fall at the R-sem/R-syn boundary at n = 1000, and to move deeper in bits as n grows (section 6).

**Why it is the preferred knob.** It changes which concept represents the context, not how the sampler treats the result.
It is one scalar with an information-theoretic unit. The same variable acts in generation (`_gap_fills`) and in parsing
(`_gain`) on the same learned memory, and the chess generator has the same variable (`kx` / `kc`, section 9).

**One confound has to be switched off: the pair gate.** `pair: gate` admits a filler only if `(l1, w)` was stored or
type-licensed. That is a bigram syntax filter that works whatever b is, so with it on, even b = 0 would be bigram-legal and
would look R-syn. The primary sweep therefore runs with **the gate off** (`pair` omitted, everything else as the recipe). The
gate is crossed in on three settings so its own contribution is visible.

### Knob B (primary, context length): read reach W'

**Definition.** The query keeps the slots l1..lW' and drops l_{W'+1}..l_W. With a product read, dropping a slot marginalises
it exactly. The learner is unchanged (learned window W = 7 at n = 1000, the recipe), so **one learned memory per (seed, n)
serves the whole W' × b grid.** The grid is W' ∈ {1, 2, 3, 5, 7}. Sentences have 3 to 8 words, so the 7th word back is the
whole left context: **W' = 7 is W = ∞ for this corpus.**

**Location.** The query is built in t3_story.py `_fills` (L1090). In the subclass, `_gap_fills` filters `q` to `l1..lW'`
(plus `r1`). `hier_fills` / `_hier_index` read `self.window` (L3341, L3429), so the n-gram baselines that go through
`hier_fills(how="ngram")` take W' through the same hook. Their cache keys include W'.

**Learned-window control (no new generation).** The r211 n=1000 dumps already hold plain TRELLIS at learned W = 2 to 7 and
edge at W = 5 to 8, at several temperatures and on 4 seeds. Re-judging them with r214a gives the learned-W curve for free.

**Predictions.**
- At b = ∞:
  - W' = 1: a bigram-level read. Local syntax mostly holds, but selection across a determiner is lost. Prediction: R-syn
    .25-.40, R-non .25-.35. **R-syn will not reach .7 purity through W' alone.** W' is a partial axis, as the user suspects.
  - W' = 2-3: highest R-sem.
  - W' = 5-7: two hypotheses, decided by the data. (H-flat) far slots are close to their back-off and change little. (H-drop)
    as with the learned window in r211 (plain W3 gram .914 against W5 .618 at t0.5), a product over more slots spreads mass
    and syntax falls.
- **Interaction prediction:** the W' effect shrinks as b falls. Below about 4 bits a concept's distribution over l2..l7 is
  close to the marginal, so W' barely matters. The (W', b) table should show a fan, with W' lines converging toward the
  coarse end. This separates "abstraction of the context" from "amount of context": they act as two distinct mechanisms.
- The learned-W curve (r211 dumps, re-judged): sem tracks syn, and the gram fall from W3 to W5 is mostly R-non, not R-syn.
  The r201 evidence (no missing state beyond the window) predicts that longer context does not buy semantics.

### Candidates considered and not chosen as primary

- **Lexical identity vs category in the window** (`window_names` = near / all / far, `nb_bits`; trellis3.py L258-263,
  L885-910). It is representational, but it changes the stored descriptions, so every setting needs a new learner. It also
  reads names through `_res_bag` reservoirs that the recipe (`type_names: false`) does not fill. It is kept as the
  **fallback for knob A** if the Phase 0 kill rule fires: `window_names: "far"` with `nb_bits` = b (l1 a token, farther slots
  categories) is the direct test of "selection lives in the far lexical slots".
- **Temperature over concept vs word.** The causal arm has no separate concept draw (section 1), so this would mean building
  a new sampler. Temperature stays fixed and is used only as a control (the guard in section 3).
- **Levels counted in edges** (`up`, `level`). They are not comparable across trees or across n. Bits are.

## 3. Guards against confounds

1. **Novelty (R-sem must not come from copying).** In every cell, report the `r204_fair.py` quantities: distinct new per
   draw, tri|d (the share of distinct new with an unstored padded trigram), fair (distinct new, regime-correct, unstored
   trigram per attempted draw), repeat rows and the share of repeats that are stored-span cuts. Report **per regime**:
   tri|d inside R-sem, R-syn and R-non separately. R-sem counts toward success only with **tri|d among its sentences ≥ .30**.
   The reference point is plain W3 at t0.5, where 87% of its grammatical sentences are recombined stored trigrams (r211); a
   setting like that does not qualify. The ML n-gram (tri|d = 0) and the 87% figure are the copying anchors in every figure.
2. **Judge validity.** No regime number is read until r214a reports per-axis confusions on three controls already in the
   refs (`r200-n*-refs`, keys `real`, `content swapped`, `word order scrambled`; r209 uses them). Real is the R-sem control,
   content-swapped the R-syn control, scrambled the R-non control. The gate is classification accuracy ≥ .80 per control
   class on held-out refs. Below that, stop and fix the judge. Every cell also gets 20 human-readable samples per regime in
   `r214_samples.md`, drawn at random with a fixed seed, with `syn_p` and `sem_p` printed.
3. **R-syn has to come from the knob, not from temperature or a bad seed.**
   - The knob sweep runs at a fixed τ = 1.0, Cobweb's own posterior with nothing tuned, with a τ = .7 replicate on the
     selection seeds.
   - **Temperature-path control:** b = ∞ at τ ∈ {1, 2, 4, 8}. Prediction: temperature flattens over all words, so its path
     goes R-sem → R-non without an R-syn plateau (max R-syn along it ≤ the knob's R-syn peak - .15). Lowering b flattens
     within a slot, so it passes through R-syn. This contrast is the main evidence that the regime comes from the level.
   - **Replication:** R-syn at the chosen b_syn must hold on all 4 seeds (each ≥ .6) and on held-out 19/23 (mean ≥ .7).
4. **Monotone dose-response, not three points.** Report the full b grid (9 settings) and the full W' grid on held-out seeds.
   Pre-registered shape, per held-out seed:
   - R-non is non-decreasing as b falls (Kendall τ ≥ .8 over the 8 numeric settings).
   - R-sem is non-increasing as b falls (τ ≥ .8).
   - R-syn has one interior peak, at least .2 above both ends.
   - Isotonic fits with bootstrap CIs over sentences and seeds.
5. **Matched lexicon and length.** Every arm uses the same 250-word vocabulary and the same 3-8 word redraw. Per regime and
   cell, report the length histogram (total variation distance to the training lengths), unigram KL to training, type count,
   and mean frequency rank. Regime purities are also reported **post-stratified to the training length distribution**. If
   the R-syn peak disappears after stratification, the knob was acting on length, and that is reported as a failure.
6. **Baselines along the same axis.**
   - **n-gram order** k = 1..5. These are the ML samplers already in t3_story.py (`{"ngram": k}`, L505), which have tri|d = 0
     by construction, and Witten-Bell n-grams with novelty (`hier_fills(how="ngram")` at W' = k-1). Prediction: k = 1 is
     R-non, k = 2 is mixed (R-syn ≤ .4), and k ≥ 3 is R-sem by copying (ML) or with low novelty (WB). Lowering the order
     moves along the same continuum, but with no R-syn plateau of its own.
   - **Class-based n-gram (the key comparison for R-syn).** P(w_i | w_{i-2}, w_{i-1}) = P(c_b(w_i) | c_b(w_{i-2}),
     c_b(w_{i-1})) · P(w_i | c_b(w_i)), Witten-Bell smoothed. The classes come **from the same concepts**: c_b(w) is the
     modal concept C_b(e) over w's committed occurrences (a helper `word_classes(b)` in the subclass, dumped by each learner
     job). Swept over the same b. Pre-registered reading:
     - If the class trigram matches TRELLIS's R-syn purity and novelty at matched b, then R-syn is a class n-gram phenomenon.
       TRELLIS's contribution is that the classes and the level are its own concepts and one dial; it is not a better R-syn
       generator.
     - If TRELLIS holds syntax better at matched b (its classes are soft and per position, the baseline's hard and per type),
       that is the representational advantage.
   - LSTM and GPT at their temperature sweep give only the temperature path, as a reference for guard 3.
7. **Parsing in each regime** (section 5).

## 4. Implementation (opt-in, default byte-identical)

`src/trellis2/trellis3_r214.py`: `class Trellis3R214(Trellis3R208)`, with class attributes `read_bits = None`,
`read_reach = None` and `parse_bits = None`. When all three are None it is `Trellis3R208` exactly, which is
`Trellis3R204`/`R207` with their flags off.

- **`_gap_fills` override.** If `read_reach` is set, it drops `l_k` for k > W' from `q`. If `read_bits` is set, it swaps
  `self.records` for light proxies (`leaf = lift(e.leaf, b)`, with `pas`, `span`, `ctx`, `weight` passed through) inside a
  try/finally, as `Trellis3R204` swaps in its filtered records. It then calls `super()._gap_fills`, so `edge_exact` (R204)
  filters the proxies by `ctx` as usual. The cache is private and its key includes (b, W'). `lift` is memoised per (node id,
  b) and invalidated when `len(records)` or the experience count changes. `basic` uses the CU rule of trellis3.py L743-750.
- **`hier_fills` override** for the WB n-gram baseline at order W'+1: the index is built with `window = W'`, cached under W'.
- **`_gain` / `_code` override** when `parse_bits` is set: the maximum runs over path concepts at level ≤ b.
- **`word_classes(b)`** returns the modal C_b concept id per word type, for the class n-gram.
- **Step diagnostic** (env `STORY_R214_LEVELS`, off by default): for each drawn step under the default read, the
  mass-weighted level in bits of the concept that dominates each occurrence's back-off product. This is used for the
  mediation test in section 7.
- **t3_story.py:** additive arm keys `"bits": b`, `"reach": W'` (the existing `reach` key belongs to r192; use `"wreach"`),
  and `"pbits"` for parse runs. Absent means unchanged, as `edge`/`escape` were added.

`tests/trellis2/test_r214_level.py` follows `test_r204_edge.py` (same CFG and toy sentences):

1. Default byte-identical: `Trellis3R214` with all flags None gives `_gap_fills`, `hier_fills`, `_gain` and parse output
   identical to `Trellis3R208` (and to `Trellis3`) on the toy queries.
2. `read_bits = ∞` and `read_reach = W` each equal the default exactly.
3. `read_bits = 0` gives mass proportional to each word's committed weight (unigram), with edge_exact boundaries respected.
4. Monotone lift: C_b is an ancestor of C_{b'} for b < b', and lifted bits ≤ b.
5. Reach: `read_reach = 1` equals the default read of a query that holds only `l1`.
6. `parse_bits = 0` gives every composite a factor of 0. Records and caches are restored after each call.
7. The full `tests/trellis2` suite passes (sbatch, compute node).

## 5. Parsing in each regime

Two readings, both with the r203 eval (`experiments/v3/story_parse_r203.py`: 300 fixed held-out sentences, NLTK
shallow-phrase gold, F1, F1+sv, crossing, cross-seed F1, phrase consistency and decisiveness against gold .116 / .982).

**(P-a) The knob on the parser.** The same memories are parsed with `parse_bits` = b over the b grid, using the parts recipe
(`consist: "parts"`, the best parser in r203). Predictions:

| b | prediction |
|---|---|
| ∞ (default) | reproduces r203 parts: F1 about .35-.37 |
| 6-8 | F1 within ±.02 of default. Consistency \|d\| and cross-seed F1 improve, because phrases are scored by slot-level concepts, not memorised spans. This is the parse counterpart of R-syn: structure without lexical identity. |
| 2-4 | F1 starts to fall. Brackets follow coarse slot classes and over-bracket. |
| 0 | all gains are 0 and the posterior is flat; F1 falls to ≤ .25 |

No arm beats right-branching (F1 .466 against end-anchored gold); it stays as the reference line. The parse-side law: the
best parse F1 sits at a coarser b than the best R-sem generation. A parser needs categories, while meaning needs lexical
selection, so the two optima need not coincide.

**(P-b) The default parser on each regime's outputs.** At n = 1000 (seeds 13 and 19), 300 distinct new sentences are drawn
per regime from that regime's purest setting. They are parsed with the default parts parser and scored against NLTK gold
computed on those same sentences. Predictions:
- R-sem and R-syn parse alike, within .03 F1 of each other and of real held-out text. Their syntax is intact, which is what
  makes R-syn "syntactic".
- R-non parses at F1 about .20-.25, below right-branching.
- Gold reliability: NLTK's chunker on word salad is itself unreliable. Report its agreement with spaCy noun chunks per
  regime. If agreement on R-non is below .6, the R-non F1 is reported as "gold undefined", not as a parser failure.

## 6. Learning curves and the law

n ∈ {125, 250, 500, 1000, 2000}, seeds 13 and 17 (selection) and 19 and 23 (held out). Learned window: W = 7 for every n (the
recipe at 250 and 1000; 500 used 6, and here it is fixed so that n is the only change). Arms per (n, seed):
b ∈ {0, 2, 4, 6, 8, ∞, basic} at W' = 7, plus W' ∈ {1, 3} at b = ∞, all at τ = 1, gate off, 1600 draws. That is 9 arms.

Laws, both pre-registered (the data decide):
- **L-abs (absolute evidence):** the regime depends on the evidence per read concept, m = N_occ · 2^-b, where N_occ is the
  number of committed occurrences. The regime boundaries collapse when plotted against log2 m = log2 N_occ - b. The n at
  which R-sem overtakes R-syn at level b follows log2 n* = b + c (slope 1 in [.7, 1.3]): each doubling of experience moves
  the crossover one bit deeper. The basic level's bits grow about one-for-one with log2 n.
- **L-rel (scale-free):** the boundaries sit at fixed b for every n (the tree's relative shape is stable). In that case n
  only changes purity within a regime.

Report the regime-share surface over (log2 n, b), both collapse fits (residual variance against each x-axis), n*(b) with
bootstrap CIs, and the basic level's mean bits against n. Prediction: **L-abs holds** (slope about 1), because a concept is
syntactic when it holds enough occurrences to mix the lexical fillers of a slot and few enough to stay one slot, and both
thresholds are counts. The same law is then tested in chess (section 9).

## 7. Does choosing the level explain the shift (mechanism tests)

1. **Phase 0 level map (no generation).** For each band b, over the committed word occurrences in the context tree, compute:
   - **slot purity**: the weighted share of the majority universal POS (spaCy) of the fillers under each C_b;
   - **filler entropy** within C_b;
   - **selection retained**: I(filler; l2 | C_b) as a share of I(filler; l2) at the leaf.

   The prediction is a band where slot purity is ≥ .8, filler entropy is ≥ 3 bits, and selection retained is ≤ .5. That band
   is the predicted b_syn, and it must be the band where generation peaks in R-syn (agreement within one band edge).
   Script: `experiments/v3/diag/r214_levels.py`, modelled on `t3_chunks_dump.py`.
2. **Within-arm mediation.** In the default arm (b = ∞), the step diagnostic gives each sentence's minimum effective level.
   Prediction: sentences judged R-syn have a coarser minimum effective level than R-sem ones (AUC ≥ .60), and R-non ones are
   coarser still. Even TRELLIS's default output would then be sorted into regimes by the level it happened to read at.
3. **The basic level.** If `b = basic` lands at the R-sem end at every n, TRELLIS's own choice of level is the one that keeps
   meaning. If it tracks the R-sem/R-syn boundary, its choice sits at the trade-off between novelty and meaning.

## 8. Run plan

All compute runs through sbatch on overcap (`--account=overcap --partition=overcap --requeue`, `--exclude` the list in
`r212_grid.sh`, which already includes sky1). One learner job runs several arms (as t3_story already does) and dumps
`word_classes(b)` for the baselines. Memory and time, from r204 and r212:

| n | memory | time |
|---|---|---|
| ≤ 250 | 32G | |
| 500 | 48G | |
| 1000 | 128G | about 1h per 1600-draw arm |
| 2000 | 180G, retry at 220G | about 2h per arm |

The n = 4000 runs in r204 needed 160-200G, and n = 4000 is excluded here.

| phase | content | jobs | stop rule |
|---|---|---|---|
| **0a** | Implement `Trellis3R214` and the t3_story keys; tests | 1 (pytest, 16G) | Default byte-identity must pass, or stop |
| **0b** | Judge gate: r214a confusions on real / content-swapped / scrambled refs; re-judge the r211 n=1000 dumps (plain W2-W7, edge W5-W8, n-gram 3/4/5, LSTM, GPT) with r214a, which gives the learned-W curve and baseline regimes at no generation cost | 1-2 GPU (a40, 32G) | Accuracy < .80 on any control class: stop, fix the judge |
| **0c** | Level map (section 7.1), n=1000 seeds 13 and 17, no generation | 2 (128G) | **Kill A:** no band meets purity ≥ .8, entropy ≥ 3 bits and selection ≤ .5 → R-syn by level is not available. Switch knob A to the `window_names: "far"` / `nb_bits` fallback (relearn per setting) and re-plan |
| **1** | Probe, seed 13, n=1000: 9 b settings (gate off, τ 1, W' 7) + gate on at b ∈ {2, 4, ∞} + temperature path b = ∞, τ ∈ {2, 4, 8}: 15 arms | 3 learner jobs (128G) + 1 judge | **Go:** each regime has a setting with purity ≥ .6, R-syn peaks at an interior b, and the temperature path's R-syn max is ≤ the knob peak - .15. **Kill:** R-syn max < .4, or the R-syn peak sits at an end of the grid |
| **2** | n=1000, 4 seeds, W' ∈ {1, 2, 3, 5, 7} × b ∈ {0, 2, 4, 6, ∞, basic} (30 arms per seed; one job per W'); τ .7 replicate of the b grid on 13 and 17; settings b_sem, b_syn, b_non chosen on 13/17 | 20 + 2 learner jobs (128G) + 1 judge | Success criteria below, read on 19/23 |
| **3** | Learning curves, n ∈ {125, 250, 500, 2000} × 4 seeds × 9 arms (n=1000 reused from Phase 2) | 16, with n=2000 split in two: 20 jobs + 1 judge | None. The law is reported whichever way it comes out |
| **4** | Baselines: ML and WB n-gram k=1..5 and the class trigram at b ∈ {1, 2, 4, 6, 8, 12}, from the training streams and the dumped classes, every n × seed | 5 CPU (16G) + 1 judge | None |
| **5** | Parsing: (P-a) `parse_bits` grid, n ∈ {250, 1000, 2000}, seeds 13/17/19 (the r203 protocol); (P-b) the default parser on each regime's outputs, n=1000, seeds 13 and 19 | 9 + 2 | None |
| **6** | Chess (section 9) | about 14 | As in section 9 |

That is about 80 jobs in total. Phases 0-1 come before anything else. Phase 2 starts only on Phase 1's go. Phases 3-5 can
run in parallel after Phase 2's selection. Phase 6 waits for Phase 3's law, because its job is to test that law.

**Success (pre-registered, held-out seeds 19/23, n=1000):**
- **S1:** each regime reaches purity ≥ .7 at its selected setting, on the mean of 19/23 and on each seed ≥ .6.
  - R-sem additionally needs tri|d ≥ .30 among its sentences.
  - R-syn additionally needs to replicate at ≥ .6 on all 4 seeds.
- **S2:** knob A meets the monotone shape of guard 4 on each held-out seed.
- **S3:** the temperature path does not reproduce R-syn (guard 3).

S1-S3 together are the result the user asked for. Knob B is expected to give a monotone partial axis (R-sem ↔ mixed) and
the W' × b fan (section 2). It is not required to reach .7 R-syn.

**The most likely failure, stated in advance.** R-sem ≥ .7 with novelty is the hardest bar. In r211, edge TRELLIS reaches
syntax ≥ .75 only at τ ≤ .7, with tri|d about .62-.66, and R-sem ≤ syntax. If R-sem stays in .55-.70 at τ = 1 and only
clears .7 at τ = .7 on 13/17, the τ = .7 setting is used for R-sem and reported as such: one temperature for all b, never per
regime. If it does not clear .7 at either τ, S1 is reported as met for R-syn and R-non only, with R-sem's frontier
(purity against tri|d) and the LSTM's on the same plot.

## 9. Chess translation

**Regimes.**

| regime | definition | language analogue |
|---|---|---|
| C-real | legal positions that look like games | R-sem |
| C-legal | legal by the rules, with no strategic sense | R-syn |
| C-illegal | breaks the rules | R-non |

C-illegal includes not exactly one king per side, pawns on rank 1 or 8, more than 16 pieces or more than 8 pawns per side,
and the side not to move in check under either side to move.

**Generator (existing code, plus an opt-in sampling mode).** `experiments/v3/chess_compress_gen.py` already holds TRELLIS's
full generative distributions as a code. Its causal route (learner config `chess_causal`, `GAP=1`) gives:
- P(another unit),
- P(gap | previous unit),
- P(symbol | context concept), with the symbol a piece kind or a content concept at level `kc`, and the context concept
  found by sorting the causal description of the board decoded so far, at `kx` levels,
- P(body | concept, anchor) (generate by offsets or on squares, or copy).

A new script `experiments/v3/r214_chess_sample.py` imports that `Model` and **draws** each choice instead of coding it, in
reading order until stop. That is the same as feeding the arithmetic decoder random bits. Nothing is added to the model.

**Knob A in chess** is the same lift. `lvl(path, k)` (chess_compress_gen.py L275) counts edges; the sampler uses
`lift(node, b)` in bits for the context concept (the `kx` read) and for the symbol's content concept (`kc`), b on the same
grid.

**Knob B in chess** is context reach. The causal description (`src/trellis2/trellis_graph.py` L96-135: backward rays, jumps,
square, previous unit `pu`, gap `gp`) is reduced by extending `chess_causal_drop` (L111-117) with the opt-in values
`"rays"` and `"jumps"`:
- {pu, gp} is reach 1, only the previous unit;
- + jumps is local geometry;
- + rays + sq is the full backward board.

Each needs its own learner.

**Predicted map.**
- b = ∞ is C-real, but close to copying. The guard reports the unstored-position share and the share of positions with a
  unit-trigram (three consecutive units in reading order) never stored.
- b at the slot level is C-legal: right piece kinds in plausible relations, wrong whole-board sense.
- b = 0 is C-illegal: piece kinds by frequency on pooled gaps give wrong king counts and pawns on back ranks.

Legality has a global part, exactly one king each, that local unit generation cannot guarantee. The illegal rate at b = ∞ is
therefore predicted to be **above 0 (.05-.2)**, and C-legal purity is read relative to it.

**Judges, validated on controls before use.**
- **Legality** (the syntax axis): python-chess (`import chess`, already used in `chess_cs1973_long_data.py`).
  `Board.is_valid()` with either side to move, and no castling or en passant rights.
- **Realism** (the semantics axis), with two scores:
  - (a) **Game likelihood:** a logistic discriminator on piece-square one-hots plus counts of Chase-Simon pair relations
    (attack, defense, proximity, colour, type, as in `chess_cs1973.py`). It is trained on real positions from games not in
    any training stream against legal shuffles of the same material, and calibrated. Realistic means P(real) > .5.
  - (b) **Engine plausibility**, secondary: Stockfish is not on PATH, so install `stockfish` from conda-forge into the
    `chunks` env. Engine-plausible means |eval| ≤ 5 pawns at depth 10 with no forced mate in ≤ 2.
  - Also reported: the `chess_ideas.py` profile (castled king, pawn chain and battery rates against real positions).
- **Controls:**
  - real held-out positions (should judge C-real);
  - the Chase-Simon random positions (the same pieces on random squares; keep the legal ones, which should judge C-legal);
  - unconstrained random placements (should judge C-illegal).
  - Gate: accuracy ≥ .80 per class, as in guard 2.

**Baseline along the axis.** `experiments/v3/chess_ngram.py` (Witten-Bell n-grams over pieces in FEN reading order,
tokenizations `abs` and `rel`). Add sampling, and order k = 1..3 gives the n-gram continuum.

**Learning curve and the law.** The `cs73L` stream (`data/chess/cs1973/train_long.fen`, GraphTrellis, the r213 config with
`chess_causal`). Checkpoints n ∈ {300, 1000, 4000, 8000, 16000} positions, seeds 13 and 17 to select, 19 and 23 held out.
Memory 64G as cs73L (`logs/v3_cs73L_submit.sh`), 128G for n ≥ 8000.

The test is whether chess follows the same law as language: the C-real/C-legal crossover against b, with
log2 n* = b + c under L-abs. That law would finalise the chess section. Expertise, more positions learned, moves the level
at which TRELLIS's chunks stay realistic one bit deeper per doubling. This is the Chase-Simon reading: masters' chunks are
both larger and more specific.

**Chess jobs.**
- 1: the sampler + legality smoke test, no learner change.
- 4: learners with checkpoints, one per seed, saving memory at each checkpoint.
- 8: sampling over b × 3 reach settings, per seed (2 jobs each).
- 1: the judges and controls.

That is about 14 jobs. **Stop rule:** if the controls fail the judge gate, or if b = ∞ is more than .3 illegal, the global
king constraint dominates. In that case, add the opt-in "kings first" step (place both kings from their learned square
distributions before the causal units) and report it as a fixed scaffold, the same at every b.
