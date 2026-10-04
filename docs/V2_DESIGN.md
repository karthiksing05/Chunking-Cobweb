# TRELLIS v2 — design and first results

Status: first implementation, October 2026, on branch `inside-outside`.

- Code: `src/trellis2/`
- Tests: `tests/trellis2/` (25, including brute-force checks of the parser and exactness checks of the search)
- Experiments: `experiments/v2/`
- Background: `reports/Trellis v2 inside outside literature review.md`
- **The framework explained end to end, with figures: [`FRAMEWORK.md`](FRAMEWORK.md)**

## Decisions taken with the user

1. The v1-era rules are relaxed (greedy-only parsing, never feeding parser output back, the generation lock, "no hints"). The parsing and generation loops are rebuilt around the new scheme.
2. Exactly two hierarchies, and each holds **both primitives and composites**. There are no extra trees (for example, for non-constituents).
3. Simplicity over patches. One probabilistic model does parsing, generation and description-length scoring, so generation samples from the very distribution the parser and the code lengths use. There are no pools, filters or fallbacks.
4. The in-house Cobweb-MDL variant is not used (not ready). Concept formation is standard Cobweb (category utility). Description length decides only which level of each hierarchy acts as the grammar, and that code lives in `grammar.py` where a Cobweb-MDL variant could replace it.

## Architecture

```
experience (tokens + analysis tree)
   │  every element (primitive or composite) is recorded
   ▼
REPRESENTATION hierarchy  (Cobweb over "how the element behaves")
   │  cut, chosen by MDL, then symbol merging  →  SYMBOLS (categories)
   ▼
COMPOSITION hierarchy     (Cobweb over "what it is made of", in symbol terms)
   │  cut, chosen by MDL  →  RULE CLASSES
   ▼
factored PCFG  P(A→w) = Σc U[A,c] pk[c] E[c,w],   P(A→B C) = Σc U[A,c](1−pk[c]) Lt[c,B] Rt[c,C]
   ├── parsing:     inside-outside → span posteriors → minimum-Bayes-risk tree
   ├── generation:  top-down sampling from the same grammar
   └── learning:    code length of the derivations decides the cuts
```

### Representation instances (`memory.py`)

Each element is described by the following attributes.

**Surface:**
- the token on each side (`l1`, `r1`);
- its first and last token (`f`, `e`);
- its kind (`k`: primitive or composite, postulate R5).

**Chunk context, in category terms:**
- its children's categories (`cl`, `cr`), i.e. what it is made of;
- a two-level **spine**: its parent and grandparent, plus the sibling chunk beside the path at each level (`a1`, `sl1`/`sr1`, `a2`, `sl2`/`sr2`).

Chunk attributes are written at **two granularities**: the grammar's symbol, and a finer node of the hierarchy.

Chunk context needs categories, and categories come from the hierarchy, so **consolidation iterates**:

1. Describe every recorded element with the previous round's categories.
2. Rebuild the representation hierarchy by replaying the elements in learning order (Cobweb, incremental).
3. Read off the grammar.
4. Repeat until the partition into symbols is stable. Keep the round with the shortest code.

The first round has blank chunk attributes.

### Choosing the grammar (`grammar.py`)

- **Code.** The Dirichlet-multinomial marginal likelihood of the training derivations under the grammar's tables. This is a prequential code that does not depend on presentation order. Concentration α = 0.001 (a sparse prior).
- **Symbols.** Search over cuts of the representation tree:
  - start points: the evidence-optimal cut (a bottom-up dynamic program) and a fine cut;
  - moves: *refine* (replace a node by its children) and *collapse* (replace a subtree's cut nodes by their ancestor), in first-improvement passes (refine-everything "kicks" were tried and removed: no effect);
  - then **Bayesian model merging**: greedily join any two symbols, siblings or not, while the code shrinks.
- **Rule classes.** The composition tree is rebuilt over the attested compositions (in symbol terms), and its cut is searched the same way under the factored grammar's code.
- **Tables.** The posterior predictive of each Dirichlet-multinomial. The model is normalized by construction, so the code lengths can see commission.

### Performance (`chart.py`)

- **Inside pass.** Per-span normalization with a log scale, so long sentences don't underflow. The factorization keeps each span at O(n·M).
- **Posteriors.** A top-down pass gives μ(i, j, A), the probability that span (i, j) is a chunk of category A given the whole sentence: its content (inside) and all of its context (outside). These are the chunks' "strength".
- **Decoders:**
  - **MBR**: maximizes the expected number of correct spans ("the best non-intersecting set going down"); used for parsing.
  - **Viterbi**: the shortest-code derivation; used by unsupervised learning.
  - **Posterior sampling**, optionally tempered.
- **Confident spans.** Spans with μ > 0.5 never cross, so they can be learned as confirmed chunks.

## How the representation got here (evidence)

All figures are for the MED or LARGE grammar at 320 training sentences, seed 13 unless stated. "Commission" is the share of generated sentences the target grammar rejects.

| Step | What went wrong without it | Effect |
|---|---|---|
| element kind `k` | a head and its phrase (`dog` / `the dog`) share right context and last word → one category → `X → Det X` (SMALL: 100% commission) | SMALL: 0.8% commission |
| fine start + kicks in the cut search | myopic single moves | no gain alone; the hierarchy was the limit |
| chunk context, 1 level | token windows cannot separate VP / PP / AdjP (all end the sentence) | LARGE 98% → 46%; MED 81% → 40% |
| spine depth 2 | V vs P are separated only by their parent's left sibling (subject vs object) | MED 40% → 8%, LARGE 46% → 17%. Depth 3 over-fragments |
| symbol merging | the true categories (all NPs) are not nodes of the hierarchy | symbols close to the true count (MED 12 vs 11); helps terminals, but MED seeds unstable |
| two granularities | merged labels erase the distinctions the next round needs | MED 20% ± 17 → 8% ± 1 |
| α = 0.001 (vs 0.01, 0.1) | smoothing mass lets generation wander into unseen expansions; codes tie between 0.001 and 0.01, and 0.1 is far worse | MED 8% → 0.5% (3 seeds) |
| composition reference in representation (tried) | — | worse and unstable (LARGE 27% ± 14); off by default |

Two checks confirm the principle behind these results:

- The gold grammar's categories give a *shorter* code than what the search finds when the search fails. MED: 6,697 vs 6,878 bits.
- Commission falls with more data from the same grammar: MED 9.2% → 0.9% and LARGE 22.5% → 8.3% from 160 to 1,280 sentences. Residual commission is description length generalizing in proportion to the evidence.

## Results (supervised regime, as in v1)

Five seeds, the paper's corpora and splits, 320 training sentences, 40 held-out test sentences, 500 generated samples. The v1 figures are the paper's 20-seed endpoints at 300 sentences.

| Condition | Omission v2 | Omission v1 | Gen. commission v2 | Gen. commission v1 | Novelty v2 |
|---|---|---|---|---|---|
| small | 0.0% | 0.0% | 0.2% ± 0.2 | 0.0% | 54% |
| med | 0.0% | 3.3% | 0.9% ± 0.9 | 1.1% | 97% |
| large | 0.7% ± 1.1 | 6.2% | 14.2% ± 1.1 | 1.6% | 99% |
| term_low | 0.0% | 5.7% | 0.3% ± 0.3 | 6.2% | 90% |
| term_med | 0.0% | 7.0% | 2.4% ± 1.1 | 4.7% | 98% |
| term_high | 0.0% | 9.7% | 2.7% ± 1.7 | 3.4% | 100% |

Learning curves: `experiments/v2/results/main/learning_curves.png`. Exploratory sweeps behind the table above: `experiments/v2/results/sweeps/`.

- **Parsing** is at 99–100% from about 40 sentences in every condition.
- **Generation** improves steadily with data. Early on it is worse than v1 at the same small n, because description length favours very general grammars when evidence is scarce.
- **LARGE** is the open case: relative clauses are rare, so they stay merged with adjective phrases at 320 sentences.

Parse commission (1 − bracket precision) equals omission here, because every parse is a complete binary tree.

## Unsupervised learning (v2.1–v2.2)

Code: `unsupervised.py`, `mdl_search.py`, `mdl.py`.

### Objective: an actual message length

The learner minimizes the bits needed to transmit the training sentences.

- **The code.** Each analysis (derivation) is sent event by event with the Dirichlet-multinomial predictive of each grammar table. This is the Bayesian mixture code that arithmetic coding achieves, a prequential code that does not depend on order. The grammar's size (number of symbols and rule classes) is sent with Elias codes.
- **The split.** The total divides into *data bits* (the cost under the best-fitting parameters) and *model bits* (the remainder: the price of learning the parameters).
- **No thresholds.** A chunk type exists only if it pays for its definition. "Minimize chunks while preserving performance" is therefore the objective itself, not a heuristic.

### Partial analyses

A sentence may be a forest of top-level chunks: GRIDS-style partial parses, and the paper's "graceful failure". The top level has the simplest proper code, a symbol distribution plus a stop probability. Fully parsed sentences reduce exactly to the previous model.

Inside-outside, Viterbi and sampling all handle forests, and brute-force tests cover them (`tests/trellis2/test_chart.py`).

### Learning by day and by night

The learner alternates two phases. Sleeping once after observing everything is batch learning.

- **Day (`observe`).** Each sentence is parsed with the current grammar and stored with that analysis. The Viterbi analysis is also the shortest-code analysis: a forest of chunks wherever no larger chunk pays, with unknown words given the category their context implies.
- **Night (`sleep`).** All stored sentences are consolidated:
  1. **Word classes.** Word types are merged while a class-bigram code shrinks (Brown clustering read as description length). The whole merge path is kept.
  2. **Structure.** A beam search over *chunk* (B, C) and *merge* (A, A') moves lowers the plain-PCFG code of the corpus. This is GRIDS, SNPR and Bayesian model merging under one probabilistic code; every move is global, so analyses stay consistent. The search runs from flat sentences in each of the last 12 partitions on the merge path, and from the stored analyses, and keeps the shortest code.
  3. **Concepts.** The analyses are consolidated into the two hierarchies, starting from the search's categories.
  4. **Re-analysis.** Hard EM: Viterbi trees under the full grammar, kept if the total code shrinks. Consolidation starts from their labels.
  5. The stored analyses are rewritten in the new grammar's categories, for the next day to perceive with and the next night to start from.

### Search

- **Exact scores.** The code is a sum of Dirichlet-multinomial row terms. A move changes a few rows, plus the alphabet size that every row's normalizer depends on, so each candidate is scored exactly from cached row sums. Scores agree with a full recomputation to 10⁻¹² bits, and greedy search is about 100× faster (MED: 17 s → 0.1 s).
- **Beam.** Each step keeps the 4 best distinct successors, even ones longer than their parent, as in GRIDS (width 3) and Stolcke (3–10). Duplicates are found by renaming categories in order of first appearance. The search stops after 3 steps without a new shortest code (Stolcke's lookahead) and returns the shortest found.
- **Several starts.** Class-bigram merging never forms a preposition class. It adds the prepositions to the verb class one at a time, because both sit between a noun and a determiner. Its merge path is still a nested family of partitions, and searching from each of its last 12 lets the code with structure, not the bigram code, make the last class merges.

Plain-PCFG code of the 320 training sentences (bits, seed 13):

| Condition | Gold trees, gold categories | Greedy from the final word classes (v2.1) | Greedy, 12 starts | Beam 4, 12 starts |
|---|---|---|---|---|
| small | 3,189 | 3,189 | 3,189 | 3,189 |
| med | 6,651 | 6,727 | 6,585 | **6,539** |
| large | 7,771 | 8,589 | 8,168 | 7,800 |
| term_low | 5,629 | 6,513 | 6,304 | 5,884 |
| term_med | 7,770 | 8,644 | 8,206 | 8,033 |
| term_high | 10,511 | 11,294 | 11,019 | 10,589 |

Over 201 searches (all six conditions, up to 12 starts each, beam widths 1, 4 and 16; `experiments/v2/results/search`, `run_search_study.py`), code length and the commission of the grammar read off the analyses have rank correlation 0.72–0.98 per condition (SMALL 0.68, where nearly every search reaches the same grammar). A wide beam (16) helps on some conditions and hurts on others, which is why the default stays at 4. The objective was right; the search was the bottleneck. From one start, the LARGE beam recovers the gold nouns, verbs, prepositions, adjectives and determiners, but keeps *who* and *which* as two classes where the gold grammar has one relative-pronoun class.

### Results: batch learning

Two seeds, the v1 splits, 320 training sentences, compared with the supervised model on the same sentences' gold trees (`experiments/v2/results/unsupervised/summary.md`).

| Condition | Train bits (unsup / gold trees) | Chunk types (unsup / gold trees) | Test bits/sentence (unsup / gold trees) | Gen. commission (unsup / gold trees) | Gen. commission in v2.1 | Brackets crossing no gold bracket |
|---|---|---|---|---|---|---|
| small | 3,251 / 3,251 | 3.0 / 3.0 | 9.5 / 9.5 | 0.1% / 0.1% | 0.1% | 75% |
| med | **6,498 / 6,528** | 11.0 / 12.5 | 18.5 / 18.5 | **0.5% / 0.6%** | 45.7% | 59% |
| large | **8,031 / 8,356** | 16.0 / 18.5 | **23.6 / 24.0** | **9.2% / 14.5%** | 10.9% | 86% |
| term_low | 5,312 / 5,284 | 13.0 / 12.0 | 15.3 / 15.3 | 0.4% / 0.1% | 25.2% | 53% |
| term_med | 7,744 / 7,712 | 19.5 / 18.0 | 21.8 / 21.9 | 2.7% / 2.9% | 40.9% | 46% |
| term_high | **10,480 / 10,538** | 14.5 / 15.0 | 30.7 / 30.8 | **0.9% / 2.2%** | 62.4% | 38% |

From sentences alone, the learner now matches the supervised model on every condition. Its code is within 0.6% of the gold-tree grammar's (shorter on MED, LARGE and TERM_HIGH), and its commission is at most 0.3 points higher (lower on four conditions). Novelty is 91–100% (SMALL 54%: its language is small).

### Results: by day and by night versus batch

Two seeds. The incremental learner perceives the training sentences one at a time and sleeps at 10, 20, 40, 80, 160 and 320 sentences. At each of those points a fresh batch learner sleeps once over the same sentences (`experiments/v2/results/incremental/`, figure `incremental_vs_batch.png`).

Generation commission (incremental / batch):

| Condition | 40 sentences | 80 | 160 | 320 |
|---|---|---|---|---|
| small | 0.5% / 0.5% | 0.3% / 0.3% | 0.2% / 0.2% | 0.1% / 0.1% |
| med | 57.3% / 57.3% | **24.8% / 48.2%** | **14.4% / 36.9%** | 0.4% / 0.5% |
| large | 34.4% / 34.4% | **27.3% / 43.2%** | 27.0% / 29.6% | 10.2% / 9.2% |
| term_low | 38.1% / 38.1% | **4.0% / 36.2%** | **1.3% / 16.9%** | 0.6% / 0.4% |
| term_med | 70.9% / 70.9% | 44.8% / 43.6% | 6.0% / 7.6% | 1.4% / 2.7% |
| term_high | 69.5% / 69.5% | **31.9% / 72.1%** | **30.3% / 52.9%** | 0.8% / 0.9% |

- **Early nights coincide.** For the first three nights the stored analyses never give the shortest code; a restart from word classes wins and the two learners are identical.
- **Then the stored analyses pay.** From 80 sentences on they win 21 of 30 nights (SMALL excluded, where both give the same grammar). At 80 and 160 sentences the incremental learner's training code is shorter in all 10 cells and its held-out bits are lower or equal. Its commission is lower in 9 of the 10 cells, by 16 to 40 points in seven of them. At 320 sentences the two are equivalent (commission within 1.3 points).
- **Perception.** By 160 sentences each day's sentences are parsed almost completely (1.0–1.1 top-level chunks per sentence), at close to the held-out rate in bits.
- **Cost.** A night costs about as much as a batch sleep over the same sentences. All six nights together cost 1.3–3.8× one batch sleep at 320 (median 1.9×; timings from a shared machine, so approximate). At 320 sentences the 12-start beam search takes 39–64% of a night, and consolidation most of the rest.

### What the experiments established

1. **Description length does not single out linguists' trees.** With gold word classes, the chunk-and-merge search on MED finds a grammar *shorter* than the gold-tree grammar (6,453 vs 6,651 bits) that generates the target language with 0.0% commission, yet shares only 19% of the gold brackets. The learned grammars above match the gold-tree grammars in code and commission, while 38–86% of their brackets cross no gold bracket. Strict MDL identifies the language and leaves its binarization underdetermined. Language-level measures (compression, held-out bits, commission) are the yardstick; bracket agreement is a diagnostic.
2. **The bottleneck was search, not the objective.** Verbs and prepositions share every local context, so the class-bigram code merges them. The code with structure keeps them apart, but greedy search from the bigram classes could not get there. Exact scores, a beam and several starts do.
3. **Consolidation must start from the search's categories.** Re-forming them from blank chunk context lost LARGE's distinctions (47.7% commission at seed 13, against 9.9% when starting from the search's categories).
4. **Incremental learning must be able to refine categories.** The search can only merge categories, while more data pays for finer ones. A first version that continued only from the stored analyses got stuck: at 10 sentences the shortest grammar has a single category, and no merge undoes that. Letting every night also start over fixed it, and the stored analyses still win whenever they are better.
5. **Earlier attempts that did not fix the verb/preposition merge:**
   - Cobweb re-formation after the search;
   - latent split-and-merge EM on the fixed trees;
   - whole-sentence context bags in the representation (`sentence_bags`);
   - Cobweb re-formation inside every search step: about 60× slower, and trapped in high-PMI non-constituents such as "found the".

### Next for unsupervised learning

- **Split moves (coupled category refinement):** split a class and the chunk categories built on it together, scored by the full code. With splits, a night could continue from the stored analyses instead of repeating the batch search, which would make nights cheaper.
- **Faster nights.** The 12-start search and consolidation (Cobweb replay) now each take about half of a night.
- **Penn Treebank** (WSJ10 with gold tags), then a non-language domain.

## Mapping to the paper's postulates

- **R1–R3:** unchanged.
- **R4–R6:** an element's two descriptions are now *representation* (behaviour: surface and chunk context) and *composition* (parts).
- **O1–O3:** unchanged.
- **O4–O5:** two taxonomies, each holding primitives and composites. The grammar is a cut through each. Representation categories become the values of composition instances.
- **P5–P7:** the recognition threshold is replaced by the chunk's posterior under the whole analysis, and by MBR decoding.
- **P8–P10:** generation samples from the same grammar.
- **Learning:**
  - Cobweb's incremental operators are unchanged.
  - Learning alternates day and night. By day each experience is perceived with the current grammar and stored. By night consolidation searches for shorter analyses, re-describes the experiences with current concepts and replays them.
  - MDL picks the level of generalization.

## Roadmap

| Stage | Content |
|---|---|
| v2.0 ✓ | supervised core: hierarchies, MDL cuts and merging, inside-outside + MBR, generation, six-condition evaluation |
| v2.1 ✓ | unsupervised learning from sentences: MDL objective, partial analyses, word classes, exact-scored beam search over chunk/merge moves from several starts |
| v2.2 (in part) | learning by day and by night ✓ (perceive with the current grammar; consolidate at night from the stored analyses or a restart); split moves; attention-like long-range context |
| v2.3 | variable-arity templates, typed relations; first non-language domain (IDS characters, then chess) |
| v2.4 | Penn Treebank (WSJ10 with gold tags, then full WSJ) with a validated evaluator |

## Reproducing

```
python -m pytest tests/trellis2 -q
python experiments/v2/run_synthetic.py --out experiments/v2/results/main      # ~6 min, 6 workers
python experiments/v2/plot_learning_curves.py experiments/v2/results/main
python experiments/v2/run_unsupervised.py --seeds 13,17   # ~5 min, 6 workers
python experiments/v2/run_incremental.py --seeds 13,17    # ~25 min, 6 workers
python experiments/v2/plot_incremental.py experiments/v2/results/incremental
```

The paper corpora are read from `../trellis_v1/data` (the v1 snapshot), with `data/` as the fallback.
