# TRELLIS v2 — design and first results

Status: first implementation, October 2026, on branch `inside-outside`.

- Code: `src/trellis2/`
- Tests: `tests/trellis2/` (31, including brute-force checks of the parser, exactness checks of the search, and the compiled Cobweb against its Python reference)
- Experiments: `experiments/v2/`
- Background: `reports/Trellis v2 inside outside literature review.md`
- **The framework explained end to end, with figures: [`FRAMEWORK.md`](FRAMEWORK.md)**

## Decisions taken with the user

1. The v1-era rules are relaxed (greedy-only parsing, never feeding parser output back, the generation lock, "no hints"). The parsing and generation loops are rebuilt around the new scheme.
2. Exactly two hierarchies, and each holds **both primitives and composites**: the **representation hierarchy** (how an element behaves) and the **composition hierarchy** (what it is made of). There are no extra trees (for example, for non-constituents). The two hierarchies are the essential core of v2 (stressed again by the user on 2026-10-04): the grammar's categories and chunk types are cuts through them, parsing and generation use that grammar, and learning replays experiences into them. Every change is stated as a change in what the two hierarchies record.
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
  5. **Choice.** Steps 3–4 run on each of the three best distinct search results; the grammar with the shortest total code wins.
  6. The stored analyses are rewritten in the new grammar's categories, for the next day to perceive with and the next night to start from.

### Search

- **Exact scores.** The code is a sum of Dirichlet-multinomial row terms. A move changes a few rows, plus the alphabet size that every row's normalizer depends on, so each candidate is scored exactly from cached row sums. Scores agree with a full recomputation to 10⁻¹² bits, and greedy search is about 100× faster (MED: 17 s → 0.1 s).
- **Incremental moves.** A child state shares every unchanged sentence and row with its parent. A chunk move rewrites only the sentences that hold the pair; a merge only records a renaming, applied when labels are read. Duplicate states are looked for only among successors with equal codes. On WSJ20 (27,000 tokens) one beam search went from 49 s to 9 s, with the same result. Codes equal to 10⁻⁷ nats are ties, broken by the order of the moves, so results do not depend on floating-point noise.
- **Word classes** are scored the same way: merging two classes pools two rows and two columns of the class-bigram counts, so every pair is scored exactly from those counts (O(K³) per step). The merge path is identical to the from-scratch search on all six conditions and about 300× faster.
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

Over 201 searches (all six conditions, up to 12 starts each, beam widths 1, 4 and 16; `experiments/v2/results/search`, `run_search_study.py`), code length and the commission of the grammar read off the analyses have rank correlation 0.71–0.98 per condition (SMALL 0.67, where nearly every search reaches the same grammar). A wide beam (16) helps on some conditions and hurts on others, which is why the default stays at 4. The objective was right; the search was the bottleneck. From one start, the LARGE beam recovers the gold nouns, verbs, prepositions, adjectives and determiners, but keeps *who* and *which* as two classes where the gold grammar has one relative-pronoun class.

### Results: batch learning

Two seeds, the v1 splits, 320 training sentences, compared with the supervised model on the same sentences' gold trees (`experiments/v2/results/unsupervised/summary.md`).

| Condition | Train bits (unsup / gold trees) | Chunk types (unsup / gold trees) | Test bits/sentence (unsup / gold trees) | Gen. commission (unsup / gold trees) | Gen. commission in v2.1 | Brackets crossing no gold bracket |
|---|---|---|---|---|---|---|
| small | 3,251 / 3,251 | 3.0 / 3.0 | 9.5 / 9.5 | 0.1% / 0.1% | 0.1% | 75% |
| med | **6,518 / 6,528** | 11.0 / 12.5 | 18.5 / 18.5 | 1.1% / 0.6% | 45.7% | 56% |
| large | **8,031 / 8,356** | 16.0 / 18.5 | **23.6 / 24.0** | **9.2% / 14.5%** | 10.9% | 86% |
| term_low | 5,312 / 5,284 | 13.0 / 12.0 | 15.3 / 15.3 | 0.4% / 0.1% | 25.2% | 53% |
| term_med | **7,677 / 7,712** | 16.5 / 18.0 | 21.8 / 21.9 | **1.7% / 2.9%** | 40.9% | 47% |
| term_high | **10,458 / 10,538** | 12.5 / 15.0 | 30.7 / 30.8 | **0.8% / 2.2%** | 62.4% | 48% |

From sentences alone, the learner now matches the supervised model on every condition. Its code is within 0.6% of the gold-tree grammar's (shorter on MED, LARGE, TERM_MED and TERM_HIGH), and its commission is at most half a point higher (lower on three conditions). Novelty is 91–100% (SMALL 54%: its language is small).

### Results: by day and by night versus batch

Two seeds. The incremental learner perceives the training sentences one at a time and sleeps at 10, 20, 40, 80, 160 and 320 sentences. At each of those points a fresh batch learner sleeps once over the same sentences (`experiments/v2/results/incremental/`, figure `incremental_vs_batch.png`).

Generation commission (incremental / batch):

| Condition | 40 sentences | 80 | 160 | 320 |
|---|---|---|---|---|
| small | 0.5% / 0.5% | 0.3% / 0.3% | 0.2% / 0.2% | 0.1% / 0.1% |
| med | 65.7% / 63.5% | **33.3% / 49.0%** | 14.4% / 14.5% | 1.1% / 1.1% |
| large | 34.4% / 34.4% | 27.3% / 27.3% | 27.0% / 29.2% | 9.2% / 9.2% |
| term_low | **13.9% / 38.3%** | 1.7% / 1.5% | **0.2% / 16.9%** | 0.1% / 0.4% |
| term_med | 60.0% / 60.0% | 38.7% / 40.8% | 5.8% / 7.3% | 0.4% / 1.7% |
| term_high | 69.5% / 69.5% | **31.9% / 60.6%** | **11.3% / 29.4%** | 0.8% / 0.8% |

- **Early nights mostly coincide.** Up to 40 sentences a restart from word classes usually gives the shortest code (the stored analyses win 3 of 20 nights at 20–40 sentences).
- **Then the stored analyses pay.** From 80 sentences on they win 16 of 30 nights (SMALL excluded, where both give the same grammar). At 80 and 160 sentences the incremental learner's training code is shorter or equal in all 10 cells, and its commission is lower in 8, by 16–29 points in 4. Both are equivalent at 320.
- **Perception.** By 160 sentences each day's sentences are parsed almost completely (1.0–1.02 top-level chunks per sentence), at close to the held-out rate in bits.
- **Cost.** A night costs about as much as a batch sleep over the same sentences; the six nights together cost 1.2–1.9× one batch sleep at 320 (timings from a shared machine, so approximate).

### What the experiments established

1. **Description length does not single out linguists' trees.** With gold word classes, the chunk-and-merge search on MED finds a grammar *shorter* than the gold-tree grammar (6,453 vs 6,651 bits) that generates the target language with 0.0% commission, yet shares only 19% of the gold brackets. The learned grammars above match the gold-tree grammars in code and commission, while 47–86% of their brackets cross no gold bracket. Strict MDL identifies the language and leaves its binarization underdetermined. Language-level measures (compression, held-out bits, commission) are the yardstick; bracket agreement is a diagnostic.
2. **The bottleneck was search, not the objective.** Verbs and prepositions share every local context, so the class-bigram code merges them. The code with structure keeps them apart, but greedy search from the bigram classes could not get there. Exact scores, a beam and several starts do.
3. **Consolidation must start from the search's categories.** Re-forming them from blank chunk context lost LARGE's distinctions (47.7% commission at seed 13, against 9.9% when starting from the search's categories).
4. **The plain code guides the search; the full code must decide.** Analyses whose plain codes differ by a bit can consolidate into very different grammars. When the search's tie-breaking changed, MED seed 17's best search result moved from 6,465.5 to 6,466.8 bits, and its final grammar from 6,506 bits and 0.5% commission to 6,771 bits and 13.4%. Each night now consolidates and re-analyses the three best distinct search results and keeps the shortest total code; choosing by the consolidated code before re-analysis was not enough (TERM_HIGH seed 17: 10,499 → 10,805 bits).
5. **Incremental learning must be able to refine categories.** The search can only merge categories, while more data pays for finer ones. A first version that continued only from the stored analyses got stuck: at 10 sentences the shortest grammar has a single category, and no merge undoes that. Letting every night also start over fixed it, and the stored analyses still win whenever they are better.
6. **Earlier attempts that did not fix the verb/preposition merge:**
   - Cobweb re-formation after the search;
   - latent split-and-merge EM on the fixed trees;
   - whole-sentence context bags in the representation (`sentence_bags`);
   - Cobweb re-formation inside every search step: about 60× slower, and trapped in high-PMI non-constituents such as "found the".

### Next for unsupervised learning

- **Split moves (coupled category refinement):** split a class and the chunk categories built on it together, scored by the full code. With splits, a night could continue from the stored analyses instead of repeating the batch search, which would make nights cheaper.
- **Faster nights.** The 12-start search and consolidation (Cobweb replay) now each take about half of a night.
- **Penn Treebank** (WSJ10 with gold tags), then a non-language domain.

## Beyond the synthetic grammars (v2.3, in progress)

Details and figures: [`FRAMEWORK.md`](FRAMEWORK.md), section 11.

### Penn Treebank: WSJ10 with gold tags

Code: `treebank.py`, `experiments/v2/run_treebank.py`. NLTK's public sample of the treebank (about 3,900 WSJ sentences, kept under `data/ptb_sample`, not committed). Tokens are gold tags; punctuation and empty elements are removed; unary chains collapse; sentences of 2–10 tags: 542. Each seed holds out 20% (108 sentences). Unlabelled brackets, ignoring single tags and the whole sentence (Klein & Manning 2002); base phrases are constituents made of tags only.

| Model (WSJ10, seeds 13 and 17) | Bracket omission | Bracket commission | Base-phrase omission | Held-out bits/sentence | Symbols | Chunk types |
|---|---|---|---|---|---|---|
| right-branching | 39.0% | 55.7% | 57.3% | – | – | – |
| left-branching | 82.9% | 87.6% | 75.0% | – | – | – |
| unigram / bigram tag model (add ½) | – | – | – | 33.4 / 27.2 | – | – |
| TRELLIS v2, tags only | 55.3% | 67.6% | 33.1% | 30.1 | 11.5 | 11.5 |
| TRELLIS v2, binarized gold trees | 19.1% | 41.3% | 20.7% | 30.9 | 9.0 | 25.5 |

Training on more sentences (every other sentence of up to 15 or 20 tags; the same held-out WSJ10 sentences; means of seeds 13 and 17):

| Training sentences | Bracket omission (unsup / sup) | Base-phrase omission (unsup / sup) | Held-out bits/sentence (unsup / sup) | Symbols | Chunk types | Learner's code below the gold-tree grammar's | Unsupervised night |
|---|---|---|---|---|---|---|---|
| 434 (≤ 10 tags) | 55.3% / 19.1% | 33.1% / 20.7% | 30.1 / 30.9 | 11.5 | 11.5 | 8–10% | about 2 min |
| about 1,100 (≤ 15 tags) | 51.0% / 17.9% | 24.9% / 18.4% | 29.7 / 31.1 | 16.5 | 24.5 | 14–15% | 16–22 min |
| about 1,900 (≤ 20 tags) | 52.1% / 17.5% | 29.5% / 16.4% | 29.6 / 31.6 | 22.5 | 34.0 | 18% | 50–66 min |

- The unsupervised learner forms base-phrase chunks (noun groups, verb groups, subject–verb pairs) and leaves sentences as forests (about 5 chunks per sentence at 434 sentences, 9 at 1,900, for longer sentences).
- Its grammar, like the supervised one, is a weaker sequence model than tag bigrams. The Dirichlet concentration is not the cause: the supervised grammar's code prefers α = 0.01 to 0.001 (15,439 vs 15,763 bits), with held-out bits unchanged (30.5 vs 30.6).
- **The objective prefers the forests.** At every size the learner's forest grammar is shorter than the grammar of the binarized gold trees, and the gap grows with data (table). Binarization is not the reason: at 434 sentences (seed 13) the learner's analyses take 14,160 bits; right-binarized gold trees 15,763, left-binarized 16,349, and forests of gold base phrases 16,120. Sentence structure does not pay for itself with this grammar family; this is not a search failure.

### Does sentence structure pay on real text?

Code: `experiments/v2/treebank_codes.py`; output: `experiments/v2/results/treebank/structure_codes.md`. Every sentence of the treebank sample (3,901 sentences, 82,356 tags), each description an actual code (Dirichlet rows, concentration chosen by code length). A description that sends a tree is scored two ways: by its *derivation*, a message that names the tree, or by its *total probability*, summed over all trees by the inside algorithm. Bits-back coding achieves the second (Hinton & van Camp 1993; Townsend et al. 2019); the difference is the price of naming one tree. Heads follow Collins (1999); dependency trees are generated head-outward (Klein & Manning 2004).

| Description of the tags | Sends | Bits | Against the tag bigram |
|---|---|---|---|
| tag unigram | – | 353,898 | +24.8% |
| tag bigram | – | 283,596 | – |
| tag trigram | – | 284,510 | +0.3% |
| gold dependency trees, first order | derivation | 370,479 | +30.6% |
| gold dependency trees, first order | total probability | 300,005 | +5.8% |
| gold dependency trees, second order (sibling) | derivation | 356,700 | +25.8% |
| gold dependency trees, second order (sibling) | total probability | 300,793 | +6.1% |
| gold base phrases as headed chunks, Markov sequence of heads | derivation | 310,730 | +9.6% |

- **Tags carry little beyond adjacent pairs at this scale.** Even a tag trigram does not pay (also at 542, 2,023 and 3,751 sentences).
- **Phrase categories are the costliest way to send sentence structure.** On WSJ10 the plain PCFG over the treebank's own labels costs 38% more than the tag bigram (21,108 against 15,284 bits). With TRELLIS's own categories the gold trees of the 434 training sentences cost 24% more than the bigram (15,439 bits at the best concentration, against 12,419). Bits back return only 1.8 bits per sentence for labelled phrase categories, which leave almost no ambiguity, so their total probability is still 32% more.
- **Heads are cheaper.** Gold dependency trees cost 31% more as derivations, but 18 bits per sentence come back, leaving 5.8% (WSJ10: 5.1%). The gap shrinks slowly with data: 7.6% at 10,000 tags, 6.8%, 6.2%, 5.8% at 82,000.
- **Lexical heads add little.** Words given their tags cost 622,578 bits given the previous word and 621,433 given the head word; both together save another 1.1% (615,481).
- **Even with heads and total probability, the code barely prefers linguists' structure.** On WSJ10, EM (40 iterations) from 18 starts:

  | Start | Bits | Against the tag bigram | Heads right |
  |---|---|---|---|
  | random trees (best of 7) | 15,348 | +0.4% | 49.6% |
  | the gold trees' parameters | 15,427 | +0.9% | 71.7% |
  | left-branching chains | 15,540 | +1.7% | 34.1% |
  | right-branching chains | 15,562 | +1.8% | 23.0% |
  | harmonic (Klein & Manning) | 15,596 | +2.0% | 44.1% |
  | uniform | 16,009 | +4.7% | 23.7% |
  | other random trees and random parameters (12) | 15,539–16,153 | +1.7% to +5.7% | 26–56% |

  Shorter codes go with better structure (rank correlation 0.54 between code length and heads missed), and the harmonic start gives 44%, close to Klein & Manning's 43%. But the shortest code found has half the heads right, 79 bits below the solution near the gold trees.
- **Within TRELLIS's own code** (WSJ10, seed 13, full consolidation) the learner's forests remain the shortest analyses: 14,313 bits (14,160 with the night's warm start), against right-branching trees 14,421, left-branching trees 14,494, the forests completed left- or right-branching above their chunks 14,869 and 15,259, and gold trees 15,763.

**Conclusion, in terms of the two hierarchies.** What the hierarchies record decides which structure can pay.

- **Today.** The representation hierarchy keeps a phrase apart from its head (the kind attribute), and the composition hierarchy records a chunk as an unheaded pair. Sentence structure is therefore sent through phrase categories, the costliest description above.
- **The cheapest description records heads in both hierarchies.** In the representation hierarchy, a composite is described by its head's behaviour and its valence (whether it has taken dependents on each side, a refinement of the kind). In the composition hierarchy, a composite is head, dependent and side. The grammar is still read off cuts through both, and each sentence is charged its total probability, which held-out bits already use.
- **Even then, structure does not pay here.** On part-of-speech tags at this scale that description comes within about 1% of adjacency. There, linguistic, half-linguistic and chain-like structures all cost about the same, so description length cannot single out the linguistic one.
- **What does pay are chunks** (base phrases), which is what the learner finds.
- **What sentence structure needs is information tags at this scale do not carry.** Lexical heads are the candidate (de Marcken 1995), but at 82,000 words they save only 1.1%. Heads in both hierarchies and the total-probability code are the changes for real text once it is scaled up to words.

### Chinese characters (IDS)

Code: `characters.py`, `experiments/v2/run_characters.py`. CJKVI IDS (under `data/ids`, not committed). Full decompositions into 270 atomic components and 12 operators, as prefix sequences; 13,297 characters of at most 11 tokens; 2,000 learned and 500 held out (seed 13). The gold tree groups an operator with its first part. Generated characters are checked for well-formedness, attested positions (operator, slot, component), and reality (held-out real characters rediscovered).

| Model (2,000 characters, seed 13) | Held-out bits/character | Structure omission | Well formed | Positions attested | Rediscovered real | Novel and valid | Symbols | Chunk types |
|---|---|---|---|---|---|---|---|---|
| unigram tokens | 44.1 | – | – | – | – | – | – | – |
| bigram tokens | 35.0 | – | 22% | 20% | 3.9% | 13% | – | – |
| TRELLIS v2, IDS structures | **31.9** | **0.0%** | **96%** | 52% | 2.5% | **49%** | 36 | 132 |
| TRELLIS v2, sequences alone | 37.2 | 28.2% | 7% | 4% | 0.7% | 3% | 35 | 53 |

The bigram row samples sequences from a maximum-likelihood token bigram trained on the same characters.

- From the structures, the representation hierarchy forms positional concepts: left-side radicals (92% on the left), top and bottom components, enclosing frames, overlaid strokes; chunk types include a radical in position (`[⿰ 氵]`).
- The supervised grammar compresses better than token bigrams (31.9 vs 35.0 bits per character), parses every held-out structure, and generates well-formed characters 96% of the time. Its positional errors come from one large category that mixes right-side and bottom components.
- **Here the search falls short.** The gold structures give a shorter code than the unsupervised learner's analyses (73,662 against 83,241 bits, 11.5% shorter), unlike the treebank, and also in the plain code the search minimizes (71,703 against 79,740). An unsupervised night on 2,000 characters takes about an hour.
- **The starting categories are the bottleneck, not the search width.** A beam of 16 instead of 4 reaches 79,207 bits (3.7 chunks per character). From the supervised model's 26 token categories the same search builds nearly complete analyses (1.4 chunks per character, 78,905 bits). Bigram word classes cannot see which slot a component fills.
- **Sleeping again does not help.** A second, third and fourth night on the same data, with the continuation of the stored analyses always evaluated, change the code by at most 0.2% (500 characters: 23,519 → 23,477 bits) and leave MED and WSJ10 unchanged. The continuation hits the same wall as the restarts.

### Tried and dropped: an attach move

Folding a recurring top-level pair directly into an existing category (a chunk move followed by a merge, in one step) is exact and cheap to score. It made the search worse: plain-PCFG code at seed 13, MED 6,539 → 8,469 bits, LARGE 7,800 → 9,964 bits, WSJ10 13,697 → 13,707 bits. The beam takes cheap early attachments that produce over-general categories. Removed.

### Speed

- Word classes: exact deltas, about 300× faster (identical merge paths).
- Search: incremental moves, about 5× faster on WSJ20 (identical result).
- Cobweb: each node caches its total sum of squares, about 10% faster with identical hierarchies.
- **Compiled Cobweb.** `cobweb_cu` (cobweb-private, branch `karthik-experimental`) reproduces the pure-Python reference (now `tests/trellis2/reference_cobweb.py`) bit for bit: a Python-compatible Mersenne Twister for the tie-breaking, CPython 3.12's compensated `sum()` where the reference sums (two operator scores tied to the last bit otherwise broke differently), insertion-ordered counts, and no fused multiply-adds. Hierarchies are identical on all six corpora and tree building is 10–20× faster. All 120 numbers of the batch results table reproduce exactly, and a batch run takes 2.2× less time (LARGE: 152 s → 60 s); the rest is now the search, the grammar read-out and the charts. Per-concept code lengths for the evidence cut are computed in the tree (`concept_codes`).

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
| v2.2 (in part) | learning by day and by night ✓ (perceive with the current grammar; consolidate at night from the stored analyses or a restart; the full code chooses among the best search results); split moves; attention-like long-range context |
| v2.3 (in progress) | beyond the paper's corpora: Penn Treebank WSJ10 with gold tags ✓ (and larger training sets ✓; which descriptions make sentence structure pay ✓); Chinese characters ✓; a compiled Cobweb ✓. Open: starting categories that see structure (characters), finer positional concepts, α by description length; for real text, chunks categorized by their head and the total-probability code, with words at scale |
| v2.4 | variable-arity templates, typed relations (two-dimensional composition as relations rather than tokens); chess |

## Reproducing

```
python -m pytest tests/trellis2 -q
python experiments/v2/run_synthetic.py --out experiments/v2/results/main      # ~6 min, 6 workers
python experiments/v2/plot_learning_curves.py experiments/v2/results/main
python experiments/v2/run_unsupervised.py --seeds 13,17   # ~10 min, 6 workers
python experiments/v2/run_incremental.py --seeds 13,17    # ~1 h, 8 workers
python experiments/v2/plot_incremental.py experiments/v2/results/incremental
python experiments/v2/run_search_study.py                  # ~2 min
python experiments/v2/run_treebank.py --train-max-len 10 --out experiments/v2/results/treebank/wsj10   # needs data/ptb_sample
python experiments/v2/treebank_codes.py --em --trellis --out experiments/v2/results/treebank/structure_codes.md   # ~40 min, 9 workers
python experiments/v2/run_characters.py                    # needs data/ids/ids.txt
python docs/figures/make_figures.py                        # figures of FRAMEWORK.md
```

The paper corpora are read from `../trellis_v1/data` (the v1 snapshot), with `data/` as the fallback.
