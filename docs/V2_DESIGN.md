# TRELLIS v2 — design and first results

Status: first implementation, October 2026, on branch `inside-outside`.

- Code: `src/trellis2/`
- Tests: `tests/trellis2/` (49, including brute-force checks of the parser over whole trees and forests, exactness checks of the search, the compiled Cobweb against its Python reference, the chess domain's star and its counted read, the relational characters' inside pass, and, in every domain, that the grammar draws experiences as often as its code says)
- Experiments: `experiments/v2/`
- Background: [the literature behind v2](#background-the-literature-behind-v2), condensed from the October 2026 review (the full report and notes are in the git history, commit `fbe61901`)
- **The framework explained end to end, with figures: [`FRAMEWORK.md`](FRAMEWORK.md)**

## Decisions taken with the user

1. The v1-era rules are relaxed (greedy-only parsing, never feeding parser output back, the generation lock, "no hints"). The parsing and generation loops are rebuilt around the new scheme.
2. Exactly two hierarchies, and each holds **both primitives and composites**: the **representation hierarchy** (how an element behaves) and the **composition hierarchy** (what it is made of). There are no extra trees (for example, for non-constituents). The two hierarchies are the essential core of v2 (stressed again by the user on 2026-10-04): the grammar's categories and chunk types are cuts through them, parsing and generation use that grammar, and learning replays experiences into them. Every change is stated as a change in what the two hierarchies record.
3. Simplicity over patches. One probabilistic model does parsing, generation and description-length scoring, so generation samples from the very distribution the parser and the code lengths use. There are no pools, filters or fallbacks.
4. The in-house Cobweb-MDL variant is not used (not ready). Concept formation is standard Cobweb (category utility). Description length decides only which level of each hierarchy acts as the grammar, and that code lives in `grammar.py` where a Cobweb-MDL variant could replace it.
5. New data types and new fundamental relations are the focus (2026-10-04). The Penn Treebank is not a priority. On real English the test is whether the grammar generates coherent language, on simple language first. For chess, the context window is the user's star: the eight queen rays at any distance, plus the eight knight jumps.

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

### Domains: one interface (`memory.py`, `model.py`)

A domain says what an experience is, and nothing else. Its `Memory` subclass records an analysed experience element by element (`add`), names the relations that join two parts (`relations`), says what the representation hierarchy sees of an element besides its chunk context (`surface`: the context window), codes the top level in the domain's reading order (`top_level_nats`, `layout_nats`, `top_level_tables`), and draws and codes an experience with the grammar (`sample`, `log_prob`). For experiences that arrive without analyses, a `Learner` subclass supplies the structure search of the night (`sleep`) and the analysis of a new experience (`analyse`). Sentences use `Memory` and `UnsupervisedLearner`, characters `CharacterMemory` (their structure is given), chess positions `BoardMemory` and `ChessLearner`. Chunk context, both hierarchies, the cuts, the grammar and its code, and consolidation are shared, and every experiment goes through the same calls: `learn`, `consolidate`, `generate`, `log_prob` with analyses; `observe`, `sleep`, `generate`, `log_prob` without. FRAMEWORK.md, section 13, tabulates the three domains.

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
- **The read's context of a rule choice** (2026-10-05). Where the domain's read gives each element a context (a sentence: the word before the element; `Memory.contexts`), the rule class is chosen in its light: `Uc[A, x, c]` = (n(A, x, c) + β U[A, c]) / (n(A, x) + β), coded prequentially in learning order (`mdl.backoff_code`). The composition cut is searched with and without the context, and description length keeps the shorter code and the weight β ∈ {1, 4, 16} (2 bits for the choice). The context of a span is the input word before it, so the chart stays exact (brute-force tests with contexts).

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
| term_low | 0.0% | 5.7% | 0.5% ± 0.5 | 6.2% | 90% |
| term_med | 0.0% | 7.0% | 2.4% ± 1.1 | 4.7% | 98% |
| term_high | 0.0% | 9.7% | 2.7% ± 1.7 | 3.4% | 100% |

Learning curves: `experiments/v2/results/main/learning_curves.png`. Exploratory sweeps behind the table above: `experiments/v2/results/sweeps/`.

- **Parsing** is at 99–100% from about 40 sentences in every condition.
- **Generation** improves steadily with data. Early on it is worse than v1 at the same small n, because description length favours very general grammars when evidence is scarce.
- **LARGE** is the open case: relative clauses are rare, so they stay merged with adjective phrases at 320 sentences. Coding each rule choice in the light of the word before it would halve its commission (7.6%), but 320 sentences do not pay for that context, and description length leaves it out.

Parse commission (1 − bracket precision) equals omission here, because every parse is a complete binary tree.

## Unsupervised learning (v2.1–v2.2)

Code: `unsupervised.py`, `mdl_search.py`, `mdl.py`.

### Objective: an actual message length

The learner minimizes the bits needed to transmit the training sentences.

- **The code.** Each analysis (derivation) is sent event by event with the Dirichlet-multinomial predictive of each grammar table. This is the Bayesian mixture code that arithmetic coding achieves, a prequential code that does not depend on order. The grammar's size (number of symbols and rule classes) is sent with Elias codes.
- **The split.** The total divides into *data bits* (the cost under the best-fitting parameters) and *model bits* (the remainder: the price of learning the parameters).
- **No thresholds.** A chunk type exists only if it pays for its definition. "Minimize chunks while preserving performance" is therefore the objective itself, not a heuristic.

### Partial analyses

A sentence may be a forest of top-level chunks: GRIDS-style partial parses, and the paper's "graceful failure". The top level codes a sentence as **one tree or a forest of two or more pieces** (2026-10-04): a two-outcome row for which; the root's symbol from S when it is one tree; each piece's symbol from S<sub>piece</sub>, and after the second piece a stop-or-continue row, when it is a forest. The representation instance marks the difference too: the spine above a whole sentence's root records ROOT, and above a piece of a forest it records nothing (the chunk that would join the pieces is unknown).

The first version had one symbol row for every top-level chunk and one stop row. On real text that made the start category hold whole sentences and leftover pieces alike (on 2,500 TinyStories sentences: 1,543 whole-sentence roots and 2,223 pieces, 919 of them single words such as *together*), so the grammar generated pieces as sentences. Under one row, telling the two apart costs as many bits in the symbol row as it saves in the rule rows, so description length never split the category. With two rows the split pays, and refitting the same analyses separated them exactly: held-out code 17.1 → 15.7 bits per sentence, and the sentences the grammar derives whole are 76% real instead of fragments ([Simple English](#simple-english-can-the-grammar-generate-coherent-sentences)). The ROOT marker alone did not split the category; the two-row code alone almost did (10 pieces stayed with the roots).

When every sentence is one tree, the code is the previous one term for term (the mode row has the old stop row's counts; the piece rows are empty), so supervised results are unchanged; unsupervised synthetic runs match the previous ones in five of six conditions and are slightly shorter on LARGE (8,090 → 8,084 bits; 23.8 → 23.6 held-out bits per sentence). The structure search keeps its own unigram top level, which is the pressure towards chunks.

Inside-outside, Viterbi and sampling all handle both modes, and brute-force tests cover them (`tests/trellis2/test_chart.py`). Generation can be conditioned on one tree (`generate(..., whole_only=True)`): the grammar's own sentences.

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
| med | **6,507 / 6,528** | 12.5 / 12.5 | 18.5 / 18.5 | 0.6% / 0.6% | 45.7% | 60% |
| large | **8,029 / 8,356** | 14.5 / 18.5 | **23.5 / 24.0** | **9.0% / 14.5%** | 10.9% | 86% |
| term_low | 5,311 / 5,284 | 12.0 / 12.0 | 15.3 / 15.3 | 0.4% / 0.1% | 25.2% | 53% |
| term_med | **7,677 / 7,712** | 16.5 / 18.0 | 21.8 / 21.9 | **1.7% / 2.9%** | 40.9% | 47% |
| term_high | **10,458 / 10,538** | 12.5 / 15.0 | 30.7 / 30.8 | **0.9% / 2.2%** | 62.4% | 48% |

From sentences alone, the learner now matches the supervised model on every condition. Its code is within 0.6% of the gold-tree grammar's (shorter on MED, LARGE, TERM_MED and TERM_HIGH), and its commission is at most half a point higher (lower on three conditions). Novelty is 91–100% (SMALL 54%: its language is small).

### Results: by day and by night versus batch

Two seeds. The incremental learner perceives the training sentences one at a time and sleeps at 10, 20, 40, 80, 160 and 320 sentences. At each of those points a fresh batch learner sleeps once over the same sentences (`experiments/v2/results/incremental/`, figure `incremental_vs_batch.png`).

Generation commission (incremental / batch):

| Condition | 40 sentences | 80 | 160 | 320 |
|---|---|---|---|---|
| small | 0.5% / 0.5% | 0.3% / 0.3% | 0.2% / 0.2% | 0.1% / 0.1% |
| med | 60.9% / 60.9% | **33.7% / 43.2%** | 16.2% / 15.2% | 0.6% / 0.6% |
| large | 31.4% / 31.4% | 26.7% / 28.5% | 18.8% / 20.1% | 8.5% / 9.0% |
| term_low | 25.9% / 25.9% | 1.6% / 3.1% | **0.2% / 5.2%** | 0.4% / 0.4% |
| term_med | 59.6% / 59.6% | 38.6% / 45.0% | 5.8% / 8.0% | 0.4% / 1.7% |
| term_high | 69.2% / 69.2% | **31.9% / 76.4%** | **5.7% / 23.8%** | 0.7% / 0.9% |

(Rerun 2026-10-05 with the read's context of a rule choice available: description length takes it in a few of the small runs, which moves some of the 10–40-sentence cells; at 320 nothing changes.)

- **Early nights nearly coincide.** Up to 40 sentences a restart from word classes almost always gives the shortest code (the stored analyses win one of the 20 nights at 20–40 sentences, LARGE at 20 sentences with seed 17), so the two learners are identical there but for that night.
- **Then the stored analyses pay.** From 80 sentences on they win 16 of 30 nights (SMALL excluded, where both give the same grammar). At 80 and 160 sentences the incremental learner's training code is shorter or equal in 9 of 10 cells, and its commission is lower in 9, by 5–45 points in 4. Both are equivalent at 320.
- **Perception.** By 160 sentences each day's sentences are parsed almost completely (1.00–1.01 top-level chunks per sentence), at close to the held-out rate in bits.
- **Cost.** The six nights together cost 1.5–2.6× one batch sleep at 320 (timings from a machine running other experiments at the same time, so approximate).

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
| TRELLIS v2, tags only | 48.3% | 62.5% | 26.2% | 27.2 | 6.5 | 4.0 |
| TRELLIS v2, binarized gold trees | 16.0% | 39.1% | 15.7% | 26.6 | 9.0 | 24.5 |

(Each rule choice in the light of the tag before it, chosen by description length, 2026-10-05; before: tags only 55.3% / 67.6% / 33.1% / 30.1 bits, gold trees 19.1% / 41.3% / 20.7% / 30.9 bits.)

Training on more sentences (every other sentence of up to 15 or 20 tags; the same held-out WSJ10 sentences; means of seeds 13 and 17):

| Training sentences | Bracket omission (unsup / sup) | Base-phrase omission (unsup / sup) | Held-out bits/sentence (unsup / sup) | Symbols | Chunk types | Learner's code below the gold-tree grammar's | Unsupervised night |
|---|---|---|---|---|---|---|---|
| 434 (≤ 10 tags) | 48.3% / 16.0% | 26.2% / 15.7% | 27.2 / 26.6 | 6.5 | 4.0 | 8% | about 2–3 min |
| about 1,100 (≤ 15 tags) | 51.9% / 16.4% | 28.2% / 15.2% | 26.7 / 26.8 | 12.5 | 11.0 | 9–10% | 28–33 min |
| about 1,900 (≤ 20 tags) | 52.3% / 15.8% | 27.5% / 15.2% | 27.1 / 27.3 | 25.0 | 35.0 | 11–12% | 85–117 min |

(Tag bigram, held out: 27.2, 27.1, 27.3 bits. Before the read's context: omission 55.3 / 51.0 / 52.1% unsupervised and 19.1 / 17.9 / 17.5% supervised; held out 30.1 / 29.7 / 29.6 and 30.9 / 31.1 / 31.6 bits; the learner's code 8–10%, 14–15% and 18% below the gold-tree grammar's. Night times are from a machine running other experiments.)

- The unsupervised learner forms base-phrase chunks (noun groups, verb groups, subject–verb pairs) and leaves sentences as forests (about 5 chunks per sentence at 434 sentences, 9 at 1,900, for longer sentences).
- With each rule choice in the light of the tag before it, its grammar codes held-out sentences as compactly as a tag bigram, and the supervised one too, at every size; without that context both were weaker sequence models than tag bigrams (and the Dirichlet concentration was not the cause: the supervised grammar's code preferred α = 0.01 to 0.001, 15,439 vs 15,763 bits, with held-out bits unchanged, 30.5 vs 30.6).
- **The objective prefers the forests.** At every size the learner's forest grammar is shorter than the grammar of the binarized gold trees, and the gap grows with data (table; with the read's context the gap is smaller, 8–12%). Binarization is not the reason: at 434 sentences (seed 13) the learner's analyses take 14,160 bits; right-binarized gold trees 15,763, left-binarized 16,349, and forests of gold base phrases 16,120. Sentence structure does not pay for itself with this grammar family; this is not a search failure.

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

Code: `characters.py`, `experiments/v2/run_characters.py`. CJKVI IDS (under `data/ids`, not committed). Full decompositions into 270 atomic components and 12 operators; 13,297 characters of at most 11 tokens; 2,000 learned and 500 held out (seed 13). A character is learned in one of two forms: as its prefix sequence with a gold tree that groups an operator with its first part (operators as tokens), or as a relational tree whose relations are the operators (below). Generated characters are checked for well-formedness, attested positions (operator, slot, component), and reality (held-out real characters rediscovered).

| Model (2,000 characters, seed 13) | Held-out bits/character | Structure omission | Well formed | Positions attested | Rediscovered real | Novel and valid | Symbols | Chunk types |
|---|---|---|---|---|---|---|---|---|
| unigram tokens | 44.1 | – | – | – | – | – | – | – |
| bigram tokens | 35.0 | – | 22% | 20% | 3.9% | 13% | – | – |
| TRELLIS v2, IDS structures, operators as relations | 30.3 | – (given) | **100%** | **84%** | **5.3%** | **77%** | 21 | 280 |
| TRELLIS v2, IDS structures, operators as tokens | **27.2** | 0.0% | 98% | 62% | 5.2% | 53% | 27 | 76 |
| TRELLIS v2, sequences alone | 30.5 | 33.8% | 15% | 12% | 1.3% | 10% | 39 | 63 |

(The two token-sequence models are sequences, so each rule choice is made in the light of the token before it, which description length takes (2026-10-05); before, operators as tokens: 31.9 bits, 96% well formed, 52% attested, 2.5% rediscovered; sequences alone: 36.8 bits, 31.7% omission, 8%, 6%. The relational model has no such context.)

The bigram row samples sequences from a maximum-likelihood token bigram trained on the same characters.

**Operators as relations** (2026-10-04). Written as tokens, an operator is a word of the sequence, and whether a part goes on the right or at the bottom is decided by an operator two or more tokens back, out of reach of the part's description. With operators as tokens, one large category mixed right-side and bottom components, and only half of the generated characters placed every component where a real character does. Read as a relational tree, a character is parts joined by relations, as pieces are on a board: 湖 = [氵 ⿰ [古 ⿰ 月]] (`CharacterMemory`). The operator is the relation of a composite and an entry of the grammar's relation table, and the representation hierarchy sees each part's **slot**: the operator that places it and which part it is (`⿰:0`, a left part). A three-part operator is written as two joins of its two-part counterpart, which lays the parts out the same way (⿲ A B C = ⿰ A ⿰ B C), so every generated tree is a well-formed character; real characters are compared in the same form. The held-out code is the inside pass over each known structure (`CharacterMemory.log_prob`). It is comparable with the token models' codes because a prefix sequence and its structure determine each other.

- **Categories become classes of position.** Top components (100% on top), bottom components (88%), left-side radicals (氵 木 亻 扌 言, 100%), right-side components (100%), overlaid strokes, upper-left frames (广 尸 厂 疒, 99%), and enclosed parts (100%).
- **Coherence.** 84% of generated characters place every component where some real character places it (52% with operators as tokens, 62% once those read each token in the light of the one before), every one is well formed, 5.3% are held-out real characters rediscovered, and 77% are novel and valid. The code was also shorter, 30.3 bits per held-out character against 31.9; with the read's context the token model codes shortest (27.2) but places components far worse.
- **What the representation hierarchy should see** (each variant re-learned on the same 2,000 characters): slot, first and last component, and kind give 83.8% attested and 30.3 bits. Adding the nearest component across the join lowers attested placement to 75.1% (categories drift toward what stands beside a part); adding the parent's slot lowers it to 34.4% (14 categories that mix positions). Slot and kind alone give 84.0% and 31.0 bits; slot, first and last 82.2% and 30.7 bits.
- **The remaining misplacements** come from the categories of composite parts (two components already joined), which mix slots.
- **More data makes it more coherent.** Learned from 6,000 characters (`--modes relational --train 6000`, `results/characters_6000`), 93.0% of generated characters place every component where real characters do, and held-out characters take 29.4 bits (token bigram 32.8); from 10,000, 92.8% and 28.3 bits (bigram 31.8), with 32 categories and 742 chunk types. Fewer real characters remain to be rediscovered as more are learned (2.9% and 1.6% of samples).
- **No read context for a character** (2026-10-05). Read in its IDS prefix order, with each rule choice in the light of the token written before it (a first part's operator, or the first part's last component), description length takes the context and the held-out code shortens (2,000 characters: 30.3 → 27.1 bits; 10,000: 28.3 → 24.6), but at scale fewer generated characters place every component where a real character does (10,000: 92.8% → 85.4%) and fewer chunk types pay (742 → 329). Given each part's slot as the context instead: 27.5 bits and 90.0% at 10,000 (14 symbols at 2,000, against 21, and 76.9%). The context takes over what categories and chunks did, and in rare contexts generation falls back on coarser categories. A character is parts placed in space, not a sequence, so `CharacterMemory` gives no context; the token-sequence models of characters are sequences and get the previous token, as sentences do.
- **Here the search falls short.** The gold structures give a shorter code than the unsupervised learner's analyses (73,662 against 82,159 bits, 10.3% shorter; 83,241 with the previous top-level code, under which the diagnostics below were run), unlike the treebank, and also in the plain code the search minimizes (71,703 against 79,740). An unsupervised night on 2,000 characters takes about an hour. (With operators as relations, the same structures take 69,014 bits.)
- **The starting categories are the bottleneck, not the search width.** A beam of 16 instead of 4 reaches 79,207 bits (3.7 chunks per character). From the supervised model's 26 token categories the same search builds nearly complete analyses (1.4 chunks per character, 78,905 bits). Bigram word classes cannot see which slot a component fills.
- **Sleeping again does not help.** A second, third and fourth night on the same data, with the continuation of the stored analyses always evaluated, change the code by at most 0.2% (500 characters: 23,519 → 23,477 bits) and leave MED and WSJ10 unchanged. The continuation hits the same wall as the restarts.
- **Starting categories read off the representation hierarchy do not help either.** Two variants were tried (2,000 characters, seed 13; the night itself reproduces 83,241 bits).
  - *Restart from the hierarchy's categories.* After the night, each token occurrence takes the category the representation hierarchy gives it, which sees the chunk context. The search restarts from flat sequences in those 31 categories. It reaches 83,244 bits in the plain code and 84,862 after consolidation, both worse than the night (79,740 and 83,241). The hierarchy can see a component's slot only where the analyses already hold the operator-plus-part chunks, and the night's analyses hold too few of them (4.2 chunks per character): a chicken-and-egg problem.
  - *Finer starts on the word-class merge path* (greedy search). Raw tokens reach 90,607 bits, 200 classes 87,487, 150 classes 83,309, 110 classes 80,679, and 82 classes 80,132. The most-merged partitions remain the best starts.
- **Even the supervised model's categories leave the search 10% above the gold structures** (78,905 against 71,703 bits). So the gap is mostly in the search.
- **Split moves, in two simple forms, do not open it either.** Both were alternated with the greedy chunk-and-merge search, each split scored exactly by the plain code. One splits a category's occurrences as the left or right part of a given chunk category; the other splits its top-level occurrences right after a given category. Two splits pay, 0.7% together (81,308 → 80,728 bits), and the search finds no new chunk after them (5.4 top-level chunks per character before and after). A component's slot depends on the operator several tokens back, and one category refined at a time does not reach it. Scoring each of the ten most promising splits by the code after a search helps a little: after eight rounds the search builds a few more chunks, and then no split pays (81,308 → 80,163 bits, 5.40 → 5.24 chunks per character). That is still above the night's own result (79,740) and far from the supervised categories (78,905, 1.4 chunks per character). The search sits in an optimum that single refinements barely move. What the supervised categories show is that a coordinated change of many categories at once can leave it, so the next attempt is a search over several moves at a time, or restarts from perturbed analyses.

### Simple English: can the grammar generate coherent sentences?

Code: `stories.py`, `experiments/v2/run_stories.py`. The question is whether a grammar learned from real sentences alone generates coherent ones, measured without a target grammar.

- A generated sentence is **real** if it occurs, word for word, somewhere among the 497,000 sentences of TinyStories. It is **new** if it is not among the training sentences. Both are also taken among generated sentences of the training sentences' length (3–5 words): a single word such as *mom* or *together* is new and real without being a coherent sentence. (The first report of these results missed this: 53 of TRELLIS v2's 83 "new and real" sentences out of 1,000 were single words, and its lead over the n-gram models disappeared within the training length.)
- Coherence is also checked one level down: the share of generated word pairs and triples that occur in TinyStories, and the share of the multi-word chunks inside generated sentences that do.
- **Consistency** is whether a generated sentence is perceived again (Viterbi) with the analysis it was generated from.

**Children's books from Project Gutenberg were too sparse.**

| Corpus (unsupervised, one night) | Sentences | Words | Uses per word | Result |
|---|---|---|---|---|
| Grimms' Fairy Tales: sentences of 3–10 words over the 500 most frequent words | 709 | 426 | about 10 | 3 categories, no whole sentence; held-out code worse than a word unigram (50.2 against 49.3 bits per sentence) |
| McGuffey readers 1–5, two Aesop collections, Grimm, Alice, Oz: clauses of 3–8 words over 300 words | 4,375 | 299 | about 72 | 18 categories, 10% whole sentences; held-out 32.8 bits against a word bigram's 30.0; generations are strings of independent chunks |

On the clause corpus, consolidation lengthened the code from the search's 142.7K to 158.1K bits. The cut through the representation hierarchy costs 143.7K, so the categories are not what lengthens it. The factored rule layer is: a rule class draws its two parts independently, and English agreement and selection make them dependent. A Markov top level and sentence templates were also tried over the search's analyses, and neither made the generations coherent. Pairs and triples of generated words stayed at about 46–55% and 4–9% found in training. The categories themselves were too coarse to generate from.

**TinyStories** (Eldan & Li 2023) are short stories written with the words a three- or four-year-old knows, designed to test whether small models produce coherent English (CDLA-Sharing-1.0; the validation file, 4.4 million words, is enough). Its sentences of 3–5 words over its 100 most frequent words number 9,955. Examples: *lily was very happy*, *he wanted to help*, *tom and sam are sad*, *max wanted to help lily*. A single greedy search there analyses half of them as whole sentences (`[[tim [and sam]] [were sad]]`, `[she [was [not happy]]]`); with 250 words and 8-word sentences, it is 7%.

**One tree or a forest of pieces** (2026-10-04; [Partial analyses](#partial-analyses)). The first grammar coded every top-level chunk from one row, so one start category held whole sentences and the pieces of forests alike, and 17% of its samples were single words. With whole sentences and pieces coded apart, and ROOT marked only above a whole sentence's root:

| 2,500 TinyStories sentences of 3–5 words over 100 words (500 held out; `results/stories_2500`) | TRELLIS v2, its own sentences | TRELLIS v2, all samples | Word bigram | Word trigram | Before: one top-level row (all samples) |
|---|---|---|---|---|---|
| held-out bits per sentence | 16.0 | 16.0 | 13.2 | 13.0 | 17.1 |
| generated sentences 3–5 words long | **94%** | 85% | 71% | 93% | 47% |
| real, among those of 3–5 words | 75% | 55% | 63% | 86% | 50% |
| new and real, among those of 3–5 words | 4.6% | 4.2% | 4.0% | 3.4% | 4.2% |
| new | 33% | 57% | 58% | 23% | 78% |
| generated word triples found in TinyStories | 85% | 54% | 84% | 100% | 63% |
| chunks inside generated sentences found in TinyStories | 92% | 88% | – | – | 92% |
| perceived with the analysis they were generated from | 100% | 99% | – | – | 99.7% |

The grammar has 34 categories and 50 chunk types, and 65% of training sentences are analysed as one tree (62% before). The categories are grammatical: subjects, subject plus copula, copulas and verbs, predicate phrases, whole sentences, and the pieces of forests. The grammar's own sentences (those it derives as one tree, `generate(..., whole_only=True)`) are English of the training length, real more often than a bigram's, about as often new and real as either n-gram model's; a trigram's are real more often because it mostly repeats its training sentences. All samples, forests included, still generate strings of pieces: a forest's pieces are drawn independently. Coding each piece given the previous piece's category (branch `markov-forest-pieces`: a Markov code over a forest's pieces, with brute-force tests) was tried on the same analyses: the code shortens (43,663 → 42,008 bits; held out 16.02 → 15.03 bits per sentence) and the pieces split by role (clauses, noun phrases, *can we …*, single words), but forests sampled from it are no more often real (real among 3–5-word samples 56.5% → 49.5%), and the grammar's own sentences are slightly less often real (75.5% → 72.8%). It is kept aside. Two words of context on each side instead of one (refit on the same analyses) left the grammar's own sentences unchanged (75% real within the length) and lengthened the training code by 292 bits.

**With twice the sentences** (5,000 learned, the same 500 held out; `results/stories`; a night of 4.6 hours):

| TinyStories, 3–5 words over 100 words | 2,500 sentences | 5,000 sentences | Word bigram (5,000) | Word trigram (5,000) |
|---|---|---|---|---|
| held-out bits per sentence | 16.0 | **15.0** | 12.8 | 12.0 |
| training sentences analysed as one tree | 65% | **73%** | – | – |
| categories / chunk types | 34 / 50 | 58 / 87 | – | – |
| own sentences 3–5 words long | 94% | 97% | 73% | 92% |
| own sentences real, among those of 3–5 words | **75%** | 64% | 65% | 87% |
| new and real, among those of 3–5 words | 4.6% | 1.2% | 1.1% | 1.2% |
| word triples of own sentences found in TinyStories | 85% | 75% | 85% | 100% |

As a model the grammar improves: a shorter held-out code, more sentences derived whole, more categories and chunk types. But its own sentences are real less often, at a bigram's level, which breaks the pattern of chess and characters, where more data made generations more coherent. In the run's final grammar two categories mix words which do not combine alike: one holds intensifiers and determiners (*very, so, not, a, big, his, their*), another adjectives and nouns (*happy, sad, fun, dog, friend, toy*), and since a rule class draws its two parts independently, the grammar writes *she was so dog*. That mix is not forced by the description: refitting the same analyses with one more round of re-analysis keeps those words apart with one word of context, and the grammar's own sentences rise to 66.5% real within the training length. More context does not help further (refit on the same analyses: two words on each side 66.7%, three 59.4%, bags of the whole sentence 63.8%; held-out 14.94, 15.11, 15.10 and 15.00 bits). The errors that remain are at the level of phrases: the grammar now reads sentences as subject and predicate (*[she [was [very happy]]]*), and its predicate categories also hold noun phrases and pieces of coordinations (*they [the mom]*, *they [and sam were happy]*). "New and real" falls to about 1% for every model, because 5,000 training sentences hold much of this small language.

**What limits the grammar's own sentences** (2026-10-05; refits of the 2,500-sentence analyses of `results/stories_2500`, each with up to five rounds of re-analysis). A fairer measure of coherence than *real* is whether every word triple of a sentence, its two edges included, occurs in TinyStories: *lily is very happy* is good English that TinyStories happens not to contain. 83.5% of the grammar's own sentences of 3–5 words pass, as do 83.1% of a word bigram's (a trigram's pass by construction); at 5,000 sentences, 71.0% against 81.6%. Nearly every sentence that fails uses only (parent, left part, right part) category triples seen in training (506 of 516), so the errors are inside categories, and they are of two kinds: rare words lumped with frequent ones of another kind (*mom was a happy*, *tim can so happy*) and agreement (*she want to play*). The training sentences are dominated by *X was very happy*: *happy* occurs 1,012 times and *very* 507, *a* 94 and *dog* 68. A rare word lumped into a frequent word's category costs few bits, and keeping *a dog* apart from *very happy* saves too few to pay for a category, so description length lumps them; generated sentences then put *a* where *very* goes about one time in ten. Nothing tried changes this:

| Refit on the same analyses | Training bits | Held-out bits per sentence | Own sentences real (3–5 words) | Every triple in TinyStories |
|---|---|---|---|---|
| as learned (34 categories) | 43,663 | 16.02 | 74.2% | 83.5% |
| no model merging (50 categories) | 43,914 | 15.92 | 74.4% | 84.0% |
| α = 0.01 | 43,810 | 16.30 | 60.9% | 70.7% |
| one granularity of chunk context | 43,875 | 15.98 | 70.9% | 81.2% |
| a one-level spine | 42,074 | 15.48 | 74.4% | 84.4% |
| both (the shortest code) | **41,552** | **15.12** | 73.5% | 83.5% |
| neighbours also by their word's category | 43,031 | 15.67 | 68.8% | 79.4% |
| each element also by the categories its word or phrase takes everywhere | 43,581 | 15.98 | 71.0% | 80.8% |
| the same, with a one-level spine and one granularity | 41,933 | 15.27 | **77.1%** | **86.3%** |
| a chunk's two parts drawn jointly in generation (drawn independently in the same test: 74.8%, 84.1%) | – | – | 74.2% | 82.9% |
| consolidated from right-branching trees instead of the search's analyses | 44,262 | 15.87 | 42.4% | 44.5% |
| from left-branching trees | 45,455 | 16.41 | 36.1% | 39.5% |

The search's analyses matter: consolidated from fixed tree shapes, which re-analysis does not improve, the grammar's own sentences are far less coherent, even though right-branching trees read *[she [was [very happy]]]* as a linguist would. The representation hierarchy itself groups *very, so, not, a* and *happy, sad, dog, cat* even at its evidence cut (61 categories): two occurrences in the same slot (*was [very happy]*, *was [a dog]*) are described alike, and the words that would tell them apart are rare. A lighter chunk context gives the shortest code (5% shorter) without changing coherence. Only one variant moves coherence: a lighter chunk context together with a bag, on every element, of the categories its word or phrase takes across all its occurrences (what it does elsewhere). It separates adjectives from nouns (*happy, sad, fun, big* apart from *dog, cat*), though *a* stays with *very*, and its own sentences are real 77.1% of the time. It does not carry over: on characters the same bag lowers attested placements (82.8% against 83.8%; 74.1% with a one-level spine), and chunk context written as bags, as the chess star is, does not help them either (83.6%). Nor would description length choose it: on characters the variant with the shortest code (the bag with a one-level spine, 68,228 bits against 69,014) places components worst, and on English the shortest code is not the most coherent variant. The representation stays as it is.

**The read's context: each rule choice in the light of the word before it** (2026-10-05). The errors above sit inside categories, where a frequent and a rare word of different kinds share one category. What tells them apart in a sentence is what was just read: after *a* a noun phrase is made one way, after *very* another. In the chess read, each question was answered in the light of what the read had already placed; here each rule choice is made in the light of the word before the element, which the chart knows for every span. The rule choices given the category and that word code 37,885 → 33,506 bits (Dirichlet rows, the same analyses), so the context pays for itself; sampling with it (without refitting) raises the grammar's own sentences with every triple attested from 84% to 95%, and among new sentences from 46% to 72%, at about the same number of new coherent sentences (the incoherent ones go). Built in (`Uc`, the back-off code, the context-aware chart and sampler) and chosen by description length:

| Refit of the 2,500-sentence analyses | Training bits | Held-out bits per sentence | Own sentences real (3–5 words) | Every triple in TinyStories | All samples, every triple |
|---|---|---|---|---|---|
| without the context | 43,663 | 16.02 | 74.2% | 83.5% | 61.7% |
| with it, weight chosen by description length | **37,461** | **12.44** | 83.5% | 89.9% | **75.9%** |
| learned again from the sentences alone with it (`results/stories_2500`; 59% whole trees, 34 categories, 56 chunk types) | 39,089 | 12.81 | **89.7%** | **92.7%** | 73.3% |
| word bigram / trigram | – | 13.2 / 13.0 | 63.6% / 86.5% | 83.1% / 100% | – |

For the first time the grammar describes held-out English more compactly than the n-gram models, and its own sentences are real more often than a trigram's, though fewer of them are new (15%, against 33% before and 24% for the trigram; new and real 4.0%, against 4.6% before). On the Penn Treebank sample (WSJ10 tags, 434 training sentences) the context is chosen too: from tags alone, bracket omission 55.3% → 48.3%, base-phrase omission 33.1% → 26.2%, held-out 30.1 → 27.2 bits per sentence (the tag bigram's 27.2); from binarized gold trees, omission 19.1% → 16.0% and held-out 30.9 → 26.6 bits, below the tag bigram. On the paper's synthetic corpora, 320 sentences do not pay for the context (it is chosen in one run of 30 at 320 sentences, mostly at 10–40), so their results are unchanged; forced on, it would halve LARGE's generation commission (14.2% → 7.6%, novelty unchanged) at a longer code (26.0 → 28.0 training bits per sentence). Characters and chess have no such context (their reads are a tree from the top, and the board's counts at the top level).

**A larger language** (2026-10-05; `run_stories.py --train 2500 --vocab 250 --max-len 8`, `results/stories_250`). Sentences of 3–8 words over the 250 most frequent words, 2,500 learned and 500 held out, with the read's context. Held out the grammar codes 26.1 bits per sentence, between the word bigram (25.8) and trigram (29.0). Only 10% of training sentences are analysed as one tree (3.7 top-level chunks per sentence), so the grammar's own sentences are the commonest forms (*tim was very scared*, *then something unexpected happened*; mean length 3.8 words): 81% of them are real and 17% new and real (bigram 9%, trigram 14%, among sentences of 3–8 words), and 91% of their word triples occur in TinyStories. All samples, mostly forests of independent pieces, are real 15% of the time (bigram 26%, trigram 55%). Fewer forests remains the open problem.

### Chess: parts joined by typed relations

Code: `chess.py`, `experiments/v2/run_chess.py`. Middlegame positions from the Lichess database of January 2013 (CC0, under `data/chess`): games in which both players are rated at least 1800 and that last at least 40 plies, the position after ply 30. That gives 8,560 positions; 4,000 are learned and 500 held out (seed 13).

**Design, in terms of the two hierarchies.**

- *Elements.* A primitive is a piece; its token is its colour and kind. A composite joins two elements whose anchors see each other, and its anchor is its first part's anchor.
- *Representation hierarchy.* An element's context window is the star: the first piece along each of the eight queen rays, at any distance, and the piece on each of the eight knight squares. The representation instance also holds the anchor piece (whole, and by colour and kind), the element's kind, its square, and the chunk context, as for sentences. The star is written as two bags, rays and jumps, whose values name their direction (`N:wP`).
- *Composition hierarchy.* A composition is (category, relation, category), where the relation is the star direction and distance (`N1`, `E3`, `NNE`). The composition's window is one direction of the star: a chunk joins an element to the first piece it sees along one forward ray, at any distance, or to the piece a forward knight's jump away; looking forward loses nothing, since of two pieces that see each other the later one in the read is always forward of the earlier. Like the star, the window looks past the element's own pieces (2026-10-05), so a chunk can grow along a ray: a pawn chain or a queen behind a bishop. On these positions that changes nothing: no longer chunk pays, and not even two pawns on a diagonal, because the square-by-square read already predicts where pawns stand. `grammar.py` gained a relation table per rule class (`Rel`). Sequences have a single relation, and their codes and grammars are bit-identical to before.
- *Top level.* A board is read square by square (a1 … h8). At every square not covered by an earlier chunk, the read asks of each kind of piece in turn whether a top-level element is anchored there on a piece of that kind, given the square and how many pieces of that kind stand on earlier squares (below); the element's category is drawn given the kind and square of its anchor, and the element from its category. So that a chunk is decoded at its first square, relations point forward in the scan: the N, NE, NW and E rays and the four upward knight jumps, 32 relations in all.

**Two first attempts failed.**

- *An unordered set at the top level.* The first top level coded a position's top-level elements as a set: categories, anchor squares, and a −log₂ k! credit for their order. Joining two elements loses log₂ k ≈ 4.7 bits of that credit, so on 1,000 positions only one chunk paid (black rook beside black king). The scan pays no such price.
- *The star as sixteen attributes.* Then it outweighs what the element is. The categories mixed kinds of piece (white pawns, knights and bishops together), and the consolidated grammar coded positions in 95.8 bits per position, against 86.5 for the plain square code (1,500 positions). As two bags, the star weighs what a sentence element's two neighbours weigh. The categories then separate the kinds of piece, and the grammar is shorter than the plain code: 85.9 bits.

**Results.**

| Bits per position (4,000 positions) | Each square on its own | TRELLIS v2 |
|---|---|---|
| training | 85.05 | 84.42 (search alone: 84.67) |
| held out (500) | 82.75 | 81.93 |

| Chunk type | Count | Main anchors |
|---|---|---|
| `[bP N1 bB]` pawn with its bishop behind it | 2,164 | g6 (fianchetto), e6, b6 |
| `[bR E1 bK]` castled black king | 2,110 | f8 |
| `[wB N1 wP]` bishop with its pawn in front | 1,826 | g2 (fianchetto), d3, b2 |
| `[bR E2 bK]`, `[bK E3 bR]`, `[bK E1 bR]` | 536, 380, 181 | e8, e8, c8 |
| `[wK E3 wR]`, `[wK E1 wR]` | 298, 237 | e1, c1 |

- **The chunks are castling and fianchetto structures.** King and rook appear at every stage of castling, with the distance as part of the relation. Castled on either wing, the rook is beside the king or has moved one square on. Uncastled, the squares between them are cleared. Bishops appear with the pawn that blocks or supports them.
- **Categories.** The representation hierarchy forms one category per kind of piece and one per chunk type (18 symbols, 19 rule classes).
- **Generation.** 93.0% of the chunks in 1,000 generated positions occur, piece for piece and square for square, in some held-out game. 7.7% of samples are rejected because a chunk would leave the board or land on an occupied square.
- **Whole positions had no global sense.** 38% of generated positions had one king of each colour, the same as when each square is drawn on its own with no chunks (38%), and 4% passed every check below. A square-by-square code that counts nothing cannot keep to one king.
- **Compression.** Chunks shorten the held-out code by 1.0%. At move 15, where pieces stand is most of what can be compressed, and a chunk pays only where pieces depend on each other beyond their squares.

(The table above and the chunk types are the read without context: `experiments/v2/results/chess_plain`.)

**The read learns its context** (2026-10-04, since replaced). A square was read in the light of the pieces on the squares before it through features "at least *m* pieces of kind *k* stand on earlier squares", added greedily while one shortened the code of the training positions. Seven paid on 4,000 positions (a king of each colour, one and two rooks of each colour, seven black pawns): one king each in 96.3% of generated positions, but only 22.9% passed every check (41.5% on 8,000 positions), because no feature about queens or minor pieces paid. A representation hierarchy over "what has been placed so far", cut by description length, compressed as well but grouped contexts by how far the read had got (one king each 46–54%). A stricter check was added then: no more of any kind of piece than a side starts with (one queen, two rooks, …); every one of the 8,560 real positions passes all five checks.

**The read counts material** (2026-10-05; `BoardMemory`, `BoardSearch` in `chess.py`). A square's content was one thirteen-way outcome (empty, or which piece), and predicting it from what is already on the board needs all twelve counts at once: too many contexts to learn, so features had to be chosen and most counts were left out. The read now asks twelve yes-or-no questions per square instead, one per kind of piece in a fixed order: is a top-level element anchored here on a piece of this kind? Each question needs one count, that of its own kind on earlier squares (0 to 10), and gets it. The code of the same events is the same whichever order the kinds are asked in (73.2–73.4 held-out bits per position across orders, read alone). The element's category is then drawn given the kind and square of its anchor (`T`), and the element from its category, its anchor's kind given (normalised by the probability that a derivation from that category is anchored on that kind). Every quantity the read conditions on is on the board, so the code of a known analysis stays exact, and the search scores a chunk move from the squares it changes: the second part's square is no longer read, and the anchor's label changes.

| 4,000 positions learned, 500 held out (`results/chess`) | Squares on their own (`results/chess_plain`) | Features chosen by description length (before) | **The read counts** |
|---|---|---|---|
| held-out bits per position | 81.93 | 76.47 | **73.33** |
| one king each | 37.7% | 96.3% | **100.0%** |
| at most 8 pawns each / at most 16 pieces each | 73.1% / 79.9% | 82.7% / 84.8% | **100.0% / 100.0%** |
| no more of any kind than at the start | 7.7% | 22.9% | **100.0%** |
| passes every check | 4.3% | 22.9% | **100.0%** |
| generated chunks found, piece for piece, in a held-out game | 93.0% | 97.0% | 98.7% |
| chunk types | 12 (castling, fianchettos) | 3 (fianchettos) | 1 (blocked pawns) |
| model bits | 10,604 | 17,591 | 15,369 |

- **The chunk that pays joins two colours.** `[wP N1 bP]`, a white pawn with a black pawn on the square in front of it (2,788 times; d4, e4, e5): the pawn rams of closed centres. Castling and fianchettos no longer pay: given how many kings, rooks and bishops are already placed, the read predicts their squares as well as a chunk does.
- **The read alone already generates legal-looking positions** (99.9% pass every check with no chunks); the grammar adds categories, the chunk, and a slightly shorter code (73.33 against 73.47 held-out bits).
- **More data.** On 8,000 positions (560 held out; `run_chess.py --train 8000 --test 560`, `results/chess_8000`), the white fianchetto pays again beside the blocked pawns (`[wP N1 bP]` 5,604 times, `[wB N1 wP]` 2,860 times), the held-out code is 72.41 bits per position (72.62 for the read alone; 74.42 with the features before), and 99.8% of generated positions pass every check (41.5% before).
- **The concentration.** With the counted read, α = 0.001 codes the training positions shortest (76.37 bits per position, against 77.56 at α = 0.01, `results/chess_alpha01`). At α = 0.01 the held-out code is 73.26 bits and no chunk pays; 99.8% of generated positions pass every check.
- **A first version asked about categories, not kinds** (each of the grammar's categories in turn, counting elements of its category): the read then changes with every candidate cut, which made consolidation many times slower, and a category's count is not the count of a kind of piece once categories mix kinds or chunks hold pieces. Asking about the anchor's kind keeps the read fixed while the cuts are searched.

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

## Hypotheses: playing chess from the two hierarchies

The chess grammar describes positions; playing needs a choice of move. The aim is a player whose built-in knowledge is only the rules of movement and one or two "stupid" rules, whose judgement comes from the two hierarchies, and whose model is measured in bits and stays small next to engines and networks. None of this is built yet.

**What the representation already offers.** A piece's star holds the first piece along each queen ray and on each knight square: the pieces it attacks or defends, and the pieces that could attack it along a line or a jump. The read counts what is already on the board, kind by kind. Read square by square without counts, the chunk types are king shelters and fianchettos, two of the stereotyped patterns in Chase & Simon's (1973) recall data; with counts, blocked pawn pairs, a piece of the pawn structures that are the most frequent pattern there.

**Hypotheses, each with its test.**

1. **Typical positions are good positions.** Among the legal moves, play the one after which the position has the shortest code under the grammar. Expected to choose quiet moves well (castling, development, fianchettos) and to be blind to tactics. Test: agreement with the move played in the Lichess games (the PGN already holds the move after each position) by rating band, against a random legal move and against the same rule with squares read on their own. The grammar must then learn positions with either side to move.
2. **A move is a composition.** In the composition hierarchy a move is a composite: the moving piece's category, the move relation (a direction and distance, or a knight jump, the relations the board already has) and what stands on the target square. Its rule class is conditioned on the moving piece's category in the representation hierarchy, which sees the piece in its star. This is the pattern-to-move association of CHUMP (Gobet & Jansen 1994) inside the two hierarchies, with no third net. Learned from (position, move) pairs; play the legal move with the shortest code. Test: move agreement, bits per move, and the Lichess puzzles (CC0) by theme and rating.
3. **Attack and defence are relations.** With "attacks" and "defends" as relation types alongside direction and distance (both read off the star and the pieces' moves), a pawn chain is a run of defences and a pin is a chunk of two parts along one ray. Test: whether pawn chains and pins appear among the chunk types, and whether move agreement improves on tactical puzzles.
4. **One stupid rule: do not hang pieces.** From the star alone: a piece is en prise if an enemy piece of lower value sees it along a ray or jump it moves on and nothing defends it. Filter such moves before applying 1 or 2. Test: puzzle solve rate, and games against a random mover and a one-ply material player.
5. **Space.** The grammar's size is its model bits, reported by every run. Compare agreement and strength against stored size: CHREST's discrimination net of 300,000 nodes for master-level recall (Gobet & Simon 2000) and a 270-million-parameter transformer that plays at grandmaster level without search (Ruoss et al. 2024) bracket the range. A Cobweb hierarchy can hold far more counts than the grammar read off it, so both sizes are reported.

**Measures.** Move agreement (top 1 and top 3) by rating band; bits per move; puzzles solved by theme and rating; results against a random mover and a one-ply material player; size in bits and in stored counts.

**A first test of hypothesis 1** (`experiments/v2/run_chess_play.py`; the 500 held-out positions, White to move after ply 30, 37.7 legal moves on average):

| Rule | Picks the move played | Among its top 3 |
|---|---|---|
| a random legal move | 3.2% | 9.3% |
| capture the most valuable piece, else a random move | 19.4% | – |
| shortest code: chunks and the read's counts | 10.6% | 22.2% |
| shortest code: the read's counts, no chunks | 9.0% | 21.4% |
| shortest code: each square on its own | 8.8% | 22.2% |
| a capture that does not lose material, else the shortest code | 20.7% | – |

Typicality triples the agreement of a random move, but chunks and the read's counts barely change it: the codes of two candidate positions differ mostly by where the moved piece lands, which squares alone already price. (With the earlier read, whose context was chosen features, the shortest code picked the move played 9.0% of the time, and 20.1% with the capture rule.) One stupid capture rule does better on its own, because many moves at this point of a game are recaptures, which a position's code cannot see; together the two reach 20.7%. Seeing tactics is the job of hypotheses 2–4.

## Background: the literature behind v2

This section condenses the October 2026 literature review that preceded v2. The full report and its seven research notes (about 750 KB) are kept in the git history, commit `fbe61901` (`reports/` and `research_notes/`). Other systems' results are quoted in their own measures; v2's are omission and commission. The review's first recommendation, re-scoping the v1-era rules, became decision 1.

| The review recommended | In v2 |
|---|---|
| Inside-outside posteriors and minimum-risk decoding | Yes |
| Chart items scored directly by Cobweb, with a temperature | No: the chart runs over the grammar read off the cuts |
| Constituent-versus-distituent odds from extra trees | No (decision 2) |
| Chunk context at several granularities, from the outside pass | Partly: two granularities and a depth-2 spine, read from stored analyses |
| Prequential description length | Yes: the Dirichlet-multinomial code of the grammar's tables |
| A normalized generator, so that code length sees commission | Yes (decision 3) |
| Per-chunk acceptance tests, a decaying frontier, probation | No: exact global search over chunk and merge moves |
| Masked chunk prediction as the training regime | No |
| Typed relations, with slot order supplied by the domain | Yes: chess (direction and distance) and characters (each IDS operator is the relation joining two parts) |
| Language-level measures first; bracket agreement and a non-crossing check as diagnostics | Yes |

### Parsing with a chart

Inside-outside gives each span the weight of all ways to build it and of all ways the rest of the sentence can surround it; their normalized product is the posterior that the span is a chunk (Baker 1979; Lari & Young 1990). Minimum-risk decoding picks the tree with the most expected correct spans. Each decoder wins on the measure it optimizes (Goodman 1996), and on the WSJ, max-rule-product decoding beat the best single derivation by 1.7 points of bracket agreement (Petrov & Klein 2007). What spans are scored by matters more than how the chart is searched: trained on raw text, inside-outside reached 37% bracketing accuracy against 90% with brackets (Pereira & Schabes 1992), and CCM succeeded by scoring each span's yield and context separately for constituents and non-constituents, or distituents (Klein & Manning 2002).

**v2.** Inside-outside over the factored grammar gives μ(i, j, A), the chunk's strength, in place of v1's recognition threshold. The minimum-risk tree is the parse; perception and unsupervised learning use the Viterbi analysis, which is also the shortest-code analysis. Spans with μ > 0.5 never cross and the chart exposes them, but learning does not use them yet. The rule tables have the low-rank form of tensor-decomposed PCFGs (Yang et al. 2021), which keeps each span at O(n·M). Not built: the review's chart of Cobweb-scored items, its distituent trees (an over-general grammar pays in code length instead), coarse-to-fine pruning, and a greedy parser trained against the chart.

### What a chunk's context should be

In Clark's syntactic concept lattice a category is a closed pair of strings and contexts, and composing two categories concatenates their strings and then closes the result through its contexts (Clark 2010): composition proposes, context categorizes. DIORA computes a span's outside from its parent's outside and its sibling's inside, which unrolls to a spine of sibling chunks (Drozdov et al. 2019). HVM defines a variable as the chunks that share preceding and succeeding chunks, and gets this chunk context by parsing first and counting afterwards (Wu, Thalmann et al. 2025). Abstraction should add to identity, not replace it: Cobweb context coded as leaf concepts did little better than chance, while whole ancestor paths did best (MacLellan et al. 2022).

**v2.** The representation instance holds the neighbouring tokens, the first and last tokens, the element's kind, its children's categories and a depth-2 spine, with chunk attributes written at two granularities (the symbol and a finer node). The spine is read from the stored analyses, and consolidation iterates until the symbols are stable, much as in MacLellan et al.'s multi-pass labelling. Chunk context reaches the parser only through the categories it forms, so the grammar stays context-free and inside-outside stays exact. The spine took MED generation commission from 40% to 8% and LARGE from 46% to 17%; the second granularity took MED from 20% ± 17 to 8% ± 1. Tried and dropped: the element's own composition concept in its representation instance (LARGE commission 27% ± 14) and whole-sentence context bags. Not tried: context from the outside pass, four-level references, seam attributes, and a third hierarchy for long-range dependencies.

### Learning a grammar by description length

SNPR learned grammars by hierarchical chunking plus disjunctive categories under a compression measure (Wolff 1982). GRIDS rebuilt it with create and merge operators, a description-length bias and a beam of three, defined errors of omission and commission, and named an incremental version as the next step (Langley & Stromsten 2000). Bayesian model merging put both operators under one posterior and found that a new chunk often pays only after several merges, so best-first search fails (Stolcke & Omohundro 1994). Heuristics should propose and description length decide (Goldsmith 2001). Brown clustering merges word classes by the mutual information of adjacent classes (Brown et al. 1992), but grouping by mutual information alone prefers [V P] to [P N] (de Marcken 1995).

**v2.** Each night's structure search is GRIDS, SNPR and model merging under one probabilistic code: global chunk (B, C) and merge (A, A′) moves, scored exactly from cached Dirichlet-multinomial rows, with a beam of 4 and Stolcke's lookahead. It starts from the last 12 partitions on the merge path of Brown clustering read as description length, and after the first night also from the stored analyses. Symbols come from a cut search followed by model merging. Re-analysis is hard (Viterbi) EM, which beat soft re-estimation for unsupervised dependency induction (Spitkovsky et al. 2010), kept only if the total code shrinks. There are no thresholds: exact scores of global moves replace the review's per-chunk tests (a compression gain with a mutual-information floor, significance margins, probation, and a decaying frontier as in online adaptor grammars; Zhai et al. 2014). An attach move was tried and dropped (MED 6,539 → 8,469 bits). One departure from the review: it read v1's history as showing that gains come from representation, never from search, but for unsupervised v2 the objective was right and the search was the bottleneck (over 201 searches, code length and commission rank-correlate at 0.71–0.98 per condition).

### Information-theoretic foundations

For multinomials with Dirichlet priors the prequential code, which charges each item its predictive probability given the past, equals the Bayesian marginal likelihood and does not depend on order (Grünwald & Roos 2019). Description length sees commission from positive data only if the code is the distribution that generates. A v1-era two-part MDL selection (met6) cut categories from 16 to 5 while generation commission rose from 18% to 38%; the review traced this to a pool-and-filter generator that defined no coded distribution. The structure function picks the best model within a class whether or not the true model is in it (Vereshchagin & Vitányi 2004), and nothing in a code prefers one of two equally compressive binarizations. Bits-back coding charges a sentence its total probability instead of the cost of naming one tree (Hinton & van Camp 1993).

**v2.** Every grammar table is coded this way (α = 0.001), the grammar's size with Elias codes, and model bits are reported apart from data bits. One normalized model parses, generates and sets code lengths (decision 3). Category utility still forms the concepts; description length chooses the cuts, starting from the exact code-optimal cut found by a bottom-up dynamic program, and merges symbols (decision 4). The tie between binarizations shows up: with gold word classes, the MED search finds a grammar shorter than the gold one, with 0.0% commission, that shares 19% of the gold brackets, so bracket agreement is a diagnostic. Held-out bits use total probability; learning still charges derivations. Not used: the review's running test-then-train score over Cobweb's own counts, and a tempered likelihood against misspecification (Grünwald & van Ommen 2017).

### Neural hierarchical chunking, the contrast

Neural chunkers such as H-Net set boundaries anew for each input, where adjacent states differ (Hwang et al. 2025); across the systems reviewed, two or three levels of 3–6× compression per level were the useful range. None keeps a reusable chunk inventory, and chunk structure that the objective does not reward fades during training (Wu, Deshmukh et al. 2025). Diffusion models learn PCFG-like data by clustering features with similar contexts, one level at a time, with deeper levels needing more data (Favero et al. 2025).

**v2.** Chunk types and categories are cuts through stored hierarchies, reused across inputs and learned without gradients, and description length rather than a target ratio sets how many there are. Masked chunk prediction and entropy-based boundaries were not taken. (The review also noted that the acs-26 `main.bib` entry `diffusion-grammar` gives arXiv:2502.12089, Favero et al., a title that matches no paper.)

### Domains beyond language

Langley's essay and the TRELLIS paper name chess as the first target beyond strings and call for relations beyond "before" (Langley 2025; Singaravadivelan & Langley 2026). Chess memory research scores recall by errors of omission and commission: a master recalled 7.7 chunks of 2.5 pieces on average, mostly pawn chains and castled kings (Chase & Simon 1973), and the CHREST model fits human omissions but not commissions (Gobet & Simon 2000). Ideographic Description Sequences give gold, relation-labelled trees for Chinese characters, whose radicals are position-specific (Taft et al. 1999). Penn Treebank scores shift by more than 20 points with the protocol (Li et al. 2020); WSJ10 with gold tags is the standard short-sentence track (Klein & Manning 2002).

**v2.** Each domain brings its context window and its relations. In chess the representation hierarchy sees the user's star (queen rays and knight jumps), the composition hierarchy records a direction-and-distance relation, chunks join only elements whose anchors see each other, and a square-by-square scan supplies the order; the chunk types found are castling and fianchetto structures. Characters are relational trees whose relations are the IDS operators, a part's slot is part of what the representation hierarchy sees, and the hierarchy forms positional radical classes (left, right, top, bottom); an attested-position check measures commission. Real text uses WSJ10 with gold tags on NLTK's sample, and TinyStories (Eldan & Li 2023), added after the review. Not yet: chess recall against Gobet & Simon's tables, attack and defence relations, chunks with two separate parts (pins), human-rated pseudocharacters, the full-WSJ protocol (decision 5), music and plans.

### References (background)

- Baker (1979). Trainable grammars for speech recognition. *Speech Communication Papers, 97th Meeting of the Acoustical Society of America.*
- Brown et al. (1992). Class-based n-gram models of natural language. *Computational Linguistics* 18(4).
- Chase & Simon (1973). Perception in chess. *Cognitive Psychology* 4(1).
- Clark (2010). Learning context free grammars with the syntactic concept lattice. *ICGI.*
- de Marcken (1995). On the unsupervised induction of phrase-structure grammars. *Third Workshop on Very Large Corpora.*
- Drozdov et al. (2019). Unsupervised latent tree induction with deep inside-outside recursive autoencoders. *NAACL.*
- Eldan & Li (2023). TinyStories: How small can language models be and still speak coherent English? arXiv:2305.07759.
- Favero et al. (2025). How compositional generalization and creativity improve as diffusion models are trained. *ICML.*
- Gobet & Jansen (1994). Towards a chess program based on a model of human memory. *Advances in Computer Chess 7*, University of Limburg.
- Gobet & Simon (2000). Five seconds or sixty? Presentation time in expert memory. *Cognitive Science* 24(4).
- Goldsmith (2001). Unsupervised learning of the morphology of a natural language. *Computational Linguistics* 27(2).
- Goodman (1996). Parsing algorithms and metrics. *ACL.*
- Grünwald & Roos (2019). Minimum description length revisited. *International Journal of Mathematics for Industry* 11(1).
- Grünwald & van Ommen (2017). Inconsistency of Bayesian inference for misspecified linear models, and a proposal for repairing it. *Bayesian Analysis* 12(4).
- Hinton & van Camp (1993). Keeping the neural networks simple by minimizing the description length of the weights. *COLT.*
- Hwang, Wang & Gu (2025). Dynamic chunking for end-to-end hierarchical sequence modeling. arXiv:2507.07955 (*ICLR* 2026).
- Klein & Manning (2002). A generative constituent-context model for improved grammar induction. *ACL.*
- Langley (2025). Concepts and chunks in cognitive systems. *Advances in Cognitive Systems* 11.
- Langley & Stromsten (2000). Learning context-free grammars with a simplicity bias. *ECML.*
- Lari & Young (1990). The estimation of stochastic context-free grammars using the inside-outside algorithm. *Computer Speech and Language* 4(1).
- Li et al. (2020). An empirical comparison of unsupervised constituency parsing methods. *ACL.*
- MacLellan, Matsakis & Langley (2022). Efficient induction of language models via probabilistic concept formation. *Advances in Cognitive Systems*; arXiv:2212.11937.
- Pereira & Schabes (1992). Inside-outside reestimation from partially bracketed corpora. *ACL.*
- Petrov & Klein (2007). Improved inference for unlexicalized parsing. *NAACL-HLT.*
- Ruoss et al. (2024). Amortized planning with large-scale transformers: A case study on chess (first circulated as "Grandmaster-level chess without search"). *NeurIPS*; arXiv:2402.04494.
- Singaravadivelan & Langley (2026). A unified account of concepts and chunks: Extending Cobweb from categorization to composition. *Advances in Cognitive Systems*; arXiv:2609.30414.
- Spitkovsky et al. (2010). Viterbi training improves unsupervised dependency parsing. *CoNLL.*
- Stolcke & Omohundro (1994). Inducing probabilistic grammars by Bayesian model merging. *ICGI.*
- Taft, Zhu & Peng (1999). Positional specificity of radicals in Chinese character recognition. *Journal of Memory and Language* 40(4).
- Vereshchagin & Vitányi (2004). Kolmogorov's structure functions and model selection. *IEEE Transactions on Information Theory* 50(12).
- Wolff (1982). Language acquisition, data compression and generalization. *Language & Communication* 2(1).
- Wu, Deshmukh et al. (2025). Unsupervised chunking with hierarchical RNN. *Computational Linguistics* 51(3).
- Wu, Thalmann et al. (2025). Building, reusing, and generalizing abstract representations from concrete sequences. *ICLR.*
- Yang, Zhao & Tu (2021). PCFGs can do better: Inducing probabilistic context-free grammars with many symbols. *NAACL.*
- Zhai, Boyd-Graber & Cohen (2014). Online adaptor grammars with hybrid inference. *TACL* 2.

## Roadmap

| Stage | Content |
|---|---|
| v2.0 ✓ | supervised core: hierarchies, MDL cuts and merging, inside-outside + MBR, generation, six-condition evaluation |
| v2.1 ✓ | unsupervised learning from sentences: MDL objective, partial analyses, word classes, exact-scored beam search over chunk/merge moves from several starts |
| v2.2 (in part) | learning by day and by night ✓ (perceive with the current grammar; consolidate at night from the stored analyses or a restart; the full code chooses among the best search results); split moves; attention-like long-range context |
| v2.3 (in progress) | beyond the paper's corpora: Penn Treebank WSJ10 with gold tags ✓ (and larger training sets ✓; which descriptions make sentence structure pay ✓); Chinese characters ✓; a compiled Cobweb ✓. Open: starting categories that see structure (characters), finer positional concepts, α by description length; for real text, chunks categorized by their head and the total-probability code, with words at scale |
| v2.4 (in progress) | new data types and relations: a domain brings its own context window (the representation hierarchy's surface context) and its own typed relations (the composition hierarchy); chess positions with the star context and direction-and-distance relations ✓; simple English (TinyStories) for generated coherence ✓; characters with operators as relations ✓ |
| v2.5 (in progress) | coherence from what the descriptions can see ✓: a sentence is one tree or a forest of pieces, coded apart; a part's slot in its description; a board's read that counts material, kind by kind. Tried and kept aside: a Markov code over a forest's pieces (shorter code, no more coherent samples). Open: fewer forests; chess moves as compositions, attack and defence as relations ([hypotheses](#hypotheses-playing-chess-from-the-two-hierarchies)); a search over several moves at a time for characters from sequences; variable arity |

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
python experiments/v2/run_characters.py                    # needs data/ids/ids.txt; ~1 h (the unsupervised mode)
python experiments/v2/run_characters.py --modes relational --train 6000 --out experiments/v2/results/characters_6000
python experiments/v2/run_chess.py --train 8000 --test 560 --out experiments/v2/results/chess_8000
python experiments/v2/run_stories.py --vocab 100 --max-len 5 --train 2500 --out experiments/v2/results/stories_2500
python experiments/v2/run_stories.py --vocab 100 --max-len 5 --train 5000 --out experiments/v2/results/stories   # hours
python experiments/v2/run_chess.py --extract && python experiments/v2/run_chess.py    # needs the Lichess file in data/chess
python experiments/v2/run_chess.py --no-context --out experiments/v2/results/chess_plain
python experiments/v2/run_chess_play.py --extract && python experiments/v2/run_chess_play.py
python docs/figures/make_figures.py                        # figures of FRAMEWORK.md
```

The paper corpora are read from `../trellis_v1/data` (the v1 snapshot), with `data/` as the fallback.
