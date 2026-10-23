# TRELLIS v2 failure modes: what to check after every iteration

User direction, 22 October 2026. Every change to the framework is checked with
`experiments/v3/diag/t3_failure.py` alongside generation and compression. The
check is evaluation only. The learner sees raw tokens. The generating grammar
labels gold spans to score, and it never teaches anything.

## Memory is an AND/OR structure

Memory is an AND/OR graph over two Cobweb trees.

- **AND nodes are content concepts (chunk types).** A chunk is its left part AND its right part, so a content concept generalizes structure: which parts go together.
- **OR nodes are context concepts (categories).** A slot is filled by one member OR another, so a context concept generalizes over compositions: which fillers can stand in the same place.

Generation walks this graph. It picks a sentence-level AND node, expands its parts, fills each part's OR node with a member, and recurses.

## What we want to guarantee

**A. Generalizations generate their constituents, and stay grammatical.** A concept should give high probability to the constituents it actually has. Every way of instantiating it should also stay grammatical in the domain; for sentences, "grammatical" means the target grammar recognizes the result.

- **OR nodes.** Measured by precision and recall of context concepts as substitution classes.
  - **Precision:** swapping one member in for another keeps the sentence grammatical.
  - **Recall:** members of the same gold category share the concept.
  - **Failure signatures:**
    - low recall at every level means the concepts **undergeneralize** (memorized fillers);
    - low precision at the level generation uses means they **overgeneralize** (a slot admits wrong fillers).
- **AND nodes.** Measured by the same substitution test on content concepts, plus **recombination**. Recombination takes one member's left part with another member's right part. A chunk type whose parts are not independently recombinable is a **memorized pairing, not a rule**.

**B. Representations are comparable across granularity.** A word, a two-word chunk and a five-word chunk that play the same role should look alike.

- **Measure:** the AUC of the path-overlap similarity (the information-weighted path bags the trees use to name parts), comparing same-category pairs with different-category pairs. It is reported separately for:
  - same-length pairs;
  - word against chunk;
  - chunks of different lengths.
- **Cross-length precision** in A tests the same thing behaviourally: can "the dog" replace "the big red dog"?
- **Failure signature:** an AUC near 0.5 for cross-length pairs, or a cross-length precision near 0. Either means categories are split by size, not role. Each level of chunking then lives in its own vocabulary, and composition across levels fails.

**F. Fodor & Pylyshyn.**

- **Systematicity.** Swap two disjoint constituents of the same category in a training sentence, keeping only swaps the grammar accepts. Memory should find the swapped constituents as often as the originals.
  - **Failure signature:** a gap between the two rates. It means a representation depends on where a chunk was learned, not on what it is.
- **Productivity.** Generated sentences longer than any training sentence should exist, and should be grammatical. This is only meaningful on recursive grammars (`large` has relative clauses).

**C. Co-evolution.** Word2Vec learns all word vectors jointly, so every update to one word moves the representations of words that share its contexts. Here the trees' concept statistics update immediately. Stored descriptions follow only when relabelled, and a stored leaf never moves. The check reports how stale memory is:

- the fraction of stored content descriptions that differ from what they would be now;
- the fraction of stored context and content leaves that no longer sit under the concept their own description now sorts to.

**Failure signature:** high staleness that also costs A or B. Run the variants, a larger refresh budget and refit, which moves the leaf, and see whether A, B or generation improve. If they do, co-evolution is a property to exploit. If staleness is high but harmless, the lag is fine.

## How to run

```
sbatch ... logs/v3_job.sh logs/v3_diag/t3_failure.py <cond> '<cfg json>' <seed> <tag>
```

Run it on `small`, `med`, `large`, `term_low`, `term_med` and `term_high` (and on characters once a scorer exists). Output is one JSON line. Levels `u1`, `u2` and `u4` count up from the leaf (generation uses `u1`); `d1` to `d3` count down from the root.

## Results

### 22 October, `small`, seed 13, current config

(`ipath`, `word_cc`, relabel, budget 300)

| Check | Result | Reading |
|---|---|---|
| A-or precision / recall, u1 | 0.999 / 0.07 | fine OR nodes near the leaf memorize fillers (undergeneralize) |
| A-or precision / recall, d2 | 0.68 / 0.67 | the best trade-off sits mid-tree; no level has both high |
| A-or precision / recall, d1 | 0.22 / 0.68 | top OR nodes overgeneralize (positional, not syntactic) |
| A-and precision / recall / recombination, d2 | 0.996 / 0.995 / 0.996 | content (AND) concepts are near perfect on `small` |
| B AUC, context, same length | 0.695 | context similarity only moderately tracks category |
| B AUC, content | 0.991 | content similarity tracks chunk type |
| A-or precision, cross-length | 0.0 | a concept never mixes lengths successfully (`small` has no same-category cross-length pairs, so this only says mixing is wrong here) |
| F systematicity | 1.0 / 1.0 | swapped NPs found as constituents as often as the originals |
| C stale content descriptions | 0.45 | almost half of stored chunks describe their parts by out-of-date paths |
| C misplaced leaves, context / content | 0.43 / 0.58 | stored leaves no longer sit where their own description sorts |

**Leading hypotheses (to confirm on the larger grammars):**

- The context tree (OR) is the weak side.
- Cross-granularity comparability is the open question for B.
- Co-evolution is weak, so whether more of it helps is the test now running (`rb3000`, `refit`).

### 22 October, `large`, seed 13, current config

| Check | Result | Reading |
|---|---|---|
| A-or precision / recall, u1 / u2 | 0.91 / 0.01, 0.84 / 0.03 | OR nodes near the leaf memorize: almost no same-category filler shares them |
| A-or precision / recall, d3 / d2 / d1 | 0.63 / 0.35, 0.43 / 0.48, 0.21 / 0.55 | no level is both precise and complete: **OR nodes fail A on both sides** |
| A-and precision / recall / recombination, u1 | 0.67 / 0.31 / 0.87 | chunk types are mediocre; recombining parts beats swapping whole members |
| A-and, d2 | 0.45 / 0.54 / 0.63 | |
| B AUC context: same length / cross length | 0.59 / 0.66 | context similarity barely tracks category |
| B AUC content: same length / cross length | 0.64 / **0.54** | **B fails**: chunk types of different sizes are not comparable |
| F systematicity | 0.851 / 0.851 | passes: no gap after swapping same-category constituents |
| F productivity | 6.4% of samples longer than any training sentence, 59% of those grammatical | partial |
| C stale content descriptions; misplaced leaves (context / content) | 0.89; 0.71 / 0.92 | co-evolution is far weaker than on `small` |

**Ranking on `large`:**
- A-or fails on both sides.
- B fails across sizes.
- A-and is weak.
- Systematicity passes.
- Memory is very stale. Whether staleness causes the A/B failures is what the `rb3000` and `refit` variants test.

**Co-evolution variants on `small`:**
- `rb3000` cuts stale descriptions from 0.45 to 0.05, but the misplaced leaves stay at 0.45 / 0.70. A-or d3 moves from 0.94 / 0.45 to 0.99 / 0.52.
- `refit` brings misplaced leaves to 0.27 / 0.47 and A-and d2 to 1.0 / 1.0. A-or u1 recall rises from 0.07 to 0.14.
- So moving stale leaves helps a little on `small`.

### 22 October, co-evolution variants on `med` and `large` (seed 13): staleness is not the cause

| | best | rb3000 | refit |
|---|---|---|---|
| `large`, A-or u1 recall | 0.01 | 0.02 | 0.04 |
| `large`, A-or d2 precision / recall | 0.43 / 0.48 | 0.42 / 0.47 | 0.50 / 0.48 |
| `large`, B content AUC, cross length | 0.54 | 0.56 | 0.54 |
| `med`, B content AUC, cross length | 0.50 | 0.50 | 0.48 |
| `large`, C stale descriptions | 0.89 | 0.49 | 0.96 |
| learning time (`large`) | 10 min | 65 min | 81 min |

- More refresh, or moving stale leaves, costs 6–8× the time and barely changes A or B.
- Staleness stays high because naming parts by whole paths makes nearly every stored name change after every insertion.
- **Conclusion:** the A and B failures come from the descriptions themselves, not from stale memory. Co-evolution by refresh is not worth exploiting here.

**B diagnosis.** A word's donated own-content facet is its token (`w:dog`), but a chunk's is its content-tree path. The two vocabularies never overlap, so "dog" and "big dog" cannot look alike in the context tree even with identical surroundings.

**Next change: `cc_naming="tokens"`.** A chunk's own content becomes the bag of its tokens, and a word is the one-token case. This is domain-general and stable: it never needs relabelling. It is being checked with this file's diagnostics, plus generation and compression.

### 22 October, `cc_naming="tokens"` (seed 13): B improves, generation drops

| | best | tokens |
|---|---|---|
| B content AUC, cross length (`large` / `med`) | 0.54 / 0.50 | **0.72 / 0.60** |
| A-and d2 precision / recall (`large`) | 0.45 / 0.54 | 0.52 / 0.79 |
| A-or | fails both sides | unchanged |
| generation grammatical (`med`, `large`) | 0.81, 0.74 | 0.77, 0.61 |
| compact grammar (`large`, MDL up) | 13% | 0.4% |

- **Kept as an option, not as the default.** The diagnosis of B was right, but the token bag throws away order, which generation needs.

**4× data (MDL up, `large_x4`):** 18–23% grammatical at 54–104 categories, against 2–13% at 11–30 with 1× data. More data buys finer categories and better compact grammars, but only partly. Held-out bits are level with the bigram model (25.8–28.0 against 24.0–26.9).

**The persistent failure is A-or.** Cobweb clusters tokens by their one context, so the same type in two positions (subject and object) lands far apart. Word2Vec instead represents a type by the aggregate of all its contexts.

**Next: `type_names=True`.** A part is named by the aggregate of the context paths of every stored instance of its type; a type is its token sequence, and a word is the one-token case. This is joint, distributional learning, and it is unsupervised.

- **On `small`:** A-and d1 precision rises from 0.73 to 0.91, and A-or d3 recall from 0.45 to 0.57. Learning takes 4× the time.
- All grammars are running.

### 23 October: generation from the part's own leaf (`up=0`) reaches 94–100% grammatical

| Grammar | `up=1` grammatical (2 seeds) | `up=0` grammatical | `up=0` grammatical and new | `up=0` distinct |
|---|---|---|---|---|
| `small` | 1.00 | 1.00 / 1.00 | 0.25 / 0.50 | 0.69 / 0.75 |
| `med` | 0.85 | 1.00 / 1.00 | 0.85 / 0.72 | 0.90 / 0.78 |
| `large` | 0.68 | 1.00 / 0.94 | 0.74 / 0.82 | 0.59 / 0.93 |
| `term_low` | 0.66 | 0.96 / 1.00 | 0.63 / 0.80 | 0.84 / 0.90 |
| `term_med` | 0.64 | 1.00 / 1.00 | 0.86 / 0.29 | 0.89 / 0.62 |
| `term_high` | 0.66 | 1.00 / 0.99 | 0.32 / 0.74 | 0.27 / 0.82 |

**How this reads against A-or.**
- Memory's fine OR nodes are precise (u1 precision 0.88–0.95 on every grammar). The generator's `up=1` slot was one level too coarse.
- Novelty comes from recombining committed chunks whose leaves are shared, not from coarse slots.

**Remaining failure: novelty collapses where leaves are sparse.**
- Cases: `term_high` seed 13 (27% distinct), and `term_med` seed 17.
- **Next:** `min_members`. Each slot takes the deepest concept whose committed members of the part's kind hold at least m distinct types. This is the same "deepest concept with enough evidence" rule as the regeneration readout. m = 2 and 3 are running.

**`min_members` fails (23 October).** m = 2 gives 35–66% grammatical on the larger grammars, and m = 3 gives 19–58%. Climbing until a concept holds 2 distinct types usually passes through single-child chains into mixed concepts.

The part's own leaf (`up=0`) is now the generation default.
- Grammatical: 94–100% on every grammar and seed.
- Grammatical and new: 0.63 on average, the same as `up=1` (whose grammatical rate is 64–85%).
- So the leaf trades variety, not correct samples.
