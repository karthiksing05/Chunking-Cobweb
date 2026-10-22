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
