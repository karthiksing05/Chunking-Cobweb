# Round 223: abstract the context, keep emission exact (b_ctx x W' x K). Verdict: NO-GO; R-syn does not appear (max .245), so n=4000, learning curves and showcase were not run

## Mechanism and its place in the original architecture
Question: the word-class routes (r214 emission or gate at class level, r219, r222) need syntactically pure classes, which the word-category tree does not give. Here the abstraction sits on the **conditioning context** and emission stays the occurrence's own word.
- Existing parts it reuses, none changed: the context tree (stored contexts keep the exact words in slots l1..l_window), the read (`Trellis3._gap_fills`: each committed word occurrence is scored by the product over the query attributes of P(slot value | its context leaf) with hierarchical Dirichlet back-off), the reach dial W' (`read_reach`, r214: the query drops l_k for k > W'), the frequency cut K (`emit_freq_cut`, r214: the K most frequent types = the protected frequent / function-word set), the word-category tree (`WordTree`, level b), the word-level pair gate of the driver.
- New: `Trellis3R223.ctx_bits = b_ctx` (and `ctx_func_bits`, default exact). In `_reach_query` a content word in a window slot is replaced by its class C_b (a `_Cls`: the word types of the class at level b of the word tree). The read for such a slot is the same read marginalised over the class: factor = P(slot in C | leaf) = sum over members of the back-off estimates (the back-off recursion is linear, so the sum of estimates is the estimate of the sum). b = inf is the exact word; b = 0 is the root class, a wildcard (the attribute is dropped). It is a read-out change on the existing tree (the same kind as `read_bits` / `read_reach`), implemented as a mixin that sits above `Trellis3` in the MRO, so every parent wrapper (edge_exact, repeat_read, the r214 lifted read, caches) is unchanged.
- Default byte-identical, with a test (`tests/trellis2/test_r223_context.py`: default reads equal Trellis3R214's at edge_exact on and off; singleton classes equal the exact read; a root class equals the read with those slots dropped; a two-member class read equals the sum of the two exact reads; mapping rules). `tests/trellis2` r214 + r223: 17 passed (job 4808054).
- Does r214 already do part of this? `read_bits` lifts the *occurrence's* context leaf (the stored side) in the context tree, `wtree` / `window_names` (Addendum 3) names the window slots *at learning time* with a word-tree level for every word including function words (stored contexts hold names), and `emit_bits` / `wgate` abstract the emitted word. None reads the query's content slots at class level against exact stored contexts with the function-word set exact; that is what this adds.

## Cheap check (n=1000, seed 13, token input, window 5, edge exact, word pair gate, tau 1, exact emission, func:100 word tree, judge2 unchanged)
36 arms: b_ctx in {leaf (exact), 7, 6, 4, 0 (root)} x W' in {1, 2, 3, 5} x K in {25, 50} (leaf: K 25 only, K does not act). Jobs 4808105-4808109, judged in 4808727 (`r223p1/judge1/report.md`). Cells R-sem / R-syn / R-non:
| b_ctx, K | W'=1 | W'=2 | W'=3 | W'=5 |
|---|---|---|---|---|
| leaf (exact) | .28/.20/.53 | .61/.21/.19 | .68/.18/.14 | .60/.17/.23 |
| 7, K25 | .29/.19/.52 | .53/.22/.25 | .65/.18/.17 | .60/.19/.21 |
| 7, K50 | .29/.21/.51 | .54/.24/.23 | .66/.19/.16 | .59/.20/.21 |
| 6, K25 | .28/.21/.51 | .49/.24/.27 | .61/.21/.19 | .59/.18/.23 |
| 6, K50 | .29/.20/.52 | .52/.25/.23 | .62/.22/.17 | .58/.20/.22 |
| 4, K25 | .29/.19/.52 | .42/.24/.34 | .54/.22/.25 | .52/.23/.26 |
| 4, K50 | .30/.20/.51 | .46/.25/.29 | .57/.23/.20 | .55/.23/.23 |
| 0 (root), K25 | .23/.18/.60 | .36/.19/.45 | .41/.22/.37 | .42/.23/.35 |
| 0 (root), K50 | .29/.16/.55 | .42/.21/.37 | .48/.24/.28 | .46/.24/.30 |
Part 2 (`judge2/report.md`, job 4808966 + judge 4810567; K25): function words abstracted too (cf) and W'=0:
W'=0 exact .07/.09/.85; W'=3 cb=leaf cf=7 .60/.17/.23, cf=6 .58/.18/.24, cf=4 .48/.17/.35; cb=7 cf=6 .55/.19/.26; cb=6 cf=4 .43/.22/.36; cb=0 cf=6 .33/.20/.47, cf=4 .28/.19/.52; W'=5 cb=0 cf=6 .36/.20/.44.

## Against the predictions
- b_ctx = leaf gives R-sem: yes (.61-.68 at W' 2-3; same as the earlier gated leaf, .63).
- b_ctx = root keeps local syntax and gives R-syn: **no**. Root with the function skeleton exact (W' 3-5) gives R-sem .41-.48, R-syn .22-.24, R-non .28-.37. Abstraction moves mass out of R-sem into R-non, and the R-syn share does not grow beyond .25 in any cell (maximum .245: b 4 K50 W'2, b 6 K50 W'2; the earlier ceiling was .265).
- Abstracting function words, or shrinking W': yes for R-non, with the same limits. W'=0 gives R-non .85; W'=1 gives .50-.60; root context + function words abstracted + W' 3 gives .47-.52. So the context-length dial (W') and the b_ctx dial reproduce R-sem and R-non, not R-syn.
Regimes reached by (b_ctx, W'): R-sem .68 (leaf, W'=3, K25), R-non .85 (W'=0) or .60 (root, W'=1, K25), R-syn max .245. Go rule: R-sem >= .6 and R-non >= .6 met; R-syn >= .5 not met.

## Why it does not make R-syn (read from the arms)
- The word-level pair gate (kept exact, as before) already forces every adjacent pair to be a stored bigram, so local syntax is largely satisfied in every cell; what the abstraction removes is the long-range selectional constraint. With the abstraction the sentences become either still-sensible generic story sentences ("tom was scared", "he saw lily", R-sem) or syntactically broken from beyond-bigram agreement/frame failures (R-non). The judged samples of `judge2` show the pattern: the middle cells give R-syn only when a repetition or odd object appears ("lucy was so happy and happy", "lily and lily thanked her"), about 1 in 5.
- This agrees with the r219 oracle: even a class-exact swap yields R-syn .45-.52 only with all content words replaced; in generation, only the context (not the emitted word) is abstracted, and the emitted word is drawn from stored pairs, which are sensible.
- The frequency cut K does not matter much (K 25 vs 50 within .03 in R-syn; K 50 gives slightly higher R-sem at coarse b).
Step 3 (n=4000 seeds 13/17, learning curves, 5 examples per regime) was gated on R-syn >= .45 or clearly above .265: not met, not run. Examples of each regime from the cheap check are in `r223p1/judge1/report.md` (5 per arm).

Files: `src/trellis2/trellis3_r223.py`, `tests/trellis2/test_r223_context.py`, `experiments/v3/diag/r223_story.py` (driver; arm keys cbits, cfbits), `r223_p1.sh`, `r223_p2.sh`, `r223_judge.sh`, `r223_test.sh`; results `experiments/v3/results/story_gen/r223p1/`. Jobs: 4808105-4808109 (arms), 4808727 (judge), 4808966 (part 2), 4810567 (judge), 4808054 (tests).
