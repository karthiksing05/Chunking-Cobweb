# Round 218: semi-incremental parsing, a spectrum from greedy to full inside-outside

Question (the user's idea): between greedy commit-as-you-go parsing and full inside-outside (IO) over the whole sentence, is there a point with a look-ahead window, IO inside the window, a frozen best parse, and a move on to the next window ("semi-incremental"; "IO for three levels of chunks at a time")? This round builds the two knobs, maps the spectrum at n = 1000 TinyStories sentences (r215/r216 recipe and evaluation sets, seeds 13/17/19/23), and measures cost, parse quality, tree statistics and generation.

## Code and exactness

- `src/trellis2/trellis3_r218.py`: `Trellis3R218(win_W=None, win_stride=None, io_depth_D=None, io_depth_mode="freeze", win_learn=False)`, subclass of `Trellis3R216` (so of `Trellis3`); `trellis3.py` is untouched. Tests: `tests/trellis2/test_r218_window.py` (25 pass; full `tests/trellis2` 174 pass, job 4804612). Drivers `experiments/v3/diag/r218_run.py`, `r218_cells.py`, `r218_analyse.py`, `r218_story.py` (t3_story.py with this learner), `r218_gen.sh`, `r218_gen_report.py`, `r218_judge.sh`, `r218_test.sh`.
- **Exactness.** With no knob set, `parse` is the parent's and the result is identical (chosen tree, logZ, every posterior, every composite gain; tested). A window as long as the sentence (`win_W=50`, with or without stride, `io_depth_D=50`, `chart` mode with a cap beyond the sentence) goes through the windowed code path and gives the same chosen tree, logZ, posteriors and gains (tested), and learning with it stores the same number of records as the baseline (tested). In the grid, the `full` cell reproduces the r215 logZ of every control sentence exactly (0 mismatches of 2437 / 2414 / 2423 / 2430 compared, seeds 13/17/19/23). Eval-only windowing (`win_learn=False`) leaves the learner untouched (tested: same records and stats).
- **Forest IO checked by enumeration.** For the capped chart (`io_depth_mode="chart"`) the analyses are forests; `_forest_io` is tested against brute-force enumeration of all forests of 4 symbols (logZ and two marginals agree).

## How freezing works (the exact definition)

A sentence is a list of symbols U (a symbol is a token or a frozen chunk). One pass slides a window over U and replaces U by a shorter list of symbols; passes repeat until one symbol (the sentence) is left.

1. **Window.** `win_W` = the number of symbols in a window (not tokens: after the first pass a symbol may be a frozen chunk). Windows start at symbol 0, S, 2S, ... (S = `win_stride`, default W: disjoint windows); the window that reaches the end of U is the last one. `win_W=None` = one window = all of U.
2. **IO inside the window.** The CKY chart of `Trellis3._chart` is built over the window's symbols: a composite's span is the token span of its symbols, its parts are the window's best sub-analyses, and `_surface/_content/_context/_factor` are the parent's (the same composite over the same parts has the same gain as in full IO). The raw tokens beyond the window edge stay visible as surface context (no hiding of the neighbours). `_inside_outside` gives the window's logZ and posteriors; `_mbr` gives its best tree.
3. **Freeze.** The commit region is the first S symbols of the window (the whole window if it is the last). The maximal nodes of the window's MBR tree that lie entirely inside the region (and, with `io_depth_mode="freeze"`, have height <= D) are frozen: each becomes one symbol of the next list (it is the chart's own composite element for that span); every MBR node inside a frozen node is appended to the final tree's `chosen` list; nothing inside a frozen node is parsed again. Nodes that cross the region's right edge are not frozen (their parts inside the region are); the S .. W-1 symbols beyond the region are look-ahead and are parsed again by the next window.
4. The next window starts at the first symbol after the commit region. Every pass freezes at least one merge (the last window holds two or more symbols, so it freezes a cherry), so passes end.
5. **Depth knob `io_depth_D`.** Two readings of "three levels at a time", both implemented.
   - `freeze`: IO over the whole window, then only nodes of height <= D are committed (height over the current symbols; a symbol has height 0). A staged commit: the whole chart is still computed, so it saves nothing (measured below).
   - `chart`: the chart itself is capped, a run evaluates composites over at most L = 2^D symbols. If the window has more than L symbols the analyses are forests of trees of at most L symbols (a lone symbol is a tree with factor 1), IO runs over forests (`_forest_io`), the MBR forest's trees are all frozen, and the next pass runs over those trees as symbols. A window of at most L symbols is plain IO. (A binary tree of height <= D has <= 2^D leaves; any shape on <= 2^D symbols is allowed here, a relaxation of "D levels".) The sentences here have <= 8 tokens, so `D3c` (cap 8) is full IO by construction.
6. **Returned parse.** `chosen` = the frozen nodes (a binary tree over the sentence); `comps` = every composite computed in any window (the last one computed for a span key); `post` = its posterior in the window that last computed it (a frozen node's: the window that froze it; normalised within that window, not over the sentence); `logZ` = the sum of the windows' logZ (equal to the sentence logZ for one window; with overlapping windows look-ahead counts twice); `logw` = the chosen tree's log weight sum(gain) ln2 / tau (comparable across cells; both scores are tabulated and agree in direction).
7. **Learning (`win_learn`).** False: `learn()` uses the full-IO parse; windowing only affects what is parsed at evaluation (and at generation time). True: `learn()` uses the windowed parse, so `chosen`, `comps` and `post` as above feed the unchanged `store="posterior"` storage (every computed composite stored with its latest window posterior; composites from windows that a later freeze superseded stay stored with their stale posterior; a cell with fewer composites therefore stores fewer records).
8. **Cost accounting.** `factors` = number of `_factor` evaluations (each = two Cobweb tree queries, the expensive step) per parse; a composite over the same two parts in a later window of the same parse is reused from a memo and not counted. `triples` = IO (i,k,j) triples summed over windows. `parse s` = wall seconds per sentence (one core). Greedy = `search="beam"`, `beam=1` (the repo's greedy parse: join the best pair, repeat); its cost counts only the composites it evaluated.

## Protocol

n = 1000 sentences (stories stream, window 7 surface context, recipe `r215_common.CFG`), seeds 13/17/19/23, exactly the r215/r216 control sets (Gcal, C1, C2full, C2one, C3scr, C3adj, C4v; 2437-2682 sentences per seed incl. seed 13's C1 for cross-seed agreement). "Eval-only" cells (`win_learn=False`) parse with the knobs under the one full-IO learner of that seed (r215's learner). "L" cells retrain a learner with `win_learn=True` (same knobs). Metrics as r215/r216: AUC of the per-word score (grammatical reference vs corrupted), P(accept) at the threshold giving 90% acceptance of Gcal, tree F1 against the NLTK shallow-phrase gold on the 300 C1 sentences (`r216_gold.json`), right-branching rate, cross-seed agreement (seed 13's C1 parsed by the four learners, mean pairwise bracket F1). Mean ± sd over 4 seeds; full tables in `r218_tables.md`, cases in `r218_cases.md`, figure data `r218_spectrum.tsv`. Jobs: parse eval 4804114/16/18/20 (seeds 13/17/19/23), learn cells 4804115/17/19/21, chart cells 4804323/25/27/29 and 4804324/26/28/30, analysis 4804393 (4804273 first pass).

## Main table (mean over 4 seeds; cost = per C1 sentence of 3-8 tokens)

`ungram AUC` = mean of AUC for scramble, adjacent swap and agreement error with the logw per-word score (logZ per word gives the same picture, `r218_tables.md`); `P(acc C2full)` = acceptance of the all-content-words-swapped sentences (grammatical, not meaningful; a syntax-only parser should accept them; C1 itself is accepted at about .93).

| cell | win_learn | composites evaluated | / full | parse ms | ungram AUC | adjacent-swap AUC | P(acc C2full) | F1 gold | non-RB trees | inner chunk len | F1 vs full tree | cross-seed F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | no | 47.5 | 1.00 | 33.9 | 0.542 | 0.533 | 0.910 | 0.352 | 0.959 | 2.83 | 1.000 | 0.866 |
| greedy | no | 9.9 | 0.21 | 7.0 | 0.541 | 0.533 | 0.914 | 0.353 | 0.972 | 2.80 | 0.934 | 0.859 |
| W2 | no | 5.4 | 0.11 | 4.8 | 0.554 | 0.540 | 0.910 | 0.287 | 1.000 | 2.59 | 0.532 | 1.000 |
| W3 | no | 9.8 | 0.21 | 7.7 | 0.558 | 0.544 | 0.913 | 0.242 | 0.995 | 2.82 | 0.557 | 0.950 |
| W4 | no | 14.6 | 0.31 | 10.1 | 0.541 | 0.531 | 0.917 | 0.308 | 0.988 | 2.85 | 0.854 | 0.939 |
| W6 | no | 30.6 | 0.64 | 21.5 | 0.557 | 0.543 | 0.910 | 0.229 | 0.959 | 3.06 | 0.734 | 0.861 |
| W3s2 | no | 14.3 | 0.30 | 10.1 | 0.538 | 0.530 | 0.918 | 0.428 | 0.934 | 2.68 | 0.788 | 0.870 |
| W4s2 | no | 22.7 | 0.48 | 16.1 | 0.540 | 0.532 | 0.916 | 0.413 | 0.936 | 2.75 | 0.799 | 0.885 |
| W6s3 | no | 35.9 | 0.76 | 24.9 | 0.542 | 0.533 | 0.910 | 0.371 | 0.959 | 2.88 | 0.915 | 0.897 |
| D1 | no | 47.7 | 1.00 | 32.4 | 0.542 | 0.533 | 0.910 | 0.354 | 0.965 | 2.82 | 0.986 | 0.870 |
| D2 | no | 47.6 | 1.00 | 33.5 | 0.542 | 0.533 | 0.911 | 0.354 | 0.964 | 2.82 | 0.990 | 0.867 |
| D3 | no | 47.5 | 1.00 | 33.6 | 0.542 | 0.533 | 0.910 | 0.352 | 0.960 | 2.83 | 0.999 | 0.869 |
| W4D2 | no | 17.5 | 0.37 | 12.6 | 0.540 | 0.532 | 0.911 | 0.338 | 0.988 | 2.80 | 0.932 | 0.885 |
| D1c | no | 8.7 | 0.18 | 6.7 | 0.557 | 0.540 | 0.912 | 0.353 | 0.994 | 2.56 | 0.660 | 0.844 |
| D2c | no | 25.2 | 0.53 | 16.9 | 0.542 | 0.533 | 0.911 | 0.362 | 0.974 | 2.80 | 0.970 | 0.890 |
| D3c | no | 47.5 | 1.00 | 31.7 | 0.542 | 0.533 | 0.910 | 0.352 | 0.959 | 2.83 | 1.000 | 0.866 |
| W6D2c | no | 22.4 | 0.47 | 14.8 | 0.554 | 0.541 | 0.911 | 0.259 | 0.974 | 2.85 | 0.771 | 0.846 |
| W3L | yes | 9.8 | 0.21 | 6.4 | 0.566 | 0.549 | 0.926 | 0.265 | 0.998 | 2.71 | 0.570 | 0.958 |
| W4L | yes | 14.6 | 0.31 | 9.5 | 0.542 | 0.533 | 0.910 | 0.306 | 0.990 | 2.86 | 0.827 | 0.935 |
| D1L | yes | 47.8 | 1.01 | 34.3 | 0.539 | 0.530 | 0.912 | 0.357 | 0.959 | 2.83 | 0.902 | 0.884 |
| D2L | yes | 47.6 | 1.00 | 32.8 | 0.542 | 0.532 | 0.914 | 0.348 | 0.968 | 2.83 | 0.935 | 0.867 |
| D3L | yes | 47.6 | 1.00 | 33.0 | 0.541 | 0.533 | 0.909 | 0.356 | 0.957 | 2.83 | 0.938 | 0.862 |
| W4D2L | yes | 17.6 | 0.37 | 11.4 | 0.541 | 0.533 | 0.913 | 0.339 | 0.995 | 2.80 | 0.864 | 0.870 |
| D1cL | yes | 8.7 | 0.18 | 5.4 | 0.567 | 0.549 | 0.926 | 0.355 | 0.995 | 2.56 | 0.644 | 0.897 |
| D2cL | yes | 25.2 | 0.53 | 15.4 | 0.543 | 0.535 | 0.912 | 0.364 | 0.958 | 2.80 | 0.924 | 0.883 |

Cell key: `W<w>` window of w symbols, disjoint (stride = w); `W<w>s<s>` stride s (w - s symbols of look-ahead); `D<d>` `freeze` mode; `D<d>c` `chart` mode (cap 2^d symbols); `L` suffix = `win_learn=True`; `greedy` = beam 1. Reference rows for trees: right-branching F1 on the same sentences = .423±.013 (above `full`, as in r216).

Per-seed paired differences of the mean ungrammatical AUC, cell minus full, are in `r218_tables.md` (`seeds positive` column).

## Findings

**1. Cost.** Composites evaluated per sentence (3-8 tokens, mean 47.5 in full IO): greedy (beam 1) 9.9 (.21), `W2` 5.4 (.11), `W3` 9.8 (.21), `W4` 14.6 (.31), `W6` 30.6 (.64), `W3s2` 14.3 (.30), `W4s2` 22.7 (.48), `W6s3` 35.9 (.76), `D1c` 8.7 (.18), `D2c` 25.2 (.53), `W4D2` 17.5 (.37), `W6D2c` 22.4 (.47). Parse wall time follows (full 33.9 ms, `D2c` 16.9 ms, `W4` 10.2 ms, greedy 7.0 ms). By sentence length (3..8 tokens): full 4 / 10 / 20 / 35 / 56 / 84; `D2c` 4 / 10 / 17 / 23 / 29 / 35; `D1c` 3 / 4.3 / 7 / 8 / 10 / 11; `W4` 4 / 10 / 11 / 12 / 15 / 21. A fixed cap makes the cost per sentence about linear in its length where full IO is cubic (the cap-4 chart is 6n - 14 composites per pass; at n = 20 full IO is 1330 composites against about 106 per pass; that extrapolation is arithmetic, not measured: all sentences here have <= 8 tokens). The `freeze` depth cells `D1/D2/D3` evaluate the same 47.5-47.7 composites as full IO (the whole chart is built before anything is frozen; their later passes hit the memo): "freeze after three levels" saves nothing, only the capped chart (`Dc`) or a window does. Learning with the windowed parse is faster in proportion: learner training 117-141 s per seed for the full learner (132 s mean over the four `freeze` L cells) vs 35 s (`W3L`), 52 s (`W4L`), 32 s (`D1cL`), 76 s (`D2cL`), 60 s (`W4D2L`) at n = 1000, and it stores fewer records (29.8k full; 13.3k `W3L`, 17.2k `W4L`, 12.3k `D1cL`, 21.0k `D2cL`, 18.9k `W4D2L`; the `freeze` cells store 29.7-30.1k).

**2. How close are the trees to full IO?** Share of C1 trees identical to full IO / F1 between the bracket sets: `D3` .994 / .999, `D2` .955 / .990, `D1` .941 / .986, `D2c` .888 / .970, `greedy` .807 / .934, `W4D2` .816 / .932, `W6s3` .662 / .915, `W4` .527 / .854, `W4s2` .363 / .799, `W3s2` .328 / .788, `D1c` .092 / .660, `W3` .202 / .557, `W2` .028 / .532. Greedy (beam 1) is already close to full IO on these short sentences (F1 .93).

**3. Parse quality, acceptance.** Every cell's per-word logZ / logw is a weak grammaticality score (mean ungrammatical AUC .54-.57 for all 25 cells, as r215/r216 found for full IO, .542); content-swap acceptance (logw) is .909-.927 for all cells (C1 .93: all accept the grammatical-but-meaningless swaps equally). No cell is worse than full IO on acceptance beyond seed noise (seed sd ~.004-.013). Several are slightly higher, consistently across the four seeds: `W2` +.013, `W3` +.017, `W6` +.016, `D1c` +.016, `W6D2c` +.012, `W3L` +.025, `D1cL` +.025 (mean ungram AUC, logw, 4/4 seeds positive); the others (`greedy`, `W4`, `W3s2`, `W4s2`, `W6s3`, `D1`, `D2`, `D3`, `D2c`, `W4D2`, `W4L`, `D1L`-`D3L`, `W4D2L`, `D2cL`) are within ±.005 of full. I do not read the gains as a better parser: the absolute AUCs stay near .55, the cells with the gain are those whose trees differ most from full IO and are not the ones with the best gold F1, and the scores are sums over windows. The point is that the acceptance score does not need full IO.

**4. Trees against the shallow-phrase gold and right-branching.** F1 on the 300 C1 sentences (right-branching .423): full .352, greedy .353, `D1/D2/D3` .354/.354/.352, `D2c` .362, `D1c` .353, `D2cL` .364, `D1cL` .355; strided windows `W3s2` .428, `W4s2` .413, `W6s3` .371; disjoint windows `W2` .288, `W3` .242, `W4` .308, `W6` .229, `W4D2` .338, `W6D2c` .259. So **disjoint windows hurt** (a chunk can never cross a window boundary in the pass that creates it, so boundaries at fixed token positions cut phrases; `W6` is worse than `W4` although it sees more), while **overlapping windows that commit only the first part** (look-ahead W - S symbols) recover and exceed full IO against this gold; `W3s2` is above right-branching (.428 vs .423; sd .009 and .013, so call it level). The gold is shallow phrases only (NP/VP/PP/AP), which favours local chunking; this is evidence about phrase-like chunks, not about the parser being more correct. Non-right-branching share: full .96, all cells .93-1.00 (`W2`: 1.00 by construction). Mean inner chunk length: full 2.83, `D1c` 2.56, `W2` 2.59, `W3s2` 2.68, `W6` 3.06. Chunk reuse (share of inner chunks whose word string is a stored composite): full .53, `D1c` .61, `W3s2` .57, `W2` .60, with the shorter chunks of those cells; the weight >= .5 variant is in `r218_tables.md`.

**5. Cross-seed agreement** (seed 13's C1 parsed by the four learners, mean pairwise F1): full .866, greedy .859, `D2c` .890, `W6s3` .897, `W3s2` .870, `D1c` .844, `W6D2c` .846, `W3` .950, `W2` 1.000. The `W2`/`W3` values are an artefact, not stability: a window of 2 symbols has one analysis, so the tree is a fixed function of the sentence length and ignores the learner (and a window of 3 offers two). Agreement is not a quality measure for these two cells.

**6. Learning with the windowed parse (`win_learn`).** Learner and eval-only cells with the same knobs agree within noise on acceptance (`W3L` .566 vs `W3` .558; `W4L` .542 vs `W4` .541; `D1cL` .567 vs `D1c` .558; `D2cL` .543 vs `D2c` .542 mean ungram AUC (logw: .566/.558, .542/.541, .567/.558, .543/.542)) and on gold F1 (`W3L` .265 vs `W3` .242; `W4L` .306 vs `W4` .308; `D1cL` .355 vs `D1c` .353; `D2cL` .364 vs `D2c` .362). Training on the windowed parse changes the stored memory (up to 59% fewer records) but not what the parse of a held-out sentence looks like or scores, at this n.

**7. Generation (n = 1000, the r212 pair-gated causal arm at temp 2.0, edge exact, window 7; judge: Qwen3-8B + phi-4 ensemble, `r214_judge2.py score` at its stored thresholds).** Learners trained with `win_learn=True` (full = the baseline learner). One draw set (1544-1600 attempted draws) per cell and seed, so the sampling error of a pass rate over about 1280 distinct new sentences is about .013 and the two seeds differ by .05 in level. Per seed (distinct new = novel flag and distinct string; pass rates are among distinct new sentences; `per draw` = distinct new syntax-passing sentences per attempted draw):

| cell | seed | distinct new | syntax pass | syntax + semantics pass | syntax-pass per attempted draw |
|---|---|---|---|---|---|
| D1cL | 13 | 1268 | 0.669 | 0.450 | 0.5300 |
| D1L | 13 | 1284 | 0.687 | 0.464 | 0.5513 |
| D2cL | 13 | 1314 | 0.686 | 0.460 | 0.5631 |
| D2L | 13 | 1296 | 0.664 | 0.438 | 0.5381 |
| D3L | 13 | 1282 | 0.672 | 0.452 | 0.5387 |
| full | 13 | 1292 | 0.688 | 0.467 | 0.5556 |
| W3L | 13 | 1302 | 0.675 | 0.440 | 0.5494 |
| W4D2L | 13 | 1288 | 0.652 | 0.426 | 0.5250 |
| W4L | 13 | 1257 | 0.660 | 0.436 | 0.5188 |
| D1cL | 17 | 1278 | 0.643 | 0.422 | 0.5324 |
| D2cL | 17 | 1236 | 0.653 | 0.426 | 0.5227 |
| full | 17 | 1313 | 0.636 | 0.427 | 0.5408 |
| W3L | 17 | 1271 | 0.630 | 0.439 | 0.5188 |
| W4D2L | 17 | 1260 | 0.636 | 0.440 | 0.5188 |
| W4L | 17 | 1278 | 0.619 | 0.402 | 0.5123 |

Seed 13 (all cells): syntax pass full .688, `D1L` .687, `D2cL` .686, `W3L` .675, `D3L` .672, `D1cL` .669, `D2L` .664, `W4L` .660, `W4D2L` .652. Seed 17: full .636, `D2cL` .653, `D1cL` .643, `W4D2L` .636, `W3L` .630, `W4L` .619. `D3L` (the learner that freezes nodes of height <= 3 and re-parses above them; its C1 trees agree with full IO's at F1 .938) differs from `full` by .016 in syntax pass at seed 13, about the size of the draw noise. No windowed or capped learner is below full by more than .04 (`W4L` is lower at both seeds, -.028 and -.017, and `W4D2L` is -.036 and 0.0 at seeds 13/17: within about 2-3 standard errors, suggestive, not confirmed), and `D2cL` / `D1cL` are level with full at both seeds. Syntax-pass distinct-new sentences per attempted draw: full .556 / .541 (13 / 17), `D2cL` .563 / .523, `D1cL` .530 / .532, `W3L` .549 / .519, `W4L` .519 / .512. The generation numbers say that learning from bounded-window parses does not visibly damage the generator at n = 1000; they cannot rank the cells (1-2 seeds, one draw set each). Any parse the generation path makes also uses the knobs.

## Case studies (seed 13, n = 1000; C1 test sentences with the most distinct trees across the shown cells; composites evaluated at the right)

**when they got home something unexpected happened**

| cell | tree | composites evaluated |
|---|---|---|
| greedy | `((((when they) (got home)) (something unexpected)) happened)` | 13 |
| W2 | `(((when they) (got home)) ((something unexpected) happened))` | 6 |
| W4 | `(((when they) (got home)) (something (unexpected happened)))` | 15 |
| W4s2 | `(((when they) (got home)) (something (unexpected happened)))` | 25 |
| D1 | `(((when they) ((got home) something)) (unexpected happened))` | 62 |
| D3 | `((when (they (got home))) (something (unexpected happened)))` | 59 |
| D1c | `(((when they) (got home)) ((something unexpected) happened))` | 10 |
| D2c | `(((when they) (got home)) (something (unexpected happened)))` | 29 |
| W4D2 | `(((when they) (got home)) (something (unexpected happened)))` | 15 |
| full | `(((when (they (got home))) something) (unexpected happened))` | 56 |
| W4L | `((((when they) got) home) (something (unexpected happened)))` | 15 |
| D3L | `((when ((they got) home)) (something (unexpected happened)))` | 56 |
| D2cL | `((when ((they got) home)) (something (unexpected happened)))` | 29 |

**soon tim's friends came to the park**

| cell | tree | composites evaluated |
|---|---|---|
| greedy | `((soon (tim's friends)) (came (to (the park))))` | 12 |
| W2 | `(((soon tim's) (friends came)) ((to the) park))` | 6 |
| W4 | `((((soon tim's) friends) came) (to (the park)))` | 15 |
| W4s2 | `((soon tim's) ((friends came) (to (the park))))` | 25 |
| D1 | `(((soon tim's) friends) (came (to (the park))))` | 56 |
| D3 | `(((soon tim's) friends) (came (to (the park))))` | 56 |
| D1c | `(((soon tim's) friends) ((came to) (the park)))` | 10 |
| D2c | `(((soon tim's) friends) (came (to (the park))))` | 29 |
| W4D2 | `(((soon tim's) friends) (came (to (the park))))` | 17 |
| full | `(((soon tim's) friends) (came (to (the park))))` | 56 |
| W4L | `((((soon tim's) friends) came) (to (the park)))` | 15 |
| D3L | `((((soon tim's) friends) came) (to (the park)))` | 56 |
| D2cL | `((((soon tim's) friends) came) (to (the park)))` | 29 |

**they were happy and they were friends**

| cell | tree | composites evaluated |
|---|---|---|
| greedy | `((((they were) happy) (and (they were))) friends)` | 13 |
| W2 | `(((they were) (happy and)) ((they were) friends))` | 6 |
| W4 | `((((they were) happy) and) (they (were friends)))` | 15 |
| W4s2 | `(((they were) happy) (and (they (were friends))))` | 28 |
| D1 | `(((they were) happy) (and (they (were friends))))` | 56 |
| D3 | `(((they were) happy) (and (they (were friends))))` | 56 |
| D1c | `(((they were) happy) ((and they) (were friends)))` | 10 |
| D2c | `(((they were) happy) (and (they (were friends))))` | 29 |
| W4D2 | `(((they were) happy) (and (they (were friends))))` | 17 |
| full | `(((they were) happy) (and (they (were friends))))` | 56 |
| W4L | `((((they were) happy) and) (they (were friends)))` | 15 |
| D3L | `((((they were) happy) and) ((they were) friends))` | 57 |
| D2cL | `(((they were) happy) (and (they (were friends))))` | 29 |

**when amy found him they both laughed**

| cell | tree | composites evaluated |
|---|---|---|
| greedy | `(when ((((amy found) him) (they both)) laughed))` | 15 |
| W2 | `(((when amy) (found him)) ((they both) laughed))` | 6 |
| W4 | `((((when amy) found) him) (they (both laughed)))` | 15 |
| W4s2 | `((when amy) ((found him) (they (both laughed))))` | 25 |
| D1 | `(when (((amy found) him) (they (both laughed))))` | 61 |
| D3 | `(when (((amy found) him) (they (both laughed))))` | 56 |
| D1c | `(((when amy) (found him)) (they (both laughed)))` | 10 |
| D2c | `((((when amy) found) him) (they (both laughed)))` | 29 |
| W4D2 | `((((when amy) found) him) (they (both laughed)))` | 17 |
| full | `(when (((amy found) him) (they (both laughed))))` | 56 |
| W4L | `((((when amy) found) him) (they (both laughed)))` | 15 |
| D3L | `(when (((amy found) him) (they (both laughed))))` | 56 |
| D2cL | `((((when amy) found) him) (they (both laughed)))` | 29 |

In the sentences above greedy and `W2` produce flat or left-leaning stacks that ignore the phrase (`((((when they) (got home)) (something unexpected)) happened)`; `(((and from) (that day)) ((they became) (good friends)))`), `W4` cuts at its fixed alignment (`((((soon tim's) friends) came) (to (the park)))` vs `(((soon tim's) friends) (came (to (the park))))`), and `D2c` / `D3` / full agree on most sentences while `D2c` evaluates 29 composites for a 7-token sentence against 56. `D1c` (cap 2) builds pairs per pass (`((came to) (the park))`) and goes wrong where a phrase needs three symbols to meet first.

## Recommendation: where on the spectrum TRELLIS should sit

1. **At the sentence lengths of the current experiments (<= 8 tokens) keep full IO as the reference.** It costs 34 ms per parse, learning n = 1000 takes about 2 minutes, and no cell shows a gain I would trust (finding 3). Nothing here motivates changing the current recipe at this length.
2. **Where cost matters (longer inputs), use bounded-chunk IO, `io_depth_D=2, io_depth_mode="chart"` (cap 4 symbols per run, passes until one symbol, `win_learn=True`).** On these sentences it evaluates .53 of full IO's composites (per length: equal up to 4 tokens, .42 at 8 tokens), has the same acceptance AUCs (.542 vs .542), the same-or-better gold F1 (.362 vs .352; `D2cL` .364), 89% identical trees and F1 .970 with full IO's, higher cross-seed agreement (.890 vs .866), 42% less learning time (76 s vs 132 s), and no generation loss visible at n = 1000. It is the point on the spectrum where the user's "a few levels of chunks at a time, freeze, move on" holds without a measured price. The saving here is modest (2x) because sentences are short; the gain is asymptotic (linear in n at a fixed cap, full IO cubic), which this round did not test on long sentences.
3. **Not the user's disjoint-window form.** `win_stride = win_W` (windows that never overlap) is the worst region (gold F1 .23-.31, trees 53-85% F1 against full IO). A window needs look-ahead (commit the first S of W symbols): `W4s2` / `W3s2` / `W6s3` are cheaper than full IO (.48 / .30 / .76 of the composites) and reach .413 / .428 / .371 gold F1, but their trees differ most from full IO's (F1 .79 / .79 / .915), so they are a different parser, not an approximation; whether that parser is better is not settled by shallow-phrase F1.
4. **Not "freeze after D levels of a full chart" (`io_depth_mode="freeze"`).** Trees near full IO but no saving.
5. **Greedy (beam 1) is the cheapest competitive point on trees and acceptance (.21 of the composites, F1 .93 with full IO's trees, same AUC and gold F1) but it was only tested at eval time**: with `store="posterior"` the learner stores the IO posterior of every analysis, and a one-analysis greedy parse stores only its chosen composites; learning with greedy was not run, so whether the IO's soft storage matters for the learned memory is untested. That is the experiment that would decide between `D2c` and greedy.

## Caveats and what was not done

- Sentences are 3-8 tokens, so W >= 8 and D3c are full IO; the cost advantage at long lengths is arithmetic, not measured. No long-sentence run.
- The acceptance score is near chance for every cell (r215/r216), so acceptance cannot discriminate parsers; the tree measures are against a shallow-phrase gold only.
- For windows with look-ahead, `logZ` double counts the overlap; `logw` (chosen-tree log weight) is the comparable score; both are tabulated and agree.
- `win_learn=True` stores composites whose window posteriors are not sentence posteriors and keeps stale ones from superseded windows; I did not test a variant that drops them.
- `io_depth_mode="chart"` caps the chunk size at 2^D symbols, not its height at D.
- Generation: one draw set per cell, seeds 13 (9 cells) and 17 (6 cells), temperature 2.0 only.
- Greedy learning was not run (see 5). Eval-only greedy used the full-IO learner.
- Raw per-sentence results (`r218_raw/*.json`, 43 MB) are not committed; regenerate with `r218_run.py <seed> 1000 eval|learn` (about 10 minutes per job) and `r218_analyse.py`. Generation dumps are in `experiments/v3/results/story_gen/r218/` (judge scores `scores.tsv`, `report.md`).

Generation jobs: seed 13 4804141-4804147 (`full`, `W3L`, `W4L`, `D1L`, `D2L`, `D3L`, `W4D2L`) and 4804335/4804336 (`D1cL`, `D2cL`); seed 17 4804337-4804342; judge 4804603. Tests 4804013, 4804314, 4804612.
