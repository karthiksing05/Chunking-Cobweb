# Changes in response to REVIEW_1.md

Status key: fixed, partly fixed, pending (another agent is running it), not fixed (reason).

## Blocking
1. cs_fig1 (superseded protocol): fixed. The figure is removed. Fig. cscurve (chess_cs1973_curve.png, identical to cs73x_curve.png, current protocol) is the only recall figure. The "12, 18, 25, 26" trial is traced to cs73x_s13.json (4,000 positions, K=7, example 1 of the first point list) and is now Table cstrial, labelled single example, seed 13, with its seven chunks and the d4 commission error. tab:cs relabelled as three-seed means (13/17/19, ten subjects each), 4-slot rows for all three position types added, random-position rise stated against the human fall, defence-chunk 0.4x stated, "barely moves" removed, "reaches 16" changed to "approaches".
2. tab:cases caption: fixed. Chosen vs drawn groups follow r213_chunks.md section 3 (draw seed 213). Extra rows added (110, 314, 163, 358).
3. Chess compression: fixed. Table chess-bits has all four test sets, Krimp, the recipe run (130.6 / 153.6), the best-arm row, and the exact_alone-off provenance of 147.2. Text states one seed, model cost not charged, BPE-shapes wins on other openings and other wing by 16.9 and 13.4 bits, and that the shifted-set edge over BPE-shapes exists only in the non-recipe run. Recipe values for other openings/other wing do not exist in the sources, so they are shown as dashes.
4. Confirmation seeds: fixed. Table gen has an all-eight block and a confirmation-four block (bootstrap CIs, from r212-analysis.txt). Text states GPT leads at 1,000 on the new seeds (-.014, bootstrap [-.027,-.001]).
5. Regimes section: fixed except r223/r221. r222 reported (endings, nearest function words, size cuts: NO-GO, best .395). Two oracle protocols (A: K=50, B: K=0 N/V/ADJ) defined and reconciled (new Table oracle); "across classes" corrected to "within" implicitly by describing within-class swaps; chess list control .103 vs leaf .085 added. %% PENDING r223 and %% PENDING r221 markers kept with delimited stub subsections.
6. Intro greedy vs IO: fixed. The intro now says the answer is negative on short sentences, and the benefit of IO (posterior weights, longer inputs) is untested.

## Major
- Configuration table: added (Appendix A, Table config, C1-C7, each pointing at its results).
- Parse sets: set S (400 silver, r204-recipe learner) vs set C (300 per-seed + 400 calibration, r215 recipe) defined in Experimental Design and named in every caption; the two cross-seed F1 values (.65-.72 vs .87) and two right-branching values (.49 vs .42) are explained.
- "Further domains" appendix: citation removed; docs/BABI.md holds only first results (600 sentences, no full run), so no appendix was added. v1v2 table no longer lists grammars as a v2 domain.
- Neural sizes: grid, optimiser, early stopping, selection by held-out loss described (from experiments/v3/diag/t3_lm.py). Parameter counts were not recorded and the paper says so. TRELLIS temperature defined in Implementation (parse_gen.py rule) and Eq. weight/posterior given.
- Related Research rewritten: Wolff/GRIDS corrected, Cowan, Krimp, CBL, PARSER, CCM, compound PCFG, CHREST recall simulations (Gobet and Simon 2000; Gobet et al. 2001) added. Missing from the review list and not added: DIORA, TRACX, CCL (not required by the task).
- References: Trestle now @article; MacLellan, Matsakis and Langley key renamed (no pages exist in any listing); CHREST entry replaced by Gobet and Lane (2012) "Learning in the CHREST cognitive architecture" pp. 1920-1922 (title and pages from a repository record, Springer page not opened); Gobet and Simon 1996 cited for what it found (small persistent skill effect on random positions).
- Case studies are now the majority of results: body metric tables reduced to four (gen, parse, chess-bits, cs) plus the v1/v2 comparison; novel, chunks, judge, dial, strict, window, auc, chessmoves moved to appendices. New case material: parse table with chosen/drawn/error groups, chunk biography table, regime-level sample table, chess boards through training, one C&S trial, judge-labelled draws.
- Question list in Experimental Design, with omission/commission link; Implementation section with equations; O1-O2 restored.

## Items from section 1a (all 27)
N1 fixed. N2 fixed. N3 fixed. N4 fixed. N5 fixed. N6 fixed. N7 fixed (.15-.395, typically .30-.38). N8 fixed (protocols A/B). N9 fixed (word "across" removed). N10 fixed (the sentence now names the read level of committed word occurrences, which is what r214 Phase 0 measures). N11 fixed (4,000 point lives in Table gen, caption of Fig gen says so). N12 fixed. N13 fixed (five-seed mean 2.5-2.7, per-seed 2.2-3.2). N14 fixed. N15 fixed (sets named, two agreement numbers explained). N16 fixed. N17 fixed (.89 to .94). N18 fixed (5/8, 4/8, 3/8 stated). N19 fixed (t and bootstrap named in appendix). N20 fixed (seed 29). N21 fixed (tables name C2, plain generator, new distinct sentences). N22 fixed (window and temperature in caption). N23 fixed. N24 partly fixed (.345 with binomial interval, validation numbers in caption). N25 fixed. N26 fixed. N27 fixed.

## Not fixed or pending
- r224 generation re-judging: %% PENDING r224 at the generation table, judge story unchanged.
- r225 language compression: %% PENDING r225 in abstract, contributions and concluding remarks; the title no longer claims compression.
- r223 context length ablation and r221 chess parse-then-regenerate: delimited stub subsections.
- Ablations of the taxonomies or outside score (review 2.9): not run; the paper now says so plainly.
- Chunker baseline for Chase and Simon (2.11a): not run; the paper says so.
- Hand-labelled judge validation on generated sentences (2.14): not run; the paper says so.
- Langley (concepts and chunks) reference could not be verified by any search; flagged in refs.bib.
