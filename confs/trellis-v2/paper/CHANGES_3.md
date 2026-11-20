# Changes in revision 3 (response to REVIEW_2.md)

Main paper: 25 pages (rev 2: 42). Supplement (`supplement.tex`, Tables S1-S17, Figures S1-S4): 13 pages. Tectonic build: 0 undefined references or citations, 0 overfull boxes.

## Blocking
- B1 / M-1 (language compression attribution). Text, abstract, contributions and conclusion now say that the word-window generator earns the margin over byte-pair coding: both-off gives 28.6 bits against 30.8 for the full system (shorter at every n), concepts cost about 2 bits, chunks add nothing. Added the sentence that chunks reach the generator only through the context tree, so the chunks-off generation null holds partly by construction, and the chunk-free parser's higher tree F1 (.383 vs .347). Fig. bits (make_figures_rev2.py, rerun on a compute node, viewed) now draws the both-off line and says chunks-off overlaps the full line. Concepts-off line kept.
- B2 / M-2 (interior peak). Rewritten: R-syn peaks at b=7 in all three runs (.255, .211, .265, against .19-.22 at the temperature-1 leaf and .08-.10 at the root); a small interior peak exists and stays far below .5 and below the oracle ceiling.
- B3 / M-3 (tab:cases caption). Groups named in table order: chosen for plausibility, drawn at random (second), chosen for a systematic error (third).

## Major
- Confirmation seeds 29/31/37/41 reported separately in Table 4 (computed from r224 per-seed values by `analysis_confirmation_seeds.py`, t intervals on 4 seeds, per-seed values rounded to 3 digits). LSTM leads at 500 on all four seeds (syntax -.012, meaningful -.022 [-.039, -.005]). The "matches at 250-500" claim is qualified in abstract, contributions, results and conclusion.
- Law restated as one statement (meaningless share of grammatical output rises with abstraction; meaning pass falls faster because meaningful is a subset of syntax pass). Caveats: permissive syntax threshold; class-free swap null (.32/.45/.43, from Addendum 7 `nocls`); three runs; target .5 equals the oracle ceiling (.45-.52). "Law" removed from the section title, abstract and conclusion; section is "Abstraction Costs Meaning Before Grammar".
- Temperature confound (M-5): the law table and Fig. law now divide by the temperature-1 leaf (.558/.206/.235, .580/.193/.228, .541/.218/.241): leaf share .25-.29 (one quarter), syntax retained at b=7 .65-.77, meaning .51-.60. The temperature-.7 leaf is kept in Table S13 and the text. Fig. law regenerated and viewed; the chess panel now also shows where-rank.
- Chess: where-file dial and the rank=grammar reading stated as chosen after the grid; file symmetry of the judge and the link to shifted-position compression are stated as untested hypotheses (old L991 removed as a claim); global vs local constraint sentence added. "As compactly as" removed from abstract, contributions and conclusion; the text gives 153.6 vs 149.7 (seed 13) and 151.5 three-seed mean vs 149.7.
- tab:showcase: "tim saw his mom" setting corrected to content words at the root, function words at b=4, W'=3 (r223p1/judge2 "W'=3 cb=0 cf=4"). Draws are described as picked by label from five random draws (M-15). Showcase cut to seven rows.
- tab:samples now shows judge-2 labels (M/S/N) looked up in r224 `scores.tsv`, replacing Qwen2.5-3B probabilities.
- Case studies now outnumber metric items in the body (11 case tables and figures: cases, bio, chunkcases, samples, regimecases, showcase, chesstypes+case (one figure), chesscases, cstrial, boards; metric: gen table, gen fig, bits, ablation, parsecurve, chess-bits, chess-abl, cscurve, law). Abstract claim rewritten to match.
- Length: appendices B-E, frontier figure, oracle table, chunk curves, chess ideas lift, chess compression curve, law table, ctx table, full chess regime table, recall values, neural grid and judge agreement moved to `supplement.tex`; instances/hierarchies figures cut to none; oracle paragraphs condensed; contributions and conclusion shortened; Appendix A kept (Table 11).
- `langley-concepts-chunks`: web search found no such paper; the citation and bib entry were removed and `%% TODO(user): Pat Langley to supply` left in the introduction.

## Minor
- M-6 oracle ceiling .45-.52 (Table S15, text). M-7 protocol named (older protocol, 50 protected types). M-8 and M-9 now use the r224 numbers (.381 -> .286, .484 -> .410, GPT .471 -> .570). M-10 "5.5 to 6.6 bits". M-11 root described as one class with the class-level gate. M-12 "about .94". M-13 500 boards (content dial) and 600 (others) in the S17 text. M-14 "about the slot count (6.3 to 7.0)". M-16 "class A player" everywhere. M-17 ctx table caption says 16 of 36 cells. M-18 C1 row has the 4,000-sentence settings; Fig. gen caption says settings and seed counts differ at 4,000. "One divergence" removed.
- Fig. chesscompress caption states its shifted panel uses the better variant (non-standard). Fig. cscurve caption defines glance and copy task.
- Prose: contrast constructions, what-clauses, summary closers, section cross-references, rhetorical questions and passives rewritten; "arm" -> variant, "families" -> models, "block" -> part; "recipe", `exact_alone`, "gated token leaf", class gate, function-word features, level b, span, piece-square list, window vs reach W', T vs tau defined or removed. Sentences over 40 words split. A scan script (compute node) reports no remaining banned constructions except words used in other senses ("short sentences", "above/below" as comparisons).

## Remaining TODOs for the author
- Pat Langley's reference for the "neither saves much alone" argument.
- Define what varies between the "ten simulated subjects" in the recall study (not documented in the result files).
- Define the storage setting behind configuration C7 (`exact_alone`); the paper now says only "one storage setting changed".
- Run the file-shift test of the realism classifier (hypothesis stays untested).
- Main paper is 25 pages against the 20-22 target; further candidates to move: Table 4 confirmation block, Fig. parsecurve, Table 5.

# Revision 4 (closing the open items of revision 3)

Main paper: 22 pages (rev 3: 25). Supplement: 17 pages, Tables S1 to S21 and Figures S1 to S7 (new section "Items Moved from the Main Paper" at the end, so every older S number is unchanged). Tectonic build on a compute node: 0 undefined references or citations, 0 overfull boxes, no "??" in either PDF.

## TODOs closed
- Configuration C7 (`exact_alone`). Defined in `src/trellis2/incremental_cobweb.py` (`_ec_alone`) and passed through `trellis3.py`. The standard chess configuration (`"exact_alone":true`, see `experiments/v3/diag/r228_submit.sh`) scores a new concept that holds one position by its expected correct guesses (sum of squared bag shares per attribute). C7 turns the flag off, so a new single-position concept counts as one correct guess per attribute, as in the original Cobweb. Run: `results/chess_compress/ab-exact-off-shift_s13.json` (147.2 at content level 2 with copy; `xfer-rel-shift_s13` and `r104-tn-noex-shift_s13` give the same value). Main text, Table 5 caption and the C7 row of the configuration table now say this in one sentence instead of "one storage setting changed".
- "Ten simulated subjects" (`experiments/v3/chess_cs1973_exact.py`, `draw_subject`). Each seed learns one memory that the ten subjects share. Subject `rep` draws its own 28 test positions with `random.Random(seed*1000+rep)`: 10 middle games (after White's 21st move), 10 end games (after the 41st) and 8 random boards (pieces of real positions on random squares), split 5/5 between the recall and copy tasks. Nothing else varies (no slot noise, no training-order change). Stated in the Chase and Simon method text and in the Table 6 and Figure captions.
- File symmetry of the judge (r229, `experiments/v3/results/chess_regimes/r229_judge_file_symmetry.md`, script `experiments/v3/diag/r229_judge_file_symmetry.py`). Hypothesis refuted. On the 1,000 real held-out positions the judge scores 95.7% real unchanged, 78.9% after the a<->h mirror (all 1,000 stay legal), 68.4% after one-file shifts (54 of 79 positions that fit) and 41.2% after two-file shifts (7 of 17). Random legal boards score 3%, so the judge is partly tolerant. r227 where-file output (68.1% real at the root, 89.0% at the leaf) lies in the same range. The shape-coder half of the hypothesis stays untested; the paper says so. Supplement Table S21.

## Length (25 -> 22 pages)
- To the supplement: confirmation-seed block of Table 4 (Table S18, key numbers kept in the body), parse-curve figure (Figure S5), language compression figure (Figure S6), language removal-controls table (Table S19), chess removal-controls table (Table S20), Chase and Simon curve figure (Figure S7). Each has a one-sentence pointer with the key numbers in the body. All case studies stay in the body (cases, bio, chunkcases, samples, regimecases, showcase, chess types and case, chesscases, cstrial, boards).
- Figure widths reduced (generation 0.62, ideas 0.56, chunk cases 0.8) and float-page fractions relaxed in the preamble.

## Prose scan (compute node, `build/scan2.py`)
Main paper: mean sentence length 21.9 words; no "rather than", ", not", "but not", "and not", "and also", causal "since", what-clauses, clefts, rhetorical questions, summary closers or section cross-references left (two section references and ten sentences over 40 words fixed). Hits that remain are false positives: "In short TinyStories sentences", "above/below" as comparisons, and merged cite or caption fragments. The word "span" stays in Section 3 where the method defines it (formal term); it is removed from captions. Supplement: mean 20.6; seven sentences over 40 words split.

# Revision 5 (r230 controlled grammar)

Main paper: 23 pages (rev 4: 22). Supplement: 21 pages, Tables S1 to S29 (S22 to S29 new, older numbers unchanged). Tectonic build on a compute node: 0 undefined references or citations, 0 overfull boxes.

## Added
- Main, new subsection "A controlled grammar with an exact judge" at the end of the regimes section, with Figure 6 (`figures/controlled_regimes.png`, script `confs/trellis-v2/make_figure_r230.py`, reads `results/story_gen/r230/judge_p1*.json`). Source of truth: `experiments/v3/results/story_gen/r230_controlled_grammar.md`. The text reports C6 on a 147-word grammar with a rule-based judge: meaningful .72-.73 at the leaf, syntax-only .60-.67 at b=6-7, nonsense .97-.98 at b=0, purity of the b=5-7 classes (syntactic about .9, semantic near chance), C9 with no syntax-only share above .41, and six caveats (func:5 chosen after reading purity tables of five feature sets; two seeds; leaf meaningful share about .75; classes of two words or fewer at b=6-7; syntax-only at b=6 needs n >= 1,000; C9 fails to separate).
- Abstract, regimes introduction, introduction's dial sentence and conclusion now say: unmet on TinyStories (law plus oracle ceiling), met on the controlled grammar. The C6 row of the configuration table names the controlled-grammar setting.
- Supplement, new section "Controlled Grammar" (grammar spec, judge, Tables S22 to S28: judge controls, func:5 purity, purity by feature set, C6 dial with reach, C9, learning curves, examples = first three stored generations per regime per seed, unselected) and a short section holding Table S29. All tables come from `confs/trellis-v2/make_tables_r230.py`, which reads the result files.

## Moved
- Judged-example table of the context-abstraction study (was Table 8 in the main paper) to the supplement as Table S29, to hold the main text near 22 pages. The main text points to it.

## Checks
- No numeric discrepancy between the r230 md and the data files. The md's "R-syn b = 6 and 7 (.64-.67 / .60-.65)" is a seed-by-level pairing (b=7 is .646 / .600). C9 max syntax-only .41 is the W'=2 setting at b=4, seed 17 (.408).
