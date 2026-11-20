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
