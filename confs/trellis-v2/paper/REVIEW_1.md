# Review 1: "Chunking as Compression: Learning Concepts and Chunks from Raw Experience" (TRELLIS v2, first draft)

Reviewer stance: tough and fair, cognitive-systems / ML venue (ACS, CogSci, CoNLL-style). Draft reviewed: `confs/trellis-v2/paper/main.tex` (814 lines) and the figures in `figures/`. Line numbers refer to `main.tex`. I checked numbers against `experiments/v3/results/` (story_gen r190/r212/r213/r214_phase01 Addendum 7/r214_judge2/r219/r222, story_parse_eval r216/r218, chess_regimes r217/r220/r221, chess_cs1973_exact), `docs/PAPER_GENERATION.md` and `docs/site/trellis_full.html`.

## Summary of the submission

The paper presents version 2 of TRELLIS, which learns a context taxonomy and a content (chunk) taxonomy with Cobweb from a token stream. Parsing uses inside-outside over binary segmentations, generation draws one word at a time with outside weighting, and learning stores every candidate chunk weighted by its posterior. On TinyStories the paper reports that TRELLIS ties a matched LSTM and beats or ties a matched GPT on a "fair count" of new grammatical sentences from 250 to 1,000 training sentences, and that it falls behind at 4,000. It reports parse case studies, chunk inventories, broken-sentence acceptance, chess chunk types, chess compression on shifted positions, a Chase and Simon recall replication, and a negative result on a syntax-only generation regime.

## Overall assessment

The paper is honest in tone and reports several negative results plainly, which counts in its favour. In its current form it is **not ready**. Two figures or captions misstate the evidence (the recall figure is from a superseded protocol, and the parse case-study caption reverses which sentences were chosen). The chess compression result is reported selectively. The headline generation claim rests on pooled selection and confirmation seeds, and the confirmation seeds alone are weaker than the paper says. Besides these problems, the paper uses at least five different TRELLIS configurations and two disjoint parse evaluation sets without saying so, and it never measures compression in language, although compression is its title and framing.

Recommendation: major revision. Fix the blocking items before any circulation.

---

## 1. Number check (text, tables, captions against sources)

### 1a. Mismatches and misstatements

| # | Line(s) | Paper says | Source says | Severity |
|---|---|---|---|---|
| N1 | 596-597 (Fig. `cs_fig1`), 566-567 | Caption: "the seven chunks TRELLIS held after its first look and the pieces it rebuilt"; text: "recalled 12 ... over repeated trials it reached 12, 18, 25, and 26" | `figures/cs_fig1.png` is a four-panel trial curve (C&S vs TRELLIS M/A/B, middle-game and random, 7 trials). It comes from the **superseded** recall protocol: TRELLIS trial-1 recall ~19/17/14.5 and the full board by trial 3. The site calls that protocol over-generous ("the whole board came back by trial 3"). It shows no board and no chunks, and it contradicts Table `tab:cs` (12.9/9.7/7.2). I found no source for "12, 18, 25, 26". | **Blocking** |
| N2 | 370-371 (Table `tab:cases` caption) | "The sentences come from a block drawn without selection for quality; the last was chosen because it fails." | `r213_chunks.md` §3: sentences 366 ("the tree had no friends") and 178 ("one day tom went to the park") are from **"Showcase (chosen)"**, picked for linguistic intuition. Sentence 219 ("i will help you said tom") is from the **"Random block (drawn, not chosen)"**. The caption reverses the selection. | **Blocking** |
| N3 | 532-535, Table `tab:chess-bits` | TRELLIS 127.7 / 147.2; "two TRELLIS figures come from separate runs of an older recipe" | Site (Familiar shapes, Correction): 147.2 came from a run with `exact_alone` **off**, which the recipe does not allow. With the recipe, shift = **153.6**, which is **worse than BPE over shapes (149.7)**. "Older recipe" hides the fact that the recipe-faithful TRELLIS loses every column. | **Blocking** |
| N4 | 559 (Fig. `fig:chesscompress` caption), 531-535 | "on control and shifted positions" | The figure has four panels. In "Other openings" (138.3 vs 121.4) and "Other wing" (154.6 vs 141.2), BPE over shapes beats TRELLIS by 14-16 bits. Krimp appears in the figure and nowhere in the text or table. The site's own reading: "TRELLIS is not the best coder. It wins only on shifted positions, narrowly." | **Blocking** (selective reporting) |
| N5 | 240 | "choose settings on four and report the other four separately" | The paper never reports the four new seeds separately. On them alone (`r212-analysis.txt`): TRELLIS−GPT at 250 +.042, t CI [−.003, +.087]; at 500 +.030, t CI [−.007, +.068] (both include 0 under t; bootstrap excludes 0). At 1,000 **GPT beats TRELLIS** on the new seeds (−.014, bootstrap [−.027, −.001], 1/4 wins). The "beats GPT at 250 and 500" claim uses pooled selection + confirmation seeds. | **Blocking** |
| N6 | 671 | "Whether classes ... for example from word endings, would reach the regime remains untested." | r222 tested word endings (`+suf`) and nearest function words (`+nf`) with depth and size cuts: **NO-GO**, best tree-class oracle .395. The PENDING comment at 610 says r222 may change the section; r222 is finished. | **Blocking** (stale) |
| N7 | 669 | "Class sources that need no labels reach .30 to .38 (best single cell .38)" | r222 best cell .395 (s17, fn25+lr+suf+nf, b=7). Write ".30 to .40" or report r222. | Major |
| N8 | 663-668 | Paragraph 1: "The cause is the substitution itself." Paragraph 2: a signature oracle reaches .45-.52. | These come from two oracle protocols. Addendum 7 protects the 50 most frequent types (K=50) and swaps all classes. r219's .45/.50/.52 needs K=0 and N/V/ADJ only (with K=50 the morph oracle gives .31-.34). r219 supersedes Addendum 7's verdict. The paper states both conclusions and does not reconcile them. | Major |
| N9 | 665 | "Swapping meaning-bearing words **across** classes" | The oracle swaps **within** classes (Addendum 7 §1). | Minor (factual slip) |
| N10 | 670 | "a level of the **content** taxonomy whose slots ..." | r214 Phase 0 measures the read level C_b of committed **word occurrences** (the context/word read level), not the content taxonomy. Please check and fix. | Minor |
| N11 | 267 (Fig. `fig:gen` caption) | "4,000 sentences use two seeds" | `generation_fair_count.png` plots 250, 500 and 1,000 only. Add the 4,000 point or drop the clause. | Minor |
| N12 | 573 (Table `tab:cs` caption) | "(seed 13)" | Site Figure C1 and its table: three seeds (13/17/19) × 10 simulated subjects. The values are three-seed means. | Major |
| N13 | 519 | "Batteries appear at 2.2 to 3.2 times chance" | `chess_ideas_lift.png` (five-seed mean, as the caption says): battery 2.5-2.7. If 2.2-3.2 is the per-seed range, say so. | Minor |
| N14 | 520 | "This matches Chase and Simon's observation ... pawn chains, castled kings, and same-color clusters made up most of the master's chunks." | Same figure: castled king 1.1×, fianchetto 0.9-1.1×, **pawn-guarded (a defence relation) 0.4-0.6×**. Chase and Simon found defence relations frequent within chunks, so the pawn-guarded split runs against them. | Major (overclaim) |
| N15 | 396 | "agree ... at F1 of about .65 to .72, where random binary trees give .34" | .652/.701/.724 are the 250/500/1,000 values (r213), while the passage is about 1,000 sentences (.72). Random .343 comes from a different 300-sentence set (r203). r218 on the r215 recipe reports cross-seed F1 **.866** at 1,000 (micro, 300 C1 sentences). Two incompatible agreement numbers coexist. | Major |
| N16 | 401-403, 465-468 vs 745-754 | Right-branching F1 .49; TRELLIS .32-.39; 82% non-right-branching | Appendix Table `tab:window`: right-branching .423, full IO .352. r218: non-RB **.96**. The body uses r213 (r204 recipe, 400 silver sentences) and the appendix uses r215/r216/r218 (r215 recipe, 300 C1 sentences). The paper never says so. | Major |
| N17 | 415 (Table `tab:parse` caption) | "Real sentences are accepted .90 to .94 by every score." | r216 C1 acceptance ranges .893 (LSTM, n=500) to .944. Write ".89 to .94". | Minor |
| N18 | 258 | "At 250 and 500 sentences TRELLIS ties the LSTM and beats the GPT on all eight seeds" | Wins against the LSTM are 5/8 and 4/8. The sentence reads as if both relations hold on all eight seeds. | Minor |
| N19 | 292, 712-713 | Strict count: TRELLIS "ties the LSTM ... only at 250" | Under the stated t rule, 500 also passes (CI [−.021, +.002], lower bound > −.03). It fails only under bootstrap (r212_seeds.md). Name the CI that is used. | Minor |
| N20 | 296 | "Over the whole run the share judged grammatical is about .59 ... .58 ... .61" | These are seed 29 only. I recomputed them from `r212-n1000-*/story_s29_n1000_arm{1,4,1}.scores.json`: .586/.578/.606 (TRELLIS t2.0, LSTM t.8, GPT t.8). Say "seed 29". | Minor |
| N21 | 337 vs 331 | Table: "Share of generations with a trigram unseen"; Figure: "share of new sentences containing a trigram unseen" | r190 `tri` is among new distinct sentences. Make the two captions agree. Table `tab:novel` and Fig. `fig:frontier` also use r190's **plain** TRELLIS (windows 3-5, no exact boundary), which is not the headline edge-exact w7 generator. The paper does not state this. | Major |
| N22 | 644-653 (Table `tab:dial`) | Leaf row = "the leaf setting with the highest meaningful share" | The leaf cells use W'=7 at 1,000 and W'=5 at 4,000, with τ=.7 (Addendum 7). The caption gives neither the window nor τ. | Minor |
| N23 | 675 | "falls from .10 at the leaf to .01 at the root" | kc=0 .085 (abs0 .097), root .006. The **piece-square list control (pieces i.i.d.) scores .103**, which equals or beats the leaf, and the paper omits it. In r221 (pending) the list with counts reaches .540 against TRELLIS's .398. | Major (missing control) |
| N24 | 674, 806-807, 794 | ".33 of a uniformly random legal move"; table .345; validation .697 sound (.303 losing) | These are three different "random" figures from different sets. Explain them, or quote one. r217 also gives a binomial ±.065 on 200 positions, which the paper omits. | Minor |
| N25 | 227; 216 | "The appendix reports further domains."; Table `tab:v1v2` Domains: "grammars, children's stories, chess" | No appendix reports further domains, and v2 reports no grammar results. | Major (factual) |
| N26 | 568 | "recall reaches the master's 16 at about 8,000 positions (15.6 ...)" | 15.6 < 16. The two seeds are 16.2 and 14.9, and this happens at twice the experience fixed in advance for M. Say "approaches". | Minor |
| N27 | 89 with 738-742 | Intro: "This paper answers all three [questions]", the first being whether a less greedy parser could repair greedy choices | Appendix Table `tab:window`: greedy (beam 1) F1 .353 vs full .352, ungrammatical AUC .541 vs .542, 93.4% identical trees, at 0.21 of the cost. On every reported measure, inside-outside does not repair greedy choices. State that the answer is negative. | **Blocking** (framing contradicted by own data) |

### 1b. Verified (match the sources)

- Table `tab:gen` and Table `tab:strict`: every mean, CI and win count matches `r212_seeds.md` / `PAPER_GENERATION.md`. The 4,000 row (.319/.358/.362, −.039/−.043) matches PAPER_GENERATION.
- L291 (.32-.42 → .07-.12; .316 → .338), L358-362 (.464 → .410; .48 vs .19; .58-.66) match PAPER_GENERATION. L359-360 (r206 innovation-rate diagnosis) is checked only against PAPER_GENERATION, not against r206 itself.
- Table `tab:samples`: the draws match seed 29 dumps (TRELLIS edge t2.0 `r212-n1000-w7/arm1`, LSTM t.8 `arm4`, GPT t.8 `arm1`). The prompt "tom said sorry cat" is itself ungrammatical. Prompts look unrelated to outputs, so explain what a prompt does.
- Table `tab:novel` values and L323-324 match `r190-frontier-n1000.means.tsv` (w3 t1.0 .339/.764; LSTM t.4 .316/.923; w4 t1.0 .515/.712; LSTM t.6 .517/.811; w5 t.5 .566/.670; GPT t.5 .573/.797). The judge ceiling of .891 matches.
- Table `tab:parse` (all 36 cells, adjacent swap from `r216_figure.tsv`) and L408-409 match r216. Table `tab:auc` matches r216 unpaired AUC.
- Table `tab:window` matches `r218_window_parse.md` (all 7 rows).
- Table `tab:chunks` and L401-402, L447-450 match `r213_chunks.md` (.332, .510/.431, 14, .822, .663, .320-.386, .488; .28-.32 vs .45-.48). The tree cases in `tab:cases` match `r213/cases.txt`. L474-476 ("the bird" biography) matches §5.
- Table `tab:judge` matches `r214_judge2.md` (ensemble test half). L617 (templates syntax 1.00) and L668 (.45/.50/.52) match r219.
- Table `tab:dial` matches Addendum 7 "one-dial story". L663-664 (.208, .181, .203; .199/.211/.217) match Addendum 7 §1.
- Table `tab:chessmoves`, L793-795 match r217. L675 "never exceeds .015" matches r220.
- Table `tab:chess-bits` numbers match the site table. Table `tab:cs` numbers match the site (with the seed error N12). L568-570 (16.7, 11.0/11.1, 2.8, 6.3/3.5) match. L498-499 (7 chunks / 26 of 27, castled king n=861) match.

---

## 2. Claims that overreach, missing controls, unclear protocol

### Major

1. **Two different judges behind one claim.** The generation headline (Tables `tab:gen`, `tab:strict`, `tab:samples`, `tab:novel`) uses a single-axis Qwen2.5-3B judge that passes .89 of real sentences. The regime study uses a Qwen3-8B + phi-4 ensemble with separate syntax and meaning questions. The paper itself reports that under the stricter judges no generator reaches .95 and that the edge-exact TRELLIS sits near .80 against the LSTM's .91-.93 (PAPER_GENERATION, r211). Re-score the r212 dumps with the ensemble judge and report the fair count under it, split into meaningful vs syntax-only. Without this, the headline is judge-dependent and does not use the two-axis judge the paper builds.
2. **Selection and confirmation seeds are pooled** (N5). Report the four confirmation seeds as their own row in Table `tab:gen`. The abstract's "as often as or more often than a matched GPT" fails at 1,000 on the confirmation seeds (GPT ahead, bootstrap CI excludes 0).
3. **Equivalence margin.** "Matches the LSTM" means a CI lower bound above −.03, on a metric whose values are .24-.33, so the margin is about 10-12% of the effect. Justify the margin. At 1,000 the lower bound is −.028.
4. **Seeds are not independent data.** Training sets overlap by 8-14% across seeds (r212). Say so next to every CI.
5. **TRELLIS-only temperature extension.** The grid was extended for TRELLIS after its optimum sat on the edge, and the n=250 optimum has since moved to t4.0-4.5 (r212 selection re-check). TRELLIS sampling temperatures of 2.0-3.5 need an explanation: what does temperature mean for count-based draws? So do the neural model sizes, layers, parameter counts, training steps and the held-out loss used for sizing. None of these appear anywhere in the paper.
6. **At least five TRELLIS configurations, none tabulated.** They are: the r212 edge-exact w7 generator; the r190 plain w3-w5 generator (frontier); the r213 learner (r204 recipe; chunks, trees, body F1); the r215/r216/r218 learner (parse curves, appendix); the regime generator (token leaf, word tree func:100, class gate, τ .7, W' 5/7); and the chess recipes (cs73L max_size 4 for recall; an "older recipe" with `exact_alone` off for compression). The text says "the same machinery". Add one configuration table and point each result at a row.
7. **Two parse evaluation sets** (N16): 400 silver sentences vs 300 C1 sentences, with right-branching .488 vs .423. Unify them, or name the set in every caption.
8. **Compression is the title, but language compression is never measured.** Bits appear only for chess, and there the model cost is "not charged" (site), so the chess numbers are not a two-part or MDL code. A paper titled "Chunking as Compression" should report held-out bits per token on TinyStories for TRELLIS, n-grams, BPE and the neural LMs, together with model description length. Alternatively, retitle it.
9. **No ablation shows that chunks matter.** The paper never removes the content taxonomy or the outside score from the generator, and never removes the context taxonomy (exact words only), so no result establishes that chunks or concepts cause the generation tie. r216 shows that the parse score adds nothing to the generator's discrimination ((a) is below the generator alone in 29/30 cells). In language, chunk inventories are less consistent than raw frequency or BPE (L450). The paper needs at least one result where chunking helps in language, or it should say plainly that none was found.
10. **Chess compression is single seed and reported selectively** (N3, N4). Report all four test sets, Krimp, the recipe-faithful TRELLIS (153.6), seeds, and the model cost.
11. **Chase and Simon.** (a) "Same ordering" is weak evidence, since any learner with more data recalls more. Add a chunker baseline (for example BPE-squares or BPE-shapes chunks in the same seven-slot protocol), or compare with CHREST's published simulations. (b) With 7 slots, TRELLIS random-position recall **rises** 4.2 → 5.5 → 6.0 while people fall 3.0 → 2.5 → 2.0. The text says recall "barely moves" (L570), which it does only on the long stream. (c) The 4-slot rows for end-game and random positions (3.4/3.9/4.5; 2.2/2.8/3.2) fit humans best and are omitted from the table. (d) Middle-game 4-slot recall at the master point is 9.5, against 16.
12. **Raw experience.** Sentences are pre-segmented, 3-8 words, and filtered to a 250-word vocabulary. Chess positions arrive as pieces with relations. "Raw stream" and "no labels" are true in the narrow sense only. Say what preprocessing occurs.
13. **Abstract claim on syntax vs meaning** (L52-53, L410). "Separates grammatical from scrambled better than meaningful from meaningless" holds for **every** model in Table `tab:parse`. Stated as a TRELLIS property it misleads. TRELLIS is worse than the neural LMs on both.
14. **Regime section negative result.** The middle regime is not reached (max .265). The paper reports this honestly. The reconciliation problem (N8) and the stale "untested" sentence (N6) must be fixed. The judge is validated only on constructed controls. Hand-label 100-200 generated sentences per regime to show that the ensemble's syntax/meaning split holds on generator output.
15. **Concluding claim** (L692): "offers a better way to handle context than retaining the whole past" has no support. No experiment compares context handling, and the models that retain the whole past win at 4,000 sentences.
16. **"Where and why it falls behind"** (Contributions, L100-101). The "why" (r206 innovation-rate imbalance) is a diagnosis whose fix did not raise the fair count (L360). Call it a hypothesis.

### Minor

- Line 85-87: the v1 parser "parsed greedily, one candidate pair at a time", yet v2's own greedy beam-1 matches full IO (N27).
- Line 245-246 and Table `tab:judge`: two different "content-word swap" classes are built differently (r216's tag-based swap vs r214_judge2's anomaly-biased swap with animacy/emotion flips and hand-listed verb frames). Name them differently or describe both.
- Line 186: "The generator holds four steps". Say what "reads the prompt" means, given that the samples look unconditioned.
- Line 232: "a neural model sampled at the default temperature copies its training data". Give the number (2-14% distinct at .1-.3 per PAPER_GENERATION).
- Line 229: the n-gram baselines are named but missing from every generation table. The fair count makes them zero by construction, so say this, because the requirement of an unstored trigram is a design choice that favours generalizing models.
- Line 497: "a chunk is a group of up to four pieces". The cap is a hyperparameter (max_size 4), and the site notes that whole formations appear only in parts.
- Line 499: "a fianchetto". The chunk is f7-g6-g8-h7 with no bishop.
- Table `tab:chunks` mixes 1,000-sentence values with "four seeds" ranges and gives no CIs.
- Line 63: "about seven items" alongside 4-slot runs. Cite Cowan (2001) for 4±1.
- PENDING comments (L610, L676) and the unused `\pend` macro remain in the source.

### Undefined or underdefined terms

gain, "share of experiences" (L173-175: write the equation), temperature (TRELLIS), outside score of a position, pair gate, exact-boundary rule / "edge", window width (W vs W'), context window size, committed chunk (L1 says every candidate is stored, so what is committed?), recognized chunk, posterior threshold (Fig. `fig:chunkcases`), level b and "information level" in bits, class-level pair gate, morphosyntactic signature, slot purity, filler entropy, K (protected frequent types), same-tag chance, retrieved-swap reference (Fig. `fig:frontier`), D2c / cap-four chart, glance / label / chunk slot, "model not charged", realism classifier (chess), material criterion.

---

## 3. Does the draft meet the user's stated asks?

| Ask | Status | Notes |
|---|---|---|
| Chunking framed as compression | **Partly met** | The framing is in the title, abstract, intro and conclusion. The evidence is weak: no language bits, chess bits with no model cost, a recipe deviation, and selective test sets. The conclusion's compression paragraph (L689) asserts and never measures. |
| Case studies as the majority of results, metrics a minority | **Unmet** | 12 metric tables (`gen, novel, parse, chunks, chess-bits, cs, judge, dial, strict, window, auc, chessmoves`) against about 6 case items (`samples`, `cases`, Figs. `chunkcases`, `chesstypes`, `chesscase`, `csboard`, and `csboard` is the wrong figure). Results prose is dominated by numbers. The abstract's "we report mostly case studies" is false as written. |
| Learning curves | **Mostly met** | Present: fair count (3 sizes, plus 4,000 in table only), parse acceptance, chunk types, chess compression, recall. Missing: regime shares against n (only 1,000 and 4,000), novelty-grammaticality frontier against n (r191 has 250/500/4,000 and is unused), and the Fig. `fig:gen` 4,000 point. |
| Judge separating syntactic from semantic grammaticality | **Partly met** | Built and validated (Table `tab:judge`), but used only in the regime section. The headline generation result uses the single-axis Qwen2.5-3B judge. |
| Three regimes (semantic / syntax-only / nonsense) via one or two abstraction settings + a context-length ablation | **Unmet** | One abstraction dial (b with the class gate) reaches the two ends only, and syntax-only peaks at .265. **No context-length ablation is reported.** r219 explicitly did not run the W' grid, and Table `tab:dial` silently mixes W'=7 and W'=5. |
| Laws carried to chess | **Partly met** | Chase and Simon recall and chunk ideas are carried over. The novelty-grammaticality law and the regime ladder are not: chess regimes are negative for moves and positions, and the leaf is no better than an i.i.d. piece list (N23). The chess compression "law" holds on one of four test sets with a non-recipe run. |

---

## 4. Prose rules

Measured on body prose (lines 40-695, tables excluded, crude split): **mean ≈ 21 words per sentence** (on target), but **about 20 sentences exceed 40 words**. The longest:

- L670 (~62 words) "A search for a level of the content taxonomy whose slots are both pure ..."
- L447-448 (~60) "The learners find chunks dominated by two-word strings: ..."
- L476 (~54) "After 250 sentences it shares a concept with ..."
- L533-534 (~53) "When the same shapes appear one file over, ..."
- L295-296 (~50) "All three models write new and fluent sentences, and all three also write word salad ..."
- L90-93 (~49) "It parses by inside-outside computation over every binary segmentation ..."
- L172-175 (~48) "A candidate's weight is 2 to the power of its gain divided by a temperature, where ..." (replace with an equation)
- L664 (~48) "Swapping within classes from the learned word taxonomy leaves at most .208 ..."
- L674-675 (~47) "Generating whole positions from the chunk and concept distributions ..."
- L669 (~46), L361 (~44), L52-55 abstract (~44), L104-106 (~43), L519 (~43), L639 (~43), L94-96 (~42), L793 (~42), L85-87 (~41), L732 (~41).

Banned constructions:

- **"X, not Y"**: L512 caption "This is the analysis the parser chose, not a recall trial."
- **"X but not Y"**: L396 "the structure is real but not identical across seeds"; L565 "but only on real positions".
- **"X and not Y"**: L361 "so the tie holds at the fair-count optimum, ... and not at the high-grammaticality end"; L321 "is grammatical and not new".
- **"and also" (near-miss)**: L295-296 "and all three also write word salad"; L57 and L104 "We also report".
- **What-clauses as nouns**: L477 "What a concept contains therefore depends on ..."; L295 "Counts hide what the output looks like."
- **Clefts / demonstrative "This is"**: L512 "This is the analysis the parser chose"; L614 "The middle regime is the interesting one, because ...".
- **Self-answered questions / rhetorical questions**: L694-695, three questions followed by "We hope to address each in later work."
- **Summary closers** ("therefore" wrap-ups that restate the paragraph): L324 "The price of a coarser code is therefore visible at this size, and it shows where the level of abstraction has to be chosen."; L410 "TRELLIS therefore separates syntax better than ..."; L477; L520 "This matches ..."; L570 "so the gain is specific to real positions"; L617 "The judge therefore does not block ..."; L677 "Recognition in TRELLIS does not encode legality, so ..."; L692 (unsupported closer).
- **Causal "since"**: none found.
- **Passive voice** (sample): L115 "called category utility"; L131 "are named by concepts"; L155 "was sorted"; L338 "are judged"; L415 "are accepted"; L569 "is capped"; L699 "was supported" (acknowledgement, fine); L733 "was extended".
- **Section cross-references in prose**: L110 "This section recalls them"; L178 "an appendix studies windowed variants"; L227 "The appendix reports further domains"; L293 and L674 "(Table ..., appendix)".
- **Contributions** (L98-106): a paragraph, as required. Its claim "as often as or more often than" needs the N5 correction.
- **Captions not self-contained**: Fig. `fig:chunkcases` ("a fixed threshold", "recognized chunks"); Fig. `fig:chunkcurves` ("committed chunks", "same-tag chance", seed count); Fig. `fig:frontier` ("retrieved-swap reference"); Fig. `fig:chessideas` (definition of chance, positions, rules); Fig. `fig:chesscompress` (test sets, seed, model cost); Fig. `fig:csboard` (wrong figure); Table `tab:chess-bits` (seed, test-set size, model not charged); Table `tab:dial` (window, τ).
- **Each table names dataset/model/judge**: mostly done. Missing pieces: `tab:chess-bits` (seed, n), `tab:cs` (wrong seed), `tab:chunks` (which recipe and held-out set), `tab:auc` (the Class column is empty, a layout bug at L778-781).

---

## 5. Structure against the model paper (confs/acs-26/paper/main.tex)

The ACS paper runs: Introduction → Cobweb framework (Representation / Organization / Performance / Learning) → theory postulates by the same four headings → implementation by the same four headings → Evaluation (Experimental Design with dependent measures, independent variables and stated questions, then Results) → Related Research → Concluding Remarks → appendices (grammars, implementation formulas and parameter values).

Departures that hurt:

1. **Organization postulates are gone.** v2 has R1-R2, P1-P2, L1, and the `opostulate` counter does not exist. The text says v2 "keeps the representation and organization of version 1", so restate the O postulates (or cite them by number), and keep the R/O/P/L scheme of the ACS paper.
2. **Theory and implementation are merged.** ACS separates postulates from TRELLIS's realization. v2 puts implementation detail (gain, temperature, pair gate, boundary rule, window) in prose under the postulates, without equations. Add an "Implementation" section or an appendix of formulas and parameter values like ACS Appendix B.
3. **No stated questions or hypotheses in Experimental Design.** ACS states its dependent variables (omission/commission) and its independent variables. v2 drops omission/commission without linking the new measures to them. Fair count is a commission measure, and parse acceptance of real sentences is an omission measure. Say so, and state 4-5 numbered questions that the results sections answer in order.
4. **Results are spread over five top-level sections** (Generation, Parsing, Chunks, Chess, Regimes). ACS keeps one Evaluation section with subsections. Consider "Evaluation in Language" and "Evaluation in Chess", each opening with case studies.
5. **Related Research is six sentences** (ACS gives it a full section). It omits the closest modern work: chunk-based learners of child language (McCauley and Christiansen's CBL; PARSER, Perruchet and Vinter; TRACX), unsupervised grammar induction (CCL, CCM, compound PCFG, DIORA), MDL itemset mining (Krimp, which is in the figure), and CHREST's own recall simulations. L683 misstates Wolff and GRIDS: both learn their grammar, i.e. their code, by compression. They do not use "a fixed code".
6. **Appendices hold overflow results**, where ACS appendices hold specification (grammars, formulas). Move the configuration table and formulas into the appendix, and promote results that matter to the main text (the confirmation-seed row, all four chess test sets).

---

## 6. References

| Key | Issue |
|---|---|
| `trestle-maclellan` | Typed `@inproceedings` with `booktitle = {Advances in Cognitive Systems}` and a volume. *Advances in Cognitive Systems* vol. 4 (2016) is a journal volume, so use `@article`, `journal`. Marked REUSED, not re-checked. |
| `matsakis-cobweb` | The key names the second author (first author MacLellan). There are no pages and no volume. The ACS 2022 proceedings entry needs verification. |
| `chrest-gobet-lane` | Page range 556-559 is REUSED and not verified against Springer. Check it. |
| `singaravadivelan-langley-acs` | "To appear", with no volume. Fine for a draft. The title matches the ACS paper. |
| `langley-concepts-chunks` | In refs.bib but never cited. Given the topic it should be cited, likely in the intro next to the predecessor. |
| `gobet-simon-random` | Cited for "on random boards the advantage largely disappears" (L67-69). Gobet and Simon (1996) argue that a small skill effect **persists** on random positions, which fits TRELLIS's rising random recall. Cite it for that and adjust L67-69. |
| `vaswani-attention` | Comment says "venue from memory". NeurIPS 30, pp. 5998-6008 is correct. |
| Missing | Cowan (2001) for 4±1 slots (the 4-slot runs); Vreeken, van Leeuwen and Siebes (2011) Krimp (in Fig. `fig:chesscompress`); McCauley and Christiansen (2019) CBL; Perruchet and Vinter (1998) PARSER; Klein and Manning (2002) CCM / Kim et al. (2019) compound PCFG; Gobet and Simon (2000) or Gobet et al. (2001) for CHREST recall simulations. |

The remaining entries (Miller 1956; Chase and Simon 1973; Fisher 1987; Feigenbaum 1961; Thompson and Langley 1991; Wolff 1982; Langley and Stromsten 2000; Lari and Young 1990; Goodman 1996; Sennrich et al. 2016; Hochreiter and Schmidhuber 1997; Eldan and Li 2023; Qwen2.5 / Qwen3 / Phi-4 reports) have plausible metadata.

---

## 7. Ranked issue list

**Blocking**
1. Fig. `cs_fig1` is from the superseded recall protocol and contradicts its caption, the text and Table `tab:cs` (N1).
2. Table `tab:cases` caption reverses which sentences were chosen and which were drawn (N2).
3. Chess compression: the 147.2 run violates the recipe (the recipe gives 153.6, which loses to BPE-shapes 149.7), and the two test sets where TRELLIS loses by 14-16 bits are omitted (N3, N4).
4. The confirmation seeds are never reported separately, although the text promises it, and on them the GPT claim weakens and reverses at 1,000 (N5).
5. The regime section is stale: the endings test is called untested, but r222 ran it with a NO-GO result (N6).
6. The intro says the paper answers whether a less greedy parser repairs greedy errors, and the appendix shows it does not (N27).

**Major**
7. The headline uses the single-axis Qwen2.5-3B judge, and the two-axis ensemble is never applied to the generation comparison (§2.1).
8. Five or more TRELLIS configurations are used without a configuration table, and two parse evaluation sets are mixed (N15, N16, N21; §2.6-2.7).
9. No language compression measurement, and chess bits omit model cost (§2.8).
10. No ablation shows that chunks or concepts drive any language result (§2.9).
11. Chase and Simon overclaims: seed caption, random recall runs opposite to humans, omitted 4-slot rows, no chunker baseline (N12, N14; §2.11).
12. Regime section: two oracle protocols are presented as one story (N8), the i.i.d. list control is missing (N23), and the context-length ablation the user asked for is absent (§3).
13. The case-study-majority ask is unmet. The abstract says "mostly case studies".
14. The paper refers to a "further domains" appendix that does not exist, and to grammars in v2 (N25).
15. Neural baseline details (size, parameters, training) and TRELLIS temperature semantics are missing (§2.5).
16. Organization postulates are missing, there is no formulas or parameters appendix, and Related Research is thin and misstates Wolff/GRIDS (§5).

**Minor**
17. N7, N9-N11, N13, N17-N20, N22, N24, N26; the minor list in §2; the prose violations in §4; the reference fixes in §6; PENDING comments; the empty "Class" column in Table `tab:auc`.
