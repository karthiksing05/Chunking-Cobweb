# Review 2: "Learning Concepts and Chunks from Raw Experience: Chunks Compress Chess Positions, and Concepts Drive Children's Stories" (TRELLIS v2, revision 2)

Draft reviewed: `confs/trellis-v2/paper/main.tex` at commit 844b4fff (1,426 lines, 42 pages). Line numbers refer to that file. Sources checked: `experiments/v3/results/story_gen/{r224_rejudge, r225_compression_ablation, r223_context_abstraction (+ r223p1/judge{1,2}/report.md), r222_representation, r219_agreement, r214_phase01 Addendum 7}.md`, `chess_regimes/{r220_position_regimes, r221_position_regimes, r226_where_what, r227_file_rank, r228_chunks_concepts_off}.md`, `story_gen/r225_bits_figure.tsv`, `confs/trellis-v2/make_figures_rev2.py` (read, not run), and the four new PNGs (viewed).

## Overall assessment

Revision 2 fixes most of review 1. The headline now uses judge 2. The chess compression table reports every test set and the recipe run. Configurations and evaluation sets are named. Removal controls exist in both domains, and the stale regime text is gone. Most numbers I checked match their sources to the printed digit, including all 24 cells of Table `tab:gen` and the derived ratios in Table `tab:law`.

Three new problems block circulation. All three can be fixed in the text, with no new runs:

1. **The language compression claim names the wrong cause** (L561, L109, L1046). The paper says that "the chunk-and-concept memory codes held-out language in fewer bits than" BPE. r225 shows the opposite: removing both chunks and concepts gives a *shorter* code (28.6 against 30.8 bits at 1,000 sentences, and shorter at every n). The win over BPE comes from the word-window generator. Neither chunks nor concepts contribute to it, and concepts lengthen the code.
2. **"No interior peak" is false by the paper's own Table `tab:dial`** (L831). The syntax-only share is highest at b=7 in all three runs (.255/.211/.265, against .170-.182 at the leaf and .08 at the root). The peak is interior and low.
3. **The `tab:cases` caption misorders its groups again** (L327). It says "the first two groups were chosen ... and the third group was drawn at random". In the table the drawn group is the second, and the third ("Systematic error") was chosen. This is the review 1 N2 defect in a new form.

Several major issues remain about how the claims are framed. The "law" that meaning degrades faster than grammar holds in every run and is real as a description. It is stated more strongly than three runs, a temperature-confounded leaf, nested definitions, an asymmetric judge and no null baseline support. The judge-2 headline is reported only for all eight seeds pooled. On the four confirmation seeds alone, the LSTM leads at 500 sentences on every seed. Case studies are still a minority of the body. At 42 pages the paper is about twice the length of a cognitive-systems paper.

**Verdict: not ready to circulate yet.** It becomes ready to circulate to coauthors once the three blocking text fixes are made (an estimated two to three hours, no new runs). It is not ready to submit until the major items and the length cut are done.

---

## 1. Review 1 items: status

| R1 item | Status | Evidence (rev 2 lines) |
|---|---|---|
| B1 / N1 recall figure from superseded protocol | **Fixed** | `cs_fig1` removed. Table `tab:cstrial` L750-770 is a labelled single example. Fig. `cscurve` L808-814. |
| B2 / N2 `tab:cases` caption reversal | **Partly fixed** | The groups are now labelled in the table (L336, L345, L355), but the caption at L327 again misstates the order. See blocking item 3. |
| B3 / N3-N4 chess compression selective | **Fixed in table, partly in claims** | Table `tab:chess-bits` L670-692 has all four sets, Krimp, the recipe run 130.6/153.6, and the 147.2 provenance. The text at L664-667 is accurate. However, the Contributions (L111, "code shifted positions as compactly as a shape-based byte-pair coder") and the Conclusion (L1052, "match a shape-based byte-pair coder on shifted positions") still claim parity. The recipe run is 3.9 bits worse (153.6 vs 149.7; three-seed r228: 151.5 vs 149.7). The abstract (L50) names only BPE over squares and omits that BPE over shapes is better. Fig. `chesscompress` (L697-698) plots "the better of its two arms" and does not say whether its shifted panel uses the non-recipe run. |
| B4 / N5 confirmation seeds | **Partly fixed** | Old-judge split in Table `tab:genold` L1133-1136. The headline judge-2 Table `tab:gen` pools all eight seeds, and L294 says the split is reported only under the earlier judge. My recomputation of the confirmation-seed split from r224 is in section 2c below. |
| B5 / N6 stale regime text | **Fixed** | L934-938 report r222 (endings, nearest function words, size cuts, .15-.395). |
| B6 / N27 greedy vs IO framing | **Fixed** | L92-95, L608-610. |
| M7 headline judge | **Fixed, with caveat** | Judge 2 is the headline (L461-471). Selection under the old judge is stated (L299, L462). No hand labels (L314, L825), which is stated. The case-study table `tab:samples` (L432-457) still shows only Qwen2.5-3B probabilities. |
| M8 configurations, two parse sets | **Fixed** | Table `tab:config` L1084-1107. Sets S/C at L267-270 and in every caption checked. Small gap: C1 omits the 4,000-sentence settings (edge w5 t2.0, LSTM t1.0, GPT t.9 per r224). |
| M9 language compression, model cost | **Fixed (measurement); claim wrong** | r225 is reported (L540-588, Table `tab:bits`). Model cost is listed beside the code, uncharged, and the paper says so. The attribution is wrong (blocking item 1). |
| M10 no ablation shows chunks matter | **Fixed** | Language: Table `tab:ablation`. Chess: Table `tab:chess-abl`. But the language chunks-off null is partly built in, because the generator never reads chunks directly (r225 Result 2, point 1: "chunks reach the generator only through the context tree they sit in"). The paper does not say this. |
| M11 Chase and Simon overclaims | **Mostly fixed** | Three-seed caption, 4-slot rows, random rise against the human fall (L777-779), defence chunks 0.4x (L647-648). The missing chunker baseline is stated (L780). L112 still says "one divergence", but the paper documents at least three: random-position rise, master-point level 12.9 vs 16, and defence chunks. |
| M12 regime section (oracles, list control, context ablation) | **Fixed** | Protocols A/B at L925-932, Table `tab:oracle`. List control in Table `tab:chesslaw`. Context ablation in Table `tab:ctx`. |
| M13 case studies a majority | **Not met** | See section 4. |
| M14 "further domains" appendix, grammars | **Fixed** | L229. No appendix reference. |
| M15 neural sizes, TRELLIS temperature | **Mostly fixed** | L273-276, L252-255. Parameter counts are missing for the generation runs, and the paper says so. T vs τ is ambiguous in the regime captions: "temperature .7" (L1365) and "temperature 1" (L834, L899, L953) do not say which temperature they mean. |
| M16 O postulates, formulas, related work | **Mostly fixed** | O1-O2 at L143-149. Equations at L237-255. Related research at L1028-1040, with Wolff/GRIDS corrected. There is no parameter-value appendix like ACS Appendix B, but Table `tab:config` partly serves. |
| N7, N9, N10, N13, N17, N19, N20, N22, N26 | Fixed | L936, L929-931, L937, L645, L1276, L1162, L436, L1365, L774-775. |
| N11 Fig. gen 4,000 point | Fixed | Fig. `gen` now plots 4,000 (open markers). |
| N12 C&S seeds | Fixed | L784. |
| N14 C&S defence relations | Fixed | L647-648. |
| N15/N16 agreement and right-branching on two sets | Fixed | L367-376. |
| N18 win counts | Fixed under judge 2 | L463-466. |
| N21 frontier caption and C2 | Fixed | L519, L524-526. |
| N23 list control | Fixed | Table `tab:chesslaw` last row. |
| N24 three "random" move figures | Partly fixed | Table `tab:chessmoves` caption L1405-1407. |
| N25 further domains | Fixed | |
| Minor: PENDING comments, `\pend`, empty Class column | Fixed | No `%%`/PENDING remains. Table `tab:auc` has been redone. |
| References (R1 §6) | Mostly fixed | See section 7. `langley-concepts-chunks` is still unverified. |
| Prose (R1 §4) | Partly fixed | Several constructions remain (section 6). |

---

## 2. Number check

### 2a. Verified against sources (match to printed digit unless noted)

- **Table `tab:gen` (L489-505)**: all 24 cells (means, t intervals, wins) match `r224_rejudge.md` summary and detail tables. Text L463-470 matches: +.015 [+.000, +.031] 6/8; +.020 7/8; meaningful +.022 [+.000, +.044]; 1,000 bootstrap [−.048, −.008] (r224 [−.0475, −.0075]) and [−.037, −.000]; −.042 lower bound; 4,000 gaps .055-.057 and .070-.082.
- **L312-313, L1113-1114, Table `tab:agree`**: pass rates .51-.72 against .41-.58, agreement 79-81%, kappa .55-.62, 77-83% one-sided, +.08 to +.15, GPT .66 to .81, +.076/+.098 against +.061/+.074. All match r224.
- **Table `tab:genold`**: all-eight block matches r224's old-judge reproduction. The confirmation block matches r212-analysis as cited in review 1.
- **Compression, L545-566, Table `tab:ablation`, Table `tab:bits`, L1210-1215**: every value matches r225: bits for all models and arms, ± sd, BPE dictionary 2k-43k bits, 0.2-1.2M / 0.23-1.7M parameters, 8.3k / 102k records, 10-36% overlap, unseen-only ranking, eps, bits per token, ablation fair counts, and content-swap .708/.707/.573/.575.
- **Table `tab:ctx` and L944-950**: the 16 shown cells match `r223_context_abstraction.md`. The .245 maximum and the 1-in-5 observation also match.
- **Table `tab:law` (L873-883)**: I recomputed it from Table `tab:dial`. Every range is correct (for example, b=7: syntax .62-.73, meaning .46-.53, share .42-.45; root share .716-.739 → .72-.74). The values in `make_figures_rev2.py` Figure 2 equal Table `tab:dial`.
- **Table `tab:dial`**: matches Addendum 7 "one-dial story" (leaf τ .7 row; class rows).
- **Table `tab:oracle`, L927-937**: A rows .208/.181/.203, .199/.211/.217 (Addendum 7); morph .31/.33/.34 and B .45/.50/.52 (r219); best tree cell .349/.395 (r222). Purity .4-.58 and .15-.395 (r222).
- **Chess, Table `tab:chess-abl` and L707-712**: matches `r228_chunks_concepts_off.md`. "Within 1.4 to 2.4 bits of the piece-square list" is correct by subtraction (183.3−181.9 … 200.7−198.3 = 2.4). The source text's own "1.4 to 1.8" is the error there.
- **Table `tab:chesslaw` and L975-990**: rows match r220 (content dial), r221 (parse and regenerate .762/.051/.188, .043/.019/.938) and r227 (all where rows, list control). L982 legality .95→.91 and realism .72 of control, L987 .254 at 500, L988-989 .054-.089 and .32 / 68% also match.
- **Chess compression L664-667**: arithmetic correct (51.9, 3.9, 2.5, 16.9, 13.4).
- **Parse text L594-596**: matches Table `tab:parse`.
- **Showcase draws (Table `tab:showcase`)**: all 12 sentences occur verbatim in `r223p1/judge{1,2}/report.md` with the stated judge labels. One row has the wrong setting (2b).
- **`tab:regimecases`**: every sentence occurs in Addendum 7 "Samples (5 per arm), n=4000" under the stated level, at τ 1.

### 2b. Mismatches and misstatements

| # | Line(s) | Paper says | Source says | Severity |
|---|---|---|---|---|
| M-1 | 561 (also 109, 1046) | "the chunk-and-concept memory codes held-out language in fewer bits than a vocabulary of byte-pair chunks does" | r225 Table: full 30.8; chunks off 30.8; concepts off 28.8; **both off 28.6** (n=1000). Both off is shorter at every n (33.7, 31.1, 28.6, 26.7, 24.7 against 35.7 … 26.6). The BPE margin belongs to the gated word-window generator with leaf reads. Chunks add nothing to it and concepts cost about 2 bits. | **Blocking** |
| M-2 | 831 | "the syntax-only share stays between .08 and .27 with no interior peak" | Table `tab:dial`: syntax-only .170 → **.255** → .208 → .140 → .095 (n=1000), .176 → **.211** → .161 → .116 → .082, .182 → **.265** → .225 → .157 → .083. The maximum is at b=7 in every run. The phrase is copied from Addendum 7 and is wrong there too. Write "peaks at b=7 at .21-.27, far below .5". | **Blocking** |
| M-3 | 327 | "The first two groups were chosen …, and the third group was drawn at random" | Table order: Chosen (2 sentences), Drawn (4), Systematic error (1, chosen). The drawn group is the second. | **Blocking** |
| M-4 | 911 (Table `tab:showcase`) | "tim saw his mom": setting "leaf, W'=3, function words at b=4" | `r223p1/judge2/report.md` L66-68: the draw comes from **"W'=3 cb=0 cf=4"**, which is root context with function words at b=4. | Major (mislabelled case study) |
| M-5 | 1365 vs 834, 1098 | Table `tab:dial` / `tab:law` / Fig. `law` leaf = τ .7; class levels τ 1 | Addendum 7: leaf at τ 1 is .558/.206/.235, .580/.193/.228, .541/.218/.241. The law normalizes τ-1 class levels by a τ-.7 leaf. With a τ-1 leaf the meaningless share at the leaf is **.25-.29** (not .21-.23), syntax retained at b=7 is .65-.78 and meaning retained .51-.60. The law still holds at every level, but L114, L863, L1054 ("about one fifth") and Table `tab:law` change. Also `tab:regimecases` shows the leaf at τ 1, which does not match the leaf row it illustrates. | Major |
| M-6 | 115, 940, 1055 | oracle ceiling ".50 to .52" | r219: **.45**/.50/.52 (n=1000 s13 / 4,000 s13 / 4,000 s17). L929 gives all three correctly. Write ".45 to .52". | Minor |
| M-7 | 932 | "the same 250-word vocabulary at 4,000 sentences leaves the ceiling at .18 to .22" | This is protocol A's ceiling (Addendum 7). The sentence sits after the protocol-B discussion and reads as B's ceiling. Name the protocol. | Minor |
| M-8 | 533 | "the share that passes both questions falls from .388 to .287 against .406 and .481" | Those are the *semantics-pass* columns. Passing both questions is .381 → .286 against .405 and .479 (r224). | Minor |
| M-9 | 532 | Old judge: TRELLIS ".464 to .410", GPT ".487 to .570" | r224's eight-seed old-judge values are .484 → .410 and GPT .471 → .570. The .464/.487 come from the earlier four-seed run (PAPER_GENERATION), while the next sentence uses eight seeds. Use one seed set. | Minor |
| M-10 | 559 | "trails ... the LSTM and GPT by 6 to 7 bits" | GPT at 250 sentences: 35.7 − 30.2 = 5.5. Write "5.5 to 6.6". | Minor |
| M-11 | 831 / 1098 | "With one class and no pair gate, 87% to 89% are nonsense" | Root in the one-dial run is a class gate at the root, which admits every pair. Equivalent in effect, but call it what it is. | Minor |
| M-12 | 978 | "R-non ... .94 or more from level 2 up" | r221 keep kc=2 = .938. Write "about .94". | Minor |
| M-13 | 973 | "600 boards per cell" | The r220 content-dial rows in Table `tab:chesslaw` use 500 per seed. | Minor |
| M-14 | 710 | "recall equals the slot count" | Chunks off K=7 gives 6.3-7.0 (r228). Write "is about the slot count". | Minor |
| M-15 | 834-835 | "Draws are the first listed in the run, without selection" | The report lists five random draws per arm, and the table picks by label. For example, at b=4 it skips "tim and sue ran very happy friends", the first nonsense draw. Say "picked by label from the five random draws per arm". | Minor |
| M-16 | 811 vs 792 | "class A" (Fig. `cscurve`) vs "club player" (Table `tab:cs`) | Use one term. | Minor |
| M-17 | 954 | "the largest syntax-only share over all 36 cells is .245" | The table shows 16 of the 36 cells. Say that the rows are a subset (b=7 and b=6 rows omitted). | Minor |
| M-18 | 1093 (C1), Fig. `gen` L476-478 | 4,000-sentence settings not given | r224: TRELLIS edge w5 t2.0, LSTM t1.0, GPT t.9. r224 also notes that under judge 2 GPT's best 4,000 setting is t1.0, which would widen the gap to −.069. | Minor |

### 2c. Judge-2 confirmation seeds (recomputed from r224 per-seed values, seeds 29/31/37/41)

| n | count | TRELLIS / LSTM / GPT | TRELLIS − LSTM, wins | TRELLIS − GPT, wins |
|---|---|---|---|---|
| 250 | syntax | .309 / .287 / .281 | +.022, 3/4 | +.027, 4/4 |
| 250 | meaningful | .191 / .176 / .162 | +.015, 3/4 | +.029, 4/4 |
| 500 | syntax | .366 / .378 / .362 | −.012, **0/4** | +.004, 2/4 |
| 500 | meaningful | .220 / .242 / .214 | **−.022, 0/4** (t CI ≈ [−.039, −.005]) | +.006, 3/4 |
| 1,000 | syntax | .399 / .414 / .427 | −.015, 1/4 | **−.028, 0/4** |
| 1,000 | meaningful | .254 / .275 / .275 | −.020, 1/4 | −.021, 0/4 |

On the seeds that chose nothing, the LSTM leads at 500 sentences on every seed, and its lead on the meaningful count excludes zero. The summary sentence "roughly matches both neural models at 250 and 500 sentences" (L470, L53, L102, L1049) holds for 250 and pooled 500 only. Add this block to Table `tab:gen`, or qualify the 500 claim. **Major.**

### 2d. Figure script (`make_figures_rev2.py`)

- **Fig. `bits`**: reads `r225_bits_figure.tsv` by model key. The keys `unigram`, `bpe`, `trellis`, `trellis_noconcept` (= "concepts off (generator reads)", the arm in Table `tab:ablation`), `kn3`, `gpt` and `lstm` all exist in the TSV. Correct. The figure omits chunks-off (it overlaps the full line) and both-off (28.6, below full). Add both-off, or say in the caption that chunks-off overlaps the full line. Showing both-off makes blocking item M-1 visible.
- **Fig. `law`**: the language values are hard-coded and equal Table `tab:dial`, so they inherit the τ confound (M-5). The chess panel hard-codes r227 where-file kc 0-99. Realism is divided by .945 (identity R-real) and legality is absolute (identity legality = 1.00). Both are correct. The panel shows only the where-file dial, which the authors picked after the grid (r227 caveat). Add where-rank and where-both.
- **Fig. `boards`**: the nine FENs are the r227 boards with the stated labels: where-file kc=0 R-real (2), kc=99 R-real (1), kc=99 R-legal (3), where-both+what kc=99 R-non (3). Correct. Harmless oddity: `Rectangle(..., 1, r and 1 or 1, ...)` always gives height 1.
- **Fig. `gen`**: hard-coded r224 means are correct. The 1,000→4,000 dotted segment joins an eight-seed mean to a two-seed mean taken at different settings. The caption gives the seed counts, but not the setting change.
- The script must run on a compute node (it imports matplotlib and chess), as its docstring says.

---

## 3. Claims against evidence

### 3a. Generation (r224)

The paper is honest that the GPT win does not survive judge 2 (L469-470, L1112-1114). It also discloses selection under the old judge (L299, L462, L1050). Remaining problems:
- The confirmation-seed result at 500 (section 2c) is not reported.
- L463 and L1049 describe 250 as "slightly ahead", which is accurate. The abstract (L53) and Contributions (L102-103) say "match" / "about as often as" for 250-500. Make that "level or slightly ahead at 250; level pooled, behind on confirmation seeds at 500".
- The judge's syntax threshold is permissive by design (L312) and passes adjacent swaps at .13. This inflates every family (L1115), which is stated. Fine.
- `tab:samples` is the only generation case study, and it shows old-judge probabilities. Show the judge-2 syntax and meaning labels, so that the headline case study uses the headline judge and shows the syntax/meaning split.

### 3b. Compression (r225, r228)

- **Language.** The abstract (L48, L51-52) and L567 now say chunking-as-compression is unsupported in short sentences. This is the honest reading. Two corrections are still needed.
  - (i) M-1: the BPE win is not a chunk-or-concept win.
  - (ii) The generator does not read chunks, so the chunks-off null for generation is partly true by construction. Say so in one sentence at L564. Say also that the chunk-free parser's tree F1 is *higher* (.383 vs .347; r225 table). "Removing the chunks changes nothing we measured" (L51) is then slightly inaccurate: one measure improves.
- **Title.** "Concepts Drive Children's Stories" holds for generation (−.04 to −.05 fair count without concepts). For coding, concepts *hurt* (+2 bits) and they make the generator accept more content swaps (.708 vs .573). L566 says this well. The title and abstract (L51 "the concepts carry the results") should name the result that concepts drive: generation.
- **Chess.** Chunks off returns the code to the piece-square list (+47 to +59 bits, three seeds) and removes the game-vs-random recall gap. That supports "chunks carry TRELLIS's compression and its recall pattern". It does not show that TRELLIS's *relational* chunks are a better code than frequency chunks. Any chunker beats a no-chunk code, and BPE over shapes beats TRELLIS on every set, including shifted (149.7 vs 151.5 three-seed, 153.6 seed 13). L667 states this well. The abstract (L50), Contributions (L111) and Conclusion (L1045, L1052) should carry the same qualifier: chunking compresses chess, and TRELLIS's chunks compress about as well as BPE over shapes, never better under the recipe.

### 3c. "Meaning degrades faster than grammar"

The ordering holds in every run (three language runs, one chess dial with two seeds, and the r223 context dial at one seed). As a description of these runs it is supported. As a "law" (section title L816, abstract L55, L113, L859, L1054) it is overstated, for these reasons:
1. **Nested definitions.** Meaningful ⊂ syntax pass, so "meaning retained < syntax retained" is the same statement as "the meaningless share of grammatical output rises". The paper presents it as two findings (L859-863, Fig. `law` left and middle). In chess R-real ⊂ legal. Realism retained < legality retained holds whenever R-legal at a level exceeds about .058 × R-real, which is nearly automatic once any legal-unrealistic board appears.
2. **Asymmetric judge.** The syntax threshold is set permissive "so that well-formed nonsense passes" (L312), and the meaning threshold is strict. Under such a judge, any degradation will cost meaning before syntax. The regularity is therefore partly a property of the thresholds.
3. **No null degradation.** Addendum 7's class-free random swap (`nocls`) turns all-meaningful input into a meaningless share of grammatical output of about .32 (.078 of .245 at n=1000). Random degradation therefore also shows the pattern. Add one null row, either random word swaps or uniform emission at matched syntax pass, so the reader can see whether abstraction along TRELLIS's taxonomy differs from damage in general.
4. **Temperature confound** (M-5).
5. **Small support.** Three runs (one seed at 1,000, two at 4,000), no intervals, one judge. "More data changes no level by more than seed noise" (L864) uses a noise estimate from two seeds at one size.
6. **The oracle ceiling.** The target share .5 equals the perfect-morphosyntax oracle (.45-.52). Under this judge and vocabulary, a generator could reach the target only by matching an oracle that swaps every noun, verb and adjective within exact cells. L940 says the oracle "explains why". It would be more accurate to say that the target was set at the ceiling, so missing it does not separate one mechanism from another. L1055 does say this.

Recommendation: call it a regularity ("abstraction costs meaning before grammar") and state it once, as the rise of P(meaningless | grammatical). Report the τ-matched leaf and one null degradation row. Drop "law" from the section title and abstract, or reserve it for the discussion with these caveats.

### 3d. The chess parallel

The parallel is fair in direction, and the paper reports the negative (R-legal ≤ .25) plainly. It is weaker than the text implies:
- The where-file dial, and the reading "rank = grammar, file = content" (L984-985), were chosen after the grid (r227 caveat: "three representative cells chosen after the grid"). Fig. `law` shows only the dial that fits best.
- The realism judge is a piece-square classifier. The paper's own hypothesis that it is nearly file-symmetric (L990) is untested. L991 then states "the same fact appears in compression" as if established. It is a second untested link. The test is cheap: score real held-out positions shifted one file, without regenerating anything. Run it, or move L991 into the hypothesis.
- Legality in chess is a global count and placement constraint (r220, r226). Grammar in the language study is local, enforced by the pair gate (L949). "The same law" (L973) compares two different kinds of constraint. Say so in one sentence.

---

## 4. The user's asks

| Ask | Status | Notes |
|---|---|---|
| Chunking as compression, scoped to where supported | **Mostly met** | The title and abstract now scope compression to chess, and the language null is reported. Blocking M-1 and the chess BPE-shapes qualifier (3b) are still open. |
| Case studies the majority of body results | **Not met** | Body tables: 15. Case studies: `cases`, `bio`, `samples`, `cstrial`, `regimecases`, `showcase` (6). Metric tables: `gen`, `ablation`, `chess-bits`, `chess-abl`, `cs`, `law`, `ctx`, `chesslaw` (8), plus `v1v2`. Body figures: 16. Case figures: `chunkcases`, `chesstypes`, `chesscase`, `chesscases`, `boards` (5). Metric figures: `chunkcurves`, `gen`, `frontier`, `bits`, `parsecurve`, `chessideas`, `chesscompress`, `cscurve`, `law` (9). Case items are 11 of 28 non-descriptive body items. The abstract's "we present most of the evidence as case studies" (L47) is false by count. The length cuts in section 5 would make it true. |
| Learning curves | **Met** | Fair count (4 sizes), bits (5), parse acceptance (4), chunk types, chess compression, recall (to 27k), chess regimes (flat, in text only). The language law has 2 sizes, and the context ablation 1 size and 1 seed (r223 stopped before n=4000). Say this in the Table `tab:ctx` caption. |
| Judge separating syntax and meaning, used for the headline | **Met** | Judge 2 is the headline. Selection still used the old judge (disclosed), there are no hand labels (disclosed), and `tab:samples` uses the old judge (3a). |
| Three regimes via one or two abstraction settings + context-length ablation, reported honestly as a law with the middle regime not reached | **Met in structure, framing overclaims** | One dial (b) and the context dials (W', b_ctx) reach the two ends. The middle regime is reported as not reached (max .265). Fix M-2 (interior peak) and the law caveats (3c). |
| Laws carried to chess | **Partly met** | The where-file dial reproduces the ordering, with two seeds and the dial chosen post hoc. The judge-symmetry hypothesis is untested. Chunks-off and C&S carry over well (r228). |
| Inspection of chunks | **Met** | Chunk biographies (Table `tab:bio`), chunk-type curves, chess chunk types, chess cases through training, one recall trial. |
| Parse curve | **Met** | Fig. `parsecurve`, Table `tab:parse`. |

---

## 5. Length and venue fit

The ACS reference paper (`confs/acs-26/paper/main.pdf`, 19 pages) has about 16 pages of body, 1.5 of references and 3 of appendix. Its source has 6 figures and 2 tables. TRELLIS v2 has 42 pages, 16 body figures, 15 body tables and 12 appendix tables. A cognitive-systems venue (ACS, CogSci long, Cognitive Systems Research) wants about 16-20 pages of body. Concrete cuts, keeping every body case study:

**Move to a supplementary document (not an appendix):**
1. Appendices B-E in full: Generation Details, Compression Details, Windowed Parsing, Further Metric Tables (Tables `genold`, `agree`, `strict`, `novel`, `bits`, `window`, `parse`, `auc`, `chunks`, `judge`, `dial`, `oracle`, `chessmoves`). That is about 9 pages. Keep only Appendix A (configurations), and add a one-page parameter list to it, as in ACS Appendix B.
2. Fig. `frontier` and the subsection "Novelty costs grammaticality" (L513-528). Replace them with two sentences in the generation subsection.
3. "Where TRELLIS falls behind" (L530-538): cut to three sentences in the fair-count subsection.
4. Fig. `chunkcurves` (L421-428) and Fig. `chessideas` (L650-656). Keep two sentences each in the text.
5. Fig. `chesscompress` (L694-700): it duplicates Table `tab:chess-bits` at one size.
6. Table `tab:law` (keep Fig. `law`, which carries the same numbers).
7. The oracle paragraphs (L925-938): cut to one paragraph of about 5 sentences, with Table `tab:oracle` in supplementary.
8. The greedy-vs-IO paragraph (L608-610): the intro already gives the answer (L92-95).
9. One of Fig. `hierarchies` and Fig. `instances` (both describe v1).

**Shorten in place:**
- Contributions (L101-115, about 15 lines) repeats the abstract and the conclusion. Cut to 5-6 sentences.
- Concluding Remarks (L1044-1065) restates nearly every number. Keep the claims and limitations, without the numbers.
- Experimental Design (L260-314): move the neural grid (L273-276) and the judge-agreement numbers (L312-313) to supplementary.
- Table `tab:chesslaw`: keep identity, where-file leaf/root, where-rank root, where-both+what root, and list counts on. Six rows.

These cuts should save about 10-11 pages of body and 9 of appendix, giving about 20-22 pages in all. The case-study tables (`cases`, `bio`, `samples`, `cstrial`, `regimecases`, `showcase`) and case figures (`chunkcases`, `chesstypes`, `chesscase`, `chesscases`, `boards`) all stay. The curves the user asked for (gen, bits, parse, recall) also stay. After the cuts, case items outnumber metric items in the body.

---

## 6. Prose rules

The constructions found by grep (on the login node) are listed below. Sentence lengths come from a script run on a compute node (srun, overcap) and were then checked by hand. The prose is much improved on review 1: no "rather than", no ", not", no causal "since", and no rhetorical questions remain.

- **Contrast constructions (X but not Y / X and not Y / unlike):**
  - L537 "The tie holds at the fair-count optimum … but fails at the high-grammaticality end."
  - L561 "… codes held-out language in fewer bits than … and it does not code language better than a trigram."
  - L661 "… held-out code lengths that leave out the model, unlike two-part description lengths."
  - L712 "it shows that recall needs the ancestor concepts but does not measure what a smaller concept tree would recall."
  - L1059 "so it is a measurement of that distribution and does not bound what a predictor … could achieve."
  - L1099 (table) "(not the recipe)".
- **What-clauses as nouns:**
  - L642 heading "What the chess chunks encode".
  - L710 "which is what the full system recalls on random positions".
  - L712 "what a smaller concept tree would recall".
  - L1059 "what a predictor built on the same memory could achieve".
  - L384 "contains whatever the learner has seen so far".
- **Demonstrative subjects / cleft-like:** L712 "That control is degenerate by construction."; L775 "That approaches the master's 16 …".
- **Summary closers:**
  - L470 "The surviving claim is that …".
  - L567 "In short sentences the concepts carry the results … so the evidence … comes from chess."
  - L713 "In chess the chunks carry the compression … which supports chunking as compression …".
  - L781 "Ordering by experience alone is weak evidence for the account."
  - L863 "… so grammar survives abstraction longer than meaning does."
  - L940 "The target … is therefore not met …, and the oracle ceiling … explains why."
  - L985 "The file is the content-like coordinate."
  - L1051 "Its competitiveness with neural systems at their scale remains unclaimed."
- **Section cross-references in prose:**
  - L96 "the rest of the paper reports them".
  - L277 "on the fair count below".
  - L294 and L471 "the appendix reports …".
  - L298 "described below".
  - L300 "the generation rule above".
  - L546 "the ranking below".
  - L719 caption "(see text)".
- **Passive voice (sample):**
  - L139 "are named by concepts" (postulate).
  - L171 "was sorted".
  - L194 "is chosen".
  - L252 "is drawn".
  - L327-328 "were chosen", "was drawn", "was chosen".
  - L560 "was built to sample".
  - L776 "is capped".
  - L825 "is validated".
  - L1182 "was extended".
  - L1188 "are judged".
  - L1215 "was selected … was selected".
  - L1276 "are accepted".
- **Sentences over 40 words (hand-checked, approximate counts):**
  - L84-85 (~45).
  - L92-94 (~41).
  - L113-114 (~50).
  - L278 (~45).
  - L382-383 (~45).
  - L533 (~55).
  - L543 (~42).
  - L709 (~48).
  - L821 (~43).
  - L824 (~44).
  - L937 (~45).
  - L946 (~43).
  - L978 (~48).
  - L984 (~47).
  - Several captions exceed 40 words: L326-327, L416-417, L554, L673-674, L784-786, L834-835, L890-891.
- **Banned vocabulary used as jargon:**
  - "arm(s)": L564, L579 (column header "Arm"), L664, L667, L673, L683, L698, L1097, L1100. Use "variant" or "setting".
  - "families": L532, L1114. Use "models".
  - "block": L484, L718, L1118, L1119. Use "part" or "top/bottom rows".
  - "span" is used throughout the postulates (L130, L181-190, L237-249) and never defined. It is the standard parsing term, so keep it, but define it once ("a contiguous run of elements").
- **Undefined or underdefined terms:**
  - "recipe" / "the system's recipe" (L664-674).
  - `exact_alone`, a code identifier in prose (L666, L674, L1099).
  - "edge-exact" (used in the body via Table `tab:config` only; the body says "exact-boundary").
  - "gated token leaf" (C6, C9, L1385).
  - "class gate" / "class-level pair gate" (L1366).
  - "function-word features" (L1366).
  - Level $b$: how levels are cut ("information level") is never stated, only "larger b gives finer classes" (L828).
  - "offsets-only and offsets-plus-squares arms" (L673).
  - "piece-square list" (L688, L707; say: pieces coded independently from their square frequencies).
  - "simulated subjects" (L718, L784; what varies between them?).
  - "glance" and "copy task" (L776, L812).
  - "anchor", "unit", "identity abstraction", "content dial" (L981-996).
  - Window vs reach $W'$: both appear in one caption (L953). Say how they differ.
  - T vs τ in the regime captions (M15 above).

---

## 7. References

- `langley-concepts-chunks` (cited L77) is still flagged UNVERIFIED in `refs.bib`. Volume 11 / 2025 is doubtful. Pat Langley is a coauthor, so ask him for the correct entry before circulation. It is cheap to fix and should not wait until submission.
- `gobet-lane-chrest`: pages 1920-1922 come from a repository record, and the Springer chapter page was not checked. The metadata is plausible (Encyclopedia of the Sciences of Learning, 2012).
- `singaravadivelan-langley-acs`: "To appear", no volume. Fine for a draft.
- `kim-compound-pcfg`: the author is listed as "Rush, A."; use "Rush, A. M.".
- `vaswani-attention`: venue "from memory", but NeurIPS 30, pp. 5998-6008 is correct.
- `feigenbaum-epam`: WJCC 1961, pp. 121-132. Plausible (volume 19 could be added).
- All other entries have plausible metadata. The Crossref/arXiv/ACL-verified entries (Gobet & Simon 1996, 2000; Gobet et al. 2001; Cowan 2001; Vreeken et al. 2011; McCauley & Christiansen 2019; Perruchet & Vinter 1998; Klein & Manning 2002; Lari & Young 1990; Goodman 1996; Sennrich et al. 2016; Hochreiter & Schmidhuber 1997) match what I know of them. REUSED entries (Miller, Chase & Simon, Fisher, Thompson & Langley, Wolff, Langley & Stromsten) are standard and correct.
- Missing, optional: DIORA (Drozdov et al. 2019) and TRACX (French et al. 2011) for completeness. A CHREST recall simulation is cited (Gobet & Simon 2000), but no CHREST baseline was run, as stated.

---

## 8. Ranked issue list

**Blocking (fix before circulating to coauthors; text only)**
1. M-1. The language compression win over BPE is attributed to chunks and concepts, but removing both gives a shorter code. Fix L561, L109, L1046. Also add the sentence that the generator does not read chunks, so the chunks-off generation null is partly by construction.
2. M-2. "No interior peak" (L831) contradicts Table `tab:dial`: syntax-only peaks at b=7 in every run.
3. M-3. The `tab:cases` caption (L327) misorders chosen and drawn groups.

**Major**
4. Judge-2 confirmation seeds are not reported. At 500 sentences the LSTM leads on all four confirmation seeds, and on the meaningful count the interval excludes zero (section 2c). Qualify L53, L102, L470, L1049.
5. The "law" is overstated: nested definitions, an asymmetric judge, no null degradation, three runs, and the oracle ceiling equal to the target (section 3c). Rename it a regularity, state it once, and add a null row.
6. Temperature confound in Table `tab:law` / Fig. `law` / `tab:dial`: the leaf is at τ .7 and the class levels at τ 1 (M-5). Recompute with the τ-1 leaf (the numbers are in Addendum 7). The ordering survives, and "one fifth" becomes "one quarter".
7. Chess compression parity claims (L50, L111, L1052) still overstate. The recipe run trails BPE over shapes on every set.
8. Chess parallel: the dial was chosen after the grid, the judge's file symmetry is untested, and L991 asserts the compression link. Fig. `law` shows one dial only (section 3d).
9. `tab:showcase` row "tim saw his mom" has the wrong setting (M-4).
10. Case studies are still a minority of the body, and the abstract's "most of the evidence as case studies" is false as written. `tab:samples` uses the superseded judge.
11. Length: 42 pages, about twice the venue norm (section 5).
12. `langley-concepts-chunks` is unverified but cited in the introduction.

**Minor**
13. M-6 to M-18. "One divergence" (L112). The Fig. `bits` caption omits that chunks-off overlaps the full line. The Fig. `chesscompress` shifted panel's provenance is not given. The C1 row omits the 4,000-sentence settings.
14. Prose items in section 6, especially the banned "arm", "families" and "block", and the undefined terms list.

**Verdict:** not ready to circulate yet. Fix the three blocking items (2-3 hours, no new runs), and it is ready to circulate to coauthors. Items 4-12 are needed before submission. Items 5, 6 and 8 need small analyses (a τ-matched leaf from existing numbers, a null degradation row, and a file-shift test of the realism classifier), and none needs a new training run.
