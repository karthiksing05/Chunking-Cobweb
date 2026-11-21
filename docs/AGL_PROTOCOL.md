# Artificial grammar learning: chunk strength against grammaticality (Knowlton & Squire 1996; Meulemans & Van der Linden 1997): protocol and numbers, plus model comparisons

Compiled 2026-11-21 for row P4 of docs/PLAN_PSYCH.md. Every number below comes from a source that was actually opened and read as text, or is marked as arithmetic or as a figure read. **Knowlton & Squire (1996) was obtained in full and is the primary source for section 1, including the grammar, all 23 training strings and all 32 test strings. Meulemans & Van der Linden (1997) was obtained in full (user-supplied PDF, 2026-11-21) and is the primary source for section 2: the grammar (Figure 1), the complete training lists and the 32 test strings of Experiments 2a and 2b (Appendices C and D), the ACS formulas (p.1010), the procedure, N, marginal means with SEMs and the ANOVAs.** The paper prints no per-item ACS values and no cell means (cells appear only as bars in Figures 2-3); the cell means come from Johnstone's thesis (1999) and agree with the paper's marginals and bars (2.8, 2.11). The grammar and all MVdL stimulus lists were checked by program (2.2, 2.5).

## Sources read (and how)

PDFs were text-extracted with pypdf on a compute node (PDF streamed over stdin). Figures and the appendix table were rendered with pymupdf and read as images. Page cites use the page numbering named in the last column.

| Tag | Source | Copy read | Page numbering used below |
|---|---|---|---|
| KS96 | Knowlton & Squire (1996), "Artificial grammar learning depends on implicit acquisition of both abstract and exemplar-specific information", JEP:LMC 22(1):169-181 | Full journal PDF, public domain, lead.u-bourgogne.fr/wp-content/uploads/2023/08/knowlton_squire_1996.pdf (13 pp). Text layer good. The Appendix (stimulus list) and Figure 1 (grammars) are images and were read from page renders | Journal pages 169-181 as printed (PDF page n = printed page 168+n) |
| MVdL97 | Meulemans & Van der Linden (1997), "Associative chunk strength in artificial grammar learning", JEP:LMC 23(4):1007-1028 | User-supplied journal PDF (22 pp, one per printed page). Text layer good (appendices render "VV" as "W"; corrected). Figure 1 (grammar), Figures 2-3 (bars) and Appendices C-D were read from page renders | Journal pages as printed (PDF page n = printed page 1006+n) |
| JS99 | Johnstone (1999), PhD thesis "Structural versus processing accounts of implicit learning", University College London | discovery.ucl.ac.uk/id/eprint/10107529/1/Structural_versus_processing_a.pdf (full thesis; only Ch. 1-2 and the ACS definition in Ch. 3 were used). Chapter 2 re-analyses MVdL97 Experiments 2a and 2b and reproduces its summary table. Figure 2 (grammar) is an image and was read from a render | Thesis printed page numbers (PDF page n = printed page n-1) |
| BD03 | Boucher & Dienes (2003), "Two ways of learning associations", Cognitive Science 27:807-842 | users.sussex.ac.uk/~dienes/Boucher & Dienes 2003.pdf (31 pp, clean text). Used for the exact scoring rules of competitive chunking and the SRN | Journal pages as printed (footers 807-842) |
| KD05 | Kuhn & Dienes (2005), "Implicit learning of nonlocal musical rules", JEP:LMC 31:1417-1432 | users.sussex.ac.uk/~dienes/Kuhn & Dienes JEP 2005.pdf (16 pp). Used only for a verbal definition of ACS and anchor ACS as applied to MVdL97 | PDF page 2 (introduction) |
| PPVG02 | Perruchet, Vinter, Pacteau & Gallego (2002), "The formation of structurally relevant units in artificial grammar learning", QJEP 55A | lead.ube.fr copy (19 pp). Abstract and text skimmed only. PARSER applied to AGL segmentation, not to endorsement | |

## Not found / not read

- **MVdL97** is now read in full (see the sources table). What it does **not** contain: per-item ACS values (only cell means of the recomputable measures), cell means as numbers (bars only), cell SDs, the exact predetermined (random) presentation orders, the response key or button, whether the 125-string series of Exp 2b was shown once or twice, and the within-group 2 x 2 ANOVAs for 2a and 2b (only the 2 x 2 x 2 with the control group). The 1a/1b appendix lists (Appendices A, B) were read as text but not transcribed or checked by program.
- **Johnstone & Shanks (1999)** (JEP:LMC 25:524-531, the comment on MVdL97): journal text not obtained; JS99 (the thesis) is the same author's longer treatment and was used instead.
- **Servan-Schreiber & Anderson (1990)** (JEP:LMC 16:592-608): no free full text located (search returned only citing papers). Not read. What is known about the model comes from BD03's reimplementation (section 3), and BD03 says its implementation differs from the original in four named respects (BD03 p.818).
- **PARSER** (Perruchet & Vinter 1998): PV98 does not simulate AGL endorsement. The only PARSER-on-AGL paper located is PPVG02, which reproduces a segmentation measure on a finite-state grammar, not endorsement of high/low chunk strength items. No PARSER numbers for KS96 or MVdL97 were found.
- **Simulation papers on these two datasets:** none located. BD03 states explicitly that "it has yet to be shown whether the CCN could simulate the Knowlton and Squire data" (BD03 discussion, about p.834) and notes Kinder & Shanks (2001) simulated KS94/KS96 amnesic versus control with an SRN by lowering the learning rate. Kinder & Shanks (2001) itself was not obtained. Boucher & Dienes, Kinder & Assmann, Pothos: no numbers for KS96 or MVdL97 found.
- All figure-only values are listed at the end.

---

## 1. Knowlton & Squire (1996), Experiment 1: the primary protocol

### 1.1 Design

Grammaticality (grammatical G versus nongrammatical NG) crossed with chunk strength (high H versus low L) at test, within participants: four cell types GH, GL, NGH, NGL, **8 test items per cell, 32 test items** (KS96 p.172, "Materials"). One training phase, one test phase. Two groups: 18 controls and 11 amnesic patients; **the amnesic data are secondary and the controls are the human target** (KS96 p.171).

### 1.2 Participants (controls, the target)

- N = 18 controls, employees or volunteers at the San Diego VA Medical Center or retirement community members; age M = 63.8 years (range 51-71), education M = 14.4 years (KS96 p.171). They were matched to the 11 amnesic patients.
- Amnesic patients: N = 11 (4 Korsakoff, 2 other diencephalic, 5 hippocampal), age M = 65.1, listed in KS96 Tables 1-2 (pp.171-172). Not needed for a simulation.
- Experiment 2 and 3 used other participants (see 1.9).

### 1.3 Grammar (KS96 Figure 1A, p.172; "Grammar A", from Abrams & Reber 1989)

Read from a 330-dpi render of the figure. Alphabet **{J, T, V, X}** (4 letters). Five states S1-S5, start at S1 ("IN"), strings are formed by traversing the diagram from the IN arrow to an OUT arrow, **one letter per transition** (KS96 p.172).

| From | Letter | To |
|---|---|---|
| S1 | X | S1 (self loop) |
| S1 | V | S2 |
| S2 | J | S3 |
| S2 | T | S5 |
| S2 | X | S4 |
| S3 | T | S1 |
| S5 | V | S4 |
| S4 | J | S4 (self loop) |

Accepting (OUT) states: **S3, S4, S5**. S1 and S2 are not accepting.

Check: this transcription accepts all 23 distinct training strings and all 16 grammatical test items, and rejects all 16 nongrammatical test items (checked by program on a compute node, 2026-11-21). The Figure 1B grammar (Grammar B, Experiments 2-3 only) is in 1.9.

### 1.4 Training strings (KS96 Appendix, p.181; copied from a page render)

The Appendix lists the training items in four columns. **The list has 26 rows but only 23 distinct strings**: XXVJ, VJTVX and XXXVTV each appear twice in the printed table (marked * below). The text says "twenty-three training items" (p.172). This is an ambiguity (see Simulation notes): either the table repeats three strings by typesetting error, or the training set really contained three repeated strings. Row order as printed, column by column:

```
col 1: XXVT   XXVXJJ   VXJJ     XVJTVJ   XXVXJ    XVX
col 2: VXJJJJ XVT      XXXVT    VJ       XVXJJJ   VJTVTV
col 3: XXVJ*  VTVJ     VJTVX*   XXXVTV*  XXVJ*    VJTVX*
col 4: XXXVTV* XVXJJ   VT       VJTVXJ   XXXVX    VJTXVJ   XVXJ   XXXXVX
```

23 distinct strings, lengths 2 to 6 ("two-six letters", p.172).

### 1.5 Test strings (KS96 Appendix, p.181), with the paper's chunk strength (CS)

Item order within a cell is as printed. CS is the paper's own value.

| GH | CS | GL | CS | NGH | CS | NGL | CS |
|---|---|---|---|---|---|---|---|
| XXVXJ | 10.4 | VJTVT | 6.0 | VJTV | 7.0 | XXJJ | 6.8 |
| XVTV | 6.8 | VTVJJ | 5.1 | XXV | 12.3 | VXJTJ | 4.9 |
| VXJ | 9.3 | VTVJ | 5.6 | XVXV | 10.0 | XXVJJ | 6.2 |
| XXVTV | 8.0 | XVJTVT | 6.7 | XVXVJ | 9.1 | JXVT | 5.0 |
| XVJTVX | 7.4 | VTV | 5.0 | XXVJJJ | 7.8 | XXTX | 2.8 |
| XXVTVJ | 7.7 | XVTVJ | 6.7 | XJJ | 7.0 | TVJ | 6.7 |
| VJTXVX | 6.8 | XVTVJ | 6.1 | VXVJ | 8.2 | VXJJX | 5.9 |
| VX | 12.0 | VTVJJ | 5.2 | XVXT | 7.0 | VJJXVT | 4.9 |
| **M** | **8.6** | **M** | **5.8** | **M** | **8.6** | **M** | **5.4** |

Transcription notes. The text layer of the Appendix scrambles the columns, so the table was read from the page image. GL row 7 is printed as XVTVJ and GL row 6 is also XVTVJ (two identical strings, as printed; likewise VTVJJ appears twice in GL, rows 2 and 8, with different CS values 5.1 and 5.2). The CS values differ for identical strings, so one of these is a typesetting error in the original; the string content is copied as printed. NGL row 3 is printed "XXVJJ" (the text layer gives "XXWJJ", an OCR error; XXVJJ is nongrammatical under the grammar above, as intended). Nongrammatical items were made "by introducing an error in each of 16 different grammatical items" (p.172). All 16 NG items violate the grammar in this transcription.

### 1.6 Chunk strength definition (exact; KS96 p.172)

"Chunks were defined as the bigrams and trigrams that appeared in the item. Thus, the item XXVXJ consisted of the chunks XX, XV, VX, XJ, XXV, XVX, and VXJ" (the printed text has typos "XI" and "VJX" for XJ and VXJ). "The associative strength of a chunk was defined as the number of times it had appeared in the training items. The chunk strength of each test item was calculated by averaging the associative strengths of each chunk in the item."

So **ACS(item) = (1/|C|) sum over c in C of n_train(c)**, where C is the multiset of all bigrams and all trigrams of the item (pooled; the item's chunk count is its length-1 bigrams plus its length-2 trigrams) and n_train(c) is the number of occurrences of chunk c in the training strings, at any position. Whether "number of times" counts each of the 23 distinct strings once or each of the 46 presentations (23 x 2) is not stated. Items of length 2 have one chunk (the bigram only). No anchor or novelty measure is used in KS96.

Reproduction check (program on a compute node): with 23 distinct training strings, pooled bigram+trigram averaging, counting every occurrence, my values are close to the printed CS but not identical (for example XXVXJ 10.3 versus 10.4 and VX 12.0 versus 12.0, but differences up to about 1.9 for some NGL items such as XXVJJ, printed as 6.2 and recomputed as 8.1). Counting the 26 printed rows instead moves most values further from the paper. Averaging bigrams and trigrams separately and then averaging the two means fits worse. Conclusion: the pooled-mean definition with 23 strings is the closest, the paper's own numbers are the reference, and **the recomputed values should not be used in place of the printed ones**. The cell means are matched by the paper's design: mean CS of grammatical items = 7.2 (8.6 and 5.8), nongrammatical = 7.0 (8.6 and 5.4), high = 8.6, low about 5.6 (KS96 p.172). The mean CS of a high item is 8.6 in both G and NG, and of a low item 5.8 versus 5.4.

### 1.7 Procedure (KS96 pp.172-173)

1. **Exposure.** Training items were printed on individual index cards. Each item was shown for **3 s**, then removed; the participant tried to reproduce it on paper from memory. If wrong, the item was shown again (a second 3 s exposure is implied; the text says "shown the item again") and a second attempt allowed, then a third showing and a third attempt, then the next item. The 23-item set was **repeated once, for a total of 46 trials**. Reproduction accuracy: controls 88.3% correct on the first attempt, 97.6% in three attempts; amnesic patients 86.2% and 95.4% (p.173). Item order within the set is not stated.
2. **Delay.** Testing began 5 min after the training phase (p.173).
3. **Instructions at test (quoted in substance, p.173).** Before the test, participants were informed that the items they had seen "had been generated by a set of rules, which allowed only certain letters to follow other letters", and that for new items "their task was to decide whether each item was or was not formed according to the same rules". They were told the rules were very complex and to base judgments on a "gut feeling" as to whether each test item obeyed the rules.
4. **Test.** The 32 test items (four types in mixed order, printed on index cards) were displayed one at a time; the participant judged each "correct" or "incorrect". No feedback, no confidence rating, no time limit stated, no second pass. Test order is not given; the Appendix gives only per-cell lists.

### 1.8 Human results (Experiment 1, controls; KS96 Table 3, p.173)

Endorsement rate = mean percentage of items of that type judged "grammatical/correct". SEM is across participants.

| Cell | Controls M (%) | SEM | Amnesic M (%) | SEM | Items |
|---|---|---|---|---|---|
| Grammatical, high chunk strength (GH) | 64.6 | 3.7 | 61.4 | 3.6 | 8 |
| Grammatical, low chunk strength (GL) | 62.5 | 6.0 | 63.7 | 4.3 | 8 |
| Nongrammatical, high chunk strength (NGH) | 54.9 | 5.6 | 56.8 | 6.4 | 8 |
| Nongrammatical, low chunk strength (NGL) | 28.1 | 3.8 | 30.7 | 3.9 | 8 |
| All 16 grammatical | 63.5 | 3.5 | 62.5 | 3.0 | 16 |
| All 16 nongrammatical | 41.5 | 4.1 | 43.8 | 4.2 | 16 |
| All 16 high chunk strength | 59.7 | 4.0 | 59.1 | 2.1 | 16 |
| All 16 low chunk strength | 45.3 | 3.9 | 47.2 | 3.3 | 16 |

Statistics (controls, N = 18; amnesic N = 11), KS96 p.173:
- Grammatical 63.5% versus nongrammatical 41.5%, t(17) = 5.27, p < .01. Amnesic: 62.5 versus 43.8, t(10) = 3.28, p < .01.
- High 59.7% versus low 45.3% chunk strength, t(17) = 3.10, p < .05. Amnesic: 59.1 versus 47.2, t(10) = 3.84, p < .05.
- Effect sizes (point-biserial): controls .50 for rule adherence and .39 for chunk strength; amnesic .52 and .27. The two effects did not differ within either group (Zs < 1.14, ps > .2); the groups did not differ on either measure (Zs < 0.52).
- Three-way ANOVA, Group x Grammaticality x Chunk strength: grammaticality F(1,27) = 34.26, MSE = 331.71, p < .01; chunk strength F(1,27) = 16.6, MSE = 284.42, p < .01; **grammaticality x chunk strength F(1,27) = 17.54, MSE = 273.9, p < .01** (chunk strength exerts most of its effect on nongrammatical items: NGH 54.9 versus NGL 28.1, but GH 64.6 versus GL 62.5). No effect of group, no interactions with group (Fs < 1).

The error-type analysis (KS96 Table 4, p.174), nongrammatical items only, mean % endorsed (controls; amnesic in brackets): high strength and location error 60.0 (61.8); high strength and chunk error 42.6 (51.5); low strength and location error 41.7 (45.5); low strength and chunk error 25.0 (25.8). SEMs controls 7.3, 7.0, 7.3, 4.1; amnesic 6.3, 12.2, 8.1, 5.2. Effects: chunk strength F(1,27) = 18.24, MSE = 570.1, p < .01; error type F(1,27) = 10.31, MSE = 680.3, p < .01; no interaction (F < 1). A "chunk error" is a nongrammatical item containing an impermissible bigram or trigram (zero strength); a "location error" is one made of permissible chunks in impermissible positions (p.174). The item-level assignment of each NG item to error type is not given in the Appendix.

A simulation target in one line: GH 64.6, GL 62.5, NGH 54.9, NGL 28.1 (controls), with grammaticality and chunk strength both significant and an interaction in which NGL is the outlier.

### 1.9 Experiments 2 and 3 (secondary; read for completeness)

- **Experiment 2** (recognition of chunks, p.174-175): 9 amnesic patients, 12 other controls. Study items: 23 training strings of Grammar A or of Grammar B (half of participants each), same 3 s plus reproduction procedure (training set shown once? The text says "The training procedure was the same as in Experiment 1", so presumably repeated once as in Exp 1; not explicitly restated). Test: 46 bigrams and trigrams, 23 that occurred in training and 23 constructed from the same letters that had not; participants judged whether each had appeared in the training strings. For Grammar A: 9 old bigrams and 14 old trigrams, 6 new bigrams and 17 new trigrams; for Grammar B: 10 and 13, 5 and 18. Mean chunk strength of the old test chunks 8.3. Result: controls 67.4% +/- 2.7% correct, amnesic 57.7% +/- 3.6%, t(19) = 2.20, p < .05 (p.175).
- **Experiment 3** (transfer to a new letter set, pp.175-177): N = 14 controls and the same 9 amnesic patients. Four grammars (A-D), two letter sets each (A: JTVX and HNPS; B: BFLZ and DGKW; C: BCJW and HNPS; D: FTVX and DGKL; p.176), 23 grammatical training strings, 23 grammatical test strings, 23 nongrammatical test strings with errors in one position, per grammar. Percent correct classification, controls: same letters 59.9 +/- 1.5 (Grammars C, D) and 65.5 +/- 2.6 (A, B); different letters 56.1 +/- 1.5 and 60.7 +/- 1.4. Amnesic: same 57.2 +/- 2.6 and 60.9 +/- 3.1; different 54.6 +/- 1.3 and 57.3 +/- 2.2 (p.177). All above chance (ts > 2.77). The training and test strings of Experiment 3 and Grammars C and D (Figure 2, p.176) were **not extracted** and are not needed for P4.
- **Grammar B** (Figure 1B, p.172; Experiments 2-3 only; from Knowlton, Ramus & Squire 1992): alphabet {B, F, L, Z}; S1 -L-> S1 (self), S1 -B-> S2, S2 -L-> S5, S2 -Z-> S4, S2 -F-> S3, S4 -F-> S4 (self), S4 -B-> S3, S3 -Z-> S1; accepting S3, S4, S5. Read from a 330-dpi render; not verified against any string list.

---

## 2. Meulemans & Van der Linden (1997), Experiments 1a/1b and 2a/2b: the primary source

Everything in 2.1 to 2.10 comes from the journal PDF itself (JEP:LMC 23(4):1007-1028, 22 PDF pages, one per printed page 1007-1028; PDF page n = printed page 1006+n), text-extracted with pypdf on a compute node, with Figure 1 (grammar), Figures 2-3 (bars) and Appendices A-D (stimuli) read from page renders (170 dpi; grammar at 400 dpi). The text layer of the appendices is clean in column order but renders **"VV" as "W"** (for example "MXRVVW" is MXRVVVV); every string below has W expanded to VV and was checked against the page image. Page cites are the printed page numbers. Experiments 2a/2b are the simulation target. Items marked "figure read" are bar heights read from Figures 2-3 (the paper prints no cell means; it prints marginal means only).

### 2.1 Design summary

| | Exp 1a | Exp 1b | Exp 2a | Exp 2b |
|---|---|---|---|---|
| Training strings | 16 (KS94 items), 3-7 letters (p.1010) | 125 of the 150 grammatical strings of length 3-7 (p.1012) | **32**, 3-7 letters (p.1015) | **125**, 3-7 letters (p.1018) |
| Share of the 150 grammatical strings | 10.7% (p.1021) | 83.3% (p.1021) | 21.3% (arithmetic, 32/150) | 83.3% (arithmetic) |
| Test strings | 32 (4 cells x 8) | 32 (4 x 8) | 32 (4 x 8) | 32 (4 x 8) |
| N, trained group | 20 | 20 (new group) | **20** | **20** |
| N, no-learning control | none | none | 20 (separate participants) | 20 (separate participants) |
| Chunk novelty controlled | no | no | yes (associated cells have none) | yes (all cells: none) |

Between-participant design for training amount (each experiment has its own participants; 2a and 2b each have 40 undergraduates, 20 experimental and 20 control, p.1015, p.1018). Grammaticality (G/NG) and associativity (A/NA) are within participants.

### 2.2 Grammar (MVdL97 Figure 1, p.1008; from Vokey & Brooks 1992, p.330; identical to KS94 Exps 2a/2b, p.1010)

Read from a 400-dpi render. Alphabet **{M, R, T, V, X}**. One start state, nine others, **three exits** ("OUT": G, H, I). One letter per transition; a string ends when an exit is reached (p.1008 caption), and "each letter of the grammar can constitute an exit" (p.1010).

State names (mine; position in the figure): S0 start; A top-left, B middle-left (T self loop), C bottom-left; D top-middle, E centre, F bottom-middle; G top-right (V self loop, OUT), H middle-right (OUT), I bottom-right (R self loop, OUT).

| From | Letter | To | | From | Letter | To |
|---|---|---|---|---|---|---|
| S0 | M | A | | E | R | B |
| S0 | V | C | | E | R | G |
| A | V | D | | E | T | I |
| A | X | E | | F | V | E |
| B | M | A | | F | T | I |
| B | T | B | | G | V | G |
| B | V | C | | G | M | H |
| C | M | E | | H | (none; exit only) | |
| C | X | F | | I | X | H |
| D | R | G | | I | R | I |
| D | X | E | | | | |

Accepting (OUT): **G, H, I**. That is **20 letter transitions plus 3 exits**. **State E has two R transitions (to B and to G)**, so the grammar is not deterministic on letters; a parser or simulator must follow both (the strings of the paper are generated by choosing among them).

Comparison with the JS99 redrawing (old section 2.2): **identical**. Under the mapping JS99 1=S0, 2=A, 3=B, 4=C, 5=D, 6=E, 7=F, 8=G, 9=H, 10=I, every one of the 20 JS99 transitions (1-M-2, 1-V-4, 2-V-5, 2-X-6, 3-M-2, 3-T-3, 3-V-4, 4-M-6, 4-X-7, 5-X-6, 5-R-8, 6-R-3, 6-R-8, 6-T-10, 7-V-6, 7-T-10, 8-V-8, 8-M-9, 10-X-9, 10-R-10) is in the table above and vice versa, and the exits 8, 9, 10 are G, H, I. The earlier worry that JS99's "17 transition rules + 6 terminal rules" did not match was a counting artefact: 20 + 3 = 23 = JS99's total of 23 rules (JS99 p.35). **Check by program** (compute node, 2026-11-21; script in /coc/flash10/ksingara3/tmp_mvdl/chk.py, not part of the repo): this grammar (as an NFA) accepts all 32 training strings of 2a and all 125 of 2b (157 strings, 0 rejected), accepts all 16 grammatical test strings of 2a and of 2b (32 of 32), and rejects all 16 nongrammatical test strings of 2a and of 2b (32 of 32). It generates **exactly 150 distinct strings of length 3 to 7**, matching the paper's "a total of 150 items from three to seven letters" (p.1011). The grammar is therefore confirmed against the primary source.

### 2.3 Training strings

Each experiment's training list is printed in the appendix as a table; the printed table is the only order given. The order shown to participants was "predetermined random" for half of the participants and **the inverse order for the other half** (p.1010, p.1012); the predetermined order itself is not printed, so the layout below is the table's. In both lists **no string appears twice** (32 distinct, 125 distinct); no training string appears among the test strings of its own experiment. Lengths: 2a: 3 letters x1, 4 x1, 5 x2, 6 x9, 7 x19; 2b: 3 x5, 4 x16, 5 x17, 6 x27, 7 x60 (arithmetic from the lists).

**Exp 2a, 32 training strings** (Appendix C, p.1027; printed as 4 columns of 8, shown here as the printed rows):

```
MXRVVVV   MVXRMXT   MVRVVV    VMRVXT
VXTRRRR   MXRVXVR   VMRVVVM   VMRMVR
VXVRVMT   MXRMVRV   MVXTRX    MVXRVMR
MXRVMRM   VMRVMRV   VMRVVM    VMRVXVT
MVXTRR    MVXRM     MXRMVRM   MXTRRRX
VMRTMXR   MXTRR     MVR       VMTX
VXVTRR    MXRMVR    MVXRVXT   VXVTRRX
VMRVMTR   MVXRMXR   MXRVXTR   MVXRVV
```

**Exp 2b, 125 training strings** (Appendix D, p.1028; printed as 5 columns of 25, shown here as the printed rows):

```
VXVTX     VXTRRX    MVXRVVV   VXVRVM    VMRMXRM
VXVT      VXTRX     MVXRMXT   MXRVXVR   MXRVXTR
VXVTR     VMRTVXT   MVRVVM    VMRMXTR   MXRM
VXVTRRR   VXTX      VXTRRR    MXRMVXT   MXRMXR
VXVTRR    MVXTRRX   MXRMVR    MXRVVV    MXRVMT
VMTRX     VXVRVVV   VXVRVVM   MXRMXT    MXRVV
VMRTMXR   VMTRR     MVXRM     MVRVM     MVXRVVM
VMTX      MVRVVVV   VXVRV     VMRVXTX   MXRVVM
VMTRRRX   MXTRRRX   MXRVXVT   MXRVMTR   VMRVXTR
VMTRRX    MXTRRR    MXRVMTX   MVXTRR    VMRVMT
MXRTVMT   MVRVV     MXRVVVV   MVXRVV    MXRVXT
MXTX      MXRTVMR   MXRMVRV   MXRVVVM   VMRMXRV
MXTRX     VXTRRRR   MXTR      VMT       VMRMXR
MXTRRX    MXRMVRM   VMRVXVT   MXRMXRM   VMRVVM
VMRTMXT   MVRVVVM   VMRMVR    VMRVXVR   VMRVV
MXRTVXT   VXVR      VXTRR     MVXRV     VMRM
VMTRRRR   VMTR      MVXTR     VXVRVMR   VMRVMRM
VMRTVMT   VXVRVV    VMRVMTX   VMRMVXT   VXT
MVRM      MVXTRRR   VMRVVVV   VXTR      MXRVMRV
VMTRRR    MXTRR     MXRVXTX   MVXT      MXRV
MVRVVV    VMRMVRM   VMRMVRV   VMRVVV    VMRVMRV
MVR       VMRMXTX   MXRMVXR   VMRVMTR   VMRVMR
MVXTX     VXVRMXR   MVXRMXR   VMRMXT    VMRV
VXTRRRX   VMRTVMR   MVXRVMT   MXT       MXRTMVR
MXTRRRR   MVRV      MVXR      VMRVVVM   VMR
```

Set relations (arithmetic from the lists): 25 of the 32 strings of the 2a training set also occur in the 2b training set, so the two training sets are not nested; 13 of the 32 2a test strings are among the 125 2b training strings; 6 strings are test strings in both experiments (MVRVXT, MXRMXTX, MXRVXRM, VMRVXRM, VXVRVXT, VXVTRX).

### 2.4 Test strings (Appendix C p.1027, Appendix D p.1028), with recomputed chunk statistics

The paper prints the 32 test strings as four columns of eight (GA, GNA, NGA, NGNA); it does **not** print the per-item chunk strengths. The three numbers after each string are **my recomputation** from the printed training list under the paper's formula (section 2.5): global ACS, anchor ACS, and the number of novel chunks (bigrams and trigrams, any position, with training count 0). They reproduce the paper's cell means (checked in 2.5). Item order within a cell is as printed.

**Exp 2a** (printed ACS: none per item; cell means in 2.5). Lengths: 4 letters x1 (VXTX), 5 x3, 6 x5, 7 x23.

| # | GA | GNA | NGA | NGNA |
|---|---|---|---|---|
| 1 | MVXRVVV 10.09 5.75 0 | MXRMXTX 7.09 4.25 1 | MVRMVXT 8.73 4.25 0 | MVRVRRX 6.36 4.25 2 |
| 2 | MXRMVXT 9.91 5.25 0 | MXRTVMR 6.82 4.50 4 | MVRMXR 8.44 4.00 0 | MXTRVX 8.67 2.75 1 |
| 3 | MXRMXRM 9.45 5.00 0 | VMRTVXT 6.55 5.50 3 | MVRVMRM 9.27 4.00 0 | VMRTMTR 5.00 5.00 1 |
| 4 | MXRVMT 9.67 4.50 0 | VMTRRRX 5.82 3.75 0 | MVRVXRM 9.64 4.00 0 | VMRTRXT 5.64 5.00 2 |
| 5 | MXRVXVT 8.91 4.50 0 | VXTX 6.60 1.50 1 | MVRVXT 9.44 4.25 0 | VXVMRRR 7.82 3.00 2 |
| 6 | VMRVMRM 10.55 5.25 0 | VXVTR 6.29 2.25 0 | MXRVXRM 10.73 5.00 0 | VXVRX 5.71 2.50 1 |
| 7 | VMRVXTR 10.36 5.00 0 | VXVTRX 5.33 2.75 0 | VMRVVMR 10.55 4.75 0 | VXVTMXT 5.36 2.75 1 |
| 8 | VXVRVXT 8.55 3.00 0 | VXVTX 4.86 2.00 1 | VMRVXRM 10.91 5.25 0 | VXVTRMR 5.82 2.00 2 |

Cell format: STRING, global ACS, anchor ACS, novel chunks.

**Exp 2b** (lengths: 5 letters x2, 6 x9, 7 x21; no test chunk is novel, p.1018).

| # | GA | GNA | NGA | NGNA |
|---|---|---|---|---|
| 1 | MVXRVMR 36.00 12.25 0 | MVXRMVR 26.18 12.50 0 | MVRVMRV 34.45 11.50 0 | MVRMVRM 20.82 11.25 0 |
| 2 | MVXRVXT 32.45 14.50 0 | MVXTRX 25.89 13.00 0 | MVXRVXR 31.82 12.00 0 | MVRMVRV 24.00 11.75 0 |
| 3 | MXRMXRV 36.09 19.50 0 | VMRTMVR 22.27 21.50 0 | MXRVXR 36.00 18.50 0 | MVRMXTX 22.09 12.25 0 |
| 4 | MXRMXTX 28.18 20.25 0 | VXVRM 22.14 11.75 0 | MXRVXRM 33.45 19.50 0 | MVRVXT 29.78 13.00 0 |
| 5 | MXRVMR 41.11 18.75 0 | VXVRMVR 21.82 11.50 0 | MXRVXRV 36.27 19.50 0 | MVXVRV 26.89 13.25 0 |
| 6 | VMRVM 43.14 21.75 0 | VXVRMXT 24.73 13.00 0 | VMRVXR 37.33 21.00 0 | MVXVTR 20.78 12.50 0 |
| 7 | VMRVXT 38.11 23.50 0 | VXVRVMT 28.00 11.75 0 | VMRVXRM 34.55 22.00 0 | MVXVTRX 18.18 13.00 0 |
| 8 | VXVRVXT 28.45 13.50 0 | VXVTRX 16.44 12.00 0 | VMRVXRV 37.36 22.00 0 | VXVRVXR 27.82 11.00 0 |

**How the nongrammatical items were made** (p.1010, p.1015, p.1018): 2a, "violating the rule system at one or two positions in the letter string", never in the first position (so every NG string starts with a legal first letter; p.1010 note that any letter can be an exit); 2b, a rule violation "in any position within the letter string, except in the first position" (one violation; the text says "a rule violation"); in 1b the 16 NG items were made by changing one letter of 16 of the training strings (p.1012). All NG test strings of 2a and 2b were confirmed to fall outside the grammar by program. Global similarity (2b, p.1018): 15 G and 14 NG test strings differ by one letter from some training string, 1 G and 2 NG by two letters (not different, t(30) = -0.59). Test items in each cell were chosen for ACS and novelty; NGA and NGNA strings were matched to GA and GNA strings on global and anchor ACS (2.5).

### 2.5 Chunk strength: MVdL's exact formulas (p.1010)

**Chunk frequency.** For each training list, "the frequency of every chunk (bigram and [tri]gram) appearing in the learning strings": the number of occurrences of the chunk as a substring of a training string, at any position (overlapping occurrences each count). The count is taken over the distinct training list **once**: for 2a, the 32 strings are shown twice, but the printed values are reproduced exactly with each string counted once (see the check below). The text says "bigram and digram" (a typo for trigram).

**Global associative chunk strength** (global ACS): the item is decomposed into all its bigrams and all its trigrams ("the test item MXTR comprised the bigrams MX, XT, and TR and the trigrams MXT and XTR"), "we then calculated the mean chunk frequencies for each test item". It is the **pooled mean over all bigrams and trigrams**: global ACS = (sum of the training frequencies of all bigrams + sum for all trigrams) / (number of bigrams + number of trigrams). Worked example (1a, p.1010), MXRVVM: bigrams MX, XR, RV, VV, VM with frequencies 8, 7, 5, 3, 8; trigrams MXR, XRV, RVV, VVM with 5, 1, 1, 0; global ACS = [(8+7+5+3+8) + (5+1+1+0)]/9 = 4.22. This is the same as KS96's definition (section 1.6) and is **not** the "mean of the bigram mean and the trigram mean" used in the Johnstone thesis (JS99 convention, 2.13).

**Anchor associative chunk strength**: frequencies are counted **by position**: "we calculated the frequency of every chunk appearing in the anchor positions in the learning strings". For a test item the four numbers are: (1) the number of training strings that begin with the item's first bigram, (2) the number that begin with its first trigram, (3) the number that end with its last bigram, (4) the number that end with its last trigram. Anchor ACS = their mean. Worked example (1a), MXRVVM: initial bigram MX = 6, initial trigram MXR = 5, final bigram VM = 2, final trigram VVM = 0; (6+5+2+0)/4 = 3.25. This is a positional count and **differs from JS99's anchor ACS** (training count of the first/last bigram/trigram anywhere in the strings; JS99 convention, 2.13).

**Chunk novelty** (p.1013, p.1015): the number of bigrams and trigrams of the test item that never occurred in the training strings, at any position; "anchor novelty" is the same for the four anchor chunks (never occurred in that location). Tables 1 and 3 and p.1015 report both the percentage of novel chunks and the count of items with at least one novel chunk (1a/1b); for 2a/2b the means are the mean number of novel chunks per item.

**Programmatic recomputation** (compute node, 2026-11-21, with the formulas above and the printed training lists, each string counted once). All cell means, and almost all margins, equal the printed values:

| | Exp 2a printed | 2a recomputed | Exp 2b printed | 2b recomputed |
|---|---|---|---|---|
| global ACS GA / NGA | 9.69 / 9.71 | 9.69 / 9.71 | 35.44 / 35.16 | 35.44 / 35.16 |
| global ACS GNA / NGNA | 6.17 / 6.30 | 6.17 / 6.30 | 23.43 / 23.79 | 23.43 / 23.79 |
| anchor ACS GA / NGA | 4.78 / 4.44 | 4.78 / 4.44 | 18.00 / 18.34 | 18.00 / **18.25** |
| anchor ACS GNA / NGNA | 3.31 / 3.41 | 3.31 / 3.41 | **12.88** / 12.28 | **13.38** / 12.25 |
| novelty (any position) GA, NGA / GNA, NGNA | 0, 0 / 1.25, 1.50 | 0, 0 / 1.25, 1.50 | 0 in all cells | 0 in all cells |
| anchor novelty GNA / NGNA | 0.50 / 0.63 | 0.50 / 0.62 (5/8) | 0 | 0 |
| global, G / NG; A / NA | 7.93 / 8.01; 9.70 / 6.23 | same | 29.44 / 29.47; 35.30 / 23.61 | same |
| anchor, G / NG; A / NA | 4.05 / 3.92; 4.61 / 3.36 | same | 15.44 / 15.31; 18.17 / 12.58 | 15.69 / 15.25; 18.12 / 12.81 |
| mean length GA, GNA, NGA, NGNA | (JS99) 6.88, 6.00, 6.75, 6.63 | 6.88, 6.00, 6.75, 6.63 | (JS99) 6.50, 6.50, 6.75, 6.63 | same |
| final-trigram frequency G vs NG | not reported | 1.06 vs 1.00 | 4.63 vs 3.31 (p.1020) | 4.62 vs 3.31 |

**Mismatches (all in 2b anchor ACS).** The printed 2b anchor means for GNA (12.88) and NGA (18.34) differ from the recomputed 13.38 and 18.25; the printed margins (15.44, 15.31, 18.17, 12.58) are consistent with the printed cells, not with the recomputed ones. The recomputed 13.38 and 18.25 are the values in JS99 Table 1 (which has 13.38 and 18.25), so the lists as printed give JS99's numbers, and MVdL's printed 12.88 and 18.34 are probably typesetting or arithmetic slips; the effect on the associative/nonassociative contrast is small (12.58 printed versus 12.81 recomputed). A further disagreement: JS99 Table 1 gives 2b NGA global ACS 34.80, but MVdL prints 35.16 and the recomputation gives 35.16, so the JS99 value is the slip. The printed 1a/1b statistics were not recomputed (Appendix A/B lists were not transcribed; see 2.9).

The terminal-trigram check from the referee comment (p.1020-1021) also reproduces: in the 2b training list VXT ends 7 strings and VXR ends 2.

### 2.6 Procedure (Exp 2a; Exp 2b changes in 2.7)

Participants were tested individually on a personal computer; each letter string appeared in a rectangle (14.5 cm x 6.0 cm) at the centre of the screen (p.1010). Letters M, V, R, T, X.

1. **Learning phase** (exposure task, p.1010, p.1015-1016). Each training string appeared for **3 s**, one at a time. As soon as it disappeared the participant was to **reproduce it with a pencil on a sheet of paper**. Instruction (p.1010): "This is an immediate memory test. On the screen, you will see letter strings composed of letters M, V, R, T, and X. These strings are of variable length. Each string appears for 3 seconds. Once it disappears from the screen, try to reproduce the string on your sheet of paper. Just try to reproduce a maximum number of letter strings at the first trial." If the reproduction was correct, the next string came; if not, **the same string was shown again, up to three presentations in all**, then the examiner moved on. **The whole series was then presented a second time in the same order** (p.1010). So 2a has 32 strings x 2 passes = 64 string-trials at least, more with re-presentations. Order: a predetermined random order for half of the participants, the inverse order for the other half. Reproduction accuracy in 2a (p.1016): 82.6% correct on the first trial, 93.4% after two, 97.0% after three. (1a: 83.6%, 95.6%, 98.6%, p.1010.)
2. **Delay.** The classification phase began five minutes after the end of the learning phase (p.1010).
3. **Classification instruction** (p.1010, repeated for 2a): "The letter strings which you memorized a few minutes ago were generated according to a set of complex rules. Now, you are going to classify new letter strings. Some of these strings are constructed according to the same rules as the previous strings, while others contain errors with regard to these rules. Note that these rules are so complex that it is impossible to unravel them. To decide whether a letter string is correct or not, just rely on your intuition, on your feeling." (Learning-phase participants were not told about the rules until after learning.)
4. **Test** (p.1010-1011). Strings presented the same way as in learning, "except that there was no presentation time limit" (a string stayed on screen until the response). **No feedback.** The 32 test strings were in a predetermined random order for half of the participants and the inverse order for the other half; **then the whole procedure was repeated, with the same 32 strings presented a second time in the same order. Each string was therefore judged twice**, 64 judgments per participant, 16 per cell per pass. Response format (key or button) is not stated; the task is a grammatical/nongrammatical classification (the wording is "classify new items as grammatical or not", p.1010).
5. **After the test.** A free report of the rules and a generation task (write 10 strings believed grammatical, on a sheet with the five letters printed, p.1012) were given to both groups in 2a and 2b (p.1015-1016). Generated strings: 2a, 13.0% grammatical (SEM 2.3) in the experimental group versus 4.0% (1.5) in controls; 2b, 15.0% (3.0) versus 1.5% (0.8), p.1016, p.1018. Not needed for the simulation.
6. **No-learning control group** (2a and 2b): classification phase only, with the instruction "The letter strings were not constructed at random, but according to a complex set of rules which you do not know. Among the strings you will see, some contain errors with regard to these rules. Your task is to classify the correct and incorrect strings..." (p.1016). Their endorsement pattern is in 2.8.

### 2.7 Exp 2b: changes in procedure (p.1018)

Same as 2a except: 125 training strings, shown **in writing-response format as in 2a** ("to equalize the procedure with that of Experiment 2A, experimental participants were asked to write the letter strings on a sheet of paper, and each string could be presented a maximum of three times"; in 1b the response was oral and the maximum was two presentations, p.1012). Reproduction accuracy: 85.0% first attempt, 96.0% within two, 98.7% within three (p.1018). **Whether the 125-string series was shown once or twice in 2b is not stated.** In 1b it was shown once, explicitly (p.1012); 2b says only that the response mode and the number of presentations were equalised with 2a. JS99 reports 125 training trials for 2b (old 2.3). The classification phase is "identical" to 2a (twice through the 32 strings, no feedback; stated for 1b/2a, p.1012, p.1015).

### 2.8 Human results: Exp 2a and 2b (experimental and control groups; N = 20 per group)

The paper prints **marginal means with SEM** (SEM across participants of each marginal mean) and shows the four cells only as bars (Figure 3, p.1016). Percent of strings endorsed as grammatical:

| | Grammatical | Nongrammatical | Associated | Nonassociated |
|---|---|---|---|---|
| **2a experimental** (p.1016) | 54.1 (SEM 2.9) | 56.4 (2.3) | 66.4 (2.4) | 44.1 (2.8) |
| **2a no-learning control** (p.1016) | 46.1 (2.5) | 48.6 (2.8) | 53.0 (2.8) | 41.7 (3.5) |
| **2b experimental** (p.1018) | 65.5 (2.7) | 58.4 (3.2) | 62.3 (3.4) | 61.6 (3.3) |
| **2b no-learning control** (p.1018) | 45.2 (3.1) | 55.0 (2.3) | 55.0 (3.2) | 45.2 (2.6) |

**Cell means, experimental groups (figure read from Figure 3, +/- about 1 point)**: 2a GA ~65, GNA ~43, NGA ~67, NGNA ~45; 2b GA ~68, GNA ~62, NGA ~56, NGNA ~60. These agree with the secondary JS99 Table 1 values (2a 65.31, 42.81, 67.50, 45.00; 2b 67.81, 62.81, 56.25, 60.31; old 2.5, kept in 2.11); the margins implied by the JS99 cells (2a 54.06, 56.25, 66.41, 43.91; 2b 65.31, 58.28, 62.03, 61.56) agree with the paper's printed margins to within 0.3 points. The JS99 values are therefore the more precise cell means, and they are **consistent with the primary source**. **Cell means, control groups (figure read, +/- about 2 points)**: 2a GA ~53, GNA ~39, NGA ~53, NGNA ~44; 2b GA ~48, GNA ~38, NGA ~57, NGNA ~50. **SDs and cell SEMs are not printed**; the error bars in Figure 3 were not measured.

ANOVAs (all 2 x 2 x 2 with condition between and grammaticality and associativity within, df (1, 38)):
- **2a** (p.1016-1017): condition F = 6.10, MSE = 408.29 (experimental participants endorse more than controls); associativity F = 43.00, MSE = 262.48; condition x associativity F = 4.69, MSE = 262.48 (the associated-minus-nonassociated difference is larger in the experimental group; Newman-Keuls: significant in both groups). Grammaticality: no effect, and no grammaticality x associativity interaction (p.1017; the statistics are not printed). 
- **2b** (p.1018): condition F = 12.58, MSE = 448.58; condition x grammaticality F = 13.51, MSE = 210.81 (grammatical > nongrammatical in the experimental group by post hoc tests, the statistic not printed; the reverse in controls). No effect of associativity in the experimental group (statistic not printed).
- 2b control analysis (p.1021): dropping strings that end in VXT or VXR leaves 13 G and 11 NG items; experimental participants, grammaticality F(1,19) = 6.02, MSE = 237.27, 66.2% versus 57.7%; no associativity effect, no interaction.
- The paper does **not** report the separate 2 x 2 within-group ANOVA for 2a and 2b; for 1a and 1b it does (2.9).
- Generation-task comparison, 2a versus 2b: no effect of experiment (p.1020): the 125-string group did not generate more grammatical strings.

Control-group pattern (explained by "internal chunk strength": the frequency of a test string's chunks among the other test strings): 2a controls show an associativity effect (53.0 versus 41.7) that correlates with internal chunk strength on the second pass (r = .50, p.1017-1018); 2b controls endorse NG more than G (55.0 versus 45.2), with internal chunk strength correlating r = .40 and .59 with endorsement on the first and second pass (p.1019). Experimental participants show no such correlation (Tables 5, 6).

### 2.9 Experiments 1a and 1b (brief)

- **1a** (pp.1009-1011). N = 20 undergraduates, one group. 16 training strings (the 16 items of KS94 Exps 2a/2b, 3-7 letters, Appendix A p.1025), shown twice as in 2a. Test: 32 strings, 4 x 8. NG strings: one or two violations, not in the first position. Printed ACS: global G 3.60 versus NG 3.62, A 4.65 versus NA 2.57; anchor G 1.95 versus NG 1.84, A 2.83 versus NA 0.97; within cells GA 4.63, NGA 4.67 (global), GNA 2.57, NGNA 2.57 (global); anchor GA 2.91, NGA 2.75, GNA 1.00, NGNA 0.94 (p.1010). Novelty not controlled (Table 1, p.1013: percentage of novel chunks GA 4.80, GNA 35.34, NGA 17.00, NGNA 26.98). Results (p.1011): G 55.6% (SEM 2.1), NG 53.1% (3.9), A 65.3% (2.5), NA 43.4% (3.4). 2 x 2 ANOVA: associativity F(1,19) = 44.98, MSE = 212.79; grammaticality F(1,19) = 0.45, MSE = 276.11; interaction F(1,19) = 16.98, MSE = 117.80. Cells (figure read, Figure 2): GA ~71, GNA ~40, NGA ~59, NGNA ~47; GA endorsed more than every other cell (ts(19) > 2.70), NGA more than NGNA (t(19) = 2.72), GNA = NGNA.
- **1b** (pp.1012-1013). N = 20 (new group). 125 training strings (Appendix B, p.1026), 16 G test strings (8 A, 8 NA) and 16 NG test strings made by changing one letter (not the first) of 16 training strings. Training: each string 3 s, **oral repetition** instead of written reproduction, maximum two presentations, the series **once**; 86.7% first attempt, 96.3% within two. Test as in 1a (twice through). Printed ACS: global G 28.82 versus NG 27.57, A 35.36 versus NA 21.03; anchor G 15.00 versus NG 14.52, A 18.08 versus NA 11.44. Results: G 70.9% (SEM 3.0), NG 62.3% (2.6), A 65.8% (2.1), NA 67.5% (3.3). ANOVA: grammaticality F(1,19) = 7.45, MSE = 198.27; associativity F(1,19) = 0.36, MSE = 165.99; interaction F(1,19) = 0.26, MSE = 227.67. Cells (figure read): GA ~71, GNA ~71, NGA ~60, NGNA ~64. Generation task: 18.0% grammatical.
- The 1a/1b appendix lists (A: p.1025; B: p.1026) were **not transcribed or checked** by program (the text layer scrambles Appendix A cell columns; Appendix B's learning list is clean). They are not needed for the 2a/2b target.

### 2.10 What MVdL conclude (p.1021-1024)

2a: with chunk novelty and ACS controlled, a learning-phase effect of associativity and no grammaticality effect and no interaction, so the 1a interaction was an artefact of novelty. 2b: grammaticality only; the paper argues for abstraction and not for a surface chunk account (the terminal-trigram check, p.1020-1021). The paper's own analysis of the KS96 items: mean novel chunks GA 0.00, GNA 0.38, NGA 0.75, NGNA 2.38, which tracks the KS96 endorsement pattern (p.1014).

### 2.11 Cross-check against the JS99 secondary values (old sections 2.3-2.5)

- **Design.** N = 20 per experiment: confirmed (p.1010 etc.; the paper adds 20 no-learning controls per experiment, which JS99 does not mention). Training amounts 32 and 125: confirmed. **"Each string presented twice at test": confirmed** (p.1011). Training trials "64 (2a) versus 125 (2b)": 2a confirmed (32 x 2 passes); 2b unconfirmed (the paper does not say whether the 125-string series was repeated, 2.7). Test length "5 to 7 letters": **wrong for 2a**; the 2a test list contains one 4-letter string (VXTX) and three 5-letter strings; 2b is 5 to 7 letters.
- **Cell means.** JS99 2a 65.31 / 42.81 / 67.50 / 45.00; 2b 67.81 / 62.81 / 56.25 / 60.31 (GA / GNA / NGA / NGNA): consistent with the paper's marginal means and with the Figure 3 bar heights (2.8). Not independently checkable to two decimals from the paper.
- **Stimulus statistics (JS99 Table 1).** Match the paper's printed 2a values exactly; for 2b they match the recomputation from the lists (anchor GNA 13.38, NGA 18.25) rather than the paper's printed 12.88 and 18.34; the JS99 2b NGA global ACS 34.80 is a slip (paper and recomputation: 35.16). Novelty and length cell means: confirmed.
- **Grammar:** identical (2.2).
- **ACS formula.** JS99's thesis convention (global = mean of the bigram mean and the trigram mean; anchor = counts of first/last bigram/trigram anywhere) is **not** MVdL's; see 2.5 and 2.13.

### 2.11a JS99 Table 1, stimulus statistics and human results (kept for reference; the values agree with 2.5 and 2.8)

Stimulus statistics (mean per cell; JS99 Table 1, p.23, as reported from MVdL's stimuli):


| | Grammatical associated (GA) | Grammatical nonassociated (GNA) | Nongrammatical associated (NGA) | Nongrammatical nonassociated (NGNA) |
|---|---|---|---|---|
| **Exp 2a** anchor ACS | 4.78 | 3.31 | 4.44 | 3.41 |
| global ACS | 9.69 | 6.17 | 9.71 | 6.30 |
| novelty (mean number of unseen chunks) | 0.00 | 1.25 | 0.00 | 1.50 |
| NCP (novel chunk position; Johnstone's measure, not MVdL's) | 0.63 | 1.75 | 1.38 | 2.13 |
| length (letters) | 6.88 | 6.00 | 6.75 | 6.63 |
| **Exp 2b** anchor ACS | 18.00 | 13.38 | 18.25 | 12.25 |
| global ACS | 35.44 | 23.43 | 34.80 | 23.80 |
| novelty | 0.00 | 0.00 | 0.00 | 0.00 |
| NCP | 0.25 | 0.25 | 0.63 | 1.88 |
| length | 6.50 | 6.50 | 6.75 | 6.63 |

The ACS values are on the scale of raw counts of the chunk in the training strings, so they are about 3.7 times larger in 2b than 2a (125 versus 32 distinct training strings; counts are per distinct string, not per presentation, 2.5).

Human results as tabulated by JS99 (JS99 Table 1, p.23): percentage of test strings classified as grammatical

"Classification" row, mean percentage of the 320 judgments per cell (N = 20), from the table:

| | GA | GNA | NGA | NGNA |
|---|---|---|---|---|
| **Exp 2a (32 training strings)** | 65.31 | 42.81 | 67.50 | 45.00 |
| **Exp 2b (125 training strings)** | 67.81 | 62.81 | 56.25 | 60.31 |

Margins (arithmetic from the four cells, equal weights, not from the paper):

| | Grammatical | Nongrammatical | Associated | Nonassociated |
|---|---|---|---|---|
| Exp 2a | 54.06 | 56.25 | 66.41 | 43.91 |
| Exp 2b | 65.31 | 58.28 | 62.03 | 61.56 |

Effect sizes (arithmetic, percentage points): Exp 2a, chunk strength (associated minus nonassociated) +22.5, grammaticality (G minus NG) -2.2; Exp 2b, chunk strength +0.5, grammaticality +7.0.

MVdL's conclusion, as stated by JS99 (p.24): "after the short training phase (Experiment 2a) participants classified test strings on the basis of ACS, with no reliable effect of grammaticality: the classification rates closely parallel the ACS measures. However, when participants received extended training (Experiment 2b), they classified test strings on the basis of grammaticality, with no reliable effect of ACS." The ANOVA statistics of the original are in 2.8 (primary source); cell-level SDs are not printed in either source. **Note that the 2b cell pattern is not clean: the nongrammatical associated cell (56.25) is below the nongrammatical nonassociated cell (60.31), and the grammatical advantage is 11.6 points in the associated column (67.81 versus 56.25) but 2.5 points in the nonassociated column (62.81 versus 60.31).**

### 2.12 Johnstone's re-analysis (JS99 pp.26-37), relevant because the controlling variable differs from ACS

- **Novel chunk position (NCP):** a chunk (bigram or trigram) that occurred in training but never at the position where it occurs in the test string counts one; a terminal chunk counts as novel only if it never occurred as the last chunk in any training string (JS99 p.26). Example: MVXVTRX has NCP 7 minus 1 = 6 (p.26).
- Mean NCP, Exp 2a: grammatical 1.19 versus nongrammatical 1.76, associated 1.01 versus nonassociated 1.94. Exp 2b: grammatical 0.25 versus nongrammatical 1.26, associated 0.44 versus nonassociated 1.07 (p.26). Mann-Whitney: 2a, associated versus nonassociated U = 76.5 (significant), grammatical versus nongrammatical U = 87 (ns); 2b, grammatical versus nongrammatical U = 75.5 (significant), associated versus nonassociated U = 122.5 (ns) (p.27).
- **Transition rule strength (TRS):** the frequency of each of the grammar's 23 rules in the training strings, summed over the rules that compose a test string; global TRS divides by the number of letters, anchor TRS averages the first and last rule counts (p.35). Example: VXVRM in Exp 2b: rule counts 64, 33, 17, 45, 19, 39 over the 125 training strings; global TRS = (64+33+17+45+19+39)/5 = 43.4; anchor TRS = (64+39)/2 = 51.5. Mean TRS, Exp 2a: anchor G 9.81 versus NG 8.00, associated 9.56 versus nonassociated 8.25; global G 13.02 versus NG 7.89, associated 11.19 versus nonassociated 9.73 (p.36). Exp 2b: anchor G 37.88 versus NG 30.88, associated 34.31 versus nonassociated 34.44; global G 47.26 versus NG 27.38, associated 41.54 versus nonassociated 33.11 (pp.36-37). Correlation of TRS with grammaticality in 2b: .81 (anchor), .83 (global).
- Per-participant multiple regression on mean number of grammatical responses per string (0, 1, 2): in 2a anchor ACS (mean coefficient .05045, t(19) = 2.80), novelty (-.11770, t = 8.15 as printed, reading garbled) and NCP (.03996, t = 4.00) are reliable and grammaticality is not (-.01002, t = 0.36); in 2b no predictor is reliable by two-tailed test (grammaticality .03477, t = 1.22; NCP -.02397, t = 1.49; length -.06370, t = 1.95); variance explained about 25% in both (Tables 3-4, pp.29-30). The first-participant table rows contain apparent copy errors (participant 9 and 10 rows are identical up to a sign in Table 3), so do not use them.
- String length against classification in 2b: 5-letter strings 65.0%, 6-letter 69.7%, 7-letter 61.8% (p.30).
- JS99's conclusion: the 2b "grammaticality" effect is confounded with NCP and TRS; positional chunk knowledge by a single learning system accounts for both experiments.

### 2.13 ACS definitions used in this literature (the exact formulas)

- **MVdL97 (primary; section 2.5):** global ACS = pooled mean over all bigrams and trigrams of the item of the training frequency of the chunk, counting occurrences anywhere in the training strings; anchor ACS = mean of four positional counts (number of training strings beginning with the item's first bigram, with its first trigram, ending with its last bigram, ending with its last trigram). Novelty = number of the item's chunks with training count 0. Reproduces every printed 2a cell mean and, apart from three 2b anchor values (2.5), every 2b one.
- **KS96 (primary):** pooled mean over all bigrams and trigrams of the test item (section 1.6), the same as MVdL's global ACS. No anchor measure (MVdL fault KS96 for omitting it, p.1009).
- **JS99 thesis convention (secondary, applies to JS99's own experiments, not to MVdL):** global ACS = (bigram mean + trigram mean)/2; anchor ACS = mean of the training counts, anywhere in the strings, of the first bigram, last bigram, first trigram and last trigram (JS99 pp.45-46). It differs from MVdL's on both measures.
- KD05 gloss for ACS in general: "the frequency with which a chunk occurred in the training set, with the frequency scores averaged across all different chunks" (p.2); "Anchor ACS refers to a measure that calculates the frequency with which bigrams and trigrams occur in the beginning and terminal positions" of the strings. Read with MVdL's p.1010 definition, "the beginning and terminal positions" means chunk counts by position, as in 2.5.
- **Novelty** (MVdL97; JS99 Table 1): the number of test-item chunks that never occurred in training (0 in all of 2b).

---

## 3. Comparison models: what is actually available as numbers

**No source reports a model's endorsement rates for the four KS96 cells or the four MVdL97 cells per training amount. I found no such numbers.** What is available:

### 3.1 Competitive chunking (Servan-Schreiber & Anderson 1990), via BD03's CCN reimplementation

SSA90 itself was not read. BD03 (pp.815-818) describes its implementation of the competitive chunking network (CCN), with four differences from SSA90 (BD03 p.818, list not extracted here). Scoring rule used by BD03 (and, per BD03, the same logistic as used for its SRN):
- Familiarity = e^(1 - n_active), where n_active is the number of active units (chunks) when the activation flow stops (BD03 formula 4, p.817).
- **p("grammatical") = 1 / (1 + e^(T - n_active))**, T a classification threshold (BD03 formula 5, p.817). In SSA90 the threshold was the running average of the last 20 stimuli rather than the whole test set (BD03 p.817). Fewer active units (the string is parsed into fewer, larger chunks) means more familiar.
- Chunk strength increments when a chunk is activated and decays with parameter d (0 < d < 1); unit activation is stochastic with competition parameter c (formula 3).
- BD03 on KS96: "It has yet to be shown whether the CCN could simulate the Knowlton and Squire data" (the amnesic/normal discussion, about p.834). So no CCN fit to KS96 was found.

### 3.2 SRN (Boucher & Dienes 2003; Kinder & Shanks 2001)

- BD03 SRN scoring (p.811-812): the net predicts the next letter; the Pearson correlation c between the stacked predicted and actual output vectors for the whole string is mapped by **p("grammatical") = 1 / (1 + e^(T - c))** (formula 1), with T set so that about half of all test strings are called grammatical (average output correlation over the whole test set); back-propagation stays on during test.
- BD03 says Kinder & Shanks (2001) simulated the amnesic versus normal pattern of KS94/KS96 by giving the SRN a lower learning rate (BD03, discussion). The numbers are in Kinder & Shanks (2001, J. Cogn. Neurosci. 13), which was not obtained.

### 3.3 PARSER (Perruchet & Vinter)

No AGL endorsement simulation of KS96 or MVdL97 located. PPVG02 reports PARSER reproducing a unit-segmentation pattern on a finite-state grammar (human N = 78 segmenting strings); no endorsement numbers were read.

### 3.4 What this means for P4

Because no published model numbers exist for these cells, P4 compares TRELLIS to the human cells and to **baselines we implement ourselves** under the published scoring rules above (pooled ACS, anchor ACS, competitive-chunk familiarity, SRN correlation), with the same stimuli. The published rules are in sections 1.6, 2.7, 3.1 and 3.2.

---

## Simulation notes (what a 1:1 simulation needs)

**Part A: KS96 Experiment 1 (everything needed is in this document).**
- Alphabet: {J, T, V, X}. Grammar: section 1.3. String lengths: training 2 to 6, test 2 to 6.
- Training: 23 distinct strings (section 1.4; decide the 26-row versus 23-string question, see Ambiguities), **each presented twice, 46 trials**, one string at a time; the human saw each for 3 s and had to reproduce it, with up to 3 attempts. For a model that learns from a stream, present the strings as 46 separate inputs, with each string isolated (a string is a sentence with boundaries), in a fixed or shuffled order; the human order is not given, so run several shuffles.
- Test: 32 strings (section 1.5), four cells of 8, mixed order, each judged once, no learning at test (the human got no feedback). Participants were controls, N = 18; a simulation should use at least 18 independently seeded learners (training order and any stochasticity vary), with the cell mean over learners.
- **Scoring of an endorsement from a model score.** The paper reports only percent endorsed per cell. Two options: (a) fixed criterion: endorse if the score exceeds a threshold set a priori on training strings (for example the mean or a low quantile of the model's score on the 23 training strings, or the median of the 32 test scores, the BD03 rule for SRN threshold); (b) Luce rule or logistic as in BD03: p(endorse) = 1/(1 + exp(T - score)), averaged over items, with T as in BD03 (running mean or the mean of all test scores) and a slope fixed in advance. Option (b) gives rates in the human range without fitting each cell; the humans' overall rate is 52.5% (mean of the four cells, 54.9+28.1+64.6+62.5 over 4 = 52.5; arithmetic), so a median-split threshold (50%) is close to the humans' overall endorsement rate.
- **Targets**: controls GH 64.6, GL 62.5, NGH 54.9, NGL 28.1 (section 1.8). Qualitative targets: G > NG (63.5 versus 41.5), H > L (59.7 versus 45.3), and the interaction (chunk strength matters for NG: 54.9 versus 28.1, almost not for G: 64.6 versus 62.5). A learner driven only by ACS will predict GH = NGH and GL = NGL (ACS is balanced across G and NG in the design, KS96 mean CS 8.6, 5.8, 8.6, 5.4), i.e. no grammaticality effect and, instead, a chunk strength effect of equal size in G and NG; the human data need an additional sensitivity to something other than pooled ACS, such as **chunk error versus location error**: NG items with impermissible chunks are rejected more than NG items with permissible chunks in impermissible positions (60.0 versus 42.6 at high strength, 41.7 versus 25.0 at low, section 1.8). This is the diagnostic that separates a learner of chunk position from a pure bigram/trigram counter.
- Report: the four cell means with SEM over learners, the two main effects, the interaction, and the correlation of the model score with the paper's CS value across the 32 items.

**Part B: MVdL97 Experiments 2a and 2b (everything needed is in section 2).**
- Alphabet {M, R, T, V, X}; grammar: section 2.2 (20 letter transitions, 3 exits, state E has two R transitions; generates exactly 150 strings of length 3-7). Verified by program against all 157 training and 64 test strings.
- Training: **2a, the 32 strings of 2.3, shown twice** in the same order (3 s each, written reproduction, up to 3 presentations of a string per pass, with the human order randomised or reversed); **2b, the 125 strings of 2.3** (series shown once or twice: unconfirmed, 2.7). Treat each string as a separate sentence with boundaries; shuffle order over learners (the printed orders are not the presented orders).
- Test: **32 strings, section 2.4 (8 per cell: GA, GNA, NGA, NGNA)**, judged **twice** each (two passes in the same order, no feedback, no learning at test), N = 20 trained participants per experiment (use at least 20 independently seeded learners; the 20 no-learning controls give a no-training baseline with a model that has seen no strings).
- ACS: the pooled global mean and the positional anchor mean of 2.5, recomputable and verified; novelty = number of chunks never seen.
- Targets (cells from JS99, equal to the paper's marginals and bars, 2.8): Exp 2a GA 65.31, GNA 42.81, NGA 67.50, NGNA 45.00; Exp 2b GA 67.81, GNA 62.81, NGA 56.25, NGNA 60.31. Marginals printed by MVdL: 2a G 54.1, NG 56.4, A 66.4, NA 44.1; 2b G 65.5, NG 58.4, A 62.3, NA 61.6. The signature effect to reproduce is the crossover: chunk strength +22.5 and grammaticality -2.2 after 32 strings, versus chunk strength +0.5 and grammaticality +7.0 after 125 strings (arithmetic from JS99 cells; printed marginals give +22.3 and -2.3, and +0.7 and +7.1).
- Controls: no-learning participants endorse A over NA in 2a (53.0 versus 41.7) and NG over G in 2b (55.0 versus 45.2), explained by chunk strength internal to the test list (2.8). A model that scores strings by their overlap with the other test strings should reproduce this, and a trained model's lift over it is the learning effect.
- Scoring: the same rules as Part A. In MVdL97, grammatical and nongrammatical items were balanced on global and anchor ACS and novelty, but not on NCP, TRS or the terminal trigram (4.63 versus 3.31 in 2b, p.1020; 2.12), so a learner of position-specific chunks or of transition rules (TRELLIS' concepts) will predict a grammaticality advantage after long training even without rule abstraction. State this in the write-up as a finding of JS99 and of the paper's own referee comment, not as evidence of abstraction.

### Ambiguities that affect a simulation

1. **Training set size in KS96.** The Appendix lists 26 rows but 23 distinct strings (XXVJ, VJTVX, XXXVTV printed twice). The text says 23 items. Use the 23 distinct strings by default; run the 26-row version as a sensitivity check.
2. **Recomputed CS differs from printed CS** (up to about 1.9 per item; the closest definition is the pooled mean over chunk occurrences with 23 distinct strings, section 1.6). Use the printed CS to define "high" and "low". Three possible duplicate rows in the GL column (section 1.5) are printed as given.
3. **Item order in training and test** is not given in KS96 (the test "in a mixed order"). Shuffle and average.
4. **Does a model learn during the test?** Humans got no feedback. TRELLIS-style incremental learners would otherwise keep learning from test strings, which inflates familiarity of repeated chunks; freeze learning at test (the BD03 SRN keeps back-propagation on, which is a model choice).
5. **Endorsement mapping**: no human-derived threshold exists; the humans' overall endorsement rate is 52.5% (arithmetic), only 2.5 points above the 50% a median-split rule would give.
6. **MVdL97 order and repetition.** The printed appendix order is not the presentation order (random order for half the participants, inverse for the other half, p.1010). In 2b it is not stated whether the 125-string series was shown once or twice (2.7); JS99's "125 training trials" suggests once. Response key and cell-level SDs are not printed.
7. **MVdL97 cell means.** The paper prints marginal means with SEM and only bars for the cells; the cell means used as targets are from JS99 Table 1 and agree with the paper's marginals to 0.3 points and with the bar heights to about 1 point. A simulation target of cell means should carry that tolerance.
8. **2b anchor ACS.** The printed anchor ACS for 2b GNA (12.88) and NGA (18.34) do not reproduce from the printed lists (13.38 and 18.25, which are JS99's values); the paper's own statistic for the A/NA anchor contrast (18.17 versus 12.58) differs from the recomputed 18.12 versus 12.81. Use the recomputed values.
9. **Nongrammatical items:** the item-level error type (chunk error versus location error) is not listed in KS96's Appendix, so the Table 4 analysis cannot be replicated item by item. A program can recompute it from the grammar and the training set (an NG item has a "chunk error" if it contains an impermissible bigram or trigram, meaning one absent from all grammatical strings or absent from the training set, which is itself ambiguous), but the paper's classification may differ.

## Figure-only and image-only values (read, not digitized)

- KS96 Figure 1 (grammars A and B) and the Appendix table were read from page renders, not digitized from a plot; transcription risk is in the letters and arrows, not in measurement. Figure 3 (Experiment 3 percent-correct bars) gives only the bar means that are also stated in the text (section 1.9); nothing was digitized from it.
- MVdL97 Figure 1 (grammar; 400-dpi render), Appendices C and D (stimulus lists; page renders, compared with the text layer) and Figures 2-3 (bar charts) were read from page renders. The bar heights quoted in 2.8 and 2.9 are figure reads (+/- about 1 point for experimental groups, about 2 for controls), not digitized. Every other MVdL number (marginals, SEMs, F values, ACS means) is printed text.
- JS99 Figure 2 (Brooks & Vokey grammar as redrawn by Johnstone): read from a render; now confirmed identical to MVdL97 Figure 1 (section 2.2).
- No other numbers in this document are figure-derived.
