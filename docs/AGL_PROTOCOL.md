# Artificial grammar learning: chunk strength against grammaticality (Knowlton & Squire 1996; Meulemans & Van der Linden 1997): protocol and numbers, plus model comparisons

Compiled 2026-11-21 for row P4 of docs/PLAN_PSYCH.md. Every number below comes from a source that was actually opened and read as text, or is marked as arithmetic or as a figure read. **Knowlton & Squire (1996) was obtained in full and is the primary source for section 1, including the grammar, all 23 training strings and all 32 test strings. Meulemans & Van der Linden (1997) could NOT be obtained.** Its design and its cell means (all four cells, both training amounts) come from a secondary source that tabulates them, Johnstone's PhD thesis (1999, UCL), read in full for the relevant chapter. Its stimulus lists (training and test strings) were NOT found anywhere; section 2 says exactly what is and is not known. Treat every MVdL value as "as reported by Johnstone", and check it against the primary paper before quoting.

## Sources read (and how)

PDFs were text-extracted with pypdf on a compute node (PDF streamed over stdin). Figures and the appendix table were rendered with pymupdf and read as images. Page cites use the page numbering named in the last column.

| Tag | Source | Copy read | Page numbering used below |
|---|---|---|---|
| KS96 | Knowlton & Squire (1996), "Artificial grammar learning depends on implicit acquisition of both abstract and exemplar-specific information", JEP:LMC 22(1):169-181 | Full journal PDF, public domain, lead.u-bourgogne.fr/wp-content/uploads/2023/08/knowlton_squire_1996.pdf (13 pp). Text layer good. The Appendix (stimulus list) and Figure 1 (grammars) are images and were read from page renders | Journal pages 169-181 as printed (PDF page n = printed page 168+n) |
| JS99 | Johnstone (1999), PhD thesis "Structural versus processing accounts of implicit learning", University College London | discovery.ucl.ac.uk/id/eprint/10107529/1/Structural_versus_processing_a.pdf (full thesis; only Ch. 1-2 and the ACS definition in Ch. 3 were used). Chapter 2 re-analyses MVdL97 Experiments 2a and 2b and reproduces its summary table. Figure 2 (grammar) is an image and was read from a render | Thesis printed page numbers (PDF page n = printed page n-1) |
| BD03 | Boucher & Dienes (2003), "Two ways of learning associations", Cognitive Science 27:807-842 | users.sussex.ac.uk/~dienes/Boucher & Dienes 2003.pdf (31 pp, clean text). Used for the exact scoring rules of competitive chunking and the SRN | Journal pages as printed (footers 807-842) |
| KD05 | Kuhn & Dienes (2005), "Implicit learning of nonlocal musical rules", JEP:LMC 31:1417-1432 | users.sussex.ac.uk/~dienes/Kuhn & Dienes JEP 2005.pdf (16 pp). Used only for a verbal definition of ACS and anchor ACS as applied to MVdL97 | PDF page 2 (introduction) |
| PPVG02 | Perruchet, Vinter, Pacteau & Gallego (2002), "The formation of structurally relevant units in artificial grammar learning", QJEP 55A | lead.ube.fr copy (19 pp). Abstract and text skimmed only. PARSER applied to AGL segmentation, not to endorsement | |

## Not found / not read

- **MVdL97** (Meulemans & Van der Linden 1997, JEP:LMC 23(4):1007-1028): no free full text located (ORBi/ULiege record 2268/1929 has "Files (0)"; APA, ResearchGate and Semantic Scholar not reachable as text). Not read. Consequences:
  - The grammar is known only through JS99's redrawing of Brooks & Vokey's (1991) grammar (section 2.2). MVdL's own figure and the exact letter-to-transition mapping were not seen.
  - **The training strings and the 32 test strings of Experiments 2a/2b are NOT available.** Only fragments appear in JS99: ten training strings containing VXR (MXRMVXR, MVXR, MVXRVV..., MVXRMXR, MVXRVMT, MVXRVV, MVXRV, MVXRM, MVXRVVM, plus one the text extraction garbled) and four nongrammatical test strings (MXRVXRM, MXRVXRV, VMRVXRM, VMRVXRV) and one grammatical example, VXVRM (JS99 pp.25, 35). These are not a usable item set.
  - Experiments 1a/1b (the first pair) are not reported by any source I read, other than the abstract-level summary in section 2.1.
  - Instructions at training and test, presentation duration, stimulus display, letter case, N for Experiments 1a/1b, SDs and cell standard errors, ANOVA statistics: not found. (JS99 reports only that MVdL concluded from ANOVAs that Exp 2a showed an ACS effect and no reliable grammaticality effect, and Exp 2b the reverse; no F values.)
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

## 2. Meulemans & Van der Linden (1997), Experiments 2a and 2b, as reported by Johnstone (1999)

### 2.1 What the original paper does (abstract-level, from the ORBi record and later citations)

Four experiments (1a, 1b, 2a, 2b); grammaticality and associative chunk strength were manipulated independently at test. "When participants learned few items of the grammar (Experiments 1a and 2a), the associated items were more often classified as grammatical than the nonassociated ones. When the learning phase included most grammatical items (Experiments 1b and 2b), the only effect observed was an effect of grammaticality" (ORBi abstract, paraphrase of the record, not a page of the paper). Experiments 1a/1b: not described in any source I read.

### 2.2 Grammar

Brooks & Vokey's (1991) grammar (JS99 p.33, Figure 2). **Alphabet {M, R, T, V, X}** (5 letters). Ten node numbers in JS99's redrawing (1 start; 8, 9, 10 have OUT arrows), read from a 300-dpi render. Transitions (from, letter, to): 1-M-2, 1-V-4, 2-V-5, 2-X-6, 3-M-2, 3-T-3, 3-V-4, 4-M-6, 4-X-7, 5-X-6, 5-R-8, 6-R-3, 6-R-8, 6-T-10, 7-V-6, 7-T-10, 8-V-8, 8-M-9, 10-X-9, 10-R-10. OUT at 8, 9, 10. JS99 states: "all legal strings ... start with MV, MX, VM, or VX" (p.33), and, as a check on a path, VXVRM = 1-V-4-X-7-V-6-R-8-M-9 plus the terminal transition, with 6 transition rules (V 1 to 4, X 4 to 7, V 7 to 6, R 6 to 8, M 8 to 9, terminal at 9; JS99 p.35, "node 9"), which this transcription reproduces. JS99 counts 17 transition rules and 6 terminal rules (p.35); my table has 20 letter transitions and 3 OUT nodes, so **the redrawing as transcribed does not match the rule count (17 + 6 = 23)**. The grammar as transcribed here is therefore unverified and must be checked against Brooks & Vokey (1991) before use. The strings in the one ten-string training fragment (JS99 p.25) are consistent with the transcription only if repeated cycles through 6-3-2 are allowed.

### 2.3 Design and procedure, as far as known (JS99 pp.22-24, 28)

- Participants: **N = 20 in each of Experiment 2a and 2b** (JS99 Tables 3-4 list 20 participants per experiment, pp.29-30; the original paper's N is not stated in any source). Different participants per experiment (not stated, assumed from the between-experiment design).
- **Training amount: 32 training strings (Exp 2a) versus 125 training strings (Exp 2b)** (JS99 p.23). Training trials: **64 (2a) versus 125 (2b)** (JS99 p.23), which is two presentations of each of the 32 strings in 2a and one presentation of each of the 125 strings in 2b. What the training task was (copying, memorising, reproducing) is not stated in JS99; assume memorisation (training was an incidental/implicit-learning task, as in the paper's title), and check.
- Test: **32 strings, 8 in each cell**, grammatical or nongrammatical crossed with associated (high ACS) or nonassociated (low ACS), 5 to 7 letters long. **Each string was presented twice in the test stage** (JS99 p.28: "each string was presented twice in the test stage"; the dependent variable per string was the number of trials, 0, 1 or 2, on which it was called grammatical). The cell percentages below are consistent with 8 strings x 20 participants x 2 presentations = 320 judgments per cell (for example 65.31% = 209/320, 42.81% = 137/320).
- Test instruction and response format: not found; presumably grammatical/nongrammatical forced choice.
- Experiments 2a and 2b differ in the number of training strings and in novel fragments: **in 2a some test strings contain bigrams/trigrams never seen in training (novelty), equal across grammatical and nongrammatical strings; in 2b every test chunk occurred at least once in training (novelty = 0 in all four cells)** (JS99 p.23).

### 2.4 Stimulus statistics (mean per cell; JS99 Table 1, p.23, as reported from MVdL's stimuli)

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

The ACS values are on the scale of raw counts of the chunk in the training strings, so they are about 3.7 times larger in 2b than 2a (125 versus 32 strings, with 2 versus 1 presentations).

### 2.5 Human results: percentage of test strings classified as grammatical (JS99 Table 1, p.23)

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

MVdL's conclusion, as stated by JS99 (p.24): "after the short training phase (Experiment 2a) participants classified test strings on the basis of ACS, with no reliable effect of grammaticality: the classification rates closely parallel the ACS measures. However, when participants received extended training (Experiment 2b), they classified test strings on the basis of grammaticality, with no reliable effect of ACS." The ANOVA statistics of the original are not available. Cell-level SDs are not available either. **Note that the 2b cell pattern is not clean: the nongrammatical associated cell (56.25) is below the nongrammatical nonassociated cell (60.31), and the grammatical advantage is 11.6 points in the associated column (67.81 versus 56.25) but 2.5 points in the nonassociated column (62.81 versus 60.31).**

### 2.6 Johnstone's re-analysis (JS99 pp.26-37), relevant because the controlling variable differs from ACS

- **Novel chunk position (NCP):** a chunk (bigram or trigram) that occurred in training but never at the position where it occurs in the test string counts one; a terminal chunk counts as novel only if it never occurred as the last chunk in any training string (JS99 p.26). Example: MVXVTRX has NCP 7 minus 1 = 6 (p.26).
- Mean NCP, Exp 2a: grammatical 1.19 versus nongrammatical 1.76, associated 1.01 versus nonassociated 1.94. Exp 2b: grammatical 0.25 versus nongrammatical 1.26, associated 0.44 versus nonassociated 1.07 (p.26). Mann-Whitney: 2a, associated versus nonassociated U = 76.5 (significant), grammatical versus nongrammatical U = 87 (ns); 2b, grammatical versus nongrammatical U = 75.5 (significant), associated versus nonassociated U = 122.5 (ns) (p.27).
- **Transition rule strength (TRS):** the frequency of each of the grammar's 23 rules in the training strings, summed over the rules that compose a test string; global TRS divides by the number of letters, anchor TRS averages the first and last rule counts (p.35). Example: VXVRM in Exp 2b: rule counts 64, 33, 17, 45, 19, 39 over the 125 training strings; global TRS = (64+33+17+45+19+39)/5 = 43.4; anchor TRS = (64+39)/2 = 51.5. Mean TRS, Exp 2a: anchor G 9.81 versus NG 8.00, associated 9.56 versus nonassociated 8.25; global G 13.02 versus NG 7.89, associated 11.19 versus nonassociated 9.73 (p.36). Exp 2b: anchor G 37.88 versus NG 30.88, associated 34.31 versus nonassociated 34.44; global G 47.26 versus NG 27.38, associated 41.54 versus nonassociated 33.11 (pp.36-37). Correlation of TRS with grammaticality in 2b: .81 (anchor), .83 (global).
- Per-participant multiple regression on mean number of grammatical responses per string (0, 1, 2): in 2a anchor ACS (mean coefficient .05045, t(19) = 2.80), novelty (-.11770, t = 8.15 as printed, reading garbled) and NCP (.03996, t = 4.00) are reliable and grammaticality is not (-.01002, t = 0.36); in 2b no predictor is reliable by two-tailed test (grammaticality .03477, t = 1.22; NCP -.02397, t = 1.49; length -.06370, t = 1.95); variance explained about 25% in both (Tables 3-4, pp.29-30). The first-participant table rows contain apparent copy errors (participant 9 and 10 rows are identical up to a sign in Table 3), so do not use them.
- String length against classification in 2b: 5-letter strings 65.0%, 6-letter 69.7%, 7-letter 61.8% (p.30).
- JS99's conclusion: the 2b "grammaticality" effect is confounded with NCP and TRS; positional chunk knowledge by a single learning system accounts for both experiments.

### 2.7 ACS definitions used in this literature (the exact formulas)

- **KS96 (primary):** pooled mean over all bigrams and trigrams of the test item of the training count of the chunk (section 1.6). No anchor measure.
- **MVdL97 (secondary description):** "global ACS" and "anchor ACS" (KD05 p.2): "Anchor ACS refers to a measure that calculates the frequency with which bigrams and trigrams occur in the beginning and terminal positions" of the strings (added because participants are sensitive to initial and terminal chunks, KD05). Johnstone and Shanks used ACS "as applied by Knowlton and Squire (1994, p. 85)" and apply, in the one experiment where JS99 shows a worked example, this form: **anchor ACS = mean of the training count of the first bigram, the last bigram, the first trigram and the last trigram** (e.g. (LF 1 + LX 1 + LFG 0 + DLX 1)/4 = 0.75, JS99 p.45-46); **global ACS = (mean over all bigrams of the item + mean over all trigrams of the item) / 2**, with counts of occurrences anywhere in the training strings (worked example JS99 p.46: ((4+3+5+3+4+2+5)/7 + (0+1+0+0+0+1)/6)/2 = 2.02, where the printed arithmetic contains typesetting damage). **This is the thesis's own convention in its own experiments and is not stated by JS99 to be MVdL's exact formula**; note that it differs from the KS96 pooled mean.
- KD05 gloss for ACS in general: "the frequency with which a chunk occurred in the training set, with the frequency scores averaged across all different chunks" (p.2). "All different chunks" could be read as distinct chunks, not occurrences; KS96's own definition averages over the chunks "in the item" (section 1.6).
- **Novelty** (MVdL97, in JS99 Table 1): the number of test-item chunks that never occurred in training (0 in all of 2b).

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

**Part B: MVdL97 Experiments 2a and 2b (not reproducible 1:1 until the stimulus lists are found).**
- Alphabet: {M, R, T, V, X}. Grammar: Brooks & Vokey (1991), section 2.2, to be verified.
- Training amount is the manipulated variable: **32 strings with 64 training trials versus 125 strings with 125 trials**; test: 32 strings, 8 per cell (G/NG x associated/nonassociated), lengths 5 to 7, each presented twice (2 judgments per string); N = 20 per experiment.
- Targets (section 2.5): Exp 2a GA 65.31, GNA 42.81, NGA 67.50, NGNA 45.00; Exp 2b GA 67.81, GNA 62.81, NGA 56.25, NGNA 60.31. The signature effect to reproduce is a crossover: chunk strength +22.5 and grammaticality -2.2 after 32 strings, versus chunk strength +0.5 and grammaticality +7.0 after 125 strings.
- Until the strings are obtained, a simulation can only (a) generate its own materials from the Brooks & Vokey grammar with the stated design (32 test strings, 8 per cell, ACS balanced across grammatical and nongrammatical strings, associated versus nonassociated differing in global and anchor ACS, 2a with some novel chunks in the nonassociated cells and 2b with none), and (b) compare qualitatively with the cell means above. It cannot reproduce the specific items. A claim of 1:1 reproduction of MVdL97 should wait for the Appendix of the original.
- Scoring: the same rules as Part A. In MVdL97, grammatical and nongrammatical items were balanced on both anchor and global ACS but not on NCP or TRS (section 2.6), so a learner of position-specific chunks or of transition rules (TRELLIS' concepts) will predict a grammaticality advantage after long training even without rule abstraction. This must be stated in the write-up as a finding of JS99, not claimed as evidence of abstraction.

### Ambiguities that affect a simulation

1. **Training set size in KS96.** The Appendix lists 26 rows but 23 distinct strings (XXVJ, VJTVX, XXXVTV printed twice). The text says 23 items. Use the 23 distinct strings by default; run the 26-row version as a sensitivity check.
2. **Recomputed CS differs from printed CS** (up to about 1.9 per item; the closest definition is the pooled mean over chunk occurrences with 23 distinct strings, section 1.6). Use the printed CS to define "high" and "low". Three possible duplicate rows in the GL column (section 1.5) are printed as given.
3. **Item order in training and test** is not given in KS96 (the test "in a mixed order"). Shuffle and average.
4. **Does a model learn during the test?** Humans got no feedback. TRELLIS-style incremental learners would otherwise keep learning from test strings, which inflates familiarity of repeated chunks; freeze learning at test (the BD03 SRN keeps back-propagation on, which is a model choice).
5. **Endorsement mapping**: no human-derived threshold exists; the humans' overall endorsement rate is 52.5% (arithmetic), only 2.5 points above the 50% a median-split rule would give.
6. **MVdL97 stimuli, instructions and training task** not found (section "Not found"); the 5-letter alphabet and grammar are unverified (the rule count 17 + 6 does not match the transcribed diagram).
7. **MVdL97 N and cell SDs** are from a secondary source (N = 20 inferred from 20 per-participant rows of JS99's tables). The original N may differ.
8. **ACS formula.** KS96 pools bigrams and trigrams; the thesis's worked example averages the bigram mean and the trigram mean (section 2.7). The exact MVdL97 formula is unconfirmed.
9. **Nongrammatical items:** the item-level error type (chunk error versus location error) is not listed in KS96's Appendix, so the Table 4 analysis cannot be replicated item by item. A program can recompute it from the grammar and the training set (an NG item has a "chunk error" if it contains an impermissible bigram or trigram, meaning one absent from all grammatical strings or absent from the training set, which is itself ambiguous), but the paper's classification may differ.

## Figure-only and image-only values (read, not digitized)

- KS96 Figure 1 (grammars A and B) and the Appendix table were read from page renders, not digitized from a plot; transcription risk is in the letters and arrows, not in measurement. Figure 3 (Experiment 3 percent-correct bars) gives only the bar means that are also stated in the text (section 1.9); nothing was digitized from it.
- JS99 Figure 2 (Brooks & Vokey grammar as redrawn by Johnstone): read from a render; unverified against MVdL97 (section 2.2).
- No other numbers in this document are figure-derived.
