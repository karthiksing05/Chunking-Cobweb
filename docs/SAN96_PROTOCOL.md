# Statistical word segmentation (Saffran, Aslin & Newport 1996 and follow-ups): protocol and numbers, plus model comparisons

Compiled 2026-11-21. Every number below comes from a source that was actually opened and read as text. Where a value could not be read it says "not found". **The three primary human papers (SAN96 Science, SNA96 JML, ASN98 Psych Sci) could NOT be obtained as text.** Their protocols below are assembled from secondary sources that were read (the 1996 CogSci precursor by the same authors, PARSER, TRACX, Frank et al. 2010, review articles, course slides), each tagged. Treat every secondary value as "as reported by that source", and check against the primary paper before quoting.

## Sources read (and how)

All PDFs were text-extracted with pypdf on a compute node (PDF streamed over stdin). Page cites use the page numbering named in the last column.

| Tag | Source | Copy read | Page numbering used below |
|---|---|---|---|
| FGGT10 | Frank, Goldwater, Griffiths & Tenenbaum (2010) "Modeling human performance in statistical word segmentation", Cognition 117:107-125 | Author preprint, 19 pp (sites.socsci.uci.edu/~lpearl/courses/readings/FrankEtAl2010_ModelingHumanPerf.pdf). Text layer good; **Figure 1 (human per-condition means) is a plot with no numbers**; Figures 4 and 5 axis labels are garbled | Preprint pages 1-19 (not the journal's 107-125) |
| PV98 | Perruchet & Vinter (1998) "PARSER: A model for word segmentation", J. Mem. Lang. 39:246-263 | lead.ube.fr PDF (full, 18 pp, clean text) | Journal pages 246-263 as printed |
| FAM11 | French, Addyman & Mareschal (2011) "TRACX", Psychological Review 118:614-636 | lead.ube.fr PDF (full, 23 pp, clean text) | Journal pages 614-636 as printed (footers) |
| SAN96b | Saffran, Aslin & Newport (1996) "Statistical cues in language acquisition: Word segmentation by infants", Proc. 18th Ann. Mtg Cognitive Science Society, pp. 376-380 | escholarship.org/uc/item/4qv000ks PDF (6 pp, OCR-like text with some spacing noise). **This is the conference precursor, NOT the Science paper.** It reports one experiment (words vs nonwords) | CogSci pages 376-380 as printed |
| PP12 | Perruchet & Poulin-Charronnat (2012) "Beyond transitional probability computations", J. Mem. Lang. 66:807-818 | lead.ube.fr PDF (12 pp) | Journal pages as printed |
| PL14 | Pearl, UCI Psych 156A (2014) Lecture 6 "Word segmentation 1", course slides | sites.socsci.uci.edu/~lpearl/.../Lecture6-WordSeg1.pdf. Slides summarising SAN96 | PDF page of the slide file (several slide builds per page). Secondary, slides |
| AN12 | Aslin & Newport (2012) "Statistical learning: From acquiring specific items to forming general rules", Curr. Dir. Psych. Sci. | UCI course copy (author manuscript, 7 pp) | Manuscript pages 1-7 |
| AS17 | Aslin (2017) "Statistical learning: a powerful mechanism that operates by mere exposure", WIREs Cogn Sci 8:e1373 | UCI course copy (7 pp) | PDF pages |

The other agent-session directories in the same scratchpad (ES87, ESC, RSS95) are unrelated and were not used.

## Not found / not read

- **SAN96** (Science 274:1926-1928): no readable full text. The huji.ac.il course copy returned 403; science.org returned an HTML wall; no other copy located. Not read. Its numbers below come only from secondary sources.
- **SNA96** (JML 35:606-621, adults): no free full text located. Not read. Nothing below comes from it directly.
- **ASN98** (Psych Sci 9:321-324): typeset/scispace copy returned an empty page; no other copy. Not read.
- **SSA97** (Saffran, Newport, Aslin, Tunick & Barrueco 1997, incidental learning): not searched; only PARSER's summary was read.
- **Robinet, Lemaire & Gordon (2011) MDLChunker** (Cognitive Science 35:1352-1389): no readable full text found (Wiley/HAL not located; the 2009 CogSci precursor on eScholarship returned a 403). Not read. MDLChunker facts below come only from PP12.
- Frank et al. (2010) **journal-page numbers** (the preprint was read) and the **per-condition human means** (Figure 1 is unlabeled dots and lines).
- TRACX's supplemental materials (doi 10.1037/a0025255.supp): not fetched.
- All figure-only values are listed at the end.
- Exact figure of the **number of infants** in the Science paper's two experiments, exact syllable inventory, speaker/voice, and per-trial counts for Experiment 2 of SAN96: not found (only the CogSci precursor's design is read, see 1.2).

---

## 1. Infant studies

### 1.1 SAN96 (Science 1996) as reported by secondary sources

Stimuli (PL14, PDF p.7; AN12 p.1-2; PV98 p.247, p.257; FAM11 p.621; AS17 p.2-3):
- Artificial language: four three-syllable nonsense words (PL14 p.7; AN12 p.1; AS17 p.2-3).
- Condition A words: tupiro, golabu, bidaku, padoti. Condition B words: dapiku, tilado, burobi, pagotu (PL14 p.7; the same lists appear in SAN96b p.378).
- Another word set, stated by PV98 p.257 as "used in one condition of their Experiment 2": pabiku, tibudo, golatu, daropi; test items in that experiment: pabiku, tibudo, tudaro, pigola (PV98 p.257). AN12 Fig. 1 (p.2) shows the stream "pa bi ku go la tu da ro pi ti bu do" for the four words, and AS17 Fig. 1 (p.2) shows the same.
- Twelve unique syllables (4 words x 3 syllables, no shared syllables in the word lists above); the number "12" is stated only in FAM11 p.621 ("an alphabet of 12 syllables"), whose description of this experiment is internally inconsistent (see 3.2).
- Stream: random order of words, no pauses or acoustic cues, speech synthesizer (PL14 p.7; SAN96b p.378). PL14 says the voice was female with monotone intonation (PL14 p.7).
- Transitional probabilities: within words 1.0, across word boundaries 0.33 (PL14 p.7-8; AN12 p.2; SAN96b p.378). AS17 Fig. 1 (p.2) labels the Science words TPs "1.0 and 1.0" and part-words "0.5 and 1.0".
- Syllable frequencies equated (AN12 p.2: "Because the frequency of syllables was equated, learners could not use this statistic to segment words").
- Rate: AS17 p.3: syllables presented "at a rate of 4 per second"; SAN96b p.378 (the CogSci precursor): 270 syllables per minute (= 4.5 per second, arithmetic).
- Exposure: 2 minutes (AN12 p.2; AS17 p.1; PL14 p.7; PV98 p.248; SAN96b p.378).
- Counts: PV98 p.257: infants heard "only 45 occurrences of each of the four trisyllabic words", i.e. 540 syllables. FAM11 p.621 trains the model on a random sequence of 180 words (= 4 x 45, arithmetic).
- Type/token ratio of the Science stimuli: .022 (PV98 p.259).

Procedure (PL14 p.6, p.8-10; PV98 p.248; SAN96b p.377-378, the same lab's headturn design):
- Familiarization-preference procedure (Jusczyk & Aslin 1995). Infants control the duration of each test trial by sustained fixation on a blinking light (PL14 p.6).
- Test: each infant hears repetitions of 4 strings, 2 words and 2 foils (PL14 p.8).
  - Experiment 1 (as summarised in PL14 p.8): foils are "fake" words with the syllables jumbled (examples ropitu, bulago), i.e. nonwords.
  - Experiment 2 (PL14 p.10): foils are part-words made of syllables from two different words in order (examples pirogo, bubida).
- Number of test trials, look-away criterion and N for the Science paper: not found. (SAN96b: 12 test trials, 2 s look-away criterion, max 15 repetitions; see 1.2.)

Results (PL14 p.9-10; FAM11 p.622 footnote 5):
| Experiment | Foil type | Looking time, words | Looking time, foils | Source |
|---|---|---|---|---|
| 1 | nonwords | 7.97 s | 8.85 s | PL14 p.9 |
| 2 | part-words | 6.77 s | 7.60 s | PL14 p.10; FAM11 p.622 footnote 5 (labels it "Saffran et al. (1996) Experiment 2") |

- Direction: longer looking to the novel (non-word / part-word) items, read as a novelty preference (PL14 p.9-10; FAM11 p.621; PV98 p.248).
- PV98 p.248: "The mean difference between words and nonwords, although relatively small (approximately 850 ms over a total exposure time of about 8 s) was statistically significant". The text does not say which experiment; 7.60 - 6.77 = 0.83 s and 8.85 - 7.97 = 0.88 s (arithmetic), both fit.
- FAM11 p.622 footnote 5 normalises: 0.83 / (6.77 + 7.60) = 0.06, the "proportion better" of words over part-words for humans.
- Statistics (t, p, N) for the Science paper: not found.

### 1.2 SAN96b (CogSci 1996 precursor, read directly)

This is a fully read primary source, but a different paper from the Science article. It is the same lab, the same design family, and the same word list as Condition A/B above. Whether its data are the same data as the Science paper's Experiment 1 is **not stated** in either source; compare by checking the Science paper.

- Subjects (p.378): 16 infants (9 male, 7 female), about 8 months old. Three more tested but excluded: experimenter error (2), crying (1).
- Stimuli (p.378): two counterbalanced conditions. 45 tokens of each of four trisyllabic words, 2-minute streams. Condition A: tupiro, golabu, bidaku, padoti. Condition B: dapiku, tilado, burobi, pagotu. Random order, the same word never twice in a row. MacinTalk synthesizer, **270 syllables/minute**, equal coarticulation between all syllables, no pauses.
- Language statistics (p.378): TP within words all 1.0, across word boundaries all .33.
- Test strings (p.378): tupiro, golabu, dapiku, tilado. Condition A infants: first two = words (tupiro, golabu), last two = nonwords (dapiku, tilado), whose syllables were heard in familiarization in a novel order, TP zero. Condition B infants: first two = nonwords, last two = words. Each of the 4 strings on 3 trials = **12 test trials per infant**, randomised per subject. Test strings were synthesized in citation form, so they differ acoustically from the stream.
- Procedure (p.378): infant on parent's lap in a sound-attenuated booth; blinking red centre light, blinking yellow side lights, hidden speakers; parent and experimenter wore headphones with masking music; experimenter blind to stimulus. Familiarization: 2-minute stream played over both speakers, lights (but not speech) contingent on looking. Test: side light blinks, the test string repeats until the infant looks away for **2 s** or until 15 repetitions. Listening times tabulated on-line.
- Results (p.378-379):
  - Condition A vs B difference scores: t(14) = 1.3, n.s., so the conditions were pooled.
  - Words vs nonwords, matched-pairs t-test: nonwords listened to longer, **t(15) = 2.8, p < .02**. **12 of 16** infants listened longer to the novel strings.
  - Mean listening times are in Figure 1 (p.379), not read as numbers (see the figures list).

### 1.3 ASN98 (Psych Sci 1998): frequency-matched part-words

ASN98 itself not read. As described by FAM11 (p.622):
- Four trisyllabic words, **two low-frequency (LF) and two high-frequency (HF)**; HF words occur twice as often as LF words in the familiarization stream.
- The part-words spanning the boundary between the two HF words ("HH-partwords") therefore occur as often as the LF words.
- TP inside an LF word = 1; TP inside an HH-part-word = .5.
- Test: LF words against HH-part-words. Infants looked longer at the HH-part-words.
- Human "proportion better" (normalised looking-time difference, defined in FAM11 footnote 5, p.622) = **.04** (FAM11 Table 1, p.622).
- Original stimuli, exposure length, N, trial counts, looking times, statistics: **not found**. (FAM11's own simulation familiarised the model on a 270-word sequence "as in Aslin et al.", p.622. A search-engine snippet said 3 minutes of exposure; that snippet is not a read source and is not used.)
- PV98 does not cover ASN98 (it predates it). PP12 p.816 says PARSER "turned out to be able to reproduce the effect of TPs (Aslin et al., 1998...)".

FAM11's own replication of ASN98 (footnote 9, p.628), British-English voice, "exact replication":
- N = 13 infants (6 girls, 7 boys; mean age 249 days, range 232-277).
- Partwords M = 12.6 s (SD 3.67), words M = 10.7 s (SD 3.72); t(12) = 2.57, p < .025, two-tailed.

### 1.4 FAM11 new infant experiment: equal forward TPs (read directly, pp.628-629)

Useful as a third infant protocol with full statistics.
- Design: "box language": three word blocks A, B, C; every forward TP (within words and between words) = .5; words occur twice as often as part-words; backward TP within words = 1, between words = .25 (pp.626-628).
- Participants: 20 infants (12 female, 8 male; mean age 251 days, range 226-279), 11 excluded (9 fussiness, 1 equipment, 1 parental interference) (p.628).
- Stimuli: MBROLA British-English voice, 16 kHz; two 3-minute familiarization sequences (810 syllables at 4.5 syllables/s) from the words of Table 3 (Grammar 1 and 2, six words each); test = the 6 words and 6 of the part-words, each presented alone with a 500 ms gap, repeated until look-away (p.628-629).
- Procedure: matches ASN98; 12 randomly ordered test trials (each test item 3 times); look-away criterion 2 s; head turn of 30 degrees; experimenter blind (p.629).
- Result: partwords M = 12.2 s (SD 2.92) vs words M = 9.4 s (SD 2.89); t(19) = 4.62, p < .001, two-tailed (p.629). Proportion better = .13 (FAM11 Table 1, p.622; also (12.2 - 9.4)/(12.2 + 9.4) = .13 by arithmetic).

---

## 2. Adult studies

### 2.1 SNA96 (JML 1996): as reported by PV98 and FGGT10 (primary text not read)

From PV98 (p.247, p.250, p.254-255):
- Language: **six trisyllabic words**, e.g. babupu, bupada; full list used in PV98's simulation of this study: babupu, bupada, dutaba, patubi, pidabu, tutibu (PV98 p.250, p.254). Whether this is SNA96's exact list is stated by PV98 as the study's list ("Six trisyllabic words, such as babupu and bupada", p.247).
- Presentation: random order, repeated for **21 minutes**, synthesised speech, no pauses or prosodic cues (p.247). PV98 p.254: "a continuous series of **4536 syllables**" (= 1512 words, arithmetic); no immediate repetition of the same word (p.254).
- Instructions: participants were told beforehand that the language contained words and asked to work out where they begin and end (p.247).
- Test: two-alternative forced choice, "which of two items sounded more like a word from the artificial language"; one item a word, the other a new three-syllable combination (p.247). PV98 p.255: with the same 21 min, "intentional conditions lead to 76%".
- Results (PV98 p.247): "participants selected the correct words on **65 or 76%** of the trials, depending on whether the alternative item included a permissible syllable pair", both significantly above chance. FGGT10 p.12: in Experiment 1 of SNA96 two groups trained on the same corpus were tested against nonwords (syllables from the language never heard together) or part-words (include a two-syllable substring from a word); "participants rejecting non-words more than part-words", a significant difference. Putting the two statements together, 76% is the nonword test and 65% the part-word test; **this mapping is my inference, not a quoted statement.**
- By-word split (PV98 p.254-255): words with higher mean TP (dutaba, pidabu, tutibu; mean TP "from 1 to .75") 79% correct vs lower-TP words (babupu, bupada, patubi; "from .50 to .37") 72%, a significant difference.
- Frequency of the syllable bu: PV98 p.255 notes bu is "the most frequent syllable in the language" and that pidabu and tutibu end in bu.

Not found: the syllable inventory (beyond the above words), the speech rate, the number of participants, the number of test items, Experiment 2's design, all statistics.

### 2.2 SSA97 (incidental-learning variant), as reported by PV98

- Participants coloured while the 21-minute tape played; not told it was a language or that a test would follow (p.247).
- Experiment 1: 1800 words, 36 test pairs (each of the 6 words against each of the 6 nonwords); mean 21.1 / 36 for adults, 21.3 / 36 for 6-7 year olds, chance 18 (p.256). 58.6% correct overall for adults (p.247).
- Experiment 2: two sessions of 1800 words; adults 26.3 / 36, children 24.6 / 36 (p.256); 73.1% (p.247).
- PV98 p.256 arithmetic: knowing one of six words gives 21/36 = 58.33%.

### 2.3 FGGT10: three adult experiments with a SNA96-style 2AFC (read directly)

Paradigm and materials (preprint pp.3, 5, 6):
- Lexicon per participant generated randomly from 18 syllables (ba, bi, da, du, ti, tu, ka, ki, la, lu, gi, gu, pa, pi, va, vu, zi, zu) into **six words: two of 2 syllables, two of 3, two of 4** (p.3). Each participant heard a unique random language.
- Sentences: random concatenation of words, no adjacent repetition (p.3). **Each participant in Exp 1 heard 600 words, equal tokens of each vocabulary item (100 each)** (p.3); Exps 2 and 3 change the token or type counts as given below.
- Synthesis: MBROLA (us3 diphone database, American male voice); consonants 25 ms, vowels 225 ms (so 250 ms per CV syllable); F0 100 Hz; no breaks inside a sentence, equal coarticulation; **500 ms break between sentences** (p.3).
- Instruction: told they would "listen to a nonsense language for 15 minutes" (p.3). (Arithmetic: 600 words x average 3 syllables x 0.25 s = 7.5 minutes before sentence breaks; the paper does not reconcile this with "15 minutes".)
- Test: **30 target-distractor pairs**; word vs part-word of the same syllable length; part-words made as in SNA96 from the end of one word and beginning of another (example: words badutu and kagi, part-word dutuka); except in the 1-word-sentence condition part-words occurred in the corpus at lower frequency than words; no adjacent test pairs shared words or part-words (p.3). 2AFC "sounded more like a word in the language they just heard"; no feedback (p.3).
- Analysis: mixed-effects logistic regression with a fixed word-length effect, separate intercepts per condition, a random effect per participant (p.4).

**Experiment 1: sentence length (pp.3-5).** 8 between-subject conditions of 1, 2, 3, 4, 6, 8, 12, 24 words per sentence. 101 tested, 3 excluded for performance more than 2 SD below their condition mean (98 analysed); participants per condition: not found.
- Word-length effect: beta = -.00022, z = .028, p = .99 (p.4). Word length x sentence length interaction did not improve fit: chi2(7) = 10.67, p = .15 (p.4).
- Condition coefficients (as printed; the paper uses them to judge reliability against chance, p.4):

| Words per sentence | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 24 |
|---|---|---|---|---|---|---|---|---|
| beta | 2.31 | 1.56 | 1.64 | 1.39 | 1.32 | 1.31 | .86 | .45 |
| z | 7.00 | 5.16 | 5.10 | 4.47 | 4.24 | 4.20 | 2.82 | 1.55 |
| p | <.0001 | <.0001 | <.0001 | <.0001 | <.0001 | <.0001 | .004 | .12 |

- Reading (p.4-5): length 24 "did not differ significantly from chance"; "longer sentences were considerably more difficult to segment".

**Experiment 2: amount of exposure (p.5).** 4 words per sentence, 6 conditions of 48, 100, 300, 600, 900, 1200 total tokens; 72 participants, **12 per condition**; no outliers excluded. Interaction pruned: chi2(5) = 1.69, p = .89.
- 48 tokens: beta = .082, z = .31, p = .76. 100 tokens: beta = .39, z = 1.47, p = .14. Both not different from chance (p.5).
- Four larger conditions (the paper prints "beta = .75, .76, 1.03, and 1.33; z = 2.78, 2.81, 3.77, 4.72; all p < .01"); the order is presumably 300, 600, 900, 1200, which the text does not label. The z for 900 prints as "3 .77" in the extraction and is read as 3.77.
- Reading (p.5): "Performance rose steeply between 48 tokens and 300 tokens, then was largely comparable between 300 and 1200 tokens."

**Experiment 3: number of word types (p.5-6).** 4 words per sentence, 600 tokens, types = 3, 4, 5, 6, 9. Token frequency per type: 200 each (3 types) to about 66 each (9 types) (p.5). 63 tested, 2 excluded (61 analysed); participants per condition: not found. Interaction pruned: chi2(4) = 7.06, p = .13; word-length effect beta = .031, z = .91, p = .36.

| Types | 3 | 4 | 5 | 6 | 9 |
|---|---|---|---|---|---|
| beta | 2.12 | 1.77 | 1.17 | .84 | .39 |
| z | 5.46 | 4.92 | 3.17 | 2.25 | 1.32 |
| p | <.0001 | <.0001 | .0015 | .025 | .19 |

- Reading: performance decreases as number of types rises; 9 types not different from chance (p.5).
- Confound control (p.6): joint analysis of Exps 2 and 3 with types and tokens-per-type as linear predictors: types beta = -.11, z = -2.81, p = .004 (negative), tokens per type beta = .0049, z = 5.31, p < .0001. So the type effect holds with tokens per type controlled.
- Between-word TPs in the languages (p.9, p.14): 3 types gives between-word TP 1/2; 9 types gives 1/8. In Exps 1-2 (six words, no repeats) any word follows any other except itself, TP about 1/5 (p.9).

**Human percent-correct per condition: not found as text** (Figure 1, p.4, only plots individual points, means and SEs).

### 2.4 FGGT10 model fits (read directly)

Evaluation (p.6): each model must give a score for each test item; Luce choice rule P(a) = S(a)/(S(a) + S(b)); averaged over test trials. **12 randomly generated training sets per condition**, each with 30 test pairs (p.6). Syllables are the unit; CV structure ignored. Pearson r against human condition means; Spearman results "comparable" (footnote 2, p.6). All models segmented the corpora correctly in absolute terms (p.6).

Models (pp.7-8): transitional probability (TP, within-word product score; start/end symbol # added; within-word minimum identical, mean slightly worse); pointwise mutual information (MI); Bayesian lexical model (Goldwater et al. 2009 unigram, Gibbs sampler, temperature 2, alpha as in that paper); MI clustering (Swingley 2005); PARSER (parameters as in the original; a reference implementation from Perruchet gave identical results, footnote 5; 100 runs per training set gave similar results, p.9).

Correlations with human means, full-data models (Fig. 2, p.10; Exp 3 values are confirmed by FAM11 p.625; the Exp 2 values' order is by position in the extracted figure text and is consistent with the prose on p.9 and with Table 1 alpha = 0 on p.19 for TP):

| Model | Exp 1 sentence length | Exp 2 exposure | Exp 3 types |
|---|---|---|---|
| TP | .84 | .43 | -.99 |
| MI | .83 | -.32 | -.99 |
| Bayesian lexical | .94 | .89 | -.98 |
| MI clustering | .11 | -.81 | .29 |
| PARSER | .00 | .86 | .00 |

- Prose (p.9): no model fitted all three; no model matched Exp 3; TP/MI fail Exp 2 (point estimates converge quickly); lexical model and PARSER fit Exp 2; MI clustering jagged.
- Target-only analysis (p.12): PARSER scores correlate with humans at r = .92, .84, .89 across Exps 1-3 if distractor scores are ignored (PARSER gives distractors zero); lexical model target probabilities r = .82, .89, .96 (pp.11-12).
- Evidence-limited models (4% of the data; Fig. 4, p.14): DM-TP (alpha = 32) and the lexical model (temperature 3) correlate with each other at r = .90 and "both showed a relatively good fit" to all three experiments (p.14). The individual r values printed in Figure 4 are garbled: **not found**. Exp 2 shape was too gradual (p.14-15).
- DM-TP with the full data (Table 1, p.19): correlation with human data by Dirichlet alpha:

| alpha | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp 1 | .84 | .84 | .84 | .84 | .84 | .84 | .84 | .85 | .85 | .85 | .85 |
| Exp 2 | .43 | .92 | .93 | .93 | .93 | .92 | .91 | .90 | .89 | .88 | .87 |
| Exp 3 | -.99 | -.99 | -.99 | -.98 | -.96 | -.90 | -.64 | .16 | .75 | .92 | .96 |

- Capacity-limited particle-filter lexical models (single particle, 10 simulated participants per input set = 120 runs per condition; forgetting by uniform type deletion, frequency-proportional type deletion, or uniform token deletion above a capacity n) "all produced a good fit across all three experiments for some parameter regime" (p.15-16). The r values in Figure 5 are garbled: **not found**. Best-fit n and temperature values: not found.

---

## 3. Model papers

### 3.1 PARSER (PV98, read directly)

Model (pp.250-252): a "percept shaper" lexicon with a weight per unit; percept sizes drawn at random from 1 to 3 units; new percepts added with weight 1; matching units get +0.5; forgetting -0.05 per time step on all units; interference -0.005 on units contained in the current percept; a unit shapes perception only if weight >= 1; longest matching unit chosen. Infant simulation raises forgetting to 0.08 (p.257).

Simulations and measures (pp.254-258):
| Study | Human data simulated | Measure | Result |
|---|---|---|---|
| 1 | SNA96 Exp 1 (6 words, 4536 syllables) | Number of syllables to reach a "loose" criterion (all 6 words in lexicon with the highest weights) and a "strict" criterion (only legal clauses left besides the words), 100 runs | Loose: mean 2064 syllables (range 735-4479); strict: mean 3837 (range 1380-8796); at 4536 syllables all runs had reached loose and 72% strict (p.254). Max units shaping perception at once: 20 (9 excluding one-syllable units) (p.254). Failed to reproduce the high-TP vs low-TP word difference: only dutaba (TP 1) beat the others, F(1,99) = 22.45, p < .0001; the contrast between the two sets of three words gave F < 1 (pp.254-255) |
| 2 | SSA97 incidental learning | Forced-choice score: word chosen if in lexicon with weight > 1, else random; attention shifts simulated by processing blocks of 5 percepts then skipping a random number of syllables | Exp 1: 21.34 / 36 (range 13-31) with 3% of the material processed (human 21.1 adults, 21.3 children). Exp 2: 25.9 and 27.02 with 4% and 5% (human 26.3 adults, 24.6 children) (p.256) |
| 3 | SAN96 infant study | Whether the test words are in the lexicon after 540 syllables (45 tokens x 4 words x 3 syllables) | 52 of 100 runs had both test words with weight > 1; 93 had at least one; 23 and 65 if only the top four units are counted; 23 runs had the single top unit as one of the two test words; part-word foils spanning word boundaries (Exp 2 test items) were never in the lexicon (p.257). Needed forgetting 0.08 instead of 0.05. For infants: loose criterion mean 1077 syllables (range 90-4908), strict mean 1188 (range 156-4908) (p.257) |
| 4 | New: words of 1 to 5 syllables (bu, bapa, dutabu, patubi, pidabupu, tutibudaba) | Same criteria | Loose mean 1910 (613-4782), strict mean 3338 (1184-8538) (p.258) |

100 simulations per study, treated as 100 participants (p.254). PV98 does not simulate ASN98 or the Frank et al. experiments.

### 3.2 TRACX (FAM11, read directly)

TRACX is a recognition-based autoassociator with truncated recursion. The abstract (p.614) lists 9 simulations of empirical data plus 2 of generalization and clustering. Simulations (pp.621-627):

| Simulation | Experiment | Participants | Measure | Human | TRACX | SRN | Page |
|---|---|---|---|---|---|---|---|
| 1 | SAN96 Exp 1 (4 words, 180-word sequence) | infant | normalised looking-time difference ("proportion better") vs output-error difference | .06 | .08 | .68 | p.622 |
| 2 | ASN98 Exp 1 (LF words vs HH-partwords, 270-word sequence) | infant | same | .04 | .13 | .58 | p.622 |
| 3 | Perruchet & Desaulty (2008) Exp 2 forward TPs (9 two-syllable words, 12 syllables, 1035-word sequence, 115 tokens each) | adult | percent-correct 2AFC | .34 | .38 | .80 | p.622 |
| 4 | Perruchet & Desaulty (2008) Exp 2 backward TPs | adult | percent-correct | .22 | .32 | -.10 (SRN prefers part-words) | pp.622-623 |
| 5 | Giroux & Rey (2009) subchunk recognition | adult | recognition error over learning | qualitative | replicates early-learning-then-forgetting rebound for internal subchunks lm, mn | SRN did not | p.623 |
| 6 | **Frank et al. (2010) Exp 1** (sentence length) | adult | correlation with human curve of word-minus-partword error difference | - | **r = .92** | **r = .60** | p.625 |
| 7 | **Frank et al. (2010) Exp 3** (vocabulary size) | adult | same | - | **r = .97** | **r = .33** | p.625 |
| 8 | Brent & Cartwright corpus scaling (9,800 sentences, 33,400 words, 1,321 word types, 95,800 phonemes) | corpus | words vs partwords error | - | words learned better than partwords | - | pp.626-627 |
| 9-10 | Generalisation and clustering of internal representations (two microlanguages Alpha/Beta) | - | cluster analysis | - | two clusters; six unseen words classified correctly | - | p.627 |
| 11 | "Box languages", equal-TP design | - | prediction | - | words learned better than partwords | SRN weak | p.628 |
| new | Infant equal-TP experiment (see 1.4) | infant | looking time | .13 | .50 | .05 | p.622, p.629 |

Details that matter for replication:
- TRACX did not simulate Frank et al.'s Experiment 2 (exposure); only Exps 1 and 3 are simulated (pp.624-625).
- Frank-Exp-1 setup in TRACX (p.625): words ab, cd, efg, hij, klmn, opqr; 8 sequences of 144 words with 144, 72, 48, 36, 24, 18, 12, 6 sentences; each sentence presented six times during learning. This is a smaller corpus than the 600 words humans heard.
- Reported on the same page: PARSER correlation with Frank's Exp 1 and Exp 3 human data r = 0; the five non-TRACX models' Exp-3 r values -.99, -.99, -.98, .29, .00 (identical to the FGGT10 Figure 2 values above).
- Values are averages over 25 runs (Table 1 note, p.622).
- Footnote 5, p.622: "proportion better" = (A - B)/(A + B) of looking times or output errors. TRACX output errors for words 1.41 and partwords 1.65, absolute difference .24, proportion .08 (SAN96 familiarization data).
- SRN: 2N-N-N architecture, learning rate .01, momentum .9 (p.621). TRACX learning rate .04 for small corpora and .005 for Brent & Cartwright (pp.621, 626).
- Discrepancy in the TRACX text: Simulation 1 (p.621) first describes SAN96 as "six different words ... drawn from an alphabet of 12 syllables" with "90 of these words (270 syllables) ... presented twice", then says "Like Saffran et al. (1996), we created a language of four trisyllabic words and trained TRACX on a random sequence of 180 words". The 12-syllable alphabet matches four words; the "six words" is not consistent with the other sources (PL14, AN12, PV98 p.257 all say four). Use four.
- Discrepancy: TRACX p.625 describes Frank et al.'s Exp 3 as varying "the number of word tokens"; FGGT10 p.5 varies word types with tokens held constant at 600.

### 3.3 MDLChunker (Robinet, Lemaire & Gordon 2011): primary text not read

What was read, from PP12 (pp.809-810, 816): MDLChunker creates all possible new chunks, shortest first, and keeps those whose coding reduces total description length; it has **no free parameters**; it was run with PARSER on Endress & Mehler's (2009) phantom-word task with 5, 10, 15, 25, 35, 55, 75, 100 repetitions of each word, 100 runs per point; score = 1 if the word's lexicon weight exceeds the foil's, 0.5 if equal, 0 if lower (p.810). Results (p.810): PARSER above chance with 5 repetitions, t(99) = 15.59, p < .001; MDLChunker at chance with 5, above chance with 10, t(99) = 3.79, p < .001; near-identical with 25 repetitions; MDLChunker asymptotes slightly faster. PP12 p.816 says PARSER was designed for the SNA96 results.

Not found: any MDLChunker simulation of SAN96, SNA96, ASN98 or the Frank et al. experiments (its own paper was not read; PP12 does not report them). Do not assume any.

---

## Signature effects a model must reproduce

Values are human data as read; "(secondary)" marks values taken from a source other than the primary paper.

1. **Infants discriminate words from foils after 2 minutes of unsegmented speech with TP as the only boundary cue** (within-word TP 1.0 vs between .33; four trisyllabic words; 45 tokens each, 540 syllables). Direction: longer looking at the foils (novelty preference). Experiment 1 (nonword foils): 7.97 s words vs 8.85 s nonwords (PL14 p.9, secondary). Experiment 2 (part-word foils, which share syllable pairs with words and occur in the stream): 6.77 s vs 7.60 s (PL14 p.10; FAM11 p.622 fn 5, secondary). Precursor paper: t(15) = 2.8, p < .02, 12 of 16 infants longer to nonwords (SAN96b p.379).
2. **The preference survives frequency matching** (ASN98): LF words (TP 1) vs HH-part-words (TP .5) of equal frequency; infants still look longer at the part-words. Human proportion-better .04 (FAM11 Table 1, p.622, secondary). Replication in British English: 10.7 s words vs 12.6 s part-words, t(12) = 2.57, p < .025, N = 13 (FAM11 p.628 fn 9).
3. **Backward TP / frequency can drive segmentation when forward TPs are equal** (FAM11's new experiment): 9.4 s words vs 12.2 s partwords, t(19) = 4.62, p < .001, N = 20; human proportion-better .13 (FAM11 p.629, p.622).
4. **Adults segment a six-word trisyllabic language, with a nonword foil easier than a part-word foil**: 76% vs 65% correct (PV98 p.247, secondary), the nonword > part-word difference being significant (FGGT10 p.12). The mapping of 76 to nonwords is my inference (see 2.1).
5. **Within-language word differences**: words with high mean TP (dutaba, pidabu, tutibu) 79% vs low-TP words (babupu, bupada, patubi) 72%, significant (PV98 p.255, secondary). PARSER failed this one (PV98 p.255).
6. **Incidental exposure gives smaller but reliable learning**: 58.6% after one 21-min session, 73.1% after two (PV98 p.247, secondary); mean 21.1/36 and 26.3/36 for adults, children not different from adults (PV98 p.256, secondary).
7. **Performance falls with sentence length** (Frank Exp 1): condition coefficients 2.31 (1 word), 1.56, 1.64, 1.39, 1.32, 1.31, .86 (12 words), .45 (24 words, p = .12, not different from chance) (FGGT10 p.4).
8. **Performance rises steeply with exposure, then plateaus** (Frank Exp 2): 48 and 100 tokens at chance (beta .082, .39); significant from 300 tokens; "largely comparable between 300 and 1200 tokens" (FGGT10 p.5).
9. **Performance falls with vocabulary size** (Frank Exp 3), opposite to what better TP statistics would predict: beta 2.12, 1.77, 1.17, .84, .39 for 3, 4, 5, 6, 9 types; 9 types at chance (p = .19); the type effect remains after controlling tokens per type (types beta = -.11, p = .004) (FGGT10 pp.5-6).
10. **Benchmark model scores a replication should beat or match** (correlation with Frank's human means, Exps 1/2/3): lexical model .94/.89/-.98; TP .84/.43/-.99; PARSER .00/.86/.00 (target-only .92/.84/.89); TRACX .92/not run/.97; SRN .60/not run/.33 (FGGT10 p.10, p.12; FAM11 p.625). No model had been shown to fit all three before TRACX's partial result; the capacity-limited lexical model fits "for some parameter regime" with unread r values (FGGT10 p.15-16).

## Numbers available only from figures

- FGGT10 Figure 1 (p.4): human percent correct per individual participant and condition mean +/- SEM for all three experiments. Only the regression coefficients above are in text.
- FGGT10 Figure 2 (p.10): model curves (only the r values are in text); Figure 3 (p.11): target and distractor relative scores per model; Figures 4 (p.14) and 5 (p.16): evidence-limited and capacity-limited fits, whose r labels are garbled in the text layer (only DM-TP vs lexical r = .90 is in prose).
- SAN96b Figure 1 (p.379): mean listening times for words and nonwords with SE.
- SAN96 Science Figure(s): looking times with error bars, and the N, t and p of both experiments: not read.
- FAM11 Figure 3 (p.624), Figure 4 (p.625), Figure 6/7 (pp.628-629): TRACX and human curves; the human equal-TP means (12.2 s, 9.4 s) are in text.
- PP12 Figure 1 (p.810): MDLChunker and PARSER proportion correct against repetitions; only the t statistics at 5 and 10 repetitions are in text.

## Practical notes for replicating with TRELLIS (chunk and concept hierarchy learner)

- Token-level protocol to copy first: SAN96b/SAN96 infant stream (4 words x 45 tokens, no immediate repeats, 12 syllables, TP 1.0/.33) and Frank Exps 1-3 (18 syllables, six words of 2, 2, 3, 3, 4, 4 syllables, 600 words, 30 word/part-word pairs, 12 random languages per condition, Luce choice rule, Pearson r on condition means). These are the most completely specified protocols read.
- For infants the measure is a looking-time difference; the community convention here is the normalised "proportion better" (A - B)/(A + B) (FAM11 p.622), which turns model error or score differences into something comparable to .04-.13.
- Before running ASN98, SNA96 and Experiment 2 variants, obtain the primary papers; the numbers for them above are second-hand.

## Human condition means read from FGGT10 Figure 1 (digitised, approximate)

Read on 2026-11-21 from the solid mean line of Figure 1 (preprint p.4, UCI copy), rendered at 450 dpi; each panel's y axis runs 30-100 at 12.0 px per point, so readings carry about +/-1 percentage point of reading error. Use these for correlations with model condition means (as FGGT10 did); the regression coefficients in 2.3 remain the text values.

| Exp 1 words per sentence | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 24 |
|---|---|---|---|---|---|---|---|---|
| mean % correct | 87.2 | 80.5 | 80.8 | ~79 (no visible vertex; on the 3-6 segment) | 75.0 | 77.0 | 68.6 | 60.3 |

| Exp 2 total tokens | 48 | 100 | 300 | 600 | 900 | 1200 |
|---|---|---|---|---|---|---|
| mean % correct | 57.2 | 63.8 | 70.8 | 70.5 | 75.3 | 80.0 |

| Exp 3 word types | 3 | 4 | 5 | 6 | 9 |
|---|---|---|---|---|---|
| mean % correct | 87.4 | 81.1 | 75.3 | 71.8 | 65.0 |

Consistency check: the Exp 2 ordering of the four larger conditions matches the printed betas .75, .76, 1.03, 1.33 (300, 600, 900, 1200), which confirms the presumed order in 2.3.
