# Psychology experiments with TRELLIS: plan and status

Started 2026-11-21. The paper is to cover two fields: language modeling, and replications of psychology
experiments. The replications follow the grounding style of MacLellan, Singaravadivelan, Lian, Wang & Langley
(CogSci 2026), "A Rational Account of Categorization Based on Information Theory" (arXiv 2603.29895).

## Template (from the CogSci 2026 paper, kept)

For each experiment:
1. **Human Study Design.** The original procedure, stimuli, N and measures, in one or two paragraphs.
   Numbers come only from the protocol documents, which cite the page each number was read from.
2. **Simulation Methodology.** The human procedure reproduced one to one: the same stimuli, trial sequence and
   stopping rules, with simulated N equal to human N. Responses are read off the model as a score.
3. **Results and Discussion.** The signature effects the experiment was designed to show, plus a rank or
   Pearson correlation on condition or item means where human means exist. Comparison models' numbers are
   taken from their papers.

## What we add beyond that paper

- **One fixed configuration.** The CogSci 2026 paper retuned alpha per experiment; we do not. All experiments
  share one parameter set:
  - K = 7 short-term slots, the chess replication's main value; K = 4 is also reported.
  - THETA = 16 recognition bits.
  - support = 3.
  - Sequence experiments use the language learner C3. Chess uses the C5 graph learner, which is the same
    theory with a different encoding of the input.
- **A removal control for every claimed mechanism.** Concepts off (leaf reads) and chunks off, as in Tables
  S19 and S20, plus experiment-specific removals. The CogSci 2026 paper listed ablations as future work.
- **One summary table:** experiment, human N, signature effects, model reproduces (yes / partly / no),
  comparison model.

## Experiments

| # | Experiment | Protocol doc | Mechanism tested | Comparison models | Status |
|---|---|---|---|---|---|
| P1 | Chase & Simon (1973) chess recall by skill; Gobet & Simon (1996) random positions | docs/CS1973_PROTOCOL.md | chunks (recognition and decoding), concepts | CHREST and MAPP (numbers in the protocol) | Done: cs73x and cs73L runs, removals in Table S20. Random-position skill effect from existing runs: middle-game random positions 4.1-4.45 / 5.15-5.65 / 5.3-6.4 at B/A/M, against game positions 7.2 / 9.7 / 12.9. That is a gain of about 1.6 pieces against 5.7; humans gain about 1 against 5 pieces per 400 Elo (GS96a p. 8). |
| P2 | Statistical word segmentation: Frank et al. (2010) Experiments 1-3 (adults; sentence length, exposure, word types); Saffran, Aslin & Newport (1996) infants (words vs nonwords and part-words) | docs/SAN96_PROTOCOL.md | chunks found from raw syllable streams; posterior storage | TP, Bayesian lexical, MI clustering, PARSER (FGGT10), TRACX (FAM11) | Done 2026-11-21 (commit 7a54727d; results/psych_segment/SUMMARY.txt). Pearson r with human means: Exp 1 .93, Exp 2 .90, Exp 3 -.96. Exp 3 has the wrong sign, as for TP and the lexical model; only TRACX and the capacity-limited models fit it, so it is stated as a limitation. Concepts off: Exp 1 .77, Exp 2 .63 (Spearman -.03, the exposure curve is gone). Infants: words beat nonwords and part-words in 16 of 16. The bits score is at chance and is dropped as diagnostic. |
| P3 | Skilled memory, digit span of DD (Richman, Staszewski & Simon 1995 data); Chase & Ericsson | docs/CE1981_PROTOCOL.md, design docs/PLAN_PSYCH_DS.md | TRELLIS as the semantic net inside EPAM IV's procedure (slots, 25% forgetting, session activation, prefix coding, associative links); concepts as semantic codes | EPAM IV (RSS95) | **Replicates (P3f, 2026-11-22).** Span 99.1 at session 865 against DD's 104; 80 first reached at sessions 606-759 against DD's ~500. 4-digit groups err more than 3-digit groups (.056 against .029; EPAM .030 against .015). Prior knowledge speeds early learning. Letter span 8.5 on the re-registered probe; the original 10-list probe failed as a probe artifact. Removal controls credit associative links (42 without them), concepts (72) and prior knowledge. Between-group codes are not credited. Discrepancies: TRELLIS stores about 9,300 strings against DD's ~3,000, and it shows no plateau near 80. History: Pilot B, P3b (2.7x fast), P3c (stall), P3d (16), P3e (42), P3f (99); all pre-registered. Figure: results/psych_digits/P3f_curve.png. |
| P4 | Artificial grammar learning: chunk strength against grammaticality (Knowlton & Squire 1996; Meulemans & Van der Linden 1997, amount of training) | docs/AGL_PROTOCOL.md (in progress) | concepts (grammaticality beyond chunk strength) against chunks (chunk strength) | ACS and bigram/trigram TP (no published model fits these data; BD03 says so) | Done 2026-11-21 (experiments/v3/psych_agl.py, results/psych_agl/SUMMARY.txt, 18 subjects). **Mechanism not supported.** Primary score (span_w), cells GH/GL/NGH/NGL 61/40/72/27 against human 65/63/55/28, r = .64. The interaction is human-like (23.6 against 24.7), but there is no grammaticality effect (+1.4: -10.4 within high CS, +13.2 within low). Concepts off is almost identical (0.0, r = .61), so the nominal pass of the criterion (1.4 > 0.0 > -10.4) is tie-breaking noise and is not counted. TP beats TRELLIS (r = .80). The secondary scores (logZ, regen) show grammaticality effects of +28 to +38, with chunk strength reversed. They were not pre-registered as decision scores and are not used. Reading: from 46 presentations, C3's concepts (support 3) barely form, so the learner reduces to chunk familiarity. |

## P3b pre-registration: digit span with a storage-time limit (fixed 2026-11-21, before any run)

The user asked to keep P3. Pilot B named two causes, and each gets one change; neither uses a constant tuned to DD.
- **Recognition for sequences = stored familiarity.** A 4-digit prefix of the buffer, else a 3-digit prefix, is a recognised group if its stored posterior weight span_w is >= 0.5. This is the P2 score; the chess bits read does not transfer to C3 (Pilot B). The learner is unchanged.
- **Storage-time limit (EPAM IV's constant).** Building a new chunk takes 8 s (RSS95 p. 313; Simon's 4-8 s, ES87 p. 204). A clock accrues presentation time (1 s per digit, as for DD). Storing a new group, i.e. a learn() call on a group or ungrouped piece whose string has span_w < 0.5, costs 8 s and happens only when the clock holds at least 8 s. Pieces are considered in list order, and unaffordable pieces are not learned. A group that is already stored is reinforced for free: its span_w is increased by 1 without a learn() call. The clock carries over between lists and sessions.
- Everything else is as in PLAN_PSYCH_DS / psych_digits.py: C3, B = 4, K = 7 STM, retrieval structure given, identity decode, the prior stream (DD's categories), learning after recall.
- **Procedure:** a staircase from 7, plus one digit after a correct list and minus one after an error, with lists per session from RSS95 Table 1, up to 865 sessions or wall time. Seeds 13, 17, 19.
- **Arms:** full (prior on); prior off; no limit (the Pilot B explosion reference); clock sensitivity (presentation plus rehearsal, 2.9 s per digit from RSS95 Table 5: (100 + 190) s / 100 digits). The sensitivity arm is secondary.
- **Probes (no learning):** at sessions 50, 100, 250, 500 and 865, a 10-list probe staircase of random consonants starting at 5 (material specificity; DD/SF span on letters ~ 6-7), plus the 4- against 3-digit group error rates.
- **Signature effects scored against DD:** (1) span rises well above 7 with practice; (2) span reaches about 80 near session 500, not before session 250 and not never (the rate); (3) 4-digit groups err more than 3-digit groups; (4) prior knowledge speeds early learning; (5) letter span stays about 7. The decision is on the primary clock.

### P3b result (2026-11-21; results/psych_digits/P3b_*.json, P3b_SUMMARY.md)

Span at sessions 15/50/100/250/500/865, mean of 3 seeds (DD: 7-9 at the start, ~80 at ~500, 104 at ~865):
- full: 20 / 33 / 51 / 105 / 198 / 266;
- prior off: 7.9 / 10 / 14 / 48 / 118 / 209;
- rehearsal clock: 22 / 37 / 57 / 117 / 181 / 264.

The session when span first reaches 80 is 175-189 for full, 330-389 for prior off, and 162-177 for the rehearsal clock.
- Effect 1 (span rises far above 7): holds.
- Effect 2 (rate, 80 near session 500): **fails, about 2.7x too fast.**
- Effect 3 (4- against 3-digit): holds only as a proxy. Recognition and identity decode coincide, so there are no decode errors; 4-digit prefixes are recognised far less often (8% against 79% in sessions 101-250).
- Effect 4 (prior speeds learning): holds.
- Effect 5 (letter span stays near 7): holds (7-8).

Diagnosis (full arm, seed 13):
- The 8 s limit binds only early: in sessions 1-50 the clock was below 8 s after learning in 178 of 680 lists, and in sessions 101-250 in 0 of 870.
- The nolimit arm is close to the full arm at session 20 (span 26-34 against 33).
- Span is therefore set by coverage of 3-digit strings, with no retrieval failure for a recognised group: a list fails only when the STM overflows.
- DD's slow rate rests on what TRELLIS lacks: forgetting of codes at about 25% per item, which EPAM IV offsets with redundant retrieval cues that grow with practice (RSS95 pp. 319, 324).
- Adding retrieval failure would be a new mechanism, so it is left to the user. The pre-registered banking of clock time across lists is not the cause: the clock is unspent after session 100 even per list (about 4.7 learns per list against length/8 affordable).

## P4b pre-registration: Meulemans & Van der Linden (1997) Exps 2a/2b (fixed 2026-11-21, before any run)

- Stimuli and procedure: docs/AGL_PROTOCOL.md section 2.
  - 2a: 32 training strings, shown twice in the same order (one random order per subject, repeated).
  - 2b: 125 training strings, shown once; shown twice is a secondary sensitivity check.
  - Test: the 32 test strings of each experiment.
  - One learn() call per presentation (TRELLIS always reproduces a string after one presentation, so the reproduction retries add nothing). Learning frozen at test.
  - N = 20 per experiment, seeds 13..32.
- Score, endorsement rule, arms (full, concepts off, ACS with MVdL's global ACS, TP) and secondary scores are as in P4. There is no time limit: P3b shows the limit is not what sets the rate there, so the fixed configuration stays the one used in P2 and P4. A secondary arm adds the limit (3 s per presentation, 8 s per new string) for completeness.
- Human targets (cells GA, GNA, NGA, NGNA; JS99 values, which match the figure reads):
  - 2a: 65.3 / 42.8 / 67.5 / 45.0 (associativity +22.5, grammaticality -2.2);
  - 2b: 67.8 / 62.8 / 56.3 / 60.3 (associativity +0.5, grammaticality +7.0).
- **Signature effect = the training shift:** the grammaticality effect grows from 2a to 2b (+9.2 in people), and the associativity effect shrinks (-22).
- **Mechanism claim:** concepts carry the shift. It is supported if the full model's grammaticality-effect shift (2b minus 2a) is positive and larger than concepts off's and ACS's.
- Also reported: Pearson r with the 8 cell means.
- Caveat stated in advance: the test sets balance global ACS but not terminal-trigram strength (4.63 against 3.31 in 2b), so a position-sensitive score could win on 2b without concepts. The primary span_w score is position-free.

### P4b result (2026-11-21; results/psych_agl/SUMMARY_MVDL.txt, 20 subjects per experiment)

**Mechanism not supported.** The full model's grammaticality shift (2b minus 2a) is -8.8 ± 1.4, against +9.2 in people. Concepts off and ACS are both -12.5, and TP is 0. The criterion's "larger than" holds only as a smaller negative shift.
- The associativity effect is far too large in every arm (+50 to +88, against people's +22 in 2a and +0.5 in 2b), and it does not fall with training (full: +6.9).
- Cell-mean r: 2a .93 (every arm .93-.995, carried by associativity); 2b .03; all 8 cells .56 (TP .70).
- The secondary limit arm gives a shift of +5.6 ± 3.3 (about 1.7 SE). It was not pre-registered as the decision arm and is not counted.
- Reading: the stored-familiarity read is a chunk-strength measure, and C3's concepts do not add a grammaticality signal that grows with training on this task.
- The nolimit P3b reference was cancelled at session 70. It tracks the full arm (span at session 50: 33-43 against 31-39), which confirms that the limit does not set the rate.

## P3c pre-registration: retrieval failure with redundant codes (fixed 2026-11-21, before any run; user chose this route)

EPAM IV's account of DD's rate (RSS95 pp. 315, 319-321, read from the PDF text): a retrieval-structure slot holds the group's digits and its codes (semantic category, pattern codes). "About one fourth of the new information that has been attached to a retrieval structure node is lost when attention shifts from it" (p. 315), once, "25% per digit" (p. 319). The semantic net is never forgotten. Recall fills lost digits by finding the category (from the slot, or guessed from the surviving digits) and searching that category for the chunk, "the digits in the retrieval structure serve as a partial image to guide the search" (p. 321). RSS95 state that the 25% "can be set to match DD's overall learning rate"; DD's own pattern-code forgetting is 23.6% (p. 324). It is taken here as EPAM's constant, not refitted.

P3c = P3b unchanged (C3, stored-familiarity recognition, the 8 s storage limit with free reinforcement, B = 4, K = 7, staircase, Table 1 lists, learning after recall, prior stream) plus one mechanism:
- **Slot contents.** For each RS group g of k digits (recognised in the stream), the slot holds its k digits and its TRELLIS label, the chess STM label used in Pilot B: c, the deepest supported content concept of g's isolated parse, and x, the deepest supported context concept (climb_path, read with learning off, before this list is learned). A label part sitting at the root, or a parse with no top composite, is not stored.
- **Forgetting.** When attention moves on, each digit and each label part is lost independently with f = .25, once. STM units are not forgotten (STM failure is overflow, as in P3b).
- **Recall of g.** (1) All k digits survive: recalled. (2) Otherwise the candidates are the familiar stored strings (span_w >= .5) of length k that agree with the surviving digits at their positions. If c and/or x survive, a candidate's score is the summed weight of its stored records whose content leaf is under c and whose context leaf is under x (the chess composite decoder with the surviving digits as a partial image); with no label part surviving, the score is span_w (EPAM's guess from the digits). The best-scoring candidate is recalled (ties: lowest string, as the chess decoder). A surviving label part with no candidate under it recalls nothing (EPAM gives up the search).
- A list is correct iff the STM holds <= 7 units and every RS group is recalled.
- **Arms:** full (primary); no rescue (any lost digit loses the group: the RS alone, RSS95's "retrieval structures alone cannot explain"); context off (x not stored); prior off; f = .20 and f = .30 (RSS95's own sensitivity values; secondary). Seeds 13, 17, 19; 865 sessions or 23.5 h.
- **Signature effects scored against DD:** (1) span rises well above 7; (2) **rate (decision): span (5-session block mean, mean of seeds) first reaches 80 between sessions 250 and 865**; (3) 4-digit groups are recalled wrongly more often than 3-digit groups (now a real recall error, sessions with lists >= 30); (4) prior knowledge speeds early learning; (5) letter span stays about 7. Redundancy claim: no rescue gives a lower span than full at session 250.
- Logged per list: groups by size, digits intact, rescues by route (c and x, c only, x only, digits only), errors by size.
- **Prediction stated before running (the risk).** Pilot B shows the context description of an isolated group is constant, so x should carry nothing. If only c rescues, then even with a perfect c decode a 3-digit group is lost with probability >= P(some digit lost) x P(c lost) x (1 - completion) = .58 x .25 x (1 - D3), so reliability <= about .86-.90 and the span ceiling is near 25-35, short of DD's 80 (which needs about .969 per group, RSS95 p. 319 logic). EPAM reaches .984 with three independent sources (1 - .25^3). A stall here would show that TRELLIS supplies two of EPAM's three codes for random digit groups.

### P3c result (2026-11-21; results/psych_digits/P3c_*.json, P3c_SUMMARY.md): stall, as predicted but lower

All 18 runs completed 865 sessions. Span at sessions 15/50/250/865 (mean of 3 seeds): full 9.5 / 9.9 / 7.7 / 6.6 (maximum block span 12.4); no rescue 6.9 / 6.2 / 5.7 / 4.6; context off 9.5 / 9.3 / 8.1 / 8.8; prior off 7.9 / 8.0 / 8.8 / 8.3; f = .20 10.7 / 10.5 / 8.3 / 9.1 (maximum 15.0); f = .30 8.5 / 9.6 / 6.7 / 6.6.
- Effect 1 (span well above 7): **fails** (maximum 12.4). Effect 2 (rate): **fails, stall**; 80 is never reached, and neither is 20. Effect 3: not reached (no lists >= 30). Effect 4 (prior helps early): holds (9.5 against 7.9 at session 15). Effect 5 (letter span about 7): holds. Redundancy (rescue raises span): holds (7.7 against 5.7 at session 250).
- **Why: the rescue does not grow with practice.** In the full arm, per-group reliability is .67 at sessions 0-50 and .63 at 500-865. Digits survive intact in 40% of groups, rescue from both label parts succeeds in .51 then .45 of attempts, from c alone in .50 then .47, and from x alone in .28 then .13. Context off is no worse (p(3) .64), so the context concept adds nothing, as predicted.
- The loop that keeps it flat: an error shortens the next list, short lists give few new pieces to learn, and the content concept of an isolated digit string is not stable enough to sharpen (Pilot B: label-only decode needed about 15k learned units to reach .89; here span stays near 9, so few units are ever learned).
- Reading: at EPAM's 25%, TRELLIS's codes (one label, about .5 rescue) give far less redundancy than EPAM's three sources (RSS95 p. 319). P3b without forgetting is 2.7x too fast, and P3c with EPAM's forgetting stalls near 10. A forgetting rate between 0 and .20 would place the curve near DD's, but it would be a parameter fitted to DD's rate, as RSS95 say theirs is (p. 319). That choice is left to the user. P3 stands as a partial replication.
- **Decision (user, 2026-11-21): fit the forgetting rate only if a psychological account motivates the value. None does, so f is not fitted.** The only human measure of slot forgetting is DD's own pattern-code loss, 23.6% (220 of 288 codes reported, RSS95 p. 324), which is what EPAM's 25% rests on. A rate below .20 would contradict that measurement, and its only support would be the learning curve it is meant to predict. P3 is reported as partial, with both bracketing curves: f = 0 (P3b) climbs 2.7x too fast, and f = .25 (P3c, DD's measured loss) stalls. The deficit the bracket locates is redundancy, not forgetting. At DD's forgetting, a 100-digit list needs about .977 reliability per group (p. 319), and EPAM's three code sources give .984. TRELLIS's one concept label gives .64, and that does not rise with practice.

## P4c pre-registration: position-sensitive context scores (fixed 2026-11-21, before any run; user chose this route)

Two new endorsement scores, never computed before on any AGL dataset. The learner, training, test items, median-split endorsement, seeds and arms are as in P4 (KS96) and P4b (MVdL).
- **pos (anchored familiarity; chunks plus position, no concepts).** For each sub-span t (length >= 2) of the test string at positions (i, j) of n, the summed posterior weight of stored records of t whose span had the same boundary status in training (begins its string iff i = 0, ends it iff j = n). Score = mean over sub-spans.
- **fit (context-concept fit; concepts).** Parse the test string with learning off. For each chosen composite e: c = deepest supported content concept, x = deepest supported context concept (climb_path, as in P3). fit(e) = W(c and x) / W(x), where W sums the weights of stored composite records whose content leaf lies under c and whose context leaf lies under x. In words: how much of what TRELLIS stored in this kind of context was this kind of chunk. Score = mean fit over chosen composites (0 if none).
- **New baseline: anchor ACS** (MVdL's positional definition, AGL_PROTOCOL 2.5: mean of the number of training strings that begin with the item's first bigram, with its first trigram, end with its last bigram, with its last trigram), computed from the training list for KS96 and MVdL.
- **Stage 1, selection on KS96 (18 subjects, full and concepts off).** A score passes if the full model's grammaticality effect (G - NG, mean of the high and low chunk-strength cells) exceeds concepts off's for the same score, the printed ACS's and anchor ACS's. Of the passing scores, the one with the larger full-model grammaticality effect is selected. If neither passes, P4c stops negative and MVdL is not run.
- **Stage 2, confirmation on MVdL 2a/2b (20 subjects each), selected score only counted.** Supported if the full model's grammaticality shift (2b - 2a) is positive, and larger than concepts off's (same score), ACS's and anchor ACS's. Reported: shift ± SE, associativity shift, Pearson r with the 8 cells. The unselected score is computed in the same runs and reported, not counted.
- Prediction stated before running: MVdL matched G and NG items on global and anchor ACS, so pos is expected to show little grammaticality effect there; only fit can carry a shift.

### P4c result (2026-11-21; results/psych_agl/p4c/SUMMARY_P4C.txt): stage 1 fails, so P4c stops negative and MVdL is not run

KS96, 18 subjects. Grammaticality effect within CS cells (human +22.0):
- pos: full -23.6, concepts off -25.0. Fails; it is a chunk-strength score with position, NGH 95%.
- fit: full +15.3 (H +20.1, L +10.4; cells 59/56/39/46, r .52), concepts off +17.4. It beats printed ACS (-9.7) but not concepts off or anchor ACS. Fails.
- anchor ACS (baseline): +21.5, cells 79/43/41/38, r .58. The positional count of first and last bigrams and trigrams alone reproduces the human grammaticality effect on the KS96 items.
- The anchor ACS code reproduces AGL_PROTOCOL's MVdL cell means exactly (2a 4.78/3.31/4.44/3.41; 2b 18.00/13.38/18.25/12.25).

Reading: the KS96 grammaticality effect is matched by anchor chunk strength, so a grammaticality score has to beat a positional chunk count before concepts are credited, and TRELLIS's context fit does not. The concepts-off control is weak for fit, because the score itself climbs the trees in both arms (concepts off changes learning and parsing, not the read). The anchor comparison decides the outcome either way. AGL is reported as a negative result. TRELLIS's stored familiarity is chunk strength (P4, P4b), and its context-concept fit adds nothing beyond anchor strength.

## P3d pre-registration: EPAM IV's architecture with TRELLIS as the semantic net (fixed 2026-11-22, before any run; user approved both changes)

Why. rescuediag (3d4120b5, P3c full arm, 3 seeds, no change to the run): the target was always among the candidates, but the decoder picks the most frequent stored string, so the prior's running times beat the string just heard. Rescue succeeds .87-.92 when the surviving label leaves one candidate (about 40% of attempts), 0 when the target's records lie outside its own label (about 10%, cause not traced), and .15-.18 when three or more remain. Picking the most recently encoded candidate would have recovered .97. RSS95 (read from the PDF text, logs/rss95_full.txt) supplies three mechanisms that P3c lacked:
- **Activation lasts one session.** "Once a branch or a node has been activated, it remains activated throughout that particular experimental 'day'. In the interval between sessions it loses its activation"; "the depth-first search mechanism restricts EPAM's search to activated portions of the net" (p. 314; cited as p. 310 when registered, corrected 2026-11-22). A node is activated when EPAM sorts through it or adds information to it.
- **Every group gets a slot, and rescue needs a stored chunk.** "DD's semantic net is incomplete—many digit groups are not represented in it—but it gets larger as the trials continue", about 3,000 groups by the end, "substantially all three-digit groups but only a fraction of the 10,000 possible four-digit groups" (Appendix, p. 329). Coding is obligatory (ESC93 p. 973). EPAM's 3- against 4-digit error difference comes from this coverage difference (p. 319).
- **Storage while the digits are still present**, at EPAM's 8 s per new chunk (p. 313).

What TRELLIS contributes, stated: the semantic net (EPAM's hand-built 14 category subnets, 99% of DD's groups, p. 325) is replaced by TRELLIS learned without supervision from the prior stream and the lists. The retrieval structure, the 25% forgetting and the activation rule are EPAM's.

Procedure (one list; everything not listed is as in P3c: C3 learner, prior stream, staircase from 7, RSS95 Table 1 lists per session, seeds 13, 17, 19, 865 sessions):
1. **Groups from EPAM's retrieval-structure patterns (p. 316), not from recognition.** From the start of the list: one d16 block (4,4,4,4) if at least 16 digits remain, then d21 blocks (3,3,3,4,4,4) while at least 21 remain. The remainder r < 21 is split into 3s and then 4s: r = 3a if r mod 3 = 0; one 4 if r mod 3 = 1; two 4s if r mod 3 = 2. Any 1-2 digits left over (r = 1, 2, 5) are singles. For lists under 16 digits, only the remainder rule applies. EPAM's 5-digit primitive is not used, since DD may form 5 as 2+3 (p. 312).
2. **Rehearsal in STM (p. 321).** Taken from the end of the list: the singles, then whole groups while the total is at most 7 digits, always at least the last group. These are recalled without forgetting. All other groups are retrieval-structure (RS) groups.
3. **Storage during presentation.** A clock starts at 0 for each list and gains 1 s per digit heard. When a group's last digit arrives:
   - if its string is familiar (span_w >= .5), span_w is raised by 1 at no time cost and the string is activated for this session;
   - otherwise, if the clock holds at least 8 s, the group is learned (learn() on the group, clock minus 8) and activated;
   - otherwise it is not stored. Singles are not stored.
   Unspent time does not carry to the next list (storage happens while the items are present). The activation set is cleared at the start of every session.
4. **Labels.** After presentation, each RS group's label (c, x) is read as in P3c: the deepest supported content and context concepts of its isolated parse, with memory frozen.
5. **Forgetting (P3c, unchanged).** Each digit, c and x of an RS group are each lost once with p = .25. A group with all digits intact is recalled.
6. **Rescue (changed).**
   - Candidates are the familiar strings of the group's length that agree with the surviving digits and are activated this session.
   - With none, the group fails. With one, it is recalled.
   - With several, the candidate with the most record weight under the surviving label parts wins (P3c's score). If that weight is 0, or no label part survives, the most frequent candidate (span_w) wins. The label guides the choice rather than gating it.
7. A list is correct iff every RS group is recalled.

Arms (3 seeds each):
- **full;**
- **actoff:** activation not required, so all familiar strings are candidates (P3c's candidate set with P3d's slots and storage);
- **conceptsoff:** labels never used, so the choice is by frequency among activated candidates (a flat string store);
- **prioroff:** no prior stream;
- **f0:** forgetting off.

Probes, as in P3b:
- **Letters:** at sessions 50/100/250/500/865, a 10-list staircase of random consonants from 5, with the same grouping, slots, forgetting and rescue but no storage. Consonant strings are never familiar, so RS groups survive only intact.

Measures and pre-registered effects (arm full, mean of 3 seeds; span = mean list length over the 5-session block, as P3b/P3c):
1. Span rises well above 7: maximum block span >= 20.
2. Rate: the first session with block span >= 80 lies in [250, 865]. Before 250 = too fast; never = too slow. Span at 865 is reported against DD's 104.
3. 4-digit RS groups err more than 3-digit RS groups (per-group error, sessions 250-865, pooled). EPAM: 3.0% against 1.5% on 100+ digit lists (p. 319).
4. Prior knowledge speeds early learning: span at session 50 is higher in full than in prioroff.
5. Letter span stays near 7: final letter-probe length <= 9 at every probe.

Reported, not scored:
- **trial order:** per-group RS error by trial order 1-6 within sessions 250-865, against DD's slowing of about 7 s per trial (Table 3, a time measure);
- **release:** error on trial 1 of a session against the last trial of the previous one;
- **net size:** familiar 3- and 4-digit strings at session 865, against RSS95's estimate of about 3,000 (nearly all 1,000 three-digit, a fraction of the four-digit);
- **post-session miss rate:** the share of a session's groups not stored, i.e. not activated, against DD's 8% missed in free recall (p. 326). An upper bound on the misses, since free recall is not simulated.

Mechanism readings:
- activation is credited if full beats actoff in span at 865 on every seed;
- concepts are credited if full beats conceptsoff on every seed, otherwise TRELLIS works as a string store here and that is reported;
- f0 is the no-forgetting reference.

Decision: P3 is a replication if effects 1, 2, 3, 4 and 5 hold. It is partial otherwise, with the failed effects named. No parameter is changed after the runs; any follow-up is a new, separately registered section.

### P3d result (2026-11-22; results/psych_digits/P3d_*.json, P3d_SUMMARY.md): partial. Slow climb, with the 4-digit groups as the wall

Span at sessions 15/50/100/250/500/865 (mean of 3 seeds):
- full: 9.1 / 11.5 / 11.0 / 15.1 / 16.6 / 15.7 (maximum block span 21.9);
- actoff: 9.2 / 9.4 / 8.8 / 10.1 / 8.9 / 9.0;
- conceptsoff: 9.7 / 10.0 / 10.4 / 13.0 / 15.8 / 18.2;
- prioroff: 8.8 / 9.5 / 9.6 / 10.6 / 13.6 / 15.6;
- f0: every list correct, so span rises by one per list (194 by session 6); stopped after about 10 sessions.

Effects:
- Effect 1: holds (21.9).
- Effect 2 (rate): **fails, too slow**; 80 is never reached.
- Effect 3 (4-digit error > 3-digit): holds, .269 against .096 (EPAM 3.0% against 1.5%).
- Effect 4 (prior helps early): holds (11.5 against 9.5 at s50).
- Effect 5 (letter span <= 9 at every probe): **fails narrowly**. Probe means are 7.7-9.7, but one seed ends a 10-list probe at 11.

Mechanisms:
- Activation is credited (full beats actoff on every seed, 15.5-16.0 against 8.0-10.0).
- **Concepts are not** (conceptsoff 15.6/16.5/22.5 against 15.5/15.7/16.0), so TRELLIS acts as a string store here.

Reading:
- **The shape is now DD-like:** reliability grows with practice. The 3-digit error falls .435 -> .262 -> .122 -> .056, while the 4-digit error stays near .25-.30.
- **The 4-digit groups are the wall.** Past 16 digits the retrieval structure opens with the d16 block of four 4-digit groups, and .75 to .70 per 4-digit group caps the span near 16-20.
- **Unrecoverable groups dominate the failures:** about 90% are "no activated candidate", i.e. the group was not stored. That is 20% of groups late, at 8 s per new chunk.
- **Net size is not the cause.** The net holds 931 three-digit and about 3,870 four-digit strings at the end, more than RSS95's ~3,000 for DD.
- **What EPAM does that P3d does not** (RSS95 p. 320, encoding steps 7-9; first cited as p. 318):
  - It sorts a 4-digit group as its first three digits plus a blank, so its semantic code comes from the 3-digit prefix chunk ("(4, 6, 4, _) sorts to ... 10-mile time").
  - It attaches between-group codes: back-to-back same category, plus symmetry and ascending or descending category patterns at the parent node.
  - Both are redundancy that does not need the 4-digit string to be stored.
  - These are candidates for a separately registered P3e.

Trial order (reported): error .215/.184/.238/.161 for trials 1-4, with no monotone build-up; release .197 -> .215 (none).

## P3e pre-registration: EPAM's 4-digit prefix coding and between-group codes (fixed 2026-11-22, before any run; user said "go for P3e")

Both mechanisms are EPAM IV encoding steps (RSS95 p. 320 (first cited as p. 318), read from logs/rss95_full.txt). Everything not listed is P3d unchanged: grouping, rehearsal, storage clock, session activation, 25% forgetting, staircase, seeds 13/17/19, 865 sessions, the same probes, effects 1-5 and their criteria.

**A. Prefix coding of 4-digit groups (step 7).** "If the RSF node will hold four digits, EPAM sorts the list (4, 2, 4, _)": a 4-digit group's semantic code comes from its first three digits.
- **Encoding.** When a 4-digit group is presented and its 3-digit prefix is familiar (span_w >= .5), the prefix is activated for the session ("a branch is activated whenever EPAM sorts through it", p. 314). The prefix is not reinforced and costs no time. Storage of the group itself is as in P3d.
- **Label.** A 4-digit group's label (c, x) is the label of its prefix, read as in P3d.
- **Recall of a damaged 4-digit group:**
  1. Run the P3d search over activated familiar 4-digit strings. If it finds any candidate, its pick is final.
  2. If there is no activated 4-digit candidate, take the prefix route. The 4th digit must have survived, since it is a raw digit with no other source. The prefix is intact, or recovered by the P3d search over activated familiar 3-digit strings from the surviving prefix digits and the label. The group is recalled as prefix plus 4th digit.

**B. Between-group codes (steps 8-9).** Codes are computed on each group's content concept c, before forgetting.
- **Parents.** The d16 block is one parent. Each d21 block is two parents: its three 3-digit groups, and its three 4-digit groups. In the remainder, the 3s form one parent and the 4s another. Rehearsed groups belong to their parent, and their labels are always known (from STM).
- **Back-to-back (step 8):** attached to a group whose c equals the c of the previous group under the same parent.
- **Identical (step 9):** attached to a parent of at least 2 groups whose c are all equal.
- **Symmetry (step 9):** attached to a parent of at least 3 groups whose first and last c are equal and not all equal.
- EPAM's ascending and descending progressions need an order on categories, which TRELLIS concepts lack, so they are omitted.
- Each code is one more item, lost once with p = .25 (as pattern codes, p. 319).
- **At recall.** Groups are taken in list order. A group's label is known if its c survived, if the group was rehearsed, or if the group has already been recalled (its label is read from the recalled string). When an RS group's c was lost:
  - a surviving back-to-back code gives it the previous group's known label;
  - otherwise a surviving identical code gives it any known sibling's label;
  - otherwise a surviving symmetry code gives the first group the last group's known label, or the reverse.
  The recovered label then guides the rescue as in P3d.
- EPAM's time costs for these checks (100 ms to check plus 200 ms to attach) are far below the 8 s storage cost and are not charged.

**Arms (3 seeds each):**
- **full:** A + B;
- **prefixoff:** B only;
- **patternoff:** A only;
- **none:** neither. This must reproduce P3d full exactly, as a check;
- **conceptsoff:** A on, no labels (so B has nothing to compare);
- **prioroff:** full without the prior stream.

**Predictions, stated before running:**
- A lowers the 4-digit error (P3d .27 at sessions 250-865) and raises span.
- The floor for an unstored 4-digit group is the loss of its 4th digit, .25. A group is unstored about 20% of the time, so A alone cannot bring 4-digit groups to 3-digit reliability.
- B is expected to be small or null, because P3d showed that labels do not change outcomes (conceptsoff = full).

**Decision:**
- Effects 1-5 are scored exactly as in P3d.
- A is credited if full beats prefixoff in span at 865 on every seed; B if full beats patternoff on every seed; concepts if full beats conceptsoff on every seed.
- P3 is a replication if effects 1-5 hold. Otherwise it is partial, with the failed effects named.
- No changes after the runs.

### P3e result (2026-11-22; results/psych_digits/P3e_*.json, P3e_SUMMARY.md): partial, but span reaches 42-51 and concepts are now credited

Span at sessions 15/50/100/250/500/865 (mean of 3 seeds; DD about 80 near 500 and 104 at 865):
- full: 9.9 / 10.9 / 11.1 / 17.1 / 26.7 / 42.2 (maximum block span 51.3, still rising);
- prefixoff: 9.1 / 10.3 / 11.0 / 14.3 / 15.7 / 15.9;
- patternoff: 10.0 / 12.2 / 11.1 / 16.5 / 28.8 / 43.9;
- none: identical to P3d full on every seed (check passed);
- conceptsoff: 10.1 / 11.6 / 11.6 / 15.1 / 25.5 / 33.1;
- prioroff: 8.9 / 8.6 / 9.7 / 13.7 / 18.4 / 39.3.

Effects:
- Effect 1: holds (51.3).
- Effect 2: **fails, too slow.** 80 is never reached, and span at 865 is 42 against 104.
- Effect 3 (4-digit error > 3-digit, sessions 250-865): holds, .105 against .072 (EPAM .030 against .015).
- Effect 4: holds (10.9 against 8.6 at s50).
- Effect 5: **fails** on the same probe draws as P3d. The probe is identical in every arm, because consonant strings are never familiar and the probe uses its own random stream: one seed ends a 10-list probe at 11. This is a property of the probe at 25% forgetting, not of an arm.

Mechanisms:
- **Prefix coding (A): credited** (41.6-43.2 against 15.6-16.3 on every seed). It cuts the 4-digit error .242 -> .080 at sessions 500-865, and the unstored share .21 -> .09. EPAM's step 7 is the whole gain.
- **Between-group codes (B): not credited** (patternoff 39.1-50.1). They recovered only 36 labels at sessions 250-865, as predicted.
- **Concepts: credited** (full 41.6-43.2 against conceptsoff 31.7-34.5 on every seed; 3-digit error .062 against .110 at 500-865). With prefix coding, the label of the 3-digit prefix picks the right activated chunk, so the concept hierarchy now does work.

Reported:
- **Trial order:** per-group error rises within a session, .080 / .094 / .102 / .155 / .180 for trials 1-5.
- **Release:** last trial .102 against the first trial of the next session .080.
- Both have the direction of DD's slowing by trial (Table 3, a time measure) and of proactive interference with release overnight.
- **Net size:** 989 three-digit and 6,303 four-digit familiar strings at the end, more than RSS95's ~3,000 for DD.

Reading: failures at sessions 500-865 are now mostly wrong picks among activated candidates (1,376) rather than unstored groups (754). EPAM has a third information source that P3e lacks: "EPAM stores information about the digits as they are presented, both in the retrieval structure and with associative links to the chunks that are already available in semantic memory" (p. 310). Ur's appendix counts the forgetting of these links at 25% (p. 329). Its exit test finds "a node in the discrimination net that points to (associates to) a particular node in the retrieval structure" (p. 313). This is P3f.

## P3f pre-registration: associative links from known chunks to retrieval-structure slots (fixed 2026-11-22, before any run; continues P3e under the user's instruction to keep iterating)

Everything is P3e full unless listed (A + B, labels, activation, storage, forgetting, staircase, seeds 13/17/19, 865 sessions, probes, effects 1-5 and criteria).

**C. Associative links (RSS95 p. 310, p. 313, Appendix p. 329).**
- **Encoding.** When an RS group is presented, a link from a chunk already available in semantic memory to the group's slot is stored, if such a chunk exists. That is the group's own string if it is familiar before presentation, else (4-digit groups) its 3-digit prefix if that is familiar. Groups stored during this list, i.e. new chunks, get no link: "chunks that are already available".
- **Forgetting.** The link is lost once with p = .25. It is drawn from a separate random stream, so the arm without links reproduces P3e full exactly.
- **Recall of a damaged RS group with a surviving link:**
  - a link to the full string gives the group;
  - a link to the prefix gives the first three digits, and the 4th digit must have survived;
  - otherwise the P3e procedure runs unchanged.

**Arms (3 seeds):**
- **full:** A + B + C;
- **linkoff:** P3e full; it must reproduce P3e full exactly, as a check;
- **conceptsoff:** C on, no labels;
- **prioroff:** full without the prior stream.

**Prediction:** wrong picks fall and span rises. The rate is still bounded by 4-digit coverage and by the loss of the 4th digit. Whether span reaches 80 inside [250, 865] is not predicted.

**Decision:**
- Effects 1-5 are scored as in P3d/P3e.
- C is credited if full beats linkoff at 865 on every seed; concepts if full beats conceptsoff on every seed.
- Replication iff effects 1-5 hold.
- Effect 5 is the same probe as before and cannot change: it depends on no arm. Its verdict is carried as "fails, probe artifact" unless the probe itself is re-registered.

### P3f result (2026-11-22; results/psych_digits/P3f_*.json, P3f_SUMMARY.md): DD's learning rate and final span reproduced; effects 1-4 hold, effect 5 fails as a probe artifact

Span at sessions 15/50/100/250/500/865 (mean of 3 seeds):
- full: 10.6 / 11.1 / 13.6 / 20.6 / 51.0 / 99.1 (maximum block span 106.3);
- linkoff: identical to P3e full on every seed (check passed);
- conceptsoff: 9.7 / 10.2 / 11.9 / 17.9 / 36.9 / 71.9;
- prioroff: 8.5 / 8.9 / 9.3 / 11.4 / 18.7 / 72.4;
- DD: 7-9 / - / - / - / ~80 / 104.

Effects:
- Effect 1: holds (106).
- **Effect 2: holds.** The first session with span >= 80 is 759, 606 and 608 (DD about 500); span at 865 is 99.1 [97.6-101.9] against DD's 104.
- Effect 3: holds, .056 against .029 for 4- against 3-digit groups (EPAM .030 against .015).
- Effect 4: holds (11.1 against 8.9 at s50).
- **Effect 5: fails as registered** (max probe final 11), with the same probe draws as P3d/P3e, independent of the arm.

Mechanisms:
- **Links: credited** (97.6-101.9 against 41.6-43.2 on every seed). Recall by a surviving link covers 37,148 of the damaged groups at sessions 250-865.
- **Concepts: credited** (against conceptsoff 70.0-74.9 on every seed).
- **Prior: credited early** (s50 11.1 against 8.9; s500 51.0 against 18.7).

Reported:
- **Trial order:** per-group error .034 / .043 / .049 / .108 / .180 for trials 1-5.
- **Release:** .049 -> .034 (direction of DD's trial-order slowing, Table 3).
- **Net size:** 1,000 three-digit and about 8,300 four-digit familiar strings at the end, against RSS95's estimate of about 3,000 groups in DD's net. **TRELLIS stores about 3x more than DD**, a discrepancy to report.
- **Unstored share** at sessions 500-865: 2.3%, against DD's 8% missed in free recall (an upper bound only).
- **No plateau near 80.** DD's plateau of about 250 sessions near 80 was not reproduced by EPAM either (protocol §9); TRELLIS crosses 80 and keeps rising.

What the P3 claim now is: with EPAM IV's procedure and its information sources (slot digits, a semantic code with prefix coding, associative links, session activation, 25% forgetting fixed by DD's measured 23.6%), TRELLIS learned without supervision in place of EPAM's hand-built semantic net reproduces DD's learning rate and final span. TRELLIS's concept hierarchy carries part of the effect: removing labels costs about 27 digits at session 865. The between-group pattern codes add nothing.

## P3f letter-probe re-registration (fixed 2026-11-22, after the P3d-P3f probe failures, before running the new probe)

The registered probe is one 10-list staircase from 5 per probe point, judged on its final length. It is independent of every arm: consonant strings are never familiar, so the result depends only on the probe's random draws. A single final length of a short staircase is too noisy to test "letter span stays near 7".

The original verdict stays on record ("fails"). This section registers a better measurement of the same quantity, using the same staircase rule as digit span:
- 5 independent consonant staircases, one per probe point (sessions 50/100/250/500/865), each of 50 lists from length 5, with the same grouping, rehearsal, forgetting and rescue as P3f and no storage.
- Letter span = mean list length over lists 11-50 of each staircase.
- Criterion: the mean over staircases and seeds is <= 9, and no staircase exceeds 10.
- The human value is from a secondary source only: SF's letter span stayed near 7 (Cowan 2013, via CE1981_PROTOCOL §L). DD's letter span is not in the opened sources.
- Seeds 13/17/19 with a new probe stream (seed * 211 + 1). Since the probe uses no memory, it is run standalone.

### Letter-probe result (2026-11-22; results/psych_digits/P3f_LETTERPROBE.md): holds

- **Spans per seed** over the 5 probes:
  - seed 13: 8.3 / 9.05 / 7.9 / 8.75 / 8.25;
  - seed 17: 8.85 / 8.35 / 8.15 / 7.9 / 9.0;
  - seed 19: 9.55 / 8.05 / 8.05 / 8.65 / 8.85.
- **Verdict:** mean 8.51, max 9.55, so the re-registered criterion **holds**. Letter span stays near the no-skill level while digit span reaches 99.
- **Caveat:** 8.5 is above SF's ~7 (secondary source). It is set by the 7-item rehearsal rule plus the chance that a 4-letter slot survives intact (.75^4).

**P3 status: replication.**
- Effects 1-4 hold as registered.
- Effect 5 holds on the re-registered probe; the original probe's "fails" stays on record.
- Figure: results/psych_digits/P3f_curve.png (experiments/v3/psych_digits_plot.py).

## P4 pre-registration (fixed 2026-11-21, before any AGL simulation)

- Target: Knowlton & Squire (1996) Experiment 1, controls (N = 18), the 23 distinct training strings shown twice in random order, and the 32 test items, all from docs/AGL_PROTOCOL.md section 1. Meulemans & Van der Linden (1997) is not simulated, because its stimuli are not available.
- Learner: C3, as in P2 (psych_segment.py CFG). Each training presentation is one learn() call. Learning is frozen at test. Simulated subject = seed; 18 seeds.
- **Primary score:** chunk familiarity, the P2 score carried over unchanged. It is the mean, over every sub-span of length >= 2 of the test string, of the stored posterior weight span_w of that token tuple. A subject endorses the items scoring above their own median, with ties at the median split at random, so about 50% endorsement against the humans' 52.5%.
- **Secondary scores (reported, not used for the decision):** (a) parse logZ per token; (b) regeneration bits per token of the MBR tree (_deriv_bits).
- Measures: the four cell means; main effects of grammaticality and chunk strength; the interaction (the chunk-strength effect for NG items larger than for G items); Pearson r with the four human cell means.
- Arms: full; concepts off (as in P2); baselines ACS (the printed KS96 chunk-strength values) and a bigram/trigram transition-probability model with the same median-split rule.
- Mechanism claim: concepts carry grammaticality beyond chunk strength. It is supported if the full model's grammaticality effect within chunk-strength cells exceeds that of ACS and of concepts off.

## Decision rules

- P2 is a go if the full model correlates positively with the human means on Experiments 1 and 2 and
  negatively-trending conditions come out in the right direction on Experiment 3, without retuning. Frank et
  al.'s benchmark is that no model fit all three; TRACX gets .92 and .97 on Experiments 1 and 3. If Experiment
  3 fails, report it the way FGGT10 reports its models.
- P3 is a go if Pilot B shows neither explosion (span reaching 80 well before DD's ~500 sessions) nor stall
  (span stuck near 7 x group size), under a decode rule that is also valid for chess. If both decode rules
  fail, P3 is reported as a partial replication: material specificity, the ceiling without a retrieval
  structure, and 4- against 3-digit errors. It may also be replaced by a better-fitting chunking experiment,
  for example Servan-Schreiber & Anderson (1990) competitive chunking in artificial grammar learning.
