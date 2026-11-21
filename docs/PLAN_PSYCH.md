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
| P3 | Skilled memory, digit span of DD (Richman, Staszewski & Simon 1995 data); Chase & Ericsson | docs/CE1981_PROTOCOL.md, design docs/PLAN_PSYCH_DS.md | concepts as meaningful codes from prior experience, chunks as groups, retrieval structure given as in EPAM IV | EPAM IV (RSS95) | Pilot B done 2026-11-21 (experiments/v3/psych_digits.py, results/psych_digits/pilotB.md). The chess recognition read (THETA = 16 bits) never fires on the sequence learner C3: 0 groups in 3 seeds x 6k units, best 4-12 bits. C3 builds the content description of an isolated group from donated context bags that change between occurrences, and its context description is the same for every isolated group (the same defect leaves the segmentation bits score at chance). With groups imposed (diagnostic arms): identity decode saturates 3-digit groups, and the projected span reaches 80 near session 130 against DD's ~500 (explosion); mixed 3/4-digit groups stall near 7 (4-digit coverage); label-only decode stalls. **Decision: P3 is not a replication.** It is reported as a failure analysis: TRELLIS stores without a rate or capacity limit, the same limit as Frank Exp 3. P4 is added in its place. |
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
