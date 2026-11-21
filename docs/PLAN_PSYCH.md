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
| P4 | Artificial grammar learning: chunk strength against grammaticality (Knowlton & Squire 1996; Meulemans & Van der Linden 1997, amount of training) | docs/AGL_PROTOCOL.md (in progress) | concepts (grammaticality beyond chunk strength) against chunks (chunk strength) | competitive chunking (Servan-Schreiber & Anderson 1990), PARSER, SRN, as the protocol papers report them | Protocol being collected |

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
