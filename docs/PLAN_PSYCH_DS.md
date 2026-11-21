# Plan: skilled-memory digit span (DD) with TRELLIS v2

Status: design only, 2026-11-21. Nothing here has been run. Human and EPAM IV numbers come only from
`docs/CE1981_PROTOCOL.md` (cited as "P" plus the RSS95 page it gives). Every other number is a design
estimate and is marked as one.

Comparison model: EPAM IV (Richman, Staszewski & Simon 1995, "RSS95"). Subject: DD (P §1). SF is not
simulated, as in RSS95 (P §9): SF's own data were not read.

Structure follows the CogSci 2026 paper: Human Study Design / Simulation Methodology / Results and
Discussion. The TRELLIS configuration stays the one used for chess (C&S 1973) and Saffran:
`Trellis3(store="best", naming="bands", restructure="sample", type_names="sample", type_key="concept",
children=False, spine=False, heights=False, tree="cu", refresh="verify", refresh_budget=300)` (the cs73x/cs73L
recipe without its GraphTrellis-only keys), window 2, boundary "marker", support 3, theta 16, K = 7 (K = 4 as
the sensitivity row, as in chess). None of these is retuned for this experiment.

---

## 1. The core problem: what limits span in a model with no time and no forgetting?

TRELLIS stores everything it learns, permanently, at no cost in time. Give it a place to keep each group
of a list (a retrieval structure), and the obvious design recalls any list perfectly from session 1. The
learning curve would then mean nothing. RSS95 bounds EPAM with two timing facts: slot information is
forgotten at 25% when attention moves away (P §9, p. 319), and storing a new node takes 5-8 s (p. 308, 313).
TRELLIS has neither. So the bound has to come from what TRELLIS can actually write into a slot and read back.

**Candidate limits.**

| | (a) Label decodability: K slots plus decoding ambiguity | (b) Retrieval interference in the context hierarchy | (c) Recognition budget per group | (d1) Slot code capped at band resolution | (d2) Learning budget = the K attended units per list |
|---|---|---|---|---|---|
| Mechanism | A slot holds one label (content concept, context concept), the same object a chess STM slot holds. Recall decodes it with the chess rules: exact if the group is a stored chunk, otherwise the most frequent stored instance under the concept pair. A group is lost when its label decodes to another group. | The context carries the RS address and the previous group. The decoding pool is the set of instances stored at that address, so earlier lists compete. | Encoding a group may cost at most a fixed number of node tests (1 digit/s with EPAM's 10 ms per test, about 300-400 per group). Past the budget the label stops at a coarser concept. | Labels named only down to the finest band (12 bits, `band_bits`). | As chess, memory learns only the K units in STM per trial. |
| Learning curve | Rises. Per-group reliability p(t) grows as more groups become decodable, through exposure in earlier lists and the prior stream. At staircase equilibrium p^G = .5, so span ≈ s̄ · ln 2 / (−ln p), with s̄ ≈ 3.5 digits. | Falls over practice. Nothing releases interference between sessions, since there is no time, so it piles up across all 865 sessions. | Falls late. Cobweb sort paths get deeper as memory grows, so encoding cost rises. DD gets faster with practice (principle c, P §8). | Flat. A random 4-digit group's share (about 1/20,000) never reaches 2^-12, so no learning. | Rises slowly (≤ 7 groups learned per list). |
| Plateau | None built in. Only by chance, as for EPAM. | None. | None. | Immediate ceiling. | None. |
| Trial order (effect 11) | Slight facilitation: each list adds familiar groups. Wrong sign compared with DD's slowing, which is a time effect. | Slowing within a session ✓, but also across sessions ✗. | None. | None. | None. |
| 4- vs 3-digit errors | 4 > 3, because 10,000 four-digit strings take ten times longer to cover than 1,000 three-digit ones. | No size prediction. | 4 > 3 (deeper sorts). | 4 ≫ 3 | 4 > 3 |
| Removal control | Identity decode off (concepts only); content-only and context-only decoding; RS off. | Address out of the context. | Budget off. | Cap off. | Learn all groups. |
| Verdict | **Recommended.** | Rejected as the limit. Kept as a weak proxy for effect 11 (§4). | Rejected (wrong direction). Node counts kept as a timing proxy. | Rejected. | Sensitivity row only. It conflicts with effect 9: DD recalls 92% of a session's groups afterwards (P §6, p. 326), so all groups reach LTM. |

**Recommendation: (a), label decodability.** Reasons:
1. It is the only candidate that predicts growth rather than decline.
2. It uses TRELLIS parts that already exist: the label is the chess STM label, and decoding is the chess decoder, both unchanged.
3. It makes testable predictions with no new parameter: 4-digit groups fail more than 3-digit groups (effect 13), and without an RS the span stops near K labels × group size.
4. It gives a numeric target from the protocol. RSS95 says 30 groups need .977 reliability per group for one perfect list in two (P §5, p. 319). By the same staircase logic, 80 digits (about 22 groups) need p ≈ .969 and 104 digits need p ≈ .977. These are the targets for p(t) at sessions ~500 and 865.

What (a) does not supply: a learning-rate limit. EPAM's net grows by about one node per list (109 to about
3,400 nodes, P §9, p. 318) because nodes cost time. TRELLIS learns every group of every list (about 14 per
list late in practice). The main risk is therefore an accelerating curve: span ∝ 1/(1−p) blows up as p → 1
(§7, R1). The pilot exists to catch this early.

---

## 2. Encoding (presentation)

**Human (P §2, §4).** Digits are read at 1 per second, ungrouped. DD imposes his own groups of 3-4 digits
(sometimes 5, as 2+3) and gives each a semantic code and an RS address. Both operations are obligatory
(ESC93 p. 973). DD rehearses the last two groups (p. 321). The auditory sensory store lasts 3,800 ms, about
4 digits (p. 320). LTM storage runs at about one chunk every 3-4 s (p. 314).

**Simulation.** Per trial:
1. The list length is announced and the RS frame is prepared (§3).
2. Auditory buffer A holds at most B = 4 digits (from the auditory store, p. 320). As digits arrive, whenever |A| = 4 (or the list has ended):
   - parse the prefixes A[:4] and A[:3], each as an isolated token sequence, with learning off (`L._quiet = True`, `L.parse`);
   - the group is the longest prefix whose top composite is recognised: `L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx) ≥ THETA`, the chess recognition rule;
   - a recognised group goes into the next RS slot with its label (the content and context climbing concepts, the chess `climb_path` ends) and leaves A;
   - if no prefix is recognised, the head digit goes to STM as a single-digit unit.
3. STM is a FIFO of at most K = 7 units. Overflow drops the oldest unit, which is then lost.
4. At the end of the list, A is flushed the same way, but its units go to STM, not to RS slots: the tail stays in STM, as DD rehearses the last groups. Only STM counts at recall, so a novice who recognises nothing has span K = 7, the normal 7-9 range (P §1).

**Group size: bounded by documented constants, chosen by recognition.** The upper bound of 4 is the
auditory store (p. 320). The lower bound of 3 is the LTM storage rate: at 1 digit/s and one chunk per 3-4 s
(p. 314), a slot can be filled at most once per 3 digits. Neither bound is fitted. Within [3, 4], the size
and the boundaries emerge from recognition, and so does whether groups form at all. DD's 5-digit leaves,
being 2+3 (p. 310), are outside this range. That is a known simplification. Measures: the share of 3- vs
4-digit groups over practice, and boundary positions on 100-digit lists against DD's first three spikes at
digit locations 4, 8 and 12 (four 4-digit groups first, p. 323). Removal control: **floor off** (2-digit
groups allowed). The prediction is a span explosion, because the 100 pairs become familiar within a few
sessions. That would show the timing-derived floor carries load.

**What is stored.** After recall, mirroring chess's learn-after-trial (labels stay fixed during a trial, so
concept ids cannot move between encoding and recall), memory learns each encoding unit of the list as one
experience (`L.learn(digits)`):
- every group, RS or STM;
- every run of ungrouped digits, cut into consecutive pieces of B = 4 (a remainder of 1-3 stays as its own piece).

This is "learn what was perceived, in auditory-store-sized pieces". Memory learns the digits heard, not the
digits recalled. With `store="best"`, a 4-digit piece also commits its 2- and 3-digit sub-composites. That
is how new groups enter the vocabulary.

**Why isolated prefix parses.** Groups are learned as isolated experiences (`<s>`/`</s>` windows), so
recognising an isolated prefix sees the same contexts as storage did. Parsing the whole list is cubic and
mixes contexts across group boundaries. Cost: two charts of ≤ 4 tokens per group.

---

## 3. Retrieval structure (RS)

**Can it emerge?** No, not as stored content. An RS has to bind new contents to fixed addresses on every list.
TRELLIS stores instances; it does not rebind variables. Supergroup chunks of a random list never repeat,
so they never become decodable. EPAM was also given its RS as an initial condition: 43 nodes for 37-digit
lists, plus the d3/d4/d5 patterns (P §9, p. 316-317).

**Given, as EPAM.** The RS is one named added part. It is an ordered frame of slots, each holding one
label for the current list, cleared before the next list. EPAM's slots are likewise overwritten per list.
The traces of earlier lists survive only in what TRELLIS learned. Slots have no size: group sizes come
from §2. For the timing proxies only, slots are arranged into a tree with RSS95's construction rules,
applied to the number of groups: an initial block of 4 groups, then blocks of 6 (3+3+3, 4+4+4 shape);
root branching ≤ 4 below 82 digits, 5 for 83-102, 6 for 103-123 (P §5, p. 316).

**Difference from EPAM, stated.** EPAM's RS starts at 37 digits and has to be extended by learning.
Ours is given at every length, which is more than EPAM was given.

**Controls.**
- **RS off:** recall from STM alone (K labels). Predicted span ≤ 7 × s̄, about 21-28 digits. ES87 p. 204 puts the best fast-rate spans before SF at about 20 digits. That is a reading, not a test.
- **RS capped at the 37-digit frame:** groups past the frame go to STM. Diagnostic only.
- **EPAM-sized slots:** RS imposes d3/d4 sizes (d16 = 4×d4, d21 = 3×d3 + 3×d4). A comparison arm, not the headline.

---

## 4. Signature effects: reachability and measures

Recall and decoding, as in chess (`chess_cs1973_exact.py`, `View.decode` / `Mem`, ported to Trellis3
records):
- **identical:** the group's digit string is a committed stored chunk; decodes exactly;
- **concept:** the most frequent stored composite string under (content concept, context concept);
- **omission:** either concept at the root.

A single-digit STM unit decodes to its token. A list is correct when the decoded sequence equals the list.
Staircase: +1 after a correct list, −1 after an error (p. 316). Lists per session follow RSS95 Table 1
exactly (P §2). The first length is 7 and the length carries over between sessions; both are modelling
choices. Span = mean list length per 5-session block (as Fig. 4), with the Woodworth 50% span as a
secondary measure (P §2).

Status key: **R** = reachable with no new part; **P** = proxy (with its defensibility); **G** = given by
design, not a result; **X** = out of scope.

| # | Effect | Status | Mechanism / measure | Human (P) | EPAM IV (P) | Removal control |
|---|---|---|---|---|---|---|
| 1 | Span 7-9 → ~104 over ~865 sessions | R | Label decodability under the staircase. Span per 5-session block at the text anchors: start 7-9; ~80 by ~session 500; ~90 after ~790; peak ~104 (p. 306-307, 316, 318). | as left | Matched DD in 16 blocks; interlaced over the first 500 sessions; two runs (p. 318) | RS off; prior off; identity off; learning off after the prior stream |
| 2 | Lists per session shrink 25 → 3; ~1 error per 2 lists | G | Table 1 is an input. The 1-in-2 error rate follows from the staircase for any model. Reported, not claimed. | Table 1; p. 316 | Same | – |
| 3 | Plateau ~80 for ~250 sessions | X (predicted absent) | RS shape does not enter decoding. Checked by a flat-segment test on blocks 100-150. | p. 318 | Not reproduced | RS capped at 37 (diagnostic) |
| 4 | Ceilings near 37/58/79/100 | X (predicted absent) | No mechanism. Span histogram modes reported. | p. 318 (mild, possibly illusory) | Reproduced | – |
| 5 | 3-4 digit groups; 12 categories; categorisation 97-99% reproducible | R (sizes, categories), P (reproducibility) | Group-size distribution (§2). Category emergence: purity/ARI of content concepts at each band against the prior-stream generator's source category, and against a DD-rubric labeller of list groups (analysis only, like chess `ideas()`). Reproducibility proxy: the same string sorted to the same band concept at two checkpoints. | 12 cats (ESC93 p. 972); 97-99% (p. 325) | 14 subnets given; 99% with DD (p. 325) | Prior off (do categories still form from random lists?) |
| 6 | Hierarchical RS, 2-4 branches, 3/4/5 leaves | G | Given (§3) | p. 316 | Given | RS off |
| 7 | Pauses: ~200 ms within, +250 ms per RS node between groups | P (by construction) | Nodes traversed in the given RS tree between consecutive groups; 0 within. This only restates the RS. The non-trivial part is where boundaries fall (locations 4, 8, 12, p. 323). No r with DD is possible: Fig. 8 was not read. | r = .932 / .964 is EPAM-vs-DD (p. 323) | .932 / .964 | EPAM-sized slots (boundary placement) |
| 8 | Rehearsal: backward; last two groups; 184-190 s | X (timing); R (structure) | Only "how many tail groups sit in STM" is measurable. Rehearsal time is out of scope. | Last two groups; 184-190 s / 199-200 s (p. 321, 326-327) | Studied last group; 91 s / 118 s (Table 5) | – |
| 9 | Free recall by category; 94% clustered; 8% missed | P | Session membership is given (records stored this session: an episodic tag TRELLIS lacks). Output order = depth-first traversal of the content tree. Measures: share of adjacent outputs in the same rubric category; within-category Spearman against numeric value; repeats output adjacently. Missed = 0% by construction (no forgetting), as EPAM. | 94%; 8% missed (p. 326) | 100%; 0 missed; DD-EPAM order ρ .782-.985 (Table 4) | Content tree replaced by a random order (chance clustering) |
| 10 | Enriched vs depleted lists | P (weak) | Test lists generated with / without DD's pattern types (test stimuli are the experimenter's role). % correct plus proxies: recognition bits, share of identity decodes, sort nodes. Predicted null: no relational codes (mirror, symmetry, add-em-up are not representable as nominal attributes). Reported as a limitation. | 99.8 vs 98.8%; 72.4 vs 115.6 s (Table 2) | 99.4 vs 99.6%; 86.3 vs 102.6 s | – |
| 11 | Slowing across trials, ~7 s per list | P (weak) | Same-session fan: the number of this session's stored instances under each label's concept pair (Anderson-style fan as retrieval time). Needs the session tag, as in 9. Spearman with Table 3's six medians. Predicted sign of accuracy by trial order: flat or improving (§1). | DD 7 s/trial (r² .544) (p. 324) | 6 s/trial (r² .974) | Fan over all sessions (shows no release) |
| 12 | Forgetting ~24-25%; redundancy needed | P | No forgetting. The analogue is decode loss = 1 − p at equilibrium, which the staircase fixes (≈ ln 2 / G), so it is not a test. The redundancy part is testable: decode with the content concept only, or the context concept only. | 23.6% (p. 324) | 24.3%; 25% parameter (p. 319, 324) | Content-only / context-only decode |
| 13 | 4-digit groups err more than 3-digit | **R** | Per-group decode error by size at the final blocks (lists ≥ 100 if reached) | "higher", not quantified (p. 319) | 3.0% vs 1.5% (p. 319) | EPAM-sized slots; identity off |
| 14 | Probe: next/previous group of a probe | R (errors), P (RT) | The probe group is parsed to a label and matched to RS slot labels. Error = label collision or decode error. RT = RS nodes traversed (by construction; forward = backward by construction). | 5.8% error (p. 325-326) | 1.9%; r = .70 with DD's RTs (p. 326) | – |
| 15 | Material-specific skill | R (direction) | Consonant lists on the session-865 memory. Predicted span ≈ K = 7 (no recognised letter groups). Plus lists built from strings the prior stream never produces. Direction only: SF's letter span was not found (P §7). | Not found (ES87 pp. 216, 222: little transfer, SP/BB) | Not simulated | – |

Rank correlations exist only where item-level data were read: Table 3 (six trial-order medians, effect
11) and Table 4. Table 4's ρ values are DD vs EPAM on DD's own lists, which we do not have, so they cannot
be recomputed. The learning curve can be correlated block by block only after RSS95 Fig. 4 is digitised
(open item, §7).

---

## 5. Prior-knowledge stream (raw experience, no labels)

DD was a runner who knew times for standard distances, plus ages and dates (P §1, p. 306, 313). Each item
is one digit string learned as one experience (`L.learn(list(digits))`). The source category is kept
outside the learner, for analysis only. Formatting drops colons and points, as in DD's codes: 359 = 3:59
mile; 8201 = 8:20.1 two-mile; 391 = age 39.1; 1776 = a year (ESC93 p. 972). Strings are clipped to 3-4
digits, matching the [3, 4] group range of §2. Each category has a truncated normal over its range (mean
at the centre, sd = range/6), rounded to the format's resolution, so common times repeat as they do for
a runner. All ranges, shapes and weights are modelling choices, documented here and not fitted.

| Cat. (RSS95 p. 317) | Range | Format → digits | Example |
|---|---|---|---|
| a quarter mile | 45.0-79.9 s | s s t (3) | 562 |
| b half mile | 1:50-2:59 | m s s (3); m s s t (4) at p = .3 | 158, 1582 |
| c three-quarter mile | 2:55-4:29 | m s s (3) / m s s t (4) at p = .3 | 312 |
| d one mile | 3:45-6:59 | m s s (3) / m s s t (4) at p = .3 | 359, 4053 |
| e 3 km | 7:30-11:59 | m s s (3) below 10:00, else m m s s (4) | 845, 1012 |
| f fast 2-mile | 8:00-9:59.9 | m s s t (4) | 8201 |
| g slow 2-mile | 10:00-13:59 | m m s s (4) | 1032 |
| h 3 mile | 13:00-19:59 | m m s s (4) | 1445 |
| i 10 km | 27:00-49:59 | m m s s (4) | 2915 |
| j 10 mile | 46:00-59:59 | m m s s (4) (times past 1 h omitted: 5 digits) | 5212 |
| k dates | years 1900-1989 (p = .7), 1492-1899 (p = .3), uniform | y y y y (4) | 1776 |
| l ages | 10-99, normal(40, 18) truncated; tenths at p = .5 | a a (2, learned but never a group) / a a t (3) | 29, 391 |
| m double ages | two ages as in l | a a a a (4) | 8969 |
| n patterns | not generated (that would hand-code relations); they occur only by chance in lists | – | – |

**Size.** Categories are equally weighted. N_pre is the smallest number of draws (seeded) whose
running-time items (a-j) contain at least 500 distinct strings. That count is RSS95's own guess, "perhaps
he had 500 or more" (p. 318). Expected order: several hundred to ~1,000 draws (estimate). The stream is
shuffled and learned once before session 1.

**Rows.**
- **Prior off:** empty memory at session 1.
- **Prior ×10:** reported as a sensitivity row and never selected.

Pre-registered check: the session 1-15 span must stay within DD's 7-9 range (P §1). A large prior stream
would break that, because identity decoding would make many list groups familiar from the start.

---

## 6. Compute estimate and pilot

**Full run (estimates).**
- Lists: 15×25 + 5×15 + 5×10 + 15×8 + 180×6 + 45×5 + 210×4 + 390×3 = **3,935 lists**.
- Digits, if the model follows DD's trajectory: about 190-200k, about 55k groups. A list is capped at 208 digits (2× DD's peak) so a runaway stays bounded; hitting the cap is logged as an explosion.
- Per group: ≤ 2 parses of ≤ 4 tokens (≤ 10 composites each) and one `learn` of 3-4 tokens.
- Memory: about 6-7 records per experience, so about 0.4M records.
- Main cost: `refresh_budget = 300` re-descriptions per experience, plus the decode index (leaf → records). The index is rebuilt once per list, only when a concept decode is needed; identity decodes are a set lookup.
- Reference points: Trellis3 learned 1,000 TinyStories sentences in 118 s with refresh_budget 0 (r212 log). GraphTrellis chess cost about 2-3 s per position at 5-27k positions.
- Guess: 0.3-2 s per experience, so roughly 6-36 h per seed on one CPU, 16-64 GB.
- Probes run in forked children at sessions 50, 100, 220, 475, 650 and 865 with learning off: fixed 30/50/100-digit probe lists, enriched/depleted, probe task, free recall, letters.
- Seeds 13, 17, 19, via sbatch on overcap (`-p overcap -A overcap --exclude=sky1`, `-c 1`). Nothing runs on the login node.
- Full runs only for: headline, RS off, prior off, identity off. Every other control runs at checkpoints (forked) or at pilot scale.

**Pilot A (mechanism, sessions 1-50, Table 1 counts).**
- 15×25 + 5×15 + 5×10 + 15×8 + 10×6 = 680 lists, about 10-12k digits, about 3k groups. Under an hour per run (estimate).
- Arms × 3 seeds: headline, identity off, RS off, prior off, floor off, K = 4.
- Logged per list:
  - each failure's cause: RS decode commission, RS omission, or STM overflow;
  - per-group p by size and by how many times the string was seen before;
  - recognised-but-not-identical rate;
  - group-size shares;
  - records and seconds per experience (throughput for the full-run projection).

**Pilot B (exposure curve, no staircase).** Learn random-list encoding units up to 15k groups (about
sessions 1-200 of exposure). Measure p(N) by group size on fixed held-out probe lists at N = 1k, 3k, 10k
and 15k. Project the staircase span with G*(N) = ln .5 / ln p(N) and DD's digits-per-session, then check
when p reaches .969 (80 digits) and .977 (104 digits), the targets from §1.

**Go/no-go gates before the full run (pre-registered).**
- G1: mean span over sessions 1-15 lies in [6, 11].
- G2: 4-digit decode error > 3-digit at matched exposure.
- G3: projected session at which span reaches 80 lies in [250, 865] (DD: ~500).
- G4: RS-off ceiling ≤ 30.
- G5: projected wall time ≤ 3 days per seed; memory ≤ 64 GB.
- Failing G3 or G4 stops the full run, and the result is reported as a negative finding on mechanism (a), with no retuning.

---

## 7. Risks and falsifiers

**Risks.**
- **R1, explosion (most likely).** Identity decoding makes any string heard once perfectly recallable. The 1,000 three-digit strings saturate within roughly 100-200 sessions of exposure (estimate). If grouping drifts to 3 digits, p → 1 and span ∝ 1/(1−p) shoots past DD. The cause is that TRELLIS has no learning-rate limit: EPAM's ~1 node per list comes from time. Signs: a convex curve, 3-digit share → 1, the list cap being hit.
- **R2, stall.** Unfamiliar 4-digit prefixes reach θ (recognised as "a kind of" group) and decode to the wrong instance. p is then bounded by 4-digit coverage (1/10,000 per string), and span stays near the STM ceiling for hundreds of sessions. Sign: a high recognised-but-not-identical rate.
- **R3, no recognition early.** Isolated 3-4 digit strings seen a few times may not reach 16 bits, so everything goes to STM and span stays at 7 until the 4-digit pieces have been learned often enough. Sign: RS slots unused in pilot A.
- **R4, degenerate context hierarchy.** Isolated groups have `<s>`/`</s>` windows, so the context tree mostly carries within-group position and the donated content concept. Here "where it occurs" lives in the given RS, not in TRELLIS. The content-only / context-only decode rows show whether context adds anything. If not, say so: this experiment supports the content hierarchy and the chunking, not the two-hierarchy claim.
- **R5, decoding becomes a lookup table.** With the chess identity rule, the concepts matter only for unfamiliar groups; TRELLIS then contributes grouping and recognition, not decoding. The identity-off arm (support-3 concept decoding only) is the test. If identity-off cannot exceed the RS-off ceiling, report it.
- **R6, expected failures.** Effects 3 and 4 (plateau, ceilings) and the 8% missed in free recall: shared with EPAM, except effect 4.
- **R7, timing effects.** Effects 7, 8, 10 and 11 are proxies or out of scope. The fan proxy for effect 11 cannot reset between sessions without a session tag.
- **R8, given more than EPAM.** The given RS works at every length (EPAM's stopped at 37), and the group-size floor comes from a timing constant. A critic can call both imposed. The floor-off and RS-37 rows quantify how much each carries.
- **R9, no curve data.** RSS95 Fig. 4 was not read, so only the text anchors are available. Open item: digitise Fig. 4A/B (gwern PDF) to get DD's 175 block means and compare block by block, as RSS95 did.
- **R10, pre-exposure size sets the early span.** Fixed a priori from RSS95's "500 or more" (p. 318); ×10 reported, never selected.
- **R11, compute.** `refresh_budget = 300` over ~0.4M records, and rebuilding the decode index per list, may be slow. Pilot A measures it.

**Results that would falsify mechanism (a)** (fixed now):
- F1: span reaches 80 before session 250, or never by session 865 (DD: ~500).
- F2: after session 50, most failed lists fail by STM overflow rather than RS decode errors. The limit is then not decodability.
- F3: 4-digit per-group error ≤ 3-digit (effect 13 reversed).
- F4: RS off reaches > 40 digits. Chunking in STM alone would then explain skilled span, against skilled-memory theory (P §8).
- F5: the prior-off run is indistinguishable from the headline at sessions 1-100. Prior knowledge would then play no role, against DD's known advantage.
- F6: consonant-list span on the final memory > 12 (no material specificity).

F1-F4 test the mechanism. F5-F6 test whether the prior stream and the material-specificity account do
any work. Any of F1-F4 means the paper reports the digit-span experiment as a limitation of a
time-free learner, not as a fit.

---

## 8. Implementation sketch (for later; not written yet)

`experiments/v3/psych_digit_span.py`, structured like `chess_cs1973_exact.py`:
- one continuous Trellis3 memory per seed (no fork between sessions);
- probes in forked children (`isolated()`), so probe learning dies with the child;
- ported unchanged: `climb_path`, the `Mem` leaf → records index (Trellis3 `e.leaf` / `e.cleaf`; instance = `_type_key(e)` string), and the identical / concept / omission decode;
- new and named: the RS frame (§3), the auditory buffer B = 4, the size floor 3, the K-FIFO STM;
- output: one JSON per seed with a line per 5-session block, flushed as it goes, plus a human-vs-model table drawn from §4.
