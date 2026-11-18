# Round 204 design: where the fair-count gap comes from, and what to try

Design only: nothing here was run on a compute node. Every number below comes from reading existing dumps
(`r200-fair-n{1000,4000}.*`, the judged `story_s*_arm*.jsonl` + `.scores.json` files, `r198-fair-*`, `r202*-classes`,
`r192_diagnosis.md`, `r201_distance.md`), using tsv arithmetic and awk/jq counts. The one thing not recomputable without the
training stream is the padded unstored-trigram flag per sentence, so per-sentence novelty splits are bounded, not measured.

Settings used for the per-sentence counts (each family at, or next to, its best fair cell):

| n | TRELLIS | LSTM | GPT | real |
|---|---|---|---|---|
| 1000 | w6 t1.6 (`r200-n1000-w6` arm1), 4 seeds | t0.8 (`r200-n1000-refs` arm3) | t0.8 (arm0) | `.real` of the 4 refs files, 1200 sentences |
| 4000 | w6 t1.3 (`r200-n4000-w6` arm0), 2 seeds | t0.9 (arm4; best is t1.0, not in this dir) | t0.9 (arm1) | 600 |

"Boundary-atypical" below means the first word starts fewer than 2, or the last word ends fewer than 2, of the 1200
held-out real sentences. Calibration: real sentences of seeds 19/23 checked against a lexicon built from seeds 13/17 come out
91.2% typical, and the neural models come out 90-92%. The proxy is therefore at the real-data rate for the neural models.

## Verdict up front

1. The gap is not driven by failed draws (0 at every best cell) or by short sentences (TRELLIS makes fewer of them than real
   data does). At n=1000 the gap is distinctness plus grammaticality; at n=4000 it is mostly grammaticality of novel
   sentences.
2. One TRELLIS-specific defect explains most of the n=1000 gap: **boundary leakage**. 37% of TRELLIS's distinct outputs at
   w6 t1.6 start or end where real sentences do not, against 8% for LSTM. That accounts for 73% of the gram|distinct gap to
   LSTM and for 48% of TRELLIS's repeats. The leak grows with window width and temperature, the two knobs TRELLIS needs for
   novelty. The cause is the per-attribute Dirichlet back-off giving the boundary markers half their parent's probability
   at single-occurrence leaves.
3. Rank 1 for round 204 is to read the boundary markers exactly (no back-off for `l1=<s>` / `r1=</s>`), with zero new
   constants. Predicted fair count at n=1000 is .32-.36 (point .34), against LSTM .344. At n=4000 the prediction is .32-.35,
   against LSTM .358 and GPT .362.
4. Beating LSTM at n=1000 is plausible only within seed noise, and beating it at n=4000 is not plausible for this
   architecture (see section 4).

## 1. Quantification

### 1a. Sentence lengths (share of all returned rows; tokens)

| | 3 | 4 | 5 | 6 | 7 | 8 | mean | share <= 4 |
|---|---|---|---|---|---|---|---|---|
| real (n=1000 refs) | .102 | .224 | .217 | .169 | .171 | .117 | 5.43 | .326 |
| TRELLIS w6 t1.6, n=1000 | .065 | .116 | .183 | .269 | **.283** | .084 | 5.84 | .181 |
| LSTM t0.8, n=1000 | .058 | .192 | .250 | .245 | .164 | .090 | 5.54 | .250 |
| GPT t0.8, n=1000 | .054 | .208 | .247 | .188 | .175 | .128 | 5.61 | .262 |
| real (n=4000 refs) | .093 | .240 | .218 | .175 | .168 | .105 | 5.40 | .333 |
| TRELLIS w6 t1.3, n=4000 | .078 | .127 | .166 | .208 | **.304** | .117 | 5.88 | .205 |
| LSTM t0.9, n=4000 | .044 | .155 | .280 | .246 | .174 | .101 | 5.65 | .199 |
| GPT t0.9, n=4000 | .067 | .167 | .241 | .182 | .194 | .148 | 5.71 | .234 |

TRELLIS is skewed long, with a 7-token mode at 28-30% against 17% in real data, and it under-produces 3-4-token
sentences (18-21% against 33%). The neural models sit between TRELLIS and real data. Lengths outside 3..8 and stored copies
are redrawn inside every generator (TRELLIS up to 50 tries, neural from a filtered pool), so the hidden rejection is
symmetric and does not enter the fair count.

### 1b. Are the short-sentence failures premature EOS?

Partly, but they hardly matter for the fair count.

- **Grammaticality by length** (distinct sentences, n=1000): for TRELLIS, length 3 is .364, length 4 .511 and lengths 5-8
  .500-.562. For LSTM, length 3 is .599, length 4 .609 and lengths 5-8 .56-.60.
- **What short sentences are worth:** short sentences are 16.6% of TRELLIS's distinct output and carry almost no unstored
  trigrams. Lifting all of them to LSTM's short-sentence grammaticality would add about .024 to gram|distinct and at most
  about .005 to the fair count.
- **Ends on a function word** (the/a/to/and/of/was/very/so/...): TRELLIS 14.9% (n=1000) and 11.9% (n=4000); LSTM 5.1-6.3%;
  GPT 5.1-5.5%; real 3.7%. That is 21.5% of TRELLIS's ungrammatical distinct sentences, against 7-8% for the neural models.
  Premature EOS is a real TRELLIS excess of about 10 points, spread over all lengths.
- **Among ungrammatical short (<= 4) TRELLIS sentences**, only 14% end on a function word. A hand classification of 59 such
  sentences (seed 13) gave:

  | class | count | examples |
  |---|---|---|
  | truncation | 11 | *they saw the*, *fish were very*, *do you want to* |
  | headless start | 11 | *said be careful*, *can have to play*, *smiled and help* (2 overlap with truncation) |
  | dialogue/vocative fragments | 7 | *see you mom*, *look ben no* |
  | local junction / agreement / repeat | 19 | *they was happy*, *can i help me*, *ben and ben smiled* |
  | semantic or borderline | 13 | |

  Premature EOS is about a fifth of the short failures, and boundaries as a whole (start plus end) about two fifths.
- **r202 classes** (w5 t1.6, s13): premature end .102 with gram .360. The r202 closure read raised premature ends to
  .18-.27, which is why it shortened sentences (len-3 share .116-.175) and lost fair count.

### 1c. Failed draws

- At every best cell, returned = attempted: TRELLIS w5/w6/w7 at T >= 1.0, and every neural cell. Failed draws contribute 0
  to the gap.
- Failures exist only at low temperature: w5 t0.5 returns 1207.5/1520 (20.6% fail), w4 t0.5 8.9%, w5 t0.7 0.6%, and
  "both" w5 t0.5 37%.
- **Why:** the window read copies a stored sentence, the `new` filter rejects it, and 50 redraws all copy (or leave 3..8).
  Those cells are far from the fair optimum.

### 1d. Decomposition of the fair count

The fair count factors exactly: fair = (distinct new / attempted) x (tri | distinct) x (gram | tri, distinct), times
returned/attempted = 1. Since the generator never returns a stored sentence, distinct new / attempted = 1 - repeats.

**Best cell against best cell** (log ratios, which sum to the total):

| | total ln(fair_T / fair_X) | distinct (repeats) | tri (novelty) | gram \| tri | failed draws |
|---|---|---|---|---|---|
| n=1000, TRELLIS .316 vs LSTM .344 (t0.8) | -.085 | **-.079** (.800 vs .866) | +.093 (.855 vs .779) | **-.097** (.462 vs .509) | 0 |
| n=1000, vs GPT .324 (t0.8) | -.025 | **-.055** (.800 vs .845) | +.018 (.855 vs .840) | +.009 (.462 vs .458) | 0 |
| n=4000, TRELLIS .317 vs LSTM .358 (t1.0) | -.122 | -.063 (.913 vs .972) | +.059 (.787 vs .742) | **-.122** (.441 vs .498) | 0 |
| n=4000, vs GPT .362 (t0.9) | -.133 | -.011 (.913 vs .923) | +.134 (.787 vs .688) | **-.257** (.441 vs .570) | 0 |

TRELLIS always buys more novelty. It loses on repeats (n=1000) and on the grammaticality of novel sentences (both n, and
dominant at n=4000).

**Matched novelty** (neural cells interpolated along temperature to TRELLIS's tri|distinct):

| | tri | TRELLIS dist / gram\|tri / fair | LSTM | GPT |
|---|---|---|---|---|
| n=1000 | .855 | .800 / .462 / .316 | .946 / .377 / .305 | .870 / .431 / .321 |
| n=1000 | .78 | .774 / .495 / .301 (w5 t1.3) | .866 / .509 / .344 | .685 / .527 / .282 |
| n=4000 | .787 | .913 / .441 / .317 | .980 / .443 / .342 | .964 / .453 / .344 |
| n=4000 | .69 | no TRELLIS cell (grid gap between w4 t1.0, tri .46, and w6 t1.3, tri .79) | .945 / .555 / .355 | .923 / .570 / .362 |

At matched novelty, TRELLIS's novel sentences are about as grammatical as the neural ones: better at n=1000, tied at
n=4000 / tri .79. Its loss at those points is repeats. The neural models win by peaking at moderate novelty (tri .68-.78).
There their novel trigrams are .51-.57 grammatical, while TRELLIS's gram|tri never exceeds .50-.60 at any window or
temperature at n=4000 (w3 t0.5 .597, w4 .50-.56, w6 .441). TRELLIS's novelty is not confidence-graded: in r192 the filler's
step probability had AUC .50 for grammaticality, so lowering the temperature cannot select its good novel junctions.

**Boundary split** (distinct sentences; "ok" = boundary-typical):

| | ok share | gram \| ok | gram \| atypical | repeat rate in ok rows | atypical share of repeat excess | distinct gram/draw from atypical |
|---|---|---|---|---|---|---|
| TRELLIS n=1000 | .630 | .589 | .389 | .171 | **.48** | **.115 of .412 (28%)** |
| LSTM n=1000 | .919 | .623 | .265 | .143 | .01 | .019 of .515 |
| GPT n=1000 | .925 | .551 | ~.22 | .165 | .006 | .014 of .444 |
| TRELLIS n=4000 | .729 | .570 | .393 | .064 | **.48** | .097 of .476 (20%) |
| LSTM t0.9 n=4000 | .902 | .685 | ~.41 | .060 | .01 | .038 of .622 |
| GPT t0.9 n=4000 | .912 | .681 | ~.48 | .080 | .05 | .039 of .612 |

- **n=1000:** with LSTM's boundary-typical share and its own conditional rates, TRELLIS gram|distinct would be .573
  instead of .515. That closes .058 of the .079 gap to LSTM (73%).
- **n=4000:** boundary closes .031 of .136 (23%). The rest is interior: ok-sentence gram .570 against .685, at higher
  novelty.
- **Repeats:** among boundary-typical rows, TRELLIS repeats at the neural rate (.171 vs .143; .064 vs .060). Its extra
  repeats are boundary-cut copies of stored sentences. The top seed-13 repeats are *little girl loved the cat very much*
  (x7), *had a lot of fun* (x7), *animals did not know what to* (x6), *knew he had found a special* (x5) and *max lived with
  a little girl named* (x4).
- **Caveat that cuts against TRELLIS:** 20-28% of its distinct grammatical output is headless or truncated fragments that
  the judge accepts (*had a lot of fun*, *wanted to be friends with the bear*). These count as novel only through their
  padded `<s>` or `</s>` trigram. For the neural models the figure is 3-6%.

### 1e. The leak grows with window width and temperature (n=1000, 4 seeds, distinct sentences)

| arm | dist/draw | gram\|d | atypical start | atypical end | ok share | gram\|ok |
|---|---|---|---|---|---|---|
| w3 t1.0 | .720 | .747 | .018 | .043 | .940 | .756 |
| w4 t1.0 | .669 | .689 | .017 | .063 | .921 | .709 |
| w5 t0.5 | .292 | .650 | .012 | .204 | .785 | .721 |
| w5 t1.0 | .697 | .614 | .061 | .174 | .771 | .671 |
| w5 t1.6 | .811 | .545 | .118 | .164 | .736 | .602 |
| w5 t2.0 | .843 | .494 | .147 | .162 | .712 | .551 |
| w6 t1.3 | .760 | .551 | .195 | .175 | .667 | .617 |
| w6 t1.6 | .800 | .515 | .247 | .170 | .630 | .589 |
| w6 t2.0 | .828 | .475 | .282 | .171 | .602 | .551 |
| LSTM t0.9 | .929 | .507 | .025 | .077 | .900 | .542 |
| GPT t0.9 | .911 | .446 | .020 | .071 | .911 | .473 |

- **End leak:** set by the window. It is .04-.06 at w3/w4 and .16-.20 at w5/w6 at every temperature, including t0.5.
- **Start leak:** set by window times temperature. It is .06 at w5 t1.0 and .28 at w6 t2.0.

**Mechanism (trellis3.py `_gap_fills`, line 2827):** a filler's mass is the sum over its occurrences of the product over
query attributes of P(q_a | leaf). Each factor is backed off along the path, P = (n_C(v) + P_parent(v)) / (n_C + 1).

- With wide windows, almost every context leaf holds a single occurrence. A non-final occurrence therefore still gets
  P(r1=`</s>`) of about half its parent's end rate, and the parent groups contexts by their shared left window.
- Every step of a copied stored sentence carries a few percent of end mass, so copies get truncated (*knew he had found a
  special*).
- At the start, position-2 occurrences match the all-`<s>` query on l2..lW and mismatch only on l1. The pads make the
  start positions siblings in the tree, so P(l1=`<s>` | parent) is large. Mid-sentence words (*had*, *wanted*, *little*,
  *said*) then start sentences, and temperature amplifies this.
- In r192 this was described for ends ("a single-occurrence leaf backs off half of its parent's end probability"). The
  dumps show it is the largest single TRELLIS-specific failure, and that it scales with the two settings the fair count
  pushes TRELLIS towards.

**r202 parse3 check (4 seeds, `r202-n1000-w5`, generated and judged but not yet fair-scored).** Compared with plain w5 at
the same temperature:

| T | parse3 dist / gram\|d / distinct-gram per draw | plain w5 |
|---|---|---|
| 1.6 | .808 / .530 / .429 | .811 / .545 / .441 |
| 2.0 | .839 / .500 / .420 | .843 / .494 / .417 |

Parse3 is null. The seed-13 ".282 vs .314" in r202 compared parse3 t1.3 with plain t1.6. At equal temperature, seed 13
gives .282 vs .268 at t1.3 (+.014, about 1 SE). The tri column needs `story_frontier_fair.py` on a compute node.

## 2. Candidate mechanisms, ranked by expected gain on the fair count

### M1 (rank 1). The boundary is observed, not inferred

**Principle.** `<s>` and `</s>` are not words. They have no distributional class, so the concept hierarchy has nothing to
generalise about them. The decision "start here" or "end here" is read from the occurrences that made it, with the same
context weighting as everything else. Interior attributes keep the back-off, and novelty in the interior is untouched.
This is the soft, mass-level version of the boundary half of r192's "both" rule, without its pivot gate, and it has no new
constant.

**Exact change.**

- `src/trellis2/trellis3.py`, `_gap_fills` (line 2827):
  - Add an opt-in learner attribute `edge_exact` (default False, beside `edge_once` / `pool`), and include it in the cache
    key (line 2879).
  - In the occurrence loop (lines 2902-2913): for every query attribute that is the gap-adjacent boundary slot, meaning
    `q["l1"] == BOS` or `q["r1"] == EOS`, replace `probs(e.leaf)[a]` with the occurrence's own value. Skip the occurrence
    when `e.ctx.get(a) != q[a]`; otherwise the factor is 1. The far pads (l2..lW at the start) keep the back-off.
  - The end read `L._gap_fills({**q, "r1": EOS})` in `t3_story.py` `_fills` (line 1033) then returns, per filler, the part
    of its context-weighted mass from occurrences that really ended a sentence. A novel end (l1, w, `</s>`) stays possible
    whenever w ended after a similar left context, but a word that never ends (*the*, *to*, *very*) gets end mass 0.
  - The start step (all-`<s>` query) reads only position-1 occurrences.
- `experiments/v3/diag/t3_story.py`, causal branch (next to `pool_` at line 1012): add an arm key `"edge": "exact"` that sets
  `L.edge_exact = True` for the arm and resets it afterwards.
- Untouched: learning, storage, the tree, `pivot_support`, the pair gate, and t3_story_r202.py.

**Predicted effect** (w6 t1.6, n=1000; the best T and W may move up, since the leak that grows with them is gone):

| factor | now | predicted | reasoning |
|---|---|---|---|
| atypical boundary share | .370 | <= .10 | neural .08 |
| gram \| distinct | .515 | .56-.59 | gram\|ok .589; some freed mass goes to longer, more novel sentences |
| tri \| distinct | .855 | .76-.82 | boundary-only novelty is lost |
| distinct / draw | .800 | .78-.84 | boundary-cut repeats (9.6% of rows) go; copy-attraction repeats come in. In r192a pairs+boundaries at T1.0, distinct fell .697 -> .656 |
| gram \| tri | .462 | .50-.54 | |
| failed draws | 0 | 0 at T >= 1.3 | may appear at T <= .7 |
| fair, n=1000 | .316 | .32-.36 (point .34) | |
| fair, n=4000 | .317 | .32-.35 | |

**Risk, stated plainly.** 28% of TRELLIS's current distinct grammatical output per draw (.115) comes from judge-accepted
boundary fragments. If the freed mass does not turn into well-formed novel sentences, the fair count falls. The closest
existing evidence is r192a "pairs" (boundary gate plus strict pair gate, w5 T1.0, 4 seeds). There gram|d rose
.614 -> .681, but distinct fell .697 -> .656, tri fell .738 -> .637 and the fair count fell about .021. That arm is
confounded: the strict pair gate also removes class-A novelty, and it ran at a temperature where the start leak was only
.06. M1 targets the w6 / T >= 1.3 regime, where the leak is .2-.28.

**Falsification probe (n=1000, seed 13).**

- **(a) Pre-test, no generation, about minutes on a compute node.** Rebuild the seed-13 learner. Replay the 1600 r200 w6 t1.6
  seed-13 sentences step by step, and record the log-probability of each sentence's start step and its end step under the
  default read and under `edge_exact`.
  - Pass if boundary-atypical sentences lose at least 1.5 nats (median) at those steps while boundary-typical sentences
    lose at most 0.3 nats.
  - Fail means the leak is not where the trace says it is; stop.
- **(b) Generation run.** Seed 13, w6 and w7, T in {1.3, 1.6, 2.0}, judged, then `story_frontier_fair.py`.
  - **Pass:** best seed-13 fair >= .335 (the best seed-13 TRELLIS cell now is w5 t1.6 at .314; w6 t1.6 is .302; one binomial
    SE is about .011), and atypical share <= .10.
  - **Kill:** best fair <= .314, or atypical share still > .15.
  - **If it passes:** run 4 seeds at n=1000 and 2 seeds at n=4000 (also fill the n=4000 grid: w5, w6 at T 1.0 and 1.6).

### M2 (rank 2). The two-sided read as a weight, not a gate

**Principle.** r192's pivot ratio, log P(pivot | left, r1=w) / P(pivot | left), is the only per-step statistic with
information about grammaticality: AUC .67 overall and .55 inside class C. As a hard gate ("both") it raised gram|tri to
.66-.73, but it cut distinct to .543 and tri to .510, so the fair count was .182 at T1.0 (`r198-fair-n1000`), and it was
never run hotter. As a weight it removes mass from new trigrams in proportion to how much they take from the pivot,
instead of deleting them, so breadth survives. On top of M1, which already replaces the boundary half of "both", this is
the remaining interior half.

**Exact change.** In `t3_story.py`, `both_` branch (lines 1094-1124):

- For a filler whose trigram is new: instead of `continue` when `go[(l1, w)] < 0`, multiply its weight by
  `exp(min(0, go[(l1, w)]))`. That is, it keeps the share of the pivot's probability it preserves; it is never boosted, and
  stored trigrams keep weight 1.
- The boundary branch of `both_` is dropped in favour of M1.
- Arm value `"both": "soft"`. No constant beyond the existing temperature.

**Predicted effect** (on top of M1, same T):

| factor | predicted change |
|---|---|
| gram\|tri | +.03 to +.06 (the gate gave +.13 at a large novelty cost) |
| tri | -.02 to -.05 |
| distinct | -.01 to -.03 |
| fair | 0 to +.02 |

The prior is weak. r202's open-chart reweighting is also a soft multiplicative read, and it was null at matched
temperature (1d).

**Falsification probe (n=1000, seed 13).** Run w6 + M1 at T 1.6 and 2.0, with and without `"soft"`. Kill if the best-T fair
gain is below +.01, or if gram|tri rises by less than .02 at fixed T.

### M3 (rank 3). The window records identity

**Principle.** A repeated word is the strongest simple predictor of a TRELLIS failure (AUC .56, r201).

- TRELLIS repeats a word in 9.1% (n=1000) and 11.6% (n=4000) of its sentences, against 4.9-5.4% in real data and 7.3-8.5% for
  the neural models.
- Those sentences are judged grammatical .28-.29 of the time (*ben and ben smiled*, *the cat and the cat were friends*, *can
  i help me*).
- The window stores which word stood at l_k, but not that the filler is that same word. A product of per-slot marginals
  cannot see "ben and ben".
- Recording the identity relation between the filler and each window slot is the Cobweb counterpart of a copy relation, it
  is read from raw tokens, and it needs no constant.

**Exact change.**

- In `trellis3.py` `_surface` (line 473): add `same_k` (k = 1..W), 1 when the occurrence's own token equals l_k, to the
  stored context of word occurrences.
- In `_gap_fills`: the query's `same_k` depends on the candidate. Per occurrence, the factor is
  P(same_k = [occurrence's filler == q[l_k]] | leaf), with the same back-off.
- This changes the stored descriptions, so the learner must be relearned. The tree changes with it (category utility sees
  the new attributes), so the parse results would need a re-check.

**Predicted effect.** The repeated-word share falls to about 6%, gram|distinct rises .010-.015, and the fair count rises
about .005-.01. It is the cheapest in principle but the most invasive, because it needs relearning.

**Falsification probe (count only, seed 13).** On the r200 w6 t1.6 seed-13 dumps, check what fraction of the ungrammatical
repeated-word sentences repeat within the window (distance <= 6). Kill if it is below 80%, since the attribute could not
see the rest. If it passes, relearn seed 13 at n=1000 with `same_k` and run w6 + M1 at T 1.6, with the same pass/kill
thresholds as M1(b).

Not proposed, because they repeat closed lines:

- Gibbs repair or chunk-unit generation (copies memory).
- KN / hier / geo read calibration (log-loss only).
- Pair-licence substitution (negative).
- The role check (AUC .455).
- The r202 closure read, which favoured premature ends; M1 removes the same end leak from the read itself.

## 3. Bookkeeping to do alongside round 204 (no new method)

- Run `story_frontier_fair.py` on `r202-n1000-w5` (4 seeds, T 1.0-2.0) to close r202 with the tri column.
- Fill the n=4000 TRELLIS grid (w5; w6 at T 1.0 and 1.6). The current n=4000 TRELLIS optimum comes from two cells, so
  "flat across n" is partly a grid artefact until those run. The n=1000 w5-w7 spread suggests at most about +.01.
- Report a boundary-strict variant of the fair count (count only boundary-typical sentences) beside the current one. The
  judge accepts TRELLIS's headless fragments, and a reviewer will find them in the repeats.

## 4. Can TRELLIS beat LSTM?

**At n <= 1000:** plausible, but only within seed noise. M1's point estimate (.34) equals LSTM's .344, and LSTM's seed range
is .308-.395. Even a clean M1 pass would most likely be a tie across 4 seeds rather than a win.

**At n = 4000:** not plausible for this architecture.

- The neural models still improve with data. LSTM goes .244 / .304 / .344 / .358 and GPT .213 / .283 / .324 / .362 at
  n = 250 / 500 / 1000 / 4000.
- Their gain sits in the grammaticality of moderately novel sentences: gram|tri .55-.57 at tri .68-.69, and .75-.82 at low T.
- TRELLIS's novel-trigram grammaticality is capped at about .44-.60 at every window and temperature at n=4000. Its step
  probability for a novel junction carries no information about grammaticality (r192: AUC .50; the pivot read .55 inside
  class C), so no sampling rule can select its good junctions.
- That limit belongs to what the context tree's back-off treats as similar, which is the representation, not the
  generator. The rounds since r156 that changed only the read or the generator have not moved it.

**Strongest honest claim, with evidence.**

- **Main claim:** with only incremental Cobweb trees (no gradients, no fitted output layer), TRELLIS matches a matched-size
  GPT on fair productivity from n = 250 to 1000:
  - n=250: .219 vs .213 (paired wins 3 of 4 seeds)
  - n=500: .280 vs .283 (1 of 4)
  - n=1000: .316 vs .324 (1 of 4; held-out seeds -.002)
- **Against LSTM:** it trails by .02-.03 at n = 250-1000 and by .04 at n=4000.
- **At matched novelty:** its novel sentences are as grammatical as the neural ones, and the deficit at that point is repeats:

  | n | tri | TRELLIS gram\|tri | LSTM | GPT |
  |---|---|---|---|---|
  | 1000 | .855 | .462 | .377 | .431 |
  | 4000 | .787 | .441 | .443 | .453 |

- **Caveat that must be stated beside any claim:** 20-28% of TRELLIS's distinct grammatical output is boundary fragments
  of stored sentences (3-6% for the neural models). A boundary-strict count would widen the gap until M1, or something
  like it, removes them.
- **What the evidence supports:** TRELLIS reaches neural-level productivity at small n without gradient training, and its
  remaining loss can be traced to a named read (boundary back-off) and a named limit (uninformative novel-junction
  probability). It does not support a claim that TRELLIS out-generalises neural LMs as data grows.
