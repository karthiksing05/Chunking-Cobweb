# Round 206: why TRELLIS's novel sentences do not get more grammatical from n=1000 to n=4000

This is a diagnosis and design only: no generation was run. Every number comes from the existing judged dumps (seeds 13 and 17)
and the rebuilt training streams. Four count-only jobs ran on overcap compute nodes (scripts and raw tables in `logs/r206/`, which git ignores:
`r206_diag.py`, `r206_junc.py`, `r206_pairA.py`, `r206_real.py`). No learner was rebuilt, and no context tree is saved from the
n=4000 runs, so nothing below reads a tree directly.

Dumps used: TRELLIS edge (`"edge":"exact"`) w5 t1.3/1.6/2.0 (`r204-n{1000,4000}-w5`), edge w7 (`r204-n1000-w7`; n=4000 has seed 17 only),
plain w5/w6 (`r200-*`), LSTM / GPT t0.3-1.2 (`r200-*-refs`, `r184-*-refs`, `r190-n1000-refs`, `r191-n4000-refs`). The unit is
distinct new sentences, as in `r204_fair.py`. "Novel" means the sentence has a padded trigram the stream never held (tri). A
"junction" is a novel trigram (a, b, c). The r192 classes are: A, the sentence holds an unstored pair; B, every pair is stored but a novel
trigram touches `<s>` or `</s>`; C, every pair is stored and the novel trigram is interior.

## Verdict up front

- **The fair count is flat because TRELLIS's gains in distinctness are cancelled by losses in grammaticality.** At the same
  setting (edge w5 t2.0), from n=1000 to n=4000:
  - distinct/draw rises .819 -> .952;
  - tri|d stays put (.836 -> .816);
  - gram|tri falls .464 -> .410.

  The neural models gain the same distinctness, and their gram|tri holds or rises (LSTM best .503 -> .498, GPT .487 -> .570).
  TRELLIS's loss is not confined to edge_exact: plain w6 t1.3 goes .491 -> .441.
- **Most of the loss is inside class C (stored pairs, new trigram).** Class C is 67-74% of TRELLIS's novel sentences. Its
  grammatical rate falls .54-.57 -> .48-.54, while LSTM and GPT class C holds at .65-.74. Class A (type-licensed new pairs)
  collapses from .22-.29 to .11-.17, but it is only 9-13% of novel sentences.
- **Of the candidate explanations:**
  - (d) TRELLIS takes on more novelty: refuted as the cause. At matched novelty TRELLIS falls 7-9 points behind at
    tri .70-.75, and ties only at tri .82.
  - (b) rarer contexts: refuted as stated, and reversed for histories.
  - (c) temperature: worth at most about +.01.
  - (a) coarser or more mixed trees: supported only through class A and the pronoun pivots. Trees are unsaved, so it is not
    tested directly.
- **What the data add (e):**
  - TRELLIS's novelty does not respond to evidence. Real held-out text and both neural models innovate after a history seen
    k times at a rate that depends on k alone, the same at n=1000 and n=4000. TRELLIS innovates too little after rare histories,
    and too much after frequent ones, increasingly so as n grows.
  - Its type-licence gate admits new pairs almost only after the most frequent words, which is where absence is best evidenced.
  - About 40% of the class-C drop is a shift toward pronoun pivots (*thank you want*, *can you are*, *give it is*). The other
    60% is a within-pivot decline that no count statistic tested here explains.
- **Mechanisms.** Rank 1: one zero-constant rule, the novel continuation gets only the history's missing mass N1/(c+N1). Rank 2:
  a soft role check on the pivot. Expected combined gain is fair +.015-.03 at n=4000.
- **Parity at n=4000 (.358-.362) is not reachable on present evidence.** The realistic ceiling is about .335-.345 (section 4).

## 1. Fair count and its factors, n=1000 vs n=4000 (seeds 13 and 17, mean)

| family, setting | n | fair | dist/draw | tri\|d | gram\|tri |
|---|---|---|---|---|---|
| TRELLIS edge w5 t1.3 | 1000 | .283 | .745 | .755 | .502 |
| | 4000 | .297 | .907 | .698 | .469 |
| TRELLIS edge w5 t1.6 | 1000 | .317 | .783 | .794 | .511 |
| | 4000 | .294 | .933 | .749 | .421 |
| TRELLIS edge w5 t2.0 | 1000 | .318 | .819 | .836 | .464 |
| | 4000 | .319 | .952 | .816 | .410 |
| TRELLIS edge w7 (best n=1000) t2.0 | 1000 | .337 | .829 | .844 | .482 |
| TRELLIS edge w7 t2.0 (seed 17 only) | 4000 | - | - | - | .399 |
| LSTM best | 1000 (t0.8) | .337 | .869 | .771 | .503 |
| | 4000 (t1.0) | .358 | .972 | .742 | .498 |
| GPT best | 1000 (t0.8) | .336 | .836 | .827 | .487 |
| | 4000 (t0.9) | .362 | .923 | .688 | .570 |

Log-ratio from n=1000 to n=4000 at each family's best cell (TRELLIS: w5 t2.0 at both n):

| family | distinct | tri | gram\|tri |
|---|---|---|---|
| TRELLIS | +.150 | -.024 | **-.124** |
| LSTM | +.112 | -.038 | -.010 |
| GPT | +.099 | -.184 | **+.157** |

Every family gains distinctness, because repeats vanish as data grows. Only TRELLIS loses grammaticality of its novel sentences.
The decline holds at every sentence length: at w5 t2.0, gram per length 3..8 goes .37/.40/.48/.51/.48/.44 at n=1000 ->
.40/.32/.44/.42/.45/.41 at n=4000. LSTM and GPT rise at every length.

## 2. The candidate explanations

### (d) More novelty taken on: refuted as the cause

gram|tri at matched tri|d (neural values linearly interpolated along temperature; seeds 13 and 17):

| n | tri | TRELLIS edge w5 | LSTM | GPT |
|---|---|---|---|---|
| 1000 | .755 | .502 | .521 | .570 |
| 1000 | .794 | .511 | .466 | .515 |
| 1000 | .836 | .464 | .398 | .471 |
| 4000 | .698 | **.469** | .536 | .559 |
| 4000 | .749 | **.421** | .489 | .500 |
| 4000 | .816 | .410 | .407 | .417 |

- **n=1000:** TRELLIS is level with the neural models at matched novelty.
- **n=4000:** it trails by .07-.09 at tri .70-.75 and ties only at tri .82. Its curve is flat (.47 -> .41 across tri .70 -> .82),
  while the neural curves are steep.
- **Consequence:** the neural optimum at n=4000 sits at moderate novelty (tri .68-.74, dist .92-.97, gram .50-.57). Lowering
  TRELLIS's temperature to reach that tri does not buy grammaticality. Even with the novelty matched, TRELLIS's novel sentences
  get worse with n.
- **Caveat:** at n=4000, tri means "outside a 2.6x larger trigram set" (stored trigram types 2744-2835 -> 7242-7302). That
  caveat applies to all families equally.

### (a) Coarser or more mixed concept trees: indirect support only

- **Class A collapses.** It is the class where the type licence (concept member sets) admits an unstored pair. Its gram rate
  goes .22-.29 -> .11-.17 (w5 t1.3/1.6/2.0: .27/.29/.22 -> .15/.11/.17; w7 seed 17: .13/.11/.16). Its share stays at
  .09-.13. LSTM/GPT class A rises (.26-.33 -> .35-.43).
- **Who the licence admits at n=4000 is the issue** (`r206_pairA.tsv`, sentences with exactly one unstored pair):
  - 91% of TRELLIS's admitted pairs follow a left word with 200 or more stored tokens, against 33% at n=1000, and are judged
    grammatical 15% of the time.
  - Top failures: *and and* (41), *a and* (39), *the and* (32), *they was* (29), *to and* (25), *you was* (22), *and to* (22),
    *she were* (13).
  - The neural models' unstored pairs are spread over left-word frequencies, at .42-.49 gram in every bin.
- **This is consistent with concepts gaining members (types) as n grows.** The member-share licence then passes pairs after
  function words that a stored history of hundreds of tokens never followed with that word.
- **Agreement-error bigrams do not grow** (.008-.019 of novel sentences at both n). The trees were not saved, so leaf
  occupancy, depth and member mixing at n=4000 are not measured.
- **Class A accounts for about .018 of the .09 gram|tri drop at t1.6.**

### (b) Novel junctions from rarer contexts: refuted as stated; reversed for histories

- **Pair rarity:**
  - The share of class C junctions that pass through a once-stored pair (b, c) *falls*, from .40-.43 to .25-.32.
  - Within every absolute pair-count bin, TRELLIS gram drops: count 4-9 .63-.66 -> .50-.55; 10-29 .65-.68 -> .55-.59;
    30+ .83-.87 -> .59-.62.
  - The neural models hold or rise in every bin (LSTM 10-29 .69-.74 -> .68-.76; 30+ .80-.92 -> .75-.80).
- **History rarity** (count of the two-word history (a, b)): at n=4000 TRELLIS's novel junctions come from *frequent* histories.
  - Class C sentences whose junction history was seen 30-99 times go from .02 to .16-.17 of the total, and are judged
    grammatical .34-.46.
  - The neural models make the same shift (.04 -> .14-.17) but stay at .58-.64 gram.
  - Re-weighting TRELLIS's n=4000 class C rates to the n=1000 history mix explains .028 of the .08 class-C drop at t1.6. The
    rest is within history bins.
- **Absence statistics carry no signal for a single junction** (class C sentences with exactly one novel trigram):
  - The chaining expectation E = c(a,b) c(b,c) / c(b) has AUC .47-.48 for TRELLIS at both n, and .43-.52 for the neural models.
  - The history count alone has AUC .52 (n=1000) and .53 (n=4000; seed 13 .56, seed 17 .50).

### (c) Temperature: at most about +.01

- **What the n=4000 grid shows (w5):**
  - fair .297/.294/.319 at t1.3/1.6/2.0;
  - gram|tri .469/.421/.410;
  - tri .698/.749/.816;
  - dist .907/.933/.952.
- **The n=1000 edge arm beyond t2.0 (w7):** t2.0 -> 2.4 gave dist +.013, tri +.035, gram|tri -.041, fair -.011.
- **Projection to n=4000 t2.4:**
  - with the n=1000 slope (dist ~.965, tri ~.86, gram|tri ~.37-.40): fair .31-.33;
  - parity would need gram|tri to stay at .41 while tri rises to .90 and dist to .98 (fair .362). No TRELLIS arm has ever held
    gram|tri flat over +.08 tri.
- **The neural optima moved up in temperature with n** (LSTM t0.8 -> t1.0, GPT t0.8 -> t0.9), because their distributions
  sharpened. TRELLIS's did not sharpen (section 3), so it has no such headroom.
- A t2.4/t2.8 column at n=4000 is worth one seed-13 run as bookkeeping. It cannot close .04.

## 3. (e) What else the data show

### 3a. TRELLIS's novelty does not respond to evidence; real text's and the neural models' does

Per-step trigram novelty by the training count of the two-word history (all generated steps after the first word;
`r206_steps.tsv`). "Real" = held-out real sentences of the same seeds (`r200-n*-refs`), with the same stream:

| history count | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64+ |
|---|---|---|---|---|---|---|---|
| real, n=1000 | .553 | .375 | .299 | .179 | .159 | .089 | .104 |
| real, n=4000 | .578 | .370 | .229 | .208 | .129 | .084 | .045 |
| LSTM t0.9, n=1000 | .581 | .440 | .300 | .215 | .153 | .077 | .041 |
| LSTM t0.9, n=4000 | .602 | .415 | .312 | .215 | .159 | .101 | .048 |
| GPT t0.9, n=1000 | .626 | .481 | .340 | .237 | .219 | .090 | .070 |
| GPT t0.9, n=4000 | .606 | .463 | .351 | .213 | .161 | .093 | .047 |
| TRELLIS edge w5 t1.6, n=1000 | .331 | .266 | .227 | .163 | .169 | .059 | .078 |
| TRELLIS edge w5 t1.6, n=4000 | .436 | .337 | .294 | .246 | .166 | **.133** | **.079** |
| TRELLIS edge w5 t2.0, n=1000 | .374 | .308 | .257 | .192 | .164 | .075 | .066 |
| TRELLIS edge w5 t2.0, n=4000 | .476 | .393 | .317 | .254 | .195 | **.170** | **.096** |
| Witten-Bell d/(d+c), n=4000 | .500 | .396 | .310 | .264 | .213 | .152 | .099 |
| Good-Turing N1/c, n=4000 | 1.00 | .462 | .269 | .207 | .150 | .081 | .040 |

- **Real text and the neural models are scale-free in the absolute count.** After a history seen k times, the novelty rate is
  the same at n=1000 and n=4000 (real .55/.58 at k=1, .089/.084 at 32-63).
- **TRELLIS is not.**
  - At fixed k its novelty rises with n (32-63: .059 -> .133 at t1.6, .075 -> .170 at t2.0).
  - Overall it innovates too little after rare histories (.33-.48 at k=1, against .55-.58 real) and too much after frequent
    ones at n=4000 (.13-.17 at 32-63, against .084 real).
  - The cause is that TRELLIS's escape is a ratio of kernel sums: partial-match mass over exact-match mass. Both grow with n, so
    absolute evidence never turns into confidence.
- **The two standard missing-mass estimators split the range.** Witten-Bell overestimates the missing mass after frequent
  histories by about 2x (.15 vs .08 at 32-63). Good-Turing N1/c matches real text from k >= 8 (.207/.150/.081/.040 against
  .208/.129/.084/.045) but is 1.0 at k=1. N1/(c+N1) (Witten-Bell counted on singletons only) approximates real at every bin.
  From the bin means this gives about .50, .32, .21, .17, .13, .075 and .038; these are approximations, not exact per-history values.

### 3b. Which pivots TRELLIS's class C junctions go through

Single-junction class C sentences. Each entry is the pivot b, its share of junctions, and the gram rate.

| family | n | pivots (share gram) |
|---|---|---|
| TRELLIS | 1000 | and .185 .71; was .160 .78; to .070 .54; the .044 .50; is .043 .50; **you .031 .13**; it .023 .39 |
| TRELLIS | 4000 | and .145 .66; was .096 .71; **you .055 .17**; the .048 .45; to .045 .53; is .036 .43; **it .033 .31**; **i .019 .24** |
| LSTM | 4000 | and .146 .61; was .104 .89; said .053 .67; were .047 .93; the .042 .68; to .033 .50; you .021 .38 |

- **Typical TRELLIS failures at n=4000:** *thank you want* (history seen 75-92 times), *would you want*, *will you are*,
  *give it is*, *yes it back*, *no i see*.
- **What these show:** a pronoun's object use and its subject use are chained through the pivot. TRELLIS's read forgives the
  l2 mismatch whenever (b, c) is frequent.
- **Standardising by pivot:**
  - Holding the n=1000 pivot mix fixed, TRELLIS's class C gram goes .649 -> .608, against the raw .648 -> .583. Pivot mix
    therefore explains about 40% of the drop.
  - The other 60% is a decline within pivots.
  - LSTM rises within pivots (.713 -> .724) and so does GPT (.719 -> .746).
- **Pivot fan-out** (distinct stored successors):
  - TRELLIS junctions move to pivots with 30-59 successors (.15-.17 -> .44-.46 of junctions), where gram is .55-.57 against
    .66-.70 at n=1000.
  - The neural models move the same way (.19 -> .33-.41) and stay at .67-.77.
- **No count statistic tested here explains the within-pivot decline.** It is what is left of the r204 diagnosis: the read's
  similarity between contexts does not sharpen with data. Testing that needs a rebuilt n=4000 tree (leaf occupancy, depth,
  per-concept member types, mass-weighted l2 mismatch factor), which was not done.

### 3c. Decomposition of the gram|tri drop (w5 t1.6, .511 -> .421)

| class | share, n=1000 -> 4000 | gram, n=1000 -> 4000 | contribution |
|---|---|---|---|
| A | .101 -> .096 | .29 -> .11 | about -.018 |
| B | .158 -> .192 | .38 -> .33 | about +.005 (more B at lower gram) |
| C | .740 -> .713 | .57 -> .49 | about -.06 |

- Within C, the history mix accounts for about .02 and the pivot mix for about .02; the rest (about .035) is a decline within
  bins.
- **Window:** edge w7 at n=4000 (seed 17) has the same gram|tri as w5 (.456/.415/.399), so widening the window does not help.

## 4. Mechanisms (ranked)

Both mechanisms use only counts that the learner already commits. The occurrences under the context tree carry their l1, l2 and
filler, and `L._pair_counts` already exists. Neither adds a constant or an n-gram table beside memory.

### Mechanism 1 (rank 1): a novel continuation gets only its history's missing mass

**Principle.** How much novelty a history allows is set by how much evidence memory holds for that history. This is the
statistic that real text follows and TRELLIS does not (3a). Which novel words get that mass is still decided by the unchanged
full-window tree read, so l2 and the class-based generalisation stay in the read. The missing mass is N1(h) / (c(h) + N1(h)):

- c(h) = committed occurrences whose context has the history h exactly;
- N1(h) = fillers seen exactly once after h.

It has no constant. The same rule applies at both history lengths that define novelty, so it is one rule, not two:

- **trigram level:** h = (l2, l1); a filler w is novel if (l2, l1, w) was never committed;
- **pair level:** h = l1; a filler w is novel if (l1, w) was never stored. This caps the type-licensed fillers that the
  `_ok` gate admits.

If memory has never seen h (c(h) = 0), the read is unchanged. With edge_exact, boundary queries are untouched.

**Exact code.**

- **New subclass.** `src/trellis2/trellis3_r207.py` defines `Trellis3R207(Trellis3R204)`, with flag `escape_mass`
  (default False) included in the cache key, in the pattern of `trellis3_r204.py`. `_gap_fills` (base `trellis3.py`
  lines 2827-2919, accumulation loop 2903-2915) returns the base masses unchanged. A new method `escape_split(q, mass)`
  re-walks `self.records` once:
  - it collects the word fillers of occurrences with `e.ctx.get("l1") == q["l1"] and e.ctx.get("l2") == q["l2"]`, giving
    c(h), N1(h) and the attested set S;
  - it rescales attested fillers to (1 - e) x m(w) / sum over S of m, and the rest to e x m(w) / sum outside S of m, with
    e = N1/(c+N1).
- **Generator hook.** In `experiments/v3/diag/t3_story.py` `_fills` (lines 1061-1089), after
  `a_, e_ = ...L._gap_fills(...)` (line 1065), apply one per-filler factor to both the word mass `m_` and the end mass `me`
  (line 1072). This keeps the (word, end) split.
- **Pair level.** In the same place, for fillers whose pair (l1, w) is not in `_st_` (those admitted by `_ok`, line 1056),
  take the pair-level missing mass N1(l1)/(c(l1)+N1(l1)) from `L._pair_counts` and nest it inside the trigram split.
- **Arm key.** `"escape": "n1"` (popped beside `edge_` at line 1011). The existing `pair` gate and `q_` are left as they are.

**How this differs from closed lines.**

- r195 `kn` backed off from (l2, l1) to the (l1) count level, dropping l2, which is the class C chain. Its five Witten-Bell
  levels multiplied the escapes and collapsed novelty (tri .03).
- r179 drew the escaped mass from the type licence with Witten-Bell d/(d+c). Section 3a shows Witten-Bell is about 2x too
  generous after frequent histories, which is exactly where TRELLIS's class A failures sit (91% after left words with 200+
  tokens).
- Here a single level per novelty unit sets only how much mass is novel. The tree read decides which words get it.

**Predicted effect at n=4000** (edge w5, optimum likely t2.0-2.4):

| factor | now | predicted |
|---|---|---|
| per-step novelty, history 32-63 / 64+ | .13-.17 / .08-.10 | <= .10 / <= .06 |
| per-step novelty, history 1 | .44-.48 | >= .52 |
| class A share of novel | .09 | about .05 |
| class A gram | .11-.17 | .20-.30 (signal: the left word's missing mass separates class A grammatical from ungrammatical at AUC .64-.65, n=1000 and n=4000) |
| class C junctions after histories >= 30 | 17% at gram .26-.46 | about 9% |
| gram\|tri | .410 | .425-.44 |
| fair | .319 | .325-.34 |

At n=1000 the pair-level part should help less, because only 33% of class A junctions follow 200+-token words there. The rare-history
under-innovation exists at n=1000 too (.33-.37 against .55), so the fair count there may move by about ±.01.

**Honest limit.** For a single junction, history count predicts grammaticality only weakly in class C (AUC .53). The class C gain
therefore comes from where the novelty goes, not from removing bad junctions directly, and the predicted gram|tri gain is small.

**Falsification probe** (n=4000, seed 13):

- **(a) Count pre-test, no generation (minutes).** On the r204-n4000-w5 t2.0 seed-13 dump, compute N1(h)/(c(h)+N1(h)) per
  generated step (training stream only). Kill if, among single-junction class C sentences, the per-sentence missing mass of the
  junction's history has AUC < .55 for grammatical, and also, among class A sentences, the left word's has AUC < .60.
- **(b) Generation.** `edge:exact` + `escape:n1`, w5, t2.0 and t2.4, seed 13, judged, `r204_fair.py`.
  - Pass: fair >= .335 (seed-13 now .318), per-step novelty at history 32-63 <= .10, and gram|tri >= .44 at tri|d >= .75.
  - Kill: fair <= .322, or gram|tri < .43 at tri|d >= .75.

### Mechanism 2 (rank 2): a soft role check on the pivot, read from the left side

**Principle.** The rest of the class C loss is pivot-role chaining that grows with n: *thank you want*, *can you are*,
*give it is*, and the share of pronoun pivots rose from 5% to 11%. The question is whether the novel word c is something the
pivot b's own occurrences after a were ever followed by. The read for that already exists: `Trellis3.pivot_support` kind
`"left"` (trellis3.py line 2921), log P(c | left, p=b) / P(c | p=b). This is r204-design M2 with the "left" kind
instead of "pivot"; it was never run.

- The "pivot" kind cannot see these failures. When the left context fixes the pivot (after *thank*, only *you*),
  P(b | left, r1=c) is about P(b | left).
- The "left" kind asks the pivot's own occurrences after *thank*, and none of them was followed by *want*.
- As a weight, not a gate, it keeps breadth. In `t3_story.py` `both_` branch (lines 1099-1140), replace `continue` at line 1131
  (`go.get((l1, w), 0.0) < 0`) with multiplying that filler's weight by `exp(min(0, go[(l1, w)]))`. Use
  `_bk_ = "left"`, arm value `"both": "soft-left"`, and drop the boundary half (edge_exact handles it).
- It has no constant.

**Predicted effect at n=4000, on top of mechanism 1:** class C gram on pronoun pivots .13-.31 -> >= .40; gram|tri +.02 to
+.03; tri -.02 to -.04; fair +0 to +.015.

**Prior is weak.** Every kernel ratio is scale-invariant, like the read itself. r192's pivot kind had AUC .55 inside class C, and
r202's soft reweighting was null.

**Falsification probe** (n=4000, seed 13, no generation, one compute job):

- Rebuild the seed-13 n=4000 learner (as `r192_trace.py` / `r204_replay.py` do). Score `pivot_support(kind="left")` for the 1456
  single-junction class C sentences of `r204-n4000-w5` seed 13, then compute the AUC against the judge.
- Pass: AUC >= .62 overall and >= .65 on pronoun pivots (you / it / i). Only then run generation (w5, t2.0/2.4, with and
  without `soft-left`).
- Kill: AUC < .57.
- The same job should log leaf occupancy, mean depth and member types per concept at n=1000 vs n=4000. That is the direct test
  of (a) that this round could not make.

## 5. Verdict: is n=4000 parity reachable?

**No, not on present evidence.**

- **The size of the gap.** It is .039-.043 fair, all of it in gram|tri at moderate novelty. Parity needs either:
  - gram|tri of about .46 at TRELLIS's tri .82 / dist .95, or
  - about .54 at tri .70 / dist .92.
- **The source of the loss.** About 40% of TRELLIS's loss with n is mechanical and fixable:
  - novelty that ignores the evidence of the history (3a);
  - type-licensed pairs after the commonest words (class A);
  - pronoun pivots.

  The other 60% is a decline within pivots and within count bins that no count-level statistic tested here predicts.
  Single-junction AUCs are at most .53 for history, .47 for chaining absence, and .55 for r192's pivot read. That is the
  representation's similarity failing to sharpen with data, which the r204 design called the limit.
- **The ceiling of the proposed mechanisms.** Mechanisms 1 + 2 at their predicted upper ends give gram|tri about .45-.46 at
  tri .78-.80, which is fair about .335-.345.
- **Probability of parity.** Matching LSTM/GPT on 2 seeds would need the upper end of both predictions plus a temperature
  gain. I put that below 20%.
- **The defensible claim:**
  - TRELLIS ties LSTM and beats GPT up to n=1000.
  - At n=4000 its loss traces to a named property: its novelty and its read are ratios of similarity sums, so they do not
    convert absolute evidence into confidence the way real text and the neural models do (3a).
  - That property is what a scaling version would have to fix.
