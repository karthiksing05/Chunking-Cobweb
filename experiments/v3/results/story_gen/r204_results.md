# Round 204 results: M1, the exact boundary read (edge_exact)

Design: `r204_design.md` (M1). Code: `src/trellis2/trellis3_r204.py` (`Trellis3R204`, flag `edge_exact`, default False),
arm key `"edge": "exact"` in `experiments/v3/diag/t3_story.py` (causal branch, additive), pre-test hook `STORY_R204_REPLAY` +
`experiments/v3/diag/r204_replay.py`, aggregation `experiments/v3/diag/r204_fair.py`. Tests: `tests/trellis2/test_r204_edge.py`
(default byte-identical, non-boundary queries untouched, start read = first words only, end read = real ends only, toy counts,
records/cache restored); the full `tests/trellis2` suite passes (115 passed, sbatch job 4779952).

Implementation: for a query with `l1 = <s>` or `r1 = </s>` the boundary attribute is dropped from the query and the committed records are
filtered to those whose own context holds that value; the unchanged `_gap_fills` then runs on that subset (private cache keyed with the flag).
That is exactly "factor 1 if the occurrence had the boundary, skip otherwise"; interior attributes keep the back-off.

Settings otherwise as r200: plain causal words arm, pair gate, new=true, story_facet, windows 5/6/7, temp 1.3/1.6/2.0, n=250/500/1000
seeds 13/17/19/23, n=4000 seeds 13/17. Same judge, same fair-count rules (`story_frontier_fair.py` definitions, re-implemented in
`r204_fair.py` and checked against r200: plain TRELLIS, LSTM and GPT values reproduce r200's).

## 1. Replay pre-test (no generation), seed 13, n=1000, r200 w6 t1.6 dump (1600 rows)

Replay of each row's start step and end step under the generator's own step distribution (pair gate, temperature 1.6), default read vs
edge_exact; delta = log P_exact - log P_default of the drawn (word, done) at that step, in nats. A boundary is atypical if the word starts
(ends) fewer than 2 of the 1200 held-out real sentences. 41.7% of the 1600 sentences have an atypical start or end.
A step the read gives zero mass is floored at -30 nats and counted separately. (`r204_replay_s13.json`)

| boundary step | group | n | median delta | share losing >= 1.5 nats | zero mass under exact | finite-only median | share losing > .3 |
|---|---|---|---|---|---|---|---|
| start | atypical | 509 | -23.8 (floor) | .884 | 450 | +.126 (59 finite) | .884 |
| start | typical (fully typical sentences) | 932 | +.125 | .003 | 3 | +.125 | .009 |
| end | atypical | 261 | -28.4 (floor) | .651 | 150 | .000 (111 finite) | .670 |
| end | typical | 932 | +.002 | .078 | 7 | +.002 | .080 |
| pooled | atypical | 770 | -24.0 | | | | |
| pooled | typical | 1864 | +.10 | | | | |

**Pass** (atypical median <= -1.5 and typical median >= -0.3), so the probe ran. Reading it honestly:
- The atypical medians are dominated by zero mass: 88% of atypical starts and 57% of atypical ends are words that never started / ended a stored
  sentence, so exact gives them probability 0 (the -24 / -28 are the floor). That is the intended effect, but the numbers are "cannot happen", not graded losses.
- Collateral: 7.8% of boundary-typical end steps also lose >= 1.5 nats (mean -2.5), 3 + 7 typical steps lose all mass. The median criterion hides this tail.
- At the start step the surviving mass is renormalised: typical starts gain +.125 nats.

## 2. Probe: seed 13, n=1000, edge_exact, w6 and w7 (fair = distinct new grammatical unstored-trigram per attempted draw)

| arm | fair | strict | dist/draw | tri \| distinct (novelty) | gram \| tri | atypical share of distinct | repeat rows | repeats that are cut copies of stored sentences |
|---|---|---|---|---|---|---|---|---|
| edge w6 t1.3 | .282 | .260 | .703 | .772 | .520 | .084 | 475 | 10 |
| edge w6 t1.6 | .301 | .276 | .757 | .814 | .488 | .110 | 389 | 11 |
| edge w6 t2.0 | .326 | .292 | .803 | .844 | .480 | .114 | 315 | 5 |
| edge w7 t1.3 | .316 | .294 | .731 | .770 | .562 | .074 | 431 | 10 |
| edge w7 t1.6 | .325 | .293 | .779 | .812 | .514 | .104 | 354 | 16 |
| **edge w7 t2.0** | **.337** | .309 | .807 | .831 | .502 | .100 | 308 | 9 |
| plain w6 t1.6 (r200, same seed) | .302 | .193 | .782 | .866 | .446 | .379 | 349 | 143 |
| plain w5 t1.6 (r200 best s13) | .314 | .212 | .801 | .825 | .474 | .318 | 318 | 85 |
| LSTM t0.8 (r200) | .342 | .318 | .871 | .776 | .506 | .100 | 207 | 65 |
| GPT t0.8 (r200) | .317 | .299 | .833 | .834 | .456 | .101 | 268 | 59 |

Pass rule: fair >= .335 (kill <= .314): best cell .337 passes, by 0.002 (about one fifth of one binomial SE, .011). Atypical share at the best cell is .100
(design target <= .10; kill > .15). The best cell is the last grid cell (w7 t2.0), so the optimum lies at or beyond the edge of the probed grid.
The mechanism claim is confirmed: atypical-boundary share falls from .32-.42 to .07-.11, boundary-cut-copy repeats from 143 rows to 9.

## 3. Full grid (4 seeds at n <= 1000)

Setting chosen on seeds 13/17, reported on held-out 19/23; "best" = best setting on the mean of all four seeds. Paired = per seed against the r200 LSTM/GPT best.
`r204-fair-n{250,500,1000}.txt` (and `.perseed.tsv`) hold every setting.

### Fair count

| n | family | best setting | mean (all seeds) | selected on 13/17 | held-out 19/23 |
|---|---|---|---|---|---|
| 250 | TRELLIS plain (r200) | w5 t2.0 | .219 | .216 | .223 |
| 250 | **TRELLIS edge** | w5 t2.0 | .197 | .201 | .194 |
| 250 | LSTM / GPT | t0.7 | .244 / .213 | .238 / .210 | .249 / .216 |
| 500 | TRELLIS plain | w6 t2.0 | .280 | .289 | .271 |
| 500 | **TRELLIS edge** | w6 t2.0 | .272 | .273 | .270 |
| 500 | LSTM / GPT | t0.8 | .304 / .283 | .296 / .277 | .312 / .288 |
| 1000 | TRELLIS plain | w6 t1.6 | .316 | .323 | .310 |
| 1000 | **TRELLIS edge** | w7 t2.0 | **.338** [.326, .353] | .337 | **.339** |
| 1000 | LSTM / GPT | t0.8 | .344 / .324 | .337 / .336 | .351 / .313 |

Paired, edge best vs LSTM / GPT best (wins of 4 seeds, mean diff; held-out 19/23 in brackets):

| n | vs LSTM | vs GPT |
|---|---|---|
| 250 | 0/4, -.046 [0/2, -.056] | 1/4, -.015 [0/2, -.022] |
| 500 | 0/4, -.033 [0/2, -.043] | 0/4, -.011 [0/2, -.018] |
| 1000 | 2/4, -.006 [1/2, -.012] | 3/4, +.014 [2/2, +.027] |

(r200 plain at n=1000 for reference: vs LSTM 1/4, -.027 [0/2, -.041]; vs GPT 1/4, -.008 [1/2, -.002].)

At n=1000 the edge read gains +.022 over plain on the fair count (3 of 4 seeds higher: s13 .337 vs .302, s17 .337 vs .343 is the exception, s19 .353 vs .324, s23 .326 vs .297; plain taken at its all-seed best, w6 t1.6),
which puts TRELLIS level with LSTM (-.006) and above GPT (+.014); held-out 2 seeds are too few to separate these from noise. At n=250 and n=500 the edge read is worse than plain (-.022, -.008): with
less data the mass freed from the boundary leak goes to copies, not to new sentences (n=250 distinct/draw .445 vs .58, 844 repeat rows of 1600).

### Boundary-strict fair count (counts only distinct new, grammatical, unstored-trigram sentences whose first word starts >= 2 and last word ends >= 2 of the 1200 real held-out sentences; every family)

| n | plain TRELLIS (best) | edge TRELLIS (best) | LSTM | GPT | edge vs LSTM / GPT (wins of 4, mean diff) |
|---|---|---|---|---|---|
| 250 | .130 (w7 t2.0) | .184 (w5 t2.0) | .226 | .202 | 0/4 -.041 / 1/4 -.017 |
| 500 | .171 (w5 t1.6) | .251 (w6 t2.0) | .288 | .271 | 0/4 -.038 / 0/4 -.020 |
| 1000 | .237 (w4 t1.3) | .307 (w7 t2.0) | .325 | .309 | 1/4 -.018 / 2/4 -.002 |

Held-out (selected on 13/17) at n=1000: edge .307, LSTM .333, GPT .296. Best-to-best, the strict count is below the fair count by 25-41% for plain TRELLIS (n=1000: .316 -> .237), 4-8% for the neural models and 7-9% for edge TRELLIS. The edge read removes most of the plain gap to the neural models on the strict count (n=1000: -.088 -> -.018 against LSTM), but does not close it.
The lexicon is built from the held-out real sentences of the same four seeds (as the design's atypicality proxy), so it is the same proxy for every family.

### Other factors at n=1000 (mean of 4 seeds, best edge cell w7 t2.0 vs plain w6 t1.6)
dist/draw .814 vs .800, tri|distinct .836 vs .855, gram|tri .497 vs .462, atypical share .115 vs .393, repeat rows 282 vs 304 (cut copies of stored sentences 11 vs 133).
So at n=1000 the gain is in gram|tri (+.035) at about unchanged distinctness and slightly lower novelty, as the design predicted (gram|tri .50-.54, tri .76-.82, dist .78-.84, fair point .34).

## 4. n = 4000 (seeds 13, 17)

Submitted (jobs 4780731-4780736, 160G, 48h); the w6 seed 17 job was Killed (node out-of-memory, as r200 had hit) after 1h06 and is resubmitted as 4781938 (200G, node droid excluded).
At the time of writing none of the n=4000 arms had finished except w5 seed 13 arm 0 (each arm takes about 2h or more), so **n=4000 is not reported here**.
The chain `logs/v3_r204_n4000_final.sh` (jobs 4782212 judge, 4782213 fair, dependent on 4780737 and 4781938) writes
`experiments/v3/results/story_gen/r204-fair-n4000.txt` (+ `.tsv`, `.perseed.tsv`) in the same format as the tables above.

## 5. Verdict

- **Mechanism: confirmed.** The exact read removes the boundary leak: atypical-boundary share of distinct outputs falls from .32-.42 (plain w5-w7 t1.6) to .07-.12, boundary-cut-copy repeats from ~130 rows to ~10,
  and the pre-test shows atypical boundary steps go to zero probability (the 88% / 57% figures) while the median typical step is unchanged. (A tail of 8% of typical end steps loses >= 1.5 nats.)
- **Fair count: a regime-dependent gain, not a general one.** Passes the seed-13 probe by .002 (.337 vs the .335 line) and, over 4 seeds at n=1000, lifts the best fair count .316 -> .338 (selected 13/17, held-out .339 vs plain .310),
  matching LSTM (.344; held-out .351) and clearing GPT (.324; held-out .313) by a margin that two held-out seeds cannot call. At n=250 and n=500 it is worse than plain TRELLIS (.197 vs .219, .272 vs .280)
  and below both neural models at every seed there. Whether it helps at n=4000 is unmeasured (pending).
- **Boundary-strict count: clear gain.** Edge TRELLIS is +.054 / +.080 / +.070 over plain at n = 250 / 500 / 1000. The fair-count gap that the leak's judge-accepted fragments were hiding is now visible and mostly closed at n=1000 (-.018 vs LSTM, -.002 vs GPT), but the neural models stay ahead at n <= 500.
- Both r204 probe thresholds were passed narrowly; every edge optimum sits at t2.0, the last temperature on the grid, so the grid is truncated on the edge arm and a t2.5/3.0 column could change the n=250/500 comparison. That was outside the stated design and not run.
- Prior prediction (design M1: fair .32-.36, point .34 at n=1000; .32-.35 at n=4000): n=1000 point matched (.338). The design said nothing about n <= 500, where it fails.

## 6. Rule slips and notes

- One `python3` invocation (empty heredoc, no script, nothing computed) was run on the login node by mistake; no other python ran there.
- The first fair step of `logs/v3_r204_probe.sh` lacked PYTHONPATH for `cobweb_cu` and failed (no compute lost); rerun as `logs/v3_r204_probe_fair.sh`, and the fix was folded into `logs/v3_r204_grid.sh`.
- One n=4000 job Killed (above); rerun pending.
- t3_story.py was edited (three additive hunks: the `edge` arm key, the reset after generation, the `STORY_R204_REPLAY` hook); its default path is unchanged (the new key is popped only in the causal branch, and the reset runs only when `edge == "exact"`).
