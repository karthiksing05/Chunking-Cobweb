# Generation: TRELLIS against neural language models

This section is the paper's generation claim. No paper draft exists in the repository yet, so it stands alone. Every number comes from `experiments/v3/results/story_gen/` (`r212_seeds.md`, `r205_results.md`, `r204_results.md`, `r200-fair-n*.txt`, `r204-fair-n4000.txt`, `r206_scaling_diag.md`).

## Claim

**Learning from 250 to 1,000 sentences, TRELLIS (Cobweb trees, no gradients) matches the LSTM (8 seeds; the paired 95% CI includes 0 at every size). It beats a matched GPT at 250 and 500 sentences (8 of 8 seeds each) and is level with it at 1,000. At 4,000 sentences both neural models pull ahead.**

The headline rests on the 8-seed confirmation of round 212 (below); the four-seed tables from rounds 200 to 205 precede it. The claim covers one metric, one corpus (TinyStories sentences), and the 250 to 1,000 range. It does not say that TRELLIS beats neural language models in general.

## Method of comparison

**Matched data.** TRELLIS, an LSTM and a GPT learn the same n sentences, for n = 250, 500, 1,000 and 4,000. Each neural model is sized by held-out loss.

**Fair count.** A draw succeeds when it yields a sentence that is distinct from the other draws, new (not stored), judged grammatical by the same Qwen judge, and holds a trigram the stream never stored. The score is the number of successes per attempted draw, so failed draws and repeats count against the generator. Low-temperature neural samples repeat themselves (2 to 14% distinct at temperature .1 to .3), and this count removes the advantage that repetition would otherwise give them.

**Temperature bracketing.** Every family is swept over its sampling temperature and scored at its best setting. Round 200 swept the LSTM and the GPT, and both optima (temperature .7 to 1.0) lie inside the grid. Round 205 extended only TRELLIS's grid, to temperature 3.5 with context windows 5, 6 and 7. That extension is fair because the neural optima are interior. TRELLIS's own optima are interior at 500 and 1,000 sentences and sit at the top of the grid at 250.

**Select on two seeds, report on two.** Settings are chosen on seeds 13 and 17. Seeds 19 and 23 are held out and reported separately. The "all seeds" columns use the best setting on the mean of the four seeds, for every family alike.

## Result

Fair count, mean over seeds 13, 17, 19 and 23 (4,000 sentences: seeds 13 and 17 only). "Wins" counts the seeds on which TRELLIS with exact boundaries scores above the neural model.

| n | TRELLIS (exact) | LSTM | GPT | wins vs LSTM, mean diff | wins vs GPT, mean diff |
|---|---|---|---|---|---|
| 250 | .241 | .244 | .213 | 2 of 4, -.003 | 4 of 4, +.028 |
| 500 | .301 | .304 | .283 | 2 of 4, -.003 | 4 of 4, +.018 |
| 1,000 | .338 | .344 | .324 | 2 of 4, -.006 | 3 of 4, +.014 |
| 4,000 | .319 | .358 | .362 | 0 of 2, -.039 | 0 of 2, -.043 |

On the held-out seeds 19 and 23, with settings chosen on 13 and 17, TRELLIS scores .233, .302 and .339 at 250, 500 and 1,000 sentences. The LSTM scores .249, .312 and .351, and the GPT scores .216, .288 and .313. Against the GPT, TRELLIS wins 2 of 2 seeds at every size (+.017, +.014, +.027). Against the LSTM it wins 0, 1 and 1 of 2 seeds (-.017, -.011, -.012).

Across the three sizes TRELLIS beats the GPT on 11 of 12 seed-size pairs on these four seeds (the 1,000-sentence edge did not replicate on eight seeds; see round 212). The differences against the LSTM range from -.003 to -.006, which sit well inside the seed spread (the LSTM's per-seed range at 1,000 sentences is .308 to .395), so the two tie. Two held-out seeds are too few to separate either comparison on their own.

The boundary-strict count drops distinct sentences whose first word never starts, or whose last word never ends, a real held-out sentence, for every family. On this stricter count TRELLIS scores .224, .275 and .307 at 250, 500 and 1,000 sentences, against .226, .288 and .325 for the LSTM and .202, .271 and .309 for the GPT. TRELLIS is level with the GPT and trails the LSTM by .002 to .018.

## 8-seed confirmation (round 212)

Round 212 adds seeds 29, 31, 37 and 41 to seeds 13, 17, 19 and 23. It uses the same pipeline, the same Qwen2.5-3B judge and the same fair count, with settings fixed from round 205: TRELLIS edge n250 w7 t3.5, n500 w6 t2.8, n1000 w7 t2.0; LSTM t.7/.8/.8; GPT t.7/.8/.8. The new seeds are independent shuffles built the same way as the old ones. Training sets overlap across seeds (mean overlap with another seed: .077 at 250, .107-.117 at 500, .136-.142 at 1,000), and the overlap between old and new seeds equals the overlap among old seeds. Source: `experiments/v3/results/story_gen/r212_seeds.md` and `r212-analysis.txt`.

Fair count, mean over 8 seeds (old 4 / new 4 in brackets):

| n | TRELLIS edge | LSTM | GPT |
|---|---|---|---|
| 250 | .2419 (.2408 / .2430) | .2323 (.2436 / .2211) | .2068 (.2127 / .2010) |
| 500 | .2982 (.3011 / .2952) | .2990 (.3044 / .2935) | .2738 (.2827 / .2649) |
| 1000 | .3281 (.3381 / .3180) | .3346 (.3439 / .3252) | .3282 (.3244 / .3321) |

Paired differences, fair count, 8 seeds:

| n | contrast | mean | t 95% CI | bootstrap 95% CI | wins |
|---|---|---|---|---|---|
| 250 | TRELLIS-LSTM | +.0096 | [-.0110, +.0301] | [-.0064, +.0252] | 5/8 |
| 250 | TRELLIS-GPT | +.0351 | [+.0163, +.0539] | [+.0217, +.0505] | 8/8 |
| 500 | TRELLIS-LSTM | -.0008 | [-.0129, +.0113] | [-.0108, +.0075] | 4/8 |
| 500 | TRELLIS-GPT | +.0244 | [+.0098, +.0390] | [+.0146, +.0368] | 8/8 |
| 1000 | TRELLIS-LSTM | -.0065 | [-.0279, +.0149] | [-.0228, +.0101] | 3/8 |
| 1000 | TRELLIS-GPT | -.0002 | [-.0227, +.0223] | [-.0161, +.0184] | 4/8 |

**Pre-registered rules and verdicts** (fair count; t and bootstrap CIs agree).

- "Matches LSTM" (CI includes 0 and lower bound above -.03) holds at n=250, 500 and 1,000. The lower bounds are -.0110/-.0064 at 250, -.0129/-.0108 at 500 and -.0279/-.0228 at 1,000 (t/bootstrap). At 1,000 the t lower bound clears -.03 narrowly.
- "Beats GPT" (CI above 0) holds at n=250 and 500 (t lower bounds +.0163 and +.0098; 8/8 seeds each). It fails at n=1,000 (CI [-.0227, +.0223], 4/8 wins).

The round 205 edge over the GPT at 1,000 sentences (+.014 on four seeds) did not replicate. The four new seeds give -.014 (GPT at t.8 scores .332 against .318 for TRELLIS). The claim that TRELLIS beats a matched GPT now holds at 250 and 500 sentences only, and TRELLIS is level with the GPT at 1,000.

**Boundary-strict count, 8 seeds.** Under the strict count TRELLIS matches and beats only at 250 sentences. At 1,000 sentences it trails both models by about .014.

| n | TRELLIS | LSTM | GPT | TRELLIS-LSTM: mean, t CI, boot CI, wins | TRELLIS-GPT: mean, t CI, boot CI, wins |
|---|---|---|---|---|---|
| 250 | .2236 | .2154 | .1959 | +.0082 [-.0102, +.0266] [-.0059, +.0221] 5/8 | +.0278 [+.0104, +.0452] [+.0146, +.0415] 7/8 |
| 500 | .2725 | .2819 | .2617 | -.0094 [-.0211, +.0023] [-.0193, -.0016] 2/8 | +.0108 [-.0055, +.0270] [-.0017, +.0235] 5/8 |
| 1000 | .3003 | .3147 | .3139 | -.0144 [-.0392, +.0104] [-.0334, +.0049] 2/8 | -.0136 [-.0363, +.0091] [-.0297, +.0052] 2/8 |

The same rules applied to the strict count (an extension; they were not pre-registered for it) give: matches LSTM holds at 250 under both CIs, holds at 500 under t and fails under bootstrap (upper bound -.0016), and fails at 1,000 under both (t lower bound -.0392). Beats GPT holds at 250 only.

**Selection re-check** (fair count means per setting).

- n=500: TRELLIS t2.8 is best on 8 seeds (.2982 against .2918 at t2.4). On the new seeds alone t2.4 (.2971) edges t2.8 (.2952). The optimum is interior. LSTM is best at t.8 on 8 seeds (t.7 on the new seeds, .3000 against .2935); GPT is best at t.8.
- n=1,000: TRELLIS t2.0 is best on the new seeds, the old seeds and all 8. The optimum is interior. LSTM and GPT are best at t.8 on every subset (GPT t.9, .3298, is close on the new seeds).
- n=250: TRELLIS t3.5 is no longer best on either subset. The new seeds favour t4.0 (.2451, against .2430 at t3.5 and .2389 at t3.0), and the old seeds favour t4.5 (.2579). LSTM and GPT are best at t.7 on every subset.

On the old seeds the n=250 temperature extension gives t2.8 .2196, t3.5 .2408, t4.5 .2579 and t6.0 .2561, so the optimum is interior and flat from 4.5 to 6.0. At t4.5 TRELLIS scores .2579 against the LSTM's .2436 (+.0143) and the GPT's .2127 (+.045). That setting came from looking at these seeds, so the result is post hoc and not confirmatory, and t4.5 was not run on the new seeds. The confirmatory n=250 comparison above uses the pre-fixed t3.5.

**Reading.** The tie with the LSTM survives eight seeds at every size (+.0096, -.0008, -.0065; all inside noise). The win over the GPT is solid at 250 and 500 sentences (8/8 each) and absent at 1,000. On the boundary-strict count the picture is weaker: TRELLIS trails the LSTM at 500 and 1,000 and is level with or below the GPT at 1,000.

## What produced the tie

The tie comes from the exact boundary read of round 204. The plain generator let a sentence start or end on a word that never starts or ends a stored sentence, and the judge accepted many of these fragments. Reading the boundary exactly (a start or end attribute counts only the stored occurrences that had that boundary) cuts the share of distinct outputs with an atypical boundary from .32-.42 to .07-.12, and cuts boundary-cut copies of stored sentences from 143 rows to 9. At 1,000 sentences the fair count rises from .316 to .338. At 250 and 500 sentences the exact read with round 204's temperature grid (up to 2.0) scored below plain TRELLIS (.197 against .219, .272 against .280); the round 205 grid, which extends to 3.5, lifts it to .241 and .301.

## Limit at 4,000 sentences

TRELLIS trails both neural models at 4,000 sentences: .319 against .358 for the LSTM and .362 for the GPT, on both seeds. The strict count shows the same gap (.264 against .325 and .328).

Round 206 diagnosed the cause. From 1,000 to 4,000 sentences all three families produce more distinct sentences, but only TRELLIS's new-trigram sentences get less grammatical (.464 to .410, while the LSTM holds at .50 and the GPT rises from .487 to .570). TRELLIS's novelty ignores absolute evidence. Real held-out text and both neural models innovate after a history seen k times at a rate that depends on k alone, the same at 1,000 and 4,000 sentences. TRELLIS innovates too little after rare histories and too much after frequent ones, and the imbalance grows with n. The round estimates that fixing it recovers .015 to .03 of fair count, which leaves TRELLIS short of parity at 4,000 sentences. A first test of such a fix (round 207) raised novelty without raising the fair count.

## Caveats

- **Judge ceiling.** The Qwen judge passes about 89% of real held-out sentences (r201 measures .89; call it .90). No generator can score above that grammaticality rate, and judge errors affect every family alike.
- **Small margins.** The TRELLIS-LSTM differences (-.003 to -.006) and the four-seed TRELLIS-GPT difference at 1,000 sentences (+.014, not replicated in round 212) are small next to the seed spread. Round 212 supports the claim with paired confidence intervals over eight seeds.
- **Selection.** In rounds 200 to 205 only two seeds are held out, and TRELLIS's grid was extended after round 204 put its optimum at the grid edge. The neural grids were not extended because their optima are interior.
- **Scope.** The result holds for one corpus, one judge and the fair count as defined here. At matched novelty (the round 190 frontier) the neural models lead, and TRELLIS's new-trigram sentences are 50 to 63% grammatical against the LSTM's 88% at low temperature.
- **Earlier claim withdrawn.** Round 183 reported that TRELLIS beats the neural models at 1,000 sentences, and round 205 reported a GPT edge there (+.014, four seeds). The round 183 result used neural models sampled at temperature 1.0 and does not survive their tuned temperatures; round 212 found no GPT edge at 1,000 sentences either.

## High-grammaticality operating points (rounds 209-211)

The default judge (Qwen2.5-3B) rates real TinyStories sentences .89 grammatical, so it cannot certify .95. A stricter prompt raises the real-sentence ceiling to .958 (Qwen3-8B) and .970 (phi-4) while still rejecting scrambled and content-swapped controls (at most .075 pass); round 209 judged all families with these strict judges.

At n=1000, no generator reaches .95 grammaticality among its distinct new sentences under either strict judge, at any temperature (round 211). The highest values are LSTM .914-.926, plain TRELLIS (3-word window) .878-.914, the 5-gram sampler .898-.926 with no unstored trigrams, GPT .850-.852, and TRELLIS with exact boundaries about .80. Where a .90 constraint can be met, the LSTM keeps more novelty than TRELLIS (unstored-trigram share .48 against .19), and 87% of plain TRELLIS's grammatical new sentences at that level recombine stored trigrams. The tie with the LSTM therefore holds at the fair-count optimum, where every model is .58-.66 grammatical, and not at the high-grammaticality end.

Repeat tracking (round 210) halves repeated words but leaves the fair count unchanged (.331 against .339 held-out), because repeats account for about 16% of ungrammatical sentences.
