# Generation: TRELLIS against neural language models

This section is the paper's generation claim. No paper draft exists in the repository yet, so it stands alone. Every number comes from `experiments/v3/results/story_gen/` (`r205_results.md`, `r204_results.md`, `r200-fair-n*.txt`, `r204-fair-n4000.txt`, `r206_scaling_diag.md`).

## Claim

**Learning from 250 to 1,000 sentences, TRELLIS (Cobweb trees, no gradients) matches the LSTM within seed noise and beats a matched GPT on the number of new, grammatical sentences it generates per draw. At 4,000 sentences both neural models pull ahead.**

The claim covers one metric, one corpus (TinyStories sentences), and the 250 to 1,000 range. It does not say that TRELLIS beats neural language models in general.

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

Across the three sizes TRELLIS beats the GPT on 11 of 12 seed-size pairs. The differences against the LSTM range from -.003 to -.006, which sit well inside the seed spread (the LSTM's per-seed range at 1,000 sentences is .308 to .395), so the two tie. Two held-out seeds are too few to separate either comparison on their own.

The boundary-strict count drops distinct sentences whose first word never starts, or whose last word never ends, a real held-out sentence, for every family. On this stricter count TRELLIS scores .224, .275 and .307 at 250, 500 and 1,000 sentences, against .226, .288 and .325 for the LSTM and .202, .271 and .309 for the GPT. TRELLIS is level with the GPT and trails the LSTM by .002 to .018.

## What produced the tie

The tie comes from the exact boundary read of round 204. The plain generator let a sentence start or end on a word that never starts or ends a stored sentence, and the judge accepted many of these fragments. Reading the boundary exactly (a start or end attribute counts only the stored occurrences that had that boundary) cuts the share of distinct outputs with an atypical boundary from .32-.42 to .07-.12, and cuts boundary-cut copies of stored sentences from 143 rows to 9. At 1,000 sentences the fair count rises from .316 to .338. At 250 and 500 sentences the exact read with round 204's temperature grid (up to 2.0) scored below plain TRELLIS (.197 against .219, .272 against .280); the round 205 grid, which extends to 3.5, lifts it to .241 and .301.

## Limit at 4,000 sentences

TRELLIS trails both neural models at 4,000 sentences: .319 against .358 for the LSTM and .362 for the GPT, on both seeds. The strict count shows the same gap (.264 against .325 and .328).

Round 206 diagnosed the cause. From 1,000 to 4,000 sentences all three families produce more distinct sentences, but only TRELLIS's new-trigram sentences get less grammatical (.464 to .410, while the LSTM holds at .50 and the GPT rises from .487 to .570). TRELLIS's novelty ignores absolute evidence. Real held-out text and both neural models innovate after a history seen k times at a rate that depends on k alone, the same at 1,000 and 4,000 sentences. TRELLIS innovates too little after rare histories and too much after frequent ones, and the imbalance grows with n. The round estimates that fixing it recovers .015 to .03 of fair count, which leaves TRELLIS short of parity at 4,000 sentences. A first test of such a fix (round 207) raised novelty without raising the fair count.

## Caveats

- **Judge ceiling.** The Qwen judge passes about 89% of real held-out sentences (r201 measures .89; call it .90). No generator can score above that grammaticality rate, and judge errors affect every family alike.
- **Small margins.** The TRELLIS-LSTM differences (-.003 to -.006) and the TRELLIS-GPT difference at 1,000 sentences (+.014) are small next to the seed spread. The claim rests on the consistent sign over seeds and sizes, not on any single cell.
- **Selection.** Only two seeds are held out, and TRELLIS's grid was extended after round 204 put its optimum at the grid edge. The neural grids were not extended because their optima are interior.
- **Scope.** The result holds for one corpus, one judge and the fair count as defined here. At matched novelty (the round 190 frontier) the neural models lead, and TRELLIS's new-trigram sentences are 50 to 63% grammatical against the LSTM's 88% at low temperature.
- **Earlier claim withdrawn.** Round 183 reported that TRELLIS beats the neural models at 1,000 sentences. That result used neural models sampled at temperature 1.0 and does not survive their tuned temperatures.
