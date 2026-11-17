# Round 192: why TRELLIS's new trigrams are ungrammatical, and one check against it

Setting: causal words-gated arm `{"causal":1,"units":"words","pair":"gate","new":true}`, learner window 3/4/5, n = 1000,
the judged r188 dumps (temp 0.5 and 0.7). Tools (all on compute nodes):

- `experiments/v3/diag/r192_trace.py`: rebuilds the seed's stream and learner exactly as `t3_story.py`, and for every
  judged sentence with a trigram the stream never had (padded with `<s>`, `</s>`) traces the step that placed the
  trigram's last word: the gated filler share p, whether the pair (l1, w) was stored or only passed the type-licence gate,
  the committed occurrences of w that carried its mass, and for the top ones the concept on the occurrence's context path
  that licensed each attribute of the query (the deepest ancestor holding the value: height above the leaf, size, lift).
- `r192_summarize.py` / `r192_cases.py`: feature splits by the judge, AUCs, classes and cases
  (`r192-diag/summary_all_s13.txt`, `r192-diag/cases_s13.txt`; seed 13, windows 3/4/5, both temps pooled:
  2719 new-trigram sentences, 48.8% judged grammatical).
- `r192_pretest.py`: candidate checks scored on the same dumps without generating (`r192-diag/pretest_w{3,4,5}_s{13,17}.txt`).

## 1. Three kinds of failure, by where the new trigram comes from

| class (seed 13, w3-5 pooled) | share of new-trigram sentences | judged grammatical |
|---|---|---|
| A. the pair (l1, w) was never stored; the type-licence gate let it through | 12.2% | 0.12 |
| B. the sentence ends where it never ended before ((l1, w, `</s>`) new) | 23.1% | 0.36 |
| C. every pair stored, only the trigram (l2, l1, w) new | 64.7% | 0.60 |

(The no-new-trigram generations are judged grammatical at about 0.88-0.91, the judge's ceiling, so the new trigram is
what costs the sentence.)

**A: the type licence licenses substitutions across agreement and function words.** 327 of the 372 class-A sentences
are judged ungrammatical. Nearly all are one of a handful of pairs: *they was*, *she were*, *max were*, *tim were*,
*she and*, *he and*, *and and*, *tom and to*, *max said thank you bird*. The trace shows the mechanism: `was` after `they`
gets its mass from occurrences of `was` after `he` / `she` / `it`, whose leaves back off to a concept 1 level up where
`l1 = they` has a count (they, he, she share almost every other neighbour), and the type licence (2.2 against a median of
1.4) passes the pair. Distributional similarity is high exactly where agreement makes the substitution wrong.

**B: ends after words that never end a sentence.** *she loved to play outside in the*, *it was a big red*, *he was very*,
*they did not want to*, *said the little*, *he wanted to play with his*. The end is read jointly with the word through
r1 = `</s>`; a single-occurrence leaf backs off half of its parent's end probability, so a determiner or preposition still
gets P(end) of 0.05-0.27 after a few levels, and the pair gate never looks at (w, `</s>`): the end marker has no type
licence and the gate skips it. The same happens at the start (`l1 = <s>` is skipped by the gate): *loved it on a special
rock*, *smiled and her friend tom*, *wanted it was very pretty*, *boy is so pretty*.

**C: bigram chaining through a pivot used in two ways.** Both pairs are stored but the middle word (the pivot) is used
differently on each side:

- *to* infinitive vs preposition: *tom wanted to the friends*, *they went to be careful*, *the ball went to play outside*
- copula vs auxiliary around *not* / *was* / *a*: *they were not share*, *they did not good friends*, *he was a happy to*,
  *you are a nice ball*, *i was fun and pretty*
- coordination that repeats or switches category: *tom and tom were*, *lily and lily*, *it was shiny and shiny*,
  *max was happy and have fun*, *it is so pretty and excited*
- pronoun / determiner slots: *they were they*, *lily and ben were they*, *sue was their mom*
- selection the judge reads as semantic: *spot loved to play with the park*, *the tree was playing*, *under a lot*
- agreement beyond the window: *she was very happy to help his mom*, *they like to play outside with his ball* (the new
  trigram is elsewhere in the sentence; the window cannot see the subject)

## 2. What does not explain it

Per step (sentence = its worst new step), AUC against the judge:

| statistic | AUC all | AUC class C (one new step) | AUC class B (one new step) |
|---|---|---|---|
| the gated filler share p of the new word | 0.50 | 0.44 | 0.49 |
| licensing concept height above the leaf (mass-weighted, l2 / r1) | 0.52 | 0.52 | 0.53 |
| share of w's mass from occurrences with the same l1 | 0.57 | 0.44 | 0.44 |
| pre-emption, N(history) x log(1 - p) (seen often, never with w) | 0.67 | 0.54 | 0.52 |
| pair stored (vs type-licensed) | 0.59 | - | - |
| **both sides: log P(pivot \| left, r1 = w) / P(pivot \| left)** | **0.67** | **0.55** | **0.58** |

The licensing concept is not an over-general concept near the root: in good and bad sentences alike the new value is
held 1-3 levels above a single-occurrence leaf (mass-weighted mean height 1.1 in judged-grammatical sentences, 1.5 in the rest), so a depth / gain-over-root threshold
(Kneser-Ney-style back-off limit) has nothing to cut. Neither the step probability nor how often the history was seen
separates them within a class. What separates them is the class (A 0.12, B 0.36, C 0.60) and, inside A and C, whether the
new word supports the previous word from the right.

## 3. The mechanism (round 192 arm key `"both"`) and why

A new word is placed by the context tree from its left context only, and every attribute of that context is backed off
independently at the occurrence's leaf; nothing asks whether the previous word, as it was placed, is the word that
stands before the new one. The check asks the context tree that question with the same reading it already uses for
generation: re-predict the previous word (the pivot) from its own left context, once alone and once with r1 = the new
word, and keep the new word only if it does not make the pivot less probable,

    log P(pivot | left, r1 = w) / P(pivot | left) >= 0     (Trellis3.pivot_support, kind "pivot")

By Bayes this is log P(w | left, pivot) / P(w | left): the new word must be at least as expected after this pivot as
after the left context with any word in the pivot's place, so a pivot whose occurrences before w are a different use
(*to* before *the* stood after *went*, not after *wanted*) or a different word altogether (*was* after *they* is
explained by *he*, not *they*) loses it. It needs no new constant: the threshold is the point where the right neighbour
stops carrying information for the pivot (zero pointwise mutual information). It only applies to trigrams memory never
stored (stored ones are licensed by the stream), it reads the context tree's committed occurrences with the same
hierarchical back-off as `_gap_fills` (no new tree, no tags, no fitted layer), and it is tried at generation time, not as
a post-filter. The sentence boundaries get the same rule where the ratio is uninformative (when the left context nearly
fixes the pivot, both reads agree and the ratio is ~0, e.g. *outside in the* `</s>`): the end and start markers are
neighbours the type licence cannot license, so a new end or start is allowed only at a word memory saw end or start a
sentence (the pair gate's own rule, which it skipped at the boundaries).

Pre-test on the judged r188 dumps (rule = every new step passes; kept = share of new-trigram sentences kept):

| w, seed | all: gram | pivot >= 0: kept, gram | + boundary words: kept, gram | "left" (PMI of left and w given pivot) >= 0 | no unstored pair at all |
|---|---|---|---|---|---|
| w3 s13 | 0.562 | 0.43, 0.718 | 0.43, 0.718 | 0.32, 0.477 | 0.88, 0.635 |
| w3 s17 | 0.618 | 0.55, 0.687 | 0.55, 0.687 | 0.20, 0.714 | 0.91, 0.665 |
| w4 s13 | 0.517 | 0.52, 0.666 | 0.52, 0.666 | 0.32, 0.352 | 0.83, 0.596 |
| w4 s17 | 0.501 | 0.46, 0.696 | 0.44, 0.715 | 0.41, 0.412 | 0.84, 0.581 |
| w5 s13 | 0.447 | 0.40, 0.613 | 0.34, 0.684 | 0.49, 0.390 | 0.91, 0.479 |
| w5 s17 | 0.436 | 0.38, 0.658 | 0.28, 0.704 | 0.56, 0.434 | 0.84, 0.501 |

The pivot ratio removes nearly all of class A (kept A share ~0) and raises class C from 0.60-0.67 to 0.66-0.77; the
boundary rule matters at w5, where new ends are 38-40% of the new-trigram sentences. The other reading of "both sides"
(the left context and w co-occurring among the pivot's own occurrences, kind "left") is worse than no check: the
back-off smooths the pivot's occurrences toward each other, so the joint read is noise. As a post-filter the check
reaches 0.67-0.72, short of the 0.80 target; what it leaves is mostly class C's coordination repeats and semantic
selection (section 1), which a window-local two-sided read cannot see.

## 4. How this differs from the closed lines (site Appendix, rounds 156-182)

Every closed line tried to make *unstored pairs* (bigrams) better or to rerank whole sentences:

| closed line | what it changed | why the r192 check is not a repeat |
|---|---|---|
| r156 / r159 gate quantiles, nearest-4 licence; r177 type-read fills (Kneser-Ney); r178 back-off over the parent's distinct fillers | how the LEFT context licenses a word, read by types or at nearer concepts | the check reads the RIGHT side: whether the new word supports the word before it (log P(pivot \| left, r1 = w) / P(pivot \| left)); no licence, quantile or type read is involved |
| r179 / r180 escape to an unstored next word from the type / pair / skip-pair licence | adds unstored pairs | the check adds nothing; it only removes fillers whose trigram is new and whose right-side read fails |
| r160 specialising a drawn word one level up | replaces a drawn word | nothing is replaced |
| r170 chart mixing (length drawn first) | a second generator | same generator; the ends that r170 left "mid-phrase" are what the boundary rule closes (r170 drew the length up front; here the end is gated where the pair gate never looked) |
| r182 parse-score critic (draw and rerank whole sentences, AUC .51-.57) | a post-hoc sentence score | the check is per step, at generation, on the step that makes the new trigram; its AUC as a sentence filter is .67 overall but only .55 (class C) / .58 (class B) inside a class, i.e. it is NOT a better critic. Its gain comes from what it removes at generation |

Where it overlaps, said plainly: (1) Most of its gain on class A is the same thing the strict pair gate does: unstored
pairs are bad (rounds 156-181 found 12-27% grammatical; here 12%), and the check rejects almost all of them (kept A share
~0 in the pre-test). The current arm (r184+) lets type-licensed unstored pairs through (12% of new-trigram sentences);
tightening the gate is not new, and the pre-test row "no unstored pair at all" (0.58-0.67) is that alone. (2) What is new
is (a) gating the sentence boundaries, which the pair gate skipped (class B, 4-40% of new-trigram sentences depending on
window, 0.36 grammatical), and (b) the right-side check on new trigrams of STORED pairs (class C, 65%), which no closed
line touched (they all worked on pairs). In the pre-test (b) adds +0.03 to +0.15 over "no unstored pair" in class C.
(3) It cannot fix the failures that are sentence-level (agreement such as *tim and sue was sad*, or *she ... his mom*,
fragments): those are in sentences whose trigrams may all be stored, and the r181 note that such failures are
sentence-level applies here unchanged.

## 5. Variant tried in pre-test and dropped: reach 2

Also checking the word two back as a pivot (l2 with r1 = l1 and the new word in r2; arm key `"reach": 2`) keeps only
5-9% of the new-trigram sentences and is no more grammatical (0.33-0.63 with the boundary rule, against 0.67-0.72 for
reach 1): with two right attributes the back-off makes almost every pair of right neighbours look unexpected. Not run.

## 6. Results (arm `"both": true`, seeds 13 17 19 23, judged; frontier computed with the earlier dirs at each n)

gram = new sentences judged grammatical, tri = share with an unstored trigram, gram|tri; mean [seed range].

n = 1000 (`r192-frontier-n1000.*`, with r184/r188/r190/r191 dirs):

| arm | temp | gram | tri | gram\|tri | without the check (r188/r190), gram\|tri |
|---|---|---|---|---|---|
| both w3 | 0.5 | 0.913 [0.877,0.945] | 0.044 [0.030,0.061] | 0.748 [0.621,0.922] | 0.633 |
| both w3 | 0.7 | 0.873 [0.850,0.890] | 0.100 [0.087,0.112] | 0.695 [0.659,0.733] | 0.621 |
| both w3 | 1.0 | 0.818 [0.790,0.842] | 0.197 [0.167,0.224] | 0.655 [0.593,0.729] | 0.583 |
| both w4 | 0.5 | 0.889 [0.868,0.904] | 0.125 [0.099,0.166] | 0.708 [0.633,0.749] | 0.516 |
| both w4 | 0.7 | 0.847 [0.820,0.860] | 0.218 [0.184,0.244] | 0.663 [0.635,0.710] | 0.543 |
| both w4 | 1.0 | 0.806 [0.796,0.824] | 0.338 [0.311,0.389] | 0.667 [0.627,0.705] | 0.552 |
| both w5 | 0.5 | 0.841 [0.780,0.867] | 0.277 [0.223,0.320] | 0.729 [0.681,0.800] | 0.507 |
| both w5 | 0.7 | 0.821 [0.781,0.846] | 0.366 [0.353,0.389] | 0.709 [0.659,0.749] | 0.498 |
| both w5 | 1.0 | 0.781 [0.742,0.820] | 0.461 [0.421,0.497] | 0.662 [0.635,0.720] | 0.531 |

Envelope (gram at fixed tri): tri .2 LSTM .968 vs both w4 .855 (margin .113; was .135 against w3); tri .3 LSTM .932 vs
both w5 .836 (margin .096; was .136); tri .4-.7 unchanged (GPT/LSTM lead; the checked arms do not reach tri .5).

n = 500 (`r192-frontier-n500.*`): gram|tri 0.67-0.78 (was 0.49-0.62). Envelope: TRELLIS both leads at tri .2 (.855 vs
TRELLIS w3 .830) and tri .3 (.821 vs .789; the neural curves start above tri .3 at n 500); tri .4 LSTM .885 vs both w5 .833
(margin .052; was .139).
n = 250 (`r192-frontier-n250.*`): gram|tri 0.66-0.81 (was 0.42-0.59). Envelope: both w4 leads at tri .3 (.815 vs .767,
no neural arm reaches down to .3); tri .4 only LSTM covers it (.891); both w5 at tri .40-.49 has gram .84-.79 against
LSTM's .90-.83 there.

Verdict against the target: gram|tri rises by .1 to .2 at every n and window, but stays at .66-.81 (target .80 reached
only by w3 at temp .5, n 250, mean .806, and with seed ranges reaching .92 at n 1000 w3 .5 on 4% of outputs). TRELLIS does
not lead the envelope at tri >= .3 where a neural curve is present; it leads at tri .3 at n 250 / 500 only where the
neural curves do not reach. The gap to the best neural arm at tri .3-.4 is cut by about a third (n 1000) to two thirds
(n 500).

## 7. Ablation: what the right-side read adds over closing the pairs and boundaries (n 1000, w5, 4 seeds)

Arm `"both": "pairs"`: the same boundary rule, and a new trigram allowed only over a stored pair, with no right-side
read (the strict pair gate the closed rounds already pointed to). `r192a-w5-frontier-n1000.jsonl`.

| w5, temp | none (r188/r190): tri, gram\|tri | pairs + boundaries: gram, tri, gram\|tri | both (pivot read): gram, tri, gram\|tri |
|---|---|---|---|
| 0.5 | 0.566, 0.507 | 0.780, 0.417, 0.633 | 0.841, 0.277, 0.729 |
| 0.7 | 0.637, 0.498 | 0.749, 0.512, 0.615 | 0.821, 0.366, 0.709 |
| 1.0 | 0.706, 0.531 | 0.695, 0.637, 0.589 | 0.781, 0.461, 0.662 |

About half of the gram|tri gain is the closed pairs and boundaries (+0.06 to +0.13), half the right-side read (+0.07 to
+0.10). On the gram-vs-tri frontier the read is worth less, because it also cuts novelty: at tri .42 the checked arm
interpolates to about .80 against .78 for pairs-only. (w3 / w4 of this ablation were still running at hand-back;
`logs/v3_r192_submit.sh 1000 r192a '"pairs"'` writes `r192a-frontier-n1000.*` when they finish.)

## 8. Round 193: two-sided Gibbs repair (pre-test: fails; not run)

Mechanism (opt-in arm key `"repair": true` on top of `"both"`; `Trellis3.gibbs_repair`, `Trellis3.fill_words`): after a
sentence is drawn, revisit each position left to right and redraw its word from the context tree given BOTH windows
(l1..lW and r1..rW, the same back-off as `_gap_fills`, vectorised in `fill_words`), among words that keep both
neighbouring pairs stored (boundary words at the ends); sweep until nothing changes (at most the sentence length);
a repaired sentence that is a stored sentence falls back to the draft. Close to round 171's sweep (two words each side,
"no better than trigram"); the differences are the full window, the strict pairs and boundaries, and the fixed point.

Pre-test (`diag/r193_pretest.py`, `r193_compare.py`; r192 n1000 "both" dumps at temp 0.7, seeds 13 17, the originals
and the repairs judged together; `r192-diag/r193_compare*.txt`):

| w, seed | original: gram, tri, gram\|tri | full repair, stored -> draft | became stored (full) | targeted repair (only words in a new trigram), stored -> draft |
|---|---|---|---|---|
| w3 s13 | .879, .099, .692 | .871, .123, .675 | 36% | .885, .069, .694 |
| w3 s17 | .890, .112, .659 | .869, .159, .614 | 32% | .897, .085, .672 |
| w4 s13 | .853, .244, .655 | .840, .254, .636 | 61% | .869, .198, .697 |
| w4 s17 | .860, .206, .710 | .849, .214, .702 | 54% | .870, .182, .751 |
| w5 s13 | .781, .389, .700 | .772, .397, .681 | 80% | .787, .359, .693 |
| w5 s17 | .819, .360, .659 | .820, .372, .661 | 78% | .821, .333, .645 |

Why it fails. Redrawing a word given 2W context attributes is dominated by the stored occurrences whose own context
matches most of those attributes, so the full repair walks the sentence back into memory (31-80% of repairs ARE stored
sentences, more with a wider window) and changes judged-good sentences as often as bad ones (70-90% of each); the new
sentences it does make are worse (raw gram|tri .43-.62 against .66-.71). The targeted version touches bad sentences 1.5 to 4
times as often as good ones, but it repairs them by removing the novelty (raw tri falls from .10-.39 to
.01-.09): on the frontier that is a lower temperature, and gram|tri of what stays new moves by -.03 to +.04. Both read
the same per-leaf, attribute-independent back-off that made the error, so a second look from the right adds no
information the left-to-right draw plus the pivot read did not already use; and class C's residue (coordination
repeats, selection, agreement beyond the window) is not a single-word junction error that one local redraw can see.
Neither variant was run at n 1000 / 4000. Instead the r192 "both" arm itself was launched at n 4000 (seeds 13 17,
w3/4/5 x .5/.7/1.0; `logs/v3_r193_both4k.sh`, dirs `r193-n4000-w*`, frontier `r193-frontier-n4000.*`).

Next candidate (not launched): stop generating word by word for novelty and make the novel step a whole committed
unit. The diagnosis says the failures sit at junctions between words whose roles the window does not fix; the content
tree already stores chunks (the arm uses units "words" only). Drawing chunk fillers ("units": "both") with the r192
check applied at chunk junctions would put every new trigram at a chunk boundary, where the parse says recombination
is licensed by a constituent, and leave within-chunk word order as stored. Round 177b found unstored pairs equally
common at all junction types, so the pre-test should be gram|tri of new trigrams at chunk junctions vs inside the
window-generated text, on dumps of the units "both" arm.
