# Round 209: can grammaticality .95+ be measured on TinyStories?

Code: `experiments/v3/diag/r209_judge.py` (job 4788083, A40). Raw scores per sentence per judge: `r209/r209_scores.json`; full tables: `r209/r209_tables.md`.
Judges: the cached original (Qwen2.5-3B-Instruct, prompt v1) and, for three larger local models (Qwen3-8B, Llama-3.1-8B base, phi-4 14B; no Qwen2.5-7B/14B-Instruct is cached locally, nothing downloaded), two prompts each.
* v1 = the original few-shot yes/no prompt (same text and shots as `t3_judge.py`), P(yes)/(P(yes)+P(no)) > .5.
* strict = longer instruction (complete sentence, odd meaning allowed; wrong order, missing/extra word, agreement error, or unfinished fails) and 16 shots.
Sets: real held-out refs (n=1000: 1200 sentences, seeds 13/17/19/23; n=4000: 600) with scrambled and content-swapped controls; distinct new generations at each family's best fair setting (n=1000 seeds 13+17: edge w7 t2.0, plain w6 t1.6, LSTM t0.8, GPT t0.8; n=4000 seed 13: edge w5 t2.0, LSTM t1.0, GPT t0.9).

## Real-sentence ceiling (share of real held-out sentences judged grammatical; controls = false-pass rate)

| judge | real n=1000 | real n=4000 | scrambled passes | content-swapped passes | >= .95? |
|---|---|---|---|---|---|
| 3B / v1 (original) | .891 | .905 | .107 | .058 | no |
| Qwen3-8B / v1 | .863 | .865 | .039 | .039 | no |
| Qwen3-8B / strict | .958 | .960 | .058 | .060 | yes |
| Llama-3.1-8B / v1 | .918 | .923 | .132 | .068 | no |
| Llama-3.1-8B / strict | .976 | .978 | .163 | .117 | yes |
| phi-4 / v1 | .914 | .918 | .089 | .052 | no |
| phi-4 / strict | .970 | .972 | .068 | .075 | yes |

The original prompt tops out at .86-.92 on every model, so the 3B ceiling is a prompt/data property, not a small-model one. The strict prompt lifts the ceiling to .96-.98 on all three models. Qwen3-8B strict and phi-4 strict do this without raising the false-pass rate on scrambled text above .07, so the gain is not just leniency. Llama strict passes 16% of scrambled sentences, so its .976 is partly leniency.
What the strict judges still fail among real sentences (Llama strict, first ones): "said the car", "and the dog", "ben and lily said", "tom and lily said", "lily and sam asked", "go tom go", "do you want some". These are dialogue-tag fragments from sentence splitting around quotes; they are unfinished as strings, so a ceiling of 1.0 is not reachable by a per-sentence judge and .95 is only just reachable.

## Families under each judge (mean over seeds; gram = share of distinct new judged grammatical; norm = gram / real ceiling of that judge; tri = has an unstored trigram)

n=1000 (seeds 13, 17):

| judge (ceiling) | family | gram | gram norm | gram\|tri | gram\|tri norm | fair/draw |
|---|---|---|---|---|---|---|
| 3B v1 (.891) | edge w7 t2.0 | .541 | .608 | .482 | .541 | .337 |
| | plain w6 t1.6 | .515 | .578 | .466 | .523 | .323 |
| | LSTM t0.8 | .591 | .663 | .503 | .564 | .337 |
| | GPT t0.8 | .556 | .624 | .487 | .547 | .336 |
| Qwen3-8B v1 (.863) | edge | .485 | .562 | .422 | .489 | .295 |
| | plain | .391 | .453 | .334 | .387 | .231 |
| | LSTM | .522 | .606 | .426 | .494 | .286 |
| | GPT | .508 | .589 | .430 | .498 | .297 |
| Qwen3-8B strict (.958) | edge | .581 | .606 | .520 | .543 | .363 |
| | plain | .469 | .489 | .407 | .425 | .282 |
| | LSTM | .615 | .642 | .522 | .545 | .350 |
| | GPT | .597 | .623 | .523 | .546 | .361 |
| Llama v1 (.918) | edge | .597 | .650 | .542 | .590 | .379 |
| | plain | .551 | .600 | .506 | .551 | .350 |
| | LSTM | .611 | .666 | .526 | .573 | .352 |
| | GPT | .583 | .635 | .515 | .561 | .356 |
| Llama strict (.976) | edge | .659 | .676 | .611 | .626 | .427 |
| | plain | .607 | .622 | .564 | .578 | .390 |
| | LSTM | .659 | .676 | .576 | .591 | .386 |
| | GPT | .621 | .637 | .555 | .569 | .383 |
| phi-4 v1 (.914) | edge | .534 | .584 | .478 | .522 | .334 |
| | plain | .434 | .475 | .374 | .409 | .258 |
| | LSTM | .580 | .635 | .486 | .532 | .326 |
| | GPT | .561 | .614 | .484 | .530 | .334 |
| phi-4 strict (.970) | edge | .579 | .597 | .521 | .537 | .364 |
| | plain | .459 | .474 | .397 | .409 | .274 |
| | LSTM | .605 | .624 | .511 | .527 | .342 |
| | GPT | .582 | .600 | .510 | .526 | .352 |

n=4000 (seed 13; edge w5 t2.0 / LSTM t1.0 / GPT t0.9):

| judge (ceiling) | edge gram (norm) | LSTM | GPT | edge gram\|tri (norm) | LSTM | GPT | fair/draw edge / LSTM / GPT |
|---|---|---|---|---|---|---|---|
| 3B v1 (.905) | .490 (.541) | .642 (.710) | .678 (.749) | .411 (.455) | .546 (.603) | .587 (.648) | .318 / .383 / .372 |
| Qwen3-8B v1 (.865) | .400 (.463) | .582 (.673) | .628 (.727) | .311 (.360) | .487 (.563) | .539 (.624) | .241 / .342 / .342 |
| Qwen3-8B strict (.960) | .539 (.561) | .715 (.745) | .754 (.785) | .455 (.474) | .630 (.656) | .674 (.702) | .352 / .442 / .427 |
| Llama v1 (.923) | .538 (.583) | .671 (.727) | .696 (.754) | .459 (.497) | .581 (.629) | .615 (.666) | .355 / .407 / .390 |
| Llama strict (.978) | .639 (.653) | .735 (.752) | .776 (.793) | .572 (.584) | .655 (.670) | .698 (.714) | .442 / .460 / .443 |
| phi-4 v1 (.918) | .481 (.523) | .629 (.685) | .686 (.747) | .398 (.433) | .539 (.587) | .599 (.652) | .307 / .378 / .379 |
| phi-4 strict (.972) | .539 (.555) | .712 (.732) | .748 (.770) | .451 (.464) | .618 (.636) | .656 (.675) | .349 / .434 / .416 |

## Does the family ranking change?

* n=4000: no. Every judge gives GPT > LSTM > edge on gram|tri, and LSTM >= GPT > edge on fair/draw. The edge deficit is .13-.19 on gram|tri raw under every judge, so it is not a 3B artefact.
* n=1000: the ordering of edge, LSTM and GPT is within noise under every judge and flips between judges (fair/draw: edge first under Qwen3 strict, Llama v1/strict, phi-4 strict; GPT or LSTM first under 3B, Qwen3 v1, phi-4 v1). The spread among the three is at most .03 fair/draw. Plain TRELLIS w6 is last under every judge except Llama v1 and strict (3rd). Under Llama strict edge leads on fair (.427 vs .386/.383) and gram|tri (.611 vs .576/.555). That is a single-judge result with the leakiest judge; the 3B, Qwen3 and phi-4 do not reproduce a lead on gram|tri.
* Ceiling normalisation helps all families the same and does not close any gap: best normalised gram|tri at n=1000 is .626 (edge, Llama strict) and at n=4000 .714 (GPT, Llama strict). No generator exceeds about .79 normalised gram under any judge.

## Agreement with the 3B judge (Cohen's kappa)

| judge | kappa on 13,355 pooled distinct new sentences | kappa on real + controls | raw agreement (generated) |
|---|---|---|---|
| Qwen3-8B v1 | .652 | .801 | .825 |
| Qwen3-8B strict | .627 | .807 | .816 |
| Llama-3.1-8B v1 | .662 | .791 | .833 |
| Llama-3.1-8B strict | .602 | .744 | .805 |
| phi-4 v1 | .633 | .774 | .817 |
| phi-4 strict | .596 | .790 | .800 |

Kappa .60-.66 on generated text: the judges agree on about 80-83% of sentences, so individual labels are noisy for borderline outputs, though family-level shares move together.

## Verdict on .95

Real sentences reach >= .95 only under the strict prompt (Qwen3-8B .958/.960, phi-4 .970/.972, Llama .976/.978) and stay at .86-.92 under the original prompt for all four models. So .95 is measurable only with a strict prompt, and only barely: the ceiling is .96-.98, which leaves 2-4 points of headroom and the residual real failures are sentence-splitting fragments. Judge disagreement (kappa about .6) is larger than that headroom. A generator at .95 raw gram would be at .97-.99 normalised, indistinguishable from real text; the current best families sit at gram .58-.78 raw (.60-.79 normalised), so the gap to .95 is 17-37 points under every judge, far above measurement noise. Reaching .95 is not blocked by the instrument; it is blocked by the generators (and TRELLIS is the furthest at n=4000: .54-.64 raw gram versus .71-.78 for LSTM/GPT).

## Parse side (no new runs; `story_parse_eval/r203_parse.md`, `r198_gold_ceiling.md`; 300 held-out sentences, seeds 13/17/19)

Reference = NLTK shallow phrases (NP/PP/VP/AP/VG, >= 2 words, not the whole sentence). It is a chunker output, not a treebank, so F1 against it is 1.0 only for the reference itself; the realistic ceiling is set by how the reference disagrees with human trees, which was not measured. Right-branching (no learning) already scores F1 .466 and random binary .275-.280, so the usable range is roughly .28 to .47 for learned methods, not to 1.0.

| method (4000 learned, mean of 3 seeds) | F1 | F1+sv | crossing | cross-seed F1 | decisive (gold .982) | consistency (gold .116) |
|---|---|---|---|---|---|---|
| TRELLIS default, posterior | .362 | .468 | .206 | .801 | .873 | .272 |
| TRELLIS default, chosen tree | .367 | .478 | .253 | .807 | .869 | .315 |
| TRELLIS parts, posterior | .357 | .466 | .209 | .799 | .876 | .288 |
| TRELLIS parts, chosen tree | .369 | .475 | .250 | .798 | .865 | .332 |
| BPE x1 | .369 | .417 | .301 | .839 | .923 | .489 |
| right-branching | .466 | .397 | .249 | 1.000 | 1.000 | 0.000 |

Distance of .95 from the current numbers (4000 learned): TRELLIS F1 .357-.369 is .58-.59 below .95; BPE F1 .369 is .58 below; right-branching F1 .466 is .48 below; cross-seed F1 .80 (TRELLIS) and .84 (BPE) are .11-.15 below .95; decisive .87-.88 (TRELLIS) is .07-.08 below .95, while BPE .923 is .03 below (BPE and right-branching are decisive mainly because they bracket the same strings the same way). At 250 / 1000 learned, TRELLIS parts post F1 is .353 / .352 and cross-seed F1 .594 / .747. Accuracy measures that exist: bracket F1, F1+sv, crossing rate, cross-seed F1, decisive, consistency (the last two are distances to gold, not accuracies; consistency rewards over-bracketing). Only cross-seed F1 and decisive are within about .1 of .95; F1 against the shallow-phrase reference is nowhere near, and for a method that is not told the chunker's conventions, .95 F1 against it is not an achievable target (right-branching, which knows nothing, is already at .466).
