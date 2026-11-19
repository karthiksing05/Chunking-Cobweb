# Round 214a: a two-axis judge (syntax and semantics) for TinyStories-style sentences

Code: `experiments/v3/diag/r214_judge2.py` (subcommands `build`, `eval`, `report`, `score`). Thresholds: `experiments/v3/diag/r214_judge2_thresholds.json`. Controls, raw scores (every sentence x model x prompt variant), full tables and a CLI demo: `experiments/v3/results/story_gen/r214/` (`controls.json`, `scores.json`, `report_tables.md`, `demo_out.tsv`). GPU jobs on A40 (4793952 eval, 4794093 CLI demo); models Qwen3-8B and phi-4, bf16, log-prob yes/no as in r209: p = P(yes)/(P(yes)+P(no)) at the answer token.

CLI: `r214_judge2.py score <sentences.txt> <out.tsv> [--models Qwen3-8B,phi-4]` (compute node, env chunks, `HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1`; `experiments/v3/results/story_gen/r214/run.sh` wraps this). The TSV has, per model, `<m>_syn_p`, `<m>_syn_pass`, `<m>_sem_p`, `<m>_sem_pass` (each model's own thresholds), plus the ensemble columns `syn_p`, `syn_pass`, `sem_p`, `sem_pass` (mean logit of the two models; with one model the ensemble columns are that model's own).

## Control sets (seed 214, 300 per class, built from held-out TinyStories sentences)

Source: the first 1500 stories of the seed-13 shuffle (the held-out stories of story_gen), sentences of 3-8 words whose words are all in the 250-word story_gen vocabulary (2188 distinct). NLTK `averaged_perceptron_tagger_eng` (compute node) tags each sentence in context; a word's pool tag is its majority tag over the held-out sentences (>= 60% of its uses, >= 3 uses). Items are paired: C1, C2 and C3 are derived from the same 300 base sentences. The dev half is the even indices, the test half the odd indices (150 each); prompt variant and threshold are chosen on dev and reported on test.

* C1 (syntax yes, semantics yes): the real sentences.
* C2full (syntax yes, semantics no): every swappable content word (noun, plural noun, adjective, verb) replaced by a different word of the same pool (same tag and inflection; function words and frame kept).
* C2one: the same with one content word swapped.
* C3scr (no, no): word order shuffled (always different from the source). C3adj: one adjacent swap of two words with different POS tags (items with no such pair are dropped, so n < 300).
* C4 (syntax no, semantics recoverable): agreement flips was/were, is/are, has/have in sentences without i/you/and ("they is not friends anymore"), drawn from all held-out sentences. Caveat: some are actually grammatical ("the fish were not happy anymore", plural fish), so C4 syntax passes are slightly overstated errors.

Two changes to the plain "swap with a random same-POS word" recipe were needed, because the plain recipe did not give the intended classes. (1) Verbs are swapped only inside an argument frame (hand-listed intransitive and monotransitive verb pools; verbs followed by "to" or lacking their object are not swapped; adverbs are not swapped at all, since their position is too free). Without this, the swaps often produced real syntax errors ("tom saw to put"), and the syntax judges rightly rejected them (C2 syntax pass only .2). Names and tagger errors are excluded from the pools; a/an is re-agreed after a swap. (2) The swap is anomaly-biased: nouns flip animacy (animate <-> inanimate), adjectives flip emotion <-> physical, and among 60 random draws the one with the most corpus-unattested bigrams touching a swapped slot is kept (bigram counts over all TinyStories sentences, no LM involved). Uniformly random swaps gave "the ball was not shiny anymore"-type sentences that are sensible, and every semantics judge passed 35-55% of them. Base sentences must have a swappable noun or verb, not only adjectives. Pools are small (20 adjectives, 24 nouns, 5 plural nouns, 18 verbs in two frames), so C2 is a narrow-vocabulary version of the colorless-green class.

## Prompts

Both: the yes/no field follows `Sentence: <s>\n<Field>:` and is scored on " yes"/" no"; the few-shot prefix is run once and its KV cache reused. Full text in the script (`SYN_HEAD_C`, `SYN_SHOTS_B`, `SEM_HEAD_B`, `SEM_SHOTS_B`).

* Syntax (chosen variant v5, field "Grammatical"): "You are a syntax checker, not a reader. Imagine every noun, verb and adjective replaced by an invented word: does the sentence frame still work as a complete, well-formed English sentence (articles before nouns, ... nothing missing, nothing in a scrambled order)? Meaning is irrelevant. A sentence in which a table eats rain, a rock smiles, or a cup is a car is fully grammatical, and so is a sentence of invented words in a grammatical frame. Answer yes whenever the frame is well-formed, however absurd the content." 24 shots, paired minimal contrasts: absurd-but-well-formed yes ("the car ate a happy tree", "the sky was very sad and the rock ran home", "the slithy toves did gyre and gimble in the wabe", "my pencil sang the quiet river", "the frabjous bandersnatch jumped over the cold milk") against the same words scrambled or with agreement errors (no).
* Semantics (chosen variant v2, field "Makes sense"): "does it describe something that could happen or exist in a children's story world (talking animals, moving toys and magic are fine when the sentence is a coherent event or description)? Answer no when the subject cannot do that action to that object, an object has a property it cannot have, or the words are an unrelated jumble. Small grammar slips do not matter here." 24 shots; sensible sentences (including ones with agreement slips, "they was very happy to see the cat") yes; absurd-but-well-formed and scrambled no.
* Other variants tried (all scored; see `report_tables.md`): v1 = first prompts (shorter, 20 shots), v2 syn = category-aware head + 24 shots, v3 = longer shots with many absurd-yes minimal pairs, v4/v5 = head/shot crosses; sem v3 = v2 + 6 shots. The first v1 pair already separated C1 from C3 but passed only .2-.5 of C2 on syntax: the judges treat nonsense as ungrammatical, so the worked examples of grammatical nonsense and the "replace content words by invented words" framing were what moved C2.

## Chosen judges and thresholds (picked on dev, threshold grid .01-.95; ensemble on a logit grid)

| judge | syntax variant, thr | semantics variant, thr |
|---|---|---|
| Qwen3-8B | v5, .01 | v2, .8 |
| phi-4 | v5, .1 | v2, .9 |
| ensemble (mean logit of the two) | logit > -5.25 | logit > 2.5 |

The syntax thresholds are low on purpose: with p > .5 the judges fail C2. At .01 and .1, C3scr stays below .03.

## Results on the test half (150 per class; rows = class, entries = share judged pass)

| judge | axis | C1 | C2full | C2one | C3scr | C3adj | C4 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | syn | 1.000 | .787 | .900 | .013 | .087 | .040 |
| Qwen3-8B | sem | .913 | .173 | .433 | .000 | .047 | .207 |
| phi-4 | syn | 1.000 | .860 | .953 | .013 | .147 | .027 |
| phi-4 | sem | .967 | .253 | .533 | .007 | .193 | .407 |
| ensemble | syn | 1.000 | .880 | .953 | .020 | .127 | .040 |
| ensemble | sem | .947 | .133 | .440 | .000 | .060 | .153 |

Targets (syntax: C1 >= .90, C2full >= .80, C3 <= .10; semantics: C1 >= .85, C2full <= .20):

* Semantics: met by the ensemble (.947 / .133) and by Qwen3-8B alone (.913 / .173); phi-4 alone misses C2full (.253 on test; its dev value was .147, so the dev pick did not transfer).
* Syntax: C1, C2full and the full-scramble C3 are met by the ensemble (1.000, .880, .020) and by phi-4 (1.000, .860, .013); Qwen3-8B alone just misses C2full (.787). The mild adjacent-swap C3adj is not met: .127 (ensemble), .147 (phi-4), .087 (Qwen3-8B, but with C2full below target). The two-model ensemble is the recommended judge, with this one residual: a single adjacent swap of different-POS words ("she a was very good girl", "then tim's friends to came the park") is passed by about 1 in 8 sentences. The syntax margin on dev was -.03 for the ensemble, driven by C3adj .127.

One-word swap (C2one): the syntax judge passes .95 (it is grammatical), the semantics judge passes .44 (ensemble; .43 Qwen3-8B, .53 phi-4). So a single swapped content word counts as a semantic failure in only about 56% of items; the rest are sensible by chance ("at the park sue thanked a big tree") or mildly odd. Treat a one-word swap as half a failure, not a clean negative; do not use it as a negative class at a .20 target.

## 2x2 confusions (ensemble; SY = syntax pass, SM = semantics pass; counts on the test half, 150 per class; full 300 in `report_tables.md`)

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- |
|---|---|---|---|---|---|
| C1 | 150 | 142 | 8 | 0 | 0 |
| C2full | 150 | 20 | 112 | 0 | 18 |
| C2one | 150 | 66 | 77 | 0 | 7 |
| C3scr | 150 | 0 | 3 | 0 | 147 |
| C3adj | 150 | 9 | 10 | 0 | 131 |
| C4 | 150 | see report_tables.md | | | |

(C4 test row, ensemble: syn pass .04, sem pass .15; on all 300: SY-SM+ 47, SY-SM- 244, SY+SM+ 6, SY+SM- 3.) Per-model confusions on test and on all 300:

Qwen3-8B (syn v5 .01, sem v2 .8), test half:

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- |
|---|---|---|---|---|---|
| C1 | 150 | 137 | 13 | 0 | 0 |
| C2full | 150 | 26 | 92 | 0 | 32 |
| C2one | 150 | 65 | 70 | 0 | 15 |
| C3scr | 150 | 0 | 2 | 0 | 148 |
| C3adj | 150 | 6 | 7 | 1 | 136 |
| C4 | 150 | 5 | 1 | 26 | 118 |

phi-4 (syn v5 .1, sem v2 .9), test half:

| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- |
|---|---|---|---|---|---|
| C1 | 150 | 145 | 5 | 0 | 0 |
| C2full | 150 | 38 | 91 | 0 | 21 |
| C2one | 150 | 79 | 64 | 1 | 6 |
| C3scr | 150 | 0 | 2 | 1 | 147 |
| C3adj | 150 | 11 | 11 | 18 | 110 |
| C4 | 150 | 3 | 1 | 58 | 88 |

Regimes for TRELLIS output: "meaningful" = SY+SM+, "grammatical but meaningless" = SY+SM-, "nonsense" = SY-SM-. The cells SY-SM+ (agreement errors with recoverable meaning, C4) are separable: C4 sits mostly in SY-SM+ for phi-4 (58/150) and in SY-SM- for Qwen3-8B (semantics .21), i.e. phi-4's semantics reads through agreement errors, Qwen3-8B's does not; the ensemble is in between (.15 semantics pass). Syntax is judged largely independently of meaning (C2full syntax .88 vs C3 .02-.13 with semantics failing in both).

## Five judged examples per class (first five of the test half; P(yes) syntax/semantics: Qwen3-8B; phi-4)

C1: "then tim's friends came to the park" 1.00/1.00; 1.00/1.00. "he told them about the big dog" 1.00/1.00; 1.00/1.00. "at the park sue found a big tree" 1.00/1.00; 1.00/1.00. "tim picked it up and took it home" 1.00/.97; 1.00/.96. "she was a very good girl" 1.00/1.00; 1.00/1.00.

C2full: "then tim's things came to the man" 1.00/.41; 1.00/.87 (a miss for phi-4 semantics). "he told them about the proud park" 1.00/.00; 1.00/.50. "at the bird sue thanked a surprised fish" .65/.00; .99/.16. "tim found it up and found it home" .27/.00; .93/.02. "she was a very careful box" 1.00/.00; 1.00/.16.

C2one: "then tim's friends came to the dog" 1.00/.22; .80/1.00. "he told them about the big food" 1.00/.27; 1.00/.88. "at the park sue thanked a big tree" 1.00/.99; 1.00/.96 (sensible by chance). "tim loved it up and took it home" .04/.00; 1.00/.09. "she was a very good park" .00/.00; .25/.04.

C3scr: "to tim's came the then friends park" .00/.00; .00/.02. "he the them about dog told big" .00/.00; .00/.01. "found a at park sue tree big the" .00/.00; .00/.00. "it it up took picked and tim home" .00/.00; .01/.00. "a very girl was she good" .00/.00; .01/.29.

C3adj: "then tim's friends to came the park" .00/.00; .00/.53. "told he them about the big dog" .00/.00; .00/.89. "at the park sue a found big tree" .00/.00; .00/.03. "tim picked it up and took home it" .00/.00; .01/.27. "she a was very good girl" .00/.01; .00/.71.

C4: "what are that" .00/.00; .00/.01. "the fish were not happy anymore" .97/1.00; .89/1.00 (grammatical, plural fish). "the cat were not sad anymore" .00/.07; .00/.98. (Examples 4 and 5 of C4 are in `report_tables.md`.)

## Caveats

* The p values are strongly bimodal (most at < .01 or > .99); the chosen syntax thresholds (.01, .1) sit in the low tail, so the result depends on Qwen3-8B and phi-4 giving non-zero mass to absurd-but-well-formed sentences. The ensemble thresholds were picked on 150 dev items per class and C2/C3adj move by .05-.10 between dev and test, so read the C3adj miss and the phi-4 semantics miss as the size of the selection noise, not as exact rates.
* C2 is built from a small word pool (see above), anomaly-biased by an n-gram rule and by animacy and emotion flips; real TRELLIS "grammatical but meaningless" output has a wider lexical range. Re-run `build` with a larger pool before drawing conclusions about the rate for other kinds of nonsense.
* C4 is rule-based and includes some grammatical items (plural fish); C3adj near-misses (a determiner moved) are the hardest class.
