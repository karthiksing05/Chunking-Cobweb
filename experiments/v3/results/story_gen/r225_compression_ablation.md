# Round 225: compression on language, and what the chunks and the concepts do

Code: `src/trellis2/trellis3_r225.py` (`Trellis3R225(concepts_off, concepts_off_parse)`, subclass of `Trellis3R204`; `trellis3.py` untouched), drivers `experiments/v3/diag/r225_{common,trellis,lms,story,gen_report,analyse}.py`, launcher `r225_launch.sh`. Raw: `r225_raw/` (one JSON per arm/seed/n and per reference-model seed/n), generation dumps and judge scores in `r225/`. Tables below are `r225_tables.md` (generated) and `r225/gen_table.md`. Figure-ready data: `r225_bits_figure.tsv` (mean over seeds: n, model, bits per sentence / token / word, sd over seeds) and `r225_bits_curve.tsv` (per seed, with the standard error over the 1,000 test sentences and the selected parameter). Seeds 13 and 17, n = 250, 500, 1000, 2000, 4000. Jobs: trellis 4808387-4808428 and 4808403-4808412 (n=1000 parse arms), LMs 4808429-4808438, generation 4808439-4808449, judge 4808451, report 4808452, analysis 4810023.

## What was measured, and the conventions

**Test sentences.** The first 1,000 kept sentences (3-8 words, 250-word vocabulary) of the seed's 1,500 held-out stories (`r215_common.stream`, the r212/r215/r216 split). 10% (n=250) to 36% (n=4000) of them also occur in the training stream (TinyStories repeats short sentences); the tables report all test sentences and a second table gives the others only. A 300-sentence dev set (the LM early-stopping stories of r212) selects each free parameter: TRELLIS's eps floor, BPE's merge count k, the best n-gram, the LM architecture. Nothing is selected on the test sentences.

**Code length** = -log2 P(sentence) as a sequence of vocabulary words plus the end, bits per sentence (the table headline), per token (words + end) and per word.

- **TRELLIS** follows the chess convention (`chess_compress_gen.py`: "the code is the sequence of choices generation makes to produce exactly that item, each costing -log2 of its probability; the model is not charged"). The probability is the one the generator samples from (r215b `Gen.score`: edge-exact, pair gate, window per n as in r212, temperature 1): at each step the gated mass of the true outcome (word, ends-here) over the gated mass of all outcomes. The inside score logZ is not used, because it is not a normalised distribution over sentences. Outcomes the generator gives no mass (unseen word or context, or the pair gate) need a floor: p -> (p + eps) / (1 + eps K), K = 500 outcomes, so every sentence has a valid code. eps is chosen on dev from {1e-1 ... 1e-8}; it was 1e-3 in 18 of 20 cells and 1e-4 for concepts-off at n=4000. So the TRELLIS number is a code for the generator's distribution with a uniform escape, not a tuned language model.
- **BPE + unigram** (chess convention): word-level BPE merges the most frequent adjacent item pair (count >= 2) up to k merges (k in 0, 10, 25, 50, 100, 200, 400, 800, 1600; chosen on dev), a sentence is rewritten by the merges in order, items are coded by their training usage counts with Laplace +1 over 250 words + k chunks + a stop item. k=0 is the word unigram.
- **n-grams**: Witten-Bell 2/3/5 (the repo baseline, `r215_lms.WB`) and interpolated Kneser-Ney 2-5 (absolute discount n1/(n1+2n2) per order); the table shows the Witten-Bell trigram, the KN trigram and the dev-best of all seven.
- **LSTM, GPT**: the r212/r215 grid and early stopping (`t3_lm._fit`, retrained here, since no checkpoints were saved: r212 jobs kept only samples and r215 kept only log-probs of its own sets). Scored on the test sentences with the PAD and BOS logits masked (renormalised over the 250 words + end).
- **Model cost** is not in any code length (as in chess). Beside it: BPE dictionary bits (words x log2 250 per merged chunk) 2.0k (n=250) to 43k (n=4000); LSTM 0.2-1.2M and GPT 0.23-1.7M parameters; TRELLIS stores 8.3k records at n=250 and 102k at n=4000 (80% composites; 21.5k at n=4000 with chunks off). A two-part code that charged these would only move TRELLIS and the neural models further from BPE; the table below is the code-only comparison the chess section uses.

## Result 1: language compression

Mean over seeds 13 and 17 (bits per sentence; the uniform code of 51.4 bits is 6.45 tokens of log2 251 each, so test sentences average about 5.5 words).

| model | n=250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| uniform over 251 | 51.4 | 51.4 | 51.4 | 51.4 | 51.4 |
| word unigram | 42.6 | 42.3 | 42.1 | 42.0 | 41.9 |
| BPE vocabulary + unigram code | 38.0 | 35.7 | 33.1 | 31.0 | 28.7 |
| **TRELLIS (generator code)** | **35.7** | **33.2** | **30.8** | **28.7** | **26.6** |
| Witten-Bell trigram | 31.3 | 28.5 | 25.7 | 23.8 | 22.2 |
| Kneser-Ney trigram | 30.3 | 27.7 | 25.1 | 23.3 | 21.7 |
| GPT | 30.2 | 27.3 | 24.4 | 22.2 | 20.6 |
| LSTM | 29.8 | 26.5 | 23.5 | 21.5 | 20.0 |

- TRELLIS beats BPE with a unigram code at every n (by 2.3, 2.5, 2.3, 2.3, 2.1 bits per sentence; the paired standard error over test sentences is about 0.3, and both seeds agree). It beats the word unigram by 7-15 bits.
- TRELLIS is worse than every n-gram, the LSTM and the GPT at every n, by about 5 bits per sentence against the trigram and 6-7 against the neural models. The gap does not close with n (5.4 bits at n=250, 4.9 at n=4000 against KN; 5.9 and 6.6 against the LSTM). In bits per token: TRELLIS 5.53 -> 4.12, KN trigram 4.70 -> 3.37, LSTM 4.61 -> 3.10.
- So the paper can say that TRELLIS's chunk-and-concept memory codes held-out language shorter than a BPE vocabulary of chunks. It cannot say that TRELLIS compresses language better than a trigram or a small neural model. The generator was built to sample, not to predict: at temperature 1 with a pair gate it spreads mass in ways a calibrated predictor would not, and a uniform escape stands in for everything unseen. This is a measurement of that distribution as it is, with no refit.
- The ranking is stable on sentences the model has not seen in training (second table in `r225_tables.md`): BPE 40.1-35.6, TRELLIS 38.6-34.2, n-gram 32.6-27.5, LSTM 32.0-24.9 bits.

## Result 2: ablations at n = 1000 (seeds 13 and 17; bits at every n)

Arms: **full** (the generation recipe, configuration C1/C3: posterior storage, window 7, edge-exact reads, pair gate). **chunks off**: `store="none"`, no composite is stored (5.4k records instead of 29.6k; the content tree stays empty; the parser still runs on the context tree alone; the content-learning half of the recipe is gone). **concepts off** (generator reads): every read of the context tree uses a record's own leaf only, with back-off straight to the root, in the generator, in its score and in the pair gate (the gate then passes only pairs stored in the stream); learning and the parser are the recipe's. **both off** = chunks off + concepts-off reads. **concepts off (reads and parser)**: also the parser's gain is taken at the leaf only, so learning stores different chunks. Generation: the r218 pipeline, 1,600 draws per seed and temperature (t = 1.2, 2.0, 3.0; t = 2.0 is the r212 setting), judge2 (Qwen3-8B + phi-4). **Fair counts here are distinct new sentences passing the judge per attempted draw (syntax; syntax + semantics); they do not require an unstored trigram, so they are not the paper's Table values** (the same pipeline's full arm reads .54 and .36 at t=2.0). Parse metrics: the r216 score `gen_lp_gate` and `logZ_pw`, thresholds from Gcal (P(grammatical) = .9), 300 test sentences per set; the full arm reproduces r216's n=1000 sentence scores and logZ exactly (difference 0.0, both seeds).

| arm | held-out bits/sentence | fair syntax t1.2 / t2.0 / t3.0 | fair syn+sem t1.2 / t2.0 / t3.0 | content-swap sentences accepted (C2full / C2one) | AUC C2full / C3adj / C4v (generator score) | tree F1 vs gold (right-branching .427) |
|---|---|---|---|---|---|---|
| full | 30.8±0.3 | .510 / .539 / .486 | .373 / .363 / .297 | .708 / .845 | .748 / .779 / .706 | .347 |
| chunks off | 30.8±0.1 | .511 / .520 / .478 | .380 / .347 / .269 | .707 / .818 | .753 / .786 / .698 | .383 |
| concepts off (reads) | 28.8±0.2 | .462 / .496 / .448 | .309 / .309 / .240 | .573 / .822 | .815 / .846 / .733 | .347 (parser unchanged) |
| both off | 28.6±0.1 | .487 / .512 / .431 | .330 / .311 / .226 | .575 / .822 | .816 / .848 / .736 | .383 |
| concepts off (reads + parser) | 28.8±0.2 | .463 / .475 / .428 | .305 / .293 / .240 | .575 / .823 | .814 / .845 / .733 | .291 (logZ AUC = .500: no signal) |

Held-out bits at every n for the arms (mean over seeds): full 35.7, 33.2, 30.8, 28.7, 26.6; chunks off 35.5, 33.5, 30.8, 28.7, 26.7; concepts off 33.8, 31.2, 28.8, 26.9, 24.9; both off 33.7, 31.1, 28.6, 26.7, 24.7.

What the ablations show, stated plainly:

1. **Chunks do not drive any language result measured here.** Removing every stored composite leaves the held-out code length unchanged at every n (differences within 0.3 bits, within the sd over seeds), leaves generation unchanged within noise (fair syntax at t=2.0: .539 full against .520 without chunks, the full arm ahead in both seeds by .018 and .020 but behind at t=1.2 and the semantics count at t=1.2; fair syntax+semantics .363 against .347), and does not change content-swap acceptance or AUC. On the parse side the chunk-free parser is not worse: tree F1 against the NLTK shallow-phrase gold is .383 against .347, both below right-branching (.427), and the generator-score AUCs match. The one place chunks show is the cost of memory: they add 24k stored composites at n=1000 and do nothing for bits or generation. In this recipe the generator reads words and their windows; chunks reach the generator only through the context tree they sit in, and removing them removes nothing it uses. The chess compression and recall results are untouched by this; the language claim "chunks drive the results" is not supported.
2. **Concepts help generation and hurt bits and content-swap rejection.** Removing the concept path from the generator's reads lowers the fair counts (syntax at t=2.0 .539 -> .496, syntax+semantics .363 -> .309; lower in both seeds at every temperature), which supports "concepts (generalisation across contexts) are what let the generator write new grammatical sentences": with leaf-only reads 89% of draws are new (against 81%) but fewer pass the judge. The same removal shortens the held-out code by about 2 bits at every n and raises the generator score's content-swap AUC from .75 to .82, with swapped sentences accepted .57 against .71 of the time (real sentences accepted at the same rate, .91 against .90): concepts are what make the generator accept a syntactically fine sentence with swapped content words. Concepts make the model more permissive, which generation rewards and prediction does not.
3. **Concepts in the parser are necessary for it to parse** (reads + parser off: logZ AUC .500 on every class, F1 .291, 31,950 stored composites with uniform weights); that is the parser's gain being clipped to 0 at every leaf, so it is the expected consequence rather than a finding about language.
4. Both off behaves like concepts off (bits 28.6, fair syntax .512 at t=2.0, above concepts off alone at t=2.0 by .016 but the semantics count equal), consistent with chunks contributing nothing and concepts contributing the generation gain.
5. Temperature matters for this reading: the ablated models were not retuned (t = 2.0 was selected for the full arm in r205); the pattern, concepts-off below full and chunks-off level with full, holds at all three temperatures for syntax, with chunks-off marginally above full at t=1.2 for syntax+semantics. Two seeds only, 1,600 draws each; the per-seed fair-count differences for chunks are .02, for concepts .02 to .07, so none of the generation differences between full and chunks-off should be read as an effect.

## Not done / limits

- Generation and parse ablations are at n=1000 only; n=4000 has bits for all arms (generation at 4000 would need its own temperature selection and judge round).
- Two seeds (13, 17); no CI beyond the paired standard error over 1,000 test sentences (per-seed table in `r225_tables.md`) and the sd over two seeds.
- TRELLIS's code is the generator's gated distribution with a uniform escape (eps on dev); a model fit for prediction (e.g. mixing it with a lower-order back-off) would code shorter, and was not tried. The model cost is reported beside, not added.
- Concepts-off is leaf-only reads (concepts as ancestors of the leaf removed); it keeps the tree as a clustering of records. The pair gate under concepts-off reduces to stored pairs.
- The generation counts use the judge pipeline of r218 and a denominator of 1,600 attempted draws per seed (a draw that returned nothing counts as failed; seed 17 returned 1,544 sentences in every arm).
- No process is left running by this round.


---

# Appendix A: generated tables (r225_tables.md)


### Held-out code length: bits per sentence (mean over test sentences, then mean ± sd over the seeds)

| model | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|
| uniform over 251 | 51.4±0.2 | 51.4±0.2 | 51.4±0.2 | 51.4±0.2 | 51.4±0.2 |
| word unigram (BPE k=0) | 42.6±0.2 | 42.3±0.4 | 42.1±0.3 | 42.0±0.3 | 41.9±0.3 |
| BPE vocabulary + unigram code (k on dev) | 38.0±0.1 | 35.7±0.3 | 33.1±0.1 | 31.0±0.1 | 28.7±0.2 |
| Witten-Bell trigram (repo baseline) | 31.3±0.2 | 28.5±0.5 | 25.7±0.2 | 23.8±0.2 | 22.2±0.3 |
| Kneser-Ney trigram | 30.3±0.1 | 27.7±0.5 | 25.1±0.1 | 23.3±0.2 | 21.7±0.3 |
| best n-gram on dev (WB 2/3/5, KN 2-5) | 30.3±0.1 | 27.7±0.5 | 25.1±0.1 | 23.3±0.2 | 21.6±0.3 |
| LSTM | 29.8±0.1 | 26.5±0.6 | 23.5±0.2 | 21.5±0.0 | 20.0±0.2 |
| GPT | 30.2±0.0 | 27.3±0.8 | 24.4±0.3 | 22.2±0.2 | 20.6±0.3 |
| TRELLIS (generator code) | 35.7±0.2 | 33.2±0.0 | 30.8±0.3 | 28.7±0.1 | 26.6±0.0 |
| TRELLIS, chunks off | 35.5±0.1 | 33.5±0.2 | 30.8±0.1 | 28.7±0.1 | 26.7±0.1 |
| TRELLIS, concepts off (generator reads) | 33.8±0.1 | 31.2±0.0 | 28.8±0.2 | 26.9±0.2 | 24.9±0.0 |
| TRELLIS, chunks and concepts off | 33.7±0.1 | 31.1±0.0 | 28.6±0.1 | 26.7±0.2 | 24.7±0.1 |
| TRELLIS, concepts off (reads and parser) | - | - | 28.8±0.2 | - | - |

### Held-out code length: bits per token (words + end; corpus ratio, mean ± sd over seeds)

| model | n=250 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|
| uniform over 251 | 7.97±0.00 | 7.97±0.00 | 7.97±0.00 | 7.97±0.00 | 7.97±0.00 |
| word unigram (BPE k=0) | 6.60±0.01 | 6.55±0.02 | 6.52±0.02 | 6.51±0.01 | 6.49±0.02 |
| BPE vocabulary + unigram code (k on dev) | 5.89±0.04 | 5.53±0.02 | 5.13±0.04 | 4.80±0.00 | 4.44±0.01 |
| Witten-Bell trigram (repo baseline) | 4.85±0.01 | 4.41±0.06 | 3.99±0.01 | 3.68±0.02 | 3.44±0.04 |
| Kneser-Ney trigram | 4.70±0.01 | 4.29±0.05 | 3.89±0.00 | 3.61±0.01 | 3.37±0.03 |
| best n-gram on dev (WB 2/3/5, KN 2-5) | 4.70±0.01 | 4.29±0.05 | 3.89±0.00 | 3.61±0.01 | 3.34±0.02 |
| LSTM | 4.61±0.03 | 4.11±0.07 | 3.65±0.01 | 3.32±0.01 | 3.10±0.01 |
| GPT | 4.68±0.02 | 4.23±0.10 | 3.78±0.04 | 3.45±0.01 | 3.19±0.02 |
| TRELLIS (generator code) | 5.53±0.00 | 5.14±0.03 | 4.77±0.06 | 4.45±0.04 | 4.12±0.02 |
| TRELLIS, chunks off | 5.51±0.01 | 5.18±0.00 | 4.77±0.04 | 4.45±0.04 | 4.13±0.00 |
| TRELLIS, concepts off (generator reads) | 5.24±0.03 | 4.84±0.03 | 4.46±0.05 | 4.16±0.05 | 3.86±0.01 |
| TRELLIS, chunks and concepts off | 5.22±0.03 | 4.81±0.03 | 4.43±0.04 | 4.14±0.05 | 3.83±0.01 |
| TRELLIS, concepts off (reads and parser) | - | - | 4.46±0.05 | - | - |

### Per seed (bits per sentence) and the paired difference to TRELLIS over the same test sentences (model minus TRELLIS; positive = TRELLIS shorter; ± = s.e. over the test sentences)

| n | seed | model | bits/sentence | model - TRELLIS | selected |
|---|---|---|---|---|---|
| 250 | 13 | uniform over 251 | 51.27 | +15.71 ± 0.37 |  |
| 250 | 13 | word unigram (BPE k=0) | 42.41 | +6.85 ± 0.34 | k=0 |
| 250 | 13 | BPE vocabulary + unigram code (k on dev) | 38.05 | +2.49 ± 0.27 | k=100 |
| 250 | 13 | Witten-Bell trigram (repo baseline) | 31.12 | -4.45 ± 0.23 |  |
| 250 | 13 | Kneser-Ney trigram | 30.25 | -5.31 ± 0.22 |  |
| 250 | 13 | best n-gram on dev (WB 2/3/5, KN 2-5) | 30.25 | -5.31 ± 0.22 | kn3 |
| 250 | 13 | LSTM | 29.83 | -5.74 ± 0.27 | {"layers": 1, "d": 256, "drop": 0.3, "epochs": 12}, 656125 params |
| 250 | 13 | GPT | 30.19 | -5.37 ± 0.24 | {"layers": 2, "d": 256, "drop": 0.1, "epochs": 7}, 1712893 params |
| 250 | 13 | TRELLIS (generator code) | 35.56 | - | eps=0.001 |
| 250 | 13 | TRELLIS, chunks off | 35.46 | -0.10 ± 0.10 | eps=0.001 |
| 250 | 13 | TRELLIS, concepts off (generator reads) | 33.87 | -1.69 ± 0.13 | eps=0.001 |
| 250 | 13 | TRELLIS, chunks and concepts off | 33.74 | -1.82 ± 0.13 | eps=0.001 |
| 250 | 17 | uniform over 251 | 51.62 | +15.79 ± 0.41 |  |
| 250 | 17 | word unigram (BPE k=0) | 42.76 | +6.92 ± 0.37 | k=0 |
| 250 | 17 | BPE vocabulary + unigram code (k on dev) | 37.95 | +2.12 ± 0.29 | k=100 |
| 250 | 17 | Witten-Bell trigram (repo baseline) | 31.46 | -4.38 ± 0.22 |  |
| 250 | 17 | Kneser-Ney trigram | 30.37 | -5.46 ± 0.21 |  |
| 250 | 17 | best n-gram on dev (WB 2/3/5, KN 2-5) | 30.37 | -5.46 ± 0.21 | kn3 |
| 250 | 17 | LSTM | 29.71 | -6.12 ± 0.26 | {"layers": 1, "d": 128, "drop": 0.3, "epochs": 26}, 197117 params |
| 250 | 17 | GPT | 30.22 | -5.61 ± 0.23 | {"layers": 2, "d": 128, "drop": 0.1, "epochs": 10}, 463357 params |
| 250 | 17 | TRELLIS (generator code) | 35.83 | - | eps=0.001 |
| 250 | 17 | TRELLIS, chunks off | 35.61 | -0.22 ± 0.12 | eps=0.001 |
| 250 | 17 | TRELLIS, concepts off (generator reads) | 33.79 | -2.05 ± 0.14 | eps=0.001 |
| 250 | 17 | TRELLIS, chunks and concepts off | 33.67 | -2.16 ± 0.14 | eps=0.001 |
| 500 | 13 | uniform over 251 | 51.27 | +18.07 ± 0.39 |  |
| 500 | 13 | word unigram (BPE k=0) | 42.04 | +8.84 ± 0.35 | k=0 |
| 500 | 13 | BPE vocabulary + unigram code (k on dev) | 35.49 | +2.29 ± 0.27 | k=200 |
| 500 | 13 | Witten-Bell trigram (repo baseline) | 28.12 | -5.08 ± 0.23 |  |
| 500 | 13 | Kneser-Ney trigram | 27.38 | -5.82 ± 0.22 |  |
| 500 | 13 | best n-gram on dev (WB 2/3/5, KN 2-5) | 27.38 | -5.82 ± 0.22 | kn3 |
| 500 | 13 | LSTM | 26.09 | -7.11 ± 0.26 | {"layers": 1, "d": 256, "drop": 0.1, "epochs": 9}, 656125 params |
| 500 | 13 | GPT | 26.73 | -6.47 ± 0.25 | {"layers": 4, "d": 128, "drop": 0.3, "epochs": 10}, 859901 params |
| 500 | 13 | TRELLIS (generator code) | 33.20 | - | eps=0.001 |
| 500 | 13 | TRELLIS, chunks off | 33.33 | +0.13 ± 0.11 | eps=0.001 |
| 500 | 13 | TRELLIS, concepts off (generator reads) | 31.23 | -1.97 ± 0.14 | eps=0.001 |
| 500 | 13 | TRELLIS, chunks and concepts off | 31.07 | -2.13 ± 0.15 | eps=0.001 |
| 500 | 17 | uniform over 251 | 51.62 | +18.43 ± 0.41 |  |
| 500 | 17 | word unigram (BPE k=0) | 42.56 | +9.36 ± 0.37 | k=0 |
| 500 | 17 | BPE vocabulary + unigram code (k on dev) | 35.87 | +2.68 ± 0.28 | k=200 |
| 500 | 17 | Witten-Bell trigram (repo baseline) | 28.86 | -4.33 ± 0.23 |  |
| 500 | 17 | Kneser-Ney trigram | 28.03 | -5.17 ± 0.23 |  |
| 500 | 17 | best n-gram on dev (WB 2/3/5, KN 2-5) | 28.03 | -5.17 ± 0.23 | kn3 |
| 500 | 17 | LSTM | 26.92 | -6.27 ± 0.28 | {"layers": 2, "d": 256, "drop": 0.1, "epochs": 13}, 1182461 params |
| 500 | 17 | GPT | 27.83 | -5.36 ± 0.25 | {"layers": 2, "d": 128, "drop": 0.1, "epochs": 8}, 463357 params |
| 500 | 17 | TRELLIS (generator code) | 33.20 | - | eps=0.001 |
| 500 | 17 | TRELLIS, chunks off | 33.57 | +0.38 ± 0.12 | eps=0.001 |
| 500 | 17 | TRELLIS, concepts off (generator reads) | 31.19 | -2.01 ± 0.15 | eps=0.001 |
| 500 | 17 | TRELLIS, chunks and concepts off | 31.05 | -2.14 ± 0.15 | eps=0.001 |
| 1000 | 13 | uniform over 251 | 51.27 | +20.30 ± 0.38 |  |
| 1000 | 13 | word unigram (BPE k=0) | 41.89 | +10.91 ± 0.35 | k=0 |
| 1000 | 13 | BPE vocabulary + unigram code (k on dev) | 33.17 | +2.20 ± 0.28 | k=400 |
| 1000 | 13 | Witten-Bell trigram (repo baseline) | 25.61 | -5.36 ± 0.23 |  |
| 1000 | 13 | Kneser-Ney trigram | 25.01 | -5.96 ± 0.22 |  |
| 1000 | 13 | best n-gram on dev (WB 2/3/5, KN 2-5) | 25.01 | -5.96 ± 0.22 | kn3 |
| 1000 | 13 | LSTM | 23.39 | -7.58 ± 0.25 | {"layers": 1, "d": 256, "drop": 0.1, "epochs": 7}, 656125 params |
| 1000 | 13 | GPT | 24.18 | -6.80 ± 0.25 | {"layers": 4, "d": 128, "drop": 0.3, "epochs": 7}, 859901 params |
| 1000 | 13 | TRELLIS (generator code) | 30.97 | - | eps=0.001 |
| 1000 | 13 | TRELLIS, chunks off | 30.89 | -0.08 ± 0.11 | eps=0.001 |
| 1000 | 13 | TRELLIS, concepts off (generator reads) | 28.89 | -2.08 ± 0.14 | eps=0.001 |
| 1000 | 13 | TRELLIS, chunks and concepts off | 28.72 | -2.25 ± 0.15 | eps=0.001 |
| 1000 | 13 | TRELLIS, concepts off (reads and parser) | 28.90 | -2.07 ± 0.14 | eps=0.001 |
| 1000 | 17 | uniform over 251 | 51.62 | +21.02 ± 0.41 |  |
| 1000 | 17 | word unigram (BPE k=0) | 42.33 | +11.72 ± 0.36 | k=0 |
| 1000 | 17 | BPE vocabulary + unigram code (k on dev) | 33.07 | +2.47 ± 0.27 | k=400 |
| 1000 | 17 | Witten-Bell trigram (repo baseline) | 25.84 | -4.77 ± 0.23 |  |
| 1000 | 17 | Kneser-Ney trigram | 25.22 | -5.39 ± 0.22 |  |
| 1000 | 17 | best n-gram on dev (WB 2/3/5, KN 2-5) | 25.22 | -5.39 ± 0.22 | kn3 |
| 1000 | 17 | LSTM | 23.66 | -6.94 ± 0.26 | {"layers": 2, "d": 256, "drop": 0.3, "epochs": 12}, 1182461 params |
| 1000 | 17 | GPT | 24.67 | -5.93 ± 0.24 | {"layers": 4, "d": 128, "drop": 0.1, "epochs": 6}, 859901 params |
| 1000 | 17 | TRELLIS (generator code) | 30.60 | - | eps=0.001 |
| 1000 | 17 | TRELLIS, chunks off | 30.71 | +0.10 ± 0.11 | eps=0.001 |
| 1000 | 17 | TRELLIS, concepts off (generator reads) | 28.65 | -1.95 ± 0.15 | eps=0.001 |
| 1000 | 17 | TRELLIS, chunks and concepts off | 28.51 | -2.09 ± 0.15 | eps=0.001 |
| 1000 | 17 | TRELLIS, concepts off (reads and parser) | 28.67 | -1.94 ± 0.15 | eps=0.001 |
| 2000 | 13 | uniform over 251 | 51.27 | +22.47 ± 0.38 |  |
| 2000 | 13 | word unigram (BPE k=0) | 41.80 | +12.99 ± 0.34 | k=0 |
| 2000 | 13 | BPE vocabulary + unigram code (k on dev) | 30.89 | +2.09 ± 0.27 | k=800 |
| 2000 | 13 | Witten-Bell trigram (repo baseline) | 23.62 | -5.18 ± 0.23 |  |
| 2000 | 13 | Kneser-Ney trigram | 23.14 | -5.66 ± 0.22 |  |
| 2000 | 13 | best n-gram on dev (WB 2/3/5, KN 2-5) | 23.19 | -5.61 ± 0.22 | kn4 |
| 2000 | 13 | LSTM | 21.44 | -7.36 ± 0.25 | {"layers": 2, "d": 256, "drop": 0.3, "epochs": 10}, 1182461 params |
| 2000 | 13 | GPT | 22.11 | -6.70 ± 0.24 | {"layers": 4, "d": 64, "drop": 0.3, "epochs": 17}, 233469 params |
| 2000 | 13 | TRELLIS (generator code) | 28.80 | - | eps=0.001 |
| 2000 | 13 | TRELLIS, chunks off | 28.80 | -0.01 ± 0.10 | eps=0.001 |
| 2000 | 13 | TRELLIS, concepts off (generator reads) | 26.99 | -1.81 ± 0.15 | eps=0.001 |
| 2000 | 13 | TRELLIS, chunks and concepts off | 26.83 | -1.97 ± 0.15 | eps=0.001 |
| 2000 | 17 | uniform over 251 | 51.62 | +23.02 ± 0.40 |  |
| 2000 | 17 | word unigram (BPE k=0) | 42.20 | +13.60 ± 0.35 | k=0 |
| 2000 | 17 | BPE vocabulary + unigram code (k on dev) | 31.07 | +2.47 ± 0.25 | k=800 |
| 2000 | 17 | Witten-Bell trigram (repo baseline) | 23.94 | -4.66 ± 0.21 |  |
| 2000 | 17 | Kneser-Ney trigram | 23.41 | -5.19 ± 0.21 |  |
| 2000 | 17 | best n-gram on dev (WB 2/3/5, KN 2-5) | 23.41 | -5.19 ± 0.21 | kn3 |
| 2000 | 17 | LSTM | 21.47 | -7.13 ± 0.25 | {"layers": 2, "d": 256, "drop": 0.3, "epochs": 11}, 1182461 params |
| 2000 | 17 | GPT | 22.38 | -6.22 ± 0.24 | {"layers": 4, "d": 128, "drop": 0.3, "epochs": 9}, 859901 params |
| 2000 | 17 | TRELLIS (generator code) | 28.60 | - | eps=0.001 |
| 2000 | 17 | TRELLIS, chunks off | 28.63 | +0.03 ± 0.10 | eps=0.001 |
| 2000 | 17 | TRELLIS, concepts off (generator reads) | 26.72 | -1.88 ± 0.15 | eps=0.001 |
| 2000 | 17 | TRELLIS, chunks and concepts off | 26.60 | -2.00 ± 0.15 | eps=0.001 |
| 4000 | 13 | uniform over 251 | 51.27 | +24.68 ± 0.38 |  |
| 4000 | 13 | word unigram (BPE k=0) | 41.69 | +15.10 ± 0.34 | k=0 |
| 4000 | 13 | BPE vocabulary + unigram code (k on dev) | 28.53 | +1.94 ± 0.23 | k=1600 |
| 4000 | 13 | Witten-Bell trigram (repo baseline) | 21.93 | -4.66 ± 0.20 |  |
| 4000 | 13 | Kneser-Ney trigram | 21.53 | -5.06 ± 0.19 |  |
| 4000 | 13 | best n-gram on dev (WB 2/3/5, KN 2-5) | 21.40 | -5.19 ± 0.19 | kn4 |
| 4000 | 13 | LSTM | 19.88 | -6.71 ± 0.23 | {"layers": 2, "d": 256, "drop": 0.3, "epochs": 12}, 1182461 params |
| 4000 | 13 | GPT | 20.40 | -6.19 ± 0.21 | {"layers": 4, "d": 64, "drop": 0.3, "epochs": 20}, 233469 params |
| 4000 | 13 | TRELLIS (generator code) | 26.59 | - | eps=0.001 |
| 4000 | 13 | TRELLIS, chunks off | 26.58 | -0.01 ± 0.10 | eps=0.001 |
| 4000 | 13 | TRELLIS, concepts off (generator reads) | 24.86 | -1.73 ± 0.15 | eps=0.0001 |
| 4000 | 13 | TRELLIS, chunks and concepts off | 24.67 | -1.92 ± 0.15 | eps=0.0001 |
| 4000 | 17 | uniform over 251 | 51.62 | +25.05 ± 0.39 |  |
| 4000 | 17 | word unigram (BPE k=0) | 42.14 | +15.57 ± 0.35 | k=0 |
| 4000 | 17 | BPE vocabulary + unigram code (k on dev) | 28.80 | +2.23 ± 0.24 | k=1600 |
| 4000 | 17 | Witten-Bell trigram (repo baseline) | 22.42 | -4.15 ± 0.20 |  |
| 4000 | 17 | Kneser-Ney trigram | 21.92 | -4.65 ± 0.19 |  |
| 4000 | 17 | best n-gram on dev (WB 2/3/5, KN 2-5) | 21.77 | -4.80 ± 0.18 | kn4 |
| 4000 | 17 | LSTM | 20.10 | -6.47 ± 0.22 | {"layers": 1, "d": 256, "drop": 0.3, "epochs": 7}, 656125 params |
| 4000 | 17 | GPT | 20.75 | -5.82 ± 0.22 | {"layers": 4, "d": 128, "drop": 0.3, "epochs": 9}, 859901 params |
| 4000 | 17 | TRELLIS (generator code) | 26.57 | - | eps=0.001 |
| 4000 | 17 | TRELLIS, chunks off | 26.72 | +0.15 ± 0.10 | eps=0.001 |
| 4000 | 17 | TRELLIS, concepts off (generator reads) | 24.91 | -1.67 ± 0.16 | eps=0.0001 |
| 4000 | 17 | TRELLIS, chunks and concepts off | 24.77 | -1.80 ± 0.16 | eps=0.0001 |

### Share of test sentences that are also training sentences (n=250/500/1000/2000/4000), and bits per sentence on the others only (mean over seeds)

| n | share seen | BPE vocabulary + unigram code (k on dev) | best n-gram on dev (WB 2/3/5, KN 2-5) | LSTM | GPT | TRELLIS (generator code) |
|---|---|---|---|---|---|---|
| 250 | 0.100 | 40.1 | 32.6 | 32.0 | 32.5 | 38.6 |
| 500 | 0.150 | 38.7 | 30.8 | 29.4 | 30.3 | 37.2 |
| 1000 | 0.209 | 37.3 | 29.0 | 27.1 | 28.0 | 35.9 |
| 2000 | 0.273 | 36.3 | 28.0 | 25.5 | 26.6 | 34.9 |
| 4000 | 0.357 | 35.6 | 27.5 | 24.9 | 25.8 | 34.2 |

### Model cost (NOT included in any code length above)

| n | seed | BPE dictionary bits (words x log2 250 per merged chunk, selected k) | LSTM chosen / params | GPT chosen / params | TRELLIS records (words + composites) / stored composites | TRELLIS, chunks off records |
|---|---|---|---|---|---|---|
| 250 | 13 | 1999 (k=100) | {"layers": 1, "d": 256, "drop": 0.3, "epochs": 12}, 656125 params | {"layers": 2, "d": 256, "drop": 0.1, "epochs": 7}, 1712893 params | 8331 / 6967 | 1364 |
| 250 | 17 | 2079 (k=100) | {"layers": 1, "d": 128, "drop": 0.3, "epochs": 26}, 197117 params | {"layers": 2, "d": 128, "drop": 0.1, "epochs": 10}, 463357 params | 8288 / 6940 | 1348 |
| 500 | 13 | 4294 (k=200) | {"layers": 1, "d": 256, "drop": 0.1, "epochs": 9}, 656125 params | {"layers": 4, "d": 128, "drop": 0.3, "epochs": 10}, 859901 params | 15299 / 12620 | 2679 |
| 500 | 17 | 4206 (k=200) | {"layers": 2, "d": 256, "drop": 0.1, "epochs": 13}, 1182461 params | {"layers": 2, "d": 128, "drop": 0.1, "epochs": 8}, 463357 params | 15784 / 13060 | 2724 |
| 1000 | 13 | 9161 (k=400) | {"layers": 1, "d": 256, "drop": 0.1, "epochs": 7}, 656125 params | {"layers": 4, "d": 128, "drop": 0.3, "epochs": 7}, 859901 params | 28673 / 23367 | 5306 |
| 1000 | 17 | 9073 (k=400) | {"layers": 2, "d": 256, "drop": 0.3, "epochs": 12}, 1182461 params | {"layers": 4, "d": 128, "drop": 0.1, "epochs": 6}, 859901 params | 30448 / 24946 | 5502 |
| 2000 | 13 | 19556 (k=800) | {"layers": 2, "d": 256, "drop": 0.3, "epochs": 10}, 1182461 params | {"layers": 4, "d": 64, "drop": 0.3, "epochs": 17}, 233469 params | 53945 / 43292 | 10653 |
| 2000 | 17 | 19835 (k=800) | {"layers": 2, "d": 256, "drop": 0.3, "epochs": 11}, 1182461 params | {"layers": 4, "d": 128, "drop": 0.3, "epochs": 9}, 859901 params | 54975 / 44118 | 10857 |
| 4000 | 13 | 42928 (k=1600) | {"layers": 2, "d": 256, "drop": 0.3, "epochs": 12}, 1182461 params | {"layers": 4, "d": 64, "drop": 0.3, "epochs": 20}, 233469 params | 102483 / 80965 | 21518 |
| 4000 | 17 | 42752 (k=1600) | {"layers": 1, "d": 256, "drop": 0.3, "epochs": 7}, 656125 params | {"layers": 4, "d": 128, "drop": 0.3, "epochs": 9}, 859901 params | 102207 / 80645 | 21562 |

### BPE: held-out bits per sentence at every merge count (mean over seeds), the dictionary cost in bits and the dev-selected k

| n | k=0 | k=10 | k=25 | k=50 | k=100 | k=200 | k=400 | k=800 | k=1600 |
|---|---|---|---|---|---|---|---|---|---|
| 250 | 42.6 | 41.3 | 40.0 | 38.8 | 38.0 | - | - | - | - |
| 500 | 42.3 | 40.8 | 39.5 | 38.1 | 36.8 | 35.7 | - | - | - |
| 1000 | 42.1 | 40.5 | 39.1 | 37.6 | 35.9 | 34.3 | 33.1 | - | - |
| 2000 | 42.0 | 40.3 | 38.9 | 37.3 | 35.6 | 33.9 | 32.1 | 31.0 | - |
| 4000 | 41.9 | 40.1 | 38.6 | 37.3 | 35.4 | 33.5 | 31.3 | 29.6 | 28.7 |

### n-gram detail (bits per sentence, mean over seeds)

| n | wb2 | wb3 | wb5 | kn2 | kn3 | kn4 | kn5 |
|---|---|---|---|---|---|---|---|
| 250 | 31.3 | 31.3 | 33.6 | 30.9 | 30.3 | 30.7 | 31.4 |
| 500 | 29.1 | 28.5 | 30.6 | 28.8 | 27.7 | 28.0 | 28.5 |
| 1000 | 27.1 | 25.7 | 27.5 | 26.8 | 25.1 | 25.3 | 25.7 |
| 2000 | 25.8 | 23.8 | 25.2 | 25.4 | 23.3 | 23.3 | 23.7 |
| 4000 | 24.7 | 22.2 | 23.3 | 24.4 | 21.7 | 21.6 | 21.9 |

### TRELLIS eps floor chosen on dev (p -> (p + eps) / (1 + eps K), K = 500), by arm

| arm | n | seed 13 | seed 17 |
|---|---|---|---|
| TRELLIS (generator code) | 250 | eps=0.001 | eps=0.001 |
| TRELLIS (generator code) | 500 | eps=0.001 | eps=0.001 |
| TRELLIS (generator code) | 1000 | eps=0.001 | eps=0.001 |
| TRELLIS (generator code) | 2000 | eps=0.001 | eps=0.001 |
| TRELLIS (generator code) | 4000 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks off | 250 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks off | 500 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks off | 1000 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks off | 2000 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks off | 4000 | eps=0.001 | eps=0.001 |
| TRELLIS, concepts off (generator reads) | 250 | eps=0.001 | eps=0.001 |
| TRELLIS, concepts off (generator reads) | 500 | eps=0.001 | eps=0.001 |
| TRELLIS, concepts off (generator reads) | 1000 | eps=0.001 | eps=0.001 |
| TRELLIS, concepts off (generator reads) | 2000 | eps=0.001 | eps=0.001 |
| TRELLIS, concepts off (generator reads) | 4000 | eps=0.0001 | eps=0.0001 |
| TRELLIS, chunks and concepts off | 250 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks and concepts off | 500 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks and concepts off | 1000 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks and concepts off | 2000 | eps=0.001 | eps=0.001 |
| TRELLIS, chunks and concepts off | 4000 | eps=0.0001 | eps=0.0001 |

## Parse metrics of the ablation arms (r216-style: generator score `gen_lp_gate`, parse score `logZ_pw`; thresholds from Gcal, P(grammatical)=.9)


Reproduction check, full arm vs r216 default learner, seed 13 n=1000: max |sentence log-prob difference| 0.00e+00, max |logZ difference| 0.00e+00.


Reproduction check, full arm vs r216 default learner, seed 17 n=1000: max |sentence log-prob difference| 0.00e+00, max |logZ difference| 0.00e+00.


### n = 1000 (mean ± sd over seeds)

| arm | C1 accepted (gen) | C2full accepted (gen) | C2one accepted (gen) | AUC C2full (gen) | AUC C2one (gen) | AUC C3scr (gen) | AUC C3adj (gen) | AUC C4v (gen) | AUC C3scr (logZ) | AUC C2full (logZ) | tree F1 vs gold | right-branching F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.895±0.012 | 0.708±0.007 | 0.845±0.016 | 0.748±0.002 | 0.626±0.009 | 0.958±0.000 | 0.779±0.009 | 0.706±0.011 | 0.606±0.006 | 0.500±0.009 | 0.347±0.021 | 0.427±0.000 |
| nochunk | 0.888±0.002 | 0.707±0.014 | 0.818±0.007 | 0.753±0.004 | 0.629±0.003 | 0.958±0.001 | 0.786±0.016 | 0.698±0.002 | 0.564±0.000 | 0.489±0.003 | 0.383±0.016 | 0.427±0.000 |
| noconcept | 0.908±0.021 | 0.573±0.024 | 0.822±0.026 | 0.815±0.011 | 0.670±0.007 | 0.973±0.004 | 0.846±0.018 | 0.733±0.002 | 0.606±0.006 | 0.500±0.009 | 0.347±0.021 | 0.427±0.000 |
| neither | 0.908±0.021 | 0.575±0.021 | 0.822±0.026 | 0.816±0.011 | 0.672±0.007 | 0.973±0.004 | 0.848±0.017 | 0.736±0.002 | 0.564±0.000 | 0.489±0.003 | 0.383±0.016 | 0.427±0.000 |
| noconceptp | 0.908±0.021 | 0.575±0.026 | 0.823±0.024 | 0.814±0.011 | 0.669±0.007 | 0.973±0.004 | 0.845±0.018 | 0.733±0.002 | 0.500±0.000 | 0.500±0.000 | 0.291±0.006 | 0.427±0.000 |

Seeds: 13, 17


## Ablation summary at n = 1000 (held-out bits, generation fair counts, mean over seeds)

| arm | bits/sentence (test) | records | stored composites | fair syntax t1.2 | fair syntax t2 | fair syntax t3 | fair syn+sem t1.2 | fair syn+sem t2 | fair syn+sem t3 | distinct-new share t2.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 30.8±0.3 | 29560 | 24156 | 0.5103 | 0.5388 | 0.4863 | 0.3728 | 0.3634 | 0.2969 | 0.814 |
| nochunk | 30.8±0.1 | 5404 | 0 | 0.5113 | 0.5197 | 0.4778 | 0.3797 | 0.3466 | 0.2687 | 0.803 |
| noconcept | 28.8±0.2 | 29560 | 24156 | 0.4622 | 0.4963 | 0.4481 | 0.3087 | 0.3090 | 0.2400 | 0.889 |
| neither | 28.6±0.1 | 5404 | 0 | 0.4872 | 0.5118 | 0.4309 | 0.3296 | 0.3106 | 0.2263 | 0.897 |
| noconceptp | 28.8±0.2 | 37354 | 31950 | 0.4628 | 0.4753 | 0.4278 | 0.3047 | 0.2925 | 0.2403 | 0.874 |

---

# Appendix B: generation per seed and mean (r225/gen_table.md)

| arm | temp | seed | sentences returned | distinct new | syntax pass (of distinct new) | syntax+semantics pass | fair syntax (per 1600 draws) | fair syntax+semantics | mean words |
|---|---|---|---|---|---|---|---|---|---|
| full | 1.2 | 13 | 1600 | 1095 | 0.732 | 0.541 | 0.5006 | 0.3700 | 5.85 |
| full | 1.2 | 17 | 1544 | 1142 | 0.729 | 0.526 | 0.5200 | 0.3756 | 6.02 |
| full | 2 | 13 | 1600 | 1292 | 0.688 | 0.467 | 0.5556 | 0.3769 | 5.75 |
| full | 2 | 17 | 1544 | 1313 | 0.636 | 0.427 | 0.5219 | 0.3500 | 5.87 |
| full | 3 | 13 | 1600 | 1370 | 0.591 | 0.360 | 0.5056 | 0.3081 | 5.61 |
| full | 3 | 17 | 1544 | 1330 | 0.562 | 0.344 | 0.4669 | 0.2856 | 5.80 |
| neither | 1.2 | 13 | 1600 | 1301 | 0.616 | 0.407 | 0.5006 | 0.3312 | 5.18 |
| neither | 1.2 | 17 | 1544 | 1214 | 0.624 | 0.432 | 0.4738 | 0.3281 | 5.34 |
| neither | 2 | 13 | 1600 | 1473 | 0.576 | 0.334 | 0.5306 | 0.3075 | 5.17 |
| neither | 2 | 17 | 1544 | 1399 | 0.564 | 0.359 | 0.4931 | 0.3137 | 5.29 |
| neither | 3 | 13 | 1600 | 1507 | 0.448 | 0.230 | 0.4219 | 0.2169 | 5.06 |
| neither | 3 | 17 | 1544 | 1476 | 0.477 | 0.255 | 0.4400 | 0.2356 | 5.13 |
| nochunk | 1.2 | 13 | 1600 | 1163 | 0.721 | 0.532 | 0.5238 | 0.3869 | 5.91 |
| nochunk | 1.2 | 17 | 1544 | 1119 | 0.713 | 0.533 | 0.4988 | 0.3725 | 6.03 |
| nochunk | 2 | 13 | 1600 | 1292 | 0.666 | 0.435 | 0.5375 | 0.3513 | 5.83 |
| nochunk | 2 | 17 | 1544 | 1278 | 0.628 | 0.428 | 0.5019 | 0.3419 | 5.86 |
| nochunk | 3 | 13 | 1600 | 1381 | 0.563 | 0.326 | 0.4856 | 0.2812 | 5.71 |
| nochunk | 3 | 17 | 1544 | 1330 | 0.565 | 0.308 | 0.4700 | 0.2562 | 5.87 |
| noconcept | 1.2 | 13 | 1600 | 1249 | 0.601 | 0.396 | 0.4694 | 0.3094 | 5.07 |
| noconcept | 1.2 | 17 | 1544 | 1165 | 0.625 | 0.423 | 0.4550 | 0.3081 | 5.22 |
| noconcept | 2 | 13 | 1600 | 1442 | 0.540 | 0.313 | 0.4869 | 0.2825 | 5.07 |
| noconcept | 2 | 17 | 1544 | 1404 | 0.576 | 0.382 | 0.5056 | 0.3356 | 5.27 |
| noconcept | 3 | 13 | 1600 | 1500 | 0.459 | 0.245 | 0.4306 | 0.2300 | 4.93 |
| noconcept | 3 | 17 | 1544 | 1466 | 0.508 | 0.273 | 0.4656 | 0.2500 | 5.12 |
| noconceptp | 1.2 | 13 | 1600 | 1231 | 0.604 | 0.396 | 0.4644 | 0.3050 | 5.10 |
| noconceptp | 1.2 | 17 | 1544 | 1165 | 0.633 | 0.418 | 0.4612 | 0.3044 | 5.31 |
| noconceptp | 2 | 13 | 1600 | 1413 | 0.537 | 0.311 | 0.4744 | 0.2750 | 5.06 |
| noconceptp | 2 | 17 | 1544 | 1384 | 0.551 | 0.358 | 0.4763 | 0.3100 | 5.21 |
| noconceptp | 3 | 13 | 1600 | 1488 | 0.444 | 0.237 | 0.4125 | 0.2206 | 4.92 |
| noconceptp | 3 | 17 | 1544 | 1458 | 0.486 | 0.285 | 0.4431 | 0.2600 | 5.07 |

Mean over seeds:

| arm | temp | seeds | returned | distinct new share of draws | syntax pass | syntax+semantics pass | fair syntax | fair syntax+semantics |
|---|---|---|---|---|---|---|---|---|
| full | 1.2 | 2 | 1572 | 0.699 | 0.730 | 0.533 | 0.5103 | 0.3728 |
| full | 2 | 2 | 1572 | 0.814 | 0.662 | 0.447 | 0.5388 | 0.3634 |
| full | 3 | 2 | 1572 | 0.844 | 0.576 | 0.352 | 0.4862 | 0.2969 |
| nochunk | 1.2 | 2 | 1572 | 0.713 | 0.717 | 0.532 | 0.5112 | 0.3797 |
| nochunk | 2 | 2 | 1572 | 0.803 | 0.647 | 0.431 | 0.5197 | 0.3466 |
| nochunk | 3 | 2 | 1572 | 0.847 | 0.564 | 0.317 | 0.4778 | 0.2687 |
| noconcept | 1.2 | 2 | 1572 | 0.754 | 0.613 | 0.410 | 0.4622 | 0.3087 |
| noconcept | 2 | 2 | 1572 | 0.889 | 0.558 | 0.348 | 0.4962 | 0.3091 |
| noconcept | 3 | 2 | 1572 | 0.927 | 0.484 | 0.259 | 0.4481 | 0.2400 |
| neither | 1.2 | 2 | 1572 | 0.786 | 0.620 | 0.420 | 0.4872 | 0.3297 |
| neither | 2 | 2 | 1572 | 0.897 | 0.570 | 0.346 | 0.5119 | 0.3106 |
| neither | 3 | 2 | 1572 | 0.932 | 0.462 | 0.243 | 0.4309 | 0.2263 |
| noconceptp | 1.2 | 2 | 1572 | 0.749 | 0.619 | 0.407 | 0.4628 | 0.3047 |
| noconceptp | 2 | 2 | 1572 | 0.874 | 0.544 | 0.335 | 0.4753 | 0.2925 |
| noconceptp | 3 | 2 | 1572 | 0.921 | 0.465 | 0.261 | 0.4278 | 0.2403 |
