# Round 215b: does TRELLIS's own generation probability discriminate where the parse score does not?

**Verdict.** Yes. The causal generator's log-probability of a sentence (`gen_lp_gate`) separates grammatical from corrupted sentences far better than the parse score (`logZ` per word), while the same learner supplies both. On full scrambles at n=4000 the paired AUC is .993 against .879 for logZ (GPT, LSTM, 5-gram: .998), and the operating-point gap is .865 against .165 (baselines .90). On the other classes the generator closes most of the distance from logZ to the baselines and stays below them. The parser ignores what the generator knows.

The alternative explanation (generation is grammatical only because it samples the mode) does not hold. The generator's distribution is spread (mean probability of the true word at each step .23 on grammatical test sentences at n=4000, argmax agreement .28), and its log-probability beats its argmax agreement everywhere (scramble AUC .98 against .87). I report the argmax and rank tests below, so the check is done even though gen_lp did not fail.

## What was scored

Code: `experiments/v3/diag/r215b_gen.py` (learner retrain + scoring) and `r215b_analyse.py` (metrics). The r215 pipeline is reused unchanged: `r215_common` stream, recipe and windows (125/250/1000/4000: w7/w7/w7/w5), the r215 sets (`r215_sets/`), Gcal calibration thresholds at the 10th percentile, AUC, paired AUC, gap. Retrained learners reproduce the r215 logZ exactly (max abs difference 0.0 on 15 sentences for every run).

- **gen_lp / gen_lp_gate.** The distribution is the one `t3_story.py` `gen` draws from for the paper arm `{"causal":1,"units":"words","pair":"gate","edge":"exact"}` at temp 1: `_fills` code copied (units "words", no escape/both/esc), `L._gap_fills(q)` for the mass of a word and `L._gap_fills({q, r1: EOS})` for the mass of the word with the sentence ending right after it, window w as the r215 learner, `edge_exact` on. Each step has outcomes (word, ends here); the sampler draws one in proportion to its weight, so the sentence probability is the product of P(w_i, end = [i is last]) = the sum over positions of log P(w_i | prefix) plus the final log P(</s> | prefix, last word). `gen_lp_gate` keeps the pair gate (median of stored pairs' type licence, as in generation, which also renormalises over the admitted fillers); `gen_lp` drops the gate (every filler the read offers stays).
- **Normalisation and floor.** Per step the weights are divided by their sum (so mass is normalised over the outcomes the generator can draw). Outcomes with no mass get a floor: p = (p0 + eps) / (1 + eps K), K = 2 x 250 outcomes, the same eps for every set. A step where the generator has no admissible outcome at all (Z = 0) gets the uniform 1/K. Score = sum of log p over steps / (len + 1) (the LMs' per-word convention). eps = 1e-6 is the primary row; eps = 1e-4 is the sensitivity row (columns `_e0.0001` in `r215b_tables.md`). The AUCs move by .01 or less on scramble and by up to .035 on C4v, in no consistent direction, and the ranking of the sets does not change.
- **Not modelled:** the sampler's length filter (sentences of 3 to 8 words, redrawn otherwise) and its sentence-level `new` rejection; the paper's sampling temperatures (3.5 at n=250 and similar, which flatten the distribution) are not used because the brief asks for tau = 1.
- **parse_norm.** Over all strings of the same length no partition is computable (250^len strings, and Z is an unnormalised weight). Over the permutations of the sentence's own words it is: parse_norm = (logZ - log mean over 16 random permutations of the sentence's bag of exp logZ) / len, a Monte Carlo estimate of log Z(s) / Sum_pi Z(pi), up to the constant log n!. It is a different score only when the corrupted sentence has a different bag (content swaps, agreement); for scrambles and adjacent swaps the denominator is shared, so the paired AUC equals that of logZ. It does help the unpaired comparison (scramble .64 -> .87 at n=4000) because it removes the word-frequency component of logZ, but the operating-point gap on the other classes stays below .15.
- **Coverage.** Seeds 13/17/19/23 at n = 250, 1000, 4000. n=125 is seed 13 only (the smoke test; shown as `(1)` in `r215b_tables.md`). n=500 and 2000 were not run.

## Results (mean over 4 seeds; sd in `r215b_tables.md`)

Classes in each table: scramble | adjacent swap | content swap all (C2full) | content swap one (C2one) | agreement (C4v). The r215 rows (logZ, GPT, LSTM, 5-gram) are the r215 scores recomputed on the same sets; they match `r215_parse_curve.md`.

### Unpaired AUC

n=4000
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ per word (r215) | .641 | .553 | .517 | .507 | .505 |
| TRELLIS parse_norm | .865 | .655 | .591 | .542 | .500 |
| **TRELLIS gen_lp_gate** | .981 | .848 | .810 | .680 | .767 |
| TRELLIS gen_lp (no gate) | .959 | .776 | .764 | .638 | .729 |
| 5-gram | .993 | .908 | .887 | .756 | .850 |
| LSTM | .993 | .914 | .886 | .739 | .831 |
| GPT | .995 | .938 | .880 | .735 | .816 |

n=1000
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ | .599 | .534 | .492 | .495 | .502 |
| TRELLIS parse_norm | .818 | .630 | .598 | .554 | .520 |
| **TRELLIS gen_lp_gate** | .958 | .781 | .739 | .623 | .708 |
| TRELLIS gen_lp | .908 | .701 | .690 | .583 | .652 |
| 5-gram | .977 | .824 | .829 | .693 | .748 |
| LSTM | .972 | .813 | .824 | .679 | .748 |
| GPT | .985 | .869 | .814 | .665 | .730 |

n=250
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ | .561 | .517 | .476 | .489 | .498 |
| TRELLIS parse_norm | .718 | .559 | .558 | .522 | .502 |
| **TRELLIS gen_lp_gate** | .921 | .724 | .658 | .576 | .658 |
| TRELLIS gen_lp | .850 | .636 | .630 | .554 | .572 |
| 5-gram | .931 | .722 | .760 | .632 | .653 |
| LSTM | .903 | .676 | .747 | .625 | .642 |
| GPT | .941 | .749 | .733 | .608 | .634 |

### Paired AUC

n=4000
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ | .879 | .778 | .584 | .533 | .608 |
| TRELLIS parse_norm | .868 | .712 | .645 | .578 | .505 |
| **TRELLIS gen_lp_gate** | .993 | .953 | .910 | .843 | .911 |
| TRELLIS gen_lp | .982 | .890 | .886 | .816 | .887 |
| 5-gram | .998 | .984 | .951 | .906 | .968 |
| LSTM | .998 | .992 | .956 | .898 | .983 |
| GPT | .998 | .991 | .953 | .895 | .974 |

n=1000
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ | .852 | .734 | .427 | .441 | .555 |
| TRELLIS parse_norm | .829 | .683 | .646 | .604 | .550 |
| **TRELLIS gen_lp_gate** | .976 | .910 | .835 | .785 | .868 |
| TRELLIS gen_lp | .951 | .837 | .823 | .746 | .834 |
| 5-gram | .993 | .953 | .921 | .865 | .920 |
| LSTM | .994 | .975 | .913 | .866 | .953 |
| GPT | .992 | .977 | .911 | .843 | .957 |

n=250
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ | .737 | .654 | .393 | .390 | .459 |
| TRELLIS parse_norm | .724 | .603 | .587 | .552 | .524 |
| **TRELLIS gen_lp_gate** | .940 | .822 | .728 | .668 | .784 |
| TRELLIS gen_lp | .901 | .743 | .757 | .691 | .739 |
| 5-gram | .972 | .902 | .868 | .797 | .869 |
| LSTM | .978 | .930 | .853 | .810 | .889 |
| GPT | .980 | .916 | .835 | .765 | .876 |

### Operating-point gap, P(success | gram) - P(success | corrupted), threshold P = .9 on Gcal

n=4000
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ | .165 | .067 | .028 | .005 | .003 |
| TRELLIS parse_norm | .522 | .148 | .094 | .036 | .005 |
| **TRELLIS gen_lp_gate** | .865 | .381 | .332 | .114 | .130 |
| TRELLIS gen_lp | .783 | .246 | .236 | .066 | .047 |
| 5-gram | .902 | .506 | .520 | .183 | .271 |
| LSTM | .903 | .558 | .499 | .142 | .186 |
| GPT | .899 | .705 | .503 | .170 | .180 |

n=1000
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ | .151 | .060 | .011 | .002 | -.004 |
| TRELLIS parse_norm | .366 | .082 | .085 | .035 | -.001 |
| **TRELLIS gen_lp_gate** | .765 | .254 | .184 | .053 | .089 |
| TRELLIS gen_lp | .606 | .133 | .116 | .032 | .024 |
| 5-gram | .856 | .260 | .328 | .087 | .139 |
| LSTM | .833 | .223 | .288 | .076 | .097 |
| GPT | .878 | .367 | .278 | .072 | .102 |

n=250
| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| TRELLIS logZ | .071 | .036 | -.008 | -.013 | -.009 |
| TRELLIS parse_norm | .218 | .032 | .058 | .022 | -.002 |
| **TRELLIS gen_lp_gate** | .639 | .184 | .068 | .027 | .070 |
| TRELLIS gen_lp | .440 | .075 | .043 | .018 | .017 |
| 5-gram | .647 | .132 | .181 | .061 | .056 |
| LSTM | .548 | .082 | .166 | .048 | .045 |
| GPT | .698 | .177 | .157 | .052 | .050 |

P(success | grammatical test C1) at the Gcal threshold is .89-.93 for every score and n (end of `r215b_tables.md`), so the operating points are comparable.

Reading the tables.
- **Scramble.** At n=250 the gated generator score already matches the 5-gram and LSTM on unpaired AUC (.92 against .93/.90) and the gap (.64 against .65/.55). At n=1000 and 4000 it trails by .02-.03 AUC and .04-.11 gap.
- **Adjacent swap, content swaps, agreement.** The generator score is well above the parse score (agreement paired .91 against .61) and tracks the baselines from below: paired C2full .91 against .95, adjacent swap .95 against .98-.99, and the gap on adjacent swap .38 against .51-.71. It is at n=250 level with or above the baselines on the unpaired and gap measures of adjacent swap and agreement.
- **Gate.** Removing the pair gate costs .02-.08 AUC and .08-.16 gap at n=1000/4000. The gate carries a lot of the discrimination.
- **Both scores trail on content swaps.** C2one (one word swapped within POS) stays at paired AUC .84, gap .11 at n=4000: the generator's word-level conditionals detect an implausible word less often than an LSTM does.

## Is it calibration or the mode? (tests when gen_lp does not fail anyway)

Unpaired AUC, gated generator, n=4000 (paired in `r215b_tables.md`):

| score | scr | adj | C2full | C2one | agr |
|---|---|---|---|---|---|
| gen_lp_gate (sum of log probabilities) | .981 | .848 | .810 | .680 | .767 |
| rank (mean -log rank of the true outcome) | .986 | .856 | .801 | .674 | .815 |
| top-1 (share of steps where the true outcome is the argmax) | .869 | .712 | .686 | .602 | .749 |
| unsup (-share of steps with zero mass on the true outcome) | .986 | .926 | .849 | .725 | .676 |

- Argmax agreement alone (top-1) is clearly weaker than the probability (scramble .87 against .98), and the rank score matches it. The distribution carries graded information, not just the mode.
- The zero-mass count is as good as the full score for scramble and adjacent swap (.99, .93) and better on adjacent swap than the log-probability. So a large part of the discrimination is licence: whether the corrupted continuation is in the support of the stored-context reads at all. The probabilities add most on agreement (.77 against .68 for unsup) and on the hardest content swaps only slightly. This also explains why the ungated score, which has a bigger support, is weaker.
- Diagnostics (gated, n=4000): the true word has zero mass at 10% of the steps of grammatical test sentences, 69% of scramble steps, 40% of adjacent-swap steps, 33% of C2full, 22% of C2one and 18% of agreement steps (`r215b_tables.md`, last table but one). Steps with Z = 0 (no admissible outcome at all) are under .3% at n=4000, 4-11% at n=250 and 9-18% at n=125.

## Proposed change (not implemented): parse with the generator's conditionals

The parser's inside weight is built only from join gains (`Trellis3._factor`, `src/trellis2/trellis3.py` l.1161, summed over analyses in `_inside_outside` l.1253, called from `parse` l.1368 on the chart from `_chart` l.1063). It uses no word-in-context probability of the kind `_gap_fills` (l.2827) provides to generation. Minimal principled change: give each leaf the generator's causal conditional as a unary factor, log P_gen(w_i | the window of the words to its left) read with `L._gap_fills` plus the pair gate, edge-exact boundary, and the end factor at the last leaf, so the weight of an analysis is its join gains times the product of leaf conditionals. The left window is a property of the sentence, not of the analysis, so the factor is analysis-independent and the new score is logZ_parse + log P_gen(sentence), computed once per `parse` call. The result is a parser whose logZ includes what generation already knows. The same score would replace `logZ / (len - 1)` as the critic in `generate_best` (l.3301). A weighted version (logZ_parse + lambda log P_gen) is a one-parameter fit on Gcal. Expected effect from this round: the combined score cannot be worse than gen_lp on the classes where logZ is at chance; whether the join gains add anything beyond gen_lp (paired adjacent swap .78 from logZ against .95 from gen_lp) needs the combined run.

## Slips and caveats

- First launch at n=4000 (seeds 13, 19, 23) was killed by the memory limit (220G). Reruns split the work into `gen` and `pn` (parse_norm) stages, cleared the learner's `_gap_cache` after each context and kept a reduced cache, at 300G. Seed 17 at n=4000 finished in the first run (with the raw-mass cache and parse_norm in the same job). Scores are identical in definition; the code change only reduces what is cached. Retrained learners reproduce the r215 logZ exactly in all runs.
- n=125 is seed 13 only, 500 and 2000 were not run (time).
- parse_norm uses 16 random permutations per sentence, so it is a noisy estimate (it matches logZ's paired AUC on scrambles within .02, as the shared denominator predicts).
- While waiting I ran `python -c 1` once on the login node (blocked by the permission classifier before it ran, no effect). All real runs went through sbatch.
- Scores are at tau = 1; the paper's generator samples at higher temperature, which flattens the distribution; I did not score at those temperatures.
- Rerunning `r215_analyse.py` rewrites `r215_parse_curve.tsv` from scratch and drops the 215b rows; rerun `r215b_analyse.py` after it.
