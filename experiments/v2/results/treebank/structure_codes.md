Penn Treebank sample: 3901 sentences, 82356 tags (punctuation removed); WSJ10: 542 sentences.

| Description of the tags | Sends | Bits | Against the tag bigram |
|---|---|---|---|
| tag unigram | – | 353,898 | +24.8% |
| tag bigram | – | 283,596 | +0.0% |
| tag trigram | – | 284,510 | +0.3% |
| dependency trees, first order, gold | derivation | 370,479 | +30.6% |
| dependency trees, first order, gold | total probability | 300,005 | +5.8% |
|  (bits back: 18.1 per sentence) | | | |
| dependency trees, second order (sibling), gold | derivation | 356,700 | +25.8% |
| dependency trees, second order (sibling), gold | total probability | 300,793 | +6.1% |
|  (bits back: 14.3 per sentence) | | | |
| headed base-phrase chunks, Markov over heads, gold | derivation | 310,730 | +9.6% |

WSJ10, TRELLIS's representation (plain PCFG over the treebank's labels):

| Description of the tags (WSJ10) | Sends | Bits | Against the tag bigram |
|---|---|---|---|
| tag bigram | – | 15,284 | +0.0% |
| labelled gold trees, plain PCFG | derivation | 21,108 | +38.1% |
| labelled gold trees, plain PCFG | total probability | 20,119 | +31.6% |
|  (bits back: 1.8 per sentence) | | | |
| dependency trees, first order, gold | total probability | 16,062 | +5.1% |

WSJ10, the first-order dependency model after 40 EM iterations (total probability), by start:

| Start | Bits | Against the tag bigram | Heads right |
|---|---|---|---|
| random trees, seed 4 | 15,348 | +0.4% | 49.6% |
| the gold trees' parameters | 15,427 | +0.9% | 71.7% |
| random trees, seed 0 | 15,539 | +1.7% | 55.1% |
| left-branching chains | 15,540 | +1.7% | 34.1% |
| random trees, seed 5 | 15,540 | +1.7% | 56.3% |
| right-branching chains | 15,562 | +1.8% | 23.0% |
| random trees, seed 2 | 15,573 | +1.9% | 45.5% |
| harmonic (Klein & Manning 2004) | 15,596 | +2.0% | 44.1% |
| random parameters, seed 3 | 15,649 | +2.4% | 53.0% |
| random parameters, seed 4 | 15,704 | +2.7% | 53.4% |
| random trees, seed 6 | 15,770 | +3.2% | 37.0% |
| random trees, seed 1 | 15,783 | +3.3% | 38.9% |
| random trees, seed 3 | 15,833 | +3.6% | 39.9% |
| random parameters, seed 5 | 15,894 | +4.0% | 40.5% |
| uniform | 16,009 | +4.7% | 23.7% |
| random parameters, seed 1 | 16,037 | +4.9% | 48.4% |
| random parameters, seed 2 | 16,114 | +5.4% | 29.8% |
| random parameters, seed 0 | 16,153 | +5.7% | 26.3% |

Rank correlation between code length and heads right: -0.54

The gap of the first-order dependency code (total probability) shrinks slowly with data:

| Sentences | Tags | Against the tag bigram |
|---|---|---|
| 487 | 10,195 | +7.6% |
| 975 | 20,595 | +6.8% |
| 1,950 | 41,352 | +6.2% |
| 3,901 | 82,356 | +5.8% |

Words given their tags (lowercased; sequential Pitman-Yor codes):

| Context of each word | Bits |
|---|---|
| tag only | 664,762 |
| previous word | 622,578 |
| head word | 621,433 |
| previous word and head word | 615,481 |

TRELLIS v2's own code for analyses of the WSJ10 training sentences (seed 13):

| Analyses | Bits |
|---|---|
| learner's night (its own code) | 14,160 |
| the learner's forests | 14,313 |
| the learner's chunks, right-branching above them | 15,259 |
| the learner's chunks, left-branching above them | 14,869 |
| right-branching trees | 14,421 |
| left-branching trees | 14,494 |
| gold trees (right-binarized) | 15,763 |
