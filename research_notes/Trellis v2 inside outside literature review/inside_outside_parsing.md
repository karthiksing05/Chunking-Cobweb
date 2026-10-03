# Inside-Outside, Chart, and Lattice Parsing as Candidate Replacements for TRELLIS's Greedy Parser

Conventions used in these notes. "Bracket score" means the PARSEVAL harmonic mean of bracket precision and recall, as each cited paper reports it; "unlabeled bracket score" is the same on unlabeled brackets. TRELLIS's own results are described in its own terms: bracket agreement, and Langley & Stromsten (2000) errors of omission and commission ([Langley & Stromsten 2000](https://doi.org/10.1007/3-540-45164-1_23)). For full binary trees over the same n words, the gold tree and the predicted tree each contain n-1 brackets, so bracket recall, bracket precision and their harmonic mean are all equal (simple arithmetic, not a cited claim). Notation: words w_0..w_{n-1}. An "item" (i,k,j) is a binary composite whose left child spans i..k and whose right child spans k+1..j. A "label" is the concept descriptor a chart cell carries.

TRELLIS facts that come from the code rather than the papers:
- The content instance has 4 attributes: the LEFT and RIGHT child's top-K bag of depth-d context-tree node ids, plus LEFT and RIGHT child complexity tags ([parse_mh.py, `create_content_instance`](/Users/karthiksing05/Documents/ISLE-Research/trellis_v1/src/parse_mh.py)).
- The context instance has the before/after word slots, a hidden complexity attribute, and a visible "content-ref" attribute holding the content-leaf id. It also already supports a `chunk_context_before/after` override mode ([parse_mh.py, `create_context_instance`](/Users/karthiksing05/Documents/ISLE-Research/trellis_v1/src/parse_mh.py)).
- The Cobweb discrete tree exposes `ifit`, `categorize`, `log_prob(instance, max_nodes)`, `log_prob_class_given_instance`, `predict` (missing-attribute prediction) and `bfs_top_k_leaves`. `ifit` has no explicit weight argument ([cobweb_discrete.cpp bindings](/Users/karthiksing05/Documents/ISLE-Research/trellis_v1/cobweb-private/src/cobweb_discrete.cpp)).

---

## 1. Classic inside-outside: what the quantities mean, expected counts and EM, complexity, partial bracketing, and the backprop/semiring views

### Takeaway
The inside quantity of a cell sums the weight of every way to build that span. The outside quantity sums the weight of every way the rest of the sentence can surround it. Their normalized product is the posterior probability that the span (or rule application) is in the parse, and these posteriors are exactly the expected counts that EM re-estimation needs.

The cost is cubic in sentence length. Full bracketing, the situation TRELLIS v1 trains in, collapses that cost to linear and greatly improves the structure inside-outside discovers.

Goodman (1999) and Eisner (2016) show that one chart engine yields recognition, the best parse, marginals, k-best lists and expectations just by swapping the semiring or by backpropagating through the inside pass. The inside/sum semiring only needs non-negative item weights, so Cobweb-derived potentials are admissible.

### Cited Findings
- **Origin and role.** The inside-outside algorithm originates with Baker's "Trainable grammars for speech recognition" (J. Acoust. Soc. Am. 65(S1):S132, 1979) — [Baker 1979](https://doi.org/10.1121/1.2017061).
- **What it computes.** Eisner describes it as computing, "given a sentence, ... the expected count of each possible grammatical substructure at each position in the sentence." Such expected counts are used "(1) to train grammar weights from data, (2) to select low-risk parses, and (3) as soft features that characterize sentence positions" — [Eisner 2016](https://aclanthology.org/W16-5901/).
- **Lari & Young (1990)** is the standard reference for estimating stochastic CFGs with inside-outside (Computer Speech & Language 4:35–56) — [Lari & Young 1990](https://doi.org/10.1016/0885-2308(90)90022-X).
- **Pereira & Schabes (1992)**, "Inside-Outside Reestimation from Partially Bracketed Corpora" (ACL 1992, pp. 128–135), identify three problems with raw inside-outside:
  - **Cost.** Inside-outside is expensive compared with HMM training, which needs "at worst O(s²|w|) time per training sentence". That complexity makes training sufficiently large grammars impractical.
  - **Local maxima.** Convergence "sharply deteriorate[s] as the number of nonterminal symbols increases" because the number of local maxima grows with the number of nonterminals.
  - **Structure.** Raw text underdetermines hierarchical structure, so "only by chance will the inferred grammar agree with" linguistic structure.

  Source: [Pereira & Schabes 1992](https://aclanthology.org/P92-1017/).
- **Pereira & Schabes' fix.** They restrict inside-outside to spans consistent with a (partial) bracketing. On ATIS part-of-speech strings, after 75 iterations, the grammar trained on raw text reached only 37.35% bracketing accuracy, versus 90.36% for the grammar trained with brackets. Bracketed training "steadily improves accuracy", while unbracketed training "does not on the whole improve accuracy" — [Pereira & Schabes 1992](https://aclanthology.org/P92-1017/).
- **Linear time with full bracketing.** With full binary bracketing, "the time for each iteration is in fact linear on the total length of the set" — [Pereira & Schabes 1992](https://aclanthology.org/P92-1017/).
- **Inside-outside is backprop.** Eisner (2016, Workshop on Structured Prediction for NLP, pp. 1–17) shows inside-outside and forward-backward "can be obtained by automatic differentiation" of the inside (respectively forward) computation. In his framing, outside quantities are the adjoints of inside quantities — [Eisner 2016](https://aclanthology.org/W16-5901/).
- **A 2026 restatement.** A 2026 preprint states the same unification for bioinformatic dynamic programs. Backward/outside quantities are adjoints of forward/inside variables, and "posterior item marginals are normalized inside–outside products" — [Asai 2026 (arXiv)](https://arxiv.org/abs/2607.09872).
- **Semiring parsing.** Goodman's "Semiring Parsing" (Computational Linguistics 25(4):573–606) unifies parsers by parameterizing one algorithm over a semiring:
  - boolean: recognition;
  - inside: string probability;
  - Viterbi: probability of the best derivation;
  - counting: number of derivations;
  - derivation forest: the set of derivations;
  - Viterbi-derivation: the best derivation;
  - Viterbi-n-best: the best n derivations.

  Source: [Goodman 1999](https://aclanthology.org/J99-4004/).
- **Expectation semirings.** Li & Eisner extend the semiring view with first- and second-order expectation semirings, for minimum-risk training over packed forests (EMNLP 2009, pp. 40–51) — [Li & Eisner 2009](https://aclanthology.org/D09-1005/).
- **Packed forest size.** Billot & Lang show that a shared (packed) forest of at most cubic size exists for any context-free grammar (ACL 1989, pp. 143–151) — [Billot & Lang 1989](https://aclanthology.org/P89-1018/).

### Inferences
**Definitions in TRELLIS notation.**

Let φ(i,k,j,B,C→A) ≥ 0 be the potential of building cell label A over i..j from children labelled B over i..k and C over k+1..j. Then:

```
inside   α(i,i,A)   = φ_leaf(i,A)
         α(i,j,A)   = Σ_{k,B,C} φ(i,k,j,B,C→A) · α(i,k,B) · α(k+1,j,C)
         Z          = Σ_A α(0,n-1,A)·φ_root(A)          (or a glue chain; see Implications)
outside  β(0,n-1,A) = φ_root(A)
         β(i,k,B)  += β(i,j,A) · φ(i,k,j,B,C→A) · α(k+1,j,C)     (B as left child)
         β(k+1,j,C)+= β(i,j,A) · φ(i,k,j,B,C→A) · α(i,k,B)       (C as right child)
posteriors μ(i,j)   = Σ_A α(i,j,A)·β(i,j,A) / Z                       ("chunk strength")
           μ(rule)  = β(i,j,A)·φ(i,k,j,B,C→A)·α(i,k,B)·α(k+1,j,C) / Z  (= ∂ log Z / ∂ log φ, Eisner 2016)
```

**Complexity.** With L labels per cell and arbitrary potentials, the cost is O(n³L³) time and O(n²L) space. In TRELLIS the label A of a cell is a deterministic function of the item: Cobweb sorts the item's content and context instances. So the effective cost is O(n³K²) item evaluations when each cell keeps K child-label hypotheses.

**What "strength" means.** The user's phrase "quantify strength" maps directly onto μ(i,j): the share of the sentence's total analysis weight in which words i..j form a chunk. Summed over a corpus, Σ μ gives a soft frequency for each chunk concept. That soft frequency could replace raw instance counts in the climbing-ancestor τ and the maturity threshold (count ≥ 50).

**TRELLIS v1 is the fully-bracketed case of Pereira & Schabes.** Gold unlabeled trees leave only the labels latent, and Cobweb's categorization acts as a hard E-step over labels. So v1's learning signal was Pereira & Schabes' "best situation".

The 37.35% vs 90.36% gap is the clearest warning for an unsupervised v2: likelihood-driven inside-outside without bracket constraints may find structure that does not match phrase structure. Partial brackets can come from several places:
- punctuation (see Section 4, Ponvert et al.);
- a teacher marking a few chunks, which fits the Teachable-AI connection;
- the parser's own high-confidence posteriors (Implications, Option 4).

Each of these restores much of the benefit and also cuts cost.

**Normalization.** The inside (sum-product) semiring needs only non-negative weights. So Cobweb class posteriors can be used as log-potentials, s ↦ exp(s/T). Z is then a partition function over trees, and μ is a proper posterior of the Gibbs distribution P(T) ∝ Π exp(s/T).

Classical EM re-estimation, however, assumes a generative, normalized model (a PCFG). With discriminative potentials you have two choices:
1. CRF-style learning: the gradient is observed minus expected counts.
2. A "Cobweb M-step": feed μ-weighted instances to `ifit`, which currently has no weight argument.

**Semiring menu for one TRELLIS chart engine:**

| Semiring | TRELLIS use |
|---|---|
| Boolean | Does any τ-recognized full parse exist? (a grammaticality judgment) |
| Counting | Ambiguity of the sentence under the gate |
| Inside | Z and the μ marginals |
| Viterbi | Best tree |
| k-best | Candidates for reranking with non-local chunk context |
| Expectation | Expected depth or number of chunks, and posterior entropy as an "uncertainty" signal |

### Gaps
- The Lari & Young full text was not consulted (paywalled), so the cubic cost is stated from Pereira & Schabes' discussion and the standard analysis.
- No published work runs inside-outside over Cobweb-derived potentials, and no work characterizes how Cobweb class posteriors behave as Gibbs potentials (calibration, temperature). This has to be measured.

---

## 2. Decoding with span posteriors: Viterbi vs MBR, and why "select the best non-intersecting set of spans" is MBR-CKY

### Takeaway
Viterbi decoding returns the single highest-scoring derivation, which is the Bayes decision under 0-1 loss on whole trees. Minimum-Bayes-risk (MBR) decoding instead maximizes the expected number of correct brackets, Σ_{span∈T} μ(span), and it is computed by the same CKY recursion run over span posteriors.

That is precisely "keep a frontier of every recognized analysis going up, then pick the best non-crossing set of spans coming down". Goodman (1996) and Petrov & Klein (2007) show that decoding for the right objective gives consistent gains: max-rule-product beats the Viterbi derivation by 1.7 bracket-score points on WSJ.

A simple corollary also holds: every span with posterior above 0.5 can be kept without conflict, giving a valid partial bracketing.

### Cited Findings
- **Goodman 1996.** Most parsers, including Viterbi, "attempt to optimize ... the probability of getting the correct labelled tree", whatever metric is used for evaluation. Goodman introduces the Labelled Recall algorithm (maximizes the expected labelled recall rate) and the Bracketed Recall algorithm (maximizes expected bracketed recall) (ACL 1996, pp. 177–183) — [Goodman 1996](https://aclanthology.org/P96-1024/).
- **Goodman's results.** Each algorithm does best on the criterion it optimizes. With a treebank grammar induced by counting (1,805 test sentences):

  | Algorithm | Labelled tree rate | Labelled recall | Bracketed recall |
  |---|---|---|---|
  | Labelled Tree (Viterbi) | 4.54% | 48.60% | 60.98% |
  | Labelled Recall | 3.71% | 49.66% | not reported here |
  | Bracketed Recall | not reported here | not reported here | 61.63% |

  Source: [Goodman 1996](https://aclanthology.org/P96-1024/).
- **Speedup for DOP.** Maximizing labelled recall instead of the labelled-tree criterion let Goodman parse with the DOP model by a much simpler algorithm, giving "a 500 times speedup" — [Goodman 1996](https://aclanthology.org/P96-1024/).
- **Petrov & Klein 2007** compare inference procedures for latent-annotation (state-split) PCFGs "from the standpoint of risk minimization" (NAACL-HLT 2007, pp. 404–411). Results:

  | Decoding method | Bracket score |
  |---|---|
  | Viterbi derivation | 89.5 |
  | Variational | 90.8 |
  | Max-rule-sum | 90.9 |
  | Max-rule-product | 91.2 |
  | Reranked 10-best, "exact" objective (for comparison) | 90.8 |
  | Oracle on that 10-best list (for comparison) | 95.0 |

  Source: [Petrov & Klein 2007](https://aclanthology.org/N07-1051/).
- **Latent labels and why they matter.** The latent-annotation setting is the one where marginalizing labels matters:
  - Matsuzaki et al.'s PCFG with latent annotations (ACL 2005) — [Matsuzaki et al. 2005](https://aclanthology.org/P05-1010/).
  - Petrov et al.'s split-merge grammars reach 90.2 on the Penn Treebank — [Petrov et al. 2006](https://aclanthology.org/P06-1055/).
- **Tree averaging (recent).** Shayegh et al. (ICLR 2024) average the outputs of several unsupervised parsers with "a CYK-like algorithm ... to search for the tree that is most similar to all teachers' outputs". This is an MBR consensus over span sets, which they then distill into a single student — [Shayegh et al. 2024](https://arxiv.org/abs/2310.01717).
- **CKY decoding of independent span scores.** Span-factored parsers decode exactly this way. Stern et al. score spans and labels independently and show this "is ... compatible with classical dynamic programming techniques" — [Stern et al. 2017](https://aclanthology.org/P17-1076/).

### Inferences
**MBR-CKY is the user's "lattice" decode.** Given posteriors μ(i,j) from Section 1, the best non-crossing complete binary bracketing is:

```
M(i,i) = 0
M(i,j) = μ(i,j) + max_k [ M(i,k) + M(k+1,j) ]          # Goodman's bracketed-recall decode
```

For binary trees the bracket count is fixed at n-1. So this simultaneously maximizes expected bracket recall and expected bracket precision; in TRELLIS terms it minimizes expected omitted brackets and expected spurious brackets together.

**Partial (forest) outputs, matching v1's "halt when nothing is recognized".** Allow a span to be left unbracketed:

```
M(i,j) = max(0, μ(i,j) - θ) + max_k [ M(i,k) + M(k+1,j) ]
```

The threshold θ trades omitted brackets against spurious brackets. The recursion is complete: any non-crossing family inside [i,j] other than [i,j] itself leaves some boundary that no maximal member straddles.

**The θ = 0.5 shortcut needs no DP.** Two crossing spans can never co-occur in one tree, so their posteriors sum to at most 1, and at most one of them can exceed 0.5. The set {(i,j) : μ(i,j) > 0.5} is therefore always non-crossing. It is also the optimum of the θ = 0.5 objective, because every member adds positive value and the whole set is feasible. This gives TRELLIS a one-line "confirmed chunks" extractor.

**Viterbi vs MBR in TRELLIS.**
- **Viterbi CKY over v1's additive score maximizes the same objective as v1's failed beam search, but exactly.** Since v1's beam lost to greedy at every width, a wider search moved the output away from gold. Exact Viterbi should therefore be expected to do no better and probably worse (see Section 5).
- **MBR is a different decision rule.** It aggregates the evidence for a span across all trees and labelings. This helps when the evidence for a correct span is spread over several labels or splits; Petrov & Klein's +1.7 comes from exactly such latent-label spreading.
- **MBR does not fix a systematic scoring bias.** If rare modifier chunks are always scored far below their distituent competitors, their marginals stay low too. The scoring itself must change; see Section 3 (CCM log-odds) and Implications, Option 2.

**Cobweb labels are latent annotations.** A TRELLIS chart cell's concept label (the context concept at a cut, plus complexity) plays the role of a latent subsymbol. Decoding unlabeled brackets should sum out labels: run MBR on μ(i,j) = Σ_A μ(i,j,A), or max-rule-product on unlabeled anchored splits μ(i,k,j) = Σ_labels μ(rule). It should not take the best single (span, concept) derivation.

### Gaps
- Goodman's Labelled Recall and Bracketed Recall algorithms were evaluated on 1990s grammars. No study tests MBR with non-probabilistic, categorization-derived potentials. How a temperature T affects MBR quality for Cobweb potentials is unknown and must be tuned.

---

## 3. Inside-outside over a span scorer rather than a normalized PCFG: CRF parsers, CCM, DIORA/S-DIORA, compound PCFGs; what breaks context-free factorization and how to repair it

### Takeaway
Dynamic programming over trees needs only one thing: the tree's weight must factor into item potentials that depend on the item, its own label, its children's labels, and the words. The words may be anywhere in the sentence.

TRELLIS v1's span representation is almost entirely of this kind. The one genuinely non-local ingredient is chunk-level context, meaning neighbors described by their chunk concepts.

The closest prior model is Klein & Manning's CCM. It scores every span by a yield (content) model and a context model, separately for constituents and distituents, and runs EM over all binary bracketings: a near-exact analog of TRELLIS's two hierarchies plus a distituent baseline.

DIORA computes literal inside (content) and outside (context) representations per cell, and shows how chunk-level context can be computed in a dynamic-programming-consistent way. Its follow-up S-DIORA shows that cells should hold a few hard hypotheses rather than soft averages, which fits Cobweb's discrete categorization.

Non-local features are handled by outside-informed second passes, by k-best or forest reranking with cube pruning, by belief propagation, or by stacking.

### Cited Findings
**CRF and span-factored parsers (arbitrary scores, not normalized rule probabilities).**
- **Finkel, Kleeman & Manning (ACL-08: HLT, pp. 959–967)** built "the first general, feature-rich discriminative parser, based on a conditional random field model ... scaled to the full WSJ parsing data". Efficiency came from "stochastic optimization ..., parallelization and chart prefiltering".
  - WSJ15: 90.9 bracket score, a 14% relative error reduction, "two orders of magnitude faster".
  - Sentences of length ≤40: 89.0.

  Source: [Finkel et al. 2008](https://aclanthology.org/P08-1109/).
- **Stern, Andreas & Klein (ACL 2017, pp. 818–827)** score labels and spans independently, decode with CKY or with "a novel greedy top-down inference algorithm based on recursive partitioning", and reach 91.79 on the Penn Treebank and 82.23 on the French Treebank — [Stern et al. 2017](https://aclanthology.org/P17-1076/).
- **Kitaev & Klein (ACL 2018, pp. 2676–2686)** reach 93.55 on the PTB with no external data and 95.13 with pretrained word representations. "Separating positional and content information in the encoder" improves accuracy — [Kitaev & Klein 2018](https://aclanthology.org/P18-1249/).
- **Torch-Struct (Rush, ACL 2020 demos, pp. 335–342)** packages tree and other structured distributions behind a "distribution-based API that connects to any deep learning model". It "exploits auto-differentiation" to compute marginals and related quantities — [Rush 2020](https://aclanthology.org/2020.acl-demos.38/).

**CCM: content and context span scoring with EM over binary bracketings (the closest analog to TRELLIS).**
- Klein & Manning's Constituent-Context Model (ACL 2002, pp. 128–135) "describes all contiguous subsequences of a sentence (spans) ..., whether they are constituents or nonconstituents (distituents)":
  - P(S,B) = P(B)·P(S|B).
  - P(S|B) = Π_{⟨i,j⟩} P(α_ij|B_ij)·P(x_ij|B_ij), where α is the yield, x is the context (the preceding and following terminals), and there is "one [multinomial] for constituents (B_ij = c) and one for distituents (B_ij = d)".

  Source: [Klein & Manning 2002](https://aclanthology.org/P02-1017/).
- If P(B) were uniform over all bracketings, including crossing ones, the model would be "equivalent to soft-clustering with two equal-prior classes". Confining P(B) to "be uniform over binary bracketings and zero elsewhere" turns distributional clustering into tree induction. EM then sums only over valid binary trees — [Klein & Manning 2002](https://aclanthology.org/P02-1017/).
- **CCM results** (unlabeled bracket score):
  - WSJ sentences of ≤10 words: 71.1 with treebank tags, 63.2 with induced tags — [Klein & Manning 2002](https://aclanthology.org/P02-1017/).
  - In the 2004 follow-up, CCM 71.9 and the product model DMV+CCM 77.6 on WSJ10 — [Klein & Manning 2004](https://aclanthology.org/P04-1061/).
- A feature-rich (log-linear) CCM variant was later proposed — [Golland, DeNero & Uszkoreit 2012](https://aclanthology.org/P12-2004/).

**DIORA and S-DIORA: neural inside and outside passes.**
- **DIORA (NAACL 2019, pp. 1129–1141)**:
  - The inside vector of a span is "a weighted average of the compositions for the ... possible segmentations", with weights from "a learned compatibility function".
  - The outside vector of a span "is a function of the outside vector of its parent ... and the inside vector of its sibling".
  - Training makes "the outside representations of the leaf cells ... reconstruct the corresponding leaf input word". The single best tree is recovered "using the CKY algorithm and compatibility scores".
  - Full WSJ test, unlabeled binary bracket score: 55.7 ± 0.4 mean and 56.2 max over five restarts, using a trailing-punctuation post-processing heuristic.

  Source: [Drozdov et al. 2019](https://aclanthology.org/N19-1116/).
- **S-DIORA (EMNLP 2020, pp. 4832–4845)** finds that DIORA's "vector averaging approach is locally greedy and cannot recover from errors when computing the highest scoring parse tree in bottom-up chart parsing". S-DIORA "encodes a single tree rather than a softly-weighted mixture of trees by employing a hard argmax operation and a beam at each cell in the chart". It improves the unsupervised WSJ state of the art "by 2.2-6%" depending on fine-tuning data — [Drozdov et al. 2020](https://aclanthology.org/2020.emnlp-main.392/).
- **ReCAT (ICLR 2024)** stacks "contextual inside-outside (CIO) layers" that learn span representations through "bottom-up and top-down passes, where ... a top-down pass combines information inside and outside a span". It does this explicitly to restore "inter-span communications" lost by strictly tree-shaped composition — [Hu et al. 2024 (ReCAT)](https://arxiv.org/abs/2309.16319).

**Compound PCFGs: global context without breaking dynamic programming.**
- In Kim, Dyer & Rush (ACL 2019, pp. 2369–2385), rule probabilities "are modulated by a per-sentence continuous latent variable, which induces marginal dependencies beyond the traditional context-free assumptions".
  - Inference is "collapsed variational inference", with "an amortized variational posterior ... on the continuous variable, and the latent trees ... marginalized with dynamic programming".
  - The MAP tree is approximated by running CKY with z set to the posterior mean μ_φ(x).
  - WSJ (mean over runs, with max in parentheses): compound PCFG 55.2 (60.1) vs neural PCFG 50.8 (52.6). Chinese: 36.0 (39.8) vs 25.7 (29.5).

  Source: [Kim, Dyer & Rush 2019](https://aclanthology.org/P19-1228/).

**Non-local features: reranking, forests, cube pruning, belief propagation, stacking.**
- **Discriminative n-best reranking** (Collins 2000, cited in [Charniak & Johnson 2005](https://aclanthology.org/P05-1022/); journal version [Collins & Koo 2005, CL 31(1):25–70](https://aclanthology.org/J05-1003/)) lets features be "essentially arbitrary functions of the parse trees". Charniak & Johnson's 50-best lists have an oracle bracket score of 96.8, and their MaxEnt reranker reaches 91.0 on sentences of ≤100 words — [Charniak & Johnson 2005](https://aclanthology.org/P05-1022/).
- **Forest reranking** "reranks a packed forest of exponentially many parses". Since "exact inference is intractable with non-local features", it uses "an approximate algorithm inspired by forest rescoring". It reaches 91.7, beating 50-best and 100-best reranking — [Huang 2008](https://aclanthology.org/P08-1067/). The related tools are:
  - cube pruning, in [Chiang 2007, CL 33(2):201–228](https://aclanthology.org/J07-2003/);
  - forest rescoring and cube growing, in [Huang & Chiang 2007](https://aclanthology.org/P07-1019/);
  - lazy k-best extraction, in [Huang & Chiang 2005](https://aclanthology.org/W05-1506/).
- **Belief propagation.** Smith & Eisner formulate parsing "as a graphical model with ... global constraints". With higher-order features that would make exact parsing slower or NP-hard, loopy BP "needs only O(n³) time", and extra features increase runtime "additively rather than multiplicatively" — [Smith & Eisner 2008](https://aclanthology.org/D08-1016/).
- **Stacking.** A second parser uses a first parser's predictions as features — [Martins et al. 2008](https://aclanthology.org/D08-1017/).
- **Dual decomposition** handles combined or global constraints — [Rush et al. 2010](https://aclanthology.org/D10-1001/).

**Recent (2025–2026) grammar-induction work relevant to interpretability.**
- A July 2026 preprint gives every PCFG rule probability "a closed form" via holographic embeddings, motivated by black-box neural rule scorers "leaving rule probabilities without an interpretable mathematical form" — [Yamaki et al. 2026 (preprint)](https://arxiv.org/abs/2607.08063).
- An EMNLP 2025 paper identifies "probability distribution collapse" as the cause of "unnecessarily large yet underperforming grammars" in neural grammar induction, and enables more compact grammars — [Park & Kim 2025](https://arxiv.org/abs/2509.20734).

### Inferences
**What context-free factorization requires.**
1. A tree's score is a sum (or product) of item scores, one per anchored binary rule (i,k,j,B,C→A).
2. Each item score depends only on that item, its own label, its children's labels, and the input words. Words may be anywhere in the sentence: CRF and span parsers condition on the whole sentence.
3. Each cell carries a bounded set of labels.

Anything that depends on how other parts of the sentence are bracketed breaks factorization. It must then be folded into labels (state splitting), approximated (outside passes, belief propagation, iteration), or handled after decoding (reranking).

**Audit of TRELLIS v1 ingredients.**

| v1 ingredient | Depends on | Context-free status | Chart treatment |
|---|---|---|---|
| Child bags (top-3 context concepts at depth 4) | Child's word window, plus the child's own content leaf via content-ref | Local if carried as the child's label | Make it part of the cell label; keep K hypotheses per cell |
| Child complexity tags | Child's internal structure, a function of its children's tags | Local if carried in the label | Label component; optionally coarsen, e.g. to {1, 2, 3+} |
| Context word slots (5 before, 5 after) | Words only | Local, exactly like CCM's context | Precompute per span |
| Context content-ref | The item's own content leaf | Local, per item | Per-item context sort |
| Climbing-ancestor gate (τ = 30) | The item's sort path | Local | Close the item, or apply a soft penalty |
| Score Σ log P(C\|x) in both hierarchies | The item | Local | Potential φ = exp(s/T) |
| Chunk-level context (neighbors' chunk concepts) | Structure outside the span | Non-local | Outside pass, iteration, reranking, or exact left context in an incremental parser |
| Easy-first commit order | Search dynamics | Not applicable (a chart has no order) | Removed |
| Learning while parsing | Changes the scorer within a sentence | Breaks dynamic programming | Freeze the trees per sentence; learn afterwards |

**Inside and outside map onto TRELLIS's two descriptions, but carefully.**
- TRELLIS's word-window context instance is a span-local factor, exactly like CCM's context x_ij. So both TRELLIS instances, content and context, belong in the item potential.
- The algorithmic outside quantity is a third thing: structural context, i.e. how the span fits a complete tree. This is where chunk-level context lives.
- DIORA shows the recursion: the outside of a span comes from its parent's outside and its sibling's inside. A TRELLIS analog is a soft chunk-context instance, a posterior-weighted bag of sibling and parent concept ids computed from pass-1 marginals and categorized by a "structural context" Cobweb tree. This would give chunk context without committing to neighbors.
- TRELLIS's attribute → {value: count} instances already hold bags, so a posterior-weighted bag is representable. Whether the C++ update handles fractional counts correctly must be checked.

**CCM's lesson for v1's "rare modifier chunk" failure.** CCM scores a span by the ratio P(yield,context | constituent) / P(yield,context | distituent). A rare constituent has a low absolute probability, but its distituent probability is low too. A log-odds potential therefore does not penalize rarity, whereas v1's raw log-posterior sum does.

TRELLIS can implement this natively with distituent Cobweb trees. These are trained on adjacent pairs of gold constituents that do not form a gold constituent, which are exactly v1's Verb+Det-type mistakes. The score is then log P(x|tree+) − log P(x|tree−), using the existing `tree.log_prob`. Keeping the distituent trees separate leaves the positive hierarchies untouched, so the generation pools stay as they are.

**S-DIORA's lesson.** Soft averaging inside cells "washes out" trees. TRELLIS cells should hold K hard Cobweb categorizations, each a concept with its score: a per-cell beam with cube pruning. They should not hold a blended instance.

**Compound PCFG's lesson.** Global context can be added without breaking context-freeness by conditioning all potentials on a per-sentence latent variable. In TRELLIS this could be a sentence-level concept sorted from the whole sentence's bag of words, held fixed during the chart pass.

**Reranking as a cheap path to chunk context.** Use k-best extraction or a forest from a pass with only local features. Then rescore each candidate tree with exact chunk-level context, since in a complete tree every neighbor is built. The trade-off is limited k-best diversity, which was Huang's motivation for forest reranking.

### Gaps
- No work combines CCM-style content/context span models with an incremental conceptual-clustering learner.
- The exact form of v1's "top-3 context concept ids at depth 4" bag across different splits of the same span was not profiled, so it is unknown how many distinct child labels per cell (K) arise in practice.
- Fractional-count support in cobweb-private's discrete update was not verified.

---

## 4. Efficiency and incrementality: middle grounds between greedy and full charts

### Takeaway
Several options sit between v1's O(n)-ish greedy parser and an exhaustive O(n³) chart:
- packed forests and GLR/Earley sharing (exact, compact);
- dynamic-programming beam shift-reduce with state merging (linear in practice, incremental, exponentially many trees per beam);
- A* with admissible outside estimates (exact; under 3–5% of edges);
- coarse-to-fine pruning through a hierarchy of coarser grammars (10× to 100× faster with no loss);
- chart constraints that close cells (provably O(n²) or O(n) worst case with no loss);
- learned beam widths per cell.

Coarse-to-fine is the most Cobweb-native: a Cobweb taxonomy literally is a nested hierarchy of coarser categories. The incremental dynamic-programming beam is the most cognitively attractive, because the left chunk context is already built on the stack.

### Cited Findings
- **Packed sharing.** Tomita's generalized LR parser for augmented context-free grammars (Computational Linguistics 13(1–2):31–46) shares parsing work through a graph-structured stack and packed forest — [Tomita 1987](https://aclanthology.org/J87-1004/). Billot & Lang show shared forests of at most cubic size exist for any context-free grammar, and that "sophistication in chart parsing schemata (e.g. use of look-ahead) may reduce time and space efficiency instead of improving it" — [Billot & Lang 1989](https://aclanthology.org/P89-1018/).
- **Earley parsing with prefix probabilities.** Stolcke's probabilistic Earley parser computes, in one framework:
  - (a) "probabilities of successive prefixes being generated by the grammar";
  - (b) substring (inside) probabilities;
  - the most likely parse;
  - parameter estimates.

  (Computational Linguistics 21(2):165–201) — [Stolcke 1995](https://aclanthology.org/J95-2002/). Earley's original algorithm is in Communications of the ACM 13(2):94–102 — [Earley 1970](https://doi.org/10.1145/362007.362035).
- **Dynamic-programming shift-reduce (Huang & Sagae, ACL 2010, pp. 1077–1086).**
  - Greedy or beam shift-reduce "only explores a tiny fraction of the whole space (even with beam search)". Dynamic programming becomes possible "by merging 'equivalent' stacks based on feature values", inspired by Earley parsing and GLR.
  - It runs "in polynomial time in theory, but linear-time (with beam search) in practice", with "up to a five-fold speedup ... with no loss in accuracy". "Better search also leads to better learning".
  - Final English PTB unlabeled dependency accuracy is 92.1%, at 0.04 s per sentence in pure Python.

  Source: [Huang & Sagae 2010](https://aclanthology.org/P10-1110/).
- **Constituency extension.** Mi & Huang extend dynamic-programming shift-reduce to constituency parsing, with a POS-tag lattice input, reaching 90.8% (PTB) and 83.9% (CTB) — [Mi & Huang 2015](https://aclanthology.org/N15-1108/). An earlier deterministic, classifier-based linear-time constituency parser is [Sagae & Lavie 2005](https://aclanthology.org/W05-1513/).
- **A\* parsing (HLT-NAACL 2003, pp. 119–126).** A\* "conservatively estimat[es] the probabilities of parse completions" (outside estimates). The most detailed estimate cuts edges to "less than 3% of that required by exhaustive parsing"; a simpler one to under 5%. It "is guaranteed to find the most likely parse" and "maintains worst-case cubic time" — [Klein & Manning 2003](https://aclanthology.org/N03-1016/).
- **Multilevel coarse-to-fine.** Charniak et al. parse with "a sequence of nested partitions or equivalence classes of the PCFG nonterminals", using coarser results to prune finer levels. Work "is decreased by a factor of ten with no decrease in parsing accuracy" — [Charniak et al. 2006](https://aclanthology.org/N06-1022/).
- **Hierarchical projections.** Petrov & Klein use "a grammar's own hierarchical projections ... for incremental pruning", which "parses up to 100 times faster than the baseline PCFG parser, with no loss in test set accuracy" — [Petrov & Klein 2007](https://aclanthology.org/N07-1051/).
- **Chart constraints.** Roark & Hollingshead classify "word positions by whether or not they can either start or end multi-word constituents", "closing" chart cells. They achieve an O(n²) worst case "without impacting parsing accuracy" — [Roark & Hollingshead 2008](https://aclanthology.org/C08-1094/). The follow-up achieves "either linear or O(N log² N)" worst case, "in some cases improving the accuracy" — [Roark & Hollingshead 2009](https://aclanthology.org/N09-1073/).
- **Beam-width prediction.** Bodenstab et al. learn "the optimal beam-search pruning parameters for each CYK chart cell" with a log-linear model. This decreases parsing time "by 65% over a standard beam-search without any loss in accuracy" and is faster than both the Berkeley parser's coarse-to-fine pruning and chart constraints — [Bodenstab et al. 2011](https://aclanthology.org/P11-1045/).
- **Unsupervised incremental parsers.**
  - Seginer's incremental parser, induced from plain text with "learning and parsing ... local and fast, requiring no explicit clustering or global optimization", scores 75.9 unlabeled on WSJ10 and 57.4 on WSJ40, from plain text — [Seginer 2007](https://aclanthology.org/P07-1049/).
  - Ponvert et al. show that cascaded finite-state chunkers (unsupervised partial parsing) "outperform[] CCL by a wide margin" for English, German and Chinese. They also study "phrasal punctuation as a heuristic indicator of phrasal boundaries" — [Ponvert, Baldridge & Erk 2011](https://aclanthology.org/P11-1108/).
- **All-subtrees approach.** The table in Seginer 2007 lists U-DOP at 78.5 and UML-DOP at 82.9 on WSJ10, parsing from part-of-speech tags — [Seginer 2007](https://aclanthology.org/P07-1049/) reporting [Bod 2006](https://aclanthology.org/P06-1109/).

### Inferences
**Approximate Cobweb sort counts per sentence.**
- **v1 easy-first with a pair cache:** about 3n pair evaluations (initial n−1, then about 2 new pairs per step), each costing a content sort and a context sort.
- **Full chart:** n(n²−1)/6 split triples; about 165 at n = 10, 1,330 at n = 20 and 10,660 at n = 40. With K child labels per cell this is ×K².
- **Content sorts are position-independent.** The content instance depends only on the two child labels (bags plus complexity tags), so a corpus-wide memo keyed on (B,C) bounds content sorts by the number of distinct label pairs, not by n³.
- **Context sorts are the real cost**, because the instance includes the span's word window and the content-ref leaf. One option is to drop content-ref during chart scoring and reattach it for the chosen tree. That makes context sorts O(n²) per sentence, cacheable on (window, leaf).

**Pruning with the existing recognition threshold.** The climbing-ancestor gate (τ) maps onto chart constraints and cell closing: an item that is never recognized is a closed item. The difference from v1 is that in greedy parsing the gate is also the halting rule. In a chart, halting becomes a "glue" chain over top-level fragments with a per-fragment penalty, in the style of Hiero's glue rules in [Chiang 2007](https://aclanthology.org/J07-2003/). This guarantees Z > 0 and allows partial parses.

The first diagnostic should be chart coverage, the analog of the n-best oracle: what fraction of gold spans survive the τ gate with non-zero inside mass? If rare modifier chunks fail the gate, no decoder can recover them.

**Cobweb is a ready-made coarse-to-fine ladder.** Cutting the context and content taxonomies at depth d gives Charniak et al.'s "nested partitions", and Petrov & Klein's projections come from split-merge refinement, the same kind of hierarchy Cobweb's split and merge operators build. The plan:
1. Coarse pass with depth-2 concepts (cheap, truncated descent).
2. Prune items with low coarse posterior.
3. Fine pass with full-depth concepts.

**Cell closing with Cobweb.** Add BEGIN/END attributes to primitive (word) context instances during training: from gold trees in supervised mode, from posteriors in unsupervised mode. `tree.predict` can then supply P(word i begins a multi-word chunk), giving Roark & Hollingshead-style constraints from the word-level context hierarchy TRELLIS already has.

**A\* with log-posterior scores.** Cobweb log-posteriors are at most 0, so h = 0 is an admissible outside estimate. That gives uniform-cost agenda parsing, which is exact but saves little. Real savings need informative bounds, such as the best possible score of any item covering each word, precomputed from the cache.

**Incremental dynamic programming fits chunk context better than a whole-sentence chart.** In a left-to-right shift-reduce parser the stack holds already-built chunks, so left chunk context is exact. Right context is only lexical lookahead, and Cobweb categorizes partial instances, with missing right-window slots, natively. Huang & Sagae's state merging requires the signature to contain every feature future scoring uses: the top two stack labels and spans, plus the input position.

### Gaps
- Per-sort latency of the C++ Cobweb tree was not measured, so the absolute per-sentence costs above are counts, not times.
- No study applies coarse-to-fine pruning with a conceptual-clustering hierarchy; the analogy rests on the structure of the methods, not on evidence.

---

## 5. Fixing the greedy parser itself: easy-first, dynamic oracles, imitation learning, and why v1's beam and supervised ranker lost

### Takeaway
TRELLIS v1's "best pair anywhere" loop is Goldberg & Elhadad's easy-first algorithm. Easy-first is competitive with global models because it commits confident decisions first and conditions later ones on structure already built, an inductive bias that a beam over elaborations scored additively throws away.

v1's two negative results have well-documented causes in the literature:
- **Beam worse at every width:** exact or better search exposes model errors in a mis-specified objective (exact NMT search prefers the empty translation for over 50% of sentences). Locally-normalized scores also suffer label bias, and search-unaware training never teaches the scorer to compare partial analyses.
- **Supervised ranker hurt by distribution shift:** exposure bias, which dynamic oracles, DAgger, SEARN and LOLS were designed to fix.

### Cited Findings
- **Easy-first (NAACL-HLT 2010, pp. 742–750).**
  - The algorithm "builds a dependency tree by iteratively selecting the best pair of neighbours to connect at each parsing step". This "allows incorporation of features from already built structures both to the left and to the right". It is "deterministic, best-first, O(n log n)".
  - It "learns both the attachment preferences and the order in which they should be performed".
  - Training: when an invalid action is chosen, the parser updates toward "the currently highest scoring valid action" and retries until a valid action is chosen.
  - PTB section 23 unlabeled accuracy (automatic POS tags, including punctuation): easy-first 89.70 vs MaltParser 88.36 vs MSTParser 90.05.

  Source: [Goldberg & Elhadad 2010](https://aclanthology.org/N10-1115/).
- **Easy-first with tree LSTMs.** A greedy bottom-up easy-first parser over hierarchical tree LSTMs achieves "very strong accuracies for English and Chinese" — [Kiperwasser & Goldberg 2016](https://aclanthology.org/Q16-1032/).
- **Dynamic oracles.** The arc-eager dynamic oracle "provides a set of optimal transitions for every valid parser configuration, including configurations from which the gold tree is not reachable". Training with exploration gives "an average improvement of over 1.2 LAS points and up to almost 3 LAS points" — [Goldberg & Nivre 2012](https://aclanthology.org/C12-1059/).
- **Why static oracles fail.** Static oracles "are only valid as long as the parser does not stray from this path". Dynamic oracles allow exploring "alternative and nonoptimal paths during training". Improvement comes "at no cost in terms of efficiency, unlike other techniques like beam search" — [Goldberg & Nivre 2013 (TACL 1:403–414)](https://aclanthology.org/Q13-1033/).
- **Constituency dynamic oracles.**
  - Cross & Huang design "the first provably optimal dynamic oracle for constituency parsing, which runs in amortized O(1) time" for a span-based shift-reduce system. On PTB development data, the static oracle and the dynamic oracle both give 91.38, and dynamic oracle plus exploration (α = 1.0) gives 91.64 — [Cross & Huang 2016](https://aclanthology.org/D16-1001/).
  - A dynamic oracle for greedy transition-based constituent parsing was evaluated on 9 SPMRL languages — [Coavoux & Crabbé 2016](https://aclanthology.org/P16-1017/).
- **DAgger (AISTATS 2011).** Sequential prediction "violate[s] the common i.i.d. assumptions". DAgger "trains a stationary deterministic policy" as "a no regret algorithm in an online learning setting" — [Ross, Gordon & Bagnell 2011](https://arxiv.org/abs/1011.0686).
- **SEARN** is search-based structured prediction (Machine Learning 75(3):297–325) — [Daumé, Langford & Marcu 2009](https://doi.org/10.1007/s10994-009-5106-x).
- **LOLS (ICML 2015)** "does well relative to the reference policy, but additionally guarantees low regret compared to deviations from the learned policy". It "can improve upon the reference policy" even "when the reference is poor" — [Chang et al. 2015](https://arxiv.org/abs/1502.02206).
- **Search errors versus model errors.** With exact search in NMT, "beam search fails to find these global best model scores in most cases". For more than 50% of sentences, "the model in fact assigns its global best score to the empty translation". The authors conclude that NMT "requires just the right amount of beam search errors" — [Stahlberg & Byrne 2019](https://aclanthology.org/D19-1331/). Exact MAP decoding "frequently leads to low-quality results", and beam search's success is attributed to an implicit uniform-information-density bias — [Meister, Cotterell & Vieira 2020](https://aclanthology.org/2020.emnlp-main.170/).
- **Label bias.** "The label bias problem implies that globally normalized models can be strictly more expressive than locally normalized models" — [Andor et al. 2016](https://aclanthology.org/P16-1231/).
- **Training for inexact search.** Early update for beam-search perceptrons — [Collins & Roark 2004](https://aclanthology.org/P04-1015/). Huang et al.'s "violation-fixing" framework "subsumes and justifies" early update; LaSO (Daumé & Marcu, 2005) is a special case, and "max-violation" cuts training time threefold — [Huang, Fayong & Guo 2012](https://aclanthology.org/N12-1015/).

### Inferences
**Why v1's beam lost, with ranked hypotheses.**
1. **Model error, not search error.** "Worse at every width" is the signature of an objective that prefers non-gold analyses: the Stahlberg & Byrne pattern. Raw log-posterior sums penalize rare modifier chunks. Fix the potentials (CCM log-odds, Section 3) before widening the search.
2. **Local normalization (label bias).** Each Cobweb posterior is normalized over sibling concepts for one instance, not across competing analyses. Global comparison of partial analyses is not what it was built for (Andor et al.).
3. **Spurious ambiguity among merge orders.** This is a hypothesis to check in v1's beam code. The number of bottom-up merge sequences yielding a given binary tree is the number of linear extensions of its internal-node order, (n−1)! / Π_v h(v), where h(v) counts the internal nodes under v (the standard hook-length count for trees).
   - A caterpillar tree has 1 order; a balanced 8-leaf tree has 80.
   - If v1's beam did not merge identical frontiers, permutations of the same partial tree crowded out alternatives. Huang & Sagae's state merging removes exactly this.
   - The same bias would affect any sampler over merge sequences: it over-weights balanced trees.
4. **The scorer was never trained for search.** Early or max-violation updates (Collins & Roark; Huang et al.) are the standard remedy when beam search is used with a learned scorer.

**Why the supervised step-ranker lost: exposure bias.** It was trained on gold-path frontiers but tested on its own. The cheapest fix is a dynamic oracle for TRELLIS's span-based easy-first system. Since the bracket loss decomposes over spans (Cross & Huang's insight), the cost of merging adjacent frontier nodes a and b given frontier F is:

```
cost(a,b | F) = #{ r ∈ R : r is still reachable from F and r crosses span(a)∪span(b) } + [span(a∪b) ∉ R]
```

Here R is the reference bracket set. The allowed (zero-cost) merges are those that destroy no reachable reference span. DAgger then aggregates the states the learned ranker actually visits, labelled by argmin cost.

**Unsupervised variant.** R can be the chart's MBR tree rather than gold. LOLS is the relevant theory here, since it allows a learned greedy policy to surpass a suboptimal reference.

**This gives a coherent division of labor:** a slow, whole-sentence inside-outside "teacher" for learning and analysis, and a fast easy-first "student" for online processing, trained by imitation with exploration. This mirrors Shayegh et al.'s ensemble-then-distill (Section 2).

### Gaps
- v1's beam implementation was not inspected to confirm whether identical frontiers were merged, so hypothesis 3 is untested.
- No dynamic-oracle work exists for non-directional constituency easy-first. The cost formula above is derived, not taken from a paper.

---

## 6. EM variants for an incremental Cobweb learner: online/stepwise, hard vs soft, lateen, and split-merge

### Takeaway
Online (stepwise) EM is the incremental version of inside-outside. It often converges faster than batch EM and to better solutions, and it is a natural fit for Cobweb's sentence-at-a-time updates.

Hard (Viterbi) EM, which trains on the single best parse, beat soft inside-outside for unsupervised dependency induction. That is good news for a TRELLIS that learns from its own parses.

Lateen EM alternates the two objectives to escape local optima. Split-merge EM over latent subsymbols is the closest statistical analog to Cobweb's split and merge operators.

A standing caution: unsupervised objectives are "provably wrong" proxies for linguistic structure, so constraints (brackets, depth bounds) matter more than the choice of optimizer.

### Cited Findings
- **Online EM (Liang & Klein, NAACL-HLT 2009, pp. 611–619).** They study "incremental EM (Neal and Hinton, 1998) and stepwise EM (Sato and Ishii, 2000; Cappé and Moulines, 2009)".
  - Stepwise EM interpolates sufficient statistics with stepsize η_k = (k+2)^−α, where "any 0.5 < α ≤ 1 is valid"; smaller α means "the more quickly we forget". A mini-batch size m adds stability.
  - Stepwise EM "reaches the same performance as batch EM, but much more quickly" and "can even surpass" it. In part-of-speech tagging, batch EM reaches 57.3% after 100 iterations, while stepwise EM reaches 65.4% after two.

  Source: [Liang & Klein 2009](https://aclanthology.org/N09-1069/).
- **Primary online-EM references:**
  - [Sato & Ishii 2000, Neural Computation 12(2):407–432](https://doi.org/10.1162/089976600300015853);
  - [Cappé & Moulines 2009, JRSS-B 71(3):593–613](https://doi.org/10.1111/j.1467-9868.2009.00698.x).
- **Viterbi training (CoNLL 2010, pp. 9–17).** Viterbi training instead of inside-outside re-estimation for the DMV "is more accurate than standard inside-outside re-estimation (classic EM), significantly faster, and simpler". It reaches 44.8% on WSJ section 23 (all sentences) "without clever initialization", 47.9% with a good initializer, and 50.8% on Brown. The authors argue that "objective functions used in unsupervised grammar induction are provably wrong", so "advantages of exact inference may not apply" — [Spitkovsky et al. 2010](https://aclanthology.org/W10-2902/).
- **Hardness of Viterbi training.** Cohen & Smith give hardness results for Viterbi training of PCFGs and show uniform initialization is competitive — [Cohen & Smith 2010](https://aclanthology.org/P10-1152/).
- **Lateen EM (EMNLP 2011, pp. 1269–1280).** It "alternates between the two objectives of ordinary 'soft' and 'hard' expectation maximization". "Switching objectives when stuck can help escape local optima"; a single alternation "already yields state-of-the-art results for English dependency grammar induction". Lateen strategies "significantly speed up training of both EM algorithms, and improve accuracy for hard EM" — [Spitkovsky, Alshawi & Jurafsky 2011](https://aclanthology.org/D11-1117/).
- **Split-merge.** Nonterminals "are alternately split and merged to maximize the likelihood of a training treebank". The learned subsymbols reproduce linguistic distinctions, are "much more compact", and reach 90.2 on the Penn Treebank — [Petrov et al. 2006](https://aclanthology.org/P06-1055/).
- **Hard EM at scale.** GPST trains an unsupervised syntactic language model plus a composition model that "induces syntactic parse trees", jointly and in parallel "in a hard-EM fashion", pre-trained on 9B tokens — [Hu et al. 2024 (GPST, ACL 2024)](https://arxiv.org/abs/2403.08293).
- **Depth bounds.**
  - Depth-bounded PCFG induction (TACL 6:211–224) acquires grammars that "demonstrate a consistent use of category labels" — [Jin et al. 2018a](https://aclanthology.org/Q18-1016/).
  - Switching bounding on and off within one chart-based Bayesian inducer shows depth-bounding "is indeed significantly effective in limiting the search space ... and thereby increasing accuracy" — [Jin et al. 2018b](https://aclanthology.org/D18-1292/).

### Inferences
**What a TRELLIS "E-step" and "M-step" look like.**
- **E-step:** chart inside-outside under the current, frozen hierarchies gives μ(rule) for every candidate composite.
- **M-step, three grades:**
  - (a) **Hard / Viterbi:** `ifit` only the decoded tree's composites. This is closest to v1 and to Spitkovsky et al.
  - (b) **Thresholded "confirmed chunks":** `ifit` only composites with μ ≥ θ_learn. With θ ≥ 0.5 these are guaranteed non-crossing (Section 2). This literally implements the user's idea, in INSIDE_OUTSIDE.md, of maintaining candidates "and then learn them once we can confirm that they're good enough" ([INSIDE_OUTSIDE.md](/Users/karthiksing05/Documents/ISLE-Research/ChunkingCobweb/INSIDE_OUTSIDE.md)).
  - (c) **Soft stepwise:** `ifit` every composite with weight η_k·μ, decaying old counts by (1−η_k). This needs a weight argument added to `ifit` and to the count updates. Cobweb's cumulative counts correspond to η_k ≈ 1/k, i.e. α = 1 with no forgetting.

**Lateen schedule.** Alternate (a) and (c) whenever bracket stability or held-out sentence likelihood stops improving.

**Self-bracketing with Pereira & Schabes.** Grade (b) followed by constrained inside-outside on the next pass turns the model's confident spans into partial brackets. This cuts cost and anchors structure.

**Guard against self-reinforcement.** v1's rule was that "parser output never feeds back to memory". An unsupervised v2 must break that rule, so it needs safeguards:
- a burn-in on short sentences;
- depth bounds (Jin et al.; Noji et al., Section 7);
- separate distituent trees, so that commission evidence also accumulates;
- periodic re-sorting of instances, Cobweb's standard defence against order effects.

**Split-merge and Cobweb.** Petrov et al.'s split-merge EM and Cobweb's split/merge operators both grow a hierarchy of category refinements under a likelihood-like criterion. Cobweb uses category utility; Petrov uses treebank likelihood with merge-back of unhelpful splits.

This supports treating Cobweb concepts as latent subsymbols of chunks and decoding with labels marginalized (Section 2). It also suggests reusing Petrov & Klein's coarse-to-fine pruning via the hierarchy (Section 4).

### Gaps
- Liang & Klein did not evaluate PCFG or inside-outside tasks (their four tasks are POS tagging, document classification, word segmentation and word alignment), so their stepwise-EM gains for chart-based grammar learning are an extrapolation.
- No work evaluates EM whose M-step is incremental conceptual clustering.

---

## 7. Cognitive plausibility of chart vs greedy parsing

### Takeaway
Exhaustive whole-sentence charts are a computational-level idealization, not a plausible online mechanism. Humans process incrementally under severe memory limits (the Now-or-Never bottleneck; 3–4 memory elements suffice for almost all treebank sentences under left-corner processing), garden-path, and often settle for "good-enough" partial representations.

Psycholinguistic models that weigh all parses do so through prefix probabilities and surprisal, which require a normalized, incremental, generative model. Mechanistic accounts use bounded parallelism (beam pruning, particle filters) or serial parsing with memory retrieval.

v1's easy-first parser is greedy but non-directional: it needs the whole sentence before acting, so it is not incremental in the psycholinguistic sense either. The most defensible TRELLIS v2 story is dual-route:
- incremental, bounded-parallel, left-corner-like processing online;
- whole-sentence inside-outside for learning, consolidation and evaluation.

### Cited Findings
**Surprisal and prefix probabilities.**
- Hale defines cognitive load as "the surprisal of word w_i given its prefix", computed with Stolcke's probabilistic Earley parser. It "correctly predicts processing phenomena associated with garden path structural ambiguity and with the subject/object relative asymmetry" — [Hale 2001](https://aclanthology.org/N01-1021/).
- Levy develops expectation-based (surprisal) comprehension theory (Cognition 106(3):1126–1177) — [Levy 2008](https://doi.org/10.1016/j.cognition.2007.05.006).
- Incremental top-down parsing as a language model — [Roark 2001](https://aclanthology.org/J01-2004/). Its lexical and syntactic expectation measures for psycholinguistic modeling — [Roark et al. 2009](https://aclanthology.org/D09-1034/).
- Eye-tracking corpora as evidence for theories of syntactic processing complexity (Cognition 109(2):193–210) — [Demberg & Keller 2008](https://doi.org/10.1016/j.cognition.2008.07.008).

**Particle filters.** Levy, Reali & Griffiths note that most models "are non-incremental, have run time superlinear in input length, and/or enforce structural locality constraints". Their limited-memory particle-filter parser "can reproduce classic results in online sentence comprehension". It gives "the first rational account" of the "digging-in" effect, where a preferred alternative "seems to grow more attractive over time even in the absence of strong disambiguating information" — [Levy, Reali & Griffiths, NIPS 21 (2008 conf.)](https://papers.nips.cc/paper_files/paper/2008/hash/a02ffd91ece5e7efeb46db8f10a74059-Abstract.html).

**Garden paths, bounded parallelism and reanalysis.**
- Eye-movement evidence on "making and correcting errors" in structurally ambiguous sentences — [Frazier & Rayner 1982](https://doi.org/10.1016/0010-0285(82)90008-1).
- A probabilistic, ranked-parallel model of lexical and syntactic access and disambiguation, in which pruning low-probability analyses explains garden paths (Cognitive Science 20(2):137–194) — [Jurafsky 1996](https://doi.org/10.1207/s15516709cog2002_1).
- "What a rational parser would do" (Cognitive Science 35(3):399–443) — [Hale 2011](https://doi.org/10.1111/j.1551-6709.2010.01145.x).
- Activation-based (ACT-R) sentence processing as skilled memory retrieval — [Lewis & Vasishth 2005](https://doi.org/10.1207/s15516709cog0000_25).
- Locality and integration-cost theory (Cognition 68(1):1–76) — [Gibson 1998](https://doi.org/10.1016/S0010-0277(98)00034-1).

**Left-corner parsing and memory limits.**
- Center-embedding causes difficulty while left- and right-branching do not. The key distinction between parsing methods "is not the form of prediction (top-down vs. bottom-up vs. left-corner), but rather the ability to instantiate the operation of composition" — [Resnik 1992](https://aclanthology.org/C92-1032/). The prior argument is in [Abney & Johnson 1991](https://doi.org/10.1007/BF01067217).
- Schuler et al. (CL 36(1):1–30) parse within a memory store "possibly constrained to as few as three or four distinct elements". Their model recognizes constituents in a right-corner (left-corner variant) transformed representation and maps it to a Hierarchical HMM. Coverage:

  | Corpus and transform | 3 stack elements | 4 stack elements |
  |---|---|---|
  | Switchboard, right-corner, no punctuation | 99.53% | 99.99% |
  | WSJ sections 2–21, right-corner, no punctuation | 97.66% | 99.96% |
  | WSJ sections 2–21, with punctuation | 93.28% | 99.54% |

  Source: [Schuler et al. 2010](https://aclanthology.org/J10-1001/).
- **Left-corner bounds in grammar induction.** "Tabulation of left-corner parsing ... captures the degree of center-embedding of a parse via its stack depth". Restricting the dependency model with valence (DMV) this way "often ... boosts the performance" across Universal Dependencies languages — [Noji, Miyao & Johnson 2016](https://aclanthology.org/D16-1004/). See also the depth-bounded PCFG work [Jin et al. 2018a](https://aclanthology.org/Q18-1016/).
- Left-corner RNNGs "outperformed top-down RNNGs and LSTM" at modeling Japanese reading times — [Yoshida, Noji & Oseki 2021](https://aclanthology.org/2021.emnlp-main.235/).

**Good-enough processing and the Now-or-Never bottleneck.**
- Good-enough representations in comprehension — [Ferreira, Bailey & Ferraro 2002](https://doi.org/10.1111/1467-8721.00158); [Ferreira & Patson 2007](https://doi.org/10.1111/j.1749-818x.2007.00007.x).
- The Now-or-Never bottleneck as a fundamental constraint on language (Behavioral and Brain Sciences 39; published online 2015) — [Christiansen & Chater 2016](https://doi.org/10.1017/S0140525X1500031X).
- A chunk-based, incremental model of child language learning as language use (Psychological Review 126(1):1–51) — [McCauley & Christiansen 2019](https://doi.org/10.1037/rev0000126).
- Lossy-context surprisal — [Futrell, Gibson & Levy 2020](https://doi.org/10.1111/cogs.12814).
- A resource-rational model of processing recursive structure — [Hahn et al. 2022](https://doi.org/10.1073/pnas.2122602119).

**Strongly incremental constituency parsing.** Humans "grow a single parse tree by adding exactly one token at each step"; the attach-juxtapose transition system makes constituency parsing strongly incremental — [Yang & Deng 2020](https://arxiv.org/abs/2010.14568).

**Grammar induction as a cognitive model (2026).** Statistical grammar induction has been used to operationalize competing maturational theories of syntactic development — [Marcheva, Salhan & Sun 2026 (CogSci)](https://arxiv.org/abs/2605.08476).

### Inferences
**Psycholinguistic standing of each parser type.**
- **v1 greedy (easy-first):** psycholinguistically it is a serial parser with no reanalysis, which would predict unrecoverable garden paths. It is also non-incremental, because it searches the whole sentence for the best pair. Its eager chunking and partial-parse halting do align with "chunk-and-pass" and "good-enough" processing.
- **Full inside-outside:** best defended as a computational-level or learning-time process: offline consolidation over a buffered sentence that computes chunk strengths and expected counts. It is not a model of word-by-word comprehension.
- **Incremental bounded-parallel parsers:** a dynamic-programming beam (Huang & Sagae) or a particle filter (Levy et al.), ideally left-corner-like with a 3–4 chunk memory bound (Schuler et al.). These are the cognitively credible online mechanisms. Their prefix quantities can produce surprisal, but only if TRELLIS supplies normalized generative probabilities.

**Surprisal from TRELLIS's generation side.** The generation side, which samples a decomposition given a parent concept, is TRELLIS's generative model. If it can be normalized, it could supply prefix probabilities without changing the locked generation behaviour. Scores from the recognition side (Cobweb class posteriors) yield only an unnormalized "prefix partition function".

**A memory bound as a learning bias.** Bounding the number of open chunks to about 4 is both cognitively motivated and, per Noji et al. and Jin et al., beneficial for grammar induction.

### Gaps
- No empirical comparison of greedy chunk-based learners (e.g., McCauley & Christiansen's) against chart-based learners on the same human data was found.
- The psycholinguistic claims above about Jurafsky 1996, Hale 2011 and the good-enough literature are summarized from these well-known papers' standard characterizations. Only bibliographic data, not full texts, were checked in this session.

---

## Implications for TRELLIS v2

### Takeaway
TRELLIS should not replace Cobweb with a grammar. It should keep every v1 ingredient as the scorer of chart items, and replace greedy commitment with:
1. a bottom-up inside pass over all recognized compositions (the user's "frontier of valid parses going up");
2. an outside pass that turns scores into chunk posteriors μ (the "strength" of each chunk);
3. a top-down MBR decode of the best non-crossing span set (the "best non-intersecting one going down").

Two warnings shape the design:
- Exact Viterbi over v1's additive score is expected to fail like the beam did, so the potentials must change. CCM-style constituent-vs-distituent log-odds is the principled fix.
- Chunk context is non-local, so it should come from an outside-informed second pass, from reranking, or from the left stack of an incremental parser.

### Cited Findings
These are the empirical anchors for the options below, all detailed and sourced in Sections 1–7:
- **MBR decoding beats the Viterbi derivation:** 89.5 to 91.2 for latent-label grammars ([Petrov & Klein 2007](https://aclanthology.org/N07-1051/)).
- **Optimize the metric you report:** each decoder wins on its own criterion ([Goodman 1996](https://aclanthology.org/P96-1024/)).
- **A content+context span model with a distituent baseline works with EM over binary bracketings:** 71.1 on WSJ10 unsupervised; 77.6 when combined with dependencies ([Klein & Manning 2002](https://aclanthology.org/P02-1017/); [Klein & Manning 2004](https://aclanthology.org/P04-1061/)).
- **Hard per-cell beams beat soft averaging in inside-outside chart encoders:** +2.2–6 points ([Drozdov et al. 2020](https://aclanthology.org/2020.emnlp-main.392/)).
- **Bracket constraints matter:** 37.35% to 90.36%, and per-iteration time becomes linear under full bracketing ([Pereira & Schabes 1992](https://aclanthology.org/P92-1017/)).
- **Exact search exposes model errors** ([Stahlberg & Byrne 2019](https://aclanthology.org/D19-1331/)).
- **Incremental DP beams are linear in practice and search far more of the space than plain beams** ([Huang & Sagae 2010](https://aclanthology.org/P10-1110/)).
- **Coarse-to-fine and cell closing give 10×–100× savings or provable O(n²)/O(n) bounds with no loss** ([Charniak et al. 2006](https://aclanthology.org/N06-1022/); [Petrov & Klein 2007](https://aclanthology.org/N07-1051/); [Roark & Hollingshead 2009](https://aclanthology.org/N09-1073/)).
- **Online and hard EM can match or beat batch soft EM** ([Liang & Klein 2009](https://aclanthology.org/N09-1069/); [Spitkovsky et al. 2010](https://aclanthology.org/W10-2902/)).
- **Dynamic oracles and DAgger fix greedy exposure bias at no test-time cost** ([Goldberg & Nivre 2013](https://aclanthology.org/Q13-1033/); [Ross et al. 2011](https://arxiv.org/abs/1011.0686)).
- **Memory-bounded, left-corner processing is cognitively supported:** 3–4 elements cover 97.7–99.96% of WSJ ([Schuler et al. 2010](https://aclanthology.org/J10-1001/)).

### Inferences

#### Compatibility summary (n words; K child-label hypotheses per cell)

| Method | Needs normalized probabilities? | Cobweb sorts per sentence | Cache | Pruning hook | Chunk context | Incremental? |
|---|---|---|---|---|---|---|
| v1 easy-first greedy | No (ranking only) | ~3n pair evaluations with pair cache | Pairs | τ gate = admission and halting | Hard (neighbors unbuilt) | No (non-directional) |
| Viterbi CKY | No | O(n³K²) | Content on (B,C); context on (window, leaf) | τ closes items | Local only | No |
| Inside-outside + MBR | No: Gibbs potentials exp(s/T); temperature matters | Same as CKY; the outside pass needs no new sorts | Same | τ, plus posterior pruning | Via outside-informed pass 2 | No |
| Generative EM (true re-estimation) | Yes | Same | Same | — | — | Online variants |
| k-best / forest reranking | No | Chart + k trees × n | Same | — | Exact per candidate | No |
| A\* / agenda | No (needs admissible bound; h = 0 is valid) | ≤ chart; <3–5% with good bounds (PCFG evidence) | Same | τ | No | No |
| Coarse-to-fine via Cobweb depth cut | Uses coarse posteriors | Cheap coarse chart + fine on survivors | Same | μ_coarse < ε | No | No |
| Chart constraints (BEGIN/END) | No | Worst case O(n²) or O(n) | Same | Closed cells | No | Partly |
| DP shift-reduce beam | No for Viterbi; yes for prefix probabilities and surprisal | O(n·b) | Same | τ per reduce | Left context exact | Yes |
| Particle filter over merges | Needs target/proposal weights | O(P·n) | Same | τ | Left, or whole sentence | Yes |
| Easy-first + dynamic oracle / DAgger | No | O(n) at test time | Pairs | τ | As v1 | As v1 |

#### Option 1 (rank 1): Cobweb-potential inside-outside chart with MBR decoding ("C-IO-MBR")

**Idea.** Keep v1's instance builders and Cobweb sorts. Make every candidate composition (i,k,j) with child labels (B,C) a chart item with potential φ = exp(s/T). Run inside-outside to get chunk posteriors μ, then decode by MBR, optionally thresholded to give partial parses.

**Pros:**
- Directly realizes "use inside-outside to quantify strength" and the user's lattice description.
- Cobweb remains the only knowledge source, which answers the earlier "straying" objection to the induced-grammar CKY. Every item is a Cobweb categorization; the chart only memoizes all merge orders.
- Decoding is O(n³) over μ.
- μ is an interpretable confidence that can be drawn as a chart heatmap for the tree-eyeballing workflow.
- The posterior over concept labels per span gives a structure-aware categorization, which bears on the "clean POS/phrase separation" goal.
- Generation is untouched.

**Cons:**
- O(n³K²) item evaluations.
- The potentials' calibration matters, so T must be swept.
- With v1's raw scores this may still inherit the rare-chunk bias; Option 2 fixes that.
- Chunk context is not yet included; that is Option 3.

**Pseudo-algorithm** (use log space, i.e. logsumexp, throughout):

```
INPUT  words w[0..n-1]; frozen trees CNT+, CTX+ (optionally CNT-, CTX- from Option 2);
       temperature T; per-cell beam K; gate τ (hard or soft); glue penalty g; MBR threshold θ
for i:  H[i,i] = top-K context concepts of word i (as in v1 build_primitives); α[i,i,A] = 1; cplx(A)=1
for len = 2..n, i = 0..n-len, j = i+len-1:
    cand = defaultdict(list)
    for k in i..j-1, B in H[i,k], C in H[k+1,j]:
        c          = content_instance(B, C)                          # bags + cplx tags: position-free
        leafC,sC,okC = MEMO_CNT.get((B,C)) or sort(CNT+, c)           # corpus-wide memo
        x          = context_instance(w, i, j, content_ref=leafC)    # 5+5 word window (word-level)
        leafX,sX,okX = MEMO_CTX.get((i,j,leafC)) or sort(CTX+, x)
        if hard_gate and not (okC and okX): continue                 # climbing-ancestor τ = closed item
        s  = sC + sX                                                 # v1 score   (Option 2: log-odds)
        s -= soft_gate_penalty(okC, okX)                             # if τ used softly
        A  = Label(cut(leafX), cplx = 1 + max(cplx(B), cplx(C)))
        cand[A].append((k, B, C, exp(s/T)))
    H[i,j] = top-K labels A by Σ φ·α[i,k,B]·α[k+1,j,C]                # S-DIORA-style hard beam per cell
    for A in H[i,j]: α[i,j,A] = Σ_{(k,B,C,φ) ∈ cand[A]} φ·α[i,k,B]·α[k+1,j,C]
# glue chain: lets the parse end as several top-level chunks (v1 halting, but inside the DP)
G[-1] = 1;  G[j] = Σ_{i ≤ j} G[i-1] · g · Σ_A α[i,j,A];   Z = G[n-1]
# outside pass: reverse the glue chain, then reverse topological order over cells/items (Section 1)
μ_span[i,j] = Σ_A α[i,j,A]·β[i,j,A] / Z ;  μ_rule = β·φ·α_left·α_right / Z
# MBR decode, complete or partial
M[i,i] = 0;  M[i,j] = max(0, μ_span[i,j] - θ) + max_k (M[i,k] + M[k+1,j])     # θ = 0: full binary tree
# θ = 0.5 shortcut: {(i,j): μ_span > 0.5} is already non-crossing
```

**Cost controls:**
- Memoize content sorts on (B,C); they are position-free.
- Drop content-ref from the context instance during scoring, and reattach it for the chosen tree, to make context sorts O(n²).
- Keep K at 1–3.
- Use the τ gate as cell closing.
- Add Option 5 when sentences grow.

**Optional decoder.** Max-rule-product over unlabeled anchored splits, Σ_labels μ_rule, following Petrov & Klein.

#### Option 2 (rank 2; do together with Option 1): CCM-style constituent-vs-distituent log-odds potentials

**Idea.** Train separate distituent trees, CNT− and CTX−. The negatives come from adjacent pairs of gold constituents whose union is not gold; these are exactly v1's Verb+Det-type errors. In unsupervised mode the weights come from 1 − μ. The score is:

```
s = [log P(c | CNT+) − log P(c | CNT−)] + [log P(x | CTX+) − log P(x | CTX−)]
```

Both terms use the existing `tree.log_prob(instance, max_nodes)`. An alternative is to add a visible `constituent ∈ {yes, no}` attribute and read P(yes | ·) with `tree.predict`. The separate-tree version is preferred because it leaves the positive hierarchies, and therefore the generation pools, unchanged.

**Pros:**
- Attacks the documented root cause of the beam failure: rare chunks are no longer penalized for rarity.
- Positive scores reward good chunks, which removes a bias against more chunks when partial parses are allowed.
- Explicit negative chunk knowledge gives commission evidence, in Langley & Stromsten's sense, for brackets.
- Proven in CCM.

**Cons:**
- Doubles the trees and the training instances.
- Distituent sampling choices matter.
- Unsupervised negatives depend on the model's own posteriors.

#### Option 3 (rank 3): Outside-informed chunk context (two-pass, or iterated to a fixed point)

**Idea.** Approximate chunk context as an expectation under pass-1 posteriors (DIORA/ReCAT outside logic, in the spirit of stacking).

**Pros:**
- Delivers chunk context, a top v2 goal, without committing to unbuilt neighbors.
- Uses Cobweb's native bag attributes.
- Interpretable: "this span's expected sibling is a NP-like concept with probability 0.8".

**Cons:**
- Approximate; no guarantee of self-consistency.
- About 2× cost.
- Needs a STRUCT tree, trained on gold sibling/parent concepts in supervised mode or on posteriors in unsupervised mode.
- Fractional-count support must be verified.

**Pseudo-algorithm:**

```
pass 1: run Option 1 (+2) with word-window context → μ_rule
for each cell (i,j):
   SIB[i,j] = Σ_{rules r with (i,j) as a child}  μ_rule[r] · {label(sibling(r)) : 1, side(r) : 1}
   PAR[i,j] = Σ_{same r}                          μ_rule[r] · {label(parent(r)) : 1}
   x_struct = {SIB: normalize(SIB[i,j]), PAR: normalize(PAR[i,j])}       # soft chunk-context instance
   s_struct[i,j] = log P(x_struct | STRUCT+) − log P(x_struct | STRUCT−)
pass 2: φ' = φ · exp(λ·s_struct[i,j]/T); rerun inside-outside; decode
(optional) iterate passes until max |Δμ| < ε   (mean-field-style refinement)
```

An alternative with exact chunk context: take the k-best trees from pass 1 (Huang & Chiang 2005) and rerank each with exact neighbor concepts (Collins; Charniak & Johnson). A forest version uses cube pruning (Huang 2008).

#### Option 4 (rank 4): Unsupervised, incremental learning loop around the chart

**Idea.** Process sentences one at a time. Run the E-step (Option 1+2 with frozen trees), then one of three M-steps:
- **(a) Hard:** `ifit` the MBR tree.
- **(b) Confirmed chunks:** `ifit` composites with μ ≥ θ_learn ≥ 0.5, which are guaranteed non-crossing. Feed low-μ adjacent pairs to the distituent trees.
- **(c) Soft stepwise:** weight w = η_k·μ, with decay (1−η_k) on old counts, η_k = (k+2)^−α.

Switch between (a) and (c) when progress stalls (Lateen EM). Optionally re-run inside-outside on later sentences constrained by the confirmed brackets (Pereira & Schabes).

**Pros:**
- Unsupervised.
- Cobweb remains the incremental learner, with the M-step done by `ifit`.
- Realizes the user's "learn once confirmed" plan.
- Bracket self-constraints cut cost.

**Cons:**
- Likelihood is not linguistic structure (Pereira & Schabes; Spitkovsky et al.).
- Rich-get-richer dynamics.
- (c) requires a weighted `ifit` in C++.
- Needs safeguards: burn-in on short sentences, depth bounds (≤4 open chunks), periodic re-sorting.

#### Option 5 (rank 5; efficiency layer): Coarse-to-fine through the Cobweb hierarchy, plus Cobweb-predicted cell closing

**Idea.**
1. Coarse chart with concepts cut at depth 1–2 (truncated sorts).
2. Prune items with μ_coarse < ε.
3. Fine chart with full-depth concepts.
4. Add BEGIN/END attributes to word context instances, learned from gold or posteriors, and close cells where P(begin_i)·P(end_j) < δ (Roark & Hollingshead).

**Pros:**
- Cobweb's taxonomy is a natural sequence of nested partitions, so the method is Cobweb-native.
- PCFG evidence: 10×–100× speedups, or O(n²)/O(n) worst-case bounds, with no accuracy loss.

**Cons:**
- Pruning errors.
- Extra training attributes.
- The benefit is unknown for small synthetic grammars, where n is small anyway.

#### Option 6 (rank 6; the processing-side model): Incremental dynamic-programming beam parser with exact left chunk context

**Idea.** Left-to-right shift-reduce over words.
- **REDUCE** builds a composite from the top two stack chunks and scores it with Cobweb. The right window holds only the words read so far plus L lookahead words; the remaining slots are missing, which Cobweb categorization tolerates.
- **SHIFT** pushes the next word.
- States are merged when their signatures match: (position, labels and spans of the top two stack items). Merged states are scored by logsumexp (inside) or max (Viterbi), with predecessor links forming a graph-structured stack.
- A memory bound of ≤4 open chunks follows Schuler et al.

**Pros:**
- Linear in practice.
- Cognitively credible: incremental and memory-bounded.
- Left chunk context is exact, because the neighbors are built.
- Gives a word-by-word "prefix strength" profile, which is true surprisal if the generative side is normalized.

**Cons:**
- Right context is only partly known.
- Shift-reduce changes the merge order from easy-first.
- Without Option 2 and search-aware training (early or max-violation updates), it risks repeating v1's beam failure.

**Pseudo-algorithm:**

```
beam = {sig(empty stack, j=0): (score=0, preds=∅)}
for step in 1..2n-1:
   new = {}
   for state in beam:
      if j < n:  new ⊕= SHIFT(state)                           # push word j's label(s)
      if |stack| ≥ 2:
          c = content_instance(s1, s0); x = context_instance(left words, words ≤ j+L, missing beyond)
          if gate_ok(c, x): new ⊕= REDUCE(state, score += s(c,x))
   for each signature σ in new: merge entries (logsumexp or max), keep predecessor links (GSS)
   beam = top-b of new (and enforce ≤ 4 open chunks)
```

#### Option 7 (rank 7; cheapest fallback, or a distillation target): Keep easy-first and train it properly

**Idea.** Train v1's step ranker with a dynamic oracle and DAgger. The reference R is either gold (supervised) or the Option 1 MBR tree (unsupervised teacher, with LOLS theory). The oracle cost is the number of still-reachable reference spans destroyed by a merge, plus 1 for a non-reference span (Section 5). Exploration follows Cross & Huang.

**Pros:**
- O(n log n)-like test-time cost.
- Directly targets the distribution shift that sank v1's supervised ranker.
- Pairs naturally with Option 1 as teacher, giving a slow learner/analyzer and a fast processor.

**Cons:**
- Does not weigh all parses at test time.
- Needs a reference.
- Gives no μ.

#### Option 8 (rank 8; research or cognitive extension): Particle-filter easy-first

**Idea.** P stochastic copies of v1's greedy parser sample merges with probability ∝ exp(s/T) and are resampled by weight. P = 1 with T → 0 recovers v1. The existing `apply_candidate`/`undo` machinery in parse_mh.py helps.

**Pros:**
- Cognitively motivated (Levy et al.).
- Minimal code change.
- Gives approximate marginals.

**Cons:**
- High variance.
- Importance weights must correct for the (n−1)!/Π h(v) merge-order multiplicity, or balanced trees are over-sampled.
- Exact charts are cheap at TRELLIS's sentence lengths anyway.

#### Suggested experiment sequence (diagnostic, using the three synthetic CFGs)
- **E0 coverage.** Fraction of gold spans that receive inside mass after the τ gate. This says whether τ must be softened.
- **E1.** Option 1 with v1 scores: compare greedy vs Viterbi-CKY vs MBR (θ = 0 and 0.5), sweeping T ∈ {0.5, 1, 2, 4}. The prediction from Sections 2 and 5 is that Viterbi ≤ beam < greedy, while MBR is uncertain.
- **E2.** Add Option 2 log-odds; repeat E1.
- **E3. Posterior calibration.** A reliability plot of μ vs the empirical constituent rate, plus per-sentence posterior entropy vs errors.
- **E4.** Option 3 (chunk context) on the "large" grammar, where v1 scored lowest.
- **E5.** Option 4 unsupervised loop starting from the supervised trees, then from scratch.

Report bracket agreement as omission and spurious-bracket counts. Leave generation untouched; the user judges it qualitatively and it is frozen.

### Gaps
- **Search tools.** The web-search budget was exhausted mid-session. Verification relied on ACL Anthology (BibTeX and PDFs), Crossref, the arXiv API and the NeurIPS proceedings site.
  - Venue details for Collins (2000, ICML), Lafferty et al. (2001) and Daumé & Marcu (2005, LaSO) were confirmed only through citing papers ([Charniak & Johnson 2005](https://aclanthology.org/P05-1022/); [Finkel et al. 2008](https://aclanthology.org/P08-1109/); [Huang et al. 2012](https://aclanthology.org/N12-1015/)).
  - Jurafsky & Martin and Manning & Schütze (chapters 11–12) were not consulted.
- **2025–2026 coverage is thin.** It is limited to arXiv keyword queries (Holographic Neural PCFG 2026 preprint; Park & Kim EMNLP 2025; Marcheva et al. CogSci 2026). Recent work on incremental or MBR parsing may exist that these queries missed.
- **Not extracted.** Numbers for Ponvert et al. 2011, Yang & Deng 2020, Coavoux & Crabbé 2016 and Golland et al. 2012 were not extracted.
- **TRELLIS-side unknowns** that must be measured before committing:
  - per-sort latency of the C++ Cobweb tree;
  - the number of distinct child labels per cell (K);
  - whether `ifit` and the count updates in cobweb-private handle fractional counts;
  - whether v1's beam merged identical frontiers;
  - whether the generation side defines a normalizable distribution (needed for true EM and surprisal).
- **No precedent.** No prior work combines incremental conceptual clustering (Cobweb) with chart inside-outside parsing. The mappings above are reasoned from the structure of each method, not demonstrated.
