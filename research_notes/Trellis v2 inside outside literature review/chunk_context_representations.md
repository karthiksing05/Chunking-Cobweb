# Chunk Context and Compositional Representations: Prior Work for TRELLIS v2

**Scope.** These notes cover what the inside (compositional) and outside (contextual) descriptions of a chunk should contain, how prior work combines the two, and how each mechanism maps onto attribute-value instances that TRELLIS's two Cobweb hierarchies could categorize incrementally. Inside-outside *algorithms* are covered by another researcher. These notes cover *representations*.

**Terminology.**
- The "composition hierarchy" is v1's content hierarchy. The "representation hierarchy" is v1's context hierarchy.
- "Spine context" is my term for describing a span by the sibling chunks met on its path to the root.

**Metric convention.** External results are quoted in the authors' own metric: usually unlabeled bracketing F1, precision/recall, or accuracy. TRELLIS's own results are framed only as omission/commission, following the project convention.

**Verification.** Unless marked †, bibliographic details and numbers were checked this session (3 Oct 2026) against arXiv, the ACL Anthology, publisher pages, or full-text PDFs.
- † means the item is cited from prior knowledge with a known DOI or URL but was not re-fetched. The session's web-search budget ran out near the end.
- Facts about TRELLIS v1 come from the local text of the ACS 2026 paper and the project's memory notes.

---

## 1. How do induction models use a span's context to decide constituency and category?

### Takeaway
In distributional learning, a span's *category* is read off its contexts, but its *constituency* needs something extra:
- a dedicated distituent class (CCM);
- statistical dependence between the left and right context (Clark 2001); or
- a closure test (substitutability and Clark's syntactic concept lattice).

The standard way to make context both general and robust to sparsity is to read a class hierarchy at several granularities, as with Brown-cluster prefixes. Clark's lattice gives a formal version of TRELLIS's "concept and chunk as two facets": a syntactic concept *is* a pair of (set of strings, set of contexts).

### Cited Findings
**Distributional foundations and multi-granularity classes**
- Clark (2001) describes his grammar inducer as "one more attempt to implement Zellig Harris's distributional analysis (Harris, 1954)" — [Clark 2001, CoNLL](https://aclanthology.org/W01-0713/); Harris's original is [Harris 1954, *Word* 10(2–3)](https://doi.org/10.1080/00437956.1954.11659520)†.
- Schütze (1995) categorizes word *tokens in context* rather than word types, evaluated on the Brown Corpus — [Schütze 1995, EACL](https://aclanthology.org/E95-1020/).
- Klein & Manning summarize the classic pattern: distributions over adjacent words induce classes close to traditional POS. Their example: "stocks" and "treasuries" both occur before "fell" and "rose", so they land in the same class — [Klein & Manning 2004](https://aclanthology.org/P04-1061/).
- Brown et al. (1992) build class-based n-gram models by agglomerative merging that preserves average mutual information between adjacent classes. This yields a binary class hierarchy — [Brown et al. 1992, CL 18(4)](https://aclanthology.org/J92-4003/)†.
  - Each node of that hierarchy is labeled by a bit-string giving its path from the root — [Koo, Carreras & Collins 2008](https://aclanthology.org/P08-1068/).
- Koo et al. (2008) found it "helpful to employ two different types of word clusters" — [Koo et al. 2008](https://aclanthology.org/P08-1068/):
  - short bit-string prefixes (4–6 bits) "as replacements for parts of speech";
  - full bit strings as substitutes for word forms, capped at about 1,000 distinct strings, so they are not equivalent to word forms.
  - Feature templates mix head/modifier words, POS tags, and the 4-bit, 6-bit and full-string clusters.
  - Results: English unlabeled second-order accuracy rose from 92.02% to 93.16%, and Czech from 86.13% to 87.13%. With only 1k training sentences, first-order gains were +1.36 (English) and +3.57 (Czech) points.

**Constituent-Context Model (CCM)**
- CCM was introduced in [Klein & Manning 2002, ACL, pp. 128–135](https://aclanthology.org/P02-1017/). Each of a sentence's O(n²) spans generates its *yield* (the tag sequence inside) and its *linear context* (the tags immediately before and after). A binary-tree-equivalent bracketing is chosen at random, and incoherent derivations get zero mass — [Klein & Manning 2004](https://aclanthology.org/P04-1061/).
- The problem CCM solves, in the authors' words: "it is easy enough to discover that DET N and DET ADJ N are similar and that V PREP DET and V PREP DET ADJ are similar, but it is much less clear how to discover that the former pair are generally constituents while the latter pair are generally not." Their answer was "earmarking a single cluster d for non-constituents" — [Klein & Manning 2004](https://aclanthology.org/P04-1061/).
- WSJ10 results (UP / UR / UF1) — [Klein & Manning 2004](https://aclanthology.org/P04-1061/):

  | Model | UP | UR | UF1 |
  |---|---|---|---|
  | CCM | 64.2 | 81.6 | 71.9 |
  | DMV+CCM (POS) | 69.3 | 88.0 | 77.6 |
  | DMV+CCM (distributionally induced classes) | 65.2 | 82.8 | 72.9 |

  - NEGRA10: CCM 61.6. CTB10: CCM 45.0.
  - With *predicted* tags, CCM drops to 63.2 on WSJ-10, as tabulated in [Drozdov et al. 2019](https://aclanthology.org/N19-1116/).

**Clark (2001): context clustering plus a mutual-information filter**
- Context is "the part of speech tag immediately preceding the sequence and the tag immediately following it". Tag sequences are clustered by their context distributions — [Clark 2001](https://aclanthology.org/W01-0713/).
- Criterion: "with real constituents, there is high mutual information between the symbol occurring before the putative constituent and the symbol after." — [Clark 2001](https://aclanthology.org/W01-0713/)
  - Intuition: an NP at the start of a sentence is likely followed by a finite verb, while an NP after the verb is likely followed by the end of the sentence or a preposition.
  - A spurious sequence such as "PRP AT0" is followed by an N-bar wherever it occurs, so its left and right contexts are independent.
- Setup and results — [Clark 2001](https://aclanthology.org/W01-0713/):
  - The criterion is embedded in an MDL search over 12M words of the BNC.
  - ATIS bracketing after 40 iterations: UR 34.6, UP 53.4, F 42.0, versus EMILE 25.4 and ABL 39.2.
  - Using automatically derived tags gave "essentially the same results", making the method fully unsupervised.

**Substitutability and lattice-based distributional learning**
- Clark & Eyraud formalize Harris's substitutability as a learnable class. Substitutable context-free languages are polynomially identifiable in the limit from positive data — [Clark & Eyraud 2007, JMLR 8:1725–1745](https://jmlr.org/papers/v8/clark07a.html).
  - The core definition: if two strings share one context, they share all contexts.
- Syntactic concept lattice — [Clark, ICGI 2010 version](https://www.its.caltech.edu/~matilde/ClarkSyntacticConceptLattice.pdf):
  - A context is a pair (l, r). Polar maps send a set of strings to the contexts that accept all of them, and a set of contexts to the strings they all accept.
  - Syntactic concepts are the closed pairs (S, C). They form a complete lattice with a concatenation operation and residuation.
  - Nonterminals correspond to lattice concepts. The finite-kernel/finite-context property guarantees each concept is characterized by finitely many strings/contexts.
  - The construction is explicitly an instance of Formal Concept Analysis.
- The lattice is a residuated lattice. Its natural representation is *not* context-free: it includes some mildly context-sensitive languages. A CFG learner based on it uses positive data plus membership queries and can learn some inherently ambiguous languages — [Clark, FG 2009](https://members.loria.fr/PdeGroote/FG09/Clark.pdf).
  - The journal version is "The syntactic concept lattice: another algebraic theory of the context-free languages?", *J. Logic and Computation* 25(5):1203–1229, 2015, as cited in [this 2025 paper](https://arxiv.org/html/2510.24853v1).
- Distributional Lattice Grammars can represent all regular languages, some CFLs, and some non-CFLs, with an efficient and correct unsupervised learner — [Clark 2010, CoNLL, pp. 28–37](https://aclanthology.org/W10-2904/).
- Related: contextual binary-feature representations where nonterminals are described by sets of contexts — [Clark, Eyraud & Habrard 2010, JMLR 11](https://jmlr.org/papers/v11/clark10a.html).
- **Primal vs dual approaches** — [Yoshinaka 2011, DLT](https://www.springerprofessional.de/towards-dual-approaches-for-learning-context-free-grammars-based/3686398):
  - *Primal* approaches build nonterminals characterized by strings.
  - *Dual* approaches build nonterminals characterized by contexts.
  - A later generalization of the dual approach uses some contexts positively and others negatively — [Kanazawa & Yoshinaka 2021, ICGI/PMLR 153](https://proceedings.mlr.press/v153/kanazawa21a.html).

### Inferences
- **TRELLIS v1 already sits inside this tradition.** The representation hierarchy is a context clusterer in the spirit of Schütze, Clark 2001, and CCM's context half. The composition hierarchy clusters yields. What v1 lacks is CCM's *distituent* model and Clark's *left×right dependence* signal. v1 instead uses a count gate on ancestors to decide merges.
- **A cheap way to bring CCM in.** v2 trains on gold unlabeled trees, so the spans that cross gold brackets are known distituents. Storing their context instances, with a `role=distituent` flag or a separate subtree, would let the parser score a merge by the CCM-style odds P(context | constituent) / P(context | distituent). This does not violate the "parser output never feeds memory" rule, because the labels come from gold trees.
- **Clark's MI criterion needs a joint attribute.** Category utility treats attributes as independent within a concept, so dependence between the left and right context is invisible to CU. A conjunctive `LR-pair` attribute (left context × right context, at a coarse depth) would let the representation hierarchy separate constituent-like contexts, where left and right co-vary, from distituent-like ones.
- **Brown/Koo prefixes vs v1's encoding.** The prefix trick is the direct analog of reading a Cobweb taxonomy at several depths. v1 reads only depth 4 (top-3 ids). Koo's evidence favors exposing *both* coarse classes (≈POS) *and* near-lexical identities as separate attributes. This matches the v1 finding that concept ids helped a dense grammar but hurt a sparse lexicon (see §3 for the Cobweb-native evidence).
- **Clark's lattice answers "compose, then enrich contextually" formally.** The concatenation of concepts (S₁,C₁)∘(S₂,C₂) is computed by concatenating strings and then *closing* the result through its contexts, (S₁S₂)′′. Composition produces a raw extent; the context closure turns it into a category.
  - In TRELLIS terms, the composition hierarchy proposes "left concept + right concept". The representation hierarchy then decides *what that composite is* from its contexts.
  - Primal ≈ composition hierarchy and dual ≈ representation hierarchy. Yoshinaka shows both are legitimate and complementary.
- **The finite-context property supports bounded windows.** Each concept needs only a finite set of characterizing contexts. This argues that bounded windows suffice, provided they are expressed at the right granularity.

### Gaps
- I did not find empirical results for lattice or DLG learners on large natural corpora. That line is mostly formal, with membership queries.
- The second-order details of Schütze's token vectors (using neighbors' learned vectors as context) are from prior knowledge and were not re-read this session.

---

## 2. Inside (compositional) vs outside (contextual) span representations, and context-perturbation signals

### Takeaway
Neural latent-tree models compute two descriptions per span:
- **Inside**: bottom-up composition of children.
- **Outside**: top-down, from the parent's outside plus the sibling's inside.

Unrolled, the outside of a span is the sequence of sibling chunks at every ancestor level, which is the user's "latents take all levels of content before and after". Practical lessons:
- Hard single trees beat soft mixtures (S-DIORA).
- Concatenating inside and outside makes good phrase representations (DIORA).
- Boundary-state and split-point features are the strongest span features in supervised parsers. This is the rediscovery behind v1's boundary/seam hints.

Perturbing a span's context, or swapping its yield (constituency tests), is a strong unsupervised constituency signal, as is co-training separate inside and outside classifiers.

### Cited Findings
**DIORA and successors**
- DIORA "predicts each word in an input sentence conditioned on the rest of the sentence and uses inside-outside dynamic programming to consider all possible binary trees". CKY extracts the best tree at inference — [Drozdov et al. 2019, NAACL](https://arxiv.org/abs/1904.02142).
  - The outside vector of "the cat" is "a function of the outside vector of its parent 'the cat drinks' and the inside vector of its sibling 'drinks'".
  - Inside vectors are compatibility-weighted mixtures over split points.
  - The root's outside vector is a learned parameter.
  - Training reconstructs each leaf from its outside vector with a max-margin loss over N negatives — [DIORA paper](https://aclanthology.org/N19-1116/).
- DIORA numbers — [DIORA paper](https://aclanthology.org/N19-1116/):

  | Benchmark (unlabeled F1) | DIORA | With trailing-punctuation fix (+PP) | Comparison |
  |---|---|---|---|
  | Full WSJ test, binarized, with punctuation | 48.9 ±0.5 mean (49.6 max) | 55.7 mean (56.2 max) | ON-LSTM 47.7 mean |
  | MultiNLI, against CoreNLP parses | — | 59.0 median | — |
  | WSJ-10 | 67.7 mean (68.5 max) | — | — |
  | WSJ-40 | 60.6 mean (60.9 max) | — | — |

  - Phrase similarity uses the *concatenation* [inside; outside]. P@1 on CoNLL-2000 chunks was 0.990 vs ELMo 0.987. On CoNLL-2012 it was 0.860 vs ELMo 0.896 and context-insensitive ELMo 0.708.
- S-DIORA notes that DIORA's soft vector averaging "is locally greedy and cannot recover from errors". It encodes a single tree with a hard argmax and a beam per chart cell, improving WSJ unsupervised parsing by 2.2–6 F1 — [Drozdov et al. 2020, EMNLP](https://aclanthology.org/2020.emnlp-main.392).
- R2D2 is a recursive Transformer over differentiable CKY-style binary trees. Its pretraining objective predicts "each word given its left and right abstraction nodes". Pruning makes encoding linear-time — [Hu et al. 2021, ACL, pp. 4897–4908](https://aclanthology.org/2021.acl-long.379/).
- Fast-R2D2 replaces heuristic pruning with a top-down parser that "casts parsing as a split point scoring task": it scores all split points, then recursively splits at the best one — [Hu et al. 2022, EMNLP](https://aclanthology.org/2022.emnlp-main.181/).
- ReCAT's Contextual Inside-Outside (CIO) layers — [Hu, Zhu, Tu & Wu 2024, ICLR](https://arxiv.org/abs/2309.16319):
  - A bottom-up pass composes high-level spans from low-level ones. A top-down pass then mixes in information from inside and outside each span.
  - The result is multi-grained, fully contextualized span representations, stacked between the embedding and attention layers.
  - Induced trees are reported as strongly consistent with human-annotated ones.
- GPST pairs a left-to-right syntactic LM with a composition model trained with a bidirectional LM loss that induces trees and constituent representations — [Hu et al. 2024, ACL](https://arxiv.org/abs/2403.08293).
  - A "representation surrogate" allows joint parallel hard-EM training without gold trees.
  - Pretrained on OpenWebText (9B tokens), it outperforms GPT-2 of comparable size.
- The StrAE line — [Self-StrAE, SemEval 2024](https://aclanthology.org/2024.semeval-1.18); [StrAE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.469):
  - Self-StrAE uses its own learned representations to define a local merge sequence, then uses that tree for the decoder.
  - With 430 non-embedding parameters it matches a 6-layer RoBERTa with 3.95M parameters. Later versions get down to 7 parameters and pretrain on 10M tokens.
- Banyan adds an entangled tree structure and diagonalized message passing. With 14 non-embedding parameters it outperforms larger transformers in low-resource settings — [Opper & Siddharth, ICML 2025](https://arxiv.org/abs/2407.17771).

**Boundary and split-point span features (compare v1's boundary/seam hints)**
- Stern, Andreas & Klein score labels and spans independently, representing a span by differences of BiLSTM states at its endpoints. They report 91.79 F1 on PTB and 82.23 on the French Treebank — [Stern et al. 2017, ACL](https://aclanthology.org/P17-1076/); numbers from the [arXiv abstract](https://arxiv.org/abs/1705.03919).
- Kitaev & Klein found that "separating positional and content information in the encoder can lead to improved parsing accuracy". They report 93.55 F1 (no external data) and 95.13 with pretrained representations, and best published results on 8 of 9 SPMRL languages — [Kitaev & Klein 2018, ACL](https://aclanthology.org/P18-1249/); [arXiv](https://arxiv.org/abs/1805.01052).
- DIORA's ELMo baseline represents phrases "as a function of its first and last hidden state", which is a boundary representation — [DIORA paper](https://aclanthology.org/N19-1116/).

**Context-perturbation and two-view constituency signals**
- Constituency tests — [Cao, Kitaev & Klein 2020, EMNLP](https://arxiv.org/abs/2010.03146):
  - Transformations such as "replacing the span with a pronoun" are judged by an unsupervised neural acceptability model.
  - Refinement alternates between improving the trees and improving the grammaticality model.
  - Result: 62.8 F1 on the PTB test set, +7.6 over the previous best.
- Perturbed Masking probes BERT without parameters by measuring how masking one token changes another token's representation. Trees recovered this way beat linguistically uninformed baselines — [Wu, Chen, Kao & Liu 2020, ACL, pp. 4166–4176](https://aclanthology.org/2020.acl-main.383/).
- Li & Lu score each span by "the distortion of contextual representations resulting from linguistic perturbations" motivated by constituency tests, then run chart parsing. They beat prior masked-LM-based methods on English and are best in 6 of 8 languages — [Li & Lu 2023, ACL](https://arxiv.org/abs/2306.00645).
- Two-view co-training — [Maveli & Cohen, Findings of ACL 2022](https://arxiv.org/abs/2110.02283):
  - An *inside* classifier sees only the span. An *outside* classifier sees "everything outside of a given span". These are the two views in a co-training scheme with seed bootstrapping.
  - With a weak branching prior, it reaches 63.1 F1 on PTB and new best results on CTB and KTB.
- Span-overlap scores how often a word sequence recurs across sentences with *equivalent predicate-argument structures*. It beats prior unsupervised parsers in 8 of 10 languages. Participant-denoting constituents score higher than event-denoting ones of the same length — [Chen, He, Bollegala & Miyao 2024](https://arxiv.org/abs/2404.12059).
- SemInfo uses a bag-of-substrings model of semantics as the training objective instead of likelihood. It gains an average of 7.85 sentence-F1 across five PCFG variants and four languages — [Chen et al., ICLR 2025](https://arxiv.org/abs/2410.02558).

### Inferences
- **Mapping to TRELLIS.** Inside maps to the composition-hierarchy instance and outside to the representation-hierarchy instance.
  - DIORA's recursion, outside(span) = g(outside(parent), inside(sibling)), unrolls into the list of sibling insides at levels 1…K plus a root term. That is exactly a *spine context*.
  - A discrete version: attributes `spine.k = (side, sibling-composition-concept@depth)` for k = 1…K.
  - R2D2's "left and right abstraction nodes" are the same idea seen from a leaf.
- **Use hard top-k, not soft mixtures.** S-DIORA's lesson favors hard top-k assignments when building outside instances. For Cobweb that means a bag of the top-k sibling concepts, as v1 already does for content slots, weighted by span posterior if one is available. Folding all candidate trees' contexts into one soft bag is what S-DIORA found leaks and errs.
- **Boundary/seam can return faithfully.**
  - The span-parsing literature's best features are boundary features (endpoint states). The modern split-point parser (Fast-R2D2) is a seam scorer.
  - v1 already describes a composite's context by the window around its *outermost positions*. So boundary words are legitimately "what neighbors see" and belong in the representation instance.
  - The seam (the junction pair) is a property of the *composition*. It belongs in the composition instance, written at several granularities rather than as raw-word hints.
- **Keep the two descriptions separate.** Kitaev & Klein's content/position factoring supports joining content and context only at scoring time rather than fusing them into one bundle. v1 already argues this, noting that a single bundle "would force role and filler into one signal".
- **Constituency tests have a discrete analog.** A span passes a "proform test" if its representation instance sorts into a representation concept that also holds *single primitives* (pronouns or names for NPs, intransitive verbs for VPs). This is a TRELLIS-native substitute for Cao et al.'s and Li & Lu's neural judges.
- **Co-training needs care.** It is the closest learning-theoretic framing of "two hierarchies as two views". But writing co-trained labels into memory would conflict with the project rule that parser output never feeds memory. Inside/outside *agreement* could serve as a parse-time gate only.

### Gaps
- I did not find any work that encodes DIORA-style outside representations as *discrete* symbolic attributes; the spine mapping is my inference.
- ReCAT and GPST quantitative results were not extracted beyond their abstracts.

---

## 3. Composition mechanisms for symbolic and attribute-value representations (including the Cobweb family and relational input)

### Takeaway
In attribute-value systems, role-filler binding is native: attribute name = role, value = filler. The real issues are the **stability** and the **granularity** of filler references when fillers are themselves learned concepts.

The Cobweb literature has already run TRELLIS's context experiment:
- representing context words by **leaf** concepts performed near chance;
- representing them by their **whole ancestor path** with counts performed best;
- iterating passes solved the "neighbors not yet categorized" problem.

TRESTLE supplies a ready encoding for relational (semantic-network) input: structure mapping followed by flattening.

### Cited Findings
**Vector binding and composition (for contrast)**
- Tensor-product binding — [Smolensky 1990, *AI* 46](https://doi.org/10.1016/0004-3702(90)90007-M)†. Holographic reduced representations — [Plate 1995, *IEEE TNN* 6(3)](https://doi.org/10.1109/72.377968)†. Hyperdimensional computing with random high-dimensional vectors — [Kanerva 2009, *Cognitive Computation* 1(2)](https://doi.org/10.1007/s12559-009-9009-8)†. All three bind fillers to roles and bundle bindings into fixed-width vectors.
- Comparing additive and multiplicative vector composition against human similarity judgments, multiplicative models were superior — [Mitchell & Lapata 2010, *Cognitive Science*](https://homepages.inf.ed.ac.uk/mlap/Papers/cogsci2010.html).

**Cobweb-family structured representations**
- LABYRINTH — [Thompson & Langley 1989, ICML workshop](https://mlanthology.org/icml/1989/thompson1989icml-incremental):
  - It extends Cobweb so a composite instance is a set of components, which may themselves be composite.
  - Attribute values of composite concepts refer to other nodes of the same hierarchy, giving an interleaved memory.
  - It "is constantly revising the structure of attributes" because those referenced nodes change.
- TRESTLE — [MacLellan, Harpstead, Aleven & Koedinger 2016, *Advances in Cognitive Systems* 4](https://tail.cc.gatech.edu/publications/maclellan-acs-journal-2016); [arXiv version](https://arxiv.org/abs/2410.10588):
  - Instances have nominal, numeric, *component*, and *relational* attributes. Relations are tuples such as `(On Component1 Component2)`.
  - Before categorizing, a structure-mapping step uses beam search (or A*) to rename the instance's components so they best match the root concept, maximizing expected correct guesses.
  - The instance is then *flattened*: relations become nominal attributes and components become dot-named attributes. Standard category utility applies after that.
  - On the RumbleBlocks task, accuracy was about 70% (human-like), versus about 83% for a non-incremental baseline. Clustering agreed with human clusterings at ARI 0.37–0.56, versus 0.42–0.51 for the baseline.
- TRESTLE's partial matching descends from structure-mapping theory — [Gentner 1983](https://doi.org/10.1207/s15516709cog0702_3)†; [SME, Falkenhainer, Forbus & Gentner 1989](https://doi.org/10.1016/0004-3702(89)90077-5)†.

**Cobweb language models (directly relevant to TRELLIS's context findings)**
- MacLellan, Matsakis & Langley (2022) built three Cobweb variants, each encoding a training case as an anchor word plus surrounding context words — [ACS 2022](https://arxiv.org/abs/2212.11937):
  - **Word**: a single bag-of-words context attribute with counts. Order is ignored, and CU gives one guess per attribute.
  - **Leaf**: context words are replaced by the labels of their *leaf* concepts. Nonterminals were avoided "because they change constantly … which makes them an unstable representation". A sequence is processed in three passes to converge on labels. The first pass has empty context, because "concept labels for adjacent words are not yet available", and categorization during the passes does not modify the tree.
  - **Path**: each context word is represented by "the entire path through the hierarchy" with counts on every ancestor. Every concept keeps back-pointers to the concepts that refer to it, so references are updated when concepts are merged or deleted.
- Results of those variants — [MacLellan et al. 2022](https://arxiv.org/abs/2212.11937):
  - "The Leaf system does little better than chance". Word and Path "do reasonable jobs".
  - On homonyms, Path reaches ARI ≈ 1 versus Word ≈ 0.25. The authors attribute this to Path matching *different* context words that share ancestors, which "makes it more robust to sparse context elements".
  - Path showed the most rapid synonym recall. Word and Path both recalled synonyms better than Word2Vec with fewer training cases.
- Cobweb/4L — [Lian, Baglodi & MacLellan, ACS 2024](https://arxiv.org/abs/2409.12440):
  - Attributes are `anchor`, `context-before`, and `context-after`. Each context word gets a weighted count of 1/(d+1), where d is its distance from the anchor. Windows are 10 words before and 10 after.
  - It uses the information-theoretic variant of category utility (Corter & Gluck).
  - Its multi-node predictor expands up to N_max nodes best-first by collocation P(c|x)·P(x|c), then averages their predictions weighted by a softmax over collocation, a form of Bayesian model averaging.
  - On the MSR Sentence Completion task, single-node prediction sat at about 0.225 (basic level) and about 0.2 (leaf, ≈chance). Multi-node prediction started at about 0.28 and kept improving. The 2000- and 3000-node versions exceeded CBOW, and both Cobweb/4L and Word2Vec beat BERT with less data.
- Lian, Wang & MacLellan report that Cobweb/4L is hyperparameter-free, robust across data scales, and outperforms transformer LMs in low-data settings — [*Cognitive Systems Research* 92:101371, 2025](https://tail.cc.gatech.edu/publications/lian-csr-2025).
- TRELLIS v1 itself (local paper text):
  - Composite context is "the same window taken over its outermost positions".
  - Content-instance values live in the context hierarchy's id space, so "the content hierarchy never sees a surface word".
  - Because the context tree restructures, references "can dangle". v1 canonicalizes them to a recent ancestor and rewrites stored instances.

### Inferences
- **TRELLIS has already replicated the Leaf/Word/Path result.** The v1 finding that concept-id context helped a dense grammar but hurt a sparse lexicon is the Leaf-vs-Word contrast. MacLellan et al.'s fix was *Path*: keep the specific item and all its generalizations as counted values. The equivalent for TRELLIS is multi-depth attributes (see §9, D1).
- **v1's dangling-reference problem is LABYRINTH's problem.** Prior art offers two fixes: v1's canonicalize-and-rewrite, or Path's back-pointer maintenance. Back-pointers scale better when many instances reference a node that is about to be merged or split.
- **The Leaf system's three-pass labeling is the earliest Cobweb-native answer to TRELLIS's chunk-context bottleneck.** Pass 1 describes with only what is available (words). Later passes describe with neighbors' concept ids from the previous pass, using non-modifying categorization until the labels settle.
- **v1's top-3 bags are already Cobweb/4L-style.** Cobweb/4L's multi-node prediction generalizes v1's "top-3 ids at depth 4" bag. A neighbor is described by a weighted set of concepts drawn from different depths, not one committed id.
- **Binding needs no new machinery.** Flattened names like `S1.rep.d4` are the roles. Mitchell & Lapata's multiplicative (intersective) composition corresponds to *conjunctive* attributes, for example a seam pair (left-edge-of-right-child ∧ right-edge-of-left-child), rather than two independent slot attributes.
- **Relational input fits TRESTLE's pipeline.** A semantic-network composite becomes component attributes (the slots) plus relational attributes such as `(agent S1 S2): True`. Structure mapping aligns slots when order or arity varies, and flattening then makes everything CU-compatible.

### Gaps
- I found no Cobweb-family system other than TRELLIS that categorizes multi-word span composites together with their context.
- The 2025–2026 sweep for new Cobweb language work found only the CSR 2025 paper. The search budget ran out before an exhaustive check.

---

## 4. Cognitive models that use chunks as context for comprehension and production

### Takeaway
HVM (Wu et al., ICLR 2025) is the closest relative of TRELLIS v2:
- it learns chunks and **variables**, where a variable is the set of chunks that share the same *preceding and succeeding chunk* (chunk context);
- it nests variables inside new chunks;
- it gets chunk context by **parsing first, then updating chunk-to-chunk statistics**.

Other models support related ideas:
- ADIOS shows that substitution classes should be **valid only inside their parent pattern's context**, which is TRELLIS's P7 conditioning.
- CBL shows that word-level statistics beat coarse class statistics in child-directed speech, and that greedy **chunk-to-chunk** production works.
- TRACX and chunk-and-pass make recognized chunks the context for what follows.

### Cited Findings
**HVM (Hierarchical Variable learning Model) in detail**
- Citation: Wu, Thalmann, Dayan, Akata & Schulz, "Building, reusing, and generalizing abstract representations from concrete sequences", ICLR 2025 — [ICLR page](https://www.iclr.cc/virtual/2025/poster/27794); [arXiv v2, June 2025](https://arxiv.org/abs/2410.21332).
- **Memory.** HVM holds a chunk dictionary ℂ and a set of variables 𝕍 — [arXiv](https://arxiv.org/html/2410.21332):
  - "A variable denotes distinct observations appearing in the same context (here defined as distinct chunks sharing preceding and succeeding chunks)."
  - Example: if ABC and DC both follow "A" and are both followed by "ED", HVM proposes a variable V = {ABC, DC} and a new chunk A⊕V⊕ED that embeds the variable.
- **Parsing.** HVM organizes chunks in a prefix trie (the "parsing graph") and greedily descends to the deepest node consistent with the upcoming input — [arXiv](https://arxiv.org/html/2410.21332):
  - This cuts search from O(|ℂ|) to O(depth), about 90% fewer steps than HCM.
  - After parsing, it updates transition counts between consecutive parsed chunks, T_ij += [i=c_L][j=c_R], and identification counts M_i.
- **Generative model.** Over d iterations, the generator creates either new objects (concatenations) or new categories (disjunctive sets). Sampling recursively replaces embedded categories with members — [arXiv](https://arxiv.org/html/2410.21332).
- **BabyLM negative log-likelihood** (lower is better) — [arXiv](https://arxiv.org/html/2410.21332):

  | Corpus | HVM | HCM | LZ78 |
  |---|---|---|---|
  | CHILDES | 1953.01 | 2783.71 | 2837.50 |
  | BNC | 3108.33 | 3591.60 | 3136.67 |
  | Gutenberg | 3252.84 | 3770.36 | 3156.61 |
  | OpenSubtitles | 3764.48 | 4151.89 | 3395.09 |

  - Note: the extracted summary claimed HVM is lowest "in all domains", but by its own table LZ78 is lower on Gutenberg and OpenSubtitles. HVM is lower than HCM everywhere.
- **Compression.** LZ78's compression ratio is sometimes better (BNC: 0.39 vs HVM 0.50). But HVM's dictionary-to-sequence-length ratio is 0.05–0.12, versus 0.34–0.39 for LZ78 — [arXiv](https://arxiv.org/html/2410.21332).
- **Humans and LLMs.** In a human sequence-recall experiment with 112 participants, HVM's likelihood correlated with recall times at R = 0.86 (training) and R = 0.70 (transfer) — [arXiv](https://arxiv.org/html/2410.21332):
  - GPT-2, Llama 2 and Llama 3 "do not differentiate" training that shares variables with the transfer block from training that does not.
  - Raising the abstraction level increases transfer likelihood but also representation complexity, a rate–distortion trade-off.
- The thesis framing (chunking and abstraction as computational principles) is in [Wu, PhD thesis, arXiv 2503.10973, revised June 2026](https://arxiv.org/abs/2503.10973).

**HCM, ADIOS, U-MILA**
- HCM learns minimal atomic sequential units as chunks, then builds a hierarchy by chunking earlier chunks "guided by sequential dependence". It comes with learning guarantees for an idealized version — [Wu, Éltető, Dasgupta & Schulz, NeurIPS 2022, pp. 36706–36721](https://proceedings.nips.cc/paper_files/paper/2022/file/ee5bb72130c332c3d4bf8d231e617506-Paper-Conference.pdf).
- ADIOS — [Solan, Horn, Ruppin & Edelman 2005, PNAS 102(33)](https://doi.org/10.1073/pnas.0409746102); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953):
  - Sentences are paths on a graph. Significant patterns are found with the MEX motif-extraction criterion (fan-through/fan-in decrease ratios, thresholds η and α≈0.01).
  - Equivalence classes are formed within a context window of width L, where "multiple alternative subpaths coexist within a fixed context".
  - Crucially, "equivalence relations only hold in the contexts specified by their parent patterns, making the ADIOS representation inherently safer than grammars that posit globally valid categories."
  - Generation reads patterns top-down and left-to-right, choosing one member from each equivalence class it meets.
  - Results: about 90% precision and recall on TA1, 100% precision and 99% recall on a simple CFG, about 70% human-judged precision on ATIS-2, and perplexity 11.5 versus 14 for trigrams.
- U-MILA learns incrementally a grammar in the form of a directed weighted graph whose nodes are recursively defined patterns. The same graph is used to parse and to generate. Its 17 experiments cover generation, representation, segmentation and chunking, artificial-grammar learning, and structure dependence — [Kolodny, Lotem & Edelman 2015, *Cognitive Science* 39(2):227–267](https://cris.tau.ac.il/en/publications/learning-a-generative-probabilistic-grammar-of-experience-a-proce/).

**CBL / CAPPUCCINO**
- Mechanism — [McCauley & Christiansen 2011, CogSci](https://csl-lab.psych.cornell.edu/files/2021/01/2011-mc-cogsci.pdf):
  - The model computes backward transitional probabilities (BTPs) between words and inserts a chunk boundary when a BTP falls below the running average.
  - Chunks go into a "chunkatory" that is consulted online.
  - "Statistics over individual words were found to be more useful than statistics over word classes" (13 classes).
- Production — [McCauley & Christiansen 2011](https://csl-lab.psych.cornell.edu/files/2021/01/2011-mc-cogsci.pdf):
  - The child's utterance is turned into a bag of known chunks. The model first picks the chunk with the highest BTP given the utterance-start marker, then repeatedly the chunk with the highest BTP given the *most recently produced chunk*.
  - Mean exact-reproduction score across 13 corpora was 59.8%, versus 52.3%, 49.6% and 54.2% for the baselines.
- The full CBL model has comprehension (shallow parsing) evaluated on 79 single-child corpora (English, French, German) and production on more than 200 corpora in 29 languages. The authors conclude that early linguistic behavior "may be supported by item-based learning through online processing of simple distributional cues" — [McCauley & Christiansen 2019, *Psych. Review* 126(1):1–51](https://doi.org/10.1037/rev0000126).

**PARSER, TRACX, chunk-and-pass, CHREST**
- PARSER forms chunks from small attentional groupings of primitives that strengthen with re-encounter and decay or interfere otherwise, so segmentation emerges without boundary statistics — [Perruchet & Vinter 1998, *J. Memory & Language* 39(2)](https://doi.org/10.1006/jmla.1998.2576)†.
- TRACX is a recognition-based autoassociator. When a pair of inputs is reconstructed well (recognized as a chunk), its hidden representation becomes the *left input* at the next step. Recognized chunks literally become context — [French, Addyman & Mareschal 2011, *Psych. Review* 118(4)](https://doi.org/10.1037/a0025255) (DOI resolves; mechanism from prior knowledge†).
- Chunk-and-pass (the Now-or-Never bottleneck): input must be recoded into chunks immediately and passed to progressively higher levels — [Christiansen & Chater 2016, *BBS* 39](https://doi.org/10.1017/S0140525X1500031X)†.
- CHREST templates are retrieval structures with "a core that remains unchanged, and a set of slots, perhaps with default values, whose value can be rapidly altered" — [Gobet & Simon 1996, *Cognitive Psychology* 31(1):1–40](https://bura.brunel.ac.uk/handle/2438/1339).

### Inferences
- **HVM maps onto TRELLIS almost one to one.**

  | HVM | TRELLIS |
  |---|---|
  | chunk | composite (composition concept) |
  | variable | representation concept |
  | "same preceding and succeeding chunk" | categorization of a *chunk-level* context instance |
  | abstraction-level knob | maturity/τ cut on the representation hierarchy |

  - HVM's variable criterion is crisp equality of neighbors. Cobweb replaces it with probabilistic categorization over multi-level context, which is a strict generalization.
- **HVM shows how to get chunk context with a greedy parser.** Parse with the current inventory, then compute chunk-to-chunk neighbor statistics on the parsed output. For TRELLIS, which learns from gold trees, this means building representation instances *after* the training tree is known, with neighbors described by the chunk ids at each level.
  - At parse time, the same works in two passes, or within an inside-outside chart (§2).
- **Literature support for P7 conditioning.**
  - ADIOS's context-local classes are exactly v1's generation filter (P7: condition each child on the class its parent expects). ADIOS presents this as *safer* than global categories, which matches v1's observation that intermediate nodes are only about 90% pure as global substitution classes.
  - CHREST templates likewise have slots whose admissible fillers are local to the template.
- **CBL and the v1 context findings.** CBL's result that words beat 13 coarse classes is a second, independent data point for v1's sparse-lexicon finding: coarse classes alone lose discriminative identity, so keep word or leaf identity alongside the classes.
  - CBL's chunk-to-chunk production also shows that *left chunk context alone* already drives decent incremental production.
- **TRACX favors asymmetric context in incremental processing.** Recognized chunks serve as left context, while raw items stay on the right. TRELLIS's greedy parser could describe left context at chunk level (already built) and right context at word level (not yet chunked).

### Gaps
- I could not fetch U-MILA's full text (the PDF link returned 404). Its details on slot-collocations and context-based similarity are not verified here.
- PARSER and TRACX quantitative results were not collected.

---

## 5. How chunk context conditions generation

### Takeaway
Syntactic language models condition each generation step on a **stack of already-composed constituents** (left chunk context) plus the open nonterminal (the parent's expectation). Treebank PCFG research shows that **parent annotation** and **horizontal markovization** (conditioning on the parent and on preceding siblings) produce large gains. Learned split hierarchies of categories (Petrov) are effectively taxonomies of contexts. The main caution comes from Transformer Grammars: forcing all context through composed chunk vectors hurts long-range (document-level) modeling.

### Cited Findings
- **RNNG.** RNNGs generate trees and words top-down and left-to-right with a stack of composed constituents. A completed constituent is composed by a BiLSTM over its children and nonterminal label — [Dyer, Kuncoro, Ballesteros & Smith 2016, NAACL, pp. 199–209](https://aclanthology.org/N16-1024/).
- **What RNNGs learn** — [Kuncoro et al. 2017, EACL](https://aclanthology.org/E17-1117/); [arXiv](https://arxiv.org/abs/1611.05774):
  - Explicit composition is crucial.
  - Attention-based composition reveals that "headedness plays a central role in phrasal representation", largely agreeing with hand-written head rules.
  - Phrasal representations "depend minimally on non-terminals", which supports endocentricity.
- **URNNG** trains RNNGs without trees using amortized variational inference — [Kim et al. 2019, NAACL](https://arxiv.org/abs/1904.03746)†.
- **Transformer Grammars** implement recursive syntactic composition with an attention mask over a transformed linearized tree — [Sartran et al. 2022, TACL](https://arxiv.org/abs/2203.00633):
  - They improve sentence-level perplexity and syntax-sensitive evaluations.
  - But "the recursive syntactic composition bottleneck which represents each sentence as a single vector harms perplexity on document-level language modeling".
- **Pushdown Layers** keep "a stack tape that tracks estimated depths of every token in an incremental parse of the observed prefix". The depths softly modulate attention, for example to skip finished constituents — [Murty, Sharma, Andreas & Manning 2023, EMNLP](https://arxiv.org/abs/2310.19089).
  - Result: 3–5× more sample-efficient syntactic generalization at comparable perplexity.
- **GPST** generates a sentence and its tree left to right as an unsupervised syntactic LM — [Hu et al. 2024, ACL](https://arxiv.org/abs/2403.08293).
- **Parent annotation** — [Johnson 1998, CL 24(4)](https://aclanthology.org/J98-4004/):
  - Johnson describes a node's label as "a 'communication channel' that conveys information between the subtree dominated by the node and the part of the tree not dominated by this node". In other words, the label is the interface between inside and outside.
  - Appending the parent's category raised labeled precision from 0.735 to 0.800 and recall from 0.697 to 0.792 on WSJ §22, with rules growing from 14,962 to 22,773.
- **Markovization** — [Klein & Manning 2003, ACL](https://aclanthology.org/P03-1054/):
  - Vertical markovization conditions on ancestors; horizontal markovization conditions on previous siblings.
  - With annotation, an unlexicalized PCFG reaches 86.36% F1. Markovization alone at v = 3, h ≤ 2 gives 79.74.
- **Split–merge refinement** — [Petrov, Barrett, Thibaux & Klein 2006, COLING-ACL](https://aclanthology.org/P06-1055/):
  - Starting from an X-bar grammar, categories are hierarchically split (for example NP into subsymbols such as subject-position NP^S) and merged to maximize likelihood.
  - Result: 90.2% F1 on PTB.
- **Context-conditioned sampling in cognitive models.** ADIOS picks one member per equivalence class *within the parent pattern*, and CBL picks the next chunk given the last chunk produced — see §4 ([ADIOS](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953); [CBL](https://csl-lab.psych.cornell.edu/files/2021/01/2011-mc-cogsci.pdf)).

### Inferences
- **v1's P7 conditioning is parent annotation.** v1's generation (a pool at the maturity level, filtered by the parent's expected context class) is parent annotation applied to sampling. Johnson and Klein & Manning show it is the cheapest big win.
  - The natural next step is **horizontal markovization**: condition each child on the parent's expected class *and* the composition or representation concept of the already-generated left sibling.
  - This is exactly the "chunk context" that is available for free during top-down left-to-right generation, as on the RNNG stack.
- **v1's representation hierarchy is an incremental Petrov hierarchy.** It splits categories by context, and reading it at increasing depth is coarse-to-fine. This supports sampling at a coarse cut and refining the filter with deeper levels only when the pool is large.
- **Endocentricity suggests a head attribute.** Kuncoro's result suggests a composite's representation concept is predictable mostly from its head child. A `head` slot marker in composition instances would let generation choose the head first and then dependents, as head-driven models do.
- **Keep long-range context outside the composition bottleneck.** The Transformer Grammars bottleneck means chunk composites should not be the *only* carriers of context. Long-range information needs a separate memory (§8).
- **Constraint.** Generation is LOCKED in the project (judged qualitatively). These are options to keep in reserve, not proposals to optimize generation scores.

### Gaps
- I found no study that compares *sampling* (as opposed to parsing) under parent vs parent+sibling conditioning in a fragment-reuse generator like TRELLIS's.
- Bayesian tree-substitution and fragment grammars (DOP, Cohn et al., O'Donnell) are relevant to "context-conditioned fragment sampling" but were not re-verified this session.

---

## 6. Theoretical analogs for the composition/representation reframing: does "a concept hierarchy for both" hold up?

### Takeaway
Yes, with nuance. The analogs all alternate two node types:
- AND-OR graphs: AND = composition, OR = alternatives.
- Sum-product networks and probabilistic circuits: product = factorization or chunk, sum = mixture or concept.
- Clark's lattice: string extents vs context intents.

A Cobweb taxonomy is itself a hierarchy of *sum* (mixture) nodes whose concepts are *products* over attributes. So "a taxonomy over compositions" plus "a taxonomy over representations", linked by slot references, is a coherent alternating structure.

Tu, Pavlovskaia & Zhu (2013) give the most actionable recipe: create a composition (AND) and its slot classes (ORs) **together**, as an "And-Or fragment". Accept it when two tables are coherent: the n-gram tensor of slot fillers, and the **context matrix** of configurations × surrounding contexts.

### Cited Findings
- Zhu & Mumford's grammar of images is an And-Or graph where "each Or-node points to alternative sub-configurations and an And-node is decomposed into a number of components". Horizontal links represent "the contexts for spatial and functional relations" — [Zhu & Mumford 2006, *Found. Trends Comput. Graph. Vis.* 2(4):259–362](https://nowpublishers.com/article/Details/CGV-018).
  - Each object category is the set of valid configurations, parsed by recursive top-down/bottom-up procedures.
- Tu, Pavlovskaia & Zhu (2013) unify stochastic And-Or grammars independently of data type. Their context-free subclass extends SCFGs and is "an extension of decomposable sum-product networks" — [NeurIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/24681928425f5a9133504de568f5f6df-Abstract.html).
  - Learning starts from a trivial grammar and iteratively adds **And-Or fragments**: a new And-node whose children are new Or-nodes. This "unifies the search for compositions and reconfigurations".
- Tu et al.'s selection criterion — [Tu et al. 2013](https://proceedings.neurips.cc/paper/2013/hash/24681928425f5a9133504de568f5f6df-Abstract.html):
  - Posterior gain = likelihood gain × prior gain.
  - The likelihood gain factorizes into the coherence of the fragment's **n-gram tensor** (counts of each covered filler configuration) and the coherence of its **context matrix** (rows = configurations, columns = "the surrounding patterns of a configuration", cells = co-occurrence counts).
  - Together these measure "the context-freeness within the And-Or fragment and the context-freeness of the And-Or fragment against its context".
  - Or-rule probabilities follow from reduction counts. The prior penalizes grammar size by α.
- Tu et al.'s results — [Tu et al. 2013](https://proceedings.neurips.cc/paper/2013/hash/24681928425f5a9133504de568f5f6df-Abstract.html):
  - And-nodes can carry *relations* among children, such as "followed-by" or "co-occurring" in event grammars.
  - Event-grammar F-measure: Data 1 was 0.831 (vs ADIOS 0.810, SPYZ 0.756). Data 2 was 0.813 (vs ADIOS 0.204, SPYZ 0.582).
- Sum-product networks — [Poon & Domingos 2011, UAI](https://arxiv.org/abs/1202.3732)†. Probabilistic circuits as a unifying framework — [Choi, Vergari & Van den Broeck 2020](http://starai.cs.ucla.edu/papers/ProbCirc20.pdf)†.
- LearnSPN learns structure recursively. It tries to "partition variables into approximately independent subsets", producing a *product*. Otherwise it clusters "subsets of similar instances", producing a *sum* — [Gens & Domingos 2013, ICML, PMLR 28(3):873–880](https://proceedings.mlr.press/v28/gens13.html).
- Clark's syntactic concepts are (strings, contexts) pairs in a Galois lattice (Formal Concept Analysis). The primal and dual learners build nonterminals from strings and from contexts respectively — §1; [Clark ICGI 2010](https://www.its.caltech.edu/~matilde/ClarkSyntacticConceptLattice.pdf); [Yoshinaka 2011](https://www.springerprofessional.de/towards-dual-approaches-for-learning-context-free-grammars-based/3686398).

### Inferences
- **The two-taxonomy design is principled.** LearnSPN's sum-vs-product decision mirrors TRELLIS's own split:
  - *clustering instances* is Cobweb categorization, i.e. sum or concept;
  - *grouping variables into a joint unit* is chunking, i.e. product or composition.
  - A Cobweb tree is a restricted SPN: concept nodes mix their children, and each concept's attribute-independence makes it a product over attributes.
  - So the "composition hierarchy" is a sum-taxonomy over product-type objects, and the "representation hierarchy" is a sum-taxonomy over context descriptions whose concepts serve as the OR-nodes, or substitution classes, of composition slots.
- **Tu's And-Or fragment is the joint creation event v2 needs.** A new composition concept and its slot representation concepts are proposed together. The **context matrix** test is directly computable from Cobweb counts:
  1. For a candidate composition concept, tabulate (slot-filler configuration) × (context concept at a coarse depth).
  2. If contexts are roughly independent of which configuration occurred (near rank 1), the fillers are true substitution classes *for that slot in that context*, and the composite is "context-free against its context".
  3. If not, the slot class should be split (ADIOS-style locality) or the composite rejected.
  - This gives an unsupervised replacement or complement for the count/maturity gate. It decides both "should we merge?" and "are these slot classes real?".
- **Zhu & Mumford's horizontal links answer "semantic network relational input".** Relations between parts are attributes of the AND node. Giving each composition instance relation attributes (TRESTLE-flattened, §3) lets the *same* composition concept license both a merge (syntax) and an idea (semantic structure). That maps composition rules onto both "deciding merging" and "deciding ideas".
- **Where trees fall short of the lattice.** Clark's lattice allows a string to belong to several concepts (multiple parents), but Cobweb trees are single-parent. v1's bag-of-concepts attributes and Cobweb/4L-style multi-node descriptions are pragmatic approximations of lattice membership. This is a known representational limit, not a reason to abandon taxonomies.

### Gaps
- I did not find an explicit published equivalence proof between Cobweb taxonomies and SPNs; the correspondence above is my inference.
- Tu et al. (2013) report results on events and images, not natural-language treebanks.

---

## 7. Variable-arity chunks and multiple templates per chunk

### Takeaway
Systems handle n-ary or variable-slot chunks in five ways:
1. **Templates with slots**: CHREST (core + slots with defaults), HVM (chunks with embedded variables), ADIOS (patterns with equivalence-class slots).
2. **And-Or fragments of arity n**: Tu et al. 2013.
3. **Component sets aligned by structure mapping**: TRESTLE, LABYRINTH.
4. **Head-plus-markovized dependents**: Klein & Manning 2003 horizontal markovization.
5. **Multi-hole contexts for discontinuous constituents**: Clark & Yoshinaka's PMCFG learning.

### Cited Findings
- CHREST templates have an unchanging core plus slots with default values — [Gobet & Simon 1996](https://bura.brunel.ac.uk/handle/2438/1339).
- HVM chunks have arbitrary length and embed variables at any position (A⊕V⊕ED) — [Wu et al. 2025](https://arxiv.org/html/2410.21332).
- ADIOS patterns are variable-length paths whose positions can be equivalence classes valid only in the pattern's context — [Solan et al. 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953).
- An And-Or fragment has n Or-nodes, each with m_i alternatives. Its evidence is an n-way tensor of configuration counts — [Tu et al. 2013](https://proceedings.neurips.cc/paper/2013/hash/24681928425f5a9133504de568f5f6df-Abstract.html).
- TRESTLE handles any number of components, aligned to a concept by structure mapping before flattening — [TRESTLE](https://arxiv.org/abs/2410.10588). LABYRINTH composites are sets of components, which may themselves be composite — [LABYRINTH](https://mlanthology.org/icml/1989/thompson1989icml-incremental).
- Horizontal markovization generates n-ary rules child by child, conditioned on h previous siblings. Klein & Manning chart how F1 and grammar size trade off across settings — [Klein & Manning 2003](https://aclanthology.org/P03-1054/).
- PMCFG learning uses tuples of strings (discontinuous constituents) with copying, learned distributionally from positive data and membership queries — [Clark & Yoshinaka 2014, *Machine Learning* 96:5–31](https://link.springer.com/article/10.1007/s10994-013-5403-2).
  - An unreviewed August 2026 preprint extends syntactic concept lattices to tuples of arbitrary arity. It finds compression-stabilization heights of only 1, 2, or ∞ — [Kuriyama 2026](https://arxiv.org/abs/2608.29639). Its relevance to TRELLIS is limited.

### Inferences
- **Option A: keep binary merges and add template concepts.** When a chain of binary composites is consistently reused, as with v1's need for `flatten=("VP","VPobj")` to remove hollow-tree artifacts, promote it to an n-ary *template* composition concept.
  - The instance carries `arity`, an ordered slot list, and a `template` id. CU will separate templates of different arity because `arity` is a visible attribute, just as the complexity tag separated NP from S.
- **Option B: head-markovized n-ary.** The composition instance holds `head` slot attributes plus *bags* of left and right dependent representation concepts, with counts by position. Generation picks the head, then dependents conditioned on the head and the previous dependent. This reuses the markovization evidence and Kuncoro's endocentricity result.
- **Option C: TRESTLE components for semantic-network input.** Use component sets with structure mapping when inputs come without a canonical order, such as semantic-network fragments or sets of relations.
- **Multi-hole contexts point to long-range structure.** A representation instance with *two* holes describes a discontinuous constituent. This connects variable arity to long-range dependencies (§8).

### Gaps
- I found no empirical comparison of template vs markovized vs component-set encodings inside a Cobweb-style categorizer.

---

## 8. Long-range dependencies with discrete symbols: the minimal attention-like mechanism

### Takeaway
The minimal, cognitively grounded mechanism is **cue-based retrieval** over a memory of chunks (Lewis & Vasishth):
- at a dependency site, features of the current element act as cues;
- all stored chunks are matched in parallel;
- activation decays with time.

Modern work adds two lessons:
- measure distance in **chunk/depth space** rather than token offsets (Pushdown Layers);
- keep this memory **separate from the composition bottleneck** (Transformer Grammars).

### Cited Findings
- Sentence processing is modeled as skilled memory retrieval over chunks. Dependencies such as subject–verb are resolved by cue-matching retrieval with activation decay and similarity-based interference — [Lewis & Vasishth 2005, *Cognitive Science* 29(3):375–419](https://doi.org/10.1207/s15516709cog0000_25) (DOI resolves; mechanism from prior knowledge†).
- Representing each sentence through a single composed vector harms document-level modeling, so successful long-text models need memory independent of syntactic composition — [Transformer Grammars](https://arxiv.org/abs/2203.00633).
- Depth-aware attention over an incremental parse lets a model skip closed constituents, improving syntactic generalization 3–5× in sample efficiency — [Pushdown Layers](https://arxiv.org/abs/2310.19089).
- R2D2 predicts a word from its left and right *abstraction nodes*, so context is accessed at the chunk level rather than the token level — [R2D2](https://aclanthology.org/2021.acl-long.379/).
- Discontinuous dependencies can be captured by multi-hole contexts in distributional PMCFG learning — [Clark & Yoshinaka 2014](https://link.springer.com/article/10.1007/s10994-013-5403-2).
- The sibling Cobweb-LLM project (project notes):
  - It categorizes (anchor, context, offset) pairs into a "pair tree" whose concepts behave like attention heads.
  - Hybrid-Cobweb multi-node prediction there beats a count table, skip-gram and CBOW at gap filling on *Oz* (0.257 top-1) and Grimm (0.260).

### Inferences
- **Minimal TRELLIS mechanism: a third, "dependency" Cobweb tree** over pair instances:
  - Attributes: the representation concept of the current chunk at multiple depths; that of a candidate earlier chunk; direction; offset in *closed chunks* (pushdown-style); and relative depth.
  - Its concepts are discrete "heads". This is the pair-tree design from the sibling project, lifted from words to chunks.
- **Use at parse and generation time** (Lewis & Vasishth-style):
  1. The current chunk's representation concept queries the dependency tree for an expected partner concept (the cue).
  2. A buffer holding recent chunks of *all levels* is scored by cue match × recency decay.
  3. The retrieved chunk's ids enter the current representation instance as a `dep.*` attribute.
  - This keeps long-range information outside the composition bottleneck, as Transformer Grammars recommend.
- **Chunk-space distance turns long dependencies into short ones.** Distance is measured over closed constituents, so a long-distance subject–verb link spanning a relative clause becomes "1–2 chunks back".

### Gaps
- I found no published Cobweb or concept-formation system implementing cue-based retrieval for syntactic dependencies. The proposal above is an inference joining two literatures.

---

## Implications for TRELLIS v2

### Takeaway
The evidence converges on six representation principles:
1. A chunk's *category* is the context closure of its *composition* (Clark's lattice; Johnson's "communication channel"; CCM; Clark 2001).
2. Every reference to a learned concept should be **multi-granular**: word or leaf plus several taxonomy depths (Koo/Brown; MacLellan Path; CBL; v1's own finding).
3. **Chunk context** becomes available after an inside pass. Its natural form is the **spine** of sibling chunks at each ancestor level (DIORA outside, R2D2, HVM parse-then-count, Leaf multi-pass).
4. Merge decisions need **distituent** evidence and **left–right context dependence** (CCM, Clark 2001, Tu's context matrix).
5. Generation should condition on the parent's expected representation, already done in v1, and optionally on left siblings (ADIOS, parent annotation, markovization).
6. Long-range dependencies need a **separate retrieval memory** (Transformer Grammars bottleneck; Lewis & Vasishth).

### Cited Findings (evidence anchors for the designs below)
- Path (all-ancestor) context beat leaf-label context in Cobweb, with homonym ARI ≈ 1 vs ≈ 0.25 for words and ≈ chance for leaves — [MacLellan, Matsakis & Langley 2022](https://arxiv.org/abs/2212.11937).
- Parsers use coarse (4–6 bit) and fine (full) cluster features together — [Koo et al. 2008](https://aclanthology.org/P08-1068/).
- Word-level statistics beat 13 coarse classes for chunking child-directed speech — [McCauley & Christiansen 2011](https://csl-lab.psych.cornell.edu/files/2021/01/2011-mc-cogsci.pdf).
- Outside(span) = f(outside(parent), inside(sibling)); [inside; outside] makes strong phrase representations — [DIORA](https://aclanthology.org/N19-1116/). A hard single tree beats a soft mixture — [S-DIORA](https://aclanthology.org/2020.emnlp-main.392).
- Variables are chunks sharing preceding and succeeding chunks; chunk statistics are updated after parsing — [HVM](https://arxiv.org/html/2410.21332).
- A distituent cluster (CCM 71.9 → DMV+CCM 77.6 UF1 on WSJ10) — [Klein & Manning 2004](https://aclanthology.org/P04-1061/). Left×right mutual information — [Clark 2001](https://aclanthology.org/W01-0713/). Context-matrix coherence — [Tu et al. 2013](https://proceedings.neurips.cc/paper/2013/hash/24681928425f5a9133504de568f5f6df-Abstract.html).
- Class validity local to parent patterns — [ADIOS](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953). Parent annotation +6–9 points P/R — [Johnson 1998](https://aclanthology.org/J98-4004/).
- Composition bottleneck harms long-range modeling — [Transformer Grammars](https://arxiv.org/abs/2203.00633). Cue-based retrieval — [Lewis & Vasishth 2005](https://doi.org/10.1207/s15516709cog0000_25).
- Relations flattened into attribute-value form after structure mapping — [TRESTLE](https://arxiv.org/abs/2410.10588).

### Inferences: ranked representation designs
Ranking weighs strength of evidence × fit with the pillars (attribute-value, incremental Cobweb, parse + generate, interpretable) ÷ implementation risk.

**D1 (rank 1): Multi-granular references everywhere ("path encoding").**
- **What.** Every attribute that points at a learned concept (content slots, context neighbors, spine siblings) is written at several granularities, each as its *own* attribute:
  - `.word` or `.leaf` for near-lexical identity;
  - `.d2` (coarse, ≈POS);
  - `.d4` (v1's current level);
  - `.cut` (the maturity-τ ancestor).
  - Values are small weighted bags (top-k), as in v1 and Cobweb/4L.
- **Why.** CU can then choose, per split, the granularity that is informative. A dense grammar gets class-level generalization, and a sparse lexicon keeps word identity.
  - This turns v1's "concept ids help dense, hurt sparse" trade-off into a learned choice.
  - Path, Koo and CBL all point the same way.
- **Risks.**
  - Attribute count grows, so context can swamp CU. Mitigate with one bag per level and per side, as Cobweb/4L uses `context-before`/`context-after`, rather than per-position attributes.
  - Reference churn when the representation tree restructures. Adopt Path-style back-pointers, or keep v1's canonicalize-and-rewrite.

**D2 (rank 2): "Compose, then contextualize" two-pass descriptions with spine chunk context.**
- **What.**
  1. **Pass A (inside).** Build the composition instance from slot references (D1) and the seam, and sort it into the composition hierarchy with no context. This "crafts the representation compositionally" first.
  2. **Pass B (outside).** Build the representation instance: the element's own composition concept (multi-depth), boundary-edge words, the distance-weighted word window, the *same window re-described by chunk ids*, and the **spine** (side + sibling concept at each ancestor level 1…K). Sort it into the representation hierarchy. This is Clark's closure step and DIORA's outside step.
  3. **Pass C (optional).** Re-describe slots with Pass-B representation concepts and re-sort, as in the Leaf system's iterations or ReCAT's stacked layers.
- **Where chunk context comes from.**
  - *Training* (gold trees): every neighbor chunk is known, so build the spine directly, as HVM's parse-then-count does.
  - *Inside-outside parsing*: the outside instance of span (i,k) under parent (i,j) is the parent's spine plus the sibling (k,j)'s inside concept. Use hard top-k per cell (S-DIORA).
  - *Greedy fallback*: describe left context with already-built chunks (stack) and right context with words (TRACX/RNNG asymmetry).
- **Why.** This answers the user's question of how to craft a representation compositionally before enriching it contextually. It implements "latents take all levels of content before and after" in discrete form.
- **Risks.**
  - Cost: O(n²) spans × K levels in a chart. Cap K at 3–4 and use the coarse depths for the spine.
  - Order effects: Pass B depends on Pass A's ids. Use non-modifying sorts during passes, as in the Leaf system, and commit only at the end.

**D3 (rank 3): Constituent/distituent context model plus left×right dependence as the merge signal.**
- **What.**
  - During training, also store representation instances for **distituent spans** (spans crossing gold brackets), with `role=distituent` or in a sibling subtree.
  - Add a conjunctive `ctx.LR.pair` attribute (left × right at a coarse depth).
  - At parse time, score a candidate merge by the CCM-style odds P(rep-instance | constituent) / P(rep-instance | distituent) from Cobweb predictions, combined with v1's ancestor-support gate.
- **Why.** This is CCM's key move and Clark 2001's mutual-information criterion, both learned from the same instances. It should mainly reduce **commission** (spurious merges) without supervision beyond the gold trees already used.
- **Risk.** Distituent instances outnumber constituents by O(n) per sentence. Subsample them, or store them in their own tree.

**D4 (rank 4): Composition concepts as And-Or fragments (chunks with variables), gated by context-matrix coherence.**
- **What.** A composition concept's slots reference representation concepts (HVM's variables, Tu's Or-nodes).
  - Promote a composition concept to "mature" when its configuration × context table, kept as counts at a coarse depth, is near rank 1. This means contexts do not depend on which fillers occurred (Tu's context-freeness).
  - If not, split the slot class locally (ADIOS) instead of globally.
- **Why.** It decides "merge?" and "is this slot a real substitution class?" together. It addresses v1's roughly 90% intermediate-node purity, and makes P7 conditioning principled rather than a filter.
- **Risk.** Count tables per composition concept cost memory. Keep them only for candidate concepts near the maturity cut.

**D5 (rank 5): Variable arity via templates and a head marker.**
- **What.** Add `arity`, `template` and `head` attributes to composition instances. Promote stable binary chains to n-ary templates (CHREST/HVM). Optionally use head-markovized dependent bags (Option B in §7).
- **Why.** It removes hollow-tree artifacts structurally and supports endocentric generation. CU separates arities the same way the complexity tag separated NP from S.
- **Risk.** It interacts with the locked generation pipeline, so introduce it only on the parsing side first.

**D6 (rank 6): Relational (semantic-network) attributes in composition instances.**
- **What.** TRESTLE-flattened relation attributes, such as `rel(agent,S1,S2)` or `rel(mod,S2,S1)`, plus component alignment by structure mapping when input order is not given.
- **Why.** The same composition concept then licenses a syntactic merge and a semantic idea. This maps composition rules onto both "deciding merging" and "deciding ideas", as AND-node relations do in Zhu–Mumford and Tu.
- **Risk.** It requires a relational input source and is a larger change. It is best staged after D1–D4.

**D7 (rank 7): Dependency ("attention") tree with cue-based retrieval for long range.**
- **What.** A third Cobweb tree over (anchor chunk, earlier chunk, offset in chunks, Δdepth) pair instances. Retrieval cues come from the current chunk's representation concept, and retrieved partners enter the representation instance as `dep.*` attributes.
- **Why.** It is the minimal discrete attention, consistent with the sibling pair-tree project and outside the composition bottleneck.
- **Risk.** It adds a third hierarchy. Validate first on agreement-style long-range phenomena.

**D8 (rank 8, diagnostic): a discrete proform/substitution test.** A span is constituent-like if its representation instance sorts into a representation concept that also contains single primitives. This is cheap and interpretable, and it serves as a check on D3.

### Example instance layouts
The examples use the sentence "the cat saw a dog".
- R… are representation-hierarchy concept ids and C… are composition-hierarchy concept ids.
- `.d2`, `.d4` and `.cut` are taxonomy depths; `.leaf` is the leaf concept.
- Values are weighted bags. Weights follow Cobweb/4L's 1/(d+1) for windows and top-k membership for concept bags.

```
# (1) COMPOSITION hierarchy instance — built in Pass A (inside), no context
#     composite "saw a dog", span (2,5)
arity          : 2
template       : T_bin                    # n-ary templates get their own ids (D5)
head           : S1                       # optional endocentric marker (D5)
S1.word        : {saw: 1.0}               # lexical anchor (primitive child; sparse-lexicon safety, D1)
S1.rep.d2      : {R3: 1.0}                # coarse class (≈ V)
S1.rep.d4      : {R31: .7, R34: .3}       # v1-style top-k bag
S1.rep.leaf    : {R3107: 1.0}
S1.cplx        : 0
S2.rep.d2      : {R1: 1.0}                # coarse class (≈ NP)
S2.rep.d4      : {R12: 1.0}
S2.comp.d3     : {C45: 1.0}               # child is a composite: its own composition concept
S2.cplx        : 1
seam.word      : (saw|a)                  # junction pair as ONE conjunctive value
seam.rep.d2    : (R3|R7)
rel            : {(obj S1 S2): 1.0}       # optional, semantic-network input (D6)
```

```
# (2) REPRESENTATION hierarchy instance — built in Pass B (outside) for the same span (2,5)
self.comp.d2   : {C4: 1.0}                # what composes it (pointer into composition tree)
self.comp.d4   : {C45: 1.0}
edge.L.word    : {saw: 1.0}               # boundary words = what neighbors see (v1 outermost positions)
edge.R.word    : {dog: 1.0}
ctx.L.word     : {cat: 1.0, the: 0.5}     # distance-weighted window (k=5 in v1)
ctx.R.word     : {</s>: 1.0}
ctx.L.rep.d2   : {R1: 1.0, R7: 0.5}       # same window re-described by representation ids (D1)
ctx.L.rep.cut  : {R1x: 1.0}
ctx.L.chunk.d2 : {R1: 1.0}                # nearest *chunk* to the left ("the cat"), not its words
spine.1        : (L, R1)                  # sibling at ancestor level 1 = "the cat" (NP-like)
spine.2        : (-, ROOT)                # no further siblings; root term (DIORA's learned root)
ctx.LR.pair.d2 : (R1|</s>)                # Clark-2001 left×right dependence as one value (D3)
role           : constituent              # 'distituent' for crossing spans stored in training (D3)
dep.subj.d2    : {R1: 1.0}                # optional, retrieved long-range partner (D7)
```

```
# (3) REPRESENTATION instance for a primitive, "dog" at position 4
self.word      : {dog: 1.0}
ctx.L.word     : {a: 1.0, saw: 0.5, cat: 0.33, the: 0.25}
ctx.R.word     : {</s>: 1.0}
ctx.L.rep.d2   : {R7: 1.0, R3: 0.5, R1: 0.33}
spine.1        : (L, R7)                  # "a"          (sibling at level 1)
spine.2        : (L, R3)                  # "saw"        (sibling of "a dog")
spine.3        : (L, R1)                  # "the cat"    (sibling of "saw a dog")
```

```
# (4) DEPENDENCY ("attention") instance — third tree (D7)
anchor.rep.d2  : {R3: 1.0}                # "saw"
other.rep.d2   : {R1: 1.0}                # "the cat"
other.rep.d4   : {R12: 1.0}
dir            : L
offset.chunks  : 1                        # closed constituents between (pushdown-style)
offset.words   : 1
depth.delta    : +1
```

**Processing-order sketch (inside-outside setting).**
1. For each chart cell, Pass A sorts the composition instance for the top-k splits.
2. Top-down, Pass B builds outside instances: spine(i,k) = [(R, inside-concept(k,j))] + spine(i,j).
3. Each span is then scored by three things: composition-concept maturity (v1 gate or D4), representation-concept constituent/distituent odds (D3), and inside–outside agreement.
4. Memory is written only from gold trees during training, preserving the project rule.

### Gaps
- None of D1–D8 has been tested inside Cobweb on TRELLIS's grammars. The rankings rest on cross-system evidence, not TRELLIS experiments, and should be confirmed in omission/commission terms on the small, medium and large grammars.
- Spine context grows with tree depth. The right K, and whether spine attributes should be coarse-only, are open empirical questions.
- The HVM summary contains an internal inconsistency on NLL vs LZ78 (noted in §4). Check the paper's table before citing those numbers externally.
- The web-search budget ran out before a complete 2025–2026 sweep. Recent latent-tree, chunking or Cobweb papers after mid-2026 may be missing. Of the 2026 items, the only ones found were Kuriyama's lattice preprint (Aug 2026) and Wu's thesis revision (June 2026).
