# Unsupervised grammar induction: how learners propose and evaluate structure, stay incremental, and score on the Penn Treebank — notes for TRELLIS v2

_Compiled 2026-10-03, covering work through October 2026. Almost every number below was checked against the primary paper's PDF text, its tables or the authors' code. Secondary aggregators are marked as such._

**How to read the numbers**
- **The literature reports unlabeled bracket F1.** "S-F1" is sentence-level: F1 computed per sentence, then averaged. "C-F1" / "corpus F1" / "micro" is computed from pooled counts.
- **Mapping to TRELLIS's GRIDS-style reporting** (bracket level):
  - omission rate = 1 − unlabeled span recall;
  - commission rate = 1 − unlabeled span precision;
  - F1 = harmonic mean of (1 − omission) and (1 − commission).

  For paper text, report omission/commission. Use the harmonic mean only where a comparison table needs it, and say it equals what other papers call unlabeled F1.
- **GRIDS's own omission and commission are language-level** (Langley & Stromsten 2000):
  - omission: target sentences the learned grammar cannot parse;
  - commission: learned-grammar outputs the target grammar rejects.

  Bracket recall and precision are the structural analogues of these (Section 5).
- **Protocols differ, so numbers move.** PTB numbers change by up to about 20 points with protocol: punctuation, binarization, micro vs macro averaging, trivial spans, vocabulary size, train = test. Every number below states its protocol; do not compare across protocols.
- **Citation fix.** The ACL 2020 "Empirical Comparison of Unsupervised Constituency Parsing Methods" is by **Li, Cao, Cai, Jiang & Tu**, not "Li, Mou & Keller". Li, Mou & Keller wrote an ACL 2019 imitation-learning parser.

## 1. The symbolic chunk-and-merge lineage: operators, search control, objectives, incrementality and results

### Takeaway
From Wolff's SNPR through GRIDS, Stolcke–Omohundro, Chen and the e-GRIDS family, the symbolic lineage converged on one design:
- two structure operators: *chunk/create*, which makes a new nonterminal for a recurring sequence, and *merge*, which forms a class of substitutable symbols;
- a simplicity criterion: compression, two-part MDL, or a Bayesian posterior with a description-length prior;
- greedy or small-beam search, monotone toward generality.

Later systems differ mainly in how they *propose* candidates:
- adjacent-pair frequency (SNPR, GRIDS, Sequitur, e-GRIDS);
- alignment and substitutability (EMILE, ABL);
- graph-path significance tests (ADIOS);
- suspicious coincidence in a decaying short-term memory (U-MILA);
- context-distribution clustering with a mutual-information filter (Clark);
- biclustering or And-Or fragments scored by posterior gain (Tu).

Incremental variants exist and work on toy grammars: Stolcke's batches of 1–10 samples, Chen's sentence-at-a-time triggers, Synapse, Sequitur, U-MILA.

On real treebank text, the bracketing results of this family were weak. ABL on WSJ §23 scored F 19.26, below random. On ATIS, EMILE (25.4), ABL (39.2) and Clark's CDC (42.0) all fell below right-branching (42.9). Generation and coverage results (ADIOS, U-MILA, GRIDS learning curves) were stronger.

Goldsmith's explicit split — "heuristics … for obtaining candidate analyses" vs MDL "for evaluating proposed analyses" — is the cleanest statement of the propose/evaluate pattern TRELLIS v2 needs.

### Cited Findings

**Wolff: MK10, SNPR (1982) and SP theory**
- **Citation.** Wolff, J. G. (1982), *Language & Communication* 2(1):57–89, DOI 10.1016/0271-5309(82)90035-0. [Source](https://doi.org/10.1016/0271-5309(82)90035-0)
- **Operators.** SNPR14's operators, with quotes from the author's full text:
  - *Building*: "selects the most frequently occurring pair … and adds that pair to its grammar as a new element".
  - *Folding*: forms disjunctive PAR (class) elements from "those two elements which occur most frequently in one or more shared contexts", then inserts them into other contexts, which can create recursion "without ad hoc provision".
  - *Rebuilding*: corrects overgeneralization when "some of the constituents of the PAR element fail to occur within the context of the containing C element".

  [Source](http://web.archive.org/web/20221017150543/https://www.cognitionresearch.org/papers/ll/lang_comm_1982/wolff_1982.doc)
- **Objective.** Compression capacity CC = (V − v)/V. "At every step, [choose] that structure which gives the greatest improvement in CC for unit increase in Sg" (grammar size). Wolff notes that "quantified tests of these suppositions have not yet been attempted". [Source](http://web.archive.org/web/20221017150543/https://www.cognitionresearch.org/papers/ll/lang_comm_1982/wolff_1982.doc)
- **Data and arity.** Trained on artificial, unsegmented letter strings only: "No attempt has yet been made to run it on natural language texts." It works by repeated scans or on new text per scan. Chunks are formed pairwise but stored as flat, variable-length strings. [Source](http://web.archive.org/web/20221017150543/https://www.cognitionresearch.org/papers/ll/lang_comm_1982/wolff_1982.doc)
- **MK10.** Reported to form a chunk "whenever a digram is seen more than 10 times", replacing all occurrences. [Source](https://arxiv.org/abs/cs/9709102)
- **SP70 (Wolff 2003).**
  - Processes new patterns one at a time through multiple alignment against stored patterns, then a sifting-and-sorting step that scores alternative grammars by minimum length encoding.
  - On a four-sentence toy, trained on 3 of the 4 sentences, it generates the missing one.
  - It is "not good at finding intermediate levels of abstraction".

  [Source](https://arxiv.org/abs/cs/0302015)
- **The 2006 SP book.** *Unifying Computing and Cognition* (ISBN 0-9550726-0-3). [Source](http://web.archive.org/web/20180813103233/http://www.cognitionresearch.org:80/books/sp_book/fly_leaf.htm)

**Langley (1994, 1995) and GRIDS (Langley & Stromsten 2000)**
- **Citation.** Langley, P. & Stromsten, S. (2000), "Learning context-free grammars with a simplicity bias", ECML 2000, LNAI 1810, pp. 220–228, DOI 10.1007/3-540-45164-1_23. It is "a rational reconstruction of Wolff's SNPR". [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf); [Source](https://mlanthology.org/ecmlpkdd/2000/langley2000ecml-learning)
- **Initial grammar.** A flat grammar of one S → X…Y rule per training sentence plus X → W for each word, which "covers all (and only) the training sentences". [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **Operators.**
  - *Create*: "all ways of creating new terms … from pairs of nonterminal symbols that occur in sequence", with the sequence substituted everywhere. These are binary chunks.
  - *Merge*: merge two nonterminals into one class; this can yield recursion.
  - "Symbol creation does not change the coverage of a grammar, and symbol merging can never decrease the coverage."

  [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **Search.** "Beam search, with a beam size of three". It alternates a merge mode and a create mode, switching mode when no successor improves, and halts when neither improves. [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **Objective.** Description length of the grammar plus the training sentences "encoded as derivations". Each nonterminal token costs log(N+1) bits; each terminal costs log P_i, where P_i is the size of its part of speech; each derivation step costs log R bits. [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **Evaluation (the GRIDS definitions).**
  - Errors of omission are "failures to parse sentences in the target language", i.e. an undergeneral grammar.
  - Errors of commission are "failures to generate only sentences in the target language", i.e. an overgeneral grammar.
  - Both are estimated by sampling: the fraction of target-generated sentences parsed by the learned grammar, and the fraction of learned-grammar sentences parsed by the target.
  - Test sentences are longer than training sentences to probe recursion. Results average 20 training sets.

  [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **Results.**
  - Adjective grammar: "after 120 training cases, the learned grammars cover 95% of the positive test set, and all generated strings are legal".
  - Relative-clause grammar: parsing reaches 100% after 15 items. Generation "falls to below 60% by the fourth case, then rebounds to perfect accuracy after processing 11 training sentences".
  - Doubling or tripling the word classes slows learning roughly linearly.

  [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **Stated limitations.** One category per word; merge cost "increases with the square of the number of words". Proposed remedies are pair filtering by co-occurrence statistics and "an incremental version of Grids that processes only a few training sentences at a time". [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **Langley 1994**, "Simplicity and representation change in grammar induction" (unpublished manuscript, Stanford Robotics Laboratory). Per Stolcke's thesis, it used merging and chunking operators with a simplicity bias measured by total RHS length, and "no incremental learning strategy is described". Its grammars reappear in GRIDS. [Source](http://web.archive.org/web/20070611231208/http://www.icsi.berkeley.edu/ftp/global/pub/ai/stolcke/thesis.ps.Z)
- **Langley 1995**, *Elements of Machine Learning*, Ch. 9: "The formation of transition networks", including §9.3 "Forming recursive transition networks". [Source](https://shop.elsevier.com/books/elements-of-machine-learning/langley/978-0-08-050545-9)

**e-GRIDS and eg-GRIDS (Petasis et al. 2004)**
- **e-GRIDS** (*Grammars* 7:69–110). Operators:
  - CreateNT, a binary pair;
  - MergeNT;
  - a new CreateOptionalNT, which makes a symbol optional.

  It uses beam search over the operator modes and an MDL objective (grammar description length + data description length). [Source](http://web.archive.org/web/20070710200647/http://www.iit.demokritos.gr:80/~paliourg/papers/GRAMMARS2004.pdf)
- **e-GRIDS efficiency result.** The analytical change in description length shows that "to create the M best scoring successor grammars, it suffices to apply the CreateNT operator using the M bigrams with the highest frequencies", which is "totally equivalent to the exhaustive enumeration". Merge gains are forecast without building the successor grammar. [Source](http://web.archive.org/web/20070710200647/http://www.iit.demokritos.gr:80/~paliourg/papers/GRAMMARS2004.pdf)
- **e-GRIDS results.**
  - On GRIDS's grammars: grammar (b) reaches 1.0 on all measures with ≥20 examples. Grammar (a) reaches 1.0 with beam 10 at 600 sentences, but still has a 0.15 probability of generating ungrammatical sentences at 700 sentences with beam 3.
  - On POS-abstracted SemCor, after 2,000 training sentences the learned grammar parsed 29 of 200 unseen sentences, vs 25 for the initial grammar.
  - An incremental mode is supported but was not evaluated.

  [Source](http://web.archive.org/web/20070710200647/http://www.iit.demokritos.gr:80/~paliourg/papers/GRAMMARS2004.pdf)
- **eg-GRIDS** (ICGI 2004, LNAI 3264, pp. 223–234).
  - Five operators: merge; create from n-grams of length 2 up to the longest rule; create-optional; detect center embedding (AABB → X → A X B); rule-body substitution.
  - Adds steady-state genetic search alongside the beam.
  - It is "more than an order of magnitude faster" than e-GRIDS, but converges to over-general grammars (commission errors) on small training sets.

  [Source](https://doi.org/10.1007/978-3-540-30195-0_20)

**Stolcke & Omohundro (1994): Bayesian model merging**
- **Citations.** ICGI-94, LNAI 862, pp. 106–118; Stolcke's 1994 Berkeley PhD thesis. [Source](https://arxiv.org/abs/cmp-lg/9409010); [Source](http://web.archive.org/web/20070611231208/http://www.icsi.berkeley.edu/ftp/global/pub/ai/stolcke/thesis.ps.Z)
- **Operators.**
  - Data incorporation: each sample enters as an ad hoc rule.
  - merge(X1, X2), "paradigmatic".
  - chunk(X1…Xk), "syntagmatic", for arbitrary k; it requires the sequence to "occur at least twice".
  - Unchunk/rechunk to undo or redo chunks.

  [Source](http://web.archive.org/web/20070611231208/http://www.icsi.berkeley.edu/ftp/global/pub/ai/stolcke/thesis.ps.Z)
- **Objective.** The structure posterior P(M_S|X) ∝ P(M_S)·P(X|M_S), with a description-length prior log P(M_S) = −DL(M_S), a Dirichlet parameter prior and Viterbi approximations. [Source](http://web.archive.org/web/20070611231208/http://www.icsi.berkeley.edu/ftp/global/pub/ai/stolcke/thesis.ps.Z)
- **Search.** Best-first "often fails because chunking typically has to be followed by several merging steps". The thesis therefore adds multi-level best-first search and beam search with widths of 3–10. [Source](http://web.archive.org/web/20070611231208/http://www.icsi.berkeley.edu/ftp/global/pub/ai/stolcke/thesis.ps.Z)
- **Incremental variant.**
  - Incorporate a few new samples, then merge best-first until the posterior drops, then repeat. Batches of "between 1 and 10 samples at a time [are] good choices", and merging should not start before about 10–20 samples.
  - A prior weight λ controls early overgeneralization.
  - It is "the default method used in all the experiments" for SCFGs.

  [Source](http://web.archive.org/web/20070611231208/http://www.icsi.berkeley.edu/ftp/global/pub/ai/stolcke/thesis.ps.Z)
- **Results.**
  - Matched all of Cook et al.'s formal-language grammars.
  - Langley's relative-clause grammar: "chunking and merging of 100 random samples produces a grammar that is weakly equivalent".
  - Presenting samples in length order sharply cut search: 212 vs 1,374 merges on the adjective grammar. The unordered run on the relative-clause grammar "had to be aborted".
  - On the 1,200-sentence BeRP corpus it found plausible categories, but "generalization … nowhere near what would be required".

  [Source](https://arxiv.org/abs/cmp-lg/9409010); [Source](http://web.archive.org/web/20070611231208/http://www.icsi.berkeley.edu/ftp/global/pub/ai/stolcke/thesis.ps.Z)

**Chen (1995): Bayesian grammar induction for language modelling**
- **Citation.** ACL 1995, pp. 228–235. [Source](https://aclanthology.org/P95-1031/)
- **Objective.** p(O|G)·p(G) with p(G) = 2^−l(G) (universal prior). [Source](https://aclanthology.org/P95-1031/)
- **Moves.** A → B C; A → B | C; A → A B | B. [Source](https://aclanthology.org/P95-1031/)
- **Triggers.** A move is considered only if it is "triggered" in the sentence currently being parsed, for example by adjacent symbols in its Viterbi parse. [Source](https://aclanthology.org/P95-1031/)
- **Search.** Greedy hill-climbing with predicted likelihood changes. [Source](https://aclanthology.org/P95-1031/)

- **Incrementality.** "We parse the first sentence of the training data and search for the optimal grammar over just that one sentence … repeat … parsing each sentence but once." Delaying parsing until the previous sentences are processed "should yield more accurate Viterbi parses". An inside-outside post-pass follows. [Source](https://aclanthology.org/P95-1031/)
- **Results (entropy, bits/word).**

  | Domain | Chen | Best n-gram | Inside-outside | Ideal grammar |
  |---|---|---|---|---|
  | English-like artificial PCFG | 2.37 | 2.46 | 2.60 | 2.30 |
  | WSJ part-of-speech sequences | 3.15 | 3.01 | 3.93 | — |

  Run time is "essentially … linear in the size of the training data". [Source](https://aclanthology.org/P95-1031/)

**Synapse (Nakamura)**
- **Citations.** Nakamura & Matsumoto 2005, *Pattern Recognition* 38(9):1384–1392; Nakamura 2006, ICGI LNCS 4201, pp. 72–83. [Source](https://doi.org/10.1016/j.patcog.2005.01.004); [Source](https://doi.org/10.1007/11872436_7)
- **Method.** Incremental learning from positive *and negative* strings, with rule generation by bottom-up (CYK) parsing. The 2006 version uses "bridging" of the missing part of a derivation. It runs an iterative-deepening search for a minimum rule set, or a faster serial search. [Source](https://www.jstage.jst.go.jp/article/tjsai/21/4/21_4_371/_article/-char/en)
- **Limitations.** Rules are binary (revised CNF). Search time is exponential in rule-set size. Formal languages only, for example #a = #b in 4.0 s with serial search. [Source](https://www.jstage.jst.go.jp/article/tjsai/21/4/21_4_371/_article/-char/en)

**EMILE (Adriaans, Vervoort)**
- **Method.** A context/expression substitution matrix over all splits of each sentence. "2-dimensional clustering … searches for maximum-sized blocks"; support thresholds handle imperfect samples. Rules come from substituting types for characteristic expressions. Batch, with n-ary expressions. [Source](http://web.archive.org/web/20060829210620/http://staff.science.uva.nl/~pietera/Emile/sofsem2000.pdf)
- **Learnability.** Shallow, separable context-free languages are learnable under simple distributions ("PACS"). [Source](https://arxiv.org/abs/cs/0205025)
- **Results.**
  - Grammars are "oversized (3000 to 4000 rules)" but "capture the recursion".
  - "EMILE almost never finds the simple basic 'sentence is noun-phrase + verb-phrase' rule."
  - ATIS: recall 16.81, precision 51.59, F 25.35.

  [Source](http://web.archive.org/web/20060829210620/http://staff.science.uva.nl/~pietera/Emile/sofsem2000.pdf); [Source](https://arxiv.org/abs/cs/0205025)

**ABL, alignment-based learning (van Zaanen 2000; thesis 2002)**
- **Alignment learning.** Aligns sentence pairs with edit distance; the unequal parts become hypothesized substitutable constituents of the same type. Constituents are variable length. [Source](https://aclanthology.org/C00-2139/); [Source](https://arxiv.org/abs/cs/0205025)
- **Selection learning.** Chooses among overlapping hypotheses with incr ("assume that the first constituent learned is the correct one"), leaf or branch probabilities, using Viterbi selection. [Source](https://aclanthology.org/C00-2139/); [Source](https://arxiv.org/abs/cs/0205025)

- **Results.**
  - Non-crossing-bracket precision on ATIS 85.31 (right-branching 82.70) and on OVIS 89.25.
  - Under unlabeled EVALB in the thesis, ATIS F 35.54 (random 31.13).
  - On WSJ §23, F 19.26 (recall 12.46, precision 42.56), *below* random at 23.27.

  [Source](https://aclanthology.org/C00-2139/); [Source](https://arxiv.org/abs/cs/0205025)

**ADIOS (Solan, Horn, Ruppin & Edelman 2005)**
- **Citation.** *PNAS* 102(33):11629–11634. Sentences are paths in a "directed pseudograph". [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953)
- **MEX proposal and significance test.**
  - P_R(e1; e2) = paths e1→e2 / paths entering e1, and likewise leftward.
  - A pattern boundary is where the decrease ratio D_R = P_R(e1; e5)/P_R(e1; e4) < η.
  - Significance requires P-values below α ≪ 1.
  - The "most significant pattern is added to the lexicon as a new unit" and the graph is rewired, either context-free (Mode A) or context-sensitive (Mode B, only where significant).
  - Equivalence classes come from slots in a window of width L; an existing class is reused when overlap exceeds ω = 0.65.

  [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953)
- **Control and order sensitivity.** Greedy: "the best available pattern in each iteration is immediately and irreversibly rewired". The syntax learned therefore "depends on the order of sentences", which the authors mitigate by training "multiple learners on different order-permuted versions of the corpus". Empirical cost is linear in corpus size. [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953)
- **Results.**
  - A 29-terminal CFG from 2,000 sentences: "100% precision and 99% recall".
  - TA1 at η = 0.6, α = 0.01, L = 5: precision and recall reach 90% at 800 sentences.
  - ATIS-2: recall 40%, human-judged precision ≈70%, perplexity 11.5. By comparison the hand-built ATIS-CFG has 45% recall but generates more than 99% ungrammatical sentences.

  [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953)
- **Limitation.** "Infinite recursion is not implemented in the current version." [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953)
- **ATIS-CFG cohort results** (thesis, 150 learners, 120k sentences): L = 5 gives recall 1.0 / precision 0.64; L = 7 gives 0.23 / 0.97. [Source](https://www.tau.ac.il/~horn/publications/ZachSolanThesis.pdf)

**U-MILA (Kolodny, Lotem & Edelman 2015)**
- **Citation.** *Cognitive Science* 39(2):227–267, DOI 10.1111/cogs.12140. [Source](http://web.archive.org/web/20240508093013/https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/KolodnyEtAl2015_LangAcqProcessLevel.pdf)
- **Incremental, memory-limited learning.**
  - It learns on every token.
  - A short-term memory queue (the "phonological loop", typically 50–300 tokens) decays exponentially.
  - All graph weights decay with a long half-life, so errors "decay and eventually become negligible".

  [Source](http://web.archive.org/web/20240508093013/https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/KolodnyEtAl2015_LangAcqProcessLevel.pdf)
- **Proposals.**
  - Recurring sequences found in the queue (top-down segmentation).
  - Binary supernodes A+B "if sanctioned by Barlow's (1990) principle of suspicious coincidence, subject to a prior". These merge recursively into longer units.
  - Slot collocations (e.g., "the ___ boy").
  - Substitutability from three similarity measures.

  [Source](http://web.archive.org/web/20240508093013/https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/KolodnyEtAl2015_LangAcqProcessLevel.pdf)
- **Results.**
  - Trained on 15,000 CHILDES (Suppes) utterances: perplexity 40.07, vs SRILM trigram 22–24.
  - Human acceptability of generated sentences: 5.87/7, vs 5.41 for a trigram matched on perplexity and 6.59 for the corpus.
  - Replicates many statistical-learning experiments (word segmentation, non-adjacent dependencies) and gets auxiliary fronting right on 89/95 pairs.

  [Source](http://web.archive.org/web/20240508093013/https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/KolodnyEtAl2015_LangAcqProcessLevel.pdf)

**Tu & Honavar (2008): PCFG-BCL; Tu, Pavlovskaia & Zhu (2013): And-Or grammars**
- **PCFG-BCL** (ICGI 2008, LNAI 5278, pp. 224–237).
  - Iteratively biclusters the bigram table. Each bicluster adds an AND-OR group N → A B, with A → rows and B → columns, chosen to maximize posterior gain under P(G) = 2^−DL(G).
  - An attach step adds O → N when the gain exceeds a threshold. Biclusters must be multiplicatively coherent.

  [Source](https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/icgi08.pdf)
- **PCFG-BCL results.** Generate-and-parse F (precision and recall from 200 samples each, 50 runs), PCFG-BCL vs EMILE vs ADIOS:
  - Langley2: 99 vs 55 vs 75.
  - TA1 (2,000 sentences): 97 vs 64 vs 62.

  [Source](https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/icgi08.pdf)
- **And-Or grammars** (NIPS 2013).
  - Objective: posterior ∝ e^−α‖G‖ × Viterbi likelihood. Starts from one And-rule per sample.
  - Iteratively adds n-ary And-Or fragments, scored by posterior gain = likelihood gain × prior gain. Greedy or beam search; stops when no fragment helps.
  - Evaluated on event grammars and images, not text.

  [Source](https://proceedings.neurips.cc/paper_files/paper/2013/hash/24681928425f5a9133504de568f5f6df-Abstract.html); [Source](https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/nips13.pdf)

**Clark (2001): distributional clustering with an MI filter**
- **Data and candidates.** 12M words of BNC with CLAWS tags. Tag sequences occurring more than 5,000 times (753 of them) are clustered by context distributions (k-means, L1) into 100 clusters. [Source](https://aclanthology.org/W01-0713/)
- **MI filter.** "With real constituents, there is high mutual information between the symbol occurring before the putative constituent and the symbol after." The threshold is the expected MI at that distance, about 0.05 for a 2-symbol sequence. MI is measured on whole clusters, pooling counts, because the plug-in MI estimator overestimates on sparse data. The filter eliminated 55 of 100 clusters. [Source](https://aclanthology.org/W01-0713/)
- **Learning loop.**
  - Greedily pick the cluster with "the best immediate reduction in description length".
  - Add rules, adding to an existing nonterminal when the cluster contains it; this yields recursion.
  - Partially parse the corpus along the shortest-description path, then repeat (40 iterations).
  - The MDL gain of chunking a pair P R "under reasonable approximations" equals the pointwise mutual information between P and R.

  [Source](https://aclanthology.org/W01-0713/)
- **Results and limitation.** ATIS F 42.0 (UR 34.6 / UP 53.4), vs EMILE 25.4, ABL 39.2 and right-branching 42.9 (EVALB, per Klein & Manning). "The greediness of the algorithm … makes the algorithm very sensitive to the order in which the rules are acquired." [Source](https://aclanthology.org/W01-0713/); [Source](https://aclanthology.org/P02-1017/)

**Goldsmith (2001): Linguistica (the proposal scheme the user flagged)**
- **Citation.** *Computational Linguistics* 27(2):153–198 (J01-2001). "We develop a set of heuristics that rapidly develop a probabilistic morphological grammar, and use MDL as our primary tool to determine whether the modifications proposed by the heuristics will be adopted or not." [Source](https://aclanthology.org/J01-2001.pdf)
- **Separation of roles.** "MDL is a framework for evaluating proposed analyses, but it does not provide a set of heuristics … essential for obtaining candidate analyses." [Source](https://aclanthology.org/J01-2001/)
- **Bootstrap heuristics.**
  - Take-all-splits: every cut of every word is scored by stem and suffix frequencies, Boltzmann-normalized, and iterated.
  - Weighted mutual information over word-final n-grams of length 2–6: the top 100 become candidate suffixes.
  - Correction: Harris's successor frequency is *not* one of the 2001 bootstrap heuristics; it appears only in prior work and a footnote.

  [Source](https://aclanthology.org/J01-2001/)
- **MDL-checked refinements.**
  - Keep "regular" signatures (≥2 stems and ≥2 suffixes).
  - Split composite suffixes (ings → ing + s): 31 of 64 tested were split.
  - Shift stem-final letters.
  - Triage, i.e. delete a signature when that lowers description length.
  - The description length is a two-part code: morphology length (lists of stems, suffixes and signatures) plus corpus length (−Σ log probabilities).

  [Source](https://aclanthology.org/J01-2001/)
- **Results and a caution.** English: precision 85.9%, recall 90.4% on 1,000 hand-checked words. Triage wrongly removed 21.9% of correct changes, so an extra non-MDL threshold had to be added. [Source](https://aclanthology.org/J01-2001/)

**Sequitur (Nevill-Manning & Witten 1997)**
- **Citation.** *JAIR* 7:67–82. [Source](https://arxiv.org/abs/cs/9709102)
- **Constraints.**
  - "No pair of adjacent symbols appears more than once in the grammar" (digram uniqueness), which creates binary rules.
  - "Every rule is used more than once" (rule utility). A rule used once is inlined, which is how rules become n-ary. [Source](https://arxiv.org/abs/cs/9709102)
- **Performance.** "Operating incrementally" in linear time and space, at about 50,000 symbols/sec. [Source](https://arxiv.org/abs/cs/9709102)
- **Evaluation.** No grammar-induction accuracy evaluation; the accept/reject paradigm of Langley, Stolcke and Cook "does not apply". [Source](https://arxiv.org/abs/cs/9709102)

### Inferences
- **TRELLIS v2 is, structurally, "incremental GRIDS with Cobweb categories".** The operator correspondence is direct:
  - GRIDS/SNPR *create* = adding a composite (content) instance or concept;
  - *merge* = Cobweb merge of context categories;
  - Wolff's *rebuilding* and Stolcke's *unchunk* = Cobweb split (undoing an overgeneral merge or chunk).

  What the lineage adds over Cobweb's category utility is an explicit grammar-plus-data code to accept or reject those moves. The authors' own suggested next steps — co-occurrence pre-filtering and incremental processing — are exactly what Cobweb supplies.
- **Proposal is cheap; acceptance is the hard part.**
  - e-GRIDS showed that, under a GRIDS code, ranking create proposals by bigram frequency is *exact*: the M best successors come from the M most frequent bigrams.
  - Clark showed the description-length gain of chunking a pair equals its pointwise MI.

  So v1's frequency-based `MERGE_POLICY` is a sound *proposer*. The missing piece is an acceptance test with (i) a grammar-cost term and (ii) a constituent/distituent filter, such as Clark's left–right context MI or ADIOS's significance test.
- **Graduation should be a significance test, not a raw count.** ADIOS (decrease ratio η with significance α), U-MILA (suspicious coincidence plus decay) and Clark (MI above its expected value, pooled per cluster to fight sparsity) all show this. That matches the user's "frontier → confirm → learn" idea, and pooling counts at the context-class level is something Cobweb's hierarchy does natively.
- **Order sensitivity is endemic to greedy symbolic learners, and the lineage already found the fixes:**
  - ADIOS trains cohorts of learners on permuted orders;
  - Clark explicitly notes order sensitivity;
  - Stolcke shows length-ordered presentation cuts merges by about 6×.

  For Cobweb-based TRELLIS: use a length curriculum, run several orderings, and combine them (Section 3: tree averaging).
- **Calibrate PTB expectations with the lineage's history.** On real treebanks this family under-performed right-branching on bracketing (ABL WSJ F 19.3 vs random 23.3; EMILE/ABL/CDC < right-branching on ATIS), while doing well on generation precision (ADIOS ≈70% human-judged; U-MILA 5.87/7). TRELLIS should expect a similar asymmetry unless it adds the CCM-style distituent modelling and soft evidence of Sections 2–3.
- **Pure MDL acceptance is miscalibrated in practice.** Goldsmith's triage wrongly deleted 21.9% of correct changes, and eg-GRIDS over-generalizes on small data. TRELLIS's v1 finding that MDL condensed categories but hurt generation is part of this pattern. Pair MDL with a commission check (see Implications for TRELLIS v2).

### Gaps
- The text of Langley's 1994 manuscript and the contents of *Elements of Machine Learning* §9.3 were not available; only secondary descriptions and the table of contents were found.
- The full texts of Synapse 2005/2006 were blocked; its description comes from a 2006 companion paper (TJSAI).
- U-MILA's exact suspicious-coincidence formula and parameter values are in supplementary appendices that were not retrieved.
- The DBLP record for e-GRIDS (volume/pages) was confirmed only through a mirror snippet, not the DBLP page itself.

## 2. Probabilistic and Bayesian induction: why inside-outside fails, CCM/DMV, Bayesian PCFGs, adaptor grammars, TSGs, U-DOP and fragment grammars

### Takeaway
Plain inside-outside EM over a PCFG is the wrong objective and the wrong search. Its failures are well documented:
- It finds a different local optimum on every start.
- It prefers "memorizing" or trivial grammars.
- Likelihood decouples from bracketing quality.
- It needs about 3× surplus nonterminals.

The successful probabilistic systems change three things:
- **The representation:** CCM's yield+context factorization of every span; heads (DMV); whole fragments (DOP, TSG, adaptor and fragment grammars).
- **The prior:** sparse Dirichlet or Pitman-Yor priors that reward reuse of few units.
- **The search:** harmonic initialization, curriculum, sampling over whole derivations.

The Pitman-Yor caching family (adaptor grammars, TSGs, fragment grammars) and Bod's shortest derivation are the most direct formalizations of "store a chunk when it is reused, and use as few chunks as possible". Online adaptor grammars already implement a "candidate pool, then promote and prune" loop.

### Cited Findings

**Inside-outside EM and its failure modes**
- **Lari & Young 1990** (*Computer Speech & Language* 4(1):35–56).
  - Batch EM for CNF SCFGs; cost is O(N³) in nonterminals and O(T³) in sentence length.
  - On 3-symbol palindromes, the minimal 7 nonterminals failed. 12 were "slightly better" and 18 learned the language. "Typically a three-fold excess of non-terminals is needed."
  - They added a grammar-minimization step that reallocates nonterminals taken over by "greedy symbols".
  - [Source](http://web.archive.org/web/20190112044228/https://courses.cs.washington.edu/courses/cse599d1/16sp/lari-young-90.pdf)
- **Carroll & Charniak 1992** (AAAI-92 workshop on statistically-based NLP; Brown TR CS-92-16).
  - Inside-outside over dependency-style PCFG rules, with sentences processed in groups of increasing length (an early curriculum).
  - "Of the 300 starting points tried, we found 300 different local minima"; none was the correct grammar.
  - A one-rule-per-sentence "memorizing grammar" maximizes likelihood.
  - [Source](https://cdn.aaai.org/Workshops/1992/WS-92-01/WS92-01-001.pdf)
- **Pereira & Schabes 1992** (ACL, pp. 128–135), ATIS POS sequences: 700 training and 70 test sentences; 15 nonterminals and 48 tags (4,095 CNF rules).
  - Cross-entropy is nearly identical for bracketed and raw training (≈2.97 vs 2.95).
  - Bracketing accuracy is 90.36% with partial brackets vs 37.35% raw. The measure is the share of Viterbi brackets compatible with the treebank.
  - Raw likelihood "underdetermines" structure; for example, pronoun+verb gets grouped.
  - [Source](https://aclanthology.org/P92-1017/)
- **de Marcken 1995** (Third Workshop on Very Large Corpora).
  - Representation argument: an entropy-minimizing PCFG over adjacent words groups high-mutual-information pairs. For V P N with I(V,P) > I(P,N), the correct [V [P N]] is the higher-entropy analysis. Greedy MI grouping (Magerman & Marcus; Olivier's "edby"; Stolcke 1994) therefore "will consequently fail to derive linguistically-plausible phrase structure in many situations".
  - Search argument: on a 3-word toy language, 12 of 20 runs converged to a suboptimal grammar, because EM is "so attracted to grammars whose terminals concentrate probability on small numbers of rules that it is incapable of performing real search".
  - Proposed remedy: head-driven, dependency-like representations.
  - [Source](https://aclanthology.org/W95-0102/)
- **de Marcken 1996** (MIT PhD thesis, arXiv cmp-lg/9611002).
  - An MDL learner with a compositional lexicon: entries are built from other entries. Each iteration adds pairwise compositions predicted to shorten the description and deletes entries whose cost exceeds their savings.
  - Brown corpus: 2.09 bits/char. Unspaced-text segmentation: word recall 90.5%, crossing brackets 1.7%.
  - Failure mode: it learns supra-word chunks ("forthepurposeof").
  - [Source](https://arxiv.org/abs/cmp-lg/9611002)

**CCM and DMV: the yield+context representation**
- **CCM (Klein & Manning, ACL 2002, pp. 128–135).**
  - P(S,B) = P(B) ∏ P(α_ij | B_ij) P(x_ij | B_ij) over all spans. Each span is a constituent or a "distituent", and generates its yield α and its context x. P(B) is uniform over binary bracketings. Training is batch EM; train = test on WSJ10.
  - WSJ-10 F1 (2002 metric): CCM 71.1; RBRANCH 60; DEP-PCFG 48.2; supervised PCFG 82.1; upper bound 87. [Source](https://aclanthology.org/P02-1017/)
  - On ATIS (trained on WSJ, scored with EVALB), CCM gets F1 51.2 vs RBRANCH 42.9. The earlier unsupervised systems did not beat right-branching: EMILE 25.4, ABL 39.2, Clark's CDC-40 (12M words of training) 42.0. [Source](https://aclanthology.org/P02-1017/)
  - Klein & Manning say Clark (2001) "must resort to a filtering heuristic to separate constituent and distituent clusters", because distituents outnumber constituents. Context clusters separate categories (labeling) far more easily than they separate constituents from non-constituents (bracketing). [Source](https://aclanthology.org/P02-1017/)
- **DMV and DMV+CCM (Klein & Manning, ACL 2004, pp. 478–485)** on WSJ10, 7,422 sentences, micro-averaged with the full-span bracket counted (UP / UR / UF1, then directed / undirected dependency accuracy):

  | System | UP | UR | UF1 | Directed | Undirected |
  |---|---|---|---|---|---|
  | RBRANCH | 55.1 | 70.0 | 61.7 | | |
  | DMV | 46.6 | 59.2 | 52.1 | 43.2 | 62.7 |
  | CCM | 64.2 | 81.6 | 71.9 | | |
  | DMV+CCM (POS) | 69.3 | 88.0 | 77.6 | 47.5 | 64.5 |
  | DMV+CCM (induced classes) | 65.2 | 82.8 | 72.9 | | |
  | Upper bound | 78.8 | 100 | 88.1 | | |

  - NEGRA10 DMV+CCM 63.9; CTB10 43.3.
  - The models succeed partly because they "minimize the amount of hidden structure".
  - [Source](https://aclanthology.org/P04-1061/)
- **Three CCM numbers from three metrics.** CCM's WSJ10 score is 71.1 (2002: whole-sentence bracket excluded, macro-averaged), 71.9 (2004: micro-averaged, whole sentence included) and 72.4 (thesis). The thesis notes CCM "drops all the way to 53.4% on sentences of length up to 15". [Source](https://people.eecs.berkeley.edu/~klein/papers/klein_thesis.pdf)
- **CCM depends on gold POS tags.** Li et al. (ACL 2020) report that run on words instead of gold POS, CCM drops from 70.14 to 57.29 (EVALB F1, ≤10 words, held-out PTB test). [Source](https://aclanthology.org/2020.acl-main.300.pdf)

**Bayesian PCFG inference**
- **Johnson, Griffiths & Goldwater 2007** (NAACL-HLT, pp. 139–146). Gibbs and collapsed Metropolis-Hastings samplers for PCFGs with Dirichlet priors.
  - On Sesotho verb morphology, inside-outside converges to the trivial "every word is one morpheme" grammar (F = 0). With sparse α = 10⁻⁵, the sampler reaches morpheme F 0.75 and exact segmentation 0.54.
  - On English syntax the Bayesian methods "obtain results very similar to those produced using IO".
  - [Source](https://aclanthology.org/N07-1018/)
- **Kurihara & Sato** (2004 IJCNLP workshop; 2006 ICGI, LNCS 4201, pp. 84–96).
  - Variational-Bayes free energy is used as a model-selection score in a greedy structure search: split high-count nonterminals, merge similar ones, delete low-count rules, and accept a change when the free energy drops.
  - Their WSJ results use bracketed training and crossing-consistency accuracy, so they are **not comparable** with raw-text unlabeled F1.
  - [Source](https://web.archive.org/web/2007/http://sato-www.cs.titech.ac.jp/reference/ICGI2006.pdf)
- **HDP-PCFG** (Liang, Petrov, Jordan & Klein, EMNLP-CoNLL 2007). An infinite-symbol PCFG with structured mean-field inference.
  - On a synthetic grammar it used only the needed subsymbols: "only 20 rules are effective, which corresponds exactly to the true grammar". A standard PCFG used 8,320 rules.
  - Treebank use was supervised state-splitting, not raw-text induction.
  - [Source](https://aclanthology.org/D07-1072/)

**Pitman-Yor caching: adaptor grammars, TSGs, fragment grammars**
- **Adaptor grammars** (Johnson, Griffiths & Goldwater; NIPS 19, pp. 641–648).
  - A Pitman-Yor adaptor caches whole subtrees of "adapted" nonterminals. The next draw reuses cached tree k with probability (n_k − a)/(n + b), or makes a new tree from the base PCFG with probability (m·a + b)/(n + b): "rich get richer". a = 1 gives an ordinary PCFG.
  - Inference is sentence-by-sentence Metropolis-Hastings, with a PCFG proposal that has one rule per cached yield.
  - [Source](https://proceedings.neurips.cc/paper/2006/hash/62f91ce9b820a491ee78c108636db089-Abstract.html)
- **Johnson & Goldwater 2009** (NAACL-HLT):
  - Table-label resampling and hyperparameter sampling lift collocation-syllable word segmentation to 0.87 token F (0.89 with incremental initialization).
  - Incremental initialization scores higher but reaches *lower* posterior probability: it greedily locks onto short common substrings.
  - [Source](https://aclanthology.org/N09-1036/)
- **Cohen, Blei & Smith 2010** (NAACL-HLT): variational inference for adaptor grammars.
  - A DMV encoded as an adaptor grammar reaches 50.2 attachment accuracy (MBR) on §23 at ≤10 words, vs 46.1 without adaptation.
  - [Source](https://aclanthology.org/N10-1081/)
- **Online adaptor grammars** (Zhai, Boyd-Graber & Cohen, TACL 2:465–476, 2014): the closest published analogue of a "frontier of candidate chunks".
  - Minibatches; 10 sampled trees per sentence.
  - Decayed sufficient statistics: f̃ ← (1−ε)·f̃ + ε·(scaled minibatch counts), with ε = (τ + l)^−κ.
  - After each minibatch, "potentially adapted" productions drawn from the base distribution are *added* to a truncated set of adapted rules.
  - Every u minibatches, rules are ranked by usefulness × a length bonus, Λ = f̃·log(ε·|s| + 1), and all but the top K are pruned.
  - Results match or beat batch MCMC (e.g., Chinese segmentation 72.98 vs 71.75 F on ctb7).
  - [Source](https://aclanthology.org/Q14-1036/)
- **Tree-substitution grammars.**
  - Cohn, Goldwater & Blunsom (NAACL 2009) and Cohn, Blunsom & Goldwater (JMLR 11:3053–3096, 2010) put Dirichlet-process / Pitman-Yor priors over elementary trees, sampled with Gibbs or blocked Metropolis-Hastings.
  - Supervised result: §23 F1 84.0 vs 70.7 for a maximum-likelihood PCFG.
  - Unsupervised TSG-DMV: §23 head attachment 65.9 (≤10) / 53.1 (all). A Baby-Steps-like length curriculum gives 66.4 / 53.4.
  - [Source](https://aclanthology.org/N09-1062/); [Source](https://jmlr.org/papers/v11/cohn10b.html)
  - Blunsom & Cohn (EMNLP 2010), lexicalized TSG-DMV: 67.7 (≤10) / 55.7 (all). Training log-likelihood correlates with accuracy at only R² = 0.2. [Source](https://aclanthology.org/D10-1117/)
- **Fragment grammars** (O'Donnell, Tenenbaum & Goodman 2009, MIT-CSAIL-TR-2009-013; O'Donnell 2015, *Productivity and Reuse in Language*, MIT Press).
  - Pitman-Yor memoization over *partial* trees: fragments with variables at their leaves, grown by a per-child "continue or leave as a variable" coin.
  - Adaptor grammars are the special case that stores only complete subtrees.
  - Storing a bigger fragment reduces choices per sentence but lowers reuse. The model ends up storing mid-sized fragments.
  - [Source](https://dspace.mit.edu/handle/1721.1/44963); [Source](http://web.archive.org/web/20240621053459/https://mitpress.mit.edu/9780262028844/productivity-and-reuse-in-language/)
  - O'Donnell et al. (CogSci 2011), on productivity of 338 English suffixes: FG correlates with Baayen's P at 0.907, vs full-parsing (PCFG) −0.0003, full-listing (adaptor grammars) 0.692 and DOP1 0.346. [Source](http://web.archive.org/web/2022/http://people.linguistics.mcgill.ca/~timothy.odonnell/papers/OdonnellCogsci2011.pdf)

**DOP family: U-DOP and the shortest derivation**
- **Bod 2000** (COLING), the shortest derivation: choose the parse built from "the fewest corpus-subtrees", breaking ties by subtree frequency rank.
  - Exact match: ATIS 85.6 vs 84.1 for probabilistic DOP; OVIS 92.2 vs 88.8.
  - WSJ §23 (≤40 words): slightly below probabilistic DOP (87.2 / 86.9 vs 89.5 / 89.3 labeled precision / recall).
  - [Source](https://aclanthology.org/C00-1011/)
- **U-DOP and UML-DOP** (Bod 2006, CoNLL and COLING-ACL). Assign all binary trees to each POS string, extract all subtrees (Goodman reduction), and take the most probable parse.
  - WSJ10 / NEGRA10 / CTB10: U-DOP 78.5 / 65.4 / 46.6; UML-DOP (EM-trained) 82.9 / 67.0 / 47.2.
  - Held-out 90/10 WSJ10: UML-DOP 82.5 vs binarized supervised PCFG 81.5. WSJ40: 66.4 vs 64.6.
  - U-DOP needs no distituent class: none of the top-10 learned constituents is a distituent, although IN DT is a frequent substring. Its most probable parse "has a tendency to be constructed by the shortest derivation".
  - [Source](https://aclanthology.org/W06-2912/); [Source](https://aclanthology.org/P06-1109/)
- **U-DOP\*** (Bod 2007, ACL, pp. 400–407). Shortest derivations on held-out data define a consistent tree-substitution grammar.
  - WSJ10 77.9 (UML-DOP 79.4 on 90/10 splits).
  - Full WSJ §23 (≤100 words, binarized gold, POS strings): 62.2 from §2–21 alone, 70.7 after adding about 4M unannotated NANC sentences. The binarized treebank PCFG scores 63.5.
  - [Source](https://aclanthology.org/P07-1051/)
- **Bod 2009** (*Cognitive Science* 33(5):752–793). The "most probable shortest derivation" formalizes a least-effort, parsimony principle.
  - WSJ10 82.7 F1 with all subtrees; 51.7 from raw words; 76.4 with Clark-induced tags.
  - On CHILDES Eve, adult→child F1 is 81.8. Depth-1 fragments (i.e., a PCFG) give only 49.5, vs 88.7 with all subtrees, and the useful fragment depth grows with age.
  - [Source](http://web.archive.org/web/20180218230842/http://onlinelibrary.wiley.com:80/doi/10.1111/j.1551-6709.2009.01031.x/full)

**Raw-text chunk cascades (Ponvert et al. 2011)**
- Ponvert, Baldridge & Erk (ACL-HLT 2011, pp. 1077–1086), method:
  - Each level is a chunker (HMM or probabilistic right-linear grammar, PRLG) with BIO tags, trained by EM.
  - Each multiword chunk is collapsed to a pseudoword (its most frequent word), then a new chunker is trained on the collapsed text. Iteration continues until no new chunks appear, taking 5–7 levels.
  - [Source](https://aclanthology.org/P11-1108/)
- Results (WSJ §23 held out, all lengths, P / R / F):
  - PRLG cascade 60.0 / 49.4 / 54.2, vs CCL 51.7.
  - On ≤10-word sentences, PRLG gets 70.5 F, CCL 72.1 and CCM (gold POS) 70.7.
  - [Source](https://aclanthology.org/P11-1108/)
- Weaknesses:
  - Phrasal punctuation is used as a boundary cue; without it, PRLG loses 10.1 precision and 7.2 recall on WSJ.
  - Pseudowords strip away cues that later levels need.
  - PP-attachment errors are common.
  - [Source](https://aclanthology.org/P11-1108/)

### Inferences
- **What doomed raw inside-outside is not "considering all parses"; it is the parameterization and the objective.** Every success either:
  - changes what a span is scored by (CCM yield+context, heads, fragments), or
  - adds a reuse-favouring prior (sparse Dirichlet, Pitman-Yor), or
  - changes the search (harmonic initialization, curriculum, sampling whole derivations, restricted parse pools).

  For TRELLIS, inside-outside is a sound *E-step* only if the scores come from the dual content/context representation. Optimizing likelihood of an unstructured PCFG would repeat the 1990s results.
- **TRELLIS's content/context duality is a hierarchical, incremental CCM.** CCM's main lessons carry over:
  - distituents need an explicit model, or a filter (Clark's MI criterion);
  - performance hinges on good word classes (gold POS 71.9 vs words 57.29);
  - constituency is harder than categorization.
- **The DOP / U-DOP results show that storing many large fragments plus a shortest-derivation preference works well on short sentences** (WSJ10 ≈ 78–83 F1). They also show what drives cognitive fits: fragment depth beyond one rule (Eve: 49.5 at depth 1 vs 88.7 with all subtrees). This argues for TRELLIS chunks being multi-level templates, not only binary rules.
- **Online adaptor grammars already implement the user's frontier idea** — propose from the base distribution, accumulate decayed counts, promote to the cache, rank by usefulness × length and prune. They can be transplanted onto Cobweb: the cache becomes stored chunk concepts, and the base distribution becomes compositional generation through the content hierarchy.

### Gaps
- No published system applies Pitman-Yor caching or CCM-style span scoring to PTB constituency from raw words incrementally. The closest are CCL (incremental, heuristic) and online adaptor grammars (applied to segmentation, not constituency).
- The internals of O'Donnell's 2015 book could not be verified; the CogSci 2011 paper was used instead.
- Bod's binarization of gold trees is described ambiguously across papers. The subagent's reading is that UP ≠ UR is hard to reconcile with fully binarized gold trees, so Bod's WSJ10 numbers may not be strictly comparable with Klein & Manning's.

## 3. Neural latent-tree and PCFG induction, and the current state of the art on PTB (2018–2026)

### Takeaway
On the standard words-only protocol (PTB WSJ sections 02–21 for training, 23 for testing, punctuation removed, unlabeled sentence-level F1 averaged over sentences, trivial spans discarded), the best single PTB-only grammar inducers sit at about 64–65 S-F1: Rank-PCFG 64.1, SN-PCFG 65.1±2.1, Hol-PCFG 64.6±0.4. Methods that take structural bias from other unsupervised parsers reach about 70: parse-focusing 69.6±0.6 and CRNP+PF 70.2±0.5. Methods that draw on pretrained LMs or LLMs reach 67–73: SemInfo PCFGs ≈67, tree-averaging ensembles 70.4 to 72.8. Reference points are right-branching 39.5 and the binarized-gold ceiling 84.3. Three things drive the gains: (a) many grammar symbols, (b) parameter sharing or smoothing across rules, and (c) aggregating several noisy structure sources. None of the gains comes from a better search procedure alone.

### Cited Findings
**The modern protocol (as established by Kim, Dyer & Rush 2019)**
- Compound PCFG paper (Kim, Dyer & Rush, ACL 2019) protocol:
  - standard PTB splits: 2–21 train, 22 validation, 23 test;
  - "discard punctuation, lowercase all tokens, and take the top 10K most frequent words as the vocabulary";
  - evaluated by unlabeled sentence-level F1 with trivial spans (width-one and sentence-level spans) discarded;
  - "Corpus-level F1 calculates precision/recall at the corpus level to obtain F1, while sentence-level F1 calculates F1 for each sentence and averages across the corpus";
  - Mean/Max over 4 random seeds.
  [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- Kim et al. 2019 Table 1 (PTB test, sentence-level F1, mean / max):

  | System | Mean | Max |
  |---|---|---|
  | Left branching | 8.7 | |
  | Right branching | 39.5 | |
  | Random trees | 19.2 | 19.5 |
  | PRPN (tuned) | 47.3 | 47.9 |
  | ON (tuned) | 48.1 | 50.0 |
  | Scalar PCFG | < 35.0 | |
  | Neural PCFG | 50.8 | 52.6 |
  | Compound PCFG | 55.2 | 60.1 |
  | Oracle (binarized gold) | 84.3 | |
  | URNNG† | | 45.4 |
  | DIORA† | | 58.9 |

  † trained with punctuation and "not strictly comparable".
  - Chinese CTB: Compound PCFG 36.0 / 39.8; right branching 20.0; oracle 81.1.
  [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- Same paper, Table 6, broken down by test-sentence length (sentence-level F1):

  | Length subset | Right branching | Compound PCFG | Oracle |
  |---|---|---|---|
  | WSJ-10 (≤10 words) | 58.5 | 70.5 | 82.1 |
  | WSJ-20 | 49.8 | 63.4 | |
  | WSJ-40 | 41.6 | 56.6 | |
  | full | 39.5 | 55.2 | 84.3 |

  - Corpus-level F1 on the full test set: right branching 36.1, Compound PCFG 52.4, oracle 84.7. Corpus-level is systematically about 3 points lower than sentence-level for the same trees.
  [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- The plain count-parameterized ("scalar") PCFG trained by EM scored < 35.0 F1 on PTB, below right-branching, "despite a thorough hyperparameter search". The authors note that "Training perplexity was much higher than in the neural case, indicating significant optimization issues". They "did not experiment with online EM (Liang and Klein, 2009)". [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- Compound PCFG setup:
  - 30 nonterminals and 60 preterminals.
  - Length curriculum: "we train only on sentences of length up to 30 in the first epoch, and increase this length limit by 1 each epoch".
  - Early stopping on validation perplexity.
  [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- Run-to-run consistency ("Self F1" between seeds): PRPN 82.3, ON 71.3, Neural PCFG 65.2, Compound PCFG 66.8. [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- Label recall for Compound PCFG (fraction of gold constituents of each label that were predicted): NP 74.7%, VP 41.7%, PP 68.8%, SBAR 56.1%, ADJP 40.4%, ADVP 52.5%. [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- Preterminal induction: Compound PCFG preterminals give 68.0 many-to-one POS accuracy. [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- Training RNNGs on induced trees and fine-tuning with URNNG lifts F1: from Compound PCFG trees 60.1 → 66.9. The authors call these numbers "optimistic", because they chose the best-performing original runs by validation F1 and used them to parse the training set. [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)
- A URNNG trained from scratch "fails to outperform a right-branching baseline on this version of PTB where punctuation is removed". [Kim, Dyer & Rush 2019](https://arxiv.org/abs/1906.10225)

**Earlier neural latent-tree models (PRPN, ON-LSTM)**
- ON-LSTM (Shen et al., ICLR 2019) Table 2:

  | System | WSJ10 (7,422 sentences) | WSJ test (2,416 sentences) |
  |---|---|---|
  | ON-LSTM 2nd layer | 65.1 (σ 1.7), max 66.8 | 47.7 (1.5), max 49.4 |
  | PRPN-LM (WSJ-trained) | 70.5 (0.4) | 37.4 (0.3) |
  | Right branching | 56.6 | 39.8 |
  | Random | 31.7 | 18.4 |
  | Balanced | 43.4 | 24.5 |

  - The paper also lists POS-tag-based WSJ10 systems: CCM 71.9, DMV+CCM 77.6, UML-DOP 82.9, noting they "are not strictly comparable".
  [Shen et al. 2019](https://arxiv.org/abs/1810.09536)
- PRPN (Shen et al., ICLR 2018) WSJ10 UF1 table:

  | System | WSJ10 UF1 |
  |---|---|
  | LBRANCH | 28.7 |
  | RANDOM | 34.7 |
  | DEP-PCFG (Carroll & Charniak 1992) | 48.2 |
  | RBRANCH | 61.7 |
  | CCM | 71.9 |
  | DMV+CCM | 77.6 |
  | UML-DOP | 82.9 |
  | PRPN | 70.02 |
  | upper bound | 88.1 |

  - Right-branching on WSJ10 is 61.7 here but 56.6 in the ON-LSTM table, which illustrates how much the evaluation conventions differ.
  [Shen et al. 2018](https://arxiv.org/abs/1711.02013)

**Scaling up the number of symbols (TN-PCFG, Rank PCFG, SimplePCFG)**
- TN-PCFG (Yang, Zhao & Tu, NAACL 2021): a tensor-decomposed PCFG with 500 preterminals and 250 nonterminals.
  - PTB test S-F1: 57.7±4.2 (max 61.4), against 51.4±4.0 with p = 60 preterminals.
  - Reimplemented baselines with MBR decoding: N-PCFG 52.3±2.3, C-PCFG 56.3±2.1.
  - Other systems in the same table: NL-PCFG (Zhu et al. 2020) 55.3; S-DIORA 57.6 (max 64.0); Constituency Test 62.8 (max 65.9).
  - Label recall for TN-PCFG p = 500: NP 75.4, VP 48.4, PP 67.0, SBAR 50.3.
  [Yang, Zhao & Tu 2021](https://arxiv.org/abs/2104.13727)
- Rank-space PCFG (Yang, Liu & Tu, NAACL 2022):
  - "Ours with 9000 PTs and 4500 NTs" reaches 64.1 PTB S-F1.
  - Comparison table: NBL-PCFG (Yang et al. 2021a) 60.4; StructFormer 54.0; DIORA+span constraint (Xu et al. 2021) 61.2.
  [Yang, Liu & Tu 2022](https://arxiv.org/abs/2205.00484)
- SimplePCFG (Liu, Yang, Kim & Tu, Findings of EMNLP 2023):
  - Decomposes π(A→BC) into π(B↶A)·π(A↷C), so left and right children are generated independently given the parent. The inside complexity becomes O(l³|N| + l²|N|²).
  - SN-PCFG with 4096 NTs: PTB test S-F1 65.1±2.1, perplexity 132.5. SC-PCFG with 2048 NTs: 60.6±3.6.
  - Same table: Fast-R2D2 57.2. Multilingual SC-PCFG 2048: Chinese 42.9, French 49.9, German 49.1.
  [Liu et al. 2023](https://arxiv.org/abs/2310.14997)

**Constituency tests and ensembles**
- Constituency tests (Cao, Kitaev & Klein, EMNLP 2020):
  - Scores each span with transformations (e.g., pronoun substitution, clefting) judged by a RoBERTa grammaticality model trained on real vs. corrupted sentences. The training data are 5 million unlabeled English Gigaword sentences.
  - Picks the binary tree with the highest total span score, then refines by alternating between tree estimates and the grammaticality model.
  - Result: "62.8 F1 on the Penn Treebank test set", the mean of four random restarts (max 65.9).
  [Cao, Kitaev & Klein 2020](https://arxiv.org/abs/2010.03146)
- Ensemble distillation (Shayegh, Cao, Zhu, Cheung & Mou, ICLR 2024):
  - Method: "tree averaging" by a CYK-like search for the tree with maximum total F1 against the teachers' trees, then distillation into an RNNG/URNNG.
  - Teacher replications (PTB test F1, mean±std over 5 runs): ON 44.3±6.0, N-PCFG 51.0±1.7, C-PCFG 55.5±2.4, DIORA 58.9±1.8, S-DIORA 57.0±2.1, ConTest 62.9±1.6, ContexDistort 47.8±0.9.
  - Ensemble results: 70.4±0.6 (corresponding runs), 71.9 (best teachers), 72.8 after URNNG, which the authors call "a new state of the art". Selective MBR (choose one teacher's tree) gives 66.3; union distillation fails (65.6 after RNNG).
  - Oracle binary tree in their setup: 83.3.
  - Under domain shift to SUSANNE: ensemble 50.3 vs. right-branching 26.9.
  [Shayegh et al. 2024](https://arxiv.org/abs/2310.01717)
- Shayegh et al. find heterogeneous teachers help beyond denoising: "different unsupervised parsers learn different aspects of the language structures". [Shayegh et al. 2024](https://arxiv.org/abs/2310.01717)
- The same paper's Appendix E shows how far replications drift. For example:
  - ON runs range 32.9–50.0;
  - the reported means are DIORA 56.8, S-DIORA 57.6 (max 64.0), ConTest 62.8 (max 65.9) and ContexDistort 49.0.
  [Shayegh et al. 2024](https://arxiv.org/abs/2310.01717)

**Simplicity bias and parse-focusing (Park & Kim)**
- Park & Kim (Findings of ACL 2024) identify two problems: "structural optimization ambiguity" (many structurally different grammars are equally optimal) and "structural simplicity bias" (models underuse rules).
  - Their fix, sentence-wise parse-focusing, restricts each sentence's parse pool to trees proposed by pretrained unsupervised parsers on the same data (StructFormer, NBL-PCFG, FGG-TNPCFG).
  - PTB S-F1 over 32 runs: 67.4±0.9 (NT=30) and 69.6±0.6 (max 70.3, NT=4500).
  [Park & Kim 2024](https://arxiv.org/abs/2407.16181)
- Park & Kim also report evaluation discrepancies:
  - DIORA gets 55.7 on CoreNLP-binarized test trees but 43.6±0.9 on non-binarized trees;
  - FGG-TNPCFG is reported at 64.1 but reproduces at 57.4±6.0;
  - NBL-PCFG is reported at 60.4 but reproduces at 53.3±11.5.
  [Park & Kim 2024](https://arxiv.org/abs/2407.16181)
- Park & Kim (arXiv, Sept 2025) diagnose "probability distribution collapse" in neural parameterizations as what forces "unnecessarily large yet underperforming grammars". Their collapse-relaxing parameterization (CRNP) with parse-focusing reaches PTB S-F1:
  - 69.4±0.3 with only 30 NTs;
  - 70.2±0.5 (max 70.9) with 90 NTs.
  - With gold trees as the focusing bias (effectively supervised), the 30-NT model reaches only 73.7±0.3.
  [Park & Kim 2025](https://arxiv.org/abs/2509.20734)

**Semantic-information objectives**
- Span-overlap (Chen, He, Bollegala & Miyao, Findings of ACL 2024):
  - Method: generate predicate-argument-structure-equivalent paraphrases with GPT-3.5, score each word sequence by its frequency across the paraphrase set, and pick the tree with maximum total score.
  - PTB S-F1 52.9 (Chinese CTB 48.7). It outperforms existing unsupervised parsers in 8 of 10 languages.
  - In its Table 4 the authors' reproductions come in below reported numbers: C-PCFG 52.6 vs. 55.2 reported; TN-PCFG 51.2 vs. 57.7.
  - A GPT-3.5 bracket-prompting parser scores 36.2 on PTB.
  [Chen et al. 2024](https://arxiv.org/abs/2404.12059)
- SemInfo (Chen, He, Miyao & Bollegala, ICLR 2025): trains PCFGs to maximize the "semantic information" of constituents, estimated with a bag-of-substrings model over GPT-4o-mini paraphrases, instead of log-likelihood (LL).
  - Protocol: PTB test SF1c, i.e., the sentence-F1 averaged over the corpus, computed only for sentences longer than two words, with trivial spans dropped.

  | PCFG variant | SemInfo | LL |
  |---|---|---|
  | CPCFG | 65.74±0.81 | 53.75 |
  | NPCFG | 64.45±1.13 | 50.96 |
  | SCPCFG | 67.27±1.08 | 49.42 |
  | SNPCFG | 67.15±0.62 | 58.19 |
  | TNPCFG | 66.55±0.96 | 53.37 |

  - Average gains are +13.09 SF1c in English, +6.02 Chinese, +7.31 French and +4.92 German.
  - Baselines in the same table: MaxTreeDecoding on SemInfo alone 58.28; direct GPT-4o-mini bracket prompting 36.16.
  - LL correlates with parsing accuracy only early in training, then its correlation "quickly diminishes".
  [Chen et al. 2025](https://arxiv.org/abs/2410.02558)

**2026 results**
- Holographic Neural PCFG (Yamaki, Mochihashi, Shimada & Taniguchi, arXiv July 2026): rule scoring by circular correlation of torus-constrained symbol embeddings.
  - PTB test SF1 (5 seeds): 64.6±0.4 with maximum likelihood; 68.1±0.5 with SemInfo training.
  - Rule-scoring parameters cut "by 99.94%" relative to the baseline model.
  - Its table lists NL-PCFG at 57.3, whereas TN-PCFG's table lists 55.3 (a discrepancy in the secondary citations).
  [Yamaki et al. 2026](https://arxiv.org/abs/2607.08063)

**Pretrained-LM probes and incremental structure**
- Contrastive hashing over pretrained LMs (Wang & Utiyama, EMNLP 2024): PTB S-F1 62.4 mean / 64.1 max (RoBERTa-base, 16 bits). [Wang & Utiyama 2024](https://arxiv.org/abs/2410.04074)
- The same table lists:
  - ReCAT (Hu et al. 2024b) 65.0;
  - co-training (Maveli & Cohen 2022) 63.1 / 66.8;
  - GPST 57.5.
  [Wang & Utiyama 2024](https://arxiv.org/abs/2410.04074)
- GPST (Hu, Ji, Zhu, Wu & Tu, ACL 2024): an unsupervised syntactic LM pretrained on raw text (OpenWebText, 9 billion tokens) that "incrementally generates a sentence with its syntactic tree in a left-to-right manner".
  - Left-to-right (incremental) unsupervised parsing F1 on WSJ: GPST-small (wiki103) 55.25, GPST-medium (OpenWebText) 54.71.
  - Earlier left-to-right systems: PRPN 37.4; NV (unsupervised) 29.0.
  - The non-incremental GPST-small reaches 57.46.
  [Hu et al. 2024](https://arxiv.org/abs/2403.08293)

**LLM prompting**
- Bai et al. (IEEE/ACM TASLP; arXiv v3 Sept 2025) prompt LLMs to output constituency trees.
  - On a 500-sentence random subset of PTB test, GPT-4 zero-shot scores 73.00 F1 and 76.87 with the authors' multi-agent "PMC" strategy, which they call "a new benchmark for end-to-end unsupervised constituency parsing".
  - Their abbreviation table defines labeled precision/recall (LP/LR), so this is not the unlabeled, punctuation-free S-F1 protocol.
  - Zero-shot scores for open models: LLaMA3-70B-IT 44.97; Qwen2.5-72B-IT 57.99.
  [Bai et al. 2025](https://arxiv.org/abs/2310.19462)

**Unsupervised chunking**
- Unsupervised (flat) chunking with a hierarchical RNN (Wu, Deshmukh, Wu, Lin & Mou; Computational Linguistics 51(3):815–841, 2025): induces non-hierarchical word-to-chunk and chunk-to-sentence structure by pretraining on an unsupervised parser's output, then fine-tuning on downstream tasks.
  - Chunk structure emergence is "transient" during downstream training.
  - Text excerpt: phrase F1 of a baseline chunker vs. Compound PCFG is 42.05 vs. 62.89 on CoNLL-2000.
  [Wu et al. 2025](https://arxiv.org/abs/2309.04919)

**Leaderboard aggregator (secondary)**
- A leaderboard aggregator (sota2) reproduces the Park & Kim 2024 table. It is secondary and lists FGG-TNPCFG at 64.1 (non-reproduced) and parse-focusing at 69.6. [sota2 leaderboard](https://www.sota2.com/research/sota/unsupervised-constituency-parsing-on-penn-treebank-english-test)
- S-DIORA (Drozdov et al., EMNLP 2020), WSJ test with punctuation removed:
  - S-DIORA trained on PTB: F1 max 63.96, mean 57.6; F1 on ≤10-word sentences 71.80.
  - The same table reports C-PCFG at 60.32 max / 55.2 mean and DIORA at 56.75 max.
  - Some DIORA-line evaluations binarize the ground truth.
  [Drozdov et al. 2020](https://aclanthology.org/2020.emnlp-main.392.pdf)
- URNNG (Kim et al., NAACL 2019) uses a different protocol: corpus-level EVALB, its own PTB version.
  - PTB F1: URNNG 40.7; right branching 34.8; supervised RNNG 68.1; oracle binary 82.5.
  - These baselines differ from the 39.5 / 84.3 of the Compound PCFG protocol.
  [Kim et al. 2019b](https://arxiv.org/abs/1904.03746)

### Inferences
- **Calibrate PTB expectations around four anchors:**
  - right-branching: 39.5 S-F1;
  - pre-2019 neural LMs: 47–48 (PRPN, ON-LSTM);
  - first-generation neural PCFGs: 50.8 / 55.2;
  - large-symbol PCFGs: 57.7–65.1.

  Systems above about 67 use either other parsers' output (parse-focusing, ensembles) or LLM/PLM signal (SemInfo, ConTest, hashing). A symbolic, incremental TRELLIS that clears right-branching on §23 would already beat the 1990s–2000s count-based EM line. Matching Compound PCFG at about 55 is a strong result for an interpretable incremental system.
- **Neural PCFG success comes from rule sharing, many symbols and aggregation, not from search.**
  - The same grammar class goes from below 35 (scalar EM) to 50.8 (shared embeddings), i.e. parameter sharing.
  - More symbols help: 60 → 500 preterminals adds 6.3 points; 4,500 nonterminals reach 64.1.
  - Pooling parsers or restricting each sentence's parse pool adds 5–8 points.

  TRELLIS has analogues for each. Cobweb's concept hierarchy provides back-off sharing across categories. Many leaves combined with a coarse cut give "many symbols for learning, few for the reported grammar". Shuffled-order ensembles, or a frontier of candidate parses, give aggregation.
- **SimplePCFG's factorization is essentially Cobweb's attribute independence within a concept.** In SimplePCFG, left and right children are independent given the parent. A TRELLIS content concept with LEFT and RIGHT child-class attributes, independent given the concept, is therefore already a simple-PCFG nonterminal. Inside-outside over the content hierarchy costs O(l³|N| + l²|N|²).
- **Low self-agreement is a known problem in neural PCFGs too.** Compound PCFG agrees with itself across seeds at self-F1 66.8, and reproductions of published results come in 3–7 points lower. The order sensitivity of an incremental Cobweb learner should be reported the same way: self-F1 across data orders.
- **LLM numbers are not a fair target.** Under the standard protocol, prompting LLMs for brackets is weak: GPT-4o-mini scores 36.16 and GPT-3.5 36.2 S-F1, below right-branching. GPT-4's 73–77 in Bai et al. is on a different metric and test subset. The LLM-based gains that do hold up (SemInfo, span-overlap) use LLMs to generate paraphrases, not parses.

### Gaps
- **2025–2026 coverage.** arXiv API queries for "unsupervised constituency", "unsupervised parsing" and "grammar induction" sorted by date, plus WebSearch, found only these new PTB-constituency results: Park & Kim 2025 (arXiv, venue not determined), Hol-PCFG (arXiv, July 2026), SemInfo (ICLR 2025) and Bai et al. (TASLP 2025). Conference-only 2025–2026 papers not on arXiv could have been missed; the WebSearch budget ran out partway through.
- **Unresolved discrepancies:**
  - The 2026 Hol-PCFG table lists NL-PCFG at 57.3, whereas TN-PCFG's table lists 55.3. The original L-PCFG PDF did not yield the number via text extraction.
  - Park & Kim's TN-PCFG max is printed as 55.6 against a mean of 57.7, which is impossible and likely a typo.
- **Unverified items:**
  - Kim et al. 2020 ("Are pre-trained LMs aware of phrases?") and other PLM-probing baselines were not verified numerically.
  - The PTB numbers in Wu et al.'s HRNN chunking paper (Computational Linguistics 2025) were not extracted beyond the CoNLL-2000 phrase-F1 excerpt.

## 4. Incremental and curriculum strategies

### Takeaway
Incremental induction is rare. A systematic review found 1 of 33 implementations processing input incrementally.

Two incremental learners stand out:
- **CCL**, a left-to-right, online, heuristic parser on raw text. It reached 75.9 UF1 on WSJ10 and 57.4 on WSJ40 at thousands of words per second.
- **GPST**, a left-to-right neural syntactic LM. It reached about 55 S-F1 on WSJ §23, on par with the batch Compound PCFG.

Curriculum by sentence length helps EM-style learners: Baby Steps / Leapfrog, and Compound PCFG's ≤30 then +1 per epoch schedule.

Hard (Viterbi) and soft (inside-outside) updates each win in different regimes, and alternating them (Lateen EM) is best. Online, stepwise EM is a drop-in way to make any count-based learner incremental.

"Starting small" is not universally beneficial (Rohde & Plaut), so curricula should be tested rather than assumed.

### Cited Findings

**Curriculum and update rules**
- **Baby Steps / Less is More / Leapfrog** (Spitkovsky, Alshawi & Jurafsky, NAACL-HLT 2010, pp. 751–759), applied to DMV dependency induction.
  - Baby Steps trains on WSJ1, then uses that model to initialize WSJ2, and so on up to WSJ45, with no other initializer.
  - Less is More trains only on sentences up to length 15.
  - Leapfrog mixes the two, then ramps up quickly.
  - Results on §23 (directed accuracy, all sentences): Baby Steps 39.4, Less is More 44.1, Leapfrog 45.0, vs the reimplemented DMV at 34.2.
  - Their diagnosis: long sentences "amplify ambiguity and pollute fractional counts with noise".
  - [Source](https://aclanthology.org/N10-1116/)
- **Viterbi training** (Spitkovsky, Alshawi, Jurafsky & Manning, CoNLL 2010).
  - Hard EM "is more accurate than standard inside-outside re-estimation (classic EM), significantly faster, and simpler": 44.8% on §23 (all), 47.9% with a good initializer.
  - "Classic EM learns better from short sentences but cannot cope with longer ones, where Viterbi thrives."
  - Baby Steps with smoothed Viterbi "fails miserably".
  - [Source](https://aclanthology.org/W10-2902/)
- **Lateen EM** (Spitkovsky, Alshawi & Jurafsky, EMNLP 2011, pp. 1269–1280).
  - Alternates the soft and hard EM objectives, switching when stuck.
  - One alternation lifts §23 from 50.4 to 52.8. Across 19 languages, simple lateen on hard EM adds 5.5 points on average.
  - Elaborate versions let "each [objective] validating the moves proposed by the other".
  - [Source](https://aclanthology.org/D11-1117/)
- **Online (stepwise) EM** (Liang & Klein, NAACL 2009, pp. 611–619).
  - Online variants "provide significant speedups and … can even find better solutions than those found by batch EM". Tested on POS tagging, document classification, word segmentation and word alignment.
  - Update: μ ← (1−η_k)μ + η_k s'_k, with η_k = (k+2)^−α, valid for 0.5 < α ≤ 1; mini-batches add stability.
  - [Source](https://aclanthology.org/N09-1069/)
- **Compound PCFG's curriculum** trains on sentences up to length 30 in epoch 1 and raises the limit by 1 per epoch. Kim et al. "did not experiment with online EM (Liang and Klein, 2009)" for the scalar PCFG that failed. [Source](https://arxiv.org/abs/1906.10225)
- **Elman 1993** (*Cognition* 48(1):71–99).
  - A simple recurrent network on a grammar with agreement and nested relative clauses "failed to master the task" when trained on the full corpus.
  - It succeeded with incremental input (simple sentences first) or incremental memory (recurrence cut every 3–4 words, then lengthened).
  - [Source](http://web.archive.org/web/20220120172243/https://crl.ucsd.edu/~elman/Papers/elman_cognition1993.pdf)
- **Rohde & Plaut 1999** (*Cognition* 72(1):67–109) found the opposite.
  - "Under no condition did the simple training regimen outperform" training on the full complex language. The advantage of starting large grew with semantic constraints.
  - Limiting memory gave "no significant difference".
  - [Source](https://www.cnbc.cmu.edu/~plaut/papers/pdf/RohdePlaut99Cog.startingSmall.pdf)

**Incremental learners**
- **CCL, common cover links** (Seginer, ACL 2007, pp. 384–391).
  - Parses raw words left to right. It only adds links ending at the current word, greedily adding the strongest positive-weight link.
  - Its lexicon of "adjacency points" (label strengths, Stop/In/Out) is updated after each utterance; there are no POS tags and no clustering.
  - UP / UR / UF1: WSJ10 75.6 / 76.2 / 75.9; WSJ40 58.9 / 55.9 / 57.4. Right-branching scores 61.7 / 40.5.
  - Speed: about 4,000 words/sec parsing and 3,200–3,600 words/sec learning.
  - Caveat: trained on the full corpus and scored on subsets, with punctuation used during learning. Without phrasal punctuation CCL loses about 14 precision and recall points on WSJ, per Ponvert et al.
  - [Source](https://aclanthology.org/P07-1049/); [Source](https://aclanthology.org/P11-1108/)
- **GPST** (Hu, Ji, Zhu, Wu & Tu, ACL 2024): an unsupervised syntactic LM that "incrementally generates a sentence with its syntactic tree in a left-to-right manner".
  - Left-to-right parsing F1 on WSJ: 55.25 for GPST-small (trained on wiki103), vs PRPN 37.4. The non-incremental reading gives 57.46.
  - [Source](https://arxiv.org/abs/2403.08293)
- **Depth-bounded PCFG** (Jin, Doshi-Velez, Miller, Schuler & Schwartz, TACL 6:211–224, 2018).
  - Applies a memory (center-embedding) bound to PCFG induction.
  - On 14,251 child-directed Eve utterances, scored against Pearl & Sprouse PTB-style trees: DB-PCFG 71.6 F1, best among induced models, but below right-branching at 76.3.
  - [Source](https://aclanthology.org/Q18-1016/); [Source](https://arxiv.org/abs/1802.08545)
- **Online adaptor grammars** (Zhai et al., TACL 2014) process minibatches with decayed statistics, add newly sampled fragments to the cache and periodically prune by usefulness. See Section 2. [Source](https://aclanthology.org/Q14-1036/)
- **Incremental initialization trade-off** (Johnson & Goldwater 2009). Incremental initialization gives higher segmentation F (0.89 max-marginal) but lower posterior: it "gets stuck" on short common substrings. [Source](https://aclanthology.org/N09-1036/)

**Reviews and recent cognitive models**
- **Muralidaran, Spasić & Knight** (*Natural Language Engineering* 27(6):647–689, DOI 10.1017/S1351324920000327) reviewed 43 studies (33 implementations).
  - 31 of 33 output hierarchical structures. Only 1 of 33 processes input incrementally.
  - Experimental and theoretical studies favour "a usage-based, incremental, sequential system".
  - [Source](https://www.cambridge.org/core/journals/natural-language-engineering/article/systematic-review-of-unsupervised-approaches-to-grammar-induction/413BE0F463829B4C23DCB8EBD67B4DFF)
- **Marcheva(-Nash), Salhan & Sun** (CogSci 2026; arXiv 2605.08476) cast maturational theories as staged grammar induction on CHILDES.
  - A bottom-up "GROWING" ordering significantly outperforms "INWARD" on F1, JS divergence and child-utterance log-likelihood.
  - [Source](https://arxiv.org/abs/2605.08476)
- **Zhou, Nagy, Dayan & Wu** (arXiv May 2026) model human sequence learning with a hierarchical adaptor grammar.
  - It has local and global libraries under memory and computation constraints (rate-distortion), and claims better rate-distortion trade-offs than "fixed grammars or shallow chunking methods".
  - Reaction times rose at inferred program boundaries, and "the order of experience shapes future abstractions".
  - [Source](https://arxiv.org/abs/2606.20623)

### Inferences
- **Soft vs. hard updates is not settled.** Soft counts help on short sentences; Viterbi helps on long ones; alternating helps most. For TRELLIS:
  - use posterior-weighted (inside-outside) evidence in the frontier, especially early and on short sentences;
  - allow hard top-1 or top-k commits once a chunk has graduated;
  - consider a Lateen-style check where the soft objective vetoes hard commits that lower it.
- **Use a length curriculum, but measure it.** Length-ordered streams are cheap and well supported for EM-family learners (Carroll & Charniak 1992; Baby Steps; Compound PCFG). The Rohde & Plaut counter-evidence means TRELLIS should ablate it, not assume it.
- **CCL is the most relevant existence proof:** an online, greedy, symbolic learner on raw words beating CCM on WSJ10 (75.9). It is a realistic yardstick for an incremental TRELLIS. CCL's use of punctuation and its train-on-test protocol must be matched, or noted, in any comparison.
- **Cobweb's order sensitivity matches recent cognitive work** that treats path dependence as a feature (Zhou et al. 2026). Ensembling across orders (Section 3) is the engineering counterpart.

### Gaps
- No incremental learner evaluated under the modern full-WSJ protocol was found other than GPST, which needs 9B-token pretraining. CCL and Ponvert were evaluated on held-out §23 by Ponvert et al. (CCL 51.7 F, all lengths).
- No study found compares soft vs. hard incremental updates specifically for constituency chunk learning. The Spitkovsky results are for dependency (DMV).

## 5. Evaluation protocols, baselines, free datasets and pitfalls (and how GRIDS-style omission/commission relates)

### Takeaway
Two PTB tracks are standard.
- **WSJ10.** 7,422 sentences of ≤10 words after removing punctuation and null elements. Gold POS tags are the input, and the same sentences are used for training and testing (Klein & Manning).
- **Full WSJ.** Train on §02–21, develop on §22, test on §23 (2,416 sentences). Words are the input, with punctuation removed, everything lowercased and a 10K vocabulary. Trivial spans are discarded and the score is the unlabeled F1 averaged per sentence (S-F1). Corpus-level F1 runs about 3 points lower.

Baselines differ a lot by convention. On §23, right-branching scores 39.5 S-F1 / 36.1 corpus-level F1, and the binarized-gold ceiling is 84.3. On WSJ10, right-branching scores 56.6–61.7 depending on the metric.

The literature's critiques boil down to four problems:
- Scores are not comparable across scripts: the same output trees can differ by more than 20 F1.
- Many systems use labeled dev trees for model selection, and as few as 15 labeled examples change conclusions.
- Punctuation and right-branching bias inflate English scores.
- Latent trees can be useful without resembling PTB.

GRIDS's errors of omission and commission are *language-level* measures: target sentences the learned grammar cannot parse, and learned-grammar samples the target rejects. PTB bracket recall and precision are their *structural* analogues. PTB has no target grammar, so language-level commission needs grammaticality judgments.

### Cited Findings

**WSJ10 conventions**
- **Definition.** WSJ-10 is "the 7422 sentences in the Penn treebank Wall Street Journal section which contained no more than 10 words after the removal of punctuation and null elements". [Source](https://aclanthology.org/P02-1017.pdf)
- **Two scoring conventions from the same authors.** Klein & Manning 2002 discarded "single words and entire sentences" and macro-averaged per sentence. Their 2004 numbers are micro-averaged and include full-span brackets; they say the two are "overall, approximately the same". [Source](https://aclanthology.org/P02-1017.pdf); [Source](https://aclanthology.org/P04-1061.pdf)
- **Train = test.** The original CCM experiment uses the whole WSJ ≤10 corpus for both training and test. Held out, CCM scores 62.97 (micro). [Source](https://aclanthology.org/2020.acl-main.300.pdf)
- **PRPN's code** treats all of WSJ/00–24 as training data and tests on those sentences. [Source](https://github.com/yikangshen/PRPN/blob/master/data_ptb.py)
- **Klein & Manning 2004 WSJ10 baselines (P / R / F1):** left-branching 25.6 / 32.6 / 28.7; random 31.0 / 39.4 / 34.7; right-branching 55.1 / 70.0 / 61.7; upper bound 78.8 / 100 / 88.1. [Source](https://aclanthology.org/P04-1061.pdf)
- **ON-LSTM WSJ10 baselines (sentence-level):** right-branching 56.6, random 31.7, balanced 43.4, left 19.6. [Source](https://arxiv.org/abs/1810.09536)
- **Seginer's punctuation-aware right-branching** reaches 65.8 on WSJ10. [Source](https://aclanthology.org/P07-1049.pdf)

**Full-WSJ protocol and its scripts**
- **Kim, Dyer & Rush 2019 protocol:**
  - sections 2–21 / 22 / 23; "discard punctuation, lowercase all tokens, and take the top 10K most frequent words";
  - "discard trivial spans and evaluate on sentence-level F1";
  - validation F1 was used to choose some hyperparameters, so the setup is "arguably not fully unsupervised".
  - [Source](https://aclanthology.org/P19-1228.pdf)
- **Kim's `process_ptb.py`** keeps only tokens whose tag is in a 36-tag word list. This removes punctuation, `-NONE-`, `#` and `$`. [Source](https://github.com/harvardnlp/compound-pcfg/blob/master/process_ptb.py)
- **Split sizes:** the test set is 2,416 sentences [Source](https://arxiv.org/abs/1810.09536); dev is 1,700 sentences and train about 40K [Source](https://aclanthology.org/2020.emnlp-main.614.pdf).
- **The compound-pcfg `eval.py`:**
  - skips length-1 sentences;
  - drops width-1 spans and the whole-sentence span;
  - collapses duplicate spans;
  - gives length-2 sentences sentence-F1 = 1.

  The README notes "quirky behavior in corner cases", whereas corpus-level F1 "does not have this issue". [Source](https://github.com/harvardnlp/compound-pcfg/blob/master/eval.py); [Source](https://github.com/harvardnlp/compound-pcfg)
- **Kim et al. 2019 baselines on §23:**
  - Sentence-level: left 8.7, right 39.5, random 19.2, binarized oracle 84.3.
  - Corpus-level: right 36.1, oracle 84.7. The oracle uses right-branching binarization.
  - [Source](https://aclanthology.org/P19-1228.pdf)
- **Li et al. 2020 (micro / macro / EVALB) on all test lengths:** right-branching 35.88 / 39.58 / 39.61; upper bound 84.41 / 83.32 / 85.34. [Source](https://aclanthology.org/2020.acl-main.300.pdf)
- **With punctuation kept,** right-branching collapses to 0.07 micro F1. [Source](https://aclanthology.org/2020.acl-main.300.pdf)
- **Different protocols give different baselines.**
  - URNNG (EVALB, corpus-level): right-branching 34.8, oracle binary 82.5. [Source](https://aclanthology.org/N19-1114.pdf)
  - Williams/DIORA (gold binarized, punctuation kept): right-branching 16.5. [Source](https://aclanthology.org/N19-1116.pdf)
- **The same model under different conventions.**
  - DIORA scores 55.7 on CoreNLP-binarized test trees vs 43.6 on n-ary trees. [Source](https://arxiv.org/abs/2407.16181)
  - Compound PCFG scores 39.2 in Shi et al. with a 35K vocabulary vs 55.2 with 10K. [Source](https://aclanthology.org/2020.emnlp-main.614.pdf)
- **EVALB** (`COLLINS.prm`) deletes only some punctuation, counts repeated constituents and is labeled by default. Li et al. state that metric variants "can sometimes differ more than 20 F1 points" on the same parses. [Source](https://nlp.cs.nyu.edu/evalb/); [Source](https://aclanthology.org/2020.acl-main.300.pdf)

**Critiques**
- **Citation correction.** The ACL 2020 "An Empirical Comparison of Unsupervised Constituency Parsing Methods" is by Jun Li, Yifan Cao, Jiong Cai, Yong Jiang and Kewei Tu, not "Li, Mou & Keller". Li, Mou & Keller wrote "An Imitation Learning Approach to Unsupervised Parsing" (ACL 2019). [Source](https://aclanthology.org/2020.acl-main.300.pdf); [Source](https://aclanthology.org/P19-1338.pdf)
- **Li, Cao, Cai, Jiang & Tu 2020.**
  - "Recent models do not show a clear advantage over decade-old models."
  - CCM trained on ≤10-word sentences beats every model trained on length-40 data without punctuation.
  - DIORA drops from 49.39 to 42.63 without pretrained embeddings.
  - They recommend reporting results at both ≤10 and all lengths, with mean±SD.
  - [Source](https://aclanthology.org/2020.acl-main.300.pdf)
- **Shi, Livescu & Gimpel 2020** (EMNLP).
  - Tuning on as few as 15 labeled examples beats unsupervised model selection. Few-shot parsing "quickly dominates once there are more than 55 examples".
  - §23 EVALB with a 35K vocabulary: a few-shot parser with augmentation and self-training reaches 61.2 with 55 labeled examples, vs 39.2–52.0 for unsupervised models even when tuned on labels.
  - They propose fully unsupervised model-selection criteria.
  - [Source](https://aclanthology.org/2020.emnlp-main.614.pdf)
- **Williams, Drozdov & Bowman 2018** (TACL 6) on latent-tree models such as ST-Gumbel:
  - their parses are "not especially consistent across random restarts" (self-F1 49.9);
  - they are shallower than PTB, and "do not resemble those of PTB or any other semantic or syntactic formalism";
  - full-WSJ F1 is 19.0 vs 21.3 for random trees;
  - yet ST-Gumbel beats conventional tree models on sentence classification.
  - [Source](https://aclanthology.org/Q18-1019.pdf)
- **Htut, Cho & Bowman 2018.**
  - Shen et al. tuned "and even trained on what is effectively the test set".
  - PRPN-LM scores WSJ10 70.5 (σ 0.4) but WSJ §23 only 37.4 (σ 0.3), below right-branching on that protocol.
  - [Source](https://aclanthology.org/D18-1544.pdf)
- **Dyer, Melis & Blunsom 2019.** The PRPN/ON tree-extraction algorithm has "a marked bias for right-branching structures". [Source](https://arxiv.org/abs/1909.09428)
- **Zhao & Titov 2021.** English-tuned configurations transfer poorly to other languages. Compound PCFG's VP recall (40.7) trails right-branching (71.5). [Source](https://aclanthology.org/2021.adaptnlp-1.17.pdf)
- **Right-branching as a baseline.** Klein & Manning: right-branching "encodes a significant fact about English structure, and an induction system need not beat it". On child-directed Eve it scores 76.3, above every induced model (best 71.6). [Source](https://aclanthology.org/P02-1017.pdf); [Source](https://arxiv.org/abs/1802.08545)
- **Reproductions run below reported numbers.** Examples: Compound PCFG 52.6 reproduced vs 55.2 reported; TN-PCFG 51.2 vs 57.7; FGG-TNPCFG 57.4 vs 64.1; NBL-PCFG 53.3 vs 60.4. [Source](https://arxiv.org/abs/2404.12059); [Source](https://arxiv.org/abs/2407.16181)

**Free and alternative data**
- **NLTK `treebank` sample.**
  - 199 files (`wsj_0001`–`wsj_0199`), 3,914 trees.
  - 100,676 leaves including null elements; 94,084 tokens without `-NONE-`; 82,369 after removing punctuation, `$` and `#`.
  - Only 555 sentences of ≤10 words.
  - NLTK's own descriptions conflict: the README says a "~5% fragment", while `index.xml` and the NLTK book say "10%". By file count it is ≈8% of Treebank-3's 2,499 WSJ files.
  - Licence: "fair use … non-commercial". The files appear to be sections 00–01, which is inferred from file naming.
  - [Source](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip); [Source](https://github.com/nltk/nltk_data/blob/gh-pages/index.xml); [Source](https://www.nltk.org/book/ch08.html); [Source](https://catalog.ldc.upenn.edu/LDC99T42)
- **Universal Dependencies.**
  - English-EWT: 16,622 sentences, CC BY-SA 4.0, dependencies only. [Source](https://universaldependencies.org/treebanks/en_ewt/index.html)
  - English-GUM: 14,353 sentences, CC BY-NC-SA 4.0. [Source](https://universaldependencies.org/treebanks/en_gum/index.html)
  - English-CHILDES (UD ≥ 2.16): 48,183 sentences, CC BY-SA 4.0. [Source](https://universaldependencies.org/treebanks/en_childes/index.html)
  - Noji et al. 2016 convert dependency trees to *non-binarized* brackets and score them with PARSEVAL. [Source](https://aclanthology.org/D16-1004.pdf)
- **GUM constituent trees** are "automatic parser output from gold POS", so they are not hand-built. Annotations are CC BY 4.0; some source texts are non-commercial. [Source](https://github.com/amir-zeldes/gum)
- **Pearl & Sprouse CHILDES Treebank.** Child-directed speech (Brown Adam/Eve/Sarah, etc.) in PTB-II-like structure, produced by a parser and hand-checked. No licence is stated. [Source](https://sites.socsci.uci.edu/~lpearl/CoLaLab/CHILDESTreebank/childestreebank.html)
- **BabyLM corpora.**
  - 2023: Strict-Small ≈10M words and Strict ≈100M, with CHILDES 5% of the data. [Source](https://aclanthology.org/2023.conll-babylm.1.pdf)
  - 2024: CHILDES raised to 29M words. [Source](https://arxiv.org/abs/2404.06214)
  - 2025: same data, with an epoch cap. [Source](https://arxiv.org/abs/2502.10645)
  - No BabyLM work evaluating unsupervised constituency against gold trees was found. Chen & Portelance 2023 trained a compound PCFG on BabyLM data but reported BLiMP-style scores. [Source](https://aclanthology.org/2023.conll-babylm.5.pdf)
- **Synthetic suites.**
  - Omphalos (ICGI 2004): 10 context-free problems. [Source](https://www.irisa.fr/Omphalos/data-sets.html)
  - Langley & Stromsten's four languages: adjective phrases, relative clauses, aⁿbⁿ, balanced parentheses. [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
  - Tu & Honavar's Num-agr / Langley1 / Langley2 / Emile2k / TA1. [Source](https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/icgi08.pdf)
  - ADIOS's ATIS-CFG: 4,592 rules. [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953/)
  - Allen-Zhu & Li's cfg3 family: depth-7 grammars with strings up to 729 tokens. [Source](https://arxiv.org/abs/2305.13673)
  - BLISS (Lan, Chemla & Katzir 2023): formal languages (aⁿbⁿ, aⁿbⁿcⁿ, Dyck-1/2). Networks trained with an MDL objective generalize better from less data. [Source](https://arxiv.org/abs/2308.08253)

**Language-level evaluation (the GRIDS framing)**
- **Langley & Stromsten 2000** (ECML) distinguish:
  - "errors of omission (failures to parse sentences in the target language), which indicate an undergeneral grammar";
  - "errors of commission (failures to generate only sentences in the target language), which indicate an overgeneral one".

  Both are estimated by sampling from the target grammar T and the learned grammar L. Omission comes from the fraction of T-samples parsed by L; commission from the fraction of L-samples parsed by T. Test sentences are longer than training sentences, to test recursion. [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **ADIOS** defines precision as "the proportion of C_learner accepted by the teacher" (judged by human referees for natural language) and recall as "the proportion of C_target accepted by the learner". [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953/)
- **Tu & Honavar 2008** use the same weak-generative-capacity precision, recall and F with 200 samples per direction. [Source](https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/icgi08.pdf)
- **Marcheva-Nash et al. 2026** combine PARSEVAL F1 on 1,000 CHILDES-Treebank parses, length-normalized log-likelihood on child utterances, and Jensen-Shannon divergence to an oracle grammar. [Source](https://arxiv.org/abs/2605.08476)
- **Caution on acceptability judgments.** A system might "just generate overly simple utterances" and still be judged acceptable. [Source](https://aclanthology.org/P07-3008.pdf)

### Inferences
- **Mapping between the frameworks:**
  - TRELLIS's bracket-level omission = 1 − unlabeled span recall.
  - Bracket-level commission = 1 − unlabeled span precision.
  - Their harmonic mean is the literature's unlabeled F1.

  For paper text the user can report "omission/commission" and give the harmonic mean only in comparison tables, labelled as what other papers call F1. The micro/macro choice and the treatment of trivial spans must be stated: sentence-averaged omission and commission correspond to S-F1, pooled counts to corpus F1.
- **Binary output is penalized as commission against n-ary gold.** "Choosing its own binarization", a known v1 issue on the medium grammar, appears in PTB scoring as commission. The 84.3 ceiling quantifies that cost. A non-crossing (compatibility) measure, as used by Pereira & Schabes, ABL and Kurihara & Sato, separates "different but compatible bracketing" from "wrong attachment". It is worth reporting as a diagnostic alongside the standard metric.
- **Run every baseline through the same script.** Because baselines move by up to about 20 points between conventions, TRELLIS should run left-branching, right-branching, random and oracle through the exact Kim et al. pipeline it uses, and should never mix numbers across protocols.
- **The NLTK sample suits development, not benchmarking.** It is sufficient (3,914 sentences, 555 of them ≤10 words) but is not a standard split and cannot be compared with §23 results. Real PTB evaluation needs LDC Treebank-3 (LDC99T42).

### Gaps
- Whether ISLE or Georgia Tech holds an LDC licence for Treebank-3 was not checked.
- How Klein & Manning handled sentences with no non-trivial brackets under macro-averaging is not stated in their papers.
- QuestionBank's licence, the Pearl & Sprouse treebank's licence and OntoNotes' fee status were not verified.

## 6. Variable-arity chunks and templates

### Takeaway
Systems decide arity in three ways:
1. **Fix it at two.** CNF PCFGs, CCM, U-DOP and every neural PCFG do this, and pay a precision ceiling against PTB's flatter n-ary gold trees: 84.3 S-F1 for binarized gold, 88.1 on WSJ10.
2. **Let it emerge from flat proposals.** ABL alignments, ADIOS patterns, cascaded chunkers and CCL's common cover links produce n-ary or skewed structures. They trade recall for precision.
3. **Store multi-level fragments.** TSG elementary trees, DOP subtrees, adaptor- and fragment-grammar cache entries make a "chunk" a template with internal structure and open slots. How large a stored template is gets decided by reuse under a Pitman-Yor or DP prior, or by a shortest-derivation preference.

Option 3 is the principled answer to "a non-uniform number of components per chunk; several templates". The parsing primitive can stay binary while stored units have arbitrary shape.

### Cited Findings

**Fixed binary arity**
- Treebank trees "are generally flatter than binary", which limits any binary system. "Any all-binary system will over-propose constituents." [Source](https://aclanthology.org/P02-1017.pdf); [Source](https://aclanthology.org/P04-1061.pdf)
- The binarized-gold oracle on §23 is 84.3 S-F1 under right-branching binarization (82.1 on the ≤10-word subset). On WSJ10 the upper bound is 88.1 (precision 78.8, recall 100). [Source](https://arxiv.org/abs/1906.10225); [Source](https://aclanthology.org/P04-1061.pdf)
- CCM's chart forces binary trees. CCL and the cascaded chunkers "can predict higher-branching constituent structures, so fewer constituents are predicted overall". As a result CCM has higher recall and lower precision than raw-text models: on WSJ ≤10 words, CCM 62.4 / 81.4 vs PRLG cascade 74.6 / 66.7 (precision / recall). [Source](https://aclanthology.org/P11-1108/)

**Emergent n-ary structure**
- **CCL** (precision / recall / F1 on WSJ10): 75.6 / 76.2 / 75.9, with near-balanced precision and recall and a bracket count close to the gold count. [Source](https://aclanthology.org/P07-1049.pdf)
- **Ponvert's "constituent chunks"** are multiword gold constituents with no sub-constituents. They make up 32.9% of WSJ constituents and cover 57.7% of words, so flat chunks are a large share of English structure. [Source](https://aclanthology.org/P11-1108/)
- **ABL** was scored with non-crossing-bracket precision and recall: ATIS 85.31 / 89.31 vs right-branching 82.70 / 92.91. Under EVALB on ATIS it gets 39.2 F1 vs right-branching 42.9. [Source](https://aclanthology.org/C00-2139.pdf); [Source](https://aclanthology.org/P02-1017.pdf)
- **Flattening hurts as well as helps.** Flattening binary output raised precision and F1 for one system on Eve (to 70.31 / 74.33). The authors treated that comparison as not fair, so how the flattening is done matters. [Source](https://arxiv.org/abs/1802.08545)

**Multi-level fragments and templates**
- **DOP / U-DOP** store all subtrees of any depth. On CHILDES Eve, restricting fragments to depth 1 (equivalent to a PCFG) drops F1 from 88.7 to 49.5, and the best fragment depth grows with the child's age. [Source](http://web.archive.org/web/20180218230842/http://onlinelibrary.wiley.com:80/doi/10.1111/j.1551-6709.2009.01031.x/full)
- **Bayesian TSGs** learn how big each stored elementary tree should be (Pitman-Yor prior with a CFG base). A compact TSG reaches 84.0 F1 on §23 (supervised setting) vs 70.7 for a maximum-likelihood PCFG, with far fewer rules than full DOP. [Source](https://aclanthology.org/N09-1062/); [Source](https://jmlr.org/papers/v11/cohn10b.html)
- **Fragment grammars** store partial trees, i.e. templates with variables, and settle on "mid-sized" fragments. Adaptor grammars are the special case that stores only complete subtrees. [Source](https://dspace.mit.edu/handle/1721.1/44963)
- **Online adaptor grammars** rank cached fragments by Λ = f̃·log(ε·|s| + 1), which adds an explicit bonus for longer yields (|s| = number of yields). The bonus "discourage[s] short constituents". [Source](https://aclanthology.org/Q14-1036/)
- **No metric for arity itself was found.** The literature measures it only indirectly, through precision vs recall. [Source](https://aclanthology.org/2020.acl-main.300.pdf)

### Inferences
- **For TRELLIS v2, keep the parse primitive binary but let stored chunks be templates.** Binary content concepts keep inside-outside cheap and match the simple-PCFG factorization. A concept may additionally cache:
  - (a) flat n-ary yields, i.e. a chain of binary merges stored and generated as one unit;
  - (b) multi-level fragments with open slots, which are class-constrained variables filled through the context hierarchy.

  Whether a template is worth storing, and how deep, can be decided by the same reuse test used for graduation: Pitman-Yor reuse or ΔDL. That operationalizes "several templates per chunk".
- **Template flattening is a lever on commission (precision).** When outputting PTB brackets, TRELLIS can emit a stored flat template as one n-ary node instead of its internal binary chain. Report both binary and flattened scores, since the binarized-gold ceiling (84.3) is the main source of unavoidable commission.

### Gaps
- No direct empirical comparison was found of "binary primitive plus cached n-ary templates" against pure binary on PTB constituency from raw text.

## Implications for TRELLIS v2

### Takeaway
Across forty years the working recipe is the same.
1. Propose cheaply from local statistics: adjacent-pair frequency, alignment and substitutability, context distributions, or inside-outside span posteriors.
2. Accumulate evidence in a pool of candidates.
3. Commit a chunk or merge only when a global criterion improves: description length or posterior gain, a significance test, or Pitman–Yor reuse.

TRELLIS v1 collapses all three into one greedy, count-gated commit and, in unsupervised mode, feeds its own hard parses back. That is the regime where the literature reports lock-in, distituent chunks and attachment errors.

TRELLIS's machinery is already close to proven designs:
- its greedy bottom-up merge-and-replace parse is a cousin of CCL and Ponvert's chunk cascades (≈52–54 F on held-out full WSJ; ≈70–76 on WSJ10);
- its content/context duality is a hierarchical CCM (≈71–72 on WSJ10 with POS tags);
- its LEFT/RIGHT content concepts are simple-PCFG nonterminals (65.1 S-F1 at scale);
- GRIDS, its direct ancestor, already named "an incremental version … that processes only a few training sentences at a time" as the next step.

The recommended v2 loop, ranked options and PTB plan follow.

### Cited Findings (key facts the recommendations rest on; full details in Sections 1–6)
- **GRIDS** is "a rational reconstruction of Wolff's SNPR". It:
  - starts from a flat grammar of the training sentences;
  - alternates a *merge* mode (merge two nonterminals into a class; this can create recursion) with a *create* mode (new nonterminal for "pairs of nonterminal symbols that occur in sequence", substituted everywhere);
  - uses beam search with a beam of three;
  - scores the description length of the grammar plus the derivations of the training sentences.

  Its authors list the quadratic cost of merges and suggest "an incremental version of Grids that processes only a few training sentences at a time and expands the grammar as necessary". [Source](http://www.isle.org/~langley/papers/grids.ecml2k.pdf)
- **Goldsmith 2001:** "We develop a set of heuristics that rapidly develop a probabilistic morphological grammar, and use MDL as our primary tool to determine whether the modifications proposed by the heuristics will be adopted or not." [Source](https://aclanthology.org/J01-2001.pdf)
- **CCM** scores every span by its yield and its context, as constituent vs distituent. Context clustering alone needs an extra filter to separate constituents from distituents (Clark 2001). [Source](https://aclanthology.org/P02-1017/)
- **Clark's criterion:** "with real constituents, there is high mutual information between the symbol occurring before the putative constituent and the symbol after". It is used inside an MDL algorithm. [Source](https://aclanthology.org/W01-0713.pdf)
- **de Marcken:** greedy grouping by mutual information of adjacent items (Magerman & Marcus; Stolcke 1994) "will consequently fail to derive linguistically-plausible phrase structure in many situations". [Source](https://aclanthology.org/W95-0102/)
- **Online adaptor grammars:**
  - decayed sufficient statistics with ε = (τ + l)^−κ;
  - after each minibatch, add candidate fragments sampled from the base distribution;
  - every u minibatches, prune to the top K by Λ = f̃·log(ε·|s| + 1).
  [Source](https://aclanthology.org/Q14-1036/)
- **Pitman–Yor adaptor:** reuse cached tree k with probability (n_k − a)/(n + b); create a new one with probability (m·a + b)/(n + b). [Source](https://proceedings.neurips.cc/paper/2006/hash/62f91ce9b820a491ee78c108636db089-Abstract.html)
- **Shortest derivation:** choose the parse built from "the fewest corpus-subtrees", breaking ties by frequency. U-DOP's most probable parse "has a tendency to be constructed by the shortest derivation". [Source](https://aclanthology.org/C00-1011/); [Source](https://aclanthology.org/W06-2912/)
- **Stepwise online EM:** η_k = (k+2)^−α. **Viterbi EM** beats classic EM on long sentences but fails at Baby Steps. **Lateen EM** alternates the two (+5.5 points on average across 19 languages). [Source](https://aclanthology.org/N09-1069/); [Source](https://aclanthology.org/W10-2902/); [Source](https://aclanthology.org/D11-1117/)
- **Counts and sharing:** a count-based ("scalar") PCFG trained by EM scores below 35 S-F1 on PTB, under right-branching (39.5). Sharing (50.8), more symbols (57.7–65.1) and aggregation (≈70) lift it. [Source](https://arxiv.org/abs/1906.10225); [Source](https://arxiv.org/abs/2310.14997); [Source](https://arxiv.org/abs/2310.01717)
- **SimplePCFG:** π(A→BC) = π(B↶A)·π(A↷C); inside cost O(l³|N| + l²|N|²). [Source](https://arxiv.org/abs/2310.14997)
- **Parse-focusing** restricts each sentence's parse pool to a few candidate trees from other unsupervised parsers: 69.6±0.6 S-F1, with lower variance and less simplicity bias. [Source](https://arxiv.org/abs/2407.16181)
- **Tree averaging** of heterogeneous parsers: +7.5 over the best teacher. [Source](https://arxiv.org/abs/2310.01717)
- **Indirect negative evidence:** reject hypotheses whose generated short strings are unsupported by the data. [Source](https://arxiv.org/abs/2312.15321)
- **Ponvert's cascade:** chunk with an HMM or probabilistic right-linear grammar (PRLG), replace each chunk by a pseudoword, re-chunk for 5–7 levels. It gets 54.2 F on held-out WSJ §23 at all lengths, vs CCL 51.7. Without phrasal punctuation it loses about 10 points of precision. [Source](https://aclanthology.org/P11-1108/)
- **CCL** (online, greedy, raw words): WSJ10 75.9, WSJ40 57.4 UF1. [Source](https://aclanthology.org/P07-1049/)
- **Stolcke's incremental model merging.**
  - Loop: incorporate a few samples, then merge until the posterior drops. "Between 1 and 10 samples at a time" are good batch sizes, and merging should not start before about 10–20 samples.
  - It is "the default method used in all the experiments" for SCFGs.
  - Ordering samples by length cut merges from 1,374 to 212 on the adjective grammar.
  [Source](http://web.archive.org/web/20070611231208/http://www.icsi.berkeley.edu/ftp/global/pub/ai/stolcke/thesis.ps.Z)
- **Chen (1995).** Moves are only considered when "triggered" in the sentence currently being parsed, and each sentence is parsed "but once". [Source](https://aclanthology.org/P95-1031/)
- **ADIOS.** Patterns are promoted by a significance test (decrease ratio < η, P < α) and then "immediately and irreversibly rewired". The resulting order dependence is mitigated with cohorts of learners trained on permuted corpora. [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187953)
- **U-MILA.** Learns from every token. It creates chunks when "sanctioned by Barlow's (1990) principle of suspicious coincidence", and exponential decay lets errors "decay and eventually become negligible". [Source](http://web.archive.org/web/20240508093013/https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/KolodnyEtAl2015_LangAcqProcessLevel.pdf)
- **e-GRIDS.** Under its description-length code, the M best create-successors come from the M most frequent bigrams, "totally equivalent to the exhaustive enumeration". [Source](http://web.archive.org/web/20070710200647/http://www.iit.demokritos.gr:80/~paliourg/papers/GRAMMARS2004.pdf)
- **Clark.** The description-length gain of rewriting a pair P R equals the pointwise mutual information of P and R. His algorithm is "very sensitive to the order in which the rules are acquired". [Source](https://aclanthology.org/W01-0713/)
- **Goldsmith's triage.** Description-length-based deletion wrongly removed 21.9% of correct changes, so an extra non-MDL threshold was needed. [Source](https://aclanthology.org/J01-2001/)

### Inferences

#### A. What each TRELLIS piece corresponds to in the literature
| TRELLIS element | Closest prior mechanism | Consequence for v2 |
|---|---|---|
| Content concept with LEFT/RIGHT child-class attributes, independent given the concept (`create_content_instance` in `src/parse_mh.py`) | Simple-PCFG nonterminal p(B,C\|A) = p_L(B\|A)·p_R(C\|A) | Inside-outside over the content hierarchy is tractable for thousands of concepts. This representation class has reached 65.1 S-F1. |
| Context hierarchy (span neighbors) | CCM's P(context \| constituent/distituent); Clark's distributional clusters; ADIOS/EMILE equivalence classes | TRELLIS is a hierarchical, incremental CCM. What is missing is a distituent model or an MI/context filter. |
| Climbing-ancestor gate (back off along the Cobweb path) | The rule sharing and smoothing that lifted PCFGs from below 35 to 50.8 S-F1 | Use the path as a probability back-off (mixture over ancestors), not only as a yes/no gate. |
| Count threshold τ for graduation | ADIOS significance test; U-MILA decay and promotion; Pitman–Yor reuse | Replace τ with a scale-free significance or reuse test that transfers from 3-nonterminal toy grammars to PTB. |
| Self-parses fed back (v1 unsupervised mode) | Viterbi/hard EM, self-training | Hard feedback locks in early bracketings. Soft or top-k evidence plus a Lateen-style veto is the likely fix for 46% on the attachment-ambiguous grammar. |
| `MERGE_POLICY` frequency merges ("BPE/GRIDS-style") | MK10/SNPR folding, Sequitur digram rules, BPE, Magerman & Marcus MI grouping | Frequency proposes frequent distituents (de Marcken's "edby"; IN DT). Keep it for proposing; accept only with context-based and description-length tests. |
| Two-part MDL that condensed categories but hurt generation | GRIDS/SNPR/Goldsmith MDL; Park & Kim's "structural simplicity bias"; GRIDS commission curves | Over-merging is commission. Add a generation-based commission check and compute data cost over all parses, not one Viterbi parse. |
| `generate_via_chunk_replay` (per-leaf chunk pools) | Adaptor-grammar cache sampling; DOP fragment reuse | v1's generator is already a cache sampler. Pitman–Yor probabilities give a principled mix of replaying a stored chunk and composing a new one. |
| Greedy merge-and-replace parse; set-aside CKY (0.93–0.997 on toy grammars) | Ponvert cascade (chunk → pseudoword → re-chunk), CCL; MBR/CYK decoding | The greedy family tops out around 52–54 F on held-out WSJ with punctuation cues. Global posteriors with MBR add about 1 point in neural PCFGs. Posterior-guided easy-first decoding keeps the greedy character while using global evidence. |

#### B. Ranked options for an unsupervised, incremental chunk-learning loop

**1. Inside-outside proposals → decaying frontier → ΔDL acceptance with a significance floor (core loop; highest expected value).** This realizes "keep candidate parses in a frontier and learn them once confirmed", in the incremental form GRIDS's authors anticipated.

*(a) Parse* each incoming sentence with inside-outside over the grammar read off the hierarchies:
- content concepts at the maturity cut are the nonterminals, with p_L / p_R taken from their child-class attributes;
- the context class of a span's neighbors acts as a CCM-style P(context | constituent) factor;
- unseen pairings back off to a "compose anew" probability at the root or basic level.

Decode with MBR (CKY over the sum of span posteriors), or with posterior-guided easy-first decoding.

*(b) Accumulate.* Each span adds posterior-weighted evidence μ(i,j) to a frontier entry keyed by its content signature (child classes plus boundary words). Entries decay as in online adaptor grammars, f̃ ← (1−ε)f̃ + ε·counts. Nothing enters long-term Cobweb memory yet, so one-off spans fade instead of becoming symbols.

*(c) Test.* When an entry passes a significance floor (an ADIOS-style significance test, or observed vs. expected co-occurrence of its child classes), compute a local ΔDL as GRIDS, Stolcke–Omohundro and Chen do:
- the grammar-code cost of the new or updated concept,
- minus the data-code saving estimated from cached expected counts.

Commit (Cobweb create/add) iff ΔDL < 0. Every few minibatches, prune frontier entries with low usefulness, as in the online adaptor-grammar ranking.

A first-cut ΔDL for a pair chunk follows from Clark's derivation:
- data-code saving ≈ n_PR · PMI(P,R), using (expected) counts of the pair P R;
- ΔDL ≈ grammar-code cost of the new concept − n_PR · PMI(P,R).

Estimate PMI at the context-class level, so that counts are pooled across the hierarchy, and compare it with its expected value at that span length (Clark's sparse-data correction).

*(d) Update* parameters with stepwise EM.
- Cobweb's `increment_counts` hard-codes `count += 1` per instance, while attribute counts add the instance's own (possibly fractional) values (`cobweb-private/src/cobweb_discrete_node.cpp:97–113`).
- Posterior-weighted incorporation therefore needs a weight argument. Otherwise, sample k trees from the inside chart and incorporate each.

**2. Replace τ with Pitman–Yor caching to decide what is stored as a chunk.**
- Reuse a stored chunk with probability ∝ (n_k − a). Compose anew with probability ∝ (b + aK) × the content-hierarchy base probability.
- "Minimize the number of chunks" then follows from rich-get-richer plus a discount, with no hand-tuned threshold, and generation (reuse vs. compose) falls out of the same model.
- This can be the data-cost model inside option 1's ΔDL.
- Watch-out: incremental initialization finds high-F but low-posterior solutions built on short common substrings (Johnson & Goldwater 2009). Use the length bonus and table resampling, or periodic re-analysis.

**3. A distituent / context filter via the context hierarchy (CCM + Clark).**
- Accept a candidate only if its contexts behave like those of a single symbol: high mutual information between left and right neighbors (Clark), or categorization into a context concept that also hosts single words or known constituents.
- Model distituents explicitly in the span score, as CCM does.
- This is the main defence against frequency-driven distituent chunks.

**4. A parse frontier with shortest-derivation tie-breaking and pool restriction.**
- Keep the top-K parses per sentence (beam or chart samples). Prefer the most probable parse among those that use the fewest stored chunks (Bod's most probable shortest derivation).
- Train only on that pool, as in parse-focusing: lower variance, less simplicity bias.
- This is the per-sentence counterpart of corpus-level description length.

**5. Order ensembles and curriculum.**
- Run k learners on shuffled streams. Combine their trees with CYK tree averaging (MBR over span agreement) and re-train on the averaged trees: +7.5 for neural teachers.
- Feed shorter sentences first, then lengthen (Baby Steps / Compound PCFG); ablate this against Rohde & Plaut's evidence.
- Optionally bound center-embedding depth (depth-bounded PCFG) for plausibility and a smaller search space.

**6. Generation as negative evidence (constituency tests; Potashnik).**
- Before committing a merge, use TRELLIS's generator to (a) substitute same-context-class alternatives into the candidate span and (b) emit short strings.
- Penalize the merge if the outputs are unattested or improbable. This targets the commission blow-up that v1's MDL caused, and it is cognitively plausible (produce, then check).

**7. Variable arity through cached templates.**
- Keep the binary primitive. Let concepts cache n-ary flat yields or multi-level fragments with class-constrained slots ("several templates") when Pitman–Yor reuse or ΔDL justifies storage.
- Emit flattened templates as n-ary PTB brackets, and report binary vs flattened scores.

**8. Diagnostic only: LLM paraphrase signals (SemInfo, span-overlap).**
- These add 7–13 S-F1 to PCFGs, but conflict with the cognitive-plausibility pillar. Use them only to estimate headroom.

#### C. How the user's "frontier, then learn once good enough" idea lines up with prior systems
| System | What sits in the "frontier" | Promotion test | Forgetting / undo |
|---|---|---|---|
| Stolcke–Omohundro (incremental) | newly incorporated, sample-specific rules (batches of 1–10 samples) | merge/chunk while the posterior (description-length prior × likelihood) improves; no merging before about 10–20 samples | unchunk/rechunk; λ weight against early overgeneralization |
| Chen 1995 | moves "triggered" by the current sentence's Viterbi parse | greedy Bayesian (2^−l(G)) gain over all sentences so far | none (an inside-outside post-pass re-estimates probabilities) |
| ADIOS | candidate path bundles (MEX) | decrease ratio < η and significance α; the most significant pattern is rewired | none ("irreversibly rewired"); cohorts over sentence orders instead |
| U-MILA | recurrences in a 50–300-token decaying short-term memory | Barlow's suspicious coincidence, subject to a prior | exponential decay of all weights |
| Clark 2001 | frequent tag-sequence clusters | MI above its expected value (pooled per cluster), then best description-length reduction | none (order-sensitive) |
| Goldsmith 2001 | heuristic analyses (take-all-splits, weighted MI) | adopt iff description length decreases | triage (plus an extra non-MDL threshold) |
| Online adaptor grammars | fragments sampled from the base distribution | Pitman–Yor reuse; decayed counts f̃ | prune all but the top K by Λ = f̃·log(ε\|s\|+1) every u minibatches |
| Parse-focusing / tree averaging | a small pool of candidate parses per sentence (other parsers' trees) | train only on the pool / MBR consensus | — |

Any of the promotion tests can sit on top of a Cobweb-backed frontier. The one with the fewest new hyperparameters, and therefore the recommended one, is a significance floor (ADIOS / suspicious coincidence) plus a ΔDL < 0 test (Stolcke / Goldsmith / GRIDS), with forgetting by decay (U-MILA / online adaptor grammars).

#### D. Recommended PTB evaluation plan
1. **Build and validate the evaluator first.**
   - Implement unlabeled span precision/recall exactly as `harvardnlp/compound-pcfg` `eval.py` does: drop width-1 and whole-sentence spans, skip length-1 sentences, compute both sentence-level and corpus-level scores.
   - Reproduce on §23: right-branching 39.5 sentence / 36.1 corpus, left 8.7, binarized oracle 84.3.
   - Report omission = 1 − recall and commission = 1 − precision, with the harmonic mean for comparability, labelled "what other papers call unlabeled F1".
   - Add a non-crossing-bracket diagnostic to separate compatible-but-different binarization from real attachment errors.
2. **Track 0, regression.** The three synthetic CFGs plus Langley & Stromsten's and Tu & Honavar's grammars, reporting bracket-level *and* language-level omission/commission as in GRIDS.
3. **Track A: WSJ10 with gold POS** (7,422 sentences, train = test). Compare with CCM 71.9 (71.1 under the 2002 metric), DMV+CCM 77.6, U-DOP 78.5, UML-DOP 82.9, CCL 75.9 (words, with punctuation cues), right-branching 61.7 (Klein & Manning metric) and upper bound 88.1. This isolates the chunking machinery from category induction.
4. **Track B: full WSJ from words**, following Kim et al.:
   - splits 02–21 / 22 / 23; punctuation stripped, lowercased, 10K vocabulary;
   - S-F1 as primary, corpus F1 as secondary;
   - at least 4–5 stream orderings, reporting mean ± std and max;
   - results by length bucket (≤10 / 20 / 30 / 40 / all);
   - recall per gold label (NP, VP, PP, SBAR, ADJP, ADVP);
   - self-F1 across orderings.

   Comparators: right-branching 39.5, PRPN/ON 47–48, Compound PCFG 55.2, TN-PCFG 57.7, SN-PCFG 65.1. Also report the held-out Ponvert / CCL setting (WSJ §23 at all lengths, raw text: PRLG 54.2, CCL 51.7), since TRELLIS is closest to those learners.
5. **Evaluate the context hierarchy as a POS inducer** with many-to-one accuracy. Compound PCFG's preterminals reach 68.0.
6. **Generation on PTB.**
   - Language-level commission: grammaticality of samples, from human ratings or minimal-pair suites such as BLiMP or Marvin & Linzen. Beware the "overly simple utterances" trap.
   - Language-level omission: coverage of §23, the fraction of sentences that get a complete parse.
7. **Model selection without labels.** Choose hyperparameters by description length or held-out likelihood, not §22 F1, or state clearly if §22 trees were used (Shi et al.).
8. **Data.** PTB requires an LDC Treebank-3 licence (LDC99T42). Before that, develop on the NLTK sample: 3,914 WSJ trees from files 0001–0199 (likely sections 00–01), non-commercial fair use. Do not compare those numbers with §23 results.

#### E. What to expect (inference)
- **Unsupervised v2 on full §23 from words:**
  - a naive first version: 35–45 S-F1. Count-based EM is below 35, and the greedy cascades without punctuation lose about 10 precision points.
  - with hierarchy back-off, a distituent filter, soft frontier evidence and ensembling: 45–58 is plausible, i.e. CCL/Ponvert/PRPN level up to Compound PCFG level.
  - above about 60 has so far needed thousands of symbols, external teachers, or pretrained-LM signal.
- **WSJ10 with POS tags:** 60–75 is plausible; CCM/CCL-level (72–76) is the stretch goal.
- **Supervised TRELLIS on PTB** is a useful ceiling check. A 30-nonterminal PCFG trained only on gold parses reaches 73.7 S-F1, against the 84.3 binary-tree ceiling. So expect 70–80 even with supervision, unlike the 0.92–0.97 bracket agreement on the synthetic grammars.

### Gaps
- No prior system combines Cobweb-style incremental concept formation with inside-outside span posteriors, so the expected-result ranges above are extrapolations from neighbouring systems, not measurements.
- Whether Cobweb's category utility, used as the acceptance criterion, behaves like ΔDL or posterior gain was not found in the literature. This needs an empirical check.
- Whether ISLE or Georgia Tech already holds an LDC licence for Treebank-3 is unknown.
