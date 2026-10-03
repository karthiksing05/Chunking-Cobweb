# Information-Theoretic Foundations for Learning a Globally Good, Minimally Viable Grammar of Chunks and Concepts, Incrementally (guidance for TRELLIS v2)

Scope: Kolmogorov complexity, Solomonoff induction, MDL/MML, prequential coding, grammar-based compression, MDL grammar/lexicon/morphology induction, MDL pattern mining and library learning, simplicity in cognition, and how all of this connects to Cobweb's category utility. The aim is to explain how to score (a) a candidate chunk, (b) a whole grammar held in TRELLIS's two linked Cobweb hierarchies (content = compositions/rewrite rules; context = distributional classes), and (c) incremental updates, using tractable stand-ins for "global optimality".

Conventions: L(.) is a codelength in bits (log base 2). D is the sentence stream s_1..s_N. G is the grammar, meaning the content tree C plus the context tree X plus a "cut" (the set of active nodes) through each. Grammar quality is discussed throughout as errors of omission (too specific) and commission (too general), after Langley & Stromsten (2000). Facts about TRELLIS itself come from the project brief and from the ACS-26 paper source in `confs/acs-26/paper/main.tex` ("A Unified Account of Concepts and Chunks: ...").

Research caveat: this session's shared web-search budget ran out partway through. Later bibliographic checks therefore used CrossRef, OpenAlex, Europe PMC, arXiv, ACL Anthology, JMLR and dblp lookups, plus full-text extraction of primary PDFs. Coverage of 2025–2026 work is thinner than coverage of the classic literature (see the Gaps sections).

---

## Q1. Formal foundations: Kolmogorov complexity, Solomonoff's universal prior, Levin search / Kt, the speed prior, and the structure function. Is there an incremental reading?

### Takeaway
Algorithmic information theory gives the ideal objective: the shortest total description, or equivalently a Bayes mixture over all computable hypotheses weighted by 2^-K. It also gives a strongly incremental reading. A sequential mixture's total excess prediction error is bounded by the prior codelength of the true source. Solomonoff's own "incremental learning" system updates its prior after every solved problem by adding the newly found definitions (concepts), so later problems get shorter descriptions. This is very close to "preserving incremental additions" in TRELLIS: a learned chunk becomes part of the reference language for everything that follows. All of these ideal quantities are incomputable. The usable stand-ins are resource-bounded versions (Kt, the speed prior), class-restricted versions (MDL over grammars, which turns undecidable into merely NP-hard), and compressor-based versions.

### Cited Findings
**Solomonoff's foundations**
- Solomonoff's "A Formal Theory of Inductive Inference" appeared in two parts in *Information and Control*: Part I in 7(1):1–22 (March 1964) and Part II in 7(2):224–254 (June 1964) — [Solomonoff bibliography, RHUL](https://cml.rhul.ac.uk/publications/solomonoff).
- Part I presents four models of induction framed as extrapolating long symbol sequences. One is equivalent to a Bayes formulation in which a priori probabilities come from the lengths of inputs to a universal Turing machine — [ML Anthology entry](https://mlanthology.org/misc/1964/solomonoff1964misc-formal).
- The grammar paper the brief dates to 1959 is listed by Solomonoff's own bibliography as "A New Method for Discovering the Grammars of Phrase Structure Languages," *Information Processing*, UNESCO, Paris, **1960** — [RHUL bibliography](https://cml.rhul.ac.uk/publications/solomonoff); [Solomonoff publications page](http://raysolomonoff.com/publications/pubs.html). Its PDF there is a scanned image (not text-searchable), so its content was not verified directly.
- Chen (1995) summarizes the lineage as follows. Solomonoff (1964) "presents a Bayesian grammar induction framework" with a factor favoring smaller grammars, and proposes the universal a priori probability (Solomonoff 1960), with p(G) = 2^-l(G), where l(G) is the grammar's description length in bits. Chen notes that this prior "dominates all other enumerable probability distributions multiplicatively" — [Chen 1995, ACL / arXiv cmp-lg/9504034](https://arxiv.org/abs/cmp-lg/9504034).
- Solomonoff, "Complexity-Based Induction Systems: Comparisons and Convergence Theorems," *IEEE Trans. Inf. Theory* IT-24(4):422–432 (1978) — [RHUL bibliography](https://cml.rhul.ac.uk/publications/solomonoff).

**Sequential (incremental) reading of Solomonoff induction**
- The Bayes mixture is ξ(x) = Σ_ν w_ν ν(x), with posterior weights w_ν(x_1:n) = P(H_ν | x_1:n) updated as data arrive. For a stochastic true environment μ, Σ_t E[h_t] ≤ ln(1/w_μ) < ∞, where h_t is the squared Hellinger distance between the predictive distributions of ξ and μ. So "ξ(x_t|x_<t) will rapidly converge to μ(x_t|x_<t) with probability one." The universal prior sets w_ν = 2^-K(ν). The main obstacle "remains its incomputability and the difficulty of approximating Solomonoff" — [Rathmanner & Hutter 2011, *Entropy* 13(6):1076–1136 (arXiv 1105.5721)](https://arxiv.org/abs/1105.5721).
- Scholarpedia's "Algorithmic probability" article (Hutter, Legg & Vitányi, 2007) covers Levin's coding theorem and the maximality and semicomputability of the universal distribution — [Scholarpedia 2(8):2572](http://www.scholarpedia.org/article/Algorithmic_probability).

**Solomonoff's explicitly incremental learners (the closest precedent for preserving incremental additions)**
- "A System for Incremental Learning Based on Algorithmic Probability," *Proc. 6th Israeli Conf. on AI, Computer Vision and Pattern Recognition*, pp. 515–527 (Dec. 1989) — [RHUL bibliography](https://cml.rhul.ac.uk/publications/solomonoff). Its text says: "The machine starts out ... with a set of primitive concepts... In solving the first problem, it acquires new concepts that help it solve the more difficult second problem, and so on." It also says that the use of Levin's search algorithm "guarantees that any describable concept will eventually be discovered" — [Solomonoff 1989 PDF](http://raysolomonoff.com/publications/IncLrn89.pdf).
- "Progress in Incremental Machine Learning" (NIPS 2002 workshop; rev. 2003) gives the update loop. After each problem, the Update Algorithm modifies the problem-solving techniques, possibly adding or deleting some, and modifies the guiding probability distribution. "After we have solved the first problem, our a priori probability distribution changes. It includes the definition for F₁(·)" (the function that solved the first problem). Plain Levin search "is only practical for simple problems, since its solution time is exponential in the complexity of the solution", but updating the guiding distribution "effectively reduces the complexity of the solutions of initially difficult problems." The paper stresses training sequences of problems that increase in difficulty — [Solomonoff 2003 PDF](http://raysolomonoff.com/publications/nips02.pdf).
- Schmidhuber's Optimal Ordered Problem Solver (OOPS), *Machine Learning* 54:211–254 (2004), extends Levin's universal search into an incremental learner. It spends part of its search time on programs that call or copy-edit earlier frozen solutions, and in experiments the reuse sped up universal search by a factor of about 1000 — [arXiv cs/0207097](https://arxiv.org/abs/cs/0207097); [ML Anthology](https://mlanthology.org/mlj/2004/schmidhuber2004mlj-optimal).

**Resource-bounded variants (Levin Kt, speed prior)**
- The speed prior (Schmidhuber, COLT 2002) is like Kolmogorov complexity but counts computation time as well as program length: the complexity of a program is its size in bits plus the log of the maximum time it is run. It yields computable, near-optimal predictions at some cost in optimality — [ML Anthology](https://mlanthology.org/colt/2002/schmidhuber2002colt-speed); [Wikipedia: Speed prior](https://en.wikipedia.org/wiki/Speed_prior).

**Structure vs. noise: the Kolmogorov structure function and algorithmic sufficient statistics**
- Algorithmic statistics uses two-part codes, "the code for the statistic (the model summarizing the regularity, the meaningful information, in the data) and the model-to-data code." It defines (minimal) sufficient statistics for individual data — [Gács, Tromp & Vitányi 2001, *IEEE TIT* 47(6):2443–2463](https://doi.org/10.1109/18.945257).
- The structure function "determines all stochastic properties of the data" and identifies the best-fitting model in a constrained class "irrespective of whether the 'true' model is in the model class considered or not" — [Vereshchagin & Vitányi 2004, *IEEE TIT* 50(12) (arXiv cs/0204037)](https://arxiv.org/abs/cs/0204037).
- Ideal MDL can be derived from Bayes' rule with the universal prior. When the model class is restricted to finite sets it becomes Kolmogorov's minimal sufficient statistic, and "data compression is almost always the best strategy, both in model selection and prediction" — [Vitányi & Li 2000, *IEEE TIT* 46(2):446–464](https://doi.org/10.1109/18.825807).
- If you approximate the optimal two-part code by successively shorter two-part codes: (i) each step may take arbitrarily long; (ii) you may never know when you have reached the optimum; (iii) the sequence of models "may not monotonically improve the goodness of fit"; but (iv) the model at the optimum has almost the best fit — [Adriaans & Vitányi 2009, *IEEE TIT* 55(1):444–457 (arXiv cs/0612095)](https://arxiv.org/abs/cs/0612095).

**Computable approximations**
- "Weakening the model from Turing machines to context-free grammars reduces the complexity of the problem from the realm of undecidability to mere intractability." The smallest-grammar problem is "a natural, but more tractable variant of Kolmogorov complexity" — [Charikar et al., STOC 2002 version](https://compression.ru/download/articles/grammar/charikar_2002_approximating_the_smallest_grammar.pdf).
- Normalized compression distance replaces the incomputable normalized information distance with real compressors and feeds a hierarchical clustering (quartet-tree) method. In other words, concept hierarchies can be built from compression alone — [Cilibrasi & Vitányi 2005, *IEEE TIT* 51(4):1523–1545](https://doi.org/10.1109/tit.2005.844059).
- "Predictive models can be transformed into lossless compressors and vice versa" (arithmetic coding). Large language models act as general-purpose compressors — [Delétang et al., "Language Modeling Is Compression" (arXiv 2309.10668)](https://arxiv.org/abs/2309.10668).
- Reference text: Li & Vitányi, *An Introduction to Kolmogorov Complexity and Its Applications*, 4th ed., Springer 2019 — [Springer](https://link.springer.com/book/10.1007/978-3-030-11298-1).

### Inferences
- **The incremental reading is the prequential one.** A sequential Bayes or MDL learner pays −log P(s_t | past) for each sentence. Its total excess codelength over the truth is bounded by the prior codelength of the truth: ln(1/w_μ), which is about K(μ)·ln 2 under the universal prior. For TRELLIS this means **a grammar that is learned incrementally and scored by its running predictive codelength pays for its structure once, implicitly, through early mistakes**. That is the formal sense in which incremental additions are "preserved": every chunk the learner keeps must have paid for itself in cumulative predictive bits.
- **"Library = learned reference machine."** The invariance theorem (Li & Vitányi) says the choice of universal machine changes K only by an additive constant. At finite N that constant is everything. Learning chunks and concepts amounts to learning a better reference language, in which later sentences, and later candidate chunks built from earlier ones, are short. Solomonoff (1989/2003), OOPS, and DreamCoder (Q5) all implement this "solutions become prior" loop. It matches the user's observation that "likely grammars influence the proposition of similarly likely grammars" (Goldsmith).
- **The structure function formalizes "minimally viable."** The knee of the structure function (the minimal sufficient statistic) separates the grammar (regularities) from accidental detail. In practice, a TRELLIS grammar should stop growing when extra chunks only encode corpus-specific accidents, that is, when their model cost exceeds the data saving. Vereshchagin & Vitányi's "irrespective of whether the true model is in the class" supports the user's position that TRELLIS's grammar need not match the linguist's grammar, provided it is the best compressor within TRELLIS's own representational class. The misspecification caveats in Q2 qualify this.
- **Adriaans & Vitányi warn against naive monotone search.** Lowering total codelength step by step does not guarantee that each intermediate grammar generalizes better. Incremental TRELLIS should expect non-monotone interpretability along the way, and should judge final grammars rather than every intermediate one.
- **Kt and the speed prior suggest a time term.** A cognitive parser pays for search. A Kt-style score, L(G) + L(D|G) + log2(parse time), would penalize chunk inventories that save bits but slow greedy matching. This is the information-theoretic form of Minton's "utility problem" (Q5).

### Gaps
- Could not verify the content of Solomonoff's 1960 grammar paper (scanned PDF only); its role is attested here only through Chen (1995).
- Levin's 1973 "Universal sequential search problems" and the original definition of Kt were not fetched from a primary source. CrossRef does not index the translated journal, so Kt is described here via the speed-prior literature and Solomonoff's descriptions of Lsearch.
- No source was found that develops an *incremental* structure function (how the minimal sufficient statistic evolves as data stream in). This looks like open territory.

---

## Q2. MDL and MML: two-part vs. refined (NML) vs. prequential codes, Bayesian equivalences, and known pitfalls

### Takeaway
MDL comes in three practical forms. **Two-part** codes, L(model) + L(data | model), are intuitive and interpretable but depend on arbitrary choices of code and parameter precision. **NML** (refined) codes are minimax-optimal but hard to compute and need the sample size in advance. **Prequential** codes, the cumulative online log-loss Σ −log p(x_i | x_<i), are naturally incremental and need no explicit model code. For multinomial count models with Krichevsky–Trofimov (Jeffreys) smoothing, the prequential code coincides *exactly* with the Bayesian marginal likelihood. That means it can be computed in closed form from a node's counts, which is exactly what Cobweb stores. The main pitfalls are crude model codes, the treatment of parameter precision, and misspecification. Under misspecification, Bayes and MDL can become inconsistent, and a learning-rate (η < 1) "Safe Bayes" correction is the proposed fix.

### Cited Findings
**Origins and definitions**
- Rissanen (1978) introduced MDL: "By finding the model which minimizes the description length one obtains estimates of both the integer-valued structure parameters and the real-valued system parameters" — [*Automatica* 14(5):465–471](https://doi.org/10.1016/0005-1098(78)90005-5). Rissanen (1984) tied universal coding to prediction and estimation and constructed optimal universal codes — [*IEEE TIT* 30(4):629–636](https://ieeexplore.ieee.org/document/1056936). Rissanen's 1989 monograph is *Stochastic Complexity in Statistical Inquiry* (World Scientific) — [DOI 10.1142/0822](https://doi.org/10.1142/0822); CrossRef lists a 1998 date, probably a reprint.
- Dawid (1984), "Statistical Theory: The Prequential Approach," *JRSS A* 147(2):278–292, proposed evaluating statistical methods by the sequence of probability forecasts they produce — [JSTOR](https://www.jstor.org/stable/2981683); [PDF copy](https://www.cs.ubc.ca/~murphyk/MLRG/dawid84Prequential.pdf).
- The standard textbooks are Grünwald, *The Minimum Description Length Principle* (MIT Press, 2007) — [book page](https://homepages.cwi.nl/~pdg/book/book.html) — and Wallace, *Statistical and Inductive Inference by Minimum Message Length* (Springer, 2005) — [Springer](https://www.springerprofessional.de/en/statistical-and-inductive-inference-by-minimum-message-length/915110).

**The four universal codes (from Grünwald & Roos 2019, "Minimum Description Length Revisited")** — [arXiv 1908.08484](https://arxiv.org/abs/1908.08484)
- *Two-part code* [Rissanen 1978]: discretize the parameter space to a countable grid, put a probability mass function w on it, and code the data with p₂ₚ(zⁿ) = max_θ p_θ(zⁿ)·w(θ). This is "historically the oldest universal distribution" and is "still important in practice."
- Discretization "makes things (unnecessarily, as was gradually discovered over the last 30 years) very complicated." Combining the choice of θ with the choice of model γ "introduces some suboptimalities." One-part codes avoid both problems. Even so, for the discrete *structure* γ "it is quite reasonable to choose a probability mass function," and designing it by thinking about codelengths −log π(γ) "comes very naturally."
- *Prequential plug-in code* [Rissanen 1984; Dawid 1984]: p_preq(zⁿ) = Π_i p_θ̆(z^{i−1})(z_i | z^{i−1}). With discrete data the raw ML plug-in should be avoided because zero probabilities are possible; use a smoothed estimate. For Bernoulli with θ̆ = (m₁ + 1/2)/(m + 1), p_preq "turns out to coincide exactly with p_bayes with Jeffreys' prior". This exact coincidence is "a special property of the Bernoulli and multinomial models."
- Fundamental identity: Σ_i −log p̄(z_i | z^{i−1}) = −log p̄(zⁿ). "Every probability distribution defines a sequential prediction strategy and ... vice versa." MDL is "quite similar in spirit to cross-validation," with the cross replaced by a forward.
- NML minimizes the worst-case regret, and its parametric complexity equals the minimax regret. Prequential plug-in codes are used when NML is "too difficult" or when the horizon n is unknown.
- With the right priors, estimators or luckiness functions, all the universal codes reach logarithmic worst-case regret. For prequential plug-in, the standard (k/2) log n behavior holds only in expectation when the data come from the model. Under misspecification a variance-ratio correction appears, and a "flattened" hybrid estimator restores the standard asymptotics.
- *Misspecification*: MDL and Bayesian inference "can become inconsistent" and may keep selecting a suboptimal model as n grows. The root cause is the failure of "no-hypercompression" when the true distribution lies outside the model class. The proposed remedy replaces likelihoods with p_θ^η (η < 1), with η learned by the Safe-Bayesian algorithm.
- Original misspecification counterexample: [Grünwald & Langford 2007, *Machine Learning* 66:119–149 (arXiv math/0406221)](https://arxiv.org/abs/math/0406221). In a natural regression setting, "the posterior puts its mass on worse and worse models of ever higher dimension" because of "hypercompression," and SafeBayes "tends to select small learning rates ... as soon as hypercompression takes place" — [Grünwald & van Ommen 2017, *Bayesian Analysis* 12(4)](https://doi.org/10.1214/17-ba1085).

**MML and classification**
- Wallace & Boulton (1968) treat "a classification as a method of economical statistical encoding of the available attribute information." Their measure compares classifications or drives a classification procedure, and was implemented as SNOB — [*Computer Journal* 11(2):185–194](https://doi.org/10.1093/comjnl/11.2.185).
- MML parameter precision is handled by coding estimates to optimal precision ("compact coding") — [Wallace & Freeman 1987, *JRSS B* 49(3):240–252](https://doi.org/10.1111/j.2517-6161.1987.tb01695.x) (CrossRef pages 240–252).

**Prequential codelength in modern ML (incremental MDL in practice)**
- Deep networks "can compress data losslessly even when taking the cost of encoding the parameters into account," as shown with prequential coding. Variational methods give "surprisingly poor compression bounds" — [Blier & Ollivier, NeurIPS 2018 (arXiv 1802.07044)](https://arxiv.org/abs/1802.07044).
- MDL probing recasts probe training as transmitting labels given representations, measuring description length rather than accuracy — [Voita & Titov, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.14).
- In MDL pattern mining, most methods use crude two-part MDL at the top level "because the aim is not just to know how much the data can be compressed, but how that compression is achieved." Prequential plug-in codes "avoid unwanted bias arising from arbitrary choices in the encoding"; NML "is optimal for fixed sample sizes" but "can be challenging to compute, or even downright infeasible" — [Galbrun 2022, *DMKD* 36(5):1679–1727 (arXiv 2007.14009)](https://arxiv.org/abs/2007.14009).

### Inferences
- **Why the earlier TRELLIS two-part MDL over-generalized** (16 → 5 categories, 177 → 104 productions, generation grammaticality 0.82 → 0.62, per the brief). Several mechanisms predicted by the literature could each produce this:
  1. **The data term must be −log P(sentence | G) under the *same* normalized probabilistic model that generates.** An over-general grammar spreads probability onto unattested strings, so every observed sentence gets less mass and L(D|G) rises linearly with N. That is how MDL detects commission errors from positive data alone (see Chater & Vitányi and Hsu & Chater in Q6). If the code counted production *uses* along fixed parses, or if TRELLIS's generator (ancestor pools filtered by context class) is a different distribution from the one being coded, the commission penalty is lost.
  2. **Crude parameter and structure codes.** Arbitrary bits per category or production, and the treatment of smoothing mass, can tip the balance. The Grünwald & Roos critique of discretized two-part codes applies directly.
  3. **Small N.** The optimal MDL grammar is coarser at small N. More data justifies more distinctions.
  4. **Misspecification.** A binarized, context-free TRELLIS grammar is misspecified for natural language, which is the regime where Bayes and MDL can "hypercompress" toward bad models. A tempered likelihood (η < 1) is a principled hedge.
- **Prequential codes fit Cobweb especially well.** Each Cobweb node stores value counts. With KT/Jeffreys smoothing, (n_a + 1/2)/(n + K/2), the prequential codelength of a node's history equals the Dirichlet-multinomial marginal likelihood:
  `L_preq(node) = −log2[ Γ(K/2)/Γ(n + K/2) · Π_a Γ(n_a + 1/2)/Γ(1/2) ]`.
  This is **order-independent and computable at any moment from the counts alone**, with no history replay. (This follows from the Bernoulli/multinomial exactness noted by Grünwald & Roos. The closed form is the standard Dirichlet-multinomial identity and is my derivation, not quoted.)
- **Keep two-part for interpretability and prequential for decisions.** Galbrun's point that two-part codes expose *how* compression is achieved matches TRELLIS's interpretability pillar. A reasonable split: report a two-part decomposition (model bits vs. data bits) for inspection, but make accept/reject decisions with prequential or marginal-likelihood codelengths, which avoid arbitrary precision choices.
- **A useful mapping from MDL terms to omission/commission.** Too much L(G) (unmerged, over-specific categories and chunks) corresponds to omission. Too much L(D|G) under a normalized code (probability wasted on non-sentences) corresponds to commission. MDL is the principled balance point, and that point moves toward more specific grammars as N grows.

### Gaps
- Rissanen 1989 was confirmed only through CrossRef metadata and Goldsmith's citations; its contents were not reviewed here.
- No source found that applies NML or Safe-Bayes to grammar induction specifically. Their transfer to TRELLIS is untested inference.

---

## Q3. Grammar as compression: grammar-based codes, the smallest grammar problem, Sequitur / RePair / BPE, and what "globally optimal" can mean

### Takeaway
Finding the smallest grammar for a string is NP-hard, and there is no polynomial-time approximation better than a factor of 8569/8568 unless P = NP. The best polynomial algorithms reach O(log(n/g*)), while practical greedy compressors have weak worst-case ratios. Two results make this much less bleak for TRELLIS. First, Kieffer & Yang show that **any *irreducible* grammar** (no repeated adjacent pair, every rule used at least twice, distinct rules expand to distinct strings) already gives an asymptotically optimal universal code. So *minimal viability*, enforced by local irreducibility checks, buys asymptotic optimality without solving the global problem. Second, greedy pair-merging (BPE) carries constant-factor guarantees on compression utility (via submodularity and APX analyses). The practical conclusion: "globally optimal" should be operationalized as *irreducible plus local-MDL-stable plus exactly optimal within a restricted family*, not as the literal minimum.

### Cited Findings
- **Grammar-based codes.** Data are compressed by first converting them to a context-free grammar from which the data can be reconstructed. "Under some weak restrictions, a grammar based code is a universal lossless source code for any finite state information source." A grammar is *irreducible* if: (a.1) it is admissible; (a.2) distinct variables expand to distinct strings; (a.3) every variable other than the start symbol appears at least twice in right-hand sides; (a.4) no pair of symbols appears more than once at non-overlapping positions in right-hand sides. "Every SLP can be easily made irreducible by a simple post-processing" — [Kieffer & Yang 2000, *IEEE TIT* 46(3):737–754](https://doi.org/10.1109/18.841160) ([PDF](https://compression.ru/download/articles/grammar/kieffer_2000_grammar_based_codes.pdf)); post-processing remark and redundancy bound as summarized by [Bannai et al. 2021 (arXiv 1908.06428)](https://arxiv.org/abs/1908.06428).
- Redundancy can be bounded by O(log log n / log n) when the compressor produces grammars of size O(n / log n), which holds for all irreducible SLPs — [Bannai et al., "The Smallest Grammar Problem Revisited," *IEEE TIT* 67(1):317–328 (2021)](https://doi.org/10.1109/tit.2020.3038147) ([arXiv](https://arxiv.org/abs/1908.06428)).
- **Hardness and approximation.** The smallest grammar problem "cannot be solved in polynomial time unless P = NP." Unless P = NP, no polynomial algorithm can produce an SLP smaller than (8569/8568)·g(w) — [Bannai et al. 2021](https://arxiv.org/abs/1908.06428). Charikar et al. give an O(log(n/g*)) approximation algorithm, an exponential improvement over Bisection's O(n^{1/2}) — [Charikar et al., *IEEE TIT* 51(7):2554–2576 (2005)](https://en.wikipedia.org/wiki/Smallest_grammar_problem); [STOC 2002 version](https://compression.ru/download/articles/grammar/charikar_2002_approximating_the_smallest_grammar.pdf). Related thesis: Lehman, *Approximation Algorithms for Grammar-Based Data Compression*, MIT PhD 2002 — [DSpace](https://dspace.mit.edu/handle/1721.1/87172).
- Tight ratios for practical compressors: LZ78 is Θ((n/log n)^{2/3}), Bisection is Θ(√(n/log n)), and RePair's lower bound improves from Ω(√log n) to Ω(log n / log log n) — [Bannai et al. 2021](https://arxiv.org/abs/1908.06428).
- **Sequitur** builds a hierarchy by "replacing repeated phrases with a grammatical rule ... recursively." It maintains digram uniqueness and rule utility (every rule used more than once), runs incrementally, and takes linear time and space — [Nevill-Manning & Witten 1997, *JAIR* 7:67–82](https://arxiv.org/abs/cs/9709102).
- **RePair** combines "a simple but powerful phrase derivation method and a compact dictionary encoding" and runs offline in linear time and space — [Larsson & Moffat 2000, *Proc. IEEE* 88(11):1722–1732](https://people.eng.unimelb.edu.au/ammoffat/abstracts/lm00procieee.html).
- **BPE** (Gage, *C Users Journal*, Feb 1994) repeatedly replaces the most frequent pair of adjacent bytes with an unused byte — [Wikipedia: Byte-pair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding); [article copy](https://jacobfilipp.com/DrDobbs/articles/CUJ/1994/9402/gage/gage.htm). It was adapted to open-vocabulary NMT subword units by [Sennrich, Haddow & Birch, ACL 2016, pp. 1715–1725](https://aclanthology.org/P16-1162/).
- Greedy BPE is a 1/σ·(1 − e^{−σ}) approximation of the optimal merge sequence, where σ is the total backward curvature (a submodularity argument), with an empirical lower bound of about 0.37 — [Zouhar et al., Findings of ACL 2023](https://aclanthology.org/2023.findings-acl.38).
- Optimal pair encoding is APX-complete, and BPE approximates the optimal compression utility "to a worst-case factor between 0.333 and 0.625" — [Kozma & Voderholzer, arXiv 2411.08671, ESA 2026](https://arxiv.org/abs/2411.08671) ([LIPIcs](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2026.80)). *Conflict note:* a search-engine summary described the ratio as "logarithmic"; the paper's own abstract says a constant factor of 0.333–0.625.
- A top-down alternative to merge-based vocabularies: a unigram-language-model subword segmentation — [Kudo, ACL 2018](https://aclanthology.org/P18-1007/).

### Inferences
- **Map the irreducibility conditions onto TRELLIS invariants (a "minimal viability" checklist).** (a.4) becomes: no adjacent pair of active categories recurs with positive MDL gain without being chunked (Sequitur's digram uniqueness, relaxed to "positive-gain digram uniqueness"). (a.3) becomes: every content concept used as a chunk has usage of at least 2, and in MDL terms enough usage to repay its definition cost, otherwise delete it and inline its parts. (a.2) becomes: no two content concepts with indistinguishable expansion distributions; merge them, which is Cobweb's own merge operator. These are cheap, local checks that can run after every sentence.
- **Compression vs. generalization.** Grammar-based codes describe *one* string exactly (straight-line programs). TRELLIS must also *generalize*, assigning probability to unseen sentences. Generalization comes from merging categories (context hierarchy), not from chunking alone (Q4). The compression literature therefore informs the *content* side (which chunks), while the generalization side needs the probabilistic, normalized code of Q2.
- **Greedy is not hopeless.** If the chunk-value function is approximately submodular (diminishing returns as overlapping chunks compete for the same occurrences), greedy best-gain-first commitment, as in TRELLIS's greedy parse and BPE, has constant-factor guarantees on compression utility. This supports the "stay greedy" design preference, provided the *objective* is a compression utility.
- **"Global" within a family.** Exact global optima are tractable in restricted families. The most useful example for TRELLIS: given a Cobweb tree, the codelength-optimal *cut* (which nodes serve as categories) can be computed exactly by bottom-up dynamic programming (see Implications). This is a far better notion of "globally optimal" than the NP-hard smallest grammar.

### Gaps
- The "most frequent pair" description of RePair comes from secondary sources; the Larsson & Moffat abstract describes it only at a high level.
- The pruning details of Kudo's unigram-LM method were not verified (abstract only).
- No source found that gives approximation guarantees for chunk-plus-merge (generalizing) grammar induction, as opposed to straight-line compression.

---

## Q4. MDL / Bayesian objectives in grammar, lexicon and morphology induction: exact objectives and how candidates are proposed

### Takeaway
One template runs through all of this work: **cheap heuristics or triggers propose local edits (chunk, merge, split, delete), and a codelength or posterior decides whether to keep them**. Three precedents are almost exact templates for an incremental TRELLIS. Chen 1995 parses each sentence once with the current best grammar, uses triggers from that parse to propose moves, and evaluates the change in log p(O|G) + log p(G). Stolcke & Omohundro 1994 combine data incorporation with chunking and merging operators under a description-length prior and Dirichlet parameter priors. de Marcken 1996 estimates the change in description length from add and delete moves, allows undo, and builds a hierarchical lexicon in which every word is a composition of other words. Wolff's SNPR, and Langley & Stromsten's GRIDS reconstruction of it, show that *chunking plus disjunctive-class formation* is exactly the dual of TRELLIS's content and context hierarchies. Goldsmith shows that the pressure to minimize the number of units comes from pointer and signature costs inside the code, not from a separate count penalty.

### Cited Findings
**Wolff's SNPR and SP theory**
- Wolff, "Language acquisition, data compression and generalization," *Language & Communication* 2(1):57–89 (1982) — [DOI](https://doi.org/10.1016/0271-5309(82)90035-0).
- SNPR learns artificial CF-PSGs without supervision "using a technique of 'hierarchical chunking' combined with a search for disjunctive (part of speech) categories and processes for generalising grammatical rules and correcting over-generalisations." Later SP/ICMAUS work replaced hierarchical chunking with multiple alignment — [Wolff 2003, arXiv cs/0311045](https://arxiv.org/abs/cs/0311045).

**Langley & Stromsten (2000), GRIDS**
- GRIDS is "a rational reconstruction of Wolff's SNPR." It uses two operators, "merging existing nonterminal symbols and creating new symbols," a "bias toward grammars that minimize description length," and "a beam search to move from complex to simpler grammars" — [ECML 2000, LNAI 1810, pp. 220–228](https://mlanthology.org/ecmlpkdd/2000/langley2000ecml-learning).

**Grünwald (1996)**
- "A Minimum Description Length Approach to Grammar Inference," in *Connectionist, Statistical and Symbolic Approaches to Learning for NLP*, LNCS/LNAI 1040, p. 203 ff. It is an abstract MDL-based model for learning grammars from a large set of training sentences — [Utah LNCS bib index](https://ftp.math.utah.edu/pub/tex/bib/idx/lncs1996a/1040/0/203-z.html).

**Stolcke & Omohundro (1994), Bayesian model merging** — [ICGI-94 (arXiv cmp-lg/9409010)](https://arxiv.org/abs/cmp-lg/9409010)
- *Data incorporation*: build an initial model that "explicitly accommodat[es] each data point individually" (maximum likelihood, no generalization).
- *Structure merging*: produce M_{i+1} = m(M_i) with operators "that coalesce substructures," guided by the posterior P(M|X) ∝ P(M)·P(X|M).
- *Prior*: P(M_S) ∝ exp(−ℓ(M_S)), a description-length prior on structure, with Dirichlet priors on parameters. For SCFGs, "each occurrence of a nonterminal contributes log N bits," and productions use symmetrical Dirichlet priors. The likelihood P(X|M_S) is approximated with the Viterbi assumption.
- *SCFG operators*: **merging** of nonterminals (which can make "inductive 'leaps'") and **chunking**, which "takes a given sequence of nonterminals and abbreviates it using a newly created nonterminal."
- *Search*: greedy works for HMMs, but for SCFGs "chunking steps typically require several following merging steps and/or additional chunking steps to improve a grammar's posterior score," so beam search (width 3–10) is used.
- *Online*: "an on-line version ... in which the data incorporation and the merging/search stages are interleaved."
- Brown et al.'s class merging can be cast as model merging with a likelihood criterion — same source; [Brown et al. 1992, *CL* 18:467–480](https://aclanthology.org/J92-4003/).

**Chen (1995), Bayesian grammar induction for language modeling** — [ACL 1995 (arXiv cmp-lg/9504034)](https://arxiv.org/abs/cmp-lg/9504034)
- *Objective*: maximize p(O|G)·p(G) with p(G) = 2^−l(G).
- *Moves*: (1) create A → B C; (2) create A → B | C; (3) create A → A B | B (iteration).
- *Evaluation*: only the *difference* in the objective is needed. p(O|G) is approximated by the Viterbi parse, and heuristics predict how a move changes that parse.
- *Incrementality*: parse the first sentence, search for the optimal grammar over it, use that grammar to parse the second, and so on, "parsing the next sentence using the best grammar found on the previous sentences ... until the entire training corpus is covered," parsing each sentence once.
- *Triggers*: a move is considered only if "triggered in the sentence currently being parsed." Example: adjacent symbols in the Viterbi parse trigger B → A_talks A_slowly.
- *Post-pass*: Inside-Outside re-estimation. On two tasks whose training data were generated by a PCFG, the algorithm "outperforms the other techniques" (n-gram models and Inside-Outside). On naturally occurring data it "does not perform as well as n-gram models but vastly outperforms the Inside-Outside algorithm."

**de Marcken (1996), Unsupervised Language Acquisition (MIT PhD)** — [arXiv cmp-lg/9611002](https://arxiv.org/abs/cmp-lg/9611002)
- *Representation*: words and sentences are both compositions of lexical parameters, and the lexicon is hierarchical ("national football league" is built from "national", "football", "league", which are built from smaller units). If an extralinguistic pattern such as "eatyourpeas" enters the lexicon, "it will be represented in terms of eat, your and peas ... This mitigates the consequences of such unavoidable mistakes."
- *Search*: "Hypothesize a set of changes ... estimate the effect on the total description length ... implement each change that is estimated to reduce the description length," with "changes that 'undo' previous modifications."
- *Add and delete rules*: a parameter is added when "the benefit of representing the chain of compositions ... by a single reference is expected to exceed the description length of the parameter." A parameter is deleted "if ... the cost of substituting its representation for it is less than the cost of its description length."
- *Candidates*: "only ... parameters that can be built by composing two other parameters," taken from "the most probable representation of some utterance."
- *Locality*: "no effect is considered if it would require reanalysis of the evidence." Data are stored flat and re-parsed, so a representation can jump from wa∘term∘el∘on to water∘melon "in a single step."
- *Statistical-test view*: a footnote shows that adding x₁₂ = x₁∘x₂ is justified at 95% confidence if p̂(x₁₂) − p(x₁)p(x₂) > 1.96·√(p(x₁₂)(1 − p̂(x₂|x₁)))/√N, a threshold that "depends inversely upon the amount of evidence."
- Goldsmith's summary of de Marcken: a lexical item of frequency F has compressed length −log F, the corpus is coded by its Viterbi-best parse, and items are added "when the improvement in compression ... is greater than the length (or 'cost') associated with the new item" — [Goldsmith 2001](https://aclanthology.org/J01-2001).

**Goldsmith (2001, 2006), Linguistica** — [*CL* 27(2):153–198](https://aclanthology.org/J01-2001); [Goldsmith 2006, *Natural Language Engineering* 12(4)](https://doi.org/10.1017/S1351324905004055)
- Method: "We develop a set of heuristics that rapidly develop a probabilistic morphological grammar, and use MDL as our primary tool to determine whether the modifications proposed by the heuristics will be adopted or not."
- "MDL is a framework for evaluating proposed analyses, but it does not provide a set of heuristics that are nonetheless essential for obtaining candidate analyses."
- *Model*: three lists (stems, suffixes, signatures). P(w = t + f) = P(σ)·P(t|σ)·P(f|σ). The compressed corpus length is Σ_w [w]·(−log P(σ(w)) − log P(t|σ) − log P(f|σ)). Model length is the sum of the lengths of the stem, suffix, and signature lists, with pointers of length log([W]/[σ]).
- *Why naive letter-counting fails*: counting letters alone would invent suffixes -t, -ted, -ts, -ting for verb stems ending in t. Pointer costs prevent this, because splitting a signature makes pointers longer: x·log(N/x) + y·log(N/y) ≥ (x+y)·log(N/(x+y)).
- Linguistica's functions "can be incrementally applied to a corpus."

**Word segmentation and morphology**
- Brent (1999), MBDP-1: "Incremental Search ... evaluates the probabilities of segmentations of successively longer prefixes of the observed corpus, adding one utterance at a time." It commits per utterance "by optimizing the prior probabilities of segmentations of the entire corpus of all utterances processed so far" — [*Machine Learning* 34:71–105 (arXiv cs/9905007)](https://arxiv.org/abs/cs/9905007).
- Goldwater, Griffiths & Johnson (2009): Bayesian ideal learners that treat words as independent units produce poorer segmentations than learners that treat words as predictive of other units. "Taking context into account is important" — [*Cognition* 112(1):21–54](https://doi.org/10.1016/j.cognition.2009.03.008).
- Pearl, Goldwater & Steyvers (2010): constrained online learners approximating the same ideal learner sometimes "perform equivalently or better than the ideal learner," consistent with "Less is More" — [*Research on Language and Computation* 8(2–3):107–132](https://doi.org/10.1007/s11168-011-9074-5) ([manuscript](https://sites.socsci.uci.edu/~lpearl/courses/readings/PearlGoldwaterSteyvers2010Manu_OnlineBayesWordSeg.pdf)).
- Morfessor is a probabilistic MAP framework that induces a lexicon of morphs storing "both the usage and form of the morphs" — [Creutz & Lagus 2007, *ACM TSLP* 4(1):1–34](https://doi.org/10.1145/1187415.1187418).

**Modern MDL learners (Katzir lab) and Bayesian program induction**
- MDL as an evaluation metric for phonology: [Rasin & Katzir 2016, *Linguistic Inquiry* 47(2):235–282](https://doi.org/10.1162/ling_a_00210). An implemented MDL learner "succeeds in learning various linguistically-relevant patterns from small corpora" — [Rasin, Berger, Lan, Shefi & Katzir 2021, *JLM* 9(1):17–66](https://jlm.ipipan.waw.pl/index.php/JLM/article/view/266).
- MDL-RNNs, which balance network complexity against accuracy, learn aⁿbⁿ, aⁿbⁿcⁿ, aⁿb²ⁿ, aⁿbᵐcⁿ⁺ᵐ, and addition, often with 100% accuracy, and are small enough to prove correct for all inputs — [Lan, Geyer, Chemla & Katzir 2022, *TACL* 10:785–799](https://aclanthology.org/2022.tacl-1.45).
- For aⁿbⁿ, the theoretically correct network "is in fact not an optimum of commonly used objectives — even with ... L1, L2 ... early-stopping, dropout," but "replacing standard targets with the Minimum Description Length objective (MDL) results in the correct solution being an optimum." L1/L2 penalize magnitude rather than information content — [Lan, Chemla & Katzir, ACL 2024, pp. 13198–13210](https://aclanthology.org/2024.acl-long.713) ([arXiv 2402.10013](https://arxiv.org/abs/2402.10013)).
- Bayesian program induction synthesizes interpretable morpho-phonology models across 70 datasets from 58 languages. Joint inference yields a cross-language meta-model of typological tendencies — [Ellis, Albright, Solar-Lezama, Tenenbaum & O'Donnell 2022, *Nature Communications* 13:5024](https://doi.org/10.1038/s41467-022-32012-w).

### Inferences
- **TRELLIS already has the SNPR/GRIDS architecture.** Content hierarchy = chunking (Stolcke's chunk operator). Context hierarchy = disjunctive classes (Stolcke's merge, SNPR's classes). What v1 lacks is a *shared codelength* that adjudicates both operators. Stolcke's posterior, with a DL prior plus Dirichlet marginals per nonterminal, is the most direct import. With Cobweb counts it can be computed per node in closed form (Q2).
- **Chen 1995 is a near-literal blueprint for an incremental unsupervised TRELLIS:** parse with the current grammar, propose moves via triggers from that parse (adjacent active categories, which TRELLIS's greedy parser already enumerates), score the change in objective, commit, move to the next sentence. Two refinements: replace Chen's Viterbi-only data term with the inside probability (aligned with the inside-outside pivot) or treat it as an upper bound; and add de Marcken's delete/undo move.
- **Stolcke's lookahead finding explains the "chunk only pays after generalization" pattern.** A new chunk often shows positive gain only once its parts or its context class are merged. For TRELLIS this argues for evaluating a candidate chunk at the *generalized (ancestor) level* rather than the instance level, and for a probation or "frontier" period (as anticipated in `INSIDE_OUTSIDE.md`) instead of immediate rejection.
- **Goldsmith answers the user's question about minimizing the number of chunks.** Fewer units emerge from the code itself. A per-unit definition cost, plus usage-based pointer codes (−log frequency), means every split or extra unit lengthens pointers: splitting a class of usage x+y costs exactly (x+y)·H₂(x/(x+y)) extra bits (algebraic restatement of Goldsmith's inequality). So "minimize chunks" and "maximize chunk quality" are two terms of one sum, not two objectives.
- **de Marcken's hierarchical lexicon matches the user's stance that a reusable grammar matters more than a "correct" one.** Wrong chunks built from right parts cause little harm. TRELLIS chunks are already compositions of concept ids, so the same robustness applies, provided chunks keep pointing at their parts rather than being flattened.
- **Goldwater et al. warn against unigram chunk codes.** A bag-of-chunks code (Krimp-style unigram) over-rewards frequent collocations and tends to merge "the dog" into a word-like unit. Chunk gains should be computed under a *context-conditioned* code, which TRELLIS's context hierarchy can supply. See Implications, P2.
- **Lan et al.'s result maps onto TRELLIS's count thresholds.** A fixed count gate (τ) or smoothing strength acts like a magnitude regularizer. A true codelength measures information content. That is the argument for replacing τ with an MDL-derived adaptive threshold.

### Gaps
- Could not verify whether Wolff's 1982 paper uses the term "folding" for class formation. The verified description says "search for disjunctive (part of speech) categories."
- Grünwald 1996: only bibliographic and abstract-level information; the exact objective and search were not extracted.
- Goldsmith 2006: only bibliographic and abstract-level information.
- Brent's exact MBDP-1 prior formula was not extracted (only the incremental-search procedure).

---

## Q5. The value of a chunk as compression gain: MDL pattern mining, library learning, fragment grammars, and the EBL "utility problem"

### Takeaway
Across pattern mining, program synthesis, and cognitive architectures, a chunk's value comes out almost the same way: **(number of uses) × (bits or work saved per use) − (cost of storing or defining it) − (cost of matching it)**. Slim gives a closed-form estimate of the gain of merging two existing code-table elements, using only their usages and co-usage. That is precisely the statistic a Cobweb-based parser can keep for adjacent concept pairs. Stitch states the same balance in words: a chunk should be "general enough that it applies in many locations, but specific enough that it captures a lot of structure at each location." DreamCoder and LOVE show that likelihood alone gives degenerate or unconstrained abstractions, while adding a description-length term on the library gives reusable ones. Fragment grammars frame this as a Bayesian store-vs-compute trade-off.

### Cited Findings
**MDL pattern mining**
- *Krimp*: the best pattern set is the one that compresses the database best, and it typically returns "only hundreds of itemsets," up to seven orders of magnitude fewer than frequent-itemset mining — [Vreeken, van Leeuwen & Siebes 2011, *DMKD* 23:169–214](https://link.springer.com/article/10.1007/s10618-010-0202-x).
- Krimp/Slim encoding: L(X|CT) = −log(usage(X)/Σ_Y usage(Y)); L(D|CT) = Σ_t Σ_{X∈cover(t)} L(X|CT); L(CT|D) = Σ_{X: usage≠0} [L(X|ST) + L(X|CT)], where ST is the singleton "standard code table"; the objective is L(CT, D) = L(CT|D) + L(D|CT) — [Smets & Vreeken, SDM 2012, pp. 236–247](https://vreeken.groups.cispa.de/pubs/2012/slim-smets,vreeken.pdf) ([DOI](https://doi.org/10.1137/1.9781611972825.21)).
- *Slim's candidates and gain*: every iteration considers "all pairwise combinations of X, Y ∈ CT as candidates in Gain Order." With x = usage(X), y = usage(Y), s = Σ usages, and xy′ = |usage(X) ∩ usage(Y)| (the estimated usage of X∪Y), x′ = x − xy′, y′ = y − xy′, s′ = s − xy′:
  - data gain: **ΔL(D) = s·log s − s′·log s′ + xy′·log xy′ − Σ_{C changed}(c·log c − c′·log c′)**;
  - model change: ΔL(CT) = log xy′ − L(X∪Y|ST) + |CT|·log s − |CT′|·log s′ + (terms for elements whose usage changes).
  - Slim assumes "only the usage of X, Y, and X∪Y will change," which gives "a very easily calculable, and generally accurate estimate of ΔL." Cascading usage changes are "unpredictable, [but] in practice the effect is not often dramatic." Exact gain is computed only for the best estimated candidate. Slim is parameter-free and evaluates orders of magnitude fewer candidates than Krimp — [Smets & Vreeken 2012](https://vreeken.groups.cispa.de/pubs/2012/slim-smets,vreeken.pdf).
- *Sequences*: SQS encodes event sequences with serial episodes, with both a select-from-candidates algorithm and a parameter-free any-time miner — [Tatti & Vreeken, KDD 2012, pp. 462–470](https://doi.org/10.1145/2339530.2339606) ([arXiv 1902.02834](https://arxiv.org/abs/1902.02834)). GoKrimp "greedily extend[s] a pattern until no additional compression benefit" — [Lam, Mörchen, Fradkin & Calders 2014, *SADM* 7(1):34–52](https://www.philippe-fournier-viger.com/spmf/gokrimp.pdf). ISM replaces a hand-designed encoding with a probabilistic subsequence-interleaving model fit by structural EM, whose E-step is "a submodular optimization problem subject to a coverage constraint" — [Fowkes & Sutton, KDD 2016, pp. 835–844](https://arxiv.org/abs/1602.05012).
- *Streams*: StreamKrimp detects distribution change by switching code tables, partitioning the stream into substreams with "only a very limited amount of data storage" — [van Leeuwen & Siebes, ECML PKDD 2008, pp. 672–687](https://doi.org/10.1007/978-3-540-87479-9_62).
- *Search strategies*: MDL "provides a basis for designing a score ... but no way to actually find the best collection"; "exhaustive search is generally infeasible"; algorithms either generate-then-select or generate candidates on the fly (levelwise, anytime); "efficiently and accurately bounding these costs" is central. Dictionary-based search goes bottom-up by combining elements, block-based search goes top-down by splitting — [Galbrun 2022](https://arxiv.org/abs/2007.14009).

**Library learning**
- *DreamCoder*: Wake: ρ_x = argmax P[ρ|x, L] ∝ P[x|ρ]·P[ρ|L]. Abstraction sleep: L = argmax_L P[L]·Π_x max_{ρ a refactoring of ρ_x} P[x|ρ]·P[ρ|L], where P[L] is "a description-length prior over libraries." This is equivalent to minimizing the description length of the library plus that of the refactored programs. Version spaces with about 10⁶ nodes represent about 10¹⁴ refactorings — [Ellis et al., PLDI 2021, pp. 835–850](https://doi.org/10.1145/3453483.3454080) ([arXiv 2006.08381](https://arxiv.org/abs/2006.08381)).
- *Stitch*: utility U(A) = −cost(A) + cost(P) − cost(Rewrite_R(P, A)). At a high level it "maximize[s] the product of the size of the abstraction and the number of locations where the abstraction can be used ... general enough that it applies in many locations, but specific enough that it captures a lot of structure at each location." Branch-and-bound uses the summed sizes of match locations as an upper bound. It is 3–4 orders of magnitude faster and uses 2 orders of magnitude less memory than DreamCoder's library learning, with comparable or better compressivity — [Bowers et al., *PACMPL* 7(POPL), Art. 41 (2023)](https://arxiv.org/abs/2211.16605).
- *babble*: library learning "modulo theory," using e-graphs, equality saturation, and e-graph anti-unification to find shared structure despite syntactic variation, with "better compression orders of magnitude faster" — [Cao et al., POPL 2023 (arXiv 2212.04596)](https://arxiv.org/abs/2212.04596).
- *LOVE (options via compression)*: maximizing likelihood admits "many solutions that maximize the likelihood equally well, including degenerate solutions." Adding "a penalty on the description length of the skills ... incentivizes the skills to maximally extract common structures" — [Jiang, Liu, Eysenbach, Kolter & Finn, NeurIPS 2022, 35:21184–21199](https://arxiv.org/abs/2212.04590).

**Store vs. compute**
- *Fragment grammars*: "more structure-building means less need to store while more storage means less need to compute structure." A hierarchical Bayesian fragment grammar explores "the optimum balance between structure-building and reuse" and generalizes adaptor grammars (Johnson, Griffiths & Goldwater 2007) — [O'Donnell, Tenenbaum & Goodman 2009, MIT-CSAIL-TR-2009-013](http://hdl.handle.net/1721.1/44963). The full treatment is O'Donnell, *Productivity and Reuse in Language* (MIT Press, 2015) — [MIT Press DOI (bibliography chapter)](https://doi.org/10.7551/mitpress/10008.003.0015).

**Cognitive architectures: the utility problem**
- Minton (AAAI-88): "Utility = (AvrSavings × ApplicFreq) − AvrMatchCost," where AvrSavings is the average savings when the rule applies, ApplicFreq is how often it applies when tested, and AvrMatchCost is the average cost of matching it. PRODIGY also "compresses" learned descriptions to reduce match cost — [Minton 1988 (AAAI-88)](https://cdn.aaai.org/AAAI/1988/AAAI88-100.pdf); journal version in [*Artificial Intelligence* 42(2–3):363–391 (1990)](https://doi.org/10.1016/0004-3702(90)90059-9).
- Chunking as a general learning mechanism in Soar — [Laird, Rosenbloom & Newell 1986, *Machine Learning* 1(1):11–46](https://doi.org/10.1023/a:1022639103969).

**Cognitive chunking models that pair chunks with variables (concepts)**
- HVM "learns chunks from sequences and abstracts contextually similar chunks as variables." Chunk proposal: if "a significant correlation is found between consecutively parsed chunk pairs (with p = 0.05)," the pair is concatenated into a new chunk. A variable denotes "distinct chunks sharing preceding and succeeding chunks." HVM learns a more efficient dictionary than Lempel–Ziv on babyLM, its sequence likelihood correlates with human recall times, and the authors frame abstraction as a rate–distortion trade-off — [Wu, Thalmann, Dayan, Akata & Schulz, ICLR 2025 (arXiv 2410.21332)](https://arxiv.org/abs/2410.21332). It builds on HCM — Wu, Éltető, Dasgupta & Schulz, NeurIPS 2022, 35:36706–36721 (as cited in the HVM paper).

### Inferences
- **A closed-form chunk gain for TRELLIS (my approximation of Slim's formula).** Take X and Y to be active nodes (concepts at the current cut), z the number of adjacent X·Y occurrences in current parses, and s the total number of tokens in the current parse "cover." When z ≪ x, y, s, Slim's data gain reduces to
  **ΔL_data ≈ z·[ PMI(X,Y) − log₂e ]**, with PMI = log₂(s·z/(x·y)).
  So **value ≈ (uses) × (bits saved per use) − (definition cost)**, the same shape as Stitch's "uses × size" and Minton's "frequency × savings − match cost". Worked checks: s = 1000, x = y = 100, z = 5 gives an exact −11.9 bits against an approximate −12.2. s = 1000, x = y = 50, z = 40 gives an exact +170 against an approximate +102; the approximation is conservative when the pair absorbs most of X's uses.
- **A pair must beat chance by more than e (about 2.7×) before its definition cost even counts.** PMI must exceed log₂e ≈ 1.44 bits. This gives a principled floor in place of an arbitrary count gate.
- **TRELLIS's climbing-ancestor gate becomes "climb to the max-gain level."** Moving X and Y up the context tree raises usages (larger z) but lowers PMI. The product z·(PMI − log₂e) − L_def has an interior maximum, a "basic level for the pair." Choosing the ancestor pair that maximizes gain replaces the fixed τ = 30 gate with an objective-driven one (see Implications, P2).
- **Match cost matters for a greedy parser.** Every committed chunk adds candidates the parser must test (Minton's AvrMatchCost). A small Kt-like penalty, λ·Δlog₂(parse steps), keeps the chunk inventory lean in a way that a pure bit count cannot.
- **Use refactoring rather than frozen parses for usage counts.** DreamCoder recomputes program representations under the new library before scoring it, and de Marcken re-parses flat data. Slim approximates the cascade and computes the exact gain only for the best candidate. For TRELLIS: estimate gains from pair counts, then re-parse a bounded buffer of recent sentences to get exact usage before final commitment.
- **HVM shows the value of chunks plus variables.** It is a recent, cognitively validated system that pairs chunk proposal (statistical dependence of adjacent units) with variable formation (shared left and right contexts). That is TRELLIS's content/context split, and it supports the claim that the two hierarchies are the right decomposition.

### Gaps
- The full O'Donnell (2015) book was not reviewed; the claims here rest on the 2009 technical-report abstract.
- GoKrimp's and SQS's exact encodings were not extracted; only their search strategies were verified.
- No source found that combines chunk-gain estimation with *context-class* (substitution) gain, which is what TRELLIS needs. This appears to be novel.

---

## Q6. Simplicity in cognition: is compression a credible cognitive principle for chunks and concepts?

### Takeaway
There is a substantial and convergent literature arguing that cognition favors simple, compressive representations. Simplicity and likelihood are formally the same principle. Concept difficulty tracks incompressibility. Working-memory chunks behave like units of a compressed code. MDL-style learners can recover linguistic restrictions from positive evidence alone, with quantifiable data requirements. This gives TRELLIS v2 a cognitive-plausibility argument for an MDL objective that a pure likelihood objective lacks. It also offers testable predictions, such as chunk-based recall and data requirements per construction.

### Cited Findings
- Simplicity and likelihood principles of perceptual organization "are not in competition, but are identical" (via Kolmogorov complexity) — [Chater 1996, *Psych. Review* 103(3):566–581](https://doi.org/10.1037/0033-295x.103.3.566).
- Simplicity principle: "Choose the pattern that provides the briefest representation of the available information." It is normatively justified and consistent with data on perception, similarity, learning, memory, and reasoning, and offers a starting point for rational analysis in Anderson's sense — [Chater 1999, *QJEP A* 52(2):273–302](https://doi.org/10.1080/027249899391070).
- Review: simplicity as a unifying principle across perception, learning, and high-level cognition — [Chater & Vitányi 2003, *TICS* 7(1):19–22](https://doi.org/10.1016/s1364-6613(02)00005-0).
- "Ideal learning" of natural language gives positive results for learning from positive evidence — [Chater & Vitányi 2007, *J. Math. Psych.* 51(3):135–163](https://doi.org/10.1016/j.jmp.2006.10.002). As summarized later, "it is possible to learn the exact generative model underlying a wide class of languages, purely from observing samples" — [Hsu, Chater & Vitányi 2011, *Cognition* 120(3):380–390](https://doi.org/10.1016/j.cognition.2011.02.013).
- MDL provides "a simple and practical methodology for estimating how much linguistic data are required to learn a particular linguistic restriction." Some restrictions are easily learnable while others appear to need additional cues or constraints — [Hsu & Chater 2010, *Cognitive Science* 34(6):972–1016](https://doi.org/10.1111/j.1551-6709.2010.01117.x).
- "The subjective difficulty of a concept is directly proportional to its Boolean complexity (the length of the shortest logically equivalent propositional formula), that is, to its logical incompressibility" — [Feldman 2000, *Nature* 407:630–633](https://doi.org/10.1038/35036586).
- "A chunk is a unit in a maximally compressed code." Span depends on pattern length after compression, with a limit of about 3–4 chunks, roughly 7 uncompressed items — [Mathy & Feldman 2012, *Cognition* 122(3):346–362](https://doi.org/10.1016/j.cognition.2011.11.003).
- Observers exploit statistical regularities to store more items in visual working memory, "quantitatively predicted by a Bayesian learning model and optimal encoding scheme" — [Brady, Konkle & Alvarez 2009, *JEP: General* 138(4):487–502](https://doi.org/10.1037/a0016797).
- Chunking as rational *lossy* compression trades "storage capacity and memory precision" — [Nassar, Helmers & Frank 2018, *Psych. Review* 125(4):486–511](https://doi.org/10.1037/rev0000101).
- Human memory for binary sequences is predicted by the "shortest description in [a recursive] language," and this improves on transition-probability models — [Planton et al. 2021, *PLoS Comp. Biol.* 17(1):e1008598](https://doi.org/10.1371/journal.pcbi.1008598).
- *Rational rules*: Bayesian inference over "a grammatically structured hypothesis space — a concept language of logical rules" — [Goodman, Tenenbaum, Feldman & Griffiths 2008, *Cognitive Science* 32(1):108–154](https://doi.org/10.1080/03640210701802071).
- Discovery of structural form by probabilistic inference "over a space of graph grammars" (trees, orders, rings, cliques, and more) — [Kemp & Tenenbaum 2008, *PNAS* 105(31):10687–10692](https://doi.org/10.1073/pnas.0802631105).
- A "universal representation language" in which learners build mental models of logic, number, CFLs, and more — [Piantadosi 2021, *Minds and Machines* 31(1):1–58](https://doi.org/10.1007/s11023-020-09540-9).
- Color-naming systems "achieve near-optimal compression" in an information-theoretic sense — [Zaslavsky, Kemp, Regier & Tishby 2018, *PNAS* 115(31):7937–7942](https://doi.org/10.1073/pnas.1800521115).
- Resource-rational analysis: "the rational use of limited resources as a unifying principle" — [Lieder & Griffiths 2020, *BBS* 43:e1](https://doi.org/10.1017/s0140525x1900061x).

### Inferences
- **Cognitive plausibility argues for MDL over likelihood.** Mathy & Feldman's chunk-as-code-unit, Brady et al.'s compression with learning, and HVM's recall-time correlation (Q5) together suggest that an MDL-trained TRELLIS could be tested against human chunking data, not just grammar metrics. It also gives a natural account of why the chunk inventory should stay small (a capacity of about 3–4 chunks in working memory).
- **Hsu & Chater offer a principled evaluation tool.** For each construction where TRELLIS commits omission or commission errors, one can compute how many sentences an ideal MDL learner would need to learn the restriction. That separates "TRELLIS is wrong" from "the data are insufficient."
- **Rate–distortion is the right lens for generation.** Generating from a coarse cut is lossy decoding. Moving up the hierarchy lowers rate and raises distortion, which shows up as commission. This matches HVM's abstraction–distortion trade-off and Nassar et al.'s capacity–precision trade-off.

### Gaps
- The abstracts for Chater & Vitányi 2007 and Piantadosi were only partly available. Claims about Chater & Vitányi 2007 rely on its title and on Hsu et al.'s (2011) summary.
- No source found that directly tests MDL-learned *phrase-structure chunks* against human phrase-chunking data. That would be a novel TRELLIS contribution.

---

## Q7. Connecting to Cobweb's objective: category utility as information, Anderson's rational model, Snob/AutoClass. How can an MDL chunk score coexist with CU-driven concept formation?

### Takeaway
The information-theoretic category utility used by the MacLellan-lab Cobweb implementations is already a *codelength saving*: CU(c) = P(c)·[U(parent) − U(c)], the expected reduction in attribute entropy from knowing the child. Summed over children, it equals the mutual information between the partition and the attributes. What CU lacks, compared with MDL, Snob, or a CRP prior, is a principled price for the partition itself. Cobweb divides by the number of children instead, and that heuristic does not scale with N. Anderson's rational model supplies the missing price through its coupling (CRP-like) prior, and single-particle incremental approximations of it fit human data. So CU can remain the fast, local, cognitively plausible sorting heuristic, while codelength governs chunk commitment, cut selection, and consolidation. Optionally, CU can be "MDL-corrected" with a label-cost term.

### Cited Findings
- Cobweb (incremental conceptual clustering) — [Fisher 1987, *Machine Learning* 2(2):139–172](https://doi.org/10.1007/bf00114265). Models of incremental concept formation — [Gennari, Langley & Fisher 1989, *Artificial Intelligence* 40(1–3):11–61](https://doi.org/10.1016/0004-3702(89)90046-5).
- The category utility hypothesis: categories are useful because they predict instance features. The derived measure predicts the basic level in natural and artificial hierarchies and is related "to certain concepts from information theory." The article explores the link between prediction and "efficient storage of information" — [Corter & Gluck 1992, *Psych. Bulletin* 111(2):291–303](https://doi.org/10.1037/0033-2909.111.2.291).
- *Information-theoretic CU (used in Cobweb/4L and the MacLellan et al. 2016 implementation)*: CU(c) = P(c)·[U(c_p) − U(c)], with U(c) = Σ_i P(W_i|c)·U(W_i|c) and U(W_i|c) = −Σ_j P(w_ij|c)·log P(w_ij|c) — [Lian, Baglodi & MacLellan, ACS 2024 (arXiv 2409.12440)](https://arxiv.org/abs/2409.12440). Cobweb chooses among add, create, merge, and split using the *average* category utility Σ_k CU(C_k)/s, where s is the number of children. This lets it "compare partitions with varying numbers of concepts (Fisher, 1987)" — [Lian, Varma & MacLellan, CogSci 2024 (arXiv 2403.03835)](https://arxiv.org/abs/2403.03835).
- Anderson's rational model of categorization assumes categorization derives optimal estimates of unseen features. Its Bayesian analysis "is placed within an incremental categorization algorithm," and it accounts for basic-level extraction and trial-by-trial learning, among other effects — [Anderson 1991, *Psych. Review* 98(3):409–429](https://doi.org/10.1037/0033-295x.98.3.409).
- The RMC connects to nonparametric Bayesian statistics. Particle filters "sequentially approximate the posterior ... updated as new data become available," and "a particle filter with a single particle provides a good description of human inferences" — [Sanborn, Griffiths & Navarro 2010, *Psych. Review* 117(4):1144–1167](https://doi.org/10.1037/a0020511).
- Snob, MML clustering of multi-state, Poisson, von Mises, and Gaussian data — [Wallace & Dowe 2000, *Statistics and Computing* 10(1):73–83](https://doi.org/10.1023/a:1008992619036). The original MML classification measure is Wallace & Boulton 1968 (Q2).
- AutoClass: a CrossRef-indexed version is Stutz & Cheeseman, "AutoClass — A Bayesian Approach to Classification," in *Maximum Entropy and Bayesian Methods* (1996), pp. 117–126 — [DOI](https://doi.org/10.1007/978-94-009-0107-0_13).

### Inferences
- **CU is a codelength saving per instance.** Summing Cobweb/4L's CU over the children of p gives Σ_k CU(c_k) = U(p) − Σ_k P(c_k)·U(c_k), which is approximately Σ_i I(A_i; K | p): the bits per instance saved by coding attribute values with child distributions instead of the parent's. (My derivation from the cited formula.)
- **An MDL-corrected partition score.** To use the split you must also transmit *which child* (H(K|p) bits per instance) and the children's parameters. So an MDL partition score for N_p instances is
  **ΔL_split(p) ≈ N_p·[ Σ_k CU_info(c_k) − H(K|p) ] − ΔL_params**, with ΔL_params ≈ ½·log₂N_p per extra free parameter, or exact via Dirichlet marginals.
  Unlike Fisher's division by K, this penalty grows sensibly with data: more data justifies finer splits. In Snob terms, a split pays only if the attributes, *jointly*, carry more information about the child than the child label costs. Equivalently, the split must capture correlations among several attributes. (Derived, consistent with Wallace & Boulton's "economical encoding" framing; not quoted from any source.)
- **A CRP prior offers a drop-in label cost.** Anderson's coupling prior and a CRP both assign a closed-form probability to a partition from its counts (exchangeable). −log₂ of that probability is a principled "number of categories" cost for create and merge decisions in both hierarchies. Sanborn et al.'s single-particle result also legitimizes Cobweb-style *greedy* commitment as a rational process model.
- **How MDL and CU can coexist in TRELLIS.**
  1. Keep CU (information form) for *sorting and restructuring* in both trees, where it is fast and local.
  2. Use codelength for decisions CU does not make: whether a pair becomes a chunk, which cut through each tree defines the active grammar, and when to delete or merge globally.
  3. Optionally replace the /K average with the MDL-corrected partition score and compare omission and commission behavior.

  Because both are entropy-based and computed from the same node counts, they share sufficient statistics and add no new bookkeeping.
- **The basic level is the codelength-optimal cut.** Corter & Gluck's basic level is where CU peaks. The MDL analogue is the cut that minimizes total codelength, which can be computed exactly by tree DP (Implications, P3). This could replace the hand-set "maturity τ" cut that the project's memory notes identify as a key lever.

### Gaps
- Gluck & Corter (1985, CogSci proceedings) could not be retrieved, so the original information-theoretic derivation is cited via Corter & Gluck 1992 and the Cobweb/4L formula.
- The commonly cited AutoClass chapter (Cheeseman & Stutz, *Advances in Knowledge Discovery and Data Mining*, 1996) could not be verified; only the CrossRef-indexed Stutz & Cheeseman version was.
- Nothing found in the published literature that compares average-CU with an MDL-corrected partition score inside Cobweb. This is open and testable.

---

## Q8. Incrementality, tractability and guarantees: online MDL, incremental model merging, rate–distortion, local search, identification in the limit, spectral methods, and feeding parses back without self-reinforcing errors

### Takeaway
There are workable recipes for incremental MDL: prequential scoring; Chen-style parse-once, trigger-propose, delta-evaluate loops; Stolcke's interleaved incorporation and merging; de Marcken's estimate-commit-undo loop; Brent's per-utterance commitment against a whole-corpus objective; and change detection in the style of StreamKrimp. Guarantees come in three kinds. Asymptotic: universality, Bayes-mixture convergence, MDL learnability from positive data. Structural: Clark & Eyraud's identification in the limit for substitutable languages, which rests on *contexts*, that is, on TRELLIS's context hierarchy. Statistical consistency: spectral methods without local optima, but only when the tree topology is observed or restricted. Unsupervised learning must feed its own parses back, because nothing else supplies structure. The documented safeguards are re-parsing flat data, soft or multiple parses, undo moves, probation, and forward (prequential) evaluation.

### Cited Findings
**Incremental procedures** (details in Q4/Q5)
- Chen 1995: incremental, parses each sentence once, triggered moves — [arXiv](https://arxiv.org/abs/cmp-lg/9504034).
- Stolcke & Omohundro: on-line interleaving of incorporation and merging — [arXiv](https://arxiv.org/abs/cmp-lg/9409010).
- de Marcken: local changes with estimated description-length deltas and undo — [arXiv](https://arxiv.org/abs/cmp-lg/9611002).
- Brent: incremental search over successive corpus prefixes — [arXiv](https://arxiv.org/abs/cs/9905007).
- StreamKrimp: code-table switches on distribution change — [DOI](https://doi.org/10.1007/978-3-540-87479-9_62).
- Online constrained learners can match or beat ideal learners — [Pearl et al. 2010](https://doi.org/10.1007/s11168-011-9074-5).

**Rate–distortion and resource-rational views**
- Chunking as lossy compression — [Nassar et al. 2018](https://doi.org/10.1037/rev0000101).
- Resource-rational analysis — [Lieder & Griffiths 2020](https://doi.org/10.1017/s0140525x1900061x).
- HVM's compression–generalization (rate–distortion) trade-off — [Wu et al. 2025](https://arxiv.org/abs/2410.21332).
- "Distributional Clustering of English Words," a precedent for context classes (cited at title level; abstract not retrieved) — [Pereira, Tishby & Lee, ACL 1993, pp. 183–190](https://aclanthology.org/P93-1024/).

**Identification and learnability**
- "Language identification in the limit" — [Gold 1967, *Information and Control* 10(5):447–474](https://doi.org/10.1016/s0019-9958(67)91165-5).
- Inductive inference of formal languages from positive data — [Angluin 1980, *Information and Control* 45(2):117–135](https://doi.org/10.1016/s0019-9958(80)90285-5).
- Clark & Eyraud formalize Harris's substitutability "and make it the basis for a learning algorithm from positive data only for a subclass of context-free languages," with "a polynomial characteristic set, and thus ... polynomial identification in the limit." "It is not necessary to identify constituents in order to learn a context-free language — it is sufficient to identify the syntactic congruence" — [Clark & Eyraud 2007, *JMLR* 8](https://jmlr.org/papers/v8/clark07a.html).
- Positive-evidence learnability under MDL — [Chater & Vitányi 2007](https://doi.org/10.1016/j.jmp.2006.10.002); [Hsu, Chater & Vitányi 2011](https://doi.org/10.1016/j.cognition.2011.02.013).

**Spectral methods (consistency, no local optima)**
- A spectral algorithm for latent-variable PCFGs. "Under a separability (singular value) condition," it gives "statistically consistent parameter estimates," resting on "a tensor form of the inside-outside algorithm" and a PAC-style bound — [Cohen et al., *JMLR* 15 (2014)](https://jmlr.org/papers/v15/cohen14a.html); [ACL 2012, pp. 223–231](https://aclanthology.org/P12-1024/); [NAACL 2013, pp. 148–157](https://aclanthology.org/N13-1015/).
- "EM suffers from local optima, while recent work using spectral methods cannot be directly applied since the topology of the parse tree varies across sentences." The "unmixing" strategy works for restricted classes, and identifiability is checked via Jacobian rank — [Hsu, Kakade & Liang 2012 (arXiv 1206.3137)](https://arxiv.org/abs/1206.3137).

**Self-training, Viterbi training, and ambiguity**
- "Viterbi Training Improves Unsupervised Dependency Parsing" — [Spitkovsky et al., CoNLL 2010, pp. 9–17](https://aclanthology.org/W10-2902/).
- "Effective Self-Training for Parsing" — [McClosky, Charniak & Johnson, NAACL 2006, pp. 152–159](https://aclanthology.org/N06-1020/).
- Chen notes that "our predicted Viterbi parse can stray a great deal from the actual Viterbi parse, as errors can accumulate as move after move is applied. To minimize these effects, we process the training data incrementally" — [Chen 1995](https://arxiv.org/abs/cmp-lg/9504034).
- Likelihood training over all parses causes "structural optimization ambiguity," which "arbitrarily selects one among structurally ambiguous optimal grammars," and "structural simplicity bias," which underuses rules. Sentence-wise "parse-focusing" reduces the parse pool — [Park & Kim, Findings of ACL 2024 (arXiv 2407.16181)](https://arxiv.org/abs/2407.16181).
- Compound PCFGs marginalize latent trees "with dynamic programming," so the sentence likelihood comes from inside-style computation — [Kim, Dyer & Rush, ACL 2019, pp. 2369–2385](https://aclanthology.org/P19-1228/).

### Inferences
- **Why parses must be fed back.** Without gold trees, the only source of structure is the learner's own analysis (EM, Viterbi or self-training). Every successful MDL induction system above (Chen, de Marcken, Brent, Stolcke) uses current-grammar parses as evidence. The question is not *whether* but *how safely*.
- **Safeguards, ranked by how strongly the literature supports them.**
  1. **Store yields, not trees, and re-parse** (de Marcken's flat storage; DreamCoder's refactoring). Committed parses must never become ground truth.
  2. **Score forward (prequential).** A chunk that only "explains" its own past parses but does not predict the *next* sentences raises cumulative codelength. Forward scoring is the cleanest guard against self-confirmation.
  3. **Undo and delete moves with hysteresis** (de Marcken; Slim/Krimp pruning).
  4. **Soft or multiple parses** (inside-outside expected counts, or a few particles) rather than one Viterbi or greedy parse, especially early. Sanborn et al. show one particle is often enough for *categorization*, but grammar structure is more ambiguous (Park & Kim).
  5. **Probation for candidates** with a significance margin (de Marcken's 1.96 test; HVM's p = 0.05).
  6. **Tempered likelihood (η < 1)** if over-generalization persists (Safe Bayes).
  7. **Curriculum** (Solomonoff's training sequences; "less is more"), for example short sentences first.
- **The binarization problem (46% bracket agreement on the medium grammar) is expected under MDL.** Equally compressive binarizations of a flat constituent have equal codelength, which is Park & Kim's structural optimization ambiguity. If the goal is "reusable ideas" rather than treebank brackets, evaluate with binarization-invariant measures (flattened n-ary chunks, yield-based omission/commission). Alternatively, let a chunk grow n-ary in GoKrimp style (extend while gain is positive), so that MDL rather than an arbitrary binarization decides granularity.
- **Clark & Eyraud justify the context hierarchy.** Their positive result rests on the syntactic congruence (contexts), not on finding constituents. TRELLIS's context hierarchy is exactly where learnability guarantees live. Candidate chunks whose context distribution matches an existing context class ("substitutable") should get a bonus. This is the MDL version of a constituency test.
- **Spectral methods fit v1, not v2.** L-PCFG spectral learning assumes observed tree skeletons, which is close to v1's gold-unlabeled-trees setting. Hsu et al. note it does not apply directly when topology varies, which is v2's setting. Spectral methods could initialize or verify *category* learning given trees, but not resolve unsupervised bracketing.
- **Tractable "global optimality," in summary:**
  1. exact DP for cuts within each hierarchy;
  2. alternating optimization across the two hierarchies (monotone in total codelength);
  3. greedy merges with submodular-style constant-factor guarantees;
  4. beam or lookahead for chunk-then-merge sequences (Stolcke);
  5. periodic consolidation ("sleep") with re-parsing;
  6. irreducibility invariants for asymptotic optimality.

### Gaps
- No source fetched on simulated annealing or stochastic local search specifically for MDL grammar induction; their mention here is conventional knowledge, not a sourced claim.
- The NIPS 2012 venue for Hsu, Kakade & Liang could not be confirmed (the dblp lookup failed); arXiv only.
- JMLR page numbers for Clark & Eyraud (vol. 8) were not confirmed.
- McClosky et al. and Spitkovsky et al. are cited at title level; their abstracts were not retrieved.

---

## Implications for TRELLIS v2

### Takeaway
Adopt **one currency, bits**, computed from the counts Cobweb already stores, and use it at three time scales:
- per sentence: a prequential test-then-train codelength, with sentence probability from the inside algorithm (or the greedy parse as an upper bound);
- per candidate chunk: a Slim / de Marcken compression gain with an adaptive, significance-guarded threshold, evaluated at the ancestor level that maximizes gain;
- per consolidation ("sleep"): an exact DP over cuts in each hierarchy, plus delete, merge and irreducibility passes.

Keep CU for Cobweb's local sorting. "Minimize the number of chunks" and "maximize chunk quality" then become two terms of one sum: definition and pointer costs against uses × bits saved per use.

### Cited Findings
The proposals below are synthesized from the cited works above. The most load-bearing sources:
- prequential = cumulative log-loss = marginal likelihood, and exact Jeffreys/KT equivalence for multinomials — [Grünwald & Roos 2019](https://arxiv.org/abs/1908.08484);
- the Slim gain formula and its usage-only estimate — [Smets & Vreeken 2012](https://vreeken.groups.cispa.de/pubs/2012/slim-smets,vreeken.pdf);
- Stolcke's DL prior plus Dirichlet marginals, chunk and merge operators, and lookahead — [Stolcke & Omohundro 1994](https://arxiv.org/abs/cmp-lg/9409010);
- Chen's incremental parse-trigger-delta loop — [Chen 1995](https://arxiv.org/abs/cmp-lg/9504034);
- de Marcken's add, delete, undo, locality, and significance test — [de Marcken 1996](https://arxiv.org/abs/cmp-lg/9611002);
- Goldsmith's pointer-cost fragmentation penalty — [Goldsmith 2001](https://aclanthology.org/J01-2001);
- Kieffer–Yang irreducibility — [Kieffer & Yang 2000](https://doi.org/10.1109/18.841160);
- information-theoretic CU — [Lian et al. 2024](https://arxiv.org/abs/2409.12440);
- the misspecification and Safe-Bayes remedy — [Grünwald & van Ommen 2017](https://doi.org/10.1214/17-ba1085);
- library-as-prior incrementality — [Solomonoff 2003](http://raysolomonoff.com/publications/nips02.pdf); [DreamCoder](https://doi.org/10.1145/3453483.3454080);
- the utility-problem shape of chunk value — [Minton 1988](https://cdn.aaai.org/AAAI/1988/AAAI88-100.pdf).

### Inferences: ranked proposals

**Notation.** For a Cobweb node v: n_v is its count and n_{v,a} the count of value a of an attribute with K values. For content concepts, the attributes are LEFT and RIGHT child-class ids at the active context cut, plus any boundary or seam features already used. For context classes, the attributes are member ids and neighbor-class ids. The cut κ_C (content) and κ_X (context) are the sets of nodes currently acting as grammar symbols. A Dirichlet-multinomial codelength with α = ½ is written
`DM(v) = −log₂[ Γ(Kα)/Γ(n_v+Kα) · Π_a Γ(n_{v,a}+α)/Γ(α) ]`.
It is order-independent and equals the prequential KT codelength.

---

**P1 (highest priority). Make the prequential codelength of the sentence stream TRELLIS's master score, and make the coding model identical to the generator.**
- *Objective*: `L_preq(D) = Σ_t −log₂ P_{G_{t−1}}(s_t)`, where `P_G(s) = Σ_T P_G(s, T)` is computed by the **inside algorithm** over the current cut grammar. This fits the inside-outside pivot. If staying greedy, use `−log₂ P_G(s, T_greedy)`, an upper bound (greedy ≥ Viterbi ≥ exact).
- *Incremental computation*: before training on s_t, parse it and record its codelength (test), then train (Cobweb add/create/merge/split). Each node's contribution under a fixed-derivation approximation is DM(v) from its live counts, so the running total equals `Σ_{v∈κ_C∪κ_X} DM(v)` plus category label costs. It can be refreshed at any time without replay, because the KT code is exchangeable.
- *Why first*: (i) it is incremental by construction; (ii) structure is paid for implicitly (no arbitrary model code, which avoids the crude two-part failure); (iii) **commission is penalized automatically** because the next real sentence gets less probability under an over-general grammar, provided generation and coding use the same normalized distribution (the context-filtered ancestor-pool sampler must *be* the coded distribution, or be replaced by it); (iv) τ, smoothing and margins can be chosen by L_preq rather than by generation metrics, in line with the project rule never to optimize generation grammaticality directly.
- *Diagnostics to log*: the L_preq curve, plus a two-part decomposition (model bits vs. data bits) for interpretability. Read high model bits as an omission tendency and high data bits as a commission tendency.

**P2. A per-chunk commit test: compression gain at the max-gain abstraction level, with a significance margin (replaces the fixed τ gate).**
- *Statistics*: a pair table over adjacent active symbols in current parses (like BPE/RePair's pair index, or Chen's triggers). For each candidate (X, Y), keep z = co-usage, x, y = usages, and s = tokens in the cover. All are maintained incrementally as each parse is produced.
- *Content gain (exact)*: Slim's ΔL(D) from Q5. *Approximate*: `G_content(X,Y) ≈ z·[log₂(s·z/(x·y)) − log₂e] − L_def`, where `L_def ≈ 2·log₂|κ| + (label cost of a new symbol)`. The adaptive threshold is `z* = L_def / (PMI − log₂e)`: strongly bound pairs need few uses, weakly bound pairs many.
- *Climb to the best level*: for ancestors (X↑i, Y↑j) in the context tree, compute G and pick the argmax. This is the principled version of the climbing-ancestor gate.
- *Context (substitution) gain*: add `G_context(XY) = bits saved by coding the chunk's occurrences as a member of its best-matching existing context class k`, versus coding X and Y separately in their classes. This rewards chunks that are *substitutable* with existing units (an MDL constituency test, following Clark & Eyraud) and counters Goldwater-style over-chunking of collocations. Compute it from the chunk's left/right neighbor counts against class k's context distribution (difference of DM terms).
- *Commit rule*: commit if `G_content + G_context − λ·Δlog₂(match steps) > m(z)`, where m(z) is a margin such as de Marcken's 1.96-σ test, an HVM-style p < 0.05 test, or c·√z. Keep committed chunks under probation, and **delete if gain later falls below −m** (hysteresis). Before final commit, re-parse a bounded buffer of recent yields to replace estimated usages with exact ones (Slim: exact gain for the best estimated candidate; DreamCoder-style refactoring).
- *Expected effect*: a smaller, more reusable inventory (minimal viability). Context gain carries the generalization pressure, and the time term (λ) controls parser match cost (Minton).

**P3. Periodic consolidation ("sleep"): an exact codelength-optimal cut per hierarchy, then delete, merge and irreducibility passes.**
- *Exact cut by tree DP* (globally optimal within the cut family): for each node v bottom-up,
  `best(v) = min{ L_leaf(v), L_label(children of v) + Σ_{u∈ch(v)} best(u) }`,
  where `L_leaf(v)` is the codelength of all data routed through v when v is used as a single symbol (DM over its pooled counts), and `L_label` is the cost of identifying the child (DM or CRP over child counts) plus structure bits. This costs O(|tree|) per hierarchy. Because content attributes refer to context-class ids, **alternate** cut optimization (κ_X given κ_C, then κ_C given κ_X). Each step cannot increase total codelength, so the procedure converges to a joint local optimum that is exact within each block.
- *Irreducibility invariants* (Kieffer–Yang / Sequitur, as MDL versions): no remaining adjacent active pair with positive gain (P2); every chunk's gain positive, otherwise delete and inline (de Marcken); no two content concepts or context classes whose merge lowers total codelength (Stolcke merge, evaluated with DM differences).
- *Lookahead*: evaluate "chunk + immediately implied merges" jointly with a small beam (width 3–10, as in Stolcke), since chunks often pay only after merging.
- *Trigger*: run every k sentences, or when the windowed L_preq rises (StreamKrimp-style change detection).

**P4 (optional, more invasive). An MDL-corrected category utility for Cobweb's own operators in both trees.**
- Replace the /K average with `ΔL_split(p) = N_p·[ Σ_k CU_info(c_k) − H(K|p) ] − ΔL_params`, or add a CRP/coupling label cost (Anderson). Both come from the same counts.
- Expected effects: the basic level shifts with N, and the number of children is naturally limited.
- Risk: it changes tree shape globally. A/B test against the v1 CU on omission/commission and qualitative tree inspection before adopting.

**P5 (optional). Treat the chunk library as the prior (Solomonoff/OOPS/DreamCoder), with a curriculum.**
- Once committed, a chunk is a symbol of the reference language. Future sentences and future candidate chunks built on it are coded more cheaply automatically, because codelengths are computed under the current grammar. This is the "preserve incremental additions" mechanism. Retirement through P2/P3 hysteresis keeps it honest.
- Present data in an easy-to-hard training sequence (Solomonoff), for example sentences ordered by length or depth, and compare L_preq against random order.

**P6 (guardrails for unsupervised feedback; apply with P1–P3).**
- Store raw yields in a replay buffer, never committed trees.
- Re-parse under the current grammar during consolidation.
- Early on, prefer inside-outside expected counts, or two or three particles, over the single greedy parse.
- Use η-tempered likelihood (Safe Bayes) in P1–P3 if commission persists.
- Report binarization-invariant evaluations, because MDL is indifferent among equally compressive binarizations.

**How "minimize the number of chunks" trades off against "maximize chunk quality" (the answer in one line).**
`Total bits = Σ_chunks L_def + Σ_tokens (pointer cost ≈ log₂(s/usage)) + Σ_sentences −log₂ P(s | G)`.
A chunk is worth keeping iff `uses × bits saved per use (≈ PMI − log₂e, plus context gain) > its definition cost + its share of the longer pointers (fragmentation cost (x+y)·H₂(·)) + λ·match cost`. Quality (bits per use) and parsimony (definition and pointer costs) are therefore weighed in the same unit. The optimum is a finite inventory, and it becomes more specific as N grows.

**Suggested experiment order** (cheap to expensive):
1. P1 logging only, with the existing v1 learner, to see whether L_preq ranks the known-good configurations correctly.
2. P2 replacing τ on the small and medium grammars.
3. P3 sleep with tree-DP cuts.
4. P4 A/B test.

Judge each step by omission/commission (Langley & Stromsten), L_preq, and qualitative tree inspection, not by optimizing generation scores directly.

### Gaps
- The closed-form identities used above (DM codelength = prequential KT codelength; the PMI approximation of Slim's gain; ΔL_split = N·[ΣCU − H(K)]; the tree-DP cut) are my derivations from the cited definitions. They should be unit-tested numerically against brute-force codelengths on small grammars before relying on them.
- No published system was found that couples two Cobweb hierarchies under one MDL objective. P1–P4 are therefore untested design proposals, not reported results.
- How generation in TRELLIS (ancestor pools filtered by context class) maps onto a normalized probability distribution needs to be worked out. Without it, P1's commission penalty does not apply.
- 2025–2026 coverage is limited, because the shared web-search budget ran out. Recent items found include Kozma & Voderholzer (ESA 2026), HVM (ICLR 2025), Lan et al. (ACL 2024) and Park & Kim (ACL Findings 2024). A targeted follow-up search of 2025–2026 venues (ACL, NeurIPS, ICLR, CogSci, ACS, ICGI) for "MDL grammar induction" and "incremental chunking compression" is recommended.
