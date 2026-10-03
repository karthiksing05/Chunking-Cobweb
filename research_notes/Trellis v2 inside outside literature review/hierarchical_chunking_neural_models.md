# Hierarchical Chunking in Neural Sequence Models, Masked/Diffusion Training Regimes, and Interpretability Evidence: Guidance for a Masked-Modeling Regime over a Chunked Context Window in TRELLIS v2

Notes compiled 2026-10-03. Coverage runs from the classic boundary-cue papers (1955) to August 2026. Dates in parentheses give the first arXiv posting and, where I could verify it, the venue. "2025–26" items are flagged as recent. Wherever TRELLIS evaluation is discussed, these notes use Langley/Stromsten-style omission/commission errors.

---

## 1. How do neural sequence models learn chunk boundaries and abstraction levels? (dynamic chunking, hierarchical tokenization, entropy patching)

### Takeaway
Neural "dynamic chunkers" decide boundaries in one of four ways:
- **Uncertainty spikes:** BLT's next-byte entropy, and the entropy-spike variant of dynamic token pooling.
- **Dissimilarity of adjacent representations:** H-Net and DLCM.
- **Learned keep/delete gates or block scorers:** MrT5 and Charformer.
- **Fixed or heuristic boundaries:** Hourglass, MEGABYTE, SpaceByte and AU-Net.

The empirical sweet spot is **2–3 levels with roughly 3–6× compression per level**. Deeper hierarchies show only a "promising trend". None of these systems keeps an explicit, reusable chunk inventory: chunks are recomputed for each input. TRELLIS's stored, categorized chunks are therefore a real point of difference (interpretability and reuse), not a gap to close.

### Cited Findings

**Summary table**

| System (date; venue) | How boundaries are decided | Levels | Explicit reusable chunks? | Headline result | Source |
|---|---|---|---|---|---|
| H-Net (Jul 2025; ICLR 2026) | Learned router: cosine dissimilarity of adjacent projected states, p_t ≥ 0.5 | 1 or 2 stages tested (recursive) | No (implicit, per input) | Byte-level 2-stage H-Net matches a BPE Transformer of twice its size; ~3.6–4× data efficiency on DNA | [arXiv](https://arxiv.org/abs/2507.07955), [HTML](https://arxiv.org/html/2507.07955v2), [ICLR 2026](https://mlanthology.org/iclr/2026/hwang2026iclr-dynamic/) |
| Byte Latent Transformer (Dec 2024) | Next-byte entropy from a small separate byte LM: global threshold or "monotonic rise" | 3 tiers (local enc → latent global → local dec) | No (patches computed during data loading) | Matches Llama 3 (FLOP-controlled) with up to 50% fewer inference FLOPs | [arXiv](https://arxiv.org/abs/2412.09871), [HTML](https://arxiv.org/html/2412.09871v1) |
| Dynamic Token Pooling (Nov 2022; ACL 2023) | Autoregressive boundary predictor, trained end-to-end (stochastic reparam.) or supervised by subword-tokenizer segmentations, conditional-entropy spikes, or linguistic (whitespace) boundaries | 1 shortening level | No | Faster and more accurate than vanilla and fixed-length pooling at equal compute | [arXiv](https://arxiv.org/abs/2211.09761v2), [ACL](https://aclanthology.org/2023.acl-long.353) |
| Hourglass (Oct 2021; Findings NAACL 2022) | Fixed-rate down/upsampling | Fixed (U-shaped) | No | Beats Transformer baseline at equal compute; SOTA Transformer on ImageNet32; more efficient enwik8 | [arXiv](https://arxiv.org/abs/2110.13711) |
| MEGABYTE (May 2023; NeurIPS 2023) | Fixed-size patches | 2 (global over patches, local within) | No | Sub-quadratic attention, million-byte sequences | [arXiv](https://arxiv.org/abs/2305.07185) |
| SpaceByte (Apr 2024; NeurIPS 2024) | Heuristic: larger "global" blocks applied only after space-like bytes | 2 | No | Outperforms other byte-level models; roughly matches subword Transformers at fixed compute | [arXiv](https://arxiv.org/abs/2404.14408), [NeurIPS](https://neurips.cc/virtual/2024/poster/95677) |
| Charformer / GBST (Jun 2021; ICLR 2022) | Enumerates candidate blocks of sizes 1..M per position, scores them with a learned network, softly mixes, then downsamples at a fixed rate | 1 | No | 28–100% faster than byte- and subword-level Transformers, quality competitive | [arXiv](https://arxiv.org/abs/2106.12672) |
| MrT5 (Oct 2024; ICLR 2025) | Learned delete gate after a fixed number of encoder layers | 1 | No | Up to 75% shorter sequences at accuracy comparable to ByT5 | [arXiv](https://arxiv.org/abs/2410.20771v3), [ICLR](https://iclr.cc/virtual/2025/poster/29408) |
| AU-Net (Jun 2025; NeurIPS 2025) | Rule-based pooling: bytes → words → word pairs → up to 4 words | 3–4 | No | Shallow hierarchies tie strong BPE baselines; deeper ones show a "promising trend" | [arXiv](https://arxiv.org/abs/2506.14761), [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/871d547a8922ca600eea526a1bd40b2c-Abstract-Conference.html) |
| Large Concept Model (Dec 2024) | Fixed: one sentence = one "concept" (SONAR embedding) | 2 (sentence over token) | Concept space is a fixed pretrained encoder | Feasibility study; MSE, diffusion and quantized variants | [arXiv](https://arxiv.org/abs/2412.08821) |
| DLCM (Dec 31, 2025) | Learned semantic boundaries from cosine dissimilarity of adjacent token embeddings (H-Net-like) | 2 (token → concept) | No | At R=4 tokens/concept: +2.69% average on 12 zero-shot benchmarks at matched inference FLOPs | [arXiv](https://arxiv.org/html/2512.24617v1) |
| Bolmo (Dec 2025) | Non-causal boundary predictor that uses a little future context | 2 | No | Byte-level 7B/1B models made from Olmo 3 with <1% of original pretraining tokens | [AI2 blog](https://allenai.org/blog/bolmo), [arXiv](https://www.arxiv.org/pdf/2512.15586) |

**H-Net mechanism details (Hwang, Wang & Gu; Jul 2025, ICLR 2026)**
- **Routing.** The router computes q_t = W_q x̂_t and k_t = W_k x̂_t, then p_t = ½(1 − cos(q_t, k_{t−1})). A boundary is set when p_t ≥ 0.5, and the first position is always a boundary — [H-Net HTML v2](https://arxiv.org/html/2507.07955v2)
- **Downsampling and smoothing.** The downsampler keeps only boundary vectors. Dechunking applies an EMA, z̄_t = P_t ẑ_t + (1−P_t) z̄_{t−1}, then confidence-weighted upsampling with a straight-through estimator — [H-Net HTML v2](https://arxiv.org/html/2507.07955v2)
- **Ratio loss.** L_ratio = N/(N−1)·((N−1)·F·G + (1−F)(1−G)), where F is the fraction of positions selected and G is the mean boundary probability. The loss is minimized at F = G = 1/N. The 1-stage model uses N = 6; the 2-stage model uses N₀ = N₁ = 3 — [H-Net HTML v2](https://arxiv.org/html/2507.07955v2)
- **Encoder/decoder layers.** Both use Mamba-2 (SSM) layers. The authors argue that SSMs give a strong inductive bias for compression, even on BPE inputs — [H-Net HTML v2](https://arxiv.org/html/2507.07955v2)
- **Learned chunks.** In the 1-stage model, boundaries cluster near whitespace (~4.8 bytes/chunk). In the 2-stage model, stage 1 marks word-initial positions and spaces, and stage 2 groups multi-word units such as "the backbone" and "such as" — [H-Net HTML v2](https://arxiv.org/html/2507.07955v2)
- **Results.**
  - Robustness on perturbed HellaSwag: 39.0 vs 20.2 (Large) and 42.8 vs 22.2 (XL) for H-Net vs a BPE Transformer.
  - Chinese XWinograd-zh rises from 59.9 to 66.3.
  - DNA data efficiency improves 3.6× — [H-Net HTML v2](https://arxiv.org/html/2507.07955v2)
  - The abstract claims the model "qualitatively learn[s] meaningful data-dependent chunking strategies without any heuristics or explicit supervision" — [arXiv abstract](https://arxiv.org/abs/2507.07955)
- **Follow-up H-Net++ (Aug 2025).** Trained on a 1.4B-token Persian corpus, it reduces bits-per-byte by 0.159 vs BPE GPT-2-fa and is 53% more robust to ZWNJ corruption — [arXiv 2508.05628](https://arxiv.org/abs/2508.05628)

**BLT mechanism details (Pagnoni et al.; Dec 2024)**
- **Patching rule.** A boundary is placed when H(x_t) > θ_g (global threshold), or when H(x_t) − H(x_{t−1}) > θ_r (an "approximate monotonic" rise detector). Entropies come from a 100M-parameter byte LM (14 layers, d=512, 512-byte sliding window) — [BLT HTML](https://arxiv.org/html/2412.09871v1)
- **Patching alternatives compared.** Strided, whitespace ("space patching"), BPE (Llama 3, ~4.4 bytes/token) and entropy patching. Mean patch size was 4.5 bytes (entropy) and 6.1 bytes (space) on the training mix, and models were trained at average patch sizes 6 and 8 — [BLT HTML](https://arxiv.org/html/2412.09871v1)
- **Hash n-gram embeddings.** n = 3..8, with 500k hash buckets. Patching runs as a lightweight preprocessing step during data loading, with no stored vocabulary — [BLT HTML](https://arxiv.org/html/2412.09871v1)
- **Results.**
  - Matches Llama 3 under FLOP-controlled training while using up to 50% fewer inference FLOPs.
  - Noisy HellaSwag: 64.3 vs 56.9. CUTE: 54.1 vs 27.5.
  - At 8B, BLT-Entropy beats Llama 3 on 4 of 7 tasks — [BLT HTML](https://arxiv.org/html/2412.09871v1)
  - It is the "first FLOP controlled scaling study of byte-level models up to 8B parameters and 4T training bytes" — [arXiv abstract](https://arxiv.org/abs/2412.09871)

**Concept-level models (2024–2026)**
- **LCM (Dec 2024).** A "concept" is a sentence embedded in SONAR space, which covers up to 200 languages in text and speech. The model predicts the next sentence embedding autoregressively. The authors tried MSE regression, diffusion-based generation and a quantized SONAR space, scaling to 7B parameters on ~2.7T tokens — [arXiv 2412.08821](https://arxiv.org/abs/2412.08821), [GitHub](https://github.com/facebookresearch/large_concept_model)
- **SONAR-LLM (Aug 2025).** It "thinks" in SONAR space but is trained with token-level cross-entropy back-propagated through a frozen SONAR decoder. This drops LCM's diffusion sampler and restores a likelihood objective; models range from 39M to 1.3B — [HF paper page](https://huggingface.co/papers/2508.05305)
- **DLCM (ByteDance Seed / Manchester / Mila / Tsinghua / M-A-P; Dec 31, 2025).** Four stages: encode → dynamic segmentation (cosine-dissimilarity boundary probability) → concept-level reasoning → token decoding — [arXiv 2512.24617](https://arxiv.org/html/2512.24617v1)

**Other relevant 2025–26 items**
- **Haltiuk (Aug 2026, position paper with preliminary measurements).** Argues that next-byte prediction and boundary placement in byte-level LMs are two separable distributions, so capabilities and boundaries could be transferred independently — [arXiv 2608.03599](https://arxiv.org/abs/2608.03599)
- **Unsupervised chunking with a hierarchical RNN (2023; extended in *Computational Linguistics* 2025).** Models word→chunk and chunk→sentence composition. The authors report that "the emergence of the chunking structure is transient during the neural model's downstream-task training" — [arXiv 2309.04919](https://arxiv.org/abs/2309.04919v2), [CL 2025](https://preview.aclanthology.org/credits/2025.cl-3.4)

### Inferences
- **Every boundary rule above can be computed from statistics a Cobweb learner already has, with no gradients:**
  - next-element predictive entropy (BLT);
  - dissimilarity between adjacent elements' representations (H-Net/DLCM);
  - a learned or recognition-based keep gate (MrT5, a Cobweb recognition threshold).

  The only thing gradient training adds is the end-to-end co-adaptation of the router and the predictor. In TRELLIS that role can be played by explicit thresholds plus a target compression band.
- **Neural chunkers need explicit anti-degeneracy pressure.** Examples are H-Net's ratio loss, BLT's thresholds tuned to hit an average patch size, and DLCM's R. TRELLIS v2 will need an analog, such as a per-level target branching band of about 3 children, or a simplicity criterion handled elsewhere in the project.
- **Chunk sizes converge across systems.** H-Net stage 1 gives ~4.8 bytes ≈ a word, and stage 2 groups ~3 of those. AU-Net uses words → pairs → 4-word groups. DLCM uses 4 tokens per concept. Together these suggest that for natural text, TRELLIS v2 should expect about 2 levels of useful chunking above the word (phrase-like, then clause-like) before returns diminish. On synthetic CFGs, the grammar's depth fixes the number of levels.
- **The HRNN "transience" result is a warning about implicit structure.** When nothing in the objective rewards keeping chunk structure, it fades. TRELLIS's explicit chunk memory avoids this, but only if the training signal keeps rewarding chunk-level prediction (see §4 and the Implications).
- **H-Net and DLCM place boundaries on dissimilarity of adjacent states, i.e., where the representation's role changes.** This is a context-taxonomy question in TRELLIS terms: a boundary falls where adjacent elements stop sharing a well-supported context concept.

### Gaps
- I did not verify which boundary source (end-to-end, entropy-spike, Unigram-supervised or whitespace) won in Dynamic Token Pooling. The abstract reports only that dynamic pooling beats vanilla and fixed pooling.
- I found no study that measures *chunk reuse or consistency across contexts* in H-Net, BLT or DLCM. None of them maintains a chunk inventory that could be inspected.
- 2026 coverage of dynamic chunking beyond H-Net++, Bolmo, DLCM and Haltiuk is likely incomplete, because the session's web-search budget ran out.

---

## 2. Classic boundary cues (successor variety, transitional probability, branching entropy) and their neural descendants

### Takeaway
The oldest boundary cue — high uncertainty about what comes next — is exactly what BLT re-implements at the byte level. Harris counted possible successors, Saffran et al. measured transitional probabilities, and Tanaka-Ishii used branching entropy. Infants use this cue after minutes of exposure, so it is a cognitively plausible primitive for TRELLIS. TRELLIS can apply it over categories (concepts), not just surface tokens.

### Cited Findings
- **Harris (1955), "From Phoneme to Morpheme," *Language* 31(2):190–222.** Morpheme boundaries can be found by counting how many distinct phonemes can follow a given prefix ("successor variety"). Boundaries fall at peaks of this count — [JSTOR](https://www.jstor.org/stable/411036)
- **Saffran, Aslin & Newport (1996), "Statistical Learning by 8-Month-Old Infants," *Science* 274(5294):1926–1928.** After about 2 minutes of exposure to a continuous syllable stream, infants distinguished "words" (within-word transitional probability 1.0) from part-words that spanned word boundaries (TP ≈ 0.33). Transitional-probability dips therefore serve as segmentation cues — [Science](https://www.science.org/doi/10.1126/science.274.5294.1926)
- **Tanaka-Ishii (2005), "Entropy as an Indicator of Context Boundaries," IJCNLP 2005.** Shows that uncertainty about the next token after a sequence (branching entropy) increases at context boundaries, verified on Chinese and Japanese with web-search counts — [ACL Anthology I05-1009](https://aclanthology.org/I05-1009/)
- **Jin & Tanaka-Ishii (2006).** Applied branching entropy to unsupervised Chinese word segmentation — [ACL Anthology P06-2056](https://aclanthology.org/P06-2056.bib)
- **Neural descendants.**
  - BLT's two patching rules, H(x_t) > θ_g and H(x_t) − H(x_{t−1}) > θ_r, are the global and "rising" forms of branching entropy, computed by a small byte LM — [BLT HTML](https://arxiv.org/html/2412.09871v1)
  - Dynamic Token Pooling includes "spikes in conditional entropy" as one supervised boundary source — [arXiv 2211.09761](https://arxiv.org/abs/2211.09761v2)

### Inferences
- **The "monotonic rise" rule is the most useful variant for an incremental learner.** It is robust to slow drift in absolute entropy as the model learns, which matters because Cobweb's predictive distributions sharpen over training.
- **Computing successor uncertainty over category symbols is TRELLIS's natural extension.** Examples are the basic-level concept of the next element, or its context concept. This is what the user's "BPE over generalizations of words" means operationally. Category-level transitional statistics are far denser than word-level ones (fewer types, more tokens per type), so they should reach stable boundary decisions with fewer sentences. That claim is testable (see Implications, P4).
- **Cobweb itself can act as the "small entropy model".** BLT trains a separate 100M-parameter byte LM for this. In TRELLIS, the context taxonomy's pattern completion of the right-neighbor attribute already gives a predictive distribution whose entropy can be read off incrementally.

### Gaps
- I found no work comparing word-level and category-level branching entropy for segmentation sample efficiency. It looks like an open, cheap experiment for TRELLIS.

---

## 3. Tokenization as compression: BPE, Unigram LM, optimality of greedy merging, and superword tokenizers

### Takeaway
BPE is greedy grammar-like compression.
- **Theory.** Optimal pair-merging is APX-complete, but greedy BPE is a constant-factor approximation of the compression *utility* (0.333 < α ≤ 0.625). This is good news for greedy chunk acquisition.
- **Compression alone is not the right target.** Minimum-token segmentation does not improve downstream quality, while a *balanced* token-frequency distribution (Rényi efficiency) predicts quality well.
- **Multi-word ("superword") units help when learned after subwords** (2025–26).

### Cited Findings
- **BPE origins.** BPE began as a data-compression algorithm (Gage 1994). Sennrich et al. adapted it to segment words into subword units for open-vocabulary NMT — [Sennrich, Haddow & Birch, ACL 2016](https://aclanthology.org/P16-1162/)
- **Unigram LM tokenization (Kudo 2018).** Learns a probabilistic subword vocabulary and supports sampling multiple segmentations ("subword regularization") — [ACL 2018](https://aclanthology.org/P18-1007/)
- **Unigram vs BPE for pretraining.** Unigram LM aligns better with morphology than BPE and matches or beats it when pretraining LMs (Bostrom & Durrett) — [Findings EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.414/)
- **Gallé (2019).** Links BPE to dictionary-based compression. At a fixed vocabulary budget, "the fewer tokens an algorithm needs to cover the test set, the better the translation" (BLEU) — [EMNLP-IJCNLP 2019, pp. 1375–1381](https://aclanthology.org/D19-1141/)
- **Zouhar et al. (2023), "Tokenization and the Noiseless Channel."** Proposes Rényi efficiency of the token unigram distribution as a tokenizer-quality measure. With α=2.5 it correlates with BLEU at Pearson 0.78, versus −0.32 for compressed length. Rényi efficiency penalizes distributions with very high- *or* very low-frequency tokens — [ACL 2023](https://arxiv.org/pdf/2306.16842)
- **Zouhar et al. (2023), "A Formal Perspective on BPE."** Formalizes BPE as combinatorial optimization and proves via submodularity that greedy BPE is a (1/σ(μ*))(1−e^{−σ(μ*)})-approximation of the optimal merge sequence, where σ is total backward curvature. The empirical lower bound is about 0.37. The paper also gives an O(N log M) implementation — [Findings ACL 2023](https://preview.aclanthology.org/setup/2023.findings-acl.38)
- **Kozma & Voderholzer, "Theoretical Analysis of BPE" (arXiv Nov 2024; ESA 2026).** Optimal pair encoding is APX-complete. BPE approximates its *compression utility* within 0.333 < α ≤ 0.625, but its approximation of *compressed length* can be Ω(n). An "EvenOdd" algorithm reaches 0.5 — [arXiv 2411.08671](https://arxiv.org/pdf/2411.08671), [ESA 2026 (LIPIcs)](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2026.80)
- **Schmidt et al. (2024), "Tokenization Is More Than Compression."** PathPiece segments into the *minimum* number of tokens; across 64 LMs (350M–2.4B parameters) fewer tokens did *not* improve downstream performance. Pre-tokenization matters, and BPE initialization of the vocabulary helps — [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.40)
- **SuperBPE (Mar 2025).** A two-stage curriculum: first learn subwords with merges blocked across whitespace, then lift the block to learn "superwords".
  - At a 200k vocabulary it uses up to 33% fewer tokens.
  - An 8B model gains +4.0% on average over 30 tasks (+8.2% on MMLU) with 27% less inference compute.
  - Segmentations are "more uniform in per-token difficulty" — [arXiv 2503.13423](https://arxiv.org/pdf/2503.13423)
- **BoundlessBPE (Schmidt, Reddy, Tanner, Pinter; COLM 2025).** Merges complete pretokens into superwords (e.g., "of the"), giving a more uniform token distribution and up to 15% more bytes per token — [arXiv 2504.00178](https://arxiv.org/pdf/2504.00178)

### Inferences
- **Greedy merging is theoretically respectable** (a constant-factor approximation of compression utility), which supports the project's stay-greedy stance for *acquiring* chunks. The Ω(n) length result and the PathPiece null result show that maximal compression is the wrong *selection* criterion on its own.
- **Rényi efficiency is a ready-made diagnostic for TRELLIS's chunk vocabulary.** Computing Rényi efficiency (α≈2.5) of each level's chunk-unigram distribution would flag over-splitting (many rare one-off chunks, which the INSIDE_OUTSIDE.md notes call "one-off symbols") and over-merging (a few giant, very frequent chunks). It is cheap, incremental and interpretable.
- **SuperBPE's two-stage curriculum supports a level-wise schedule in TRELLIS.** SuperBPE (subwords, then cross-whitespace superwords) and the 2-stage H-Net both grow structure level by level. For TRELLIS this means stabilizing level-1 chunks (POS/phrase-like) before allowing merges across level-1 boundaries.
- **"BPE over generalizations" is grammar-based compression over a typed alphabet.** Merges are proposed over concept symbols (e.g., basic-level categories) instead of surface words. Because frequency spreads over far fewer types, it should both compress more and generalize further.

### Gaps
- I found no theoretical analysis of BPE-style merging over *category* alphabets, i.e., merging generalizations rather than strings.
- A 2026 follow-up titled "Faster Superword Tokenization" (arXiv 2604.05192) appeared in search results, but I did not review it.

---

## 4. Masked and span modeling as a training regime (and masked LMs as implicit parsers)

### Takeaway
Masking whole correlated units (spans, collocations, chunks) rather than random tokens is consistently better. Random-token masking lets a model "cheat" from a visible collocation partner. Unit masking forces it to use *outside* context, which is exactly the signal a context taxonomy needs.

Masked LMs also carry recoverable constituency structure:
- perturbed masking and contextual distortion extract trees with no extra parameters;
- DIORA learns trees by reconstructing each word from its *outside* representation.

### Cited Findings
- **BERT.** Masks 15% of tokens (80% [MASK], 10% random, 10% unchanged) and predicts them from bidirectional context — [NAACL 2019](https://aclanthology.org/N19-1423/)
- **SpanBERT.**
  - Masks contiguous random spans whose lengths follow a geometric distribution (p=0.2, clipped at 10; mean ≈3.8 words).
  - Adds a Span Boundary Objective (SBO) that predicts each masked token from the two tokens just *outside* the span plus a relative-position embedding.
  - In its ablations, random spans matched or beat linguistically informed masking (whole-word, named-entity, noun-phrase) on most tasks — [TACL 2020](https://aclanthology.org/2020.tacl-1.5/)
- **T5 span corruption.** Replaces corrupted spans (15% corruption rate, mean span length 3 was the chosen setting) with sentinel tokens. The decoder emits only the missing spans — [JMLR 2020](https://jmlr.org/papers/v21/20-074.html)
- **MASS.** Masks one contiguous fragment (about half of a sentence) in the encoder and has the decoder generate it — [ICML 2019](https://arxiv.org/abs/1905.02450)
- **ELECTRA.** Replaces MLM with "replaced token detection" (a discriminator decides whether each token is original or sampled), learning from every position — [ICLR 2020](https://arxiv.org/abs/2003.10555)
- **PMI-Masking (Levine et al.).** Uniform token masking lets MLMs "latch onto shallow local signals". Jointly masking n-grams with high corpus PMI (collocations) unifies and improves on whole-word, entity/phrase and random-span masking, reaching prior methods' performance in half the training time — [ICLR 2021](https://arxiv.org/pdf/2010.01825), [mlanthology](https://mlanthology.org/iclr/2021/levine2021iclr-pmimasking)
- **Perturbed Masking (Wu et al.).** A parameter-free probe. It measures how masking word j changes the MLM's representation/prediction for word i (an "impact matrix") and extracts dependency and constituency trees from BERT — [ACL 2020](https://aclanthology.org/2020.acl-main.383/)
- **Contextual Distortion (Li & Lu).** Scores each span by how much linguistically motivated perturbations (constituency-test-like) distort contextual representations, then chart-parses for the minimum-score tree.
  - Beats prior methods with masked LMs on English, and the state of the art in 6 of 8 languages.
  - Uses no parameter updates — [ACL 2023, pp. 5208–5222](https://aclanthology.org/2023.acl-long.285)
- **DIORA (Drozdov et al.).** "Predicts each word in an input sentence conditioned on the rest of the sentence." Training runs inside-outside dynamic programming over all binary trees, and inference extracts trees with CKY. It reported the best unsupervised binary constituency parsing on WSJ at the time — [NAACL 2019](https://aclanthology.org/N19-1116/)
- **ReCAT.** Contextual inside-outside (CIO) layers alternate:
  - a bottom-up pass that composes low-level spans into high-level spans;
  - a top-down pass in which each span merges information from itself, its siblings and its parents.

  Stacked under a Transformer, this yields "multi-grained representations fully contextualized with other spans". The induced trees agree strongly with human-annotated syntax — [ICLR 2024](https://arxiv.org/html/2309.16319v2)

### Inferences
- **Whether you mask chunks or tokens decides which representation gets trained:**
  - *Masking parts within a chunk while the chunk's other parts stay visible* trains **content** (part-to-part, part-to-whole) predictability.
  - *Masking a whole chunk* and predicting it from its neighbors trains **context** (outside) predictability.

  TRELLIS has two taxonomies for exactly these two roles. A masked regime should therefore deliberately mix both kinds of masking and send each kind's evidence to its taxonomy.
- **SpanBERT's SBO and DIORA's outside reconstruction are the same idea: a span's identity can be recovered from its boundary and outside.** In TRELLIS terms, the outside score of a candidate chunk is "how well its neighbors' concepts predict its content concept". The user already added LEFT/RIGHT child complexity attributes to `content_instance`. That is a boundary-feature move in the SBO spirit, and the neural literature backs extending it to context instances.
- **Perturbed masking can be ported directly to Cobweb as a parameter-free boundary detector.** The impact matrix becomes the change in Cobweb's pattern-completion distribution for element i when element j is also hidden. High mutual impact means "same chunk"; low impact means a likely boundary. This needs no gold trees, which fits v2's unsupervised goal.
- **SpanBERT's finding that random spans compete with linguistic spans holds for token-level neural models.** It does not contradict chunk-level masking for TRELLIS: TRELLIS's spans *are* its learned units, and PMI-masking shows that statistically cohesive units are the better masking unit.
- **ReCAT's CIO layers give the user's "latents take all levels of content before and after" a concrete form.** Bottom-up content composition is followed by a top-down pass that injects sibling and parent context into every span representation.

### Gaps
- I found no masked-modeling study that masks at multiple *learned* hierarchy levels at once (e.g., word, phrase and clause chunks with level-specific objectives). Masking practice is single-level (tokens, spans or n-grams).
- I found no non-gradient or symbolic learner trained with span or chunk masking, apart from the Cobweb/4L line (word-level masked prediction, §7).

---

## 5. Discrete diffusion and insertion-based generation; structure-aware diffusion; status of the "anonymous" 2025 bibliography entry

### Takeaway
Masked (absorbing-state) discrete diffusion is, formally, a *time-agnostic masked model*. Its loss is a weighted mixture of MLM losses, and its sampler is iterated masked completion, which works best in an adaptive "most-confident-first" order. So Cobweb's pattern completion is already the core operator of an absorbing-state diffusion model.

2024–2025 theory from the Wyart group shows that diffusion models learn PCFG-like data by **hierarchically clustering features by their context**, with deeper levels needing more data, and that changes during noising and denoising happen in **correlated chunks**.

The project's anonymous bibliography entry ("Diffusion models for unsupervised sentence parsing through chunk construction", arXiv:2502.12089) **does not match any paper I could find**. The arXiv ID resolves to Favero et al. (ICML 2025), described below.

### Cited Findings

**Core discrete diffusion**
- **D3PM (Austin et al., NeurIPS 2021).** Introduces structured transition matrices (uniform, Gaussian-like, nearest-neighbor in embedding space, and absorbing/[MASK]) and connects absorbing diffusion to masked and autoregressive generative models — [arXiv 2107.03006](https://arxiv.org/abs/2107.03006)
- **MDLM (Sahoo et al., NeurIPS 2024).** Simple masked diffusion is "more performant than previously thought". The Rao-Blackwellized objective reduces to a mixture of classical MLM losses, and the model approaches AR perplexity — [arXiv 2406.07524](https://arxiv.org/abs/2406.07524)
- **SEDD (Lou, Meng & Ermon; ICML 2024 best paper).** A score-entropy loss over ratios of the data distribution. It is competitive with GPT-2 and well ahead of earlier diffusion LMs — [arXiv 2310.16834](https://arxiv.org/abs/2310.16834)
- **Zheng et al. (ICLR 2025).** "Both training and sampling of MDMs are theoretically free from the time variable … and are instead equivalent to masked models." Their first-hitting sampler is equivalent to the original process and runs 20× faster — [arXiv 2409.02908](https://arxiv.org/abs/2409.02908v5), [mlanthology](https://mlanthology.org/iclr/2025/zheng2025iclr-masked)
- **Kim et al., "Train for the Worst, Plan for the Best" (ICML 2025).** MDMs train on an exponentially large set of infilling subproblems, some computationally intractable. Adaptively choosing the decoding order at inference (decode the easy positions first) sidesteps them. Sudoku accuracy rose from <7% to ~90%, beating AR models with 7× more parameters that were trained on the right order — [PMLR v267](https://proceedings.mlr.press/v267/kim25ah.html)

**Scale (2025–26)**
- **LLaDA (Feb 2025).** An 8B masked diffusion LM trained from scratch on 2.3T tokens. It is competitive with LLaMA3-8B on in-context learning and beats GPT-4o on a reversal poem-completion task — [arXiv 2502.09992](https://arxiv.org/abs/2502.09992)
- **LLaDA2.0 (Dec 2025).** Scales to 100B (MoE) by converting AR checkpoints with a 3-phase block-level schedule: grow the block size, run full-sequence diffusion, then return to small blocks — [arXiv 2512.15745](https://arxiv.org/pdf/2512.15745)
- **Block Diffusion (Arriola et al., ICLR 2025).** Interpolates AR and diffusion: autoregressive over blocks, diffusion within each block — [arXiv 2503.09573](https://arxiv.org/abs/2503.09573)

**Structured / hierarchical noise**
- **HDLM, "Next Semantic Scale Prediction via Hierarchical Diffusion Language Models" (Zhou, Wang, Zhang, Tong, Wang, Bates & Jaakkola; NeurIPS 2025, Oct 2025).**
  - The forward process moves each token to "its higher-level ancestor with more abstract semantics": word → cluster token → [MASK]. The reverse process predicts progressively finer semantics.
  - MDLM is the special case with one cluster — [arXiv 2510.08632](https://arxiv.org/abs/2510.08632)
  - Clusters come from k-means over GPT-2 token embeddings, and experiments use one intermediate level.
  - OpenWebText validation perplexity is ≤23.36 with 64 clusters (vs MDLM ≤27.39) and ≤19.22 with 128 clusters.
  - The best cluster count is ~64–128, "approximately the square root of the vocabulary size", which splits generation into two stages of comparable complexity. Stochastic perturbation (ξ=0.8) cuts generative perplexity by >62% via self-correction — [HTML](https://arxiv.org/html/2510.08632v1)
- **DiffusionBERT (ACL 2023).** A "spindle" noise schedule sorts tokens by information. The most informative tokens are masked first in the forward process, so the least informative (easiest) tokens are recovered first in the reverse — [arXiv 2211.15029](https://arxiv.org/pdf/2211.15029)
- **2026 survey.** Argues that discrete diffusion is "fundamentally shaped by how the discrete state space is constructed: the tokenization scheme, the vocabulary topology, and domain-specific structural alphabets" — [arXiv 2607.13431 (Jul 2026)](https://arxiv.org/abs/2607.13431)

**Theory: diffusion on hierarchical (PCFG-like) data — the Wyart group, 2024–2025**
- **Favero, Sclocchi, Cagnetta, Frossard & Wyart, "How Compositional Generalization and Creativity Improve as Diffusion Models are Trained" (arXiv:2502.12089, Feb 2025; ICML 2025, PMLR 267).**
  - On a probabilistic CFG, diffusion models learn composition rules "with the sample complexity required for clustering features with statistically similar context, a process similar to the word2vec algorithm". This clustering "emerges hierarchically: higher-level features associated with longer contexts require more data".
  - Sample complexity is polynomial in context size, so models trained on intermediate amounts of data generate text that is coherent only up to a certain scale. Text and image experiments confirm that "coherence length" grows with data and training time — [arXiv abs](https://arxiv.org/abs/2502.12089)
- **Sclocchi, Favero, Levi & Wyart, "Probing the Latent Hierarchical Structure of Data via Diffusion Models" (Oct 2024; rev. Feb 2025).** In forward-backward (noise, then denoise) experiments, changes happen "by correlated chunks". The chunk length scale "diverges at a noise level where a phase transition is known to take place", confirmed on text and image datasets — [arXiv 2410.13770](https://arxiv.org/abs/2410.13770)
- **Sclocchi, Favero & Wyart, PNAS 122(1), Jan 2025.** Running the backward process from time t shows a phase transition: the probability of reconstructing high-level features (e.g., class) drops abruptly, while low-level features change smoothly — [arXiv 2402.16991](https://arxiv.org/abs/2402.16991v1), [UCL Discovery PDF](https://discovery.ucl.ac.uk/10206005/7/Sclocchi_sclocchi-et-al-2025-a-phase-transition-in-diffusion-models-reveals-the-hierarchical-nature-of-data.pdf)

**Insertion / edit-based generation (variable length)**
- **Insertion Transformer (Stern et al., ICML 2019).** Generates by insertion in arbitrary orders, including balanced-binary-tree order — [arXiv 1902.03249](https://arxiv.org/abs/1902.03249)
- **Levenshtein Transformer (Gu, Wang & Zhao, NeurIPS 2019).** Uses insertion and deletion as primitive operations, enabling iterative refinement — [arXiv 1905.11006](https://arxiv.org/abs/1905.11006)
- **Edit Flows (Havasi, Karrer, Gat & Chen; Meta FAIR; NeurIPS 2025).** A discrete flow over sequences using insertions, deletions and substitutions in a continuous-time Markov chain. It is trained via auxiliary alignment variables and outperforms mask-based models on text and code — [arXiv 2506.09018](https://arxiv.org/abs/2506.09018v2), [NeurIPS](https://neurips.cc/virtual/2025/poster/119031)
- **Insertion Language Models (NeurIPS 2025).** Jointly choose a position and a token to insert. They beat ARMs and MDMs on planning tasks and handle arbitrary-length infilling that MDMs cannot, because MDMs must know the span length in advance — [arXiv 2505.05755](https://arxiv.org/pdf/2505.05755)

**The anonymous bibliography entry**
- `confs/acs-26/paper/main.bib`, key `diffusion-grammar`, lists author "Anonymous", the title "Diffusion models for unsupervised sentence parsing through chunk construction", "arXiv preprint", and arXiv:2502.12089. That arXiv ID is the Favero et al. ICML 2025 paper above, whose actual title is different — [arXiv 2502.12089](https://arxiv.org/abs/2502.12089)
- Searching for the quoted title returned no matching paper. Results only surfaced unrelated unsupervised-chunking work, e.g., [Unsupervised Chunking with Hierarchical RNN](https://arxiv.org/abs/2309.04919v2).
- A broader search for diffusion-based unsupervised constituency parsing or grammar induction also found no such paper. It returned only classic latent-tree-induction work, e.g., [Compound PCFGs](https://aclanthology.org/P19-1228.pdf) and [DIORA](https://aclanthology.org/N19-1116/).
- Local check: the key is not cited in `main.tex` or `chunk.tex` (grep for "diffusion" in the paper's `.tex` files returned nothing).

### Inferences
- **The title in the `diffusion-grammar` entry looks invented or mislabeled.** It should be corrected before any TRELLIS v2 write-up cites it. The real paper is still highly relevant: it is a theory of *unsupervised grammar learning through hierarchical context-clustering*. That is the closest neural-theory analog to TRELLIS's context taxonomy, but it is not a parser.
- **Cobweb pattern completion is already a denoiser.** MDMs are equivalent to time-agnostic masked models (Zheng), and the MDM loss is a mixture of MLM losses (MDLM). So a Cobweb learner that imputes hidden attributes is the denoiser of an absorbing-state discrete diffusion over its attribute slots. "Masked modeling" and "diffusion" are the same regime for TRELLIS. The diffusion framing adds two things: (a) a *schedule* over how much is hidden, and (b) an *order* for re-completion, where Kim et al. show adaptive easiest-first ordering helps a lot.
- **HDLM is a neural version of "noise into generalizations".** Its noise replaces a word with its cluster ancestor before masking. TRELLIS's content taxonomy *is* a hierarchical vocabulary with many levels, so "taxonomic noise" (replace an element with its ancestor at depth d) is immediately available. HDLM's best intermediate level (~√V clusters, two equal-complexity stages) gives a principled target: an abstraction cut where "which concept?" and "which member, given the concept?" carry similar entropy. That is close in spirit to Cobweb's basic level.
- **The user's "noise two tokens at a time by frequency" idea is a coarsening forward process.** Collapsing the most cohesive adjacent pair into a chunk symbol (the BPE direction) and replacing items by their category (the HDLM direction) are both coarsening steps. Read this way, **parsing is the forward (noising) process and top-down generation is the reverse (denoising) process**. DiffusionBERT's information-ordered schedule and the Wyart-group results (hierarchical clustering by context; correlated-chunk changes; a phase transition at the level where categories flip) give both the theory and the diagnostics.
- **TRELLIS's generator is an insertion/edit process, not a fixed-length masked one.** Expanding a chunk into parts changes sequence length. Masked diffusion cannot represent this natively; insertion/edit models (ILM, Edit Flows, Insertion Transformer) can. Block diffusion's "plan the next block, fill within the block" mirrors chunk-level planning followed by content filling.

### Gaps
- No paper found on diffusion used *for unsupervised constituency parsing*, or on syntax-guided diffusion that induces trees. Diffusion work on structure is about controllable generation or hierarchical vocabularies (HDLM), not parsing.
- HDLM tests only one intermediate level; multi-level hierarchical noise is left to future work in the paper.
- Two 2026 items surfaced but were not reviewed: "Unifying Masked Diffusion Models with Various Generation Orders and Beyond" ([arXiv 2602.02112](https://arxiv.org/pdf/2602.02112)) and "A Unification of Discrete, Gaussian, and Simplicial Diffusion" ([arXiv 2512.15923](https://arxiv.org/pdf/2512.15923)).

---

## 6. Interpretability evidence of hierarchical and compositional structure in transformers

### Takeaway
Transformers build multi-level structure in a *residual* manner, keeping lower-level information as they add higher-level features:
- early and middle layers "detokenize" multi-token words into word-level representations at the last token;
- specialized "concept induction heads" copy whole lexical units by attending to word ends;
- syntactic tree distance is linearly recoverable from middle layers;
- on tree-structured data, successive layers reconstruct correlations at successive hierarchical scales (belief-propagation-like up/down passes).

Hierarchical generalization comes from language-modeling objectives, sometimes only after long training ("structural grokking"). The parts a symbolic chunker should copy are residual multi-level context, boundary-anchored chunk representations, and tree-based (not linear) context.

### Cited Findings
- **Structural probe (Hewitt & Manning).** Parse-tree distances are encoded as squared L2 distances under a linear transformation of ELMo/BERT representations, and depth as a norm — [NAACL 2019](https://aclanthology.org/N19-1419/)
- **Tree projections (Murty, Sharma, Andreas & Manning).** Transformers become more tree-structured over training on compositional tasks, and tree-structuredness predicts compositional generalization — [ICLR 2023](https://arxiv.org/abs/2211.01288)
- **Structural grokking (Murty, Sharma, Andreas & Manning).** Transformer LMs generalize hierarchically only after training "far beyond the point when in-domain accuracy has saturated". Depth shows an inverted U, and the tree-structuredness score peaks for models that grok — [ACL 2023](https://aclanthology.org/2023.acl-short.38)
- **Ahuja et al., "Learning Syntax Without Planting Trees."**
  - On five synthetic datasets, the *language-modeling* objective consistently produced hierarchical generalization; seq2seq and classification objectives often did not.
  - Pruning reveals coexisting subnetworks for hierarchical and linear rules.
  - From a Bayesian view, transformers generalize hierarchically when a hierarchical grammar is the *simplest* explanation of the data — [TACL 2025](https://aclanthology.org/2025.tacl-1.6)
- **Tree-Planted Transformers (Yoshida, Someya & Oseki).** "Plant" syntactic trees into the *attention weights* of unidirectional LMs as implicit supervision. On SyntaxGym they significantly outperform vanilla LMs *and* explicit syntactic LMs — [Findings ACL 2024](https://arxiv.org/abs/2402.12691v1)
- **Hierarchical filtering (Garnier-Brun, Mézard, Moscato & Saglietti; ICML 2025).** Encoder-only transformers can implement exact belief propagation on tree-structured data for both root classification and masked language modeling. "Correlations at larger distances, corresponding to increasing layers of the hierarchy, are sequentially included by the network during training." Attention maps and probes show reconstruction at successive length scales — [PMLR v267](https://proceedings.mlr.press/v267/garnier-brun25a.html), [arXiv 2408.15138](https://arxiv.org/html/2408.15138v3)
- **Token erasure (Feucht, Atkinson, Wallace & Bau).** Last-token representations of multi-token words and named entities rapidly "erase" information about previous and current tokens in early layers. This erasure signature was used to read out the LLM's implicit vocabulary (Llama-2-7B, Llama-3-8B) — [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.543)
- **Inner lexicon (Kaplan, Oren, Reif & Schwartz).** LLMs perform "intrinsic detokenization": sub-word sequences are combined into whole-word representations *at their last token*, mainly in early and middle layers. The process is robust to arbitrary splits, typos and out-of-vocabulary words, implying "a latent vocabulary beyond the tokenizer's scope" — [ICLR 2025](https://arxiv.org/abs/2410.05864)
- **Dual-route induction (Feucht, Todd, Wallace & Bau).** Besides token-level induction heads (verbatim copying), LLMs have "concept-level induction heads, which copy entire lexical units" by "attending to the ends of multi-token words". Their outputs carry language-independent word representations that mediate translation — [COLM 2025](https://arxiv.org/abs/2504.03022)
- **Induction heads (Olsson et al.).** The [A][B]…[A]→[B] completion circuit forms at the same point as a sharp jump in in-context learning ability — [transformer-circuits.pub, 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- **Towards Monosemanticity (Bricken et al.).** Sparse-dictionary features "split" into families of finer features as the dictionary grows — [transformer-circuits.pub, 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- **On the Biology of a Large Language Model (Lindsey et al., Mar 2025).**
  - When writing rhyming verse, Claude 3.5 Haiku "often activates features corresponding to candidate end-of-next-line words prior to writing the line", i.e., it plans a high-level target and then fills content.
  - Multi-token words (e.g., French "contraire") are "detokenized" into abstract, multilingual features.
  - Middle layers are more language-agnostic — [transformer-circuits.pub, 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- **Chunks in neural embeddings (Wu, Alaniz, Schulz & Akata).**
  - RNN hidden states trained on sequences with imposed regularities can be extracted as a dictionary of recurring "chunks" that causally influence responses.
  - In LLaMA, similar recurring embedding states correspond to input concepts, and perturbing them activates or inhibits those concepts — [arXiv 2502.01803 (Feb 2025)](https://arxiv.org/abs/2502.01803)
  - Follow-up: [Concept-Guided Interpretability via Neural Chunking, arXiv 2505.11576](https://arxiv.org/abs/2505.11576) (not reviewed in detail).

### Inferences
- **Copy residual multi-level context, not replacement.** Detokenization keeps word-level information at the word's last token while later layers add phrase- and concept-level features, and the residual stream keeps all of them. In TRELLIS, an element's context description at level ℓ should *add* chunk-level slots on top of its lower-level slots (e.g., left/right word, left/right basic-level concept, left/right level-1 chunk concept, parent context concept) rather than replace them. Garnier-Brun's layer-by-scale reconstruction is the neural version of an inside-outside pass over levels.
- **Anchor chunk identity at chunk boundaries.** Neural chunk representations collect at the last token (erasure, inner lexicon), concept induction heads attend to word ends, and SpanBERT's SBO predicts span content from boundary tokens. This supports giving TRELLIS chunks explicit first-part/last-part attributes and using neighbors' boundary parts as context features.
- **Use tree-based context.** Tree-Planted Transformers show that telling a model *which* positions should influence a token (syntactic distance) beats both vanilla attention and explicit tree generation. For TRELLIS, part of the context description should come from tree neighbors (parent, siblings, the parent's neighbors), not only from linear ±k windows. This is exactly the "outside" of inside-outside.
- **Measure early hierarchical generalization.** Ahuja et al. tie hierarchical generalization to the generative objective and grammar simplicity, and Murty et al. show it arrives late for transformers. A symbolic learner with explicit hierarchy could show it *early*: that is a measurable sample-efficiency claim for TRELLIS.
- **Interpretability parallels.** Feature splitting in dictionary learning mirrors Cobweb's coarse-to-fine taxonomy. The poem-planning result supports top-down, chunk-first generation (choose a high-level target, then expand), which TRELLIS already does.

### Gaps
- I found no interpretability study that tests whether LLM-internal "implicit vocabulary" units extend beyond words to multi-word phrases or constituents, i.e., phrase-level erasure. Feucht et al. (2024) cover multi-token words and named entities.
- I found no transformer-circuits work specifically on phrase or constituent-level features. Evidence for "chunk-level" circuits is mostly word-level (detokenization, concept induction).

---

## 7. Sample efficiency and incrementality: BabyLM findings; symbolic hierarchical learners vs neural LMs

### Takeaway
Across three BabyLM rounds (2023–2025), the best sample-efficient LMs came from **new objectives and architectures**. Notably, the 2024 winner, GPT-BERT, mixed causal and masked objectives. Architectures with an explicit hierarchical bias (StructFormer) helped on some tasks but not consistently.

Under data constraints, masked/diffusion objectives beat AR models with repeated epochs. A 2025 ablation, however, attributes much of this to input masking acting as regularization.

The Cobweb line (MacLellan's TAIL lab) shows Cobweb/4L learning masked word prediction faster than transformers in low-data settings, and Cobweb/4V resisting catastrophic forgetting. A 2025 paper explains this robustness by sparse, selective updates and structural reorganization.

### Cited Findings
- **BabyLM 2023 findings (Warstadt et al.).** Published in Proc. BabyLM Challenge at CoNLL 2023, pp. 1–34 — [ACL Anthology](https://aclanthology.org/2023.conll-babylm.1/). Specific winners and curriculum-learning conclusions were not verified in this session.
- **BabyLM 2023 structure-building entry (Momen, Arps & Kallmeyer).** StructFormer-style MLMs that induce hierarchical sentence structure without supervision improved on "some particular tasks" but "fail[ed] to consistently outperform the baseline" across 39 tasks — [arXiv 2310.20589](https://arxiv.org/abs/2310.20589v1)
- **BabyLM 2024 findings (Hu et al.).** Across 31 submissions, a hybrid causal-masked model won. GPT-BERT (Charpentier & Samuel) won both the Strict and Strict-Small tracks.
  - It shifts MLM predictions one position right to align them with next-token prediction, and trains on duplicated data with a causal:masked ratio of about 1:7.
  - "Combining causal and masked language modeling objectives clearly improves performance over single objective baselines" — [arXiv 2412.05149](https://arxiv.org/pdf/2412.05149), [GPT-BERT](https://preview.aclanthology.org/setup/2024.conll-babylm.24)
- **BabyLM 2025 findings (Charpentier et al., First BabyLM Workshop, Nov 2025, pp. 399–420).**
  - Added an interaction track (student learns from teacher feedback), cognitive/linguistic-plausibility evaluations, compute limits and intermediate-checkpoint scoring.
  - "New training objectives and architectures tend to produce the best-performing approaches". Training FLOPs and performance were not fully correlated — [ACL Anthology](https://aclanthology.org/2025.babylm-main.28/)
- **Diffusion vs AR under data constraints.**
  - *Prabhudesai et al. (NeurIPS 2025).* Masked diffusion "significantly outperform[s] AR models when compute is abundant but data is scarce", which they attribute to the random-masking objective acting as implicit augmentation over token orders. They give a closed-form critical compute threshold — [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/0f705a932553c08ebf0d1bc520b7cbc6-Abstract-Conference.html), [arXiv 2507.15857](https://arxiv.org/abs/2507.15857v7)
  - *Ni et al. (Nov 2025).* Report a "crossover": a 1.7B diffusion LM trained on 10B unique Python tokens (~1.5T-token compute budget) overtakes a matched AR coder, and a 1B diffusion LM reaches >56% HellaSwag and >33% MMLU from 1B tokens — [HF paper page](https://huggingface.co/papers/2511.03276)
  - *Counterpoint, Gao et al. (Oct 2025).* Ablations show "random masking of input tokens plays the dominant role", and similar gains come from MLP dropout and weight decay in AR models, so "stochastic regularization broadly enhances data efficiency in multi-epoch training" — [arXiv 2510.04071](https://arxiv.org/abs/2510.04071)
- **Theory of data needs for hierarchical structure.** Cagnetta & Wyart (NeurIPS 2024): on PCFG data, token-token correlations can build representations of the grammar's hidden variables, with longer-range correlations corresponding to deeper variables. A finite training set limits the resolvable correlation range, which grows with training-set size. Confirmed on Shakespeare and Wikipedia — [arXiv 2406.00048](https://arxiv.org/abs/2406.00048v3), [NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/9740da1c07c7b451af14e11523f95271-Abstract-Conference.html)
- **Cobweb/4L (Lian, Baglodi & MacLellan; ACS 2024).**
  - Encodes words and their surrounding context as attribute-value instances and uses the information-theoretic form of category utility.
  - A new multi-concept prediction mechanism "significantly outperforms" single-node prediction.
  - It "learns rapidly and achieves performance comparable to and even superior to Word2Vec", and both outperform BERT on the same task with less training data — [arXiv 2409.12440](https://arxiv.org/pdf/2409.12440)
- **Cobweb/4L journal version (Lian, Wang & MacLellan; *Cognitive Systems Research* 2025).** Cobweb/4L is "hyperparameter-free", "robust across varying scales of training data", and "outperforms transformer-based language models in a low-data setting by learning more rapidly and achieving better final performance" — [TAIL publication page](https://tail.cc.gatech.edu/publications/lian-csr-2025)
- **Cobweb/4V (Barari, Lian & MacLellan; 2024).** Learns visual concepts with less data, keeps performance stable over time and avoids catastrophic forgetting — [arXiv 2402.16933](https://arxiv.org/abs/2402.16933)
- **Explaining the robustness (Barari et al., ACS 2025).** Tests three explanations: adaptive structural reorganization, sparse and selective updates, and information-theoretic learning from sufficient statistics. These factors "help mitigate interference and preserve prior knowledge" — [arXiv 2510.23756](https://arxiv.org/html/2510.23756v1), [TAIL page](https://tail.cc.gatech.edu/publications/barari-acs-2025)
- **Category utility as a prediction objective.** Cobweb's evaluation function is the expected gain in correctly guessing attribute values given category membership. Cobweb's own objective is thus a form of "masked attribute prediction" — [Fisher 1987, *Machine Learning* 2:139–172](https://doi.org/10.1007/BF00114265)
- **Earlier Cobweb language-model work.** MacLellan et al. (ACS 2022), "Efficient Induction of Language Models via Probabilistic Concept Formation" — [arXiv 2212.11937](https://arxiv.org/pdf/2212.11937) (not reviewed in detail here).
- **HVM (Wu, Thalmann, Dayan, Akata & Schulz; Oct 2024, rev. Jun 2025).** A cognitive model that learns chunks and then abstracts "contextually similar chunks" into variables, which is effectively chunks plus categories. It learns a more efficient dictionary than Lempel-Ziv on BabyLM data, its recall correlates with human sequence recall, and LLMs transfer abstract variables less well than humans. An adjustable abstraction level exposes a compression–generalization trade-off — [arXiv 2410.21332](https://arxiv.org/abs/2410.21332)

### Inferences
- **Train with a mix of bidirectional and left-to-right prediction.** BabyLM's strongest signal is that mixing objectives (GPT-BERT's causal plus masked) beats either alone, at a ratio heavily favoring masked prediction. For TRELLIS this suggests training/scoring with mostly bidirectional chunk-masked prediction plus a minority of left-to-right (incremental) prediction. That keeps incremental parsing and generation cognitively plausible while still benefiting from bidirectional context.
- **Masked training's value for Cobweb is the extra prediction problems, not the regularization.** Gao et al. imply that much of the "diffusion data-efficiency" result in neural nets is regularization, which a count-based Cobweb learner does not need. The transferable part is the *any-order, many-subproblem* coverage: each sentence becomes many partial-instance → completion problems at several levels.
- **Do not re-insert every masked variant into Cobweb.** Doing so would distort count statistics. Masked variants should be used for *prediction, scoring and structure selection*, while each observed element is incorporated once (or with fractional weight).
- **Learn levels in order.** Cagnetta & Wyart and Favero et al. predict that deeper chunk levels need more data, because their context correlations are weaker and longer-range, and that each level becomes learnable once the level below has been clustered. A symbolic learner can exploit this by learning levels in order and expressing level-ℓ contexts in level-(ℓ−1) categories, which shrinks the effective context.
- **HVM is the closest external relative of TRELLIS in cognitive modeling.** It shares the chunks-plus-variables design and compression-vs-generalization framing. It is worth citing as related work and possibly using as a baseline.

### Gaps
- I found no head-to-head continual-learning comparison between Cobweb-family *language* learners (Cobweb/4L or TRELLIS) and neural LMs under distribution shift. The forgetting evidence is from vision (Cobweb/4V).
- Specific BabyLM 2025 track winners were not listed on the page I retrieved. I did not check BabyLM 2026 (fourth round) results because the search budget ran out.

---

## Implications for TRELLIS v2

### Takeaway
The literature points to one regime that keeps all of TRELLIS's pillars:
- **What is masked:** whole chunks at several levels of a stored chunk hierarchy.
- **How a chunk is predicted:** first its *content concept at an intermediate (basic-level) abstraction cut* from a *multi-level, chunk-unit context window* using the context taxonomy, then its parts from that concept using the content taxonomy.
- **How the score is used:** as the "outside" term of inside-outside, and as the acceptance test for frontier chunks.

This is masked modeling and absorbing/hierarchical discrete diffusion at the same time, implemented with Cobweb pattern completion and no gradients.

### Cited Findings
These are the anchors for the proposals; full citations are in §§1–7.
- **Masked diffusion = masked modeling; easy-first order matters.** MDMs are time-agnostic masked models ([Zheng et al. 2025](https://arxiv.org/abs/2409.02908v5)); the MDM loss is a mixture of MLM losses ([MDLM](https://arxiv.org/abs/2406.07524)); adaptive easiest-first decoding rescues hard inference ([Kim et al. 2025](https://proceedings.mlr.press/v267/kim25ah.html)).
- **Coarse-to-fine noise works.** Coarse-to-fine "ancestor" noise (word → cluster → mask) beats plain masking, with the best intermediate level near √V ([HDLM](https://arxiv.org/html/2510.08632v1)).
- **Mask cohesive units and predict them from outside.** Masking cohesive units beats random tokens ([PMI-Masking](https://arxiv.org/pdf/2010.01825)); predicting span content from its outside/boundaries works ([SpanBERT](https://aclanthology.org/2020.tacl-1.5/); [DIORA](https://aclanthology.org/N19-1116/)).
- **Hierarchy is learned by context clustering, level by level.** Generative models learn hierarchical grammars by clustering features by context, level by level, with deeper levels needing more data ([Favero et al. 2025](https://arxiv.org/abs/2502.12089); [Cagnetta & Wyart 2024](https://arxiv.org/abs/2406.00048v3)).
- **Boundaries come from local statistics plus a compression target.** Boundaries are decided by next-element entropy rises ([BLT](https://arxiv.org/html/2412.09871v1)) or adjacent dissimilarity with a compression target ([H-Net](https://arxiv.org/html/2507.07955v2)).
- **Representations are residual and boundary-anchored.** Transformers keep residual, boundary-anchored multi-level representations ([Kaplan et al.](https://arxiv.org/abs/2410.05864); [Feucht et al. 2025](https://arxiv.org/abs/2504.03022); [Garnier-Brun et al.](https://proceedings.mlr.press/v267/garnier-brun25a.html)).
- **Context can come from the tree.** Tree-based context guidance beats linear attention ([TPT](https://arxiv.org/abs/2402.12691v1)), and contextual inside-outside passes give multi-grained contextualized spans ([ReCAT](https://arxiv.org/html/2309.16319v2)).
- **Mixed objectives win at small data.** Mixed causal+masked objectives won BabyLM 2024 ([GPT-BERT](https://arxiv.org/pdf/2412.05149)).
- **Cobweb is good at masked prediction with little data and without forgetting.** Cobweb learns masked word prediction faster than transformers at low data ([Cobweb/4L](https://arxiv.org/pdf/2409.12440); [CSR 2025](https://tail.cc.gatech.edu/publications/lian-csr-2025)) and resists forgetting ([Cobweb/4V](https://arxiv.org/abs/2402.16933); [Barari et al. 2025](https://arxiv.org/html/2510.23756v1)).

### Inferences — ranked proposals

**P1 (highest). Adopt "outside-masked chunk prediction" (OMCP) as v2's core self-supervised signal, and use it as the outside score in inside-outside.**
- **Objective.** For a parsed sentence with levels ℓ = 0 (words), 1, 2, …, and for each chunk c hidden at level ℓ:
  `J(c) = log P_ctx( K_b(c) | Ctx_ℓ(c) ) + log P_cont( parts(c) | K_b(c) )`
  - K_b(c) is c's content concept at the basic-level / maturity cut.
  - The first term is Cobweb pattern completion in the **context** taxonomy (the outside model).
  - The second term is the **content** taxonomy's likelihood of the surface parts (the inside model).
  - Keep the second term surface-grounded. LCM's pure embedding regression was brittle, and SONAR-LLM had to restore token-level likelihood.
- **Uses of J.**
  - (a) The outside × inside product scores candidate spans in inside-outside (DIORA/ReCAT logic).
  - (b) Frontier → memory promotion: a candidate chunk is committed once its mean J over occurrences clears a threshold. This addresses INSIDE_OUTSIDE.md's "maintain candidate parses in a frontier and then learn them once we can confirm that they're good enough".
  - (c) Held-out J is the main evaluation number.
- **Why it fits Cobweb.** Category utility already measures expected attribute-prediction gain (Fisher 1987). OMCP extends Cobweb/4L's word-level masked prediction to chunk-level, multi-level masked prediction.
- **What to mask.** Mostly whole chunks, sometimes runs of 2–3 adjacent same-level chunks (T5/SpanBERT-like span lengths) to force longer-range outside use. Mask *parts within a chunk* only for the content-taxonomy term. Avoid masking a part while its collocated partner stays visible (the PMI-Masking shortcut).

**P2. "Chunk the context window" as a residual, multi-level pyramid of concept-ID slots with a constant width in chunk units.**
- **Contents of Ctx_ℓ(c):**
  - (i) ±k same-level neighbors (k≈2–3 chunks) as basic-level concept IDs;
  - (ii) the boundary parts (first/last child concepts) of those neighbors, which is residual level-(ℓ−1) information;
  - (iii) tree-outside slots: the parent's context concept and the sibling concepts, which is the outside of inside-outside;
  - (iv) optionally, one coarser slot from level ℓ+1.
- **Why chunk units.** The window is measured in chunks of the target level, not in words. That keeps it small at every level, as AU-Net, H-Net and DLCM do for their inner networks. Favero/Cagnetta's theory says higher-level categories are identifiable *because* contexts are expressed in already-clustered lower-level units, which shrinks the correlation range to be learned.
- **What level of abstraction the regression needs.**
  - *Inputs:* concept IDs at the basic-level cut of level ℓ and ℓ−1.
  - *Target:* the basic-level concept of the hidden chunk first, then its leaf or parts.

  This is consistent with the project's earlier finding that concept-ID slots plus a maturity-τ cut was the standout representation lever.

**P3. Use taxonomic (ancestor) noise as a "soft mask", unifying the BPE-over-generalizations and diffusion intuitions.**
- **Forward process:** replace an element or chunk with its ancestor at a sampled depth (leaf → basic level → root/[MASK]), as in HDLM. **Reverse process:** context-conditioned descent of the content taxonomy, which is ordinary Cobweb categorization with context attributes.
- **Merge-coarsening as noise:** collapsing a cohesive adjacent pair into its chunk concept is the user's "noise two tokens at a time" idea. Parsing then *is* the forward process, and top-down generation the reverse.
- **Default intermediate cut:** the basic level, which HDLM's √V finding suggests should split the entropy roughly evenly. Schedule lower levels earlier and higher levels later (level curriculum), as the Wyart-group results predict deeper levels need more data.
- **Generation stays locked.** This proposal concerns the *training and scoring* regime and the *framing* of parse/generate as a forward/reverse pair. It does not change generation or optimize generation metrics.

**P4. Pick boundaries from Cobweb's own predictive statistics, computed over categories, with a per-level compression band.**
Options, in order of preference:
- (a) **Category-level branching-entropy rise (BLT's monotonic rule).** Let H_t be the entropy of the context taxonomy's predicted distribution for the next element's basic-level concept. Mark a boundary when H_t − H_{t−1} > θ_r.
- (b) **The existing recognition threshold as arbiter.** The climbing-ancestor gate / maturity τ accepts a merge only when the content taxonomy recognizes it with a mature concept. This is the analog of H-Net's downsampler selection, but it yields a *stored, reusable* chunk.
- (c) **H-Net/DLCM-style adjacent dissimilarity.** p_t = ½(1 − sim(context concept of x_{t−1}, context concept of x_t)), where sim is, for example, the depth of their lowest common ancestor normalized by tree depth.

Two supporting pieces:
- **Calibration.** Adapt θ so that mean children per chunk stays in a target band per level (H-Net: N₀ = N₁ = 3; DLCM: R = 4). Treat the band as a regularizer, not a target, on CFG data whose branching factor is known.
- **Offline validator.** A Cobweb "perturbed masking" impact matrix (§4) needs no gold trees.

**P5. Use confidence-ordered (easy-first) greedy inference as the default approximation to full inside-outside.**
- Commit the span with the highest inside × outside (P1) score first, recompute the outside context of its neighbors, and repeat.
- This is the parsing analog of Kim et al.'s adaptive unmasking order. It keeps the "stay greedy" property the project prefers, avoids the "straying" seen with CKY, and still uses outside context.
- Run full inside-outside as a diagnostic upper bound.

**P6. Use a mixed regime: mostly bidirectional chunk masking, with a minority of left-only (incremental) prediction (GPT-BERT-like, about 1:7 causal:masked).**
This keeps incremental, left-to-right parsing and generation (cognitive plausibility, compatibility with the causal Cobweb-LLM sibling) while letting chunk identity use right context. It also links naturally to the Cobweb-LLM "pair-tree heads": a chunk-level version would use (anchor chunk, context chunk, offset in chunk units), the symbolic analog of concept-level induction heads.

**P7. Keep chunks explicit and anchored at their boundaries; measure reuse.**
- Every neural system reviewed recomputes chunks per input. TRELLIS's stored chunk categories are its interpretability advantage.
- Add first-part/last-part attributes to *context* instances (they already exist for content). Track reuse statistics and per-level Rényi efficiency (α≈2.5) of the chunk distribution to catch "one-off symbols" and over-merged giants.

**P8. Evaluation suite (ranked).** All TRELLIS error reporting uses omission/commission errors.
1. **Held-out multi-level masked chunk infilling.** Accuracy of the hidden chunk's basic-level concept and of its exact content, with two context ablations: (a) a word-only ±k window (Cobweb/4L-style) vs (b) the P2 chunk pyramid. A gap in favor of (b) is direct evidence that chunking the context window helps.
   - Report learning curves (sentences seen vs accuracy) against Cobweb/4L, Word2Vec/CBOW, a small BERT/MLM, an MDLM, and a same-size AR model on identical data.
   - This extends the user's earlier gap-filling win over Word2Vec to chunk-level gap filling.
2. **Unsupervised boundary and bracket recovery** against gold CFG trees, with no gold trees used in training, reported as omission/commission errors per level.
3. **Forward-backward chunk probe** (Sclocchi et al.). Noise at taxonomy depth d or level ℓ, regenerate, and measure the size distribution of changed spans. It should peak at chunk sizes and show an abrupt category flip at the level where noise crosses a concept boundary. This is a structural diagnostic, not a generation metric.
4. **Coherence-length diagnostic** (Favero et al.). The maximum depth or span up to which outputs stay grammatical, as a function of training-set size. It is judged qualitatively, in keeping with the generation lock, and serves to compare learning speed with neural diffusion models' coherence-length curves.
5. **Continual learning across grammar or domain shifts.** Train on grammar A, then B, and measure retention on A against an MLM/MDLM trained sequentially (Cobweb/4V-style test, now on language).
6. **Robustness to perturbation.** Parse stability under single-token substitutions or typos (cf. H-Net/BLT robustness).
7. **Chunk-vocabulary diagnostics.** Rényi efficiency, reuse rate, and the depth of the basic-level cut per level.
8. **Stretch: natural-text minimal pairs.** BLiMP-style minimal pairs ([BLiMP, TACL 2020](https://aclanthology.org/2020.tacl-1.25/)) scored by TRELLIS likelihood, and possibly a BabyLM strict-small subset.

**Cautions carried into the design**
- Do not re-incorporate masked duplicates into Cobweb counts (§7).
- Compression alone is not quality (PathPiece; Kozma & Voderholzer's length result), so pair OMCP with simplicity criteria handled elsewhere in the project.
- Implicit chunk structure fades without objective pressure (HRNN transience), so keep chunk-level masking in the regime permanently.
- Fix the `diffusion-grammar` bib entry (§5) before citing it.

### Gaps
- No published system combines (i) explicit stored chunks, (ii) multi-level masked prediction and (iii) non-gradient incremental learning. P1–P4 are therefore extrapolations from neural and theoretical results, not replications of a tested recipe.
- How to set thresholds for "basic-level cut ≈ equal-entropy split" in a Cobweb taxonomy (vs HDLM's flat k-means clusters) is untested.
- Whether category-level branching entropy (P4a) beats word-level entropy for boundary detection has not been measured anywhere I could find.
- Web searching ended early (session budget exhausted), so October 2026 coverage of new dynamic-chunking, hierarchical-diffusion and BabyLM work may be incomplete.
