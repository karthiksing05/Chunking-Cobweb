# TRELLIS v1 → v2: project context for the inside-outside / chunk-context / unsupervised-grammar literature review

*Source key.* All paths are relative to the ChunkingCobweb repo root unless noted. `src/parse_mh.py` is the TRELLIS core (6,616 lines; whitespace-normalised `diff` against `../trellis_v1/src/parse_mh.py` is empty, so the v1 snapshot and the working repo run the same code). `confs/acs-26/paper/main.tex` is the ACS-26 paper and the newest version (modified 2026-10-03). `memory/` abbreviates the user's saved notes at `~/.claude/projects/-Users-karthiksing05-Documents-ISLE-Research-ChunkingCobweb/memory/`. `../Cobweb-LLM/`, `../trellis_v1/` and `../Papers/` are sibling folders of the repo.

*Evidence tags* show where each claim comes from:
- **[PAPER]**: what the paper claims.
- **[CODE]**: what the code does, from my reading.
- **[MEMORY]**: what the memory notes report. I did not re-verify these unless I say so.
- **[DOC]**: the design notes.
- **[AUDIT]**: `../trellis_v1/PAPER_CODE_AUDIT.md`, dated 2026-07-23 and written against an earlier paper draft. Its code findings still apply because the code is unchanged.
- **[DATA]**: the shipped result CSVs.
- **[LIT]**: Langley 2025 or CHREST.
- **[SIBLING]**: the Cobweb-LLM repo.

---

## 1. What TRELLIS is and what it claims (theory, paper, evaluation)

### Takeaway
TRELLIS implements a unified theory of concepts and chunks built on Cobweb. Every element has a *content* description (its parts) and a *context* description (its surroundings), and the two descriptions are sorted through two linked Cobweb taxonomies. A greedy bottom-up recognition loop parses, a top-down recall loop generates, and learning is plain Cobweb `ifit` over the instances of *gold* parse trees. TRELLIS is close to a literal realisation of the third candidate framework, "probabilistic concept hierarchies", in Langley (2025, §6.3). It reports parse accuracy of 0.89–1.00 and grammaticality of 0.93–1.00 on six synthetic CFG conditions.

### Cited Findings
**Langley (2025), "Concepts and Chunks in Cognitive Systems", Advances in Cognitive Systems 11, 1–12** [LIT]
- **Shared postulates (§4).** Concepts and chunks share five postulates: discrete structures, organisation in memory, access through recognition, response time as the metric, and incremental, piecemeal learning. They differ in emphasis (Table 1): categories vs composites, taxonomic vs partonomic, flexible vs strict matching, recognition vs reconstruction, generalization vs composition. "Chunking supports cumulative learning". — [concepts-chunks.pdf pp.5–6](cobweb-papers/concepts-chunks.pdf)
- **§5.1 Representation.** Calls for one structure that "describes a relational pattern of elements while also characterizing a class of instances". Examples are a word as a letter sequence with alternative spellings, a chess-board structure as "a spatial pattern of pieces that threaten or defend each other, but different types of pieces might occupy the same role", and a passage of music. Logical descriptions vs prototypes vs exemplars is left open. — [concepts-chunks.pdf pp.6–7](cobweb-papers/concepts-chunks.pdf)
- **§5.2 Organization.** Structures refer to constituents (part-of) yet reside in a taxonomy. "Each node in a taxonomic hierarchy could specify constituents that make up that concept, but these components would also reside in the same taxonomy or a kindred one." An open question is whether relations between constituents are decomposable or variable. "Context-free grammars and logic programs have some of the required features, but more appear necessary." — [concepts-chunks.pdf p.7](cobweb-papers/concepts-chunks.pdf)
- **§5.3 Performance.** Recognition by partial match plus reconstruction from relational patterns. Bottom-up grouping goes letters→words→phrases into parse trees, and recall fills in missing letters or words. The same mechanisms could recognise chess configurations "even if they involve novel pieces". Top-down retrieval is also possible. — [concepts-chunks.pdf pp.7–8](cobweb-papers/concepts-chunks.pdf)
- **§5.4 Acquisition.** Learning should build both generalisations and compositions, cumulatively: shared contexts → word classes → phrases → phrasal classes. The key question is whether one learning mechanism suffices or two are needed, as in CFG induction (Wolff 1982; Langley & Stromsten 2000). — [concepts-chunks.pdf p.8](cobweb-papers/concepts-chunks.pdf)
- **§6, three candidate frameworks:**
  - *Discrimination nets (EPAM/CHREST)* "equated chunks with images" and "did not provide a way to acquire high-level chunks in terms of simpler ones". They need pre-organised partonomic input; the suggested fix is nonterminal-introducing grammar induction plus parsing.
  - *Neural nets*: hidden units ≈ chunks, but learning is too slow.
  - *Probabilistic concept hierarchies (Cobweb)*: instances become elements plus relations, concepts specify constituents and surrounding contexts, and parsing sorts candidate chunks through the taxonomy. "Every nonprimitive concept would have constituents and thus also be a chunk." The same mechanisms should work "in domains like chess".

  — [concepts-chunks.pdf pp.9–10](cobweb-papers/concepts-chunks.pdf)

**Paper postulates.** The paper uses one counter per letter, so Cobweb's postulates come first and the extensions continue the numbering. [PAPER]
- **Cobweb**:
  - **R1–R3**: memory holds instances and concepts; an instance is a set of attribute-value pairs; a concept is a per-attribute distribution. — [main.tex L205-218](confs/acs-26/paper/main.tex)
  - **O1–O3**: a single-root taxonomy with instances as terminals; nonterminals summarise; children partition. — [main.tex L258-271](confs/acs-26/paper/main.tex)
  - **P1–P4**: classification and prediction; sorting downward greedily by *category utility* (Cobweb/4V best-first explicitly not adopted); halting at a terminal "or no child is better than the current node"; imputation at the halting node. — [main.tex L312-358](confs/acs-26/paper/main.tex)
  - **L1–L4**: incremental and interleaved; unsupervised; counts updated along the path; four operators (add / new / split / merge), plus a deterministic fringe split at terminals. — [main.tex L377-443](confs/acs-26/paper/main.tex)
- **Extensions:**
  - **R4**: an experience is a set of elements and their local relations (before / left-of). **R5**: every element is primitive or composite, and composites are chunks. **R6**: every element can be described by content or context ("content instances", "context instances"). — [main.tex L475-515](confs/acs-26/paper/main.tex)
  - **O4**: a content taxonomy. **O5**: a context taxonomy. The two are "intertwined in that nodes in one point to nodes in the other". — [main.tex L547-578](confs/acs-26/paper/main.tex)
  - **P5**: parsing iteratively creates a partonomic tree bottom-up. **P6**: parsing sorts candidates through both taxonomies and selects the best *recognized* chunk, which introduces a *recognition threshold*. **P7**: parsing halts when no candidate is recognised or one composite remains ("graceful failure"). — [main.tex L616-656](confs/acs-26/paper/main.tex)
  - **P8**: generation expands a composite top-down. **P9**: generation "recalls candidate decompositions from the content taxonomy and conditions them on the context taxonomy" using a *recall threshold*. Too low a threshold gives very specific nodes; too high gives general nodes "that will mix constituents". **P10**: generation stops when everything is primitive. The content taxonomy plays the role of rewrite rules, and the context taxonomy the knowledge of which rule to choose. — [main.tex L664-716](confs/acs-26/paper/main.tex)
- **Learning bullets.** These are unnumbered; no new mechanisms are introduced.
  - Content and context instances go into the content and context hierarchies.
  - Composite instances go into both taxonomies; primitive instances go only into the context taxonomy.
  - "There is no separate learning mechanism for creating new chunks, as parsing already produces candidate chunks."

  — [main.tex L781-806](confs/acs-26/paper/main.tex)

**Implementation (paper §4)** [PAPER]
- **Representation.** Primitives have only a context field. A composite's content encodes each child by "an identifier that points to the context taxonomy … and a complexity tag". Both live in the context hierarchy's identifier space, "so the content hierarchy never sees a surface word". The complexity tag separates NP=Det+N from S=NP+VP. — [main.tex L820-835](confs/acs-26/paper/main.tex)
- **Parsing:**
  - Search is greedy over adjacent pairs on a frontier.
  - Each candidate's content and context are sorted, and their scores are summed.
  - Admission requires that "the count of some nonroot ancestor exceeds the recognition threshold τ". Primitives are gated on the context hierarchy and composites on the content hierarchy. The gate "suppresses selection early in learning but has little effect later".
  - Candidates are ranked by posterior probabilities, and parsing halts with a partial parse if nothing is recognised.

  — [main.tex L897-919](confs/acs-26/paper/main.tex)
- **Generation.** Generation starts from a seed composite and walks up from its leaf to an ancestor whose count exceeds τ. It samples a decomposition from the constituent pairs stored there, then filters with the context hierarchy. If the filter removes everything, it falls back to a wider pool. — [main.tex L933-956](confs/acs-26/paper/main.tex)
- **Learning.**
  - Desired parse trees are given; the paper calls this "a form of supervised learning", with nonterminals unlabeled.
  - Learning runs bottom-up. Primitives go only through the context hierarchy; composites go through both, after their constituents.
  - "Every composite structure becomes a new node in the hierarchies."

  — [main.tex L970-1001](confs/acs-26/paper/main.tex)
- **Cut text that anticipates the v2 reframing.** A commented-out passage reads: "The content taxonomy describes *how elements combine* … The context taxonomy does the reciprocal work of *representation*: sorting a composite by its surroundings crafts a new identifier for it in the same representational space … letting later composites name it as a part." — [main.tex L583-593](confs/acs-26/paper/main.tex)
- **Cut closure argument.** Another commented-out passage argues that a composite named in the same vocabulary as its parts makes "a single recognition-and-composition mechanism applies unchanged at every depth". — [main.tex L519-533](confs/acs-26/paper/main.tex)
- **Cut learning claim.** A per-commit (within-parse) learning claim was cut with Pat's comment "It can't be right." — [main.tex L988-992](confs/acs-26/paper/main.tex)

**Evaluation design and results** [PAPER]/[DATA]
- **Method:**
  - Follows Langley & Stromsten's GRIDS methodology. *Omission* means the learned grammar is overly specific: it fails to parse or generate legal sentences. *Commission* means it overgeneralises.
  - Omission is scored by parsing novel target-grammar sentences, with partial credit per matched span ("substructure", the unlabeled bracket).
  - Commission is scored by parsing TRELLIS's generations with the target grammar.

  — [main.tex L1024-1054](confs/acs-26/paper/main.tex)
- **Conditions.** The grammars have 3, 6 and 8 nonterminals (small, med, large). Lexicons have 11, 22 and 39 terminals on the 6-nonterminal grammar. Each condition has 400 sentence-parse pairs and one fixed parameter set. The final numbers appear only in figures; the tables are commented out. — [main.tex L1079-1094, L1420-1453](confs/acs-26/paper/main.tex)
- **5-seed final results (n=320).**

  | Condition | Parse accuracy | Grammaticality | Grammatical+novel | Exact match |
  |---|---|---|---|---|
  | small | 1.000 | 1.000 | 0.22 | 1.00 |
  | med | 0.971±0.021 | 0.989 | 0.41 | 0.91 |
  | large | 0.939±0.027 | 0.982 | 0.34 | 0.825 |
  | term_low | 0.948±0.040 | 0.930 | 0.58 | 0.795 |
  | term_med | 0.961±0.012 | 0.956 | 0.47 | 0.815 |
  | term_high | 0.893±0.018 | 0.959 | 0.37 | 0.615 |

  — [confs/acs-26/RESULTS.md L18-25](confs/acs-26/RESULTS.md); [grammar_experiment/*/aggregated.csv](confs/acs-26/grammar_experiment/large/aggregated.csv); [terminal_experiment/*/aggregated.csv](confs/acs-26/terminal_experiment/high/aggregated.csv)
- **Precision = recall = F1 everywhere** [DATA]. Every aggregated row has P = R = F1, because the parser always reduces to one complete binary tree (n−1 spans, the same count as gold). Parse "omission" and parse "commission" are therefore the same number. — [aggregated.csv](confs/acs-26/grammar_experiment/med/aggregated.csv)
- **The paper's figures come from 20-seed runs** [DATA]. `paper/graphics/grids_grammar_experiment.png` is byte-identical (`cmp`) to `confs/acs-26/grammar_experiment_20seed/grids_overlay.png` (20 seeds, endpoint n=300), and `render_paper_pdfs.py` reads the `*_20seed` dirs. The 20-seed endpoints are:

  | Condition | F1 | Grammaticality |
  |---|---|---|
  | small | 0.9996 | 1.000 |
  | med | 0.967±0.016 | 0.989 |
  | large | 0.938±0.019 | 0.985 |
  | term_low | 0.943±0.047 | 0.938 |
  | term_med | 0.930±0.062 | 0.953 |
  | term_high | 0.903±0.032 | 0.966 |

  The large F1 curve runs 0.31 (n=10), 0.77 (20), 0.87 (100), 0.91 (200), 0.94 (300). — [render_paper_pdfs.py L222-226](confs/acs-26/render_paper_pdfs.py); [grammar_experiment_20seed/large/aggregated.csv](confs/acs-26/grammar_experiment_20seed/large/aggregated.csv)
- **Related work.** The paper places TRELLIS among:
  - Labyrinth: no contextual taxonomy, order effects.
  - Trestle: flattens partonomies.
  - Convolutional Cobweb: 2D, multi-level, chunks not explicit.
  - EPAM / CHREST: CHREST's lateral links are likened to TRELLIS's interleaving.
  - Neural nets: they would unify concepts and chunks "if only it included operations for creating new structures" and learned rapidly.
  - CFG induction (Wolff 1982; Langley & Stromsten 2000): inventing nonterminals = chunks, merging = classes.
  - ILP predicate invention.

  — [main.tex L1215-1266](confs/acs-26/paper/main.tex)
- **Future work:**
  - learn syntax "from sentences alone, without parse trees";
  - context-sensitive grammars, "as suggested by its context taxonomy";
  - other hierarchical arenas such as vision;
  - "a new type of large language model that has interpretable structures and is vastly more sample efficient".

  — [main.tex L1291-1305](confs/acs-26/paper/main.tex)
- **Appendix A.** All productions are strictly binary. Rule weights: med NP 3:2; large NP 6:2:1 (bare-noun / AdjP / Nbar); AdjP and Nbar 2:1; AdjP recursion decays as (1/3)^d; RelClause sentences are about 20% of the large corpus. The grammars nest strictly: small (S→NP VP; NP→Det N; VP→V NP), then med (+AdjP, VPobj→NP PP, PP→P NP), then large (+Nbar→N RelClause | AdjP RelClause; RelClause→RelPro VP). The terminal lexicons are low 11, med 22 and high 39 words, with med productions fixed. — [main.tex L1333-1418](confs/acs-26/paper/main.tex)
- **Appendix B formulas:**
  - log P(i|C) = Σ log P(v|C,a), with P(v|C,a) = (n_{C,a,v}+α_a)/(n_{C,a}+V_a α_a).
  - Ranking uses the "class posterior" log P(C|i) = log P(i|C) + log P(C) − log P(i), summed over the two hierarchies (s_c + s_x).
  - Recognition threshold τ_parse = 30. It walks the sort path upward and rejects only if it reaches the root. The paper says it is "essentially transparent" after the first few sentences.
  - Recall threshold τ_gen = 50, using the deepest ancestor whose count exceeds it. Committed chunks are recorded as (parent-leaf, left-child, right-child), indexed by leaf and anchor. Seeds are sentence-root chunks, and slots are filtered by expected context class.
  - α_content = 1e-4 and α_context = 1e-5. The window is 5 tokens with distance weighting. The content bag holds up to 3 concept ids drawn from depth 4.
  - Complexity is visible. References are canonicalised to a recent ancestor, and the paper says stored instances are rewritten "whenever the context tree restructures".

  — [main.tex L1455-1475](confs/acs-26/paper/main.tex)

**Paper-internal inconsistencies I found** (in the current main.tex unless noted):
- **Stale postulate references.** App. B says "the level P6 samples from" and "P7 conditioning". In the current numbering these are P9 and P9's context conditioning; P6 and P7 are now parse postulates. The same passage contains an unfinished sentence: "whereas parsing utilizes the ." — [main.tex L1469](confs/acs-26/paper/main.tex)
- **The split arithmetic does not add up.** The protocol says "five-fold cross validation, holding out … 40 pairs … remaining 320 pairs", and 40 + 320 ≠ 400. The code makes 5 random 80/20 shuffles (320/80) and parse-scores the first 40 held-out items. Commission uses 500 generations at the final checkpoint, not 40. — [main.tex L1089-1093](confs/acs-26/paper/main.tex); [../trellis_v1/experiments/learning_curves.py L196-199, L289-290, L136](../trellis_v1/experiments/learning_curves.py); [AUDIT §1.4](../trellis_v1/PAPER_CODE_AUDIT.md)
- **Captions say five data sets.** The plotted curves are the 20-seed runs above. — [main.tex L1149-1151, L1184-1186](confs/acs-26/paper/main.tex)
- **Lexicon description.** The text calls the lexicon sizes "words per part of speech class", but 11/22/39 are total lexicon sizes. — [main.tex L1082-1084 vs L1402-1416](confs/acs-26/paper/main.tex)
- **Memory describes a different paper version** [MEMORY]. It describes "A Unified Framework of Concepts and Chunks" with a "Table 1" and a separate "Section 8". The current main.tex is titled "A Unified Account of Concepts and Chunks: Extending Cobweb from Categorization to Composition", has no Table 1, and puts future work inside "Concluding Remarks". Memory's "Table 1" numbers (e.g. med 0.96±0.04, large 0.95±0.02, term_low gen 0.91±0.07) do not match the shipped CSVs. — [memory/project_paper_shipped.md](memory/project_paper_shipped.md)

### Inferences
- The v2 language "composition hierarchy / representation hierarchy" is already in the cut paper text (L583-593), and the v1 code behaves that way. Composites are *named* by where their context sorts (label = context-leaf id), while *how* they combine lives in the content tree. The rename is a re-description of v1 roles, not a new mechanism.
- Langley (2025) §5.2's open question is whether relations between constituents are decomposable or variable. v1 hard-codes binary, ordered (left/right) composition, so any non-binary, multi-template or 2D extension departs from v1's representation.
- Because P = R = F1 for complete binary parses, the paper's parse metric equals unlabeled bracket accuracy. That makes it directly comparable to unlabeled-F1 conventions in unsupervised parsing work, apart from trivial-span conventions.

### Gaps
- No numeric results appear in the current paper text, only figures. I could not check whether the rendered main.pdf matches main.tex exactly.
- Who ran the 20-seed experiment and how (n=300 endpoint) is not documented anywhere I found.

---

## 2. Exactly what the content and context instances are, and how the code differs from the paper

### Takeaway
In the faithful configuration, the context instance has:
- 5 before-slots and 5 after-slots of **raw neighbour word ids**, weighted 1/2^(j+1);
- a hidden complexity tag;
- a visible "content-ref" attribute (word id for primitives; content-leaf id for composites, written only at learning time).

The content instance has two attributes per child, giving four in all:
- a *bag of the top-3 depth-4 context-tree concepts* that best fit that child's context instance, stored as leaf ids and canonicalised to depth-4 ancestors at evaluation time;
- a visible complexity tag.

A composite's context is the *words* outside its span, not the neighbouring *chunks*. Chunk context exists only as the optional `chunk_context` mode.

### Cited Findings
**Primitive context instance** [CODE], `build_primitives` [src/parse_mh.py L1295-1366](src/parse_mh.py)
- **Slot mode** (default; `bow=False`). Attributes 0…L−1 are the before-slots, `{word_id_of(i−j−1): w_j, 0: 0}` with w_j = 1/2^(j+1) (`_context_weight` "binary", [L86-103](src/parse_mh.py)). Attributes L…2L−1 are the after-slots. A missing neighbour gives `{0: 0}`, or `{0: w}` if `empty_weighting`.
- **Complexity.** Attribute −2 holds complexity `{C1: 1}`. It is *hidden*: negative attributes are skipped in `log_prob_instance` ([cobweb-private/src/cobweb_discrete_node.cpp L1786-1791](cobweb-private/src/cobweb_discrete_node.cpp)).
- **Content-ref.** Attribute 2L (`content_ref_attr`, [L3941-3948](src/parse_mh.py)) is the **visible** word identity `{word_id: 1}`. If `prim_word_hidden` is set, a shared constant `__PRIMWORD__` is used instead.
- **Neighbours are raw word ids.** The slots hold raw neighbour word ids, never concept ids, unless `chunk_context=True`. [MEMORY] calls this "drift" from the paper's "neighbouring words' CONCEPT identifiers". — [memory/project_faithful_representation_push.md L12](memory/project_faithful_representation_push.md)
- **BOW alternative.** Attribute 0 is a before-bag and attribute 1 an after-bag, each holding summed distance weights; content-ref sits at attribute 2. — [L1299-1320](src/parse_mh.py)

**Composite context instance** [CODE], `create_context_instance` [L1023-1125](src/parse_mh.py)
- **Span.** The before-slots come from the *left child's* `context_before` and the after-slots from the *right child's* `context_after`, i.e. the words immediately outside the composite's span.
- **Complexity.** max(left, right) + 1 at hidden attribute −2.
- **Content-ref timing.** Content-ref is omitted while parsing (`content_ref_id=None`, [L1806-1815](src/parse_mh.py)). It is written only during learning, step 4 of `add_parse_tree`.
- **The `chunk_context_before/after` parameters** ([L1029-1030, L1069-1070](src/parse_mh.py)) override these slots with `{label_path: 1}` concept ids of the *current frontier neighbours*. `evaluate_pair` and `apply_candidate` fill them when `ltm.chunk_context` is set. — [L1782-1804, L2018-2040](src/parse_mh.py)

**Composite content instance** [CODE], `create_content_instance` [L866-1019](src/parse_mh.py)
- **Attributes 0/1** are `ltm._bag_for_context_inst(child.get_context_instance())`, which calls `TopKPoolEncoder.bag_for`:
  1. Score every context-tree node at depth `content_pool_depth` (4) by `log_prob_instance(child_ctx)`.
  2. Keep the top `content_top_k` (3).
  3. For each kept node, store its *highest-count descendant leaf* id with value 1.0. The weighting is binary by default; "posterior" is optional.
  4. A `value_remap` dict pushed to the content tree makes C++ `canonical(v)` map each stored leaf id to its *current* depth-4 ancestor at evaluation time.

  — [cobweb-private/src/cobweb/leaf_remap.py L39-125, L365-424](cobweb-private/src/cobweb/leaf_remap.py); [cobweb_discrete_tree.cpp L60-70](cobweb-private/src/cobweb_discrete_tree.cpp)
- **Attributes 2/3** are the left and right child complexity tags `{C{cplx}: 1}`. They are visible unless `content_drop_cplx`. — [L914-920](src/parse_mh.py)
- **Optional "hint" attributes**, all off in the faithful config: 4… are boundary edge words (`wordid` / `ctxbag` / `posclass`); 20/21 are seam words; 22/23 are child class (the context cluster at a fixed depth). — [L929-1018](src/parse_mh.py)
- **Composite identity.** `label = {context-leaf concept id: 1}`, and `label_path` is that leaf or a cut ancestor (`_cut_ancestor` leaf / basic / maturity, [L593-648](src/parse_mh.py)). The content tree does not use `label_path` ([comment L2076-2077](src/parse_mh.py)). Parents encode a child through the TopK bag of the child's *context instance*.

**Faithful configuration actually run** [CODE]/[MEMORY]
- **Settings:** `context_length=5`, `context_alpha=1e-5`, `content_alpha=1e-4`, `content/context_bl_alpha=10`, `content_pool_depth=4`, `content_top_k=3`, `content_drop_cplx=False`, all hints off, `rank_mode="class_lp"`, `maturity_gate=("climb_ancestor_count", 30)`, `gate_mode="skip"`, `gen_pool_mode="mat"`, `gen_pool_tau=50`, `gen_anchor_mode="maturity"` with τ=20 (unused by the replay generator), `primitives_first=0`.
- **Seeds:** [13, 17, 7, 42, 100]. Seed 23 was dropped as "a genuine outlier producing a type-confused low-terminal generator".
- **Code defaults differ.** `TRELLIS(context_length=3)`, `content_top_k=5`, and `rank_mode` defaults to `'context_forward'` ([L2397](src/parse_mh.py)).

— [../trellis_v1/experiments/run_grammar_experiment.py L136-173](../trellis_v1/experiments/run_grammar_experiment.py); [memory/project_faithful_configuration.md](memory/project_faithful_configuration.md); [RESULTS.md L13-16](confs/acs-26/RESULTS.md)

**Paper vs code: representation differences**
1. The paper says context names "other elements in its vicinity" ([main.tex L511-515](confs/acs-26/paper/main.tex)). The code's slots hold raw word ids at every level, and a composite's context is the surrounding words, not the surrounding chunks.
2. The paper says the content field holds "an identifier … referring to the child itself". The code holds a 3-element bag computed from the child's context instance, plus complexity. That bag is chosen while the primitive child's visible word-id attribute is part of the scored instance. So surface identity shapes the bag even though the content tree stores only concept ids.
3. The paper says stored instances are *rewritten* when the context tree restructures ([L1475](confs/acs-26/paper/main.tex)). The code deliberately does *not* rewrite content-tree values on context-tree restructures; it remaps at evaluation time instead ([comment L4107-4113](src/parse_mh.py)). It rewrites context-tree av_counts only for content-tree SPLIT actions (`_apply_rewrite_rules` [L4009-4046](src/parse_mh.py), called at [L4191-4192, L4206-4207](src/parse_mh.py)).
4. The module docstring still says "Only frozen/accepted chunks are added to BOTH hierarchies; unfrozen candidates go only to the content hierarchy" ([L46-47](src/parse_mh.py)). But `CompositeParseNode.frozen` is only set and serialised ([L775, L2097, L3653, L3718](src/parse_mh.py)); `add_parse_tree` never reads it.

### Inferences
- **What v1 calls chunk context is really word context.** A composite's context is the window of *surface words* around its span. Any higher-level, phrase-level context comes only from `chunk_context=True`, and that mode reads the *frontier at the moment of the merge*. The frontier depends on merge order: gold order in training, greedy order at test time. This is the train/test mismatch INSIDE_OUTSIDE.md alludes to: "chunk context is extremely hard to standardize in a parsing scheme that is inherently greedy".
- **The content bag is a 3-hot sparse code.** It is drawn from a fixed depth-4 slice of the context tree, so its resolution is set by tree depth, not by data. That is why depth (pool_depth) and α levers move results (§5).

### Gaps
- I did not measure how often the depth-4 pool changes, or how often the encoder's dead-leaf rescue path fires during training.

---

## 3. How scoring, the recognition gate, ranking and halting work, and what one parse step costs

### Takeaway
Parsing proceeds as follows:
1. A greedy loop runs over adjacent frontier pairs.
2. Each new pair is evaluated once and cached.
3. A count-based "climbing-ancestor" gate is applied. In the shipped experiments it is effectively vacuous.
4. The admitted pair with the highest **class_lp** is merged. class_lp is the content tree's tree-wide class score plus the context tree's.
5. The loop repeats until one root remains.

Each candidate costs at least two root-to-leaf Cobweb categorisations, one per hierarchy, plus four best-first tree-wide scoring passes of up to 200 nodes each, plus six TopK bag encodings and several diagnostics that the ranker never reads. A sentence of n words needs about 3n candidate evaluations.

### Cited Findings
**The build loop** [CODE], `FiniteParseTree.build` [L2208-2503](src/parse_mh.py)
1. **Primitives.** `build_primitives` categorises each word in the context tree. `stable` is set from the maturity gate on its context-path score. An unstable primitive cannot take part in any merge, because `_find_root_child_by_index` returns only composites or stable primitives ([L1250-1255, L1432-1441](src/parse_mh.py)).
2. **Candidates.** Candidates are adjacent pairs of root children ([`get_parentless_pairs` L1590-1617](src/parse_mh.py)). A pair cache keyed by position indices is invalidated only for pairs that touch the merged nodes. — [L2276-2308, L2492-2498](src/parse_mh.py)
3. **Gate (Stage 1).**
   - If `MERGE_POLICY` sets a frequency gate (basic_level_count > freq_min), use that.
   - Else, if a `maturity_gate=(name, thr)` is passed, admit when `content_score_data[name] > thr`.
   - Else, admit when not `climb_hit_root`.

   — [L2310-2335](src/parse_mh.py)
4. **Rank (Stage 2)**, by `ltm.rank_mode` ([L2381-2465](src/parse_mh.py)):
   - `root_lp`: content root lp + w·context root lp.
   - `class_lp`: content `tree_class_log_prob` + `class_lp_ctx_w`·context `tree_class_log_prob`. This is the shipped ranker.
   - `learned`: a linear model over 23 features ([L2522-2571](src/parse_mh.py)).
   - `context_forward` (the default): ctx leaf lp + 3·log(1+climb count) + 0.05·content root lp.
   - legacy: root lp + 0.3·(content and context leaf lps) + `chunk_pool_weight`·attestation logs.
5. **Commit.** The top-scoring candidate is merged (a stable sort, so ties keep enumeration order) and `apply_candidate` runs ([L1998-2139](src/parse_mh.py)). It creates a composite at position = midpoint of its children, re-parents the children, categorises the new context instance to label it, and builds the content instance. *No learning happens.* The loop stops when nothing is admitted or one root remains ([L2467-2501](src/parse_mh.py)). The comment that ties resolve "right-to-left" is wrong: pairs are enumerated left-to-right. — [AUDIT §2.4](../trellis_v1/PAPER_CODE_AUDIT.md)

**Gate internals** [CODE]
- `_climbing_ancestor` walks from the leaf to the root and returns the first (deepest) node with count > τ, or count/root > ρ when τ < 1. Otherwise it falls back to the root with `hit_root=True`. — [L422-478](src/parse_mh.py)
- **The shipped gate is vacuous** [AUDIT]. `maturity_gate=("climb_ancestor_count", 30)` reads `climb_ancestor_count`, which is the *root's* count when nothing deeper qualifies. So it passes as soon as the root count exceeds 30, a handful of sentences in. P7's graceful failure never fires, and `p_parse_legal` is hard-coded to 1.0. — [AUDIT §1.2](../trellis_v1/PAPER_CODE_AUDIT.md); [learning_curves.py L418](../trellis_v1/experiments/learning_curves.py)
- **The paper partly describes this** [PAPER]. App. B now says the gate is "essentially transparent", but the Fig. 3 caption and §4.3 still describe non-root semantics. — [main.tex L911-914, L924-929, L1466](confs/acs-26/paper/main.tex)
- **Primitive admission.** The `prim_active` fraction is 0.978–0.988 across conditions. — [DATA, aggregated.csv](confs/acs-26/grammar_experiment/med/aggregated.csv)
- **Even non-vacuous count gates do not discriminate** [MEMORY]:
  - the climbing-count gate "separates constituents from non-constituents 0% (100% leak)";
  - the wrong-bracket climb median is 45 vs gold 49;
  - coverage stays 1.0 even at τ=80 or with a relative-support gate.

  — [memory/project_class_lp_heuristic.md](memory/project_class_lp_heuristic.md); [memory/project_cky_induced_grammar.md](memory/project_cky_induced_grammar.md); [memory/project_composite_threshold_sweep.md](memory/project_composite_threshold_sweep.md)

**Score internals** [CODE]
- **`_score_along_path`** ([L481-565](src/parse_mh.py)) computes per-node `log_prob_instance` on the path; `tree_log_prob = tree.log_prob(inst, 200, False)`; `tree_class_log_prob = tree.log_prob_class_given_instance(inst, 200, False)`; `get_basic(200, 100, eval_alpha=bl_alpha, use_root=True)`, cached per leaf; `basic_level_count`, alias `cost`, which is −1 if the basic level is the root; `root_log_prob`; `leaf_log_prob`; and the `climb_*` fields.
- **The class score has no −log P(i) term.** The C++ `log_prob_class_given_instance(inst, max_nodes=200, greedy=False)` does a best-first expansion of up to 200 nodes ordered by log P(i|C). It returns a weighted average, over the expanded nodes, of [log P(i|C) + log(n_C/n_root)], with weights ∝ P(i|C). The node-level function returns `log_prob_instance + log(count/root.count)`, i.e. the joint log P(i, C). App. B's −log P(i) normalisation does not appear. — [cobweb_discrete_tree.cpp L970-1036](cobweb-private/src/cobweb_discrete_tree.cpp); [cobweb_discrete_node.cpp L1773-1777](cobweb-private/src/cobweb_discrete_node.cpp)
- **`tree.log_prob`** is the same weighted average applied to log P(i|C). — [cobweb_discrete_tree.cpp L868-930](cobweb-private/src/cobweb_discrete_tree.cpp)
- **Descent.** `_categorize_dfs` follows the argmax of `log_prob_children_given_instance`, the children's log P(i|c) + log P(c) normalised among siblings, *always down to a leaf*. It uses neither category utility (P2) nor P3's early halting. — [L110-201](src/parse_mh.py); [cobweb_discrete_node.cpp L1757-1771](cobweb-private/src/cobweb_discrete_node.cpp)
- **The halting concept is not what is scored.** The paper scores a candidate "by how well the candidate matches the concept at which sorting halts relative to its competitors" ([main.tex L901-903](confs/acs-26/paper/main.tex)). The code's class_lp is the tree-wide ≤200-node average, not a halting-node score.

**Cost of one `evaluate_pair`** [CODE], [L1756-1994](src/parse_mh.py). At eval checkpoints `leaf_to_chunks` and `content_leaf_transitions` are loaded ([learning_curves.py L404-408](../trellis_v1/experiments/learning_curves.py)), so each new candidate does all of the following:
- **2 categorisations**: a context-tree descent ([L1819-1820](src/parse_mh.py)) and a content-tree descent ([L1824-1825](src/parse_mh.py)).
- **6 TopK bag encodings.** Each scores every depth-4 context node and does a pure-Python DFS of 3 subtrees for best leaves ([leaf_remap.py L250-264](cobweb-private/src/cobweb/leaf_remap.py)). There are 2 for the content instance, then 2 more in each attestation helper, which rebuild the content instance ([L1648-1649, L1721-1722](src/parse_mh.py)).
- **Up to 6 extra content-tree descents** inside `_chunk_pool_attestation` and `_leaf_transition_attestation` ([L1621-1754](src/parse_mh.py)). Their outputs are unused by class_lp.
- **4 tree-wide best-first passes** of up to 200 pops each, two per hierarchy, from the two `_score_along_path` calls ([L1827-1853](src/parse_mh.py)). Each pop scores the node and all of its children.
- **Partition-utility (PU) diagnostics.** Per hierarchy: `pu_for_best_insert` and `pu_for_new_child` at the leaf's parent and at the basic-level ancestor's parent, plus an **uncached** `get_basic(100, 1000)` ([L1855-1909](src/parse_mh.py)). With `use_root=True`, `expected_pmi` enumerates all leaves under every ancestor on the leaf→root path ([cobweb_discrete_node.cpp L584-611, L693-722](cobweb-private/src/cobweb_discrete_node.cpp)).
- **Per-child context scoring** (cached per child object): a descent plus `_score_along_path`. — [L1914-1935](src/parse_mh.py)
- **What decides.** Only `tree_class_log_prob` ×2 and the gate field feed the decision under class_lp.
- **Greedy total.** About (n−1) + 2(n−2) ≈ 3n evaluations per sentence, thanks to the pair cache.

**Beam variant** [CODE], `_build_beam` [L2593-2667](src/parse_mh.py), enabled when `ltm.parse_beam_width > 1` ([L2271-2274](src/parse_mh.py))
- Each state is re-materialised from scratch (`build_primitives` plus replayed merges), and every frontier pair is re-evaluated *without* a cache. That is O(beam·n²) evaluations per sentence.
- States are scored by cumulative additive `_rank_score`, which is hard-coded to the **context_forward** formula, or `learned` ([L2573-2591](src/parse_mh.py)). It ignores `class_lp` and `root_lp`.
- The beam prefers complete parses and falls back to the best partial parse.

**Measured cost** [MEMORY]/[DOC]. A full sweep was "HOURS-slow" ([memory/project_acs26_f1_representation_push.md](memory/project_acs26_f1_representation_push.md)), and "BASIC LEVEL SAMPLING IS SLOWWWW" ([docs/MULTIHIERARCHY.md L149](docs/MULTIHIERARCHY.md)).

### Inferences
- **What class_lp actually measures.** It equals tree_lp plus a likelihood-weighted average of log node-mass. It therefore rewards candidates whose best-fitting nodes are *large, well-populated* concepts, i.e. "belongs confidently to a recognised class". It is not a true posterior. An inside-outside chart score built on Cobweb would need a principled per-span likelihood. Note that the same "prior" weighting, log P(x|c) + log P(c), was also the best-calibrated multi-node readout in Cobweb-LLM (§10).
- **Where the cost goes.** Most per-candidate work is diagnostics and attestations that class_lp ignores. Removing them should make O(n²) or O(n³) span scoring (chart or lattice) far cheaper than today's code implies. The basic-level/EPMI and TopK-encoder costs scale with tree size and would dominate.
- **Beam results do not test class_lp.** Because `_rank_score` ignores class_lp, the memory finding "beam is worse at every width" does not test beam search with the shipped ranker.

### Gaps
- I found no wall-clock profiling per candidate or per sentence. Only operation counts are available.
- The memory beam experiments predate class_lp; their exact ranker settings are not recorded.

---

## 4. The learning loop, what is stored, and how generation samples

### Takeaway
Learning in the experiments works as follows:
1. Replay the *gold* merge sequence with `apply_candidate`.
2. At sentence end, `add_parse_tree` makes four passes: fit context instances; refresh labels and rebuild content instances bottom-up; fit content instances, plus orphan candidate pairs; write content-leaf references back into the context leaves.
3. At each checkpoint, rebuild per-leaf **chunk records**, a discovered PCFG over content leaves, and **leaf transitions**.

Generation samples a whole training sentence-root chunk. It then recursively fills each composite slot by *uniform* sampling from a maturity-level pool (τ=50) of recorded chunks, filtered to those whose own context leaf exactly matches the slot's expected context leaf, with a fallback to the unfiltered pool.

### Cited Findings
**Training loop** [CODE]
- **Data.** Each item is `{"sentence", "merges": [{"left": pos, "right": pos}, …]}`, an ordered bottom-up derivation where composites sit at midpoint positions, e.g. a large-grammar sentence with 9 merges. — [data/cfg_grammar_experiment_large/*.json](data/cfg_grammar_experiment_large)
- **Per sentence:** `build_primitives` (with the gate) → `apply_candidate` for each gold merge, silently skipped if a primitive is unstable → `ltm.add_parse_tree(tree, shuffle=True)`.
- **At checkpoints:** `learn_leaf_transitions` + `learn_chunk_records` over all trained trees, then evaluation with all three RNGs snapshotted and restored.

— [../trellis_v1/experiments/learning_curves.py L376-416](../trellis_v1/experiments/learning_curves.py)

**`add_parse_tree`** [CODE], [L4048-4248](src/parse_mh.py)
- **Step 1.** `ifit` every node's context instance into the context tree, shuffled. Composites strip content-ref; primitives keep their word-id. — [L4083-4113](src/parse_mh.py)
- **Step 2.** Bottom-up label refresh: a composite's label is its context-leaf concept id. Each composite's content instance is **rebuilt** from its refreshed children, giving new TopK bags. — [L4118-4167](src/parse_mh.py)
- **Step 3.** `ifit` composite content instances into the content tree, shuffled. Content-tree SPLITs are applied to context-tree av_counts. The code **also fits leftover adjacent "orphan" candidate pairs** into the content tree only. For complete gold trees this is a no-op; for partial or unsupervised parses it accumulates counts for unadmitted candidates. — [L4172-4207](src/parse_mh.py); [AUDIT §3](../trellis_v1/PAPER_CODE_AUDIT.md)
- **Step 4.** Write the content-leaf concept id (or `label_path` in chunk_context mode) into each composite's context leaf via `increment_attr_value`, which propagates to *all ancestors*. — [L4212-4241](src/parse_mh.py); [cobweb_discrete_node.cpp L154-166](cobweb-private/src/cobweb_discrete_node.cpp)
- **Cobweb actions.** `_ifit_and_update_vocab` maps the NEW / MERGE / SPLIT actions into vocabulary entries and rewrite rules. — [L3990-4007](src/parse_mh.py)
- **No within-parse learning.** In the experiments, training never parses and evaluation never learns. — [AUDIT §1.5](../trellis_v1/PAPER_CODE_AUDIT.md)
- **Self-training hook.** `parse_sentence(learning=True)` feeds the parser's *own* tree to `add_parse_tree`. It is used by met6, not by the paper. — [L5046-5049](src/parse_mh.py)

**Stored structures** [CODE]
- **`learn_chunk_records`** ([L4558-4766](src/parse_mh.py)): `leaf_to_chunks` maps a parent content-leaf hash to a list of records `{L/R_leaf_hash, L/R_word_id, cplx, L_cplx, R_cplx, ctx_leaf, L_ctx_leaf, R_ctx_leaf}`.
- **Sentence roots.** `sentence_root_chunks` holds composites whose context slots are all empty. — [L4621-4630](src/parse_mh.py)
- **Pools:**
  - `leaf_to_bl` / `bl_to_chunks`: max-EPMI basic-level pooling;
  - `leaf_to_mat` / `mat_to_chunks`: the deepest ancestor with count ≥ `gen_pool_tau` (default 20, shipped 50);
  - `bl_to_sentence_root_chunks`;
  - `leaf_to_shapes`: the set of (L_cplx, R_cplx) pairs per leaf.

  These are recomputed from scratch at every checkpoint by re-descending stored content instances. — [L4684-4764](src/parse_mh.py)
- **`learn_leaf_transitions`** ([L4872-4953](src/parse_mh.py)): per parent content leaf, a count plus Counters `L_children` / `R_children` (child content leaves) and `L_words` / `R_words`. This is a marginal production table.
- **Parse-time use.** `_chunk_pool_attestation` reads `leaf_to_chunks` as joint-production attestation, and `_leaf_transition_attestation` reads the transitions as marginal attestation. Only the legacy ranker uses them. — [L1621-1754](src/parse_mh.py)

**Generation** [CODE], `generate_via_chunk_replay` [L4768-4870](src/parse_mh.py)
- **Seed.** A uniform random sentence-root record.
- **Recursion.** A primitive slot emits its word. A composite slot calls `_pick(child_leaf, expected_child_ctx_leaf)`:
  1. Take the pool from mat / bl / leaf.
  2. Filter by `c["ctx_leaf"] == want_ctx`, an *exact* context-leaf match.
  3. If the filter empties the pool, fall back to the unfiltered pool.
  4. Choose uniformly with `random.choice` and recurse.
- **Depth cap.** `max_depth=8` silently drops words. — [L4830-4835](src/parse_mh.py); [AUDIT §3](../trellis_v1/PAPER_CODE_AUDIT.md)
- **Output.** The generated text is re-parsed to return a tree. — [L4863-4869](src/parse_mh.py)
- **Paper wording vs code.** P9 says pool pairs "each … has an associated probability" and that the context hierarchy is used "to rank them" ([main.tex L687-695, L940-945](confs/acs-26/paper/main.tex)). The code samples uniformly and filters with a hard exact match. [MEMORY] says not to weight by count, because uniform sampling is what gives novelty. — [memory/feedback_generation_strategy.md](memory/feedback_generation_strategy.md)
- **Unused faithful path.** The basic-level resampling path described by older postulates, plus masked completion, is still in the code but unused for paper numbers: `generate_sentence` / `_generate_sentence_impl` [L5214-6560](src/parse_mh.py), `_basic_sample` [L5469-5500](src/parse_mh.py), `_expand` [L5618](src/parse_mh.py), `_resolve_bag` [L5798](src/parse_mh.py), masked input [L6169-6170](src/parse_mh.py).
- **Commission metric.** A generation counts as grammatical if every token is in the lexicon and the target CFG's CYK recogniser accepts it. "Novel" means not verbatim in the training sentences. — [learning_curves.py L303-326](../trellis_v1/experiments/learning_curves.py)

### Inferences
- **Chunk records already form a PCFG.** `leaf_to_chunks` plus `learn_leaf_transitions` is a *discovered PCFG* whose nonterminals are content leaves or maturity ancestors and whose rules are recorded (child-identity) pairs with counts. That is the substrate an inside-outside pass would need: rule probabilities per nonterminal. Today it is rebuilt offline at checkpoints, not maintained incrementally.
- **Generation's grammaticality comes from context, not content.** It relies on *exact* context-leaf matching, so it depends on how context leaves fragment. Making contexts coarser or more abstract (chunk context, concept ids) would change the conditioning and could raise commission errors. Any v2 change to context will alter generation even though generation is "locked" (§7).

### Gaps
- I did not trace how often the context filter falls back to the unfiltered pool. RESULTS.md attributes residual commission to it but gives no rate.

---

## 5. Design levers tried, and what happened

### Takeaway
The levers that moved results were *representational*: visible child complexity, a longer and sharper context window, content-bag shape, and α. On ranking, class_lp was the one win. Search changes failed: beam, learned ranker and greedy plus validity gate all lost. CKY over an induced grammar worked but was set aside as "straying too far". Most numbers come from memory notes written in different eras with different harnesses, so they are not mutually comparable. The shipped CSVs are the authoritative endpoint.

### Cited Findings
In the table, "Era" means WEBSTER/hollow-learn (before June 2026), acs-26 push (July 2026), or met6 (the unsupervised branch). Sources: CC = [memory/project_content_instance_cplx_attrs.md](memory/project_content_instance_cplx_attrs.md); FP = [memory/project_faithful_representation_push.md](memory/project_faithful_representation_push.md); F1P = [memory/project_acs26_f1_representation_push.md](memory/project_acs26_f1_representation_push.md); CL = [memory/project_class_lp_heuristic.md](memory/project_class_lp_heuristic.md); CB = [memory/project_clustering_blocker.md](memory/project_clustering_blocker.md); PS = [memory/feedback_parse_strategy.md](memory/feedback_parse_strategy.md); CKY = [memory/project_cky_induced_grammar.md](memory/project_cky_induced_grammar.md); PM = [memory/project_primitive_maturity_gate.md](memory/project_primitive_maturity_gate.md); CZ = [memory/project_acs26_cost_zero_gate.md](memory/project_acs26_cost_zero_gate.md); GS = [memory/feedback_generation_strategy.md](memory/feedback_generation_strategy.md); M6 = [memory/project_composite_threshold_sweep.md](memory/project_composite_threshold_sweep.md); DET = [memory/project_cobweb_determinism.md](memory/project_cobweb_determinism.md); PD = [memory/project_parser_drifts_with_training.md](memory/project_parser_drifts_with_training.md).

| # | Lever | Era / harness | Outcome (numbers) | Status | Src |
|---|---|---|---|---|---|
| R1 | Child complexity tags as content attrs (2→4 attrs) | WEBSTER, seed 13 | F1 78.7→82.0%; EM 47.8→52.2%; step-pick 78.2→87.4%; probe 97.5→99.2%; from-scratch generation still 0% | kept | CC |
| R2 | Complexity visible vs dropped | acs-26 faithful base (pool4 k3, class_lp), 3 seeds | large 0.848→0.899; term_high 0.770→0.907; term_med 0.840→0.871. "Contradicts complexity-is-hidden belief" | kept (in paper) | FP L30 |
| R3 | Context window 5 + context α 1e-5 | faithful base, complexity visible | 3-seed: large 0.951, term_med 0.957 (a1e-5: term_high 0.916). 5-seed after gate/α push: small .974, med .957, large .944, term_low .963, term_med .950, term_high .928 | kept (faithful config) | FP L32-36 |
| R4 | Hint-free baseline (word-id slots, complexity dropped) | faithful, 3 seeds | large 0.777, term_high 0.791, term_med 0.794 ("faithful rep looks ceilinged ~0.85") | baseline | FP L17, L28 |
| R5 | Boundary + seam word-id features (edge and junction words) | acs-26, 5 seeds | +0.03 then +0.02 F1. small 0.962→0.994; large 0.862→0.903; med 0.920 | later rejected as "hint/drift" | F1P |
| R6 | Child-class feature (context cluster at depth 3–4) | class_lp era | Purity →0.99 but F1 only ~0.85 alone. With class_lp + boundary/seam: large 0.92, term_high 0.818→0.923. Final per-variant: small .984, med .985, large .951, term_low .955, term_med .858 (seed-17 crater .734), term_high .923 | rejected as hint | CL |
| R7 | Concept-id context slots ("C1": `chunk_context` + cut + 1 iteration) | faithful base | large: leaf 0.738 / basic 0.654 (unstable) / maturity τ=20 **0.829**. term_high 0.791→**0.48** (mat20) | "killed as universal": helps dense, hurts sparse | FP L18, L23 |
| R8 | Content-bag shape | faithful base | pool4 k3 best on large .848 and term_med .840, worse on term_high (.770 vs .791). pool_basic ~.55–.68. Posterior-weighted bag and k1 no help | pool4 k3 kept | FP L27, L30 |
| R9 | content_alpha 1e-5 (finer split) | class_lp without hints | term_med 3-seed 0.803→0.892 (seed17 0.577→0.938) but "relocates the seed craters". 5-seed: small 1.000, med .945, large .913, term_low .882, term_med .898, term_high .818 | not kept | CL |
| R10 | Per-variant context (large α1e-5 len5; term_high α1e-4 len3) | class_lp + hints | large .923→.951, but the same change took term_high .923→.888 | superseded by one shared config | CL |
| R11 | α / depth / k vs content over-splitting | WEBSTER | α=1.0, d2, k3: leaf mean count 3.4 but F1 85→51%. α=1e-2, d4, k7: 98% of chunks at count<2, F1 77.7%. α=0.1: 98.6%, F1 74.0% | rejected | CB |
| S1 | Old count gate (basic_level_count > τ) | WEBSTER | F1 ~47%, recall ~32%; 22% of sentences got zero chunks | superseded | PS |
| S2 | Climbing-ancestor gate + cnt_root_lp (+0.3 leaf-lp tie-break) | WEBSTER | F1 ~80%, recall ~78%, EM ~50–65%, step-pick 85–94% (118/119 after a 3D sweep) | superseded | PS; [L2352-2378](src/parse_mh.py) |
| S3 | **class_lp ranker** | acs-26 | Gold-trajectory step-pick 98.3% vs 83.4% for cnt_root_lp (~83% for pure PMI). Greedy F1 med 0.859→0.966 (≈ CKY 0.973); large 0.924→0.926. 5-seed: small .997, med .962, large .904, term_low .866, term_med .828, term_high .900 | **shipped** | CL |
| S4 | root_lp (paper-literal generative sum) | faithful base | Best on large (.792) and term_high (.809), worse on term_med (.723 vs .794). Hurts once complexity is visible | rejected | FP L26, L32 |
| S5 | context_forward ranker | acs-26 | large 0.86 (neutral to worse); ~0.55 on faithful base | rejected, yet still the code default | F1P; FP L26 |
| S6 | Supervised learned step-pick ranker (logreg) | acs-26 | 0.836; fails from distribution shift (self-made prefix ≠ gold prefix) | failed | F1P |
| S7 | Beam over elaboration sequences | acs-26 | Worse at every width. Additive cumulative score is anti-correlated with gold, because gold contains rare low-root_lp modifier chunks | failed (note: context_forward scoring) | F1P; [L2573-2591](src/parse_mh.py) |
| S8 | Greedy + production-validity gate | acs-26 | Hard gate: F1 0.30 (dead ends). Soft preference: 0.61 | failed | CKY |
| S9 | CKY over grammar induced from both hierarchies (type = context ancestor at fixed depth; productions mined from training chunks) | acs-26, 5 seeds, boundary+seam rep | small .997 (hybrid), med .973 (depth 3), large .928 (depth 5), term_low ~.950 (depth 6). Type-depth ~4 keeps ~99% of gold brackets valid and makes ~77% of greedy's wrong brackets impossible | proven, then "set aside as baseline"; code not in repo or git history | CKY; [memory/feedback_stay_greedy_clean_pos.md](memory/feedback_stay_greedy_clean_pos.md) |
| S10 | ctx_root_lp gate on top of 4-attr content | WEBSTER | F1 82→55%; step-pick 87→71% | rejected | CC |
| S11 | sum_class_lp per-candidate filter | WEBSTER | Kills 40–60% of gold candidates | rejected | PS |
| G1 | Primitive maturity gate root_log_prob > −12 (replaces 200-sentence warm-up) | acs-26, 5 seeds | small 98.75→98.72%; med 85.51→88.93%; large 85.77→84.17%. τ=−8.5: med 54%; τ=−10 rejects too many in large | superseded by climb count 30 | PM |
| G2 | (cost, 0) composite gate (basic_level_count > 0) | acs-26 Phase 6 | ppl 0.84→1.000 on all six. F1 at convergence: grammar .94/.93/.90, terminal .90/.84/.80. Tighter cost>2/3/4 makes LARGE worse (.68/.61/.58) | rejected: ppl "too permissive"; F1 + gen are the real bars | CZ; [memory/feedback_grids_strict_metrics.md](memory/feedback_grids_strict_metrics.md) |
| G3 | Climb τ sweep | faithful, 5 seeds | gate20: large .944 (α1e-4), term_high .928 (α1e-5). gate50 "catastrophic (starves parser)" | τ=30 shipped | FP L36 |
| G4 | Relative-support gate (τ<1 ⇒ count/root) | met6 | Coverage still 1.0; "added (but didn't need)" | dormant hook | M6; [L445-459](src/parse_mh.py) |
| Gen1 | `_resolve_bag` unpacking variants | WEBSTER | 0% → 12% → 36% → 48% grammatical | superseded | GS |
| Gen2 | Subtree-exchange replay (leaf pool) | WEBSTER, seed 13 | 100% supervised / 98% unsupervised grammatical; ~18% novel | superseded by maturity pool | GS |
| Gen3 | Paper-faithful basic-level resampling (`generate_sentence`) | acs-26 | basic: 47.5% / 11.2% legal (large / term_high). Maturity anchor τ40: 63.8% / 18.8%. Multi-seed reality ~40% / ~20% mean (range 23–67%) | shelved | FP L40-54 |
| Gen4 | Pool width: leaf / basic level / maturity | acs-26, 3 seeds | leaf 100% grammatical but 2–19% novel; BL 70–87% grammatical; **mat τ=50**: large 99.7% grammatical / 44% novel, term_high 98.3% / 31%; τ=90: large novelty 52% but term_high grammatical 94% | mat50 shipped | FP L56-60 |
| Gen5 | Context-class filter + fallback | acs-26 | "You DON'T NEED purity, you need CONDITIONING" | shipped, LOCKED | FP L58; [memory/feedback_gen_locked_qualitative.md](memory/feedback_gen_locked_qualitative.md) |
| U1 | met6 sentences-only training (parser output fed back) then FREEZE | met6, SMALL/MED | τ∈[2,10]: determinism 1.0, coverage 1.0, 12–20 categories, self-embedding .2–.3, gen .82–.88. τ=20: grammatical .40, novelty .70. SMALL τ=8 matches supervised (F1 99.3 vs 100; EM 97.5 vs 100; gen 100 vs 100). MED τ=2: F1-vs-gold 46 vs 84.6 (τ-invariant cap) but gen 98.3 vs 88.3 | exploratory; script deleted (in git history) | M6 |
| U2 | Simplicity via content_alpha chosen by two-part MDL | met6 | At α≈0.1: categories 16→5, productions 177→104, singletons .12→0, MDL 6014→4380. Generation .82→.62. α=1.0 overshoots (MDL 6726) | exploratory | M6 |
| I1 | Cobweb C++ RNG determinism patch | infra | Before the patch, the same config gave F1 47.2–73.1% (σ 11pp); across-seed σ≈7–8pp afterwards | needed. "NOT active in the current environment (checked 2026-09-21)" | DET |
| I2 | Harness fixes (context_length 3 vs 5 mismatch; eval RNG bleeding into training) | acs-26 | Curves under-reported before the fix; afterwards every curve point reproduces single-pass runs | fixed | [RESULTS.md L97-109](confs/acs-26/RESULTS.md) |
| I3 | Corpus VP flattening | WEBSTER→acs-26 | `flatten=("VP","VPobj")` fixed an apparent "parser drift" (term gen 0.87→0.75 became 0.98→0.95). Later commit `2f2c467f` "regenerate CFG data with strictly-binary grammars and no VP flattening"; App A grammars include VP/VPobj | superseded | PD; git log |

### Inferences
- **Context abstraction cuts both ways.** It helps dense, structural variants and hurts lexically sparse ones (R7, R10). The same tension will face v2's "latents take all levels of content before and after". The right abstraction level for context is the central open design variable, not a detail.
- **The win came from the representation, not the search.** Every search-side change (S6–S8) lost to greedy with a better representation, except CKY over a *typed* induced grammar (S9). That suggests inside-outside will help only if spans are typed by a sharp categorical signal (context ancestor at fixed depth) with production statistics, not by raw additive log-probs (S7's failure mode).
- **Unsupervised operation is feasible already.** met6 (U1–U2) shows sentence-only operation is viable on SMALL and generates well on MED. The cost shows up only against gold brackets. A two-part MDL criterion over (categories, productions) already exists conceptually.

### Gaps
- The CKY module (`src/cky_grammar.py`, `_cky_parse.py`) and most `_calib_*` / `_diag_*` harnesses cited by memory are absent from the repo and its git history (`git log --all` returns nothing for them). Their numbers cannot be re-verified locally.
- The met6 test is recoverable from history (`tests/met6/gen_learn_test.py`, last at `8c8fafc6`). The `unsupervised` branch named in memory does not exist locally.

---

## 6. Known failure modes and their diagnosed causes

### Takeaway
Residual errors cluster into four groups:
1. Greedy commits that cannot be undone: non-constituents such as V+Det "found the", plus attachment order.
2. Genuine attachment ambiguity: RelClause/PP and adjective-chain branching, where class_lp ties.
3. Lexical sparsity and POS confusion (V–P, Det–Adj) at 39 terminals.
4. Seed- and order-fragile Cobweb clustering: impure intermediate nodes and over-split leaves.

The gate is vacuous and the Cobweb-native basic level is unreliable. Reproducibility depends on a C++ RNG patch that may not be active.

### Cited Findings
1. **Greedy early commits compound.** "found the" (V+Det) passes because a broad impure content ancestor covers it. Wrong brackets are *invalid productions*, and greedy can't backtrack. — [memory/feedback_stay_greedy_clean_pos.md](memory/feedback_stay_greedy_clean_pos.md); [memory/project_cky_induced_grammar.md](memory/project_cky_induced_grammar.md). [DOC] "if we have an incorrect parse at the beginning, it balloons up" ([MULTIHIERARCHY.md L24](docs/MULTIHIERARCHY.md)); "Building the wrong chunks first probably has problems … could also be a nod to the need for inside-outside parsing" ([L287-292](docs/MULTIHIERARCHY.md)). INSIDE_OUTSIDE.md: greedy "doesn't account for the creation of new symbols with respect to the broad grammar, resulting in an easy goal for one-off symbols to be repeatedly generated" ([INSIDE_OUTSIDE.md L9](INSIDE_OUTSIDE.md)).
2. **Attachment ambiguity.** About 10pp of the WEBSTER-era gap came from it (e.g. "with a dog" attaching to NP vs VP), and about 8pp from clear bad merges ("man admired" before "the man"). — [memory/project_content_instance_cplx_attrs.md](memory/project_content_instance_cplx_attrs.md). On large, "RelClause/PP attachment where BOTH options confidently belong to their class, so class_lp ties". — [memory/project_class_lp_heuristic.md](memory/project_class_lp_heuristic.md)
3. **Lexical sparsity on term_high.** The genuine V/P distributional identity limits term_high at 39 terminals. It is "NOT data-limited (N=400/900/1500 all ~0.90–0.92)" and not context-tunable. Concept-id context slots collapse distinct contexts (0.48). Shipped term_high EM is only 0.615. — [memory/project_class_lp_heuristic.md](memory/project_class_lp_heuristic.md); [memory/project_faithful_representation_push.md L23](memory/project_faithful_representation_push.md); [DATA](confs/acs-26/terminal_experiment/high/aggregated.csv)
4. **Seed fragility of content clustering.** Parse F1 tracks the phrase-type purity of the content hierarchy: term_med seed100 has purity 0.990 → F1 0.955, while seed17 has purity 0.901 → F1 0.577 ("Cobweb training-order"). — [memory/project_class_lp_heuristic.md](memory/project_class_lp_heuristic.md). The 20-seed term_med F1 is 0.930±0.062, against the 5-seed 0.961±0.012. — [DATA](confs/acs-26/terminal_experiment_20seed/med/aggregated.csv)
5. **Impure intermediate nodes.**
   - Content leaves are pure substitution classes: ~0.96–0.98 by type, rising to 0.994 with context 9.
   - The max-EPMI basic level is ~0.60–0.62, mixing PP/NP/VP. Maturity τ=20 gives ~0.90; the context hierarchy at maturity gives ~0.74.
   - "Generalization = merging eventually crosses type boundaries". Per-node errors compound (~0.90^4 ≈ 0.65).

   — [memory/project_faithful_representation_push.md L40-58](memory/project_faithful_representation_push.md)
6. **Over-splitting, "clustering blocker" (WEBSTER era).** 95–99% of supervised chunks landed at content leaves with count < 2, so counts could never clear a gate. — [memory/project_clustering_blocker.md](memory/project_clustering_blocker.md). In the v1 code, leaves are near-singletons, and the climbing walk sidesteps this. — [MEMORY.md index line](memory/MEMORY.md)
7. **Gate vacuity and an uninformative count signal.** See §3. P7 never fires, and the count gate does not separate constituents from non-constituents. — [AUDIT §1.2](../trellis_v1/PAPER_CODE_AUDIT.md); [memory/project_class_lp_heuristic.md](memory/project_class_lp_heuristic.md)
8. **Raw log-prob is density-confounded.** "A dense cluster inflates the log-prob of anything force-categorized into it — that's why 'found the' passed the old gate." — [memory/project_class_lp_heuristic.md](memory/project_class_lp_heuristic.md). [DOC] "Unseen things are being seen with the log probabilities in bad ways"; "Log-probability scores are HECKA weird". — [MULTIHIERARCHY.md L265, L26](docs/MULTIHIERARCHY.md)
9. **The basic-level detector is unreliable.** "Basic level not working RIP"; "Current basic level definition favors leaves" ([MULTIHIERARCHY.md L118-119](docs/MULTIHIERARCHY.md)); "the basic-level definition also needs fixing (tuning alpha is NOT the method)" ([L33](docs/MULTIHIERARCHY.md)). In the Cobweb-LLM v0 work, "EPMI basic level collapses to small deep nodes", so a frontier cut was used instead. — [memory/project_attention_skipgram_test.md](memory/project_attention_skipgram_test.md)
10. **Unsupervised churn and binarisation mismatch.** Log-prob or frequency ranking never converges epoch to epoch ("formation churn"). The unsupervised MED parser categorises consistently (86% assignment consistency) but binarises differently from gold (F1 46 vs 84.6). Expansion consistency is ~49–55%. — [memory/project_composite_threshold_sweep.md](memory/project_composite_threshold_sweep.md)
11. **Exposure and distribution shift.** The learned ranker fails on its own prefixes, and beam scoring is anti-correlated with gold. — [memory/project_acs26_f1_representation_push.md](memory/project_acs26_f1_representation_push.md)
12. **Generation residue.** Commission comes from fallback to the unfiltered pool. term_low is weakest (0.930±0.036, σ 3.6%). Seed 23 produced a "type-confused" generator (0.796) and was replaced. Replay truncates at depth 8. — [RESULTS.md L14-16, L30-34](confs/acs-26/RESULTS.md); [AUDIT §3](../trellis_v1/PAPER_CODE_AUDIT.md)
13. **Reproducibility risk.** Before the patch, identical configs ranged over F1 47–73%. The seeding patch was reported "NOT active in the current environment" because `import cobweb` resolves to another checkout. — [memory/project_cobweb_determinism.md](memory/project_cobweb_determinism.md)
14. **Sensitivity to order and α.** "ALPHA TUNING IS BIGGEST PROBLEMMMM" ([MULTIHIERARCHY.md L232](docs/MULTIHIERARCHY.md)). Cobweb's incremental order effects are acknowledged in the paper ([main.tex L429-432](confs/acs-26/paper/main.tex)).
15. **Paper↔code mismatches that matter for v2.**
    - No within-parse learning ([AUDIT §1.5](../trellis_v1/PAPER_CODE_AUDIT.md)).
    - The orphan-pair fitting contradicts "committed composites only" ([AUDIT §3](../trellis_v1/PAPER_CODE_AUDIT.md)).
    - The code default ranker (context_forward) is not the shipped one.
    - `_rank_score` (beam) ignores class_lp ([L2573-2591](src/parse_mh.py)).

### Inferences
- Failures 1, 2 and 11 are the classic motivation for chart or inside-outside methods. Failures 4–6, 9 and 10 say that *the categories feeding any chart are themselves unstable*. A v2 chart parser over Cobweb nodes inherits Cobweb's order sensitivity unless the categorical "types" are stabilised, e.g. by fixed-depth cuts as in S9 or by frozen or staged trees (§9).
- Failure 3 suggests multi-level context, the user's v2 idea, should probably be *weighted or gated per level* rather than concatenated. Abstracted context helped dense grammars and hurt sparse lexicons (R7). In Cobweb-LLM, neighbours' paths "hurt at this scale" (§10).

### Gaps
- No PTB-scale or natural-language failure analysis exists; all evidence is from ≤39-terminal synthetic CFGs.

---

## 7. v2 goals, open questions, the user's latest notes, and tensions with v1-era decisions

### Takeaway
The v2 pivot (branch `inside-outside`, October 2026) wants:
- inside-outside or lattice parsing instead of greedy;
- chunk information as context for parsing and generation;
- *unsupervised* learning of a "globally optimal and minimally viable grammar" using information-theoretic principles, incrementally;
- stronger, explicitly compositional representations;
- domains beyond language (chess, 2D, PTB, masked modelling).

Several of these directly conflict with v1-era rules that are still recorded in memory: stay greedy, train only on gold, never feed parser output back, generation locked, no hints, single shared config. These tensions are flagged below, not resolved.

### Cited Findings
**INSIDE_OUTSIDE.md** (working copy; 28 lines, with an uncommitted diff) [DOC]
- **Definition.** Inside-outside "considers multiple parsing probabilities at once … traditional inside-outside parsing considers every set of possible parses, computing the likelihood of each parse and 'freezing' / selecting the best one". — [INSIDE_OUTSIDE.md L7](INSIDE_OUTSIDE.md)
- **Critique of v1 greedy.** "Shown to have successful results for smaller grammars but doesn't account for the creation of new symbols with respect to the broad grammar, resulting in an easy goal for one-off symbols to be repeatedly generated". — [L9](INSIDE_OUTSIDE.md)
- **Goal of structure.** "Structure helps you understand purely what is necessary for the target of coherence in language, not necessarily the same as a higher-level understanding". — [L11](INSIDE_OUTSIDE.md)
- **Targets:**
  - Chunk context: "full-parsing will allow us to refine existing representations with context".
  - Unsupervised learning: "having whole parses allows us to set up thresholds in a way that's far easier. We can also … maintain candidate parses in a frontier and then learn them once we can confirm that they're good enough".
  - "Globally optimal and minimally viable grammar, borrowing from information-theory principles … in an incremental way".

  — [L17-19](INSIDE_OUTSIDE.md)
- **Enriching representations.** Chunk context: "need to enrich context with higher-level structure". Attention in Cobweb, for long-range dependencies; "hopefully we can simply do something more naive" than Cobweb-LLM. — [L21-29](INSIDE_OUTSIDE.md)
- **Text removed by the uncommitted diff.** The committed version (HEAD `2120f1f6`) had a section: "Immediate Test - Encode pairs of words with Cobweb!! Additionally encode distance?? … moved this to Cobweb-LLM", plus "relationships need to be context-enriched and content-enriched, and that's what Q, K, V is for in Transformers … maintain a representation of each word or symbol over time and then locally enrich it but update that representation as well". — `git diff INSIDE_OUTSIDE.md`

**TRELLIS_v2.md** [DOC]. "INITIAL DUMP, 7/22/2026 … let go of many of the 'quick fixes' … most of the early brainstorming in the Google Doc". — [TRELLIS_v2.md L1-7](TRELLIS_v2.md)

**MULTIHIERARCHY.md design history relevant to v2** [DOC]
- **Methodology 6.1:**
  - "UNSUPERVISED CHUNKING IS FIRST PRIORITY!!!"
  - Other discrete relations "may need a different way to assign content and context". Possibly weight attributes per hierarchy so information is "not lost completely but … also not used to build the hierarchy".
  - Domains: context-sensitive grammar; "Chess - more discrete relations and no distinct notion of content / context!!! Can do spatial awareness".

  — [L5-16](docs/MULTIHIERARCHY.md)
- **Methodology 6:**
  - An unsupervised and supervised learning threshold; basic level plus generation to be fleshed out.
  - "maximum-value parse where we look ahead X number of steps … Can do multiple partial parses until we get our final parse and freeze the maximum-value parse from the top-down".
  - "We need the idea of chunk context to properly scaffold the breaking down of chunks".

  — [L18-33](docs/MULTIHIERARCHY.md)
- **Methodology 5:**
  - Cobweb splits maximise average mutual information per child, MI(X;C)/|C| (from Chris).
  - Path information is not the best representation, hence "Mixture-Of-Concepts", with log-prob activations as representation.
  - "Parsing: … creating a parse lattice from the bottom up (inside) and then freezing the most probable sequence in a top-down manner (outside)".

  — [L104-144](docs/MULTIHIERARCHY.md)
- **Methodology 4.1:**
  - *Better Hierarchy Brainstorm*: "Spreading activation theory for restructuring?? … restructure the tree such that BFSes produce DFSes in the long run … mass-based merge?"
  - *Chunk Context Brainstorm*: "the order of chunks matters!! The best bet … a Matasakis-akin implementation … iterative addition"; "even in the context hierarchy under the 'content-ref' attribute, do we represent nodes by their surrounding or by their composition?"; "formalize the observational buffer … layer multiple hierarchies to keep track of different options"; "all data is considered 'out-of-distribution' until it is seen repeatedly - we really need … a datastructure that holds importance and reinforces importance".

  — [L146-167](docs/MULTIHIERARCHY.md)
- **Methodology 4.0** chunk-context TODO: "see if POS retains its part of speech with iterative context! (Basically, build a diffusion model with Cobweb)". — [L201-204](docs/MULTIHIERARCHY.md)
- **Methodology 3.1:** "We need a simplicity bias … build a chunk that it recognizes before trying to create a new chunk"; "LAYERS of Cobweb hierarchies". — [L216-226](docs/MULTIHIERARCHY.md)
- **Methodology 2.0:**
  - Recognition should be graded and work over generalisations.
  - "Stability" tiers.
  - A "distributional buffer … preliminary short-term Cobweb hierarchy that feeds into a longer-term Cobweb hierarchy".
  - The data-structure requirements: incremental, discrete attribute-value input, a recognition score combining frequency and accuracy, a basic level.

  — [L270-314](docs/MULTIHIERARCHY.md)
- **Methodology 1.0:**
  - "We need to find some method of multi-level context that explains all the contextual levels of the parse tree". — [L378-380](docs/MULTIHIERARCHY.md)
  - Masked-sentence performance by "build a parse from the bottom-up … then denoise and generate from the top down". — [L366-384](docs/MULTIHIERARCHY.md)
- **"Parallels to Diffusion Models, BPE + Tokenization":**
  - Generation ≈ diffusion; "If we introduce noise in a structured way, unraveling this noise will induce compositionality"; "noise two tokens at the time … based on frequency".
  - "BPE does this!!! … our method basically does BPE over generalizations of words in addition to just words".
  - "An LLM with an adaptive vocabulary and latent space"; "MOVING THIS TO ITS OWN REPOSITORY".

  — [L489-521](docs/MULTIHIERARCHY.md)

**Other local design notes** [DOC]
- **SOFT_V_HARD.md.**
  - Defines "hard" (relation-focused) vs "soft" (blended) composition, and replicates Zekun's PoE soft composition in Cobweb.
  - "Reconstructing unseen instances in terms of existing prototypes is more analogous to a representation learning scheme (fitting in with the bag-of-concepts idea used in TRELLIS v1)".
  - "The closest image to a given query is not always the best image for stealing from"; distilling primitives and relations from correlation structure.
  - A "Compositional Autoencoder … train the concept map to minimize the number of concepts needed".

  — [docs/SOFT_V_HARD.md L5-56](docs/SOFT_V_HARD.md)
- **Archived formalisation** (Nov 2025):
  - "Virtual and Real Chunks": candidate chunks accumulate statistics in a virtual hierarchy until promoted.
  - "Chunks are a theory of information compression"; "we should add a new chunk **only when we have to**".
  - Proposed "**chunk utility**, the combination of a score of recognition and resistance towards creation … Resistance is gradually lowered by the recognition threshold".

  — [src/archive/FORMALIZATION.md L17-82](src/archive/FORMALIZATION.md)
- **Memory, as of 2026-05-09:** "Inside-outside parsing direction is on the table (bottom-up parse lattice, freeze top-down) but not implemented". — [memory/project_methodology_evolution.md](memory/project_methodology_evolution.md)

**Paper's open issues** [PAPER]: learning from sentences alone; CSGs; vision; LLM-scale. — [main.tex L1291-1305](confs/acs-26/paper/main.tex)

#### User's v2 notes (verbatim)
```text
User request: Take the time to thoroughly understand the structure of this repository and what it desires (looking over confs paper as needed) and then look at INSIDE_OUTSIDE.md and do a comprehensive lit review to figure out how we can bring the following capabilities:
- inside-outside parsing
- chunk information as context (to aid in parsing and generation)
- unsupervised parsing (i.e. create a globally optimal and minimally viable grammar, borrowing from information-theory principles to do so in an incremental way)
- better representations (needed to develop all of this stuff)

New and improved:
Necessary additions:
NEED Unsupervised Chunk aggregation
NEED better representations (and to argue stronger for the idea of composition representations)
NEED more than just grammar / language as a domain
Nice to have:
New parsing scheme that considers a lattice / inside-outside?
A non-uniform amount of components-per-chunk? No need for just one template - maybe have several?

Reframings we need to emphasize:
Content / context → BEGONE!!! New ideas → a composition hierarchy and a representation hierarchy
One thing needs to represent (and keep track of generalizations of representations)
One thing needs to compose (and keep track of generalizations of compositions)
Because we highlight the ability to generalize, a concept hierarchy makes a lot of sense for both data structures!!
Parsing process: more than just greedy parsing! Use the inside-outside algorithm to quantify strength!
Can also do a lattice-style parsing, where we maintain a frontier of valid parses as we go up and select the best (non-intersecting) one as we go down
Generally: Trellis is a process that builds an internal grammar and utilizes it to make OOD inferences!

Unsupervised Chunking:
Intuitively, what do we need:
Need to figure out what constitutes a valuable chunk!!
To me, this has always been done in terms of how good the chunk is as well as a goal to minimize the overall number of chunks! (Funnily enough, this idea is reiterated by https://aclanthology.org/J01-2001.pdf)
Definitely need to look at Cobweb-MDL!!  [USER LATER CLARIFIED: the Cobweb-MDL variant isn't ready yet — ignore this note]
Inside-outside parsing may be the method here - important to highlight that our grammar may not line up with the real grammar, but as long as we learn ideas reusable for our given task, it's great!
NEED TO LOOK AT KOLMOGOROV COMPLEXITY AND SOLOMONOFFS PRIOR??!
Could be a way to preserve incremental additions!!

Better Representations:
Semantic network relational input something something - we need a way to map the composition rules to both rules on deciding merging and deciding ideas
Generally, how do you craft a representation compositionally before enriching it contextually?
In language, representations are PURELY contextually driven, but we can derive some hints from the structure of the parse tree, the words in intermediary ideas, and the words present to begin with!
CHUNK CONTEXT IS CRUCIAL!!!! For developing latents at various levels, it is important to have distributional context at multiple levels of abstraction!!
I think that we're going to just have latents take all levels of content before and after, keep things simple!!

Beyond Grammars / Language:
Masked modeling as a training regime for training TRELLIS v2??
Chunk the context window!??! What level of abstraction is necessary for that style of regression!
2D something something - pick a domain where hierarchy matters for the sake of the parse
Definitely want to do Chess here - it would make the most sense with respect to the literature!
Do Penn Tree Bank!! Unsupervised induction would be clutch, especially with some of the problems we've been running into

Breakdown of https://aclanthology.org/J01-2001.pdf:
Basically, this paper proposes a scheme for evaluating a grammar, as well as a way of proposing grammars that doesn't iterate over every one.
The way that the paper proposes grammars is such that likely grammars influence the proposition of similarly likely grammars
```

**Tensions between v1-era decisions and v2 goals** (flagged, not resolved)
1. **"Stay greedy" vs inside-outside / lattice parsing.** The memory rule "Do not solve parsing with global search (CKY / induced-grammar chart parsing) — that strays too far from the norm" ([memory/feedback_stay_greedy_clean_pos.md](memory/feedback_stay_greedy_clean_pos.md)) and "CKY … SET ASIDE as baseline" ([memory/project_cky_induced_grammar.md](memory/project_cky_induced_grammar.md)) were v1 paper-push constraints. v2 explicitly asks for "more than just greedy parsing! Use the inside-outside algorithm" (verbatim notes above). Note that CKY and inside-outside share the same chart.
2. **"Parser output NEVER fed back to memory"; gold trees are the only parse signal** ([memory/feedback_unsupervised_chunk_thresholding.md](memory/feedback_unsupervised_chunk_thresholding.md)) vs v2 "unsupervised learning", "learn [candidate parses] once we can confirm that they're good enough" ([INSIDE_OUTSIDE.md L18](INSIDE_OUTSIDE.md)), and PTB unsupervised induction. met6 already fed parser output back via `parse_sentence(learning=True)` ([memory/project_composite_threshold_sweep.md](memory/project_composite_threshold_sweep.md)), so "never fed back" was itself scoped to the supervised acs-26 goal.
3. **"Generation LOCKED; never optimize gen_gram; judged qualitatively"** ([memory/feedback_gen_locked_qualitative.md](memory/feedback_gen_locked_qualitative.md)) vs v2's changes to representation and context. Replay generation depends on exact context-leaf matching (§4), so it cannot stay untouched.
4. **"No hints" (faithful config; edge, seam and child-class features rejected as "drift")** ([memory/project_faithful_representation_push.md L10](memory/project_faithful_representation_push.md)) vs the v2 note that representations can draw "hints from the structure of the parse tree, the words in intermediary ideas, and the words present to begin with". This closely resembles the rejected boundary and seam features, which had gained +0.05 F1 (R5).
5. **"Content / context → BEGONE" vs the pillar "two hierarchies"** ([memory/feedback_best_chunker_reframe.md](memory/feedback_best_chunker_reframe.md): pillars stay; "Still ask before touching a pillar"). The reframe keeps two concept hierarchies (composition, representation), so it may count as a re-description rather than a pillar change (§1 inference). This needs the user's confirmation.
6. **"Ground truth alignment doesn't matter" vs the bracket-match metrics.** met6 says "We do NOT measure match to source CFG (project owner: don't care about ground-truth alignment)". v2 says "our grammar may not line up with the real grammar". The paper's metrics are gold bracket matches, and unsupervised MED looked bad on them (46 vs 84.6) while generating well. Evaluation for v2 is unsettled. — [memory/project_composite_threshold_sweep.md](memory/project_composite_threshold_sweep.md)
7. **One parameter set vs per-domain behaviour.** The paper claims one parameter set across conditions ([main.tex L1086-1087](confs/acs-26/paper/main.tex)), but the best structural and sparse-lexicon settings conflict (R7, R10). The tension will grow with PTB and chess.
8. **Complexity visible vs "hidden/generation-only".** Resolved in v1 (visible is load-bearing, R2). It may reappear if v2 drops the fixed binary template ("non-uniform components-per-chunk"), since complexity tags assume binary children.
9. **Binary-only composition vs "several templates".** v1 composes strictly binary, ordered pairs: adjacent-pair candidates ([L1590-1617](src/parse_mh.py)), content attrs L/R ([L866-1019](src/parse_mh.py)), and grammars "strictly binary" ([main.tex L1336](confs/acs-26/paper/main.tex)).
10. **Linear order vs 2D or chess.** R4 relations are before / left-of, and contexts are before/after windows ([main.tex L475-484](confs/acs-26/paper/main.tex)). MULTIHIERARCHY 6.1 notes chess has "no distinct notion of content / context" ([L16](docs/MULTIHIERARCHY.md)).

### Inferences
- **INSIDE_OUTSIDE.md conflates two algorithms.** It describes inside-outside as "computing the likelihood of each parse and freezing / selecting the best one". In standard usage, inside-outside computes *span marginals / expected rule counts* summed over all parses, typically for EM. Picking the single best parse is Viterbi/CKY decoding. The user's lattice description ("frontier of valid parses as we go up … select the best (non-intersecting) one as we go down") matches inside (bottom-up chart) plus top-down Viterbi or MBR extraction. The lit review should separate *scoring and learning from marginals* from *decoding*.
- **The archive already states the MDL idea.** The "chunk utility = recognition + resistance to creation" proposal and "add a new chunk only when we have to" anticipate the v2 "minimize the overall number of chunks" (MDL / simplicity) framing. met6 even ran a two-part MDL selection over α (U2).
- **J01-2001 is probably Goldsmith (2001).** That ID is, I believe, John Goldsmith's "Unsupervised Learning of the Morphology of a Natural Language" (Computational Linguistics 27(2)), an MDL-based method that proposes candidate analyses heuristically and evaluates them by description length. It matches the user's breakdown, but I did not verify it in this local-only pass.

### Gaps
- The Google Doc where v2 brainstorming happens (TRELLIS_v2.md L7) is not available locally.
- The memory note `project_trellis_v2_direction`, referenced by two other notes, does not exist in the memory directory.

---

## 8. Constraints and pillars that must survive, and terminology conventions

### Takeaway
Four pillars are non-negotiable without asking the user: concepts AND chunks, two (concept) hierarchies, parsing AND generation, and heavy use of categorisation. Learning must stay Cobweb-style: incremental, unsupervised clustering, with no category labels on chunks and gates expressed over chunk statistics. Everything else (representation, ranker, gate, search, training regime) is explicitly "up for grabs". Paper vocabulary uses omission and commission, never "F1".

### Cited Findings
- **Pillars** [MEMORY]: "concepts AND chunks …, two hierarchies, parsing AND generation, and heavy use of categorization. **Everything else is up for grabs** — instance representation, ranker, gate, search algorithm, training regime … When a change wins, it becomes the new postulate." — [memory/feedback_best_chunker_reframe.md](memory/feedback_best_chunker_reframe.md)
- **v2 reaffirms concept hierarchies for both structures** (verbatim notes): "a concept hierarchy makes a lot of sense for both data structures!!"
- **Theory constraints** [PAPER]. Learning is incremental and interleaved (L1), unsupervised in the sense of no class labels (L2), and uses no new learning mechanism, "parsing already produces candidate chunks" (learning bullets). — [main.tex L377-384, L781-798](confs/acs-26/paper/main.tex)
- **Label and gate constraints** [MEMORY]:
  - Chunks are never labelled NP/VP.
  - Gates and thresholds operate over chunk-level statistics, "never over a gold category label".
  - "NOT acceptable: condition any gate on a syntactic category label, feed parser-generated parses back into training [v1 scope], or relax the 95% bar".

  — [memory/feedback_unsupervised_chunk_thresholding.md](memory/feedback_unsupervised_chunk_thresholding.md)
- **Evaluation conventions** [MEMORY]:
  - Use omission/commission (Langley & Stromsten 2000) in all paper text and plots, and NEVER say "F1". Parse side: "parse accuracy". Generation: "grammaticality".
  - Grammar induction is the empirical testbed, not the theory.
  - Smolensky TPR / Plate HRR are the connectionist counterpart in related work.
  - CHREST gets a proper exposition.
  - Do not describe generation as "subtree-exchange replay"; it is the "pool + context-class filter".

  — [memory/feedback_paper_positioning.md](memory/feedback_paper_positioning.md). The paper also uses "substructure" rather than "bracket" ([main.tex L1048-1052](confs/acs-26/paper/main.tex)).
- **Metric strictness** [MEMORY]. `p_parse_legal` is too permissive; F1 (omission) and gen_gram (commission) ≥ 0.95 were the acs-26 bars. — [memory/feedback_grids_strict_metrics.md](memory/feedback_grids_strict_metrics.md)
- **Reproducibility** [MEMORY]/[DOC]:
  - Call `cobweb_set_seed` with Python and NumPy seeds.
  - Snapshot and restore all three RNGs around evaluation.
  - Verify that the patched cobweb-private is the one being imported.

  — [memory/project_cobweb_determinism.md](memory/project_cobweb_determinism.md); [RESULTS.md L97-109](confs/acs-26/RESULTS.md)
- **Repository constraints** [MEMORY]. `trellis_v1/` (now at `../trellis_v1/`) is a frozen snapshot: "Do NOT edit `trellis_v1/` for v2 features". — [memory/project_trellis_v1_repository.md](memory/project_trellis_v1_repository.md). Two Cobweb copies exist: `cobweb-private/` is live, and `concept_formation/` is a read-only reference. — [memory/project_overview.md](memory/project_overview.md)
- **Working style** [MEMORY]. "Treat this as active research … frame suggestions in research terms (mechanism, hierarchy quality, basic-level behavior)". — [memory/user_role.md](memory/user_role.md)
- **Terminology:**
  - **Paper terms:** element, experience, primitive, composite (= chunk), content/context instance, content/context taxonomy, recognition threshold τ_parse, recall threshold τ_gen, recombination pool. — [main.tex L475-515, L1465-1469](confs/acs-26/paper/main.tex)
  - **Code terms:** `TRELLIS` (renamed from WEBSTER on 2026-06-05, commit `f626c47c`), `LongTermMemory`, `FiniteParseTree`, `PrimitiveParseNode`, `CompositeParseNode`, frontier = `global_root_node.children`, "climbing ancestor", "maturity cut", "basic level" (max-EPMI ancestor), "label_path", "content-ref", "chunk records".
  - **v2 terms (user):** "composition hierarchy" (≈ content) and "representation hierarchy" (≈ context).

### Inferences
- Any inside-outside design that trains on its own parses must be squared with the scoped "no self-training" rule. The rule was written for the supervised paper goal; the user should re-scope it explicitly for v2.
- "No labels on chunks" fits unsupervised grammar-induction evaluation (unlabelled brackets) and fits chess or 2D domains, because there are no gold categories there either.

### Gaps
- No written statement yet defines v2's evaluation bars (successors to the 0.95 omission/commission bars) or which domain (PTB, chess, 2D) comes first.

---

## 9. Existing code hooks that could support inside-outside parsing, chunk context and unsupervised learning

### Takeaway
v1 already contains partial machinery for each v2 goal:
- a `chunk_context` mode with iterative primitive relabelling and LCA soft-matching;
- a beam over elaboration sequences;
- a frequency merge-policy hook;
- tree-wide marginal scores;
- Cobweb's own NEW-vs-INSERT partition utility as a "novel chunk" signal;
- orphan-candidate ("virtual chunk") fitting;
- a self-training flag;
- PCFG-like chunk records and leaf transitions;
- leaf-remap canonicalisation that survives restructuring;
- frozen or fixed trees;
- a masked-completion generator.

None is wired for chart/span parsing. Spans exist only as frontier adjacencies.

### Cited Findings
**Chunk context**
- **`LongTermMemory(chunk_context=True, context_n_iterations=k)`** makes the context tree its own `ref_tree`, with every context slot and content-ref treated as a ref attribute ([src/parse_mh.py L3825-3837](src/parse_mh.py)). Values are then soft-matched by LCA depth in `log_prob_instance` (C++ `lca_similarity`, [cobweb_discrete_node.cpp L1810-1820](cobweb-private/src/cobweb_discrete_node.cpp); [cobweb_discrete_tree.cpp L367](cobweb-private/src/cobweb_discrete_tree.cpp)).
- **Iterative relabelling.** `build_primitives` iteratively rebuilds each primitive's context from its neighbours' `label_path` concept ids and re-categorises until labels stop changing or k passes have run. This is a mean-field / iterative-refinement loop ("build a diffusion model with Cobweb", MULTIHIERARCHY L202). — [L1453-1571](src/parse_mh.py)
- **Composite chunk context.** Composites use `create_context_instance(chunk_context_before/after=…)` with the frontier neighbours' `label_path` ids. — [L1029-1030, L1069-1070, L1782-1804, L2018-2040](src/parse_mh.py)
- **Label granularity.** `context_label_cut` ∈ {leaf, basic, maturity} with `context_label_tau` sets how abstract the neighbour labels are (`_cut_ancestor` [L593-625](src/parse_mh.py)). This was the C1 experiment (R7).

**Search and parsing**
- **Beam.** `ltm.parse_beam_width` → `_build_beam`; the ranker inside needs replacing (§3). — [L2271-2274, L2593-2667](src/parse_mh.py)
- **Undo.** `FiniteParseTree.undo()` reverses `apply_candidate`. — [L2143-2179](src/parse_mh.py)
- **`evaluate_pair` reads the long-term memory without learning.** Its side effects are vocabulary registration (complexity and class symbols) and the TopK encoder's interning and push of its `value_remap` to the content tree (`bag_for`, [leaf_remap.py L365-424](cobweb-private/src/cobweb/leaf_remap.py)). It does not `ifit` anything. It is therefore a candidate span scorer if spans are materialised as frontier pairs. — [L1756-1994](src/parse_mh.py)
- **`MERGE_POLICY`** (`{"rank": "freq_basic"|"freq_leaf", "gate": "freq_basic"|"climb", "freq_min": …}`) gives a deterministic, frequency-ranked "BPE/GRIDS-style" merge order. — [L2191-2206, L2312-2350](src/parse_mh.py)
- **Gates.** `_climbing_ancestor` supports a relative-support gate (τ<1), and `maturity_gate=(name, thr)` can gate on any score field. — [L445-459](src/parse_mh.py); [L2321-2332](src/parse_mh.py)
- **Marginal scores.** `tree.log_prob` and `tree.log_prob_class_given_instance` with `max_nodes` / `greedy` give best-first, multi-node scores over the whole tree, a natural soft "inside" score for one span. `bfs_top_k_leaves` returns the top-K leaves. — [cobweb_discrete_tree.cpp L868-1036, L932-968](cobweb-private/src/cobweb_discrete_tree.cpp)
- **"Is this a new chunk?"** Each candidate already logs `leaf_insert_minus_new` and `bl_insert_minus_new`: Cobweb's own NEW-vs-INSERT partition utility at the leaf's and the basic level's parent ("Cobweb's own categorize-time signal for 'novel chunk'"). This hook is directly relevant to MDL-style "resistance to creation". — [L1855-1909](src/parse_mh.py)

**Unsupervised learning**
- **Self-training.** `parse_sentence(learning=True)` sends the parser's own tree to `add_parse_tree` (met6). — [L5046-5049](src/parse_mh.py)
- **Virtual chunks.** `add_parse_tree` fits *orphan candidate pairs* into the content tree, so un-admitted candidates accumulate counts, as in the archived "virtual chunk" design. The met6 bootstrap depended on it. — [L4194-4207](src/parse_mh.py); [memory/project_composite_threshold_sweep.md](memory/project_composite_threshold_sweep.md); [src/archive/FORMALIZATION.md L17-60](src/archive/FORMALIZATION.md)
- **Production statistics.** `learn_chunk_records` (`leaf_to_chunks`, `mat_to_chunks`, `bl_to_chunks`, `sentence_root_chunks`) and `learn_leaf_transitions` give rule-like statistics per content-leaf "nonterminal". The attestation helpers already query them at parse time. — [L4558-4766, L4872-4953, L1621-1754](src/parse_mh.py)
- **met6 harness** (sentences-only training, then freeze; determinism, coverage and simplicity metrics; MDL over categories and productions) is recoverable via `git show 8c8fafc6^:tests/met6/gen_learn_test.py`; I inferred this from the log entry and did not open the file. — [memory/project_composite_threshold_sweep.md](memory/project_composite_threshold_sweep.md)

**Restructuring and canonicalisation**
- **TopKPoolEncoder.** It stores stable leaf ids and recomputes a leaf→depth-d-ancestor `value_remap` whenever `context_tree.structure_generation` bumps. A "dead leaf" is rescued through its former ancestor's best leaf; orphans stay identity-mapped. A basic-level anchoring mode exists (`use_basic_level`). — [cobweb-private/src/cobweb/leaf_remap.py L39-125, L267-363](cobweb-private/src/cobweb/leaf_remap.py)
- **Restructuring actions.** `_ifit_and_update_vocab` exposes Cobweb's NEW / MERGE / SPLIT actions, and `_apply_rewrite_rules` rewrites av_counts after SPLITs. — [L3990-4046](src/parse_mh.py)
- **Hidden attributes** (negative indices) are tracked but not scored, which is how complexity is stored. `increment_attr_value` writes a value to a node and all its ancestors without changing counts. — [cobweb_discrete_node.cpp L154-166, L1786-1791](cobweb-private/src/cobweb_discrete_node.cpp)

**Frozen or fixed trees, and other hooks**
- **`FrozenCobwebDiscreteTree`** builds a CobwebDiscreteTree with a *fixed* hierarchy from a nested dict (for evaluation); all node methods work. — [cobweb-private/src/cobweb/frozen_discrete.py L1-60](cobweb-private/src/cobweb/frozen_discrete.py)
- **The `CompositeParseNode.frozen` flag** is vestigial (§2).
- **The met6 regime trains then freezes.** — [memory/project_composite_threshold_sweep.md](memory/project_composite_threshold_sweep.md)
- **Cobweb-LLM's add-only and staged training** keep symbols stable: "a concept's ancestors never change, so a symbol stored in another tree keeps its meaning". — [../Cobweb-LLM/experiments/v1/METHODOLOGY_v1.md L46, L85-89](../Cobweb-LLM/experiments/v1/METHODOLOGY_v1.md)
- **Masked modelling.** `generate_sentence(masked_sentence="the [mask] dog …")` has a mid-sentence path. [MEMORY] reports POS recovery 87%→95.7% and exact-token recovery 17%. — [L5214-5262, L6169-6170](src/parse_mh.py); [memory/feedback_generation_strategy.md](memory/feedback_generation_strategy.md)
- **CSG generator.** `src/util/csg.py` builds toy context-sensitive grammars (tuple-LHS productions) for the paper's CSG future-work item. — [src/util/csg.py L1-37](src/util/csg.py)
- **Hybrid Cobweb.** `CobwebHybridTree` puts discrete attribute-values and a continuous Gaussian block in one node under one PU. Offsets and positions can then be continuous attributes, relevant to 2D or chess coordinates and to attention. — [memory/project_cobweb_hybrid_module.md](memory/project_cobweb_hybrid_module.md)
- **Persistence.** `save_state` / `load_state` on both the LTM and TRELLIS. — [L4300-4352, L6569-6591](src/parse_mh.py)

### Inferences
- **Inside-outside needs span scoring.** The minimal path is a chart over spans [i,j) whose cells hold candidate composites scored by the two hierarchies; `evaluate_pair` logic is generalised from frontier adjacency to span pairs. Rule probabilities come from `leaf_to_chunks` and `leaf_transitions` maintained *incrementally*. The current code assumes a single frontier (`global_root_node.children`) and position-midpoint indices, so a chart needs a new data structure. Per-candidate diagnostics should be stripped first (§3).
- **Chunk context in a chart has no single answer.** A span's context can be read from the chart's outside side, which gives a principled "outside" analogue of chunk context: expected neighbouring categories under the outside distribution. That is one way to resolve the greedy-frontier dependence of today's `chunk_context`.

### Gaps
- No existing code computes outside probabilities or span marginals. No chart data structure exists. I found no test of `chunk_context` on composites beyond the C1 sweep (R7).

---

## 10. Adjacent threads: Cobweb-LLM ("attention in Cobweb", multi-layer, hybrid trees) and CHREST (chess)

### Takeaway
Cobweb-LLM shows that a pair tree over (query word, key word, signed offset) learns readable "relationship heads". It also shows that stacked Cobweb layers, with soft per-depth paths as representations, form a causal LM that matches a Kneser-Ney trigram on top-1 (0.219 vs 0.214) and beats a tiny GPT. Attention *weights* barely matter, head *symbols* do, and passing neighbours' paths to the next layer hurt. CHREST (1996) is the canonical chess chunking model, with a discrimination net, a 4-chunk STM and eye-movement heuristics. Its stated shortcomings are overlapping chunks, low master-level recall (48.6% vs human 82.9%) and the need for slot-bearing *templates*. Those map onto v2's "non-uniform components-per-chunk" and chess goals.

### Cited Findings
**Cobweb-LLM v0: pairwise "attention" tree** [SIBLING]
- **Setup.** Instances are `{ANCHOR, CONTEXT}` plus a continuous signed offset in a hybrid tree (α 1e-3, prior_var 0.1).
- **Gap-filling results (Oz, ±3, 2,000 held-out gaps):** top-1 / top-5 0.257 / 0.517, vs Cobweb/4L replication 0.263 / 0.460, count table 0.199 / 0.463, Word2Vec CBOW 0.130 and SG 0.162.
- **Position-aware neural baselines beat it on top-1:** positional CBOW 0.277 / 0.498 and structured skip-gram 0.276 / 0.519. "Word2Vec loses because it is position-blind, not because it is neural". Across 3 splits the pairwise hybrid leads on top-5 (0.509 vs 0.468 for Cobweb/4L).

— [../Cobweb-LLM/experiments/v0/EXPERIMENTAL_SETUP.md §3, §7](../Cobweb-LLM/experiments/v0/EXPERIMENTAL_SETUP.md)
- **Heads.** A frontier cut finds relationship "heads", e.g. `the → {scarecrow, lion, woodman, tin} @+1` and `i → {shall, should, will} @+1`. The EPMI basic level collapses to small deep nodes. — [memory/project_attention_skipgram_test.md](memory/project_attention_skipgram_test.md)
- **Earlier memory claim superseded.** The earlier note "Cobweb leads at every training size" ([memory/project_attention_gap_eval.md](memory/project_attention_gap_eval.md)) predates the positional baselines that beat it on top-1.

**Cobweb-LLM v1: Cobweb-LM** [SIBLING]
- **Design.** Each layer has a *pair tree* (attention; "nodes are heads") and a *token tree* (residual / FFN / unembedding). Representations are soft per-depth paths, "renormalised within each depth … top-2 symbols per depth". Positional encoding uses distance-weighted bags plus a Gaussian offset. Add-only training keeps shared symbols stable, and the readout is a tempered multi-node mixture. — [../Cobweb-LLM/experiments/v1/METHODOLOGY_v1.md L9-46, L69-128](../Cobweb-LLM/experiments/v1/METHODOLOGY_v1.md)
- **Results on Oz (3 splits):**

  | Model | Perplexity | Top-1 |
  |---|---|---|
  | 2-layer Cobweb-LM | 70.6±2.3 | 0.219±0.011 |
  | KN trigram | 65.4 | 0.214 |
  | Tiny GPT | 102.4 | 0.167 |

  Cobweb stores 6.0M counts vs 41k for the trigram ("parameter economy" is open). Grimm shows the same ordering. — [METHODOLOGY_v1.md L177-226, L262-268](../Cobweb-LLM/experiments/v1/METHODOLOGY_v1.md)
- **Ablations:**
  - Distance-weighted bags beat exact slots "by a wide margin"; a flat bag is "terrible".
  - Attention *weights* barely matter (uniform ≈ PMI); "the information is in which heads fire".
  - A second layer helps through diversity, and soft representations matter for layer 2.
  - "**neighbours' paths hurt at this scale**".
  - Window 5 beats 3 and 8; add-only beats four operations.
  - The `prior` node weighting (log P(x|c) + log P(c)) is best calibrated, and a product combination of layers is catastrophic (ppl 207).

  — [METHODOLOGY_v1.md L117-128, L237-250](../Cobweb-LLM/experiments/v1/METHODOLOGY_v1.md)
- **Open questions:**
  - calibration with little data;
  - why attention weights do not matter;
  - "**What should a neighbour hand to the next layer?** … truncating to depth 2 makes it harmless, not useful. Candidates: a basic-level cut, or the heads it participated in";
  - "**Chunks.** The stable heads of the pair tree could become constituents of higher-order instances (Langley's concepts-and-chunks agenda)";
  - a diffusion-style masked loop, not built.

  — [METHODOLOGY_v1.md L331-359, L376-384](../Cobweb-LLM/experiments/v1/METHODOLOGY_v1.md); [../Cobweb-LLM/HANDOFF.md L138-160](../Cobweb-LLM/HANDOFF.md)
- **Journal:**
  - Multi-layer stacking with residual connections "feels very similar to the idea of 'chunk context' with not only including low-level data, but higher-level counterparts as well".
  - Attention spans a spectrum from "keep everything" to "compressed state".
  - Diffusion LMs could "be improved using my chunking ideas".

  — [../Cobweb-LLM/JOURNAL.md L47-57, L74-90](../Cobweb-LLM/JOURNAL.md)

**Other compositional-representation threads** [MEMORY, index level only]. Cobweb-PoE on ColorMNIST composes held-out combinations (joint 27%) where retrieval fails (0%). Per-pixel τ→0 PoE reaches 48% on raw pixels. "Cobweb tree IS the manifold" for sampling. — [memory/MEMORY.md](memory/MEMORY.md)

**CHREST**, chapter 8 "Learning, perception and memory in chess: Model and simulations". From the chapter text this appears to be De Groot & Gobet, *Perception and Memory in Chess* (1996). [LIT]
- **Architecture.** CHREST stands for Chunk Hierarchy and REtrieval STructures, an EPAM-family model. Long-term memory is a discrimination net learned by *discrimination* (add tests and branches) and *familiarisation* (add detail to a node's image). STM is a queue of about 4 chunk pointers plus a privileged "hypothesis" chunk that drives eye movements. The visual field is 5×5 squares. — [../Papers/CHREST-chapter-1996.pdf §8.4, §8.7.1, §8.7.4, §8.7.8](../Papers/CHREST-chapter-1996.pdf)
- **Training.** About 9,500 positions from Tal's games. The Novice net has 200 nodes; the Master net has 24,000 nodes after two passes. Heuristics switch from Novice to Master at 2,000 nodes. — [CHREST §8.8.1](../Papers/CHREST-chapter-1996.pdf)
- **Results:**
  - Board coverage is close to human: BC 0.90 vs 0.91 for masters, 0.64 vs 0.69 for novices (Table 8.6).
  - Judges could not tell simulated from human eye-movement diagrams (Table 8.4).
  - Recall is CHREST-Master 48.6% vs human masters 82.9%, and CHREST-Novice 13.3% vs 20.5% (Table 8.8). With 20 STM slots the model reaches 70–80%.
  - Low recall is blamed on chunks overlapping in STM. "The concept of template … may offer such a data structure".

  — [CHREST §8.8.4, §8.9](../Papers/CHREST-chapter-1996.pdf)
- **Extensions and limits.** CHUMP links a pattern net to a move net through associative links, giving adaptive productions. Growing multiple nets plus inter-net links (semantic memory) is proposed. Primitives are semantic relations (attack, defence, proximity). — [CHREST §8.9](../Papers/CHREST-chapter-1996.pdf)
- **How the TRELLIS paper uses CHREST** [PAPER]: it "extends EPAM … including the addition of lateral links among nodes, similar to the interleaving in TRELLIS". — [main.tex L1236-1242](confs/acs-26/paper/main.tex)

### Inferences
- **Context abstraction again.** Cobweb-LLM's finding that neighbours' concept paths hurt the next layer echoes TRELLIS's C1 result (R7: concept-id context helps dense grammars and kills sparse ones). Two independent projects suggest that naively stacking abstracted context is risky. The "level of abstraction" question in the v2 notes ("Chunk the context window!??! What level of abstraction is necessary") is empirically live.
- **Joint scoring converges.** Both projects found the joint "prior" score log P(x|c) + log P(c) best: Cobweb-LLM for readout, TRELLIS as the node-level term inside class_lp. That suggests v2 inside-outside span scores should be built on it.
- **Chess fits the v2 asks.** CHREST's chunks are unordered sets of piece-on-square primitives with overlapping membership, and its fix is templates with slots. Chess therefore naturally needs non-binary, multi-template chunks and 2D relations, both v2 asks. Langley (2025) §5.1/§5.3 and §6.3 name chess as the target domain for the same Cobweb mechanisms.

### Gaps
- The CHREST PDF is OCR'd and noisy, so table values were read from garbled text. They are consistent across mentions but should be spot-checked against the PDF images.
- The book's full citation (authors, publisher) is inferred from the chapter text, not from a title page.
