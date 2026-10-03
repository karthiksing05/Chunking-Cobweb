# Domains Beyond Language for TRELLIS v2: Chess, 2D/Visual Compositional Hierarchies, Music, Plans/Programs, and Compositional-Generalization Benchmarks

*Research notes compiled 2026-10-03; they cover work up to October 2026.*
- **Scope:** candidate non-language domains and compositional/out-of-distribution test suites for TRELLIS v2, the content+context dual Cobweb hierarchy with parsing and generation (Singaravadivelan & Langley, ACS-26).
- **Verification:** bibliographic details were checked against Crossref, publisher or DOI pages, arXiv, PubMed/Europe PMC, and full-text PDFs where reachable. Dataset licenses were checked against repository LICENSE files and official pages.
- **Search quota:** the shared web-search quota ran out partway through. Later checks used direct retrieval (Crossref/arXiv/GitHub/Europe PMC APIs and PDFs). Anything not verified that way is listed under Gaps.
- **Terminology:** precision/recall-style scores reported by sources are restated as **commission-side** (are produced items legal or correct?) and **omission-side** (are legal or true items covered?) measures. This follows Langley & Stromsten (2000), "Learning Context-Free Grammars with a Simplicity Bias," ECML, LNCS pp. 220–228 ([Crossref doi](https://doi.org/10.1007/3-540-45164-1_23)).

## Q1. What do Langley's essay, the TRELLIS paper and Cobweb's own structured-domain lineage require of a v2 domain?

### Takeaway
Both founding documents name chess first and describe the v2 problem the same way: TRELLIS needs a variable number of constituents per chunk and relations beyond linear order. Cobweb's lineage already has two working templates for that (Labyrinth's interleaved component references; TRESTLE's partial matching plus flattening). The open technical risk the lineage exposes is the **correspondence/role-assignment problem**: deciding which constituent of a new instance plays which role in a stored chunk. v1 sidestepped it with ordered LEFT/RIGHT children.

### Cited Findings
- **Essay, bibliographic.** Langley, P. (2025). "Concepts and Chunks in Cognitive Systems." *Advances in Cognitive Systems* 11, 1–12 (submitted 8/2025, published 10/2025). — [Langley 2025 PDF](http://www.cogsys.org/proceedings/2025/paper-2025-3.pdf)
- **Essay, definition of a chunk.** "A chunk is a cognitive structure that denotes a collection of entities, possibly typed, that form a specified configuration or relational pattern." The relations "may be spatial (as in constellations), temporal (as in music), or something more abstract," and "purely qualitative (e.g., above, behind)" or quantitative ("distance, angle"). — [Langley 2025](http://www.cogsys.org/proceedings/2025/paper-2025-3.pdf)
- **Essay, chess and music as unified concept+chunk structures.** "a structure for a chess board would involve a spatial pattern of pieces that threaten or defend each other, but different types of pieces might occupy the same role, denoting a set of possible configurations. One could describe a passage of music as a sequence of notes, but this could allow variations in timing and emphasis." On performance: "The same mechanisms could recognize familiar configurations in chess boards, even if they involve novel pieces, and reconstruct their layout on the board from a chunk identifier and its constituent entities." — [Langley 2025](http://www.cogsys.org/proceedings/2025/paper-2025-3.pdf)
- **Essay, how chunks are measured.** Chunk recognition is "typically ... a bottom-up, all-or-none match process, with relations among elements playing a key role." Recall is "recreating positions on a chess board after a brief exposure ... [via] some form of top-down generative process. Typical measures here are memory capacity and reconstruction time." EPAM and CHREST assume acquisition "is incremental and largely unsupervised, with mastering the ability to recognize chunks preceding the ability to reconstruct them." — [Langley 2025](http://www.cogsys.org/proceedings/2025/paper-2025-3.pdf)
- **Essay, on CHREST templates and their gap.** CHREST "further extended these images into 'templates' whose roles can be filled in different ways, making them even more like generalized concepts." But EPAM and CHREST "did not provide a way to acquire high-level chunks in terms of simpler ones," and "appear to have required stimuli that were already organized into partonomic structures." — [Langley 2025](http://www.cogsys.org/proceedings/2025/paper-2025-3.pdf)
- **Essay, open question for any new domain.** "The account should also state whether the relations between chunk constituents are themselves decomposable or whether they allow variation across instances." — [Langley 2025](http://www.cogsys.org/proceedings/2025/paper-2025-3.pdf)
- **TRELLIS paper, bibliographic.** Singaravadivelan, K. & Langley, P. "A Unified Account of Concepts and Chunks," arXiv:2609.30414 (submitted 24 Sep 2026; accepted to ACS-26, oral). — [arXiv abs](https://arxiv.org/abs/2609.30414)
- **TRELLIS paper, on domain generality.** The postulates assume only "a set of elements with local relations among them, a description that fits a board position, a visual scene, or a motor sequence as readily as a sentence." — [arXiv PDF](https://arxiv.org/pdf/2609.30414)
- **TRELLIS paper, stated limits and future work.** "Its composition operator is binary and its only relation is before, which suffices for strings but not for richer structures." "Allowing a variable number of constituents, and relations beyond linear order, would let the same postulates address board positions, visual scenes, and plans, which is where the claim of generality must ultimately be tested." "Chess is the natural first target, given the role it has played in the chunking literature, and treebanks offer a demanding test of unsupervised induction." — [arXiv PDF](https://arxiv.org/pdf/2609.30414)
- **Labyrinth (Cobweb for composite objects).** Thompson & Langley, "Incremental Concept Formation with Composite Objects," ICML 1989, pp. 371–374. A composite instance is "a set of components, which may themselves be composite objects." Attribute values point to other nodes in the hierarchy, giving "an interleaved memory structure." — [ML Anthology](https://mlanthology.org/icml/1989/thompson1989icml-incremental). Fuller account: Thompson & Langley (1991), "Concept formation in structured domains," in Fisher, Pazzani & Langley (Eds.), *Concept Formation*, pp. 127–161. — [Crossref DOI 10.1016/b978-1-4832-0773-5.50011-0](https://doi.org/10.1016/b978-1-4832-0773-5.50011-0)
- **The TRELLIS paper's own verdicts on its relatives.** Labyrinth "lacked a contextual taxonomy and suffered substantial order effects." TRESTLE "operated over instances organized in a partonomic hierarchy, but flattened the representation during processing." Convolutional Cobweb (MacLellan & Thakur, 2021) "learns over multiple levels of image descriptions, giving some effects of chunks without treating them explicitly." — [arXiv PDF](https://arxiv.org/pdf/2609.30414)
- **TRESTLE, bibliographic and mechanism.** MacLellan, Harpstead, Aleven & Koedinger (2016), *Advances in Cognitive Systems* 4. — [TAIL lab page](https://tail.cc.gatech.edu/publications/maclellan-acs-journal-2016)
  - It has four attribute types (nominal, numeric, component, relational), with relations as tuples such as `(On Component1 Component2)`.
  - It partially matches each instance to the root concept, "renaming" components with a beam search used "under the assumption that greedy approaches are more psychologically plausible than optimal ones."
  - It then flattens the instance into dot-notation attributes (e.g., `Component1.type`) for ordinary Cobweb.
  - Source: [arXiv:2410.10588](https://arxiv.org/pdf/2410.10588)
- **TRESTLE, human comparison (a 2D block-tower domain).** RumbleBlocks is a 2D tower-building educational game.
  - In a sequential success-prediction task, 20 Mechanical Turk participants "converge around 70% accuracy." A nonincremental baseline (CFE) reached "roughly 83%." TRESTLE "performs roughly equal to the humans."
  - In unsupervised clustering against researchers' hand-sorts (inter-rater adjusted Rand index 0.88), TRESTLE scored ARI 0.16–0.56 depending on level and depth. CFE scored 0.47–0.51.
  - Source: [arXiv:2410.10588](https://arxiv.org/pdf/2410.10588)
  - The `concept_formation` library (Cobweb, Cobweb/3, TRESTLE) is MIT-licensed. — [GitHub cmaclell/concept_formation](https://github.com/cmaclell/concept_formation)
- **Oxbow (Cobweb-family concept formation over movements).** Iba (1991), "Learning to Classify Observed Motor Behavior," IJCAI-91, pp. 732–738. It uses "a temporal structure relating components of a single complex movement" and can forecast "the latter portions of a partially observed movement." — [ML Anthology](https://mlanthology.org/ijcai/1991/iba1991ijcai-learning). Oxbow is described as the recognition component of MAEANDER, which also generates movements. — [NASA NTRS record](https://ntrs.nasa.gov/citations/19920019931)
- **Cobweb over plans and problem-solving traces (cited in the TRELLIS paper).**
  - Yang & Fisher (1989) clustered means-ends plans.
  - Yoo & Fisher (1991, IJCAI, pp. 630–636) formed concepts over explanations and problem-solving traces, "which are themselves derivation trees much like the parses we learn here."
  - Carlson, Weinberg & Fisher (1990) used Cobweb to manage search over strings of a synthetic CFL.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2609.30414)
  - Also relevant: Langley & Allen (1993), "A Unified Framework for Planning and Learning," in *Machine Learning Methods for Planning*, pp. 317–350. — [Crossref DOI 10.1016/b978-1-4832-0774-2.50015-9](https://doi.org/10.1016/b978-1-4832-0774-2.50015-9)
- **A 2025 chunking-based concept model from the CHREST tradition.** Bennett & Gobet's CogAct (arXiv, Dec 2025) grounds concept learning in "chunking, attention, STM and LTM." It simulates individual participants' subjective conceptual judgments in music, and also engages with literature and chess. — [arXiv:2512.18665](https://arxiv.org/abs/2512.18665)

### Inferences
- **Minimum representational change for v2.** TRELLIS needs two things:
  - n-ary content descriptions whose slots are linked by typed relation edges, not just LEFT/RIGHT.
  - A role-assignment step that says which constituent fills which slot.
- **Three options for role assignment, from cheapest to most general:**
  - **(a) Canonical ordering.** The domain supplies the order. Chinese IDS operators fix positions; music has time order; chess has board coordinates.
  - **(b) Roles defined by relations.** For example, attacker/target and defender/defended in chess.
  - **(c) TRESTLE-style partial matching.** The general solution, but costly; TRESTLE itself matches only against the root.
- **Pick domains where (a) or (b) is available.** Choosing such domains is the safest route, because it keeps v1's parse/generate machinery intact.
- **What evaluation must cover.** The essay names recognition before reconstruction, plus memory capacity and reconstruction time, as the chunk-side measures. A v2 domain should therefore support:
  - a recognition/parse test, scored by omission;
  - a reconstruction/generation test, scored by commission plus omission on recall;
  - ideally, human recall data. Chess is the canonical case.
- **What separates TRELLIS from its nearest predecessors.** Labyrinth and TRESTLE lacked a context taxonomy, and TRESTLE flattens. That is the differentiator a v2 domain must exercise: the domain should have clear substitution classes, defined by the contexts things appear in (not their content), that act like "pieces that occupy the same role."

### Gaps
- I could not open the text of Langley & Allen (1993, Dædalus) or Yoo & Fisher (1991). Only their bibliographic records, and the TRELLIS paper's one-line characterisations, are verified here.
- Oxbow's generation abilities (via MAEANDER) are verified only from a NASA NTRS search snippet, not the full paper.

## Q2. Chess: prior chunk models, datasets, human data, omission/commission-compatible metrics, and representation

### Takeaway
Chess is the strongest v2 domain on every criterion except gold structure.
- **Metric fit:** the chess-memory literature already scores position reconstruction as **errors of omission and errors of commission**, by skill, presentation time, game vs random positions, number of boards, and distortion. That is the same vocabulary TRELLIS uses.
- **Competitor:** it has a symbolic one, CHREST/template theory, whose template "core + slots" maps almost one-to-one onto Cobweb concepts.
- **Data:** huge CC0 game and puzzle data with motif labels, plus an automated legality checker.

The missing pieces are gold partonomic parses (TRELLIS must induce chunks or use proxy segmentations) and public trial-level human recall data.

### Cited Findings
**Classic cognitive findings**
- **Chase & Simon (1973), bibliographic.** "Perception in chess," *Cognitive Psychology* 4(1):55–81. — [doi](https://doi.org/10.1016/0010-0285(73)90004-2); [full text](http://matt.colorado.edu/teaching/highcog/fall8/cs73.pdf)
- **Chase & Simon, design.** One master, one Class A player and one beginner. Stimuli were 10 middle-game positions (24–26 pieces), 10 endgames (12–15 pieces) and 8 random positions. The memory task gave a 5-s view and repeated trials. — [full text](http://matt.colorado.edu/teaching/highcog/fall8/cs73.pdf)
- **Chase & Simon, chunk relations.** Chunks were analysed with five inter-piece relations:
  - **attack** (either piece attacks the other);
  - **defense**;
  - **proximity** (one of the 8 adjacent squares);
  - **common color**;
  - **common type**.

  Pieces placed within ~2 s of each other were treated as one chunk.
  - Within-chunk relation profiles correlated .89 between the copy and recall tasks.
  - Between-chunk profiles resembled random placement (.81–.87).
  - "the 2-second criterion in fact marks chunk boundaries."
  - Sources: [Chase & Simon 1973](http://matt.colorado.edu/teaching/highcog/fall8/cs73.pdf); [Simon & Chase 1973, *American Scientist*](https://www.Gwern.net/doc/psychology/chess/1973-simon.pdf)
- **Chase & Simon, chunk statistics** (middle games, first trial).

  | Measure | Master | Class A | Beginner |
  |---|---|---|---|
  | Chunks per trial | 7.7 | 5.7 | 5.3 |
  | Pieces per chunk | 2.5 | 2.1 | 1.9 |
  | Pieces recalled correctly | ~16 | 8 | 4 |

  - Of the master's 77 chunks, 47 were pawn chains and 10 castled-king configurations; 75% were "highly stereotyped."
  - Placing an "average" prototype master position already scores "better than 44%." This is a prior-only generation baseline.
  - Sources: [CS73](http://matt.colorado.edu/teaching/highcog/fall8/cs73.pdf); [SC73](https://www.Gwern.net/doc/psychology/chess/1973-simon.pdf)
- **MAPP, the first EPAM-style chess simulation.** It learned 2–7-piece patterns. About 1,000 patterns gave roughly Class-A recall, and Simon & Chase extrapolated about 50,000 patterns for a master. — [SC73](https://www.Gwern.net/doc/psychology/chess/1973-simon.pdf)
- **Gobet & Simon (1996), "Templates in chess memory."** *Cognitive Psychology* 31(1):1–40.
  - Players recalled up to 5 boards, and one player recalled 9 boards at >70% (160 pieces). Recall of single random boards was 21% for Masters, 16% for Experts and 12% for Class A.
  - **Templates** come from familiar openings. Each fixes the locations of "perhaps a dozen pieces," with **slots** fillable "in a matter of a second or two" and revisable default values.
  - Learning a new chunk takes ~8 s; adding to an existing one takes ~1 s.
  - Sources: [doi](https://doi.org/10.1006/cogp.1996.0011); [PDF](https://www.Gwern.net/doc/psychology/chess/1996-gobet.pdf)
- **Gobet & Simon (2000), "Five seconds or sixty?"** *Cognitive Science* 24(4):651–682. Source for all points below: [doi](https://doi.org/10.1207/s15516709cog2404_4); [PDF](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)
  - **Participants:** Masters (n=5), Experts (n=8) and Class A (n=7).
  - **Presentation times:** 1, 2, 3, 4, 5, 10, 20, 30 and 60 s, for game and random positions.
  - **Masters** reached about 92% at 10 s.
  - **Skill gap:** about 20 points on random positions versus about 60 on game positions.
  - **Example template** (from a 300k-node net):
    - core: `Pc4 Pe4 Pf3 Pg2 Ph2 Be3 Nc3 / pc6 pd6 pf7 pg6 nc5 bg7`;
    - **square-slots** such as `g1:<white king>`;
    - **piece-slots** such as `white rook:<e1>`.
- **Other Gobet results.**
  - Strong players keep a small advantage on random positions shown for 3–10 s. — [*Psychon. Bull. Rev.* 3(2):159–163](https://pubmed.ncbi.nlm.nih.gov/24213863/)
  - Masters' chunks are much larger than in 1973; the largest was 17 pieces. — [*Memory* 6(3):225–255](https://pubmed.ncbi.nlm.nih.gov/9709441/)
  - Usually no more than 3 chunks are replaced, and Masters' chunks reach 15 pieces. — [Gobet & Clarkson 2004, *Memory* 12(6):732–747](https://pubmed.ncbi.nlm.nih.gov/15724362/)
  - A skill effect survives full randomization of both locations and piece distribution (N=36), which supports template theory over constraint attunement. — [Gobet & Waters 2003, *JEP:LMC* 29(6):1082–1094](https://pubmed.ncbi.nlm.nih.gov/14622048/)
  - Gobet et al. (2001), "Chunking mechanisms in human learning," *TiCS* 5(6):236–243, distinguishes deliberate from perceptual chunking. — [doi](https://doi.org/10.1016/s1364-6613(00)01662-4)
- **Location specificity.** Recall drops for mirror-reflected positions. This "implies that each chunk represents a specific pattern of pieces in specific location." — [Gobet & Simon 1996, *Memory & Cognition* 24(4):493–503](https://pubmed.ncbi.nlm.nih.gov/8757497/). See also Saariluoma (1994), "Location coding in chess," *QJEP A* 47(3):607–630 ([doi](https://doi.org/10.1080/14640749408401130)).

**CHREST, the closest symbolic competitor** (as specified in [Gobet & Simon 2000](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf))
- **Memory structure.** An EPAM discrimination net indexes LTM. STM holds 3 visual chunks, the visual field is ±2 squares, and a "mind's eye" is included. — [GS00](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)
  - The primitive is a **piece-on-square (POS)**. A chunk is a list of POS, and the only test is "what is the next item."
  - There are **no relation tests for threats or plans.**
- **Learning parameters.** — [GS00](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)
  - Discrimination takes 8 s, familiarization 2 s, and template-slot filling 250 ms.
  - Slots are created when more than 3 nodes below a node share a square, a piece type, or a chunk, and the chunk has at least 5 elements.
- **Skill.** It is simulated with nets of **500, 10,000 and 300,000 nodes** (Class A, Expert, Master), trained on master-game positions. — [GS00](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)
  - Without slots, the 300k net recalled 64.5% at 10 s. With slots it reached 84%; human Masters reach 85–96%.
- **Error profile.** **Errors of omission and commission are reported for both humans and model.** — [GS00](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)
  - Omissions are fit well.
  - Commissions are fit poorly. The program "underestimates errors of commission by the Class A players, most severely in the random condition, but generally overestimates such errors by the Experts and Masters." The reason is that "the image of a chunk may contain information about the locations of more pieces than just those used to recognize the pattern."
  - The authors suggest a signal-detection-like mechanism to trade the two off.
- **Code.** A Java CHREST repository exists, last active in 2012, under the Open Works License. — [GitHub mrgschiller/chrest](https://github.com/mrgschiller/chrest); [README](https://raw.githubusercontent.com/mrgschiller/chrest/master/README.md)
- **Recall scoring convention** ("Following Chase and Simon"):
  - **% correct** = pieces placed correctly.
  - **Omission** = pieces in stimulus − pieces placed.
  - **Commission** = pieces placed wrongly.
  - Verbatim: "The number of errors of omission is the number of pieces in the stimulus position minus the number of pieces placed by the subject. The errors of commission are the pieces placed incorrectly by the subject."
  - Sources: [GS96 PDF](https://www.Gwern.net/doc/psychology/chess/1996-gobet.pdf); [GS00](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)
- **Human (CHREST) mean errors, game positions** (Gobet & Simon 2000, Table 5):

  | Error type | Masters | Experts | Class A |
  |---|---|---|---|
  | Omission | 0.6 (1.3) | 5.8 (6.2) | 9.5 (12.5) |
  | Commission | 2.2 (2.8) | 4.2 (2.8) | 4.5 (1.7) |

  - Random positions: omissions are 13.6–16.9 for humans; commissions are 2.1–3.7.
  - Largest human chunk: 7.6 (Class A), 12.3 (Expert), 14.5 (Master), with about 3 chunks at every skill level (Table 4).
  - Source: [GS00](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)

**Chess AI that chunks**
- **Chunker** (Campbell & Berliner, AAAI-83; Berliner & Campbell 1984, "Using chunking to solve chess pawn endgames," *Artificial Intelligence* 23(1):97–120 — [doi](https://doi.org/10.1016/0004-3702(84)90006-7)).
  - Chunks were "groups of pawns and king that can be handled relatively independently." Each chunk type had a library of instances with property lists. — [chessprogramming (secondary)](https://www.chessprogramming.org/Chunker)
- **PARADISE** (Wilkins 1980, "Using patterns and plans in chess," *Artificial Intelligence* 14(2):165–203 — [doi](https://doi.org/10.1016/0004-3702(80)90039-9)). Production-rule patterns propose plans such as fork, skewer and trapping, which a small search then verifies. — [chessprogramming](https://www.chessprogramming.org/Paradise)
- **Morph** (Levinson & Snyder, AAAI-91). Positions are **graphs whose nodes are pieces and squares and whose edges are attack/defense relations**. Pattern weights are learned by temporal-difference learning. — [chessprogramming](https://www.chessprogramming.org/Morph)
- **CHUMP** (Gobet & Jansen 1994). It is CHREST with two discrimination nets, one for patterns and one for moves, with moves linked to patterns. — [chessprogramming](https://www.chessprogramming.org/CHUMP)
- **KRKPA7** (Shapiro's structured-induction endgame). 3,196 instances, binary class, CC BY 4.0. — [UCI](https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn)

**Modern work**
- **McGrath et al. (2022).** "Acquisition of chess knowledge in AlphaZero," *PNAS* 119(47):e2206625119 ([doi](https://doi.org/10.1073/pnas.2206625119)).
  - Sparse linear probes targeted Stockfish evaluation terms and **116 custom concepts** such as pins, forks and mate threats.
  - Material emerged early in training; king safety and mobility later. — [arXiv PDF](https://arxiv.org/pdf/2111.09259v3)
- **Schut et al. (2025).** "Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero," *PNAS* 122(13):e2406675122 ([doi](https://doi.org/10.1073/pnas.2406675122)). Concept vectors were filtered for teachability and novelty; four top grandmasters improved after studying them.
- **Chess language models.**
  - Toshniwal, Wiseman, Livescu & Gimpel, "Chess as a Testbed for Language Model State Tracking," AAAI-22 36(10):11385–11393 ([doi](https://doi.org/10.1609/aaai.v36i10.21390)).
  - Karvonen, COLM 2024: linear probes recover board state and player skill from a character-level PGN model; code is MIT. — [arXiv 2403.15498](https://arxiv.org/abs/2403.15498); [repo](https://github.com/adamkarvonen/chess_llm_interpretability)
- **Ruoss et al. / ChessBench.**
  - 10M games with 15B Stockfish-16 annotations; a 270M transformer reaches Lichess blitz Elo 2895 without search.
  - Code Apache-2.0; data CC0 (Lichess-derived) or CC-BY 4.0.
  - The NeurIPS 2024 venue is stated in the repo README only.
  - Sources: [arXiv 2402.04494](https://arxiv.org/abs/2402.04494); [repo](https://github.com/google-deepmind/searchless_chess)
- **Human-move baselines.**
  - Maia: McIlroy-Young et al., KDD '20, pp. 1677–1687 ([doi](https://doi.org/10.1145/3394486.3403219)); GPL-3.0 ([repo](https://github.com/CSSLab/maia-chess)).
  - Maia-2: NeurIPS 2024, skill-aware ([arXiv 2409.20553](https://arxiv.org/abs/2409.20553)); MIT ([repo](https://github.com/CSSLab/maia2)).
- **Chess960 as a test of transfer.** In a 270M chess transformer, human concepts decode best from early layers (up to 85%). A released Chess960 set of 240 annotated positions over 6 concepts shows **10–20% recognition drops** in Chess960. — [Lomasov et al. 2025, arXiv 2510.26025](https://arxiv.org/abs/2510.26025)

**Data and tools**
- **Lichess open database.**
  - "Released under the Creative Commons CC0 license." Monthly PGN .zst files total about **8.22 billion** standard rated games, 2013-01 to 2026-09.
  - The evaluation DB has about 416M positions in JSONL.
  - Sources: [database.lichess.org](https://database.lichess.org/); [counts](https://database.lichess.org/standard/counts.txt)
- **Lichess puzzle DB.**
  - About **6.16M** puzzles (CC0), with CSV columns `PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags`.
  - Themes are auto-tagged and include fork, pin, skewer, discoveredAttack, xRayAttack, interference, deflection, attraction, clearance, hangingPiece, trappedPiece, doubleCheck, named mates and mateIn1–5.
  - Ratings are Glicko-2 estimates from human solvers, so they double as human difficulty data.
  - Sources: [database.lichess.org](https://database.lichess.org/); [theme keys](https://raw.githubusercontent.com/lichess-org/lila/master/translation/source/puzzleTheme.xml)
- **python-chess** (GPL-3.0+).
  - `Board.status()` flags positions with no king, too many kings, pawns on the back rank, the side not to move in check, impossible checks, and similar defects.
  - STATUS_VALID means basic requirements are met, but **"reachability is not guaranteed."**
  - Relation helpers: `attacks`, `attackers`, `is_attacked_by`, `is_pinned`, `pin`, `SquareSet.ray/between`.
  - Sources: [docs](https://python-chess.readthedocs.io/en/latest/core.html); [repo](https://github.com/niklasf/python-chess)
- **Adjacent board games with recall data** (stepping stones).
  - **Go recall:** Reitman (1976), "Skilled perception in Go: Deducing memory structures from inter-response times," *Cognitive Psychology* 8(3):336–356. — [Crossref doi](https://doi.org/10.1016/0010-0285(76)90011-6)
  - **4-in-a-row:** van Opheusden et al. (2023), "Expertise increases planning depth in human gameplay," *Nature* 618:1000–1005 ([doi](https://doi.org/10.1038/s41586-023-06124-2)). Its abstract reports a heuristic-search model validated on choices, response times and eye movements, plus "a Turing test and a reconstruction experiment"; "Experts memorize and reconstruct board features more accurately." — [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41586-023-06124-2&format=json&resultType=core)

### Inferences
- **The recall paradigm is ready-made TRELLIS evaluation.**
  - **Parse:** recognise a position's chunks.
  - **Generate:** reconstruct the position from a few chunk identifiers (≈3–7 in STM) by top-down decomposition.
  - **Score:** % correct, omissions and commissions, in exactly the units Gobet & Simon report for humans and CHREST.
- **The decisive comparison is commission.** CHREST mis-predicts commission errors. A Cobweb taxonomy, which keeps probabilistic slot distributions and defaults at every level, might predict them better. That would be a crisp, publishable cognitive-modelling result.
- **CHREST's template mechanism is a special case of Cobweb concept formation.** Low-entropy attributes are the core; high-entropy attributes are slots; modal values are defaults. CHREST's slot-creation threshold ("more than 3 nodes share it") parallels TRELLIS's count/maturity cut. Langley's "different pieces occupy the same role" becomes a distribution over fillers in a role slot.
- **What TRELLIS adds that CHREST lacks** (all flagged in Langley's essay):
  - **relations** (attack, defense, proximity, line, pin), which CHREST's next-POS tests lack;
  - **context** (what attacks or defends the chunk from outside);
  - **compositional acquisition** of higher chunks from lower ones.
- **Location coding is an empirical design question to settle with data.** Human chunks are location-specific (mirror-image recall drops).
  - Encode an absolute anchor square **and** relative offsets as content attributes, and let Cobweb's attribute distributions decide.
  - A peaked anchor distribution means a location-bound chunk; a diffuse one means a location-general chunk.
  - Mirror and translated positions, and Chess960, then become out-of-distribution tests with both human (Gobet & Simon) and machine (Lomasov et al.) comparisons.
- **Chess is not context-free.** Legality and reachability are global constraints. python-chess `status()` gives an automatic **necessary-condition commission check** (illegal piece counts, back-rank pawns, impossible checks) for generated positions. It is not a reachability check.

### Gaps
- **Human data:** no public trial-level human recall or latency dataset was found. Human targets must come from published tables (Chase & Simon 1973; Gobet & Simon 1996, 2000).
- **Unread sources:**
  - Saariluoma (1994): findings not read.
  - Gobet & Simon (1996): which mirror axes and translations the *Memory & Cognition* study used was not retrieved.
  - van Opheusden et al. (2023): board size, dataset sizes and data license were not retrieved; the OSF page did not render.
- **Secondary sources only:** the details for Chunker, Morph, PARADISE and CHUMP come from chessprogramming.org.
- **Software:** the current maintained CHREST release (and its license) is unclear; chrest.info had an expired TLS certificate.
- **No gold parse trees** exist for chess positions. Gold-like proxies:
  - Chase & Simon-style 2-s latency chunks, which are not public;
  - CHREST chunks;
  - puzzle themes and opening tags.

## Q3. 2D and visual compositional hierarchies: which offer discrete relational structure where hierarchy matters for the parse?

### Takeaway
The image-grammar literature (Zhu & Mumford's And-Or graph; Tu et al. 2013) already formalizes three things that line up with TRELLIS:
- **And = chunk content.** An And-node decomposes into components.
- **Or = concept over alternatives.** An Or-node chooses among alternative sub-configurations.
- **Horizontal relation links = context.**

The most TRELLIS-relevant learning ideas are Zhu et al.'s (2008) **suspicious coincidence** (a frequency gate) and **competitive exclusion** (pruning overlapping chunks), and Tu et al.'s separate **content-coherence and context-coherence** terms.

For a *discrete* 2D domain with gold structure, a real legality notion, and human data, the standout is **Chinese characters via Ideographic Description Sequences (IDS)**:
- The structures are relation-labelled trees, typically 2–5 levels deep.
- Arrangement matters: 呆=⿱口木 and 杏=⿱木口 have the same parts in different positions.
- There are ~89k entries.
- Human lexical-decision data exist for thousands of characters *and* ~4.9k structurally generated pseudocharacters.

Orbán/Fiser–Aslin grid scenes are the smallest exact human-comparison pilot. Lake & Piantadosi's L-system figures are the cleanest test of using a grammar out of distribution (deeper recursion than seen in training).

### Cited Findings
**Image grammars and structure learning**
- **Zhu & Mumford, "A Stochastic Grammar of Images."** *Foundations and Trends in Computer Graphics and Vision* 2(4):259–362 (book 2006; journal issue dated 2007) ([doi](https://doi.org/10.1561/0600000018)).
  - The abstract calls it "a stochastic and context sensitive grammar of images."
  - Decompositions run "from scenes, to objects, parts, primitives and pixels."
  - "Horizontal links between the nodes" carry spatial and functional relations.
  - "each Or-node points to alternative sub-configurations and an And-node is decomposed into a number of components."
  - A category is "the set of all possible valid configurations produced by the grammar."
  - Source: [Crossref abstract](https://doi.org/10.1561/0600000018)
- **Zhu, Lin, Huang, Chen & Yuille (ECCV 2008)**, LNCS pp. 759–773 ([doi](https://doi.org/10.1007/978-3-540-88688-4_56); [PDF](https://people.csail.mit.edu/leozhu/paper/usl_eccv_2008.pdf)). The method recursively composes triplets of lower-level parts, clustered on relative position, scale and orientation, from 4 oriented edgelet primitives.
  - **Suspicious coincidence:** "keep compositions which occur frequently"; concepts whose instances appear in <90% of images are removed.
  - **Competitive exclusion:** remove concepts whose instances "have significant overlap with instances of other concepts (and the other concepts have better scores)."
  - **Effect of the two filters at level 1** (12 training images):

    | Stage | Count |
    |---|---|
    | Proposals | 167,431 |
    | Clusters | 14,684 |
    | After suspicious coincidence | 262 |
    | After competitive exclusion | 48 |

  - Competitive exclusion "is the main factor that causes the bottom-up process to stop" after 4–5 levels.
  - A top-down pass relaxes both rules to fill gaps.
  - Weizmann-horse segmentation reached 93.3% from 12 unlabeled images.
- **Fidler & Leonardis (CVPR 2007)** ([doi](https://doi.org/10.1109/CVPR.2007.383269)).
  - It learns "spatially flexible compositions" layer by layer, keeping the "statistically most significant compositions."
  - Lower layers are category-independent and shared; higher layers are category-specific.
- **Si & Zhu (2013)**, "Learning AND-OR Templates for Object Recognition and Detection," *IEEE TPAMI* 35(9):2189–2205 ([doi](https://doi.org/10.1109/TPAMI.2013.35)).
  - Node types: AND (composition); geometric OR (deformation); structural OR (alternative compositions).
  - Learning is unsupervised: block pursuit builds the dictionary, then graph compression "minimize[s] model structure."
- **Tu, Pavlovskaia & Zhu (NeurIPS 2013)**, "Unsupervised Structure Learning of Stochastic And-Or Grammars." Sources: [abstract](https://proceedings.neurips.cc/paper/2013/hash/24681928425f5a9133504de568f5f6df-Abstract.html); [PDF](https://papers.neurips.cc/paper_files/paper/2013/file/24681928425f5a9133504de568f5f6df-Paper.pdf)
  - **Learning procedure:**
    - It starts from a grammar that generates exactly the training samples.
    - It greedily adds "And-Or fragments" (a new And-node plus Or-nodes beneath it) by posterior gain.
    - The gain factorizes into coherence of the fragment's n-gram tensor (**content**) and of its context matrix (**context**).
  - **Experiments:** events and images only, with no string-grammar experiments.
  - **Synthetic animal-face sketches:**
    - The authors "estimated the precision and recall of the sets of images generated from the learned grammars versus the true grammar." That is, commission-side and omission-side scoring, as in Langley & Stromsten.
  - **Real animal faces:** 10-fold perplexity 67.5, against 129.4 for Si & Zhu.

**Characters and drawings**
- **BPL** (Lake, Salakhutdinov & Tenenbaum 2015, *Science* 350(6266):1332–1338; [doi](https://doi.org/10.1126/science.aab3050); [PDF](https://www.cs.cmu.edu/~rsalakhu/papers/LakeEtAl2015Science.pdf)).
  - A character type is "an abstract schema of parts, subparts, and relations." Parts are strokes; subparts come from a learned discrete set of primitive actions.
  - Each part begins "independently, at the beginning, at the end, or along previous parts."
  - **One-shot 20-way classification error:**

    | Learner | Error |
    |---|---|
    | Humans | 4.5% |
    | BPL | 3.3% |
    | Deep convnet | 13.5% |
    | BPL without compositionality | 14.0% |

  - **Visual Turing tests:**
    - New exemplars: judges identified machine vs human at 52% (chance is ideal), N=147.
    - New concepts sampled from a part-reusing prior: 51%.
- **Omniglot dataset.** 1,623 characters from 50 alphabets, 20 drawers per character, with stroke data `[x, y, t]`; MIT license. — [GitHub](https://github.com/brendenlake/omniglot); [TFDS](https://www.tensorflow.org/datasets/catalog/omniglot)
  - The 2019 progress report found classification gains came partly from "new splits and procedures that make the task easier," with "less progress on the other four tasks." — [doi](https://doi.org/10.1016/j.cobeha.2019.04.007)
- **RCN** (George et al. 2017, *Science* 358(6368):eaag2612; [doi](https://doi.org/10.1126/science.aag2612)). A hierarchical generative vision model; reference code is MIT. — [repo](https://github.com/vicariousinc/science_rcn)
- **Lake & Piantadosi**, "People Infer Recursive Visual Concepts from Just a Few Examples," *Computational Brain & Behavior* 3(1):54–65 ([doi](https://doi.org/10.1007/s42113-019-00053-y); [arXiv](https://arxiv.org/abs/1904.08034)).
  - Stimuli: L-system "alien crystals" with recursion depth *d*.
  - Classification of novel exemplars: 64.9% (chance 16.7%). Comparisons: pretrained ConvNet 4.4%, Hausdorff 17.4%.
  - Generation was "highly structured and generally consistent with the underlying program."
- **Tian, Ellis, Kryven & Tenenbaum (2020).** People drawing objects built from composable geometric rules learn reusable abstract procedures. A model constrained to efficient motor actions discovers human-like routines. — [arXiv 2008.03519](https://arxiv.org/abs/2008.03519)

**Human visual statistical learning (chunks in 2D scenes)**
- **Fiser & Aslin (2001)**, *Psychological Science* 12(6):499–504 ([doi](https://doi.org/10.1111/1467-9280.00392)). Under passive viewing, people learn joint and conditional shape co-occurrence statistics, including "shape-pair arrangements independent of position." The authors framed this as Barlow's "suspicious coincidences."
- **Fiser & Aslin (2005)**, *JEP: General* 134(4):521–537 ([doi](https://doi.org/10.1037/0096-3445.134.4.521)). Shape combinations "that are parts of larger configurations are less well remembered" than the same kind of combination appearing on its own.
- **Orbán, Fiser, Aslin & Lengyel (2008)**, "Bayesian learning of visual chunks by human observers," *PNAS* 105(7):2745–2750 ([PMC2268207](https://pmc.ncbi.nlm.nih.gov/articles/PMC2268207); [Crossref doi](https://doi.org/10.1073/pnas.0708424105)).
  - **Stimuli:** 12 shapes on 3×3 or 5×5 grids; each scene holds 2–3 fixed "combos"; 20–32 participants per experiment.
  - **Familiarity:** true combos are judged familiar, while embedded sub-pairs and sub-triplets are at chance.
  - **Model:** a Bayesian chunk learner keeps only chunks "minimally sufficient" to encode the scenes.
  - **Fit to human data:**

    | Comparison | Bayesian chunk learner | Associative learner |
    |---|---|---|
    | 12 earlier tests | r = 0.88 | r = 0.74 |
    | New balanced-statistics triplet experiment | r = 0.92 | r = −0.23 |

- **Brady, Konkle & Alvarez (2009)**, *JEP: General* 138(4):487–502 ([doi](https://doi.org/10.1037/a0016797)). Covarying color pairs increase how many items fit in visual working memory, through more efficient (compressed) representations.

**Construction and library learning; ARC; design grammars**
- **Block-tower instruction study** (McCarthy, Hawkins, Wang, Holdaway & Fan 2021). Pairs of people rebuilding block-tower scenes develop shorter instructions that capture "each scene's hierarchical structure." The model is library learning plus convention formation. — [arXiv 2107.00077](https://arxiv.org/abs/2107.00077)
  - Code and data: [cogtoolslab/compositional-abstractions](https://github.com/cogtoolslab/compositional-abstractions) (no license detected).
- **Concept libraries from language** (Wong et al., CogSci 2022). About 2k procedurally generated objects with language descriptions. Humans favour libraries that balance description length against lexicon size. — [arXiv 2205.05666](https://arxiv.org/abs/2205.05666); code MIT ([repo](https://github.com/cogtoolslab/lax-cogsci22))
- **KiloGram** (Ji et al., EMNLP 2022, pp. 582–601). More than 1k tangrams with part annotations; MIT, except the images, which are research-only. — [doi](https://doi.org/10.18653/v1/2022.emnlp-main.38); [repo](https://github.com/lil-lab/kilogram)
- **ARC.**
  - Chollet (2019) assumes priors such as objectness ("parse grids into 'objects' based on continuity criteria") and geometry/topology. — [arXiv 1911.01547](https://arxiv.org/abs/1911.01547); [data (Apache-2.0)](https://github.com/fchollet/ARC-AGI)
  - **ARC-AGI-2:** 407 human testers; every task was solved by at least 2 people within 2 attempts. — [arXiv 2505.11831](https://arxiv.org/abs/2505.11831)
  - **H-ARC:** 1,729 people on all 800 ARC-1 tasks. Mean accuracy was 76.2% (training) and 64.2% (evaluation); action traces and rule descriptions are on OSF under CC0. — [arXiv 2409.01374](https://arxiv.org/abs/2409.01374); [*Scientific Data* 12 (2025)](https://doi.org/10.1038/s41597-025-05687-1)
  - **ARC-AGI-3** (interactive, 64×64 grids) launched 25 March 2026, with frontier AI below 1% at launch. — [launch post](https://arcprize.org/blog/arc-agi-3-launch); [arXiv 2603.24621](https://arxiv.org/abs/2603.24621)
- **Design grammars.**
  - Talton et al. (UIST 2012, pp. 63–74) induce probabilistic grammars from "labeled, hierarchical designs" (web pages, 3D models) by Bayesian model merging. — [doi](https://doi.org/10.1145/2380116.2380127)
  - Martinovic & Van Gool (CVPR 2013, pp. 201–208) learn 2D attributed stochastic CFGs for facades. Their parsing is on par with a hand-written grammar. — [doi](https://doi.org/10.1109/CVPR.2013.33)

**Chinese characters as a discrete 2D grammar**
- **Unicode operators.** Ideographic Description Characters define the relation operators:
  - ⿰ left→right, ⿱ above→below, ⿲/⿳ three-part, ⿴ full surround, ⿵–⿺ partial surrounds, ⿻ overlay;
  - newer additions: U+2FFC–U+2FFF and U+31EF.
  - An IDS is a prefix expression over components, which makes it a relation-labelled parse tree. — [UnicodeData.txt](https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt)
- **IDS data.**
  - Licenses: CHISE IDS is GPL-2.0-or-later ([README](https://gitlab.chise.org/CHISE/ids/-/raw/master/README.md)). cjkvi-ids `ids.txt` derives from CHISE and follows the GPLv2 ([repo](https://github.com/cjkvi/cjkvi-ids)).
  - The researcher's counts from [`ids.txt`](https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt):
    - 88,937 entries, 20,976 of them in U+4E00–9FFF;
    - 3,502 entries with alternative IDSs;
    - operator counts: ⿰ 60,086, ⿱ 26,442;
    - recursively expanded depth: mostly 2–5 (mode 3, maximum 8), with a mean of 5.25 leaves.
- **Zero-shot recognition of unseen characters.** The Radical Analysis Network (Zhang, Zhu, Du & Dai, ICME 2018) decodes radicals plus 2D structure, so it recognizes characters never seen in training. More than 20k characters decompose into ~500 radicals. — [doi](https://doi.org/10.1109/ICME.2018.8486456)
- **Human data.**
  - **Taft, Zhu & Peng (1999)**, *JML* 40(4):498–519 ([doi](https://doi.org/10.1006/jmla.1998.2625)): no transposition effect when the two radicals swap positions. Radical representations are therefore **position-specific**.
  - **Yeh et al. (2003)**, *Visual Cognition* 10(6):729–764 ([doi](https://doi.org/10.1080/13506280344000077)): literate Taiwanese and Japanese students sort characters by **configuration**. American students, illiterate adults and kindergartners sort by strokes or components.
  - **Chinese Lexicon Project:** lexical decision for 2,500 characters. — [doi](https://doi.org/10.3758/s13428-013-0355-9)
  - **Simplified Chinese Lexicon Project** (Wang, Wang, Chen & Keuleers 2025, *BRM* 57(7)): lexical decision for all 8,105 standard characters plus **4,864 pseudocharacters** generated "using a novel method that leveraged the hierarchical nature of Chinese characters." — [doi](https://doi.org/10.3758/s13428-025-02701-7)

**Cobweb's image lineage.** Both of these categorize only; neither has parts or a partonomy.
- Convolutional Cobweb (MacLellan & Thakur), presented at ACS 2021. — [arXiv 2201.06740](https://arxiv.org/abs/2201.06740)
- Cobweb/4V (Barari, Lian & MacLellan). — [arXiv 2402.16933](https://arxiv.org/abs/2402.16933); journal version in *Cognitive Systems Research* 96:101447 (2026), [doi](https://doi.org/10.1016/j.cogsys.2026.101447)

### Inferences
- **Which visual domains offer discrete relational structure where hierarchy matters for the parse:**
  - **IDS characters:** discrete relations, gold trees, human data on legal novel items.
  - **Synthetic 2D And-Or grammars:** fully controllable; Tu-style scoring of generated vs accepted samples.
  - **L-system figures:** recursion depth gives an out-of-distribution axis.
  - **Block towers:** discrete grid placement and subassemblies.
- **Weaker fits:**
  - Omniglot: its sub-strokes are continuous and must be quantized.
  - Tangrams: continuous geometry.
  - ARC: no fixed grammar, so commission and omission are undefined.
  - Facades: little human data.
- **IDS characters fit v1's training regime almost exactly.** They are gold trees labelled with relations but not with categories. They differ from v1 in two ways TRELLIS v2 needs: n-ary operators (⿲/⿳) and relation types beyond "before." Concretely:
  - **Content** = (operator, ordered components).
  - **Context** = (parent operator, slot index, sibling).
  - **Commission** = a generated character has a component in an unattested slot or operator, or is rejected as a non-character in a rating study. SCLP pseudocharacters give a human-validated comparison set.
  - **Omission** = held-out real characters (including rare Extension-A+ ones) that fail to parse to a single root.
- **The human data make two falsifiable predictions for TRELLIS's concept facet:**
  - Position-specific radicals (Taft et al. 1999) say the slot index must stay in the content description.
  - The novice→expert shift from component-based to configuration-based similarity (Yeh et al. 2003) should appear as content-taxonomy organization changing with training.
- **Orbán et al. provide a sharp test of chunk minimality.** Humans (and the Bayesian chunk learner) do not find embedded sub-chunks familiar.
  - If TRELLIS builds triplets through recognizable intermediate pairs, it wrongly predicts embedded-pair familiarity.
  - Competitive-exclusion-style pruning of subsumed chunks (Zhu et al. 2008), or a compression criterion, would restore the human pattern.
  - This is a direct, cheap experiment for v2's "decide which chunks are worth keeping" goal.

### Gaps
- **Unverified data:**
  - whether Orbán/Fiser–Aslin, Brady et al., and Lake & Piantadosi stimuli or raw data are public;
  - SCLP data-release details and the pseudocharacter construction method;
  - the exact numbers in Fiser & Aslin (2005) and Brady et al. (2009).
- **Unverified procedure:** Tu et al. (2013) do not describe their precision/recall sampling protocol in the main text.
- **Licensing:** the GPLv2 on IDS data may matter if TRELLIS code or derived datasets are redistributed. Using it for research is unaffected.
- **ARC-AGI-3:** status after launch rests on non-peer-reviewed preprints and was not verified.

## Q4. Music and other symbolic sequences: GTTM, harmonic grammars, IDyOM segmentation, datasets and human data

### Takeaway
Music is the most turnkey second domain for TRELLIS because it supplies **gold hierarchical trees** in v1's own training regime. The Jazz Harmony Treebank has 150 expert binary trees over chord sequences. It also has human segmentation data for testing **unsupervised** chunk discovery, at two hierarchical levels. Its weakness for v2's goals is that it is still mostly a 1-D sequence. The genuinely new relations are interval, metre and, in polyphony, simultaneity and voice. So it stresses the concept/context side more than the non-linear-relation side.

### Cited Findings
**Grammars and treebanks**
- **GTTM components.** Lerdahl & Jackendoff (1983) define four components: grouping structure; metrical structure; a time-span tree (a binary tree of the relative structural importance of notes); and a prolongational tree (a binary tree of tension and relaxation).
  - Grouping preference rules are Gestalt-based: rest or slur (GPR 2a), attack-point (2b), register/dynamics/articulation/length change (3a–d), intensification (4) and parallelism (6).
  - Sources: [Hamanaka, Hirata & Tojo 2014](https://archives.ismir.net/ismir2014/paper/000257.pdf); [Pearce, Müllensiefen & Wiggins 2010](https://research.gold.ac.uk/id/eprint/5380/1/p6507.pdf)
- **Computational GTTM.** Hamanaka, Hirata & Tojo (2006), "Implementing 'A Generative Theory of Tonal Music'," *JNMR* 35(4):249–277 ([doi](https://doi.org/10.1080/09298210701563238)).
  - The analyser line runs ATTA (manual rule priorities) → FATTA → σGTTM (decision trees learned from 100 analysed pieces) → σGTTM II. — [ISMIR 2014](https://archives.ismir.net/ismir2014/paper/000257.pdf)
- **GTTM database.** 300 eight-bar *monophonic* classical excerpts, in MusicXML plus Grouping/Metrical/Timespan/Prolongational/Harmonic XML. One expert analysed them all and three others cross-checked.
  - A second musicologist's re-analysis matched on 267/300 pieces. In the 33 pieces that differed, 233 of 2,310 time-spans disagreed, which gives a human ceiling.
  - Download: gttm.jp. Terms of use are not stated in the paper. — [ISMIR 2014](https://archives.ismir.net/ismir2014/paper/000257.pdf)
- **Jazz chord grammars.** Steedman (1984), "A Generative Grammar for Jazz Chord Sequences," *Music Perception* 2(1):52–77 ([doi](https://doi.org/10.2307/40285282)).
  - Granroth-Wilding & Steedman (2014), *JNMR* 43(4):355–374 ([doi](https://doi.org/10.1080/09298215.2014.910532)), built a CCG-based parser-interpreter that beat an HMM baseline on a small corpus. — [talk page](https://www.musica.ed.ac.uk/archive/2013/mark-granroth-wilding/)
- **Tonal and jazz syntax.** Rohrmeier (2011), "Towards a generative syntax of tonal harmony," *J. Mathematics and Music* 5(1):35–53 ([doi](https://doi.org/10.1080/17459737.2011.573676)).
  - Rohrmeier (2020), "The Syntax of Jazz Harmony," *Music Theory and Analysis* 7(1):1–63 ([doi](https://doi.org/10.11116/mta.7.1.1)), treats prolongation and preparation as the two basic principles. — [JHT paper](https://program.ismir2020.net/static/final_papers/80.pdf)
- **Harmonic parsing.** Harasim, Rohrmeier & O'Donnell (2018), ISMIR, pp. 152–159, introduce probabilistic abstract CFGs that handle modulation and long sequences. — [anthology](https://ir.webis.de/anthology/2018.ismir_conference-2018.20)
- **Jazz Harmony Treebank (JHT)** (Harasim, Finkensiep, Ericson, O'Donnell & Rohrmeier, ISMIR 2020). Sources: [paper](https://program.ismir2020.net/static/final_papers/80.pdf); [GitHub](https://github.com/DCMLab/JazzHarmonyTreebank)
  - **Size:** 150 complete standards, mean 27.75 chords each; 11,697 chords; 92 chord symbols. Average tree height is 7.57, so hierarchy matters for the parse.
  - **Rules:** 3,899 binary rule applications over 512 unique rules. Three schemas: strong prolongation X→X X; weak prolongation X→Y X with Y functionally equivalent; preparation X→Y X.
  - **Heads:** internal nodes are labelled by head chord, and trees are mostly right-headed.
  - **Exclusions:** tunes with crossing dependencies are left out.
  - **Format and license:** JSON; CC BY-NC-SA 4.0 ([LICENSE](https://github.com/DCMLab/JazzHarmonyTreebank/blob/master/LICENSE.md)).
- **Published JHT parse scores** (reported in the source as a single combined span score). Cartuyvels, Koslovsky & Moens (BNAIC 2024) trained the first fully unsupervised neural PCFG on raw chord sequences. — [PDF](https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/11/Unsupervised-Induction-of-Harmonic-Syntax.pdf)

  | Model | Combined span score on JHT |
  |---|---|
  | Unsupervised neural PCFG | 0.387 |
  | + ChoCo data | 0.455 |
  | + fifth-relation loss | 0.477 |
  | Supervised MuDeP dependency parser (Foscarin, Harasim & Widmer, ISMIR 2023) | 0.623 |
  | Random | 0.178 |

  The unsupervised model recovers ii–V–I chunks. — [PDF](https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/11/Unsupervised-Induction-of-Harmonic-Syntax.pdf); MuDeP: [arXiv 2306.16955](https://arxiv.org/abs/2306.16955)
- **Polyphony as 2-D structure.** Finkensiep & Rohrmeier's proto-voice model (ISMIR 2021, pp. 189–196) encodes sequential *and* vertical relations through recursive operations such as neighbour and passing insertion and "horizontalization," and parses with chart parsing. — [anthology](https://ir.webis.de/anthology/2021.ismir_conference-2021.23)
- **Schenkerian data.** SCHENKER41 has 41 excerpts with machine-readable Schenkerian analyses and a "please cite" condition but no formal license. — [Kirlin data page](https://www.cs.rhodes.edu/~kirlinp/schenker)
- **Recent grammar-induction work on music.**
  - Perkins & Ventura (2024) use CFG induction for phrase segmentation. — [arXiv 2405.18742](https://arxiv.org/abs/2405.18742)
  - Ren, Guan & Rohrmeier (2025) detect repeats of hierarchical relations with "Template" programs that add repetition combinators to CFGs. — [arXiv 2504.10065](https://arxiv.org/abs/2504.10065)
  - Tsushima et al. (2017) learned latent harmonic-function categories without supervision. This is the closest precedent for T/S/D *concepts* emerging. — [arXiv 1708.02255](https://arxiv.org/abs/1708.02255)

**Information-theoretic segmentation and human boundary data**
- **IDyOM.** It predicts each note from variable-order context, combining a long-term corpus model with a short-term model of the current piece. It places boundaries before high-information-content notes and is never trained on boundaries.
  - Software is Common Lisp under GPL-3.0. — [GitHub](https://github.com/mtpearce/idyom); [Pearce et al. 2010](https://research.gold.ac.uk/id/eprint/5380/1/p6507.pdf)
  - Pearce (2018), *Annals NYAS* 1423(1):378–395 ([doi](https://doi.org/10.1111/nyas.13654)), frames enculturation as statistical learning plus probabilistic prediction.
- **Pearce, Müllensiefen & Wiggins (2010)**, *Perception* 39(10):1367–1391 ([doi](https://doi.org/10.1068/p6507)). Source for all points below: [PDF](https://research.gold.ac.uk/id/eprint/5380/1/p6507.pdf)
  - **Study:** 25 trained listeners marked strong and weak phrase boundaries on 15 melodies of 39–131 notes.
  - **Scoring:** models were scored against the best-matching participant cluster.

  | Model | Precision (commission side) | Recall (omission side) |
  |---|---|---|
  | Grouper | 0.86 | 0.82 |
  | LBDM | 0.79 | 0.81 |
  | IDyOM | 0.57 | 0.73 |
  | Transition probability | 0.25 | 0.45 |
  | Always-boundary | 0.13 | 1.0 |

  - **Hybrid:** a logistic hybrid scored 0.78/0.70 at phrase level and 0.88/0.66 at subphrase level. That gives two hierarchical levels of human boundaries.
- **Sequence chunking models.** PARSER (Perruchet & Vinter 1998, *JML* 39(2):246–263; [doi](https://doi.org/10.1006/jmla.1998.2576)) is a classic chunking model of sequence segmentation. Its bibliographic record is verified; applications of it to music were not.

**Human data on hierarchical and relational music memory**
- **Hierarchical coding.** Deutsch & Feroe (1981), "The internal representation of pitch sequences in tonal music," *Psych. Review* 88(6):503–522 ([doi](https://doi.org/10.1037/0033-295x.88.6.503)), proposes hierarchical, alphabet-and-operator coding.
- **Recall.** Deutsch (1980), "The processing of structured and unstructured tonal sequences," *Perception & Psychophysics* 28(5):381–389 ([doi](https://doi.org/10.3758/bf03204881)), is a recall paradigm comparing structured and unstructured sequences, the music analogue of game vs random chess positions.
- **Relational memory.** Dowling (1978), "Scale and contour," *Psych. Review* 85(4):341–354 ([doi](https://doi.org/10.1037/0033-295x.85.4.341)), separates relational contour from scale information.
- **Hierarchy vs local structure.** Koelsch, Rohrmeier, Torrecuso & Jentschke (2013), *PNAS* 110(38):15443–15448 ([doi](https://doi.org/10.1073/pnas.1300272110)), showed that musicians and non-musicians respond differently to Bach chorales whose **hierarchical structure was made irregular while local structure stayed intact**. — [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1073/pnas.1300272110%22&resultType=core&format=json)
- **Tension.** Lerdahl & Krumhansl (2007), "Modeling Tonal Tension," *Music Perception* 24(4):329–366 ([doi](https://doi.org/10.1525/mp.2007.24.4.329)).

**Other datasets.** These are bulk material for unsupervised training and the cross-culture tests in the Inferences; none has gold trees.

| Dataset | Content | License or status |
|---|---|---|
| Essen (EsAC, kern) | About 8.5k songs with phrase marks | Unverified |
| Meertens Tune Collections | Tune families and phrases | Collection license unverified; the MTCFeatures package is CC BY-NC-SA 3.0 |
| DCML ABC | Beethoven quartets with Roman numerals and phrases | CC BY-NC-SA 4.0 |
| Nottingham (cleaned) | Folk tunes | GPL-3.0 |

Sources: [MTC](https://liederenbank.nl/mtc); [zenodo](https://zenodo.org/record/3551003); [ABC](https://github.com/DCMLab/ABC); [Nottingham](https://github.com/jukedeck/nottingham-dataset)

### Inferences
- **JHT is a near drop-in replacement for v1's synthetic CFG trees.** TRELLIS would:
  - strip head labels to keep v1's unlabeled-tree regime;
  - train on binary trees whose elements are chords;
  - score span omission and commission against held-out gold trees.
- **Baselines exist.** The unsupervised N-PCFG (0.39–0.48) and supervised MuDeP (0.62) are published, though they are reported only as combined span scores. TRELLIS should report the omission and commission sides separately.
- **The concept facet would be tested directly.**
  - Do functional classes (T/S/D, and "dominant-of-X" including tritone substitutes) emerge in the **context** taxonomy?
  - Do prolongation and preparation emerge as distinct **content** concepts?
- **Natural out-of-distribution tests:**
  - Transposition to unseen keys, if roots are encoded relative to the key or as intervals.
  - Held-out substitutions.
  - Longer forms.
  - Koelsch-style hierarchical-violation sequences: the top-level parse cost should separate them, while n-gram/IDyOM surprisal should not.
- **The unsupervised track** compares chunk boundaries at two levels (phrase and subphrase) against human data, with published precision/recall baselines.
- **Fit with v2 goals.** Music adds less to the "2-D/non-linear relations" goal than chess or characters do, unless polyphony (proto-voices) is tackled. It is therefore best cast as the third domain, or as a fast sanity domain for the inside-outside parser.

### Gaps
- **Licenses and counts:**
  - GTTM-DB license and terms are unstated.
  - Essen and Meertens licenses are unverified, and the Essen count rests on a search snippet.
  - No public dataset of human grammaticality/acceptability judgments for chord sequences was found.
- **Human data:** whether Pearce et al.'s (2010) raw participant boundary data are public is unknown.
- **Code:** availability for MuDeP, the PACFG parser and the JHT neural PCFG was not checked.
- **Unverified sources:**
  - GTTM's well-formedness vs preference-rule distinction (the MIT Press page was blocked).
  - deepGTTM.
  - Harasim et al. (2019): the rhythm-and-harmony details were not retrieved.

## Q5. Action, plans and programs: HTN learning, Soar chunking, macro-operators, option discovery, and library learning as chunking

### Takeaway
Plans and programs are the domain where Langley's own lineage is strongest: ICARUS and HTN learning by observation. Langley (2025, AAAI) breaks HTN acquisition into three subproblems that line up with TRELLIS's three facets:
- **identifying hierarchical structure** → chunk formation, i.e. content;
- **unifying method heads** → concepts over chunks;
- **finding method conditions** → context.

Two things make this domain attractive:
- **A crisp, automatic commission oracle.** Generated plans can be run in the VAL validator or a simulator.
- **Human chunk data.** Program-recall studies (normal vs scrambled code) replicate the chess paradigm, and the HVM variable-transfer task tests context-defined categories directly.

The main cost is that plans are **partial orders**: a chunk need not be contiguous in time. TRELLIS would therefore need causal or data-flow relations and argument co-reference (variable binding), not just "before."

### Cited Findings
**Hierarchical skill and plan learning**
- **Nejati, Langley & Könik (2006)**, "Learning hierarchical task networks by observation," ICML 2006, pp. 665–672 ([Crossref doi](https://doi.org/10.1145/1143844.1143928)). It learns HTNs from expert operator sequences. Unlike explanation-based learning, it acquires hierarchical structure and more general conditions, which allows transfer and recursive procedures. — [ML Anthology](https://mlanthology.org/icml/2006/nejati2006icml-learning)
- **Langley (2025)**, "Learning hierarchical task knowledge for planning," AAAI-25, pp. 28652–28656. It frames HTN learning as "identifying hierarchical structure, unifying method heads, and finding method conditions." — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35091); [ML Anthology](https://mlanthology.org/aaai/2025/langley2025aaai-learning)
- **ICARUS** lineage, bibliographic details verified only. Langley, Choi & Rogers (2009), "Acquisition of hierarchical reactive skills in a unified cognitive architecture," *Cognitive Systems Research* 10(4):316–332, DOI 10.1016/j.cogsys.2008.07.003. — [table of contents](https://mailman.srv.cs.cmu.edu/pipermail/connectionists/2009-October/023746.html)
- **HTN-MAKER** (Hogg, Muñoz-Avila & Kuter, AAAI 2008). It learns methods by regressing task effects through operator subsequences. The task definitions it is given are supervision, analogous to v1's gold trees. — [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/7571/7432)
- **Li, Kambhampati & Yoon (2009, IJCAI).** They learn probabilistic HTNs from user plans as **probabilistic grammar induction**, and the learned pHTNs generate plans distributed like users' preferred plans. This is the closest "plans as sentences of an induced grammar" baseline. — [ML Anthology](https://mlanthology.org/ijcai/2009/li2009ijcai-learning); [arXiv 1006.0274](https://arxiv.org/pdf/1006.0274)
- **Cobweb over plans.**
  - Yang & Fisher (1989) clustered means-ends plans.
  - Yoo & Fisher (1991) formed concepts over problem-solving traces, "which are themselves derivation trees much like the parses we learn here."
  - Source: [TRELLIS paper](https://arxiv.org/pdf/2609.30414)

**Chunking, macros, options**
- **Soar chunking** (Laird, Rosenbloom & Newell 1986, *Machine Learning* 1(1):11–46; [doi](https://doi.org/10.1023/a:1022639103969)). It acquires rules from goal-based problem solving, and its demonstrations include macro-operator acquisition. — [ML Anthology](https://mlanthology.org/mlj/1986/laird1986mlj-chunking)
  - Langley's essay notes that Soar's procedural "chunks" "bear little resemblance to Miller's original notion." — [Langley 2025](http://www.cogsys.org/proceedings/2025/paper-2025-3.pdf)
- **Macro-operators**, verified bibliographically. Per its abstract, Macro-FF covers macro discovery, filtering and ranking.
  - Korf (1985), "Macro-operators: A weak method for learning," *Artificial Intelligence* 26(1):35–77. — [doi](https://doi.org/10.1016/0004-3702(85)90012-8)
  - Iba (1989), "A heuristic approach to the discovery of macro-operators," *Machine Learning* 3(4):285–317. — [doi](https://doi.org/10.1023/a:1022693717366)
  - Botea et al. (2005), "Macro-FF," *JAIR* 24:581–621, DOI 10.1613/jair.1696. — [OpenAlex](https://api.openalex.org/works?filter=doi:10.1613/jair.1696&select=id,doi,title,abstract_inverted_index)
- **LOVE** (Jiang, Liu, Eysenbach, Kolter & Finn, NeurIPS 2022). Source for all points below: [arXiv PDF](https://arxiv.org/pdf/2212.04590)
  - **Problem:** maximum-likelihood skill learning is underspecified, because degenerate segmentations (one skill per trajectory, or one-step skills) reconstruct the data equally well.
  - **Fix:** minimize the code length of the skill-label sequence, subject to an ELBO constraint.
  - **Boundary agreement, Kipf grid world:**

    | Method | Commission side | Omission side |
    |---|---|---|
    | LOVE | 0.90 | 0.94 |
    | VTA | 0.19 | 0.99 |
    | DDO | 0.19 | 1.0 |

    VTA and DDO over-segment.
  - **Out of distribution:** only LOVE succeeded on 5-object tasks after training on 3-object tasks.
- **CompILE** (Kipf et al., ICML 2019). Unsupervised segmentation of demonstrations into latent-coded segments, which generalizes to longer sequences. — [arXiv 1812.01483](https://arxiv.org/abs/1812.01483)

**Program-library learning as chunking**
- **DreamCoder** (Ellis et al., PLDI 2021, pp. 835–850; [Crossref doi](https://doi.org/10.1145/3453483.3454080)). Wake-sleep learning alternately adds abstractions to the DSL and trains a neural guide. Extended version in *Phil. Trans. R. Soc. A* (2023). — [arXiv 2006.08381](https://arxiv.org/abs/2006.08381v1)
- **Stitch** (Bowers et al. 2023, *PACMPL* 7(POPL):1182–1213; [Crossref doi](https://doi.org/10.1145/3571234)). Corpus-guided top-down synthesis of abstractions; 3–4 orders of magnitude faster than DreamCoder's compressor, with equal or better compression. — [arXiv 2211.16605](https://arxiv.org/abs/2211.16605v2)
- **babble** (Cao et al., POPL 2023). Library learning modulo equational theories, using e-graph anti-unification. — [arXiv](https://arxiv.org/pdf/2212.04596)
- **LILO** (Grand et al., ICLR 2024). LLM-guided synthesis plus Stitch compression plus "AutoDoc" naming of abstractions. — [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/819cebb05f993840e8a52d7564c5c282-Abstract-Conference.html)
- **ShapeCoder** (Jones et al., *ACM TOG* 42(4), 2023). Abstraction discovery over shape programs. — [arXiv](https://arxiv.org/pdf/2305.05661)
- **Code idioms** (Allamanis & Sutton, FSE 2014, pp. 472–483; [Crossref doi](https://doi.org/10.1145/2635868.2635901)). Idioms are mined as fragments of **probabilistic tree-substitution grammars** with metavariables (slots). They capture object creation, exception handling and resource management. — [arXiv 1404.0417](https://arxiv.org/pdf/1404.0417)

**Human data**
- **Program recall (the chess paradigm for code).**
  - Shneiderman (1976): experts recall programs better than novices only when the lines are in order; the advantage disappears when the order is scrambled.
  - McKeithen, Reitman, Rueter & Hirtle (1981), *Cognitive Psychology* 13(3):307–325 ([Crossref doi](https://doi.org/10.1016/0010-0285(81)90012-8)): experts' recall reflects meaningful chunks, inferred as trees from recall orders.
  - Sources: [Détienne review](https://arxiv.org/pdf/cs/0702003); [Deep Blue](https://deepblue.lib.umich.edu/items/1d548813-cc8c-4d00-bc3b-ef1cbd11ed0d)
- **Concept shift with expertise.** Adelson (1981), *Memory & Cognition* 9(4):422–433 ([Crossref doi](https://doi.org/10.3758/bf03197568)): novices categorize code by syntax; experts by function or procedure. This is the same novice→expert shift Yeh et al. (2003) found for Chinese characters.
- **Programming plans as templates with slots and context.** Soloway & Ehrlich (1984), *IEEE TSE* SE-10(5):595–609 ([Crossref doi](https://doi.org/10.1109/tse.1984.5010283)).
  - Plans are frames with slot types, fillers and **context**. For example, Counter_Variable has init `:=0`, update `+1`, context iteration.
  - Slots have prototypical values, and experts give plan-like answers even to unplan-like programs: errors regress toward the prototype.
  - Source: [Détienne review](https://arxiv.org/pdf/cs/0702003); [Soloway PDF](https://research.cs.queensu.ca/home/cordy/cisc860/Biblio/hurd/misc/soloway84.pdf)
- **Hierarchical task decomposition.**
  - Solway et al. (2014), "Optimal Behavioral Hierarchy," *PLOS CB* 10(8):e1003779 ([Crossref doi](https://doi.org/10.1371/journal.pcbi.1003779)): people spontaneously discover optimal hierarchies.
  - Correa, Sanborn, Ho, Callaway, Daw & Griffiths (2025), "Exploring the hierarchical structure of human plans via program generation," *Cognition* 255:105990, doi 10.1016/j.cognition.2024.105990 ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22hierarchical%20structure%20of%20human%20plans%22&format=json&resultType=core)).
    - Participants wrote programs in a language with explicit hierarchy.
    - They were sensitive to both utility and MDL, but "people prefer programs with reuse over and above the predictions of MDL."
    - The authors' account extends MDL to a generative model, "modeling hierarchy choice as the induction of a grammar over actions."
  - Botvinick & Plaut (2004), *Psych. Review* 111(2):395–429 ([doi](https://doi.org/10.1037/0033-295x.111.2.395)), is the no-explicit-hierarchy recurrent-network foil.
- **Motor chunking.** Sakai, Kitaguchi & Hikosaka (2003), *Exp. Brain Research* 152(2):229–242 ([doi](https://doi.org/10.1007/s00221-003-1548-8)): chunks form spontaneously. Shuffles that preserve chunks spare performance; shuffles that break chunks hurt it. Chunks differ across people learning the same sequence.
- **Events have both taxonomies and partonomies.** Zacks & Tversky (2001), *Psych. Bulletin* 127(1):3–21: "Events belong to categories, and, like objects, events have parts." — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22event%20structure%20in%20perception%20and%20conception%22&format=json&resultType=core)
- **HCM and HVM, the closest chunk-plus-variable competitors** (Wu et al.).
  - **HCM** (NeurIPS 2022) learns chunks of chunks and parses greedily, taking the biggest consistent chunk. — [ML Anthology](https://mlanthology.org/neurips/2022/wu2022neurips-learning-a/)
  - **HVM** (ICLR 2025): source for all points below: [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e46984e056185b21ddb1e7973c365f14-Abstract-Conference.html); [arXiv HTML](https://arxiv.org/html/2410.21332v2)
    - It merges adjacent chunks when an independence test rejects.
    - It defines variables by context: "A variable denotes distinct observations appearing in the same context (here defined as distinct chunks sharing preceding and succeeding chunks)." That is, **a category defined by context**.
    - Human experiment: 112 participants learned colour sequences such as BXDF with X∈{A,C,E}.
    - HVM's sequence likelihood correlated with recall time at R=0.86 (training) and R=0.70 (transfer). LLMs did not show the human transfer pattern.
- **Language-of-thought compression.** Planton et al. (2021), *PLOS CB* 17(1):e1008598: memory for binary sequences tracks the shortest nested description. — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22mental%20compression%20algorithm%20in%20humans%22&format=json&resultType=core)

**Environments and oracles**

| Resource | What it offers | License and access |
|---|---|---|
| VAL plan validator | Commission oracle for plans | BSD-3-Clause since 2019 ([GitHub](https://github.com/KCL-Planning/VAL)) |
| Crafter | 22-achievement tech tree (diamond ← iron pickaxe ← furnace/table/...); human experts score 50.5±6.8 vs DreamerV2 10.0±1.2 | MIT ([repo](https://github.com/danijar/crafter); [paper](https://arxiv.org/pdf/2109.06780)) |
| BabyAI | Grid world with compositional instructions | BSD-3 ([repo](https://github.com/mila-iqia/babyai)) |
| The Stack | Permissively licensed code | Gated terms ([HF](https://huggingface.co/datasets/bigcode/the-stack)) |
| Breakfast (Kuehne, Arslan & Serre, CVPR 2014) | Activity videos with an explicit action grammar | Current hosting unverified ([OpenAlex](https://api.openalex.org/works?filter=doi:10.1109/cvpr.2014.105&select=id,doi,title,abstract_inverted_index)) |

### Inferences
- **Plans are the natural third domain for v2's "variable arity + relations beyond linear order".** The relations needed are:
  - precedence as a partial order;
  - causal or enabling links (an effect satisfies a precondition);
  - argument co-reference (variable binding);
  - for programs, AST parent–child edges labelled by slot.
- **Chunk-with-slots is the shared abstraction across the program literature.** A λ-abstraction, a TSG idiom with metavariables, or a Soloway plan with a context slot is a TRELLIS chunk whose slots are Cobweb concepts. Soloway's errors that regress toward the prototype are what a Cobweb slot distribution would predict when recall is under-determined.
- **HVM is the most direct published competitor to TRELLIS's dual hierarchy.** HVM's variables are exactly categories defined by context. TRELLIS could run the same 112-participant stimuli and should:
  - form the X∈{A,C,E} concept in its context taxonomy;
  - predict the variable-group transfer advantage.

  The difference to emphasize: HVM has no taxonomy over chunks and no generation evaluation.
- **Plans offer crisper scoring than any other domain.**
  - Omission = held-out valid plans that fail to get a complete parse.
  - Commission = generated decompositions that VAL rejects.
- **Plans do not suit a first v2 domain.** Human comparison data are thinner than in chess, and partial orders plus variable binding make the correspondence problem harder than in board positions.

### Gaps
- **ICARUS:** mechanism details (goal-indexed skills learned from impasses; separate concept and skill hierarchies) were not verified from primary text here. Some venue details for Choi & Langley (2005) and Langley & Choi (2006) are also unverified.
- **Data and licenses:** availability is unverified for:
  - the HVM human data and the HVM code license;
  - the Correa et al. (2025) data;
  - the LOVE code;
  - the Crafter human dataset license;
  - Breakfast's current host and license;
  - IPC domain licenses and HTN-track benchmarks.
- **HTNLearn:** the method details of Zhuo et al. (2014) were not verified.

## Q6. Compositional/OOD generalization benchmarks (SCAN, COGS, SLOG, CFQ, PCFG SET): which suit a Cobweb-based dual hierarchy, and what do symbolic or grammar-based learners achieve?

### Takeaway
Every classic compositional benchmark maps an input to a meaning or action sequence. TRELLIS has no semantics module, so only their **input languages** are directly usable. That is still valuable, because three of them come from explicit grammars, which give exact membership oracles for omission/commission scoring:
- COGS (a hand-written grammar);
- SLOG (a probabilistic synchronous CFG with a released generator);
- PCFG SET (a recursive PCFG).

The literature's clearest lesson matches TRELLIS's thesis. Grammar-inducing learners get length/depth productivity almost for free; Transformers, and even meta-learned MLC, score 0% on deeper recursion. Grammar inducers instead pay in **coverage**, which is an omission-side failure.

### Cited Findings
**Benchmarks**
- **SCAN** (Lake & Baroni, ICML 2018).
  - Over 20k command→action pairs.
  - Splits: simple, add-primitive, template, length and MCD.
  - A seq2seq RNN scores 99.7 / 1.7 / 2.5 / 13.8 on simple / jump / around-right / length.
  - The license file is a Facebook "BSD License."
  - Sources: [arXiv 1711.00350](https://arxiv.org/abs/1711.00350); [GitHub](https://github.com/brendenlake/SCAN); [Kim 2021, Table 1](https://arxiv.org/abs/2109.01135)
- **COGS** (Kim & Linzen, EMNLP 2020, pp. 9087–9105; [doi](https://doi.org/10.18653/v1/2020.emnlp-main.731)).
  - Sizes: 24,155 training, 3,000 dev, 3,000 test, and a generalization set of 21 cases × 1,000.
  - Recursion depth is 0–2 in training and 3–12 in the generalization set. 18 of the 21 cases are lexical; the 3 structural cases are PP recursion, CP recursion and obj-PP→subj-PP.
  - Generalization accuracy: Transformer 0.35, BiLSTM 0.16. License: MIT.
  - Sources: [PDF](https://aclanthology.org/2020.emnlp-main.731.pdf); [GitHub](https://github.com/najoungkim/COGS)
- **ReCOGS** (Wu, Manning & Potts, *TACL* 11:1719–1733, 2023; [doi](https://doi.org/10.1162/tacl_a_00623)). It shows COGS's negative results partly trace to incidental features of its logical forms. On original COGS, every model listed scores 0 on Obj-PP→Subj-PP. — [arXiv 2303.13716](https://arxiv.org/abs/2303.13716)
- **SLOG** (Li, Donatelli, Koller, Linzen, Yao & Kim, EMNLP 2023, pp. 3213–3232; [doi](https://doi.org/10.18653/v1/2023.emnlp-main.194)). It has 17 structural cases in four groups:
  - novel recursion depth (deeper PP, tail CP and center embedding; training depth ≤4, test 5–12);
  - modified phrases in new grammatical roles;
  - novel gap positions;
  - novel wh-questions.

  Results:
  - Best Transformer, including pretrained ones, 40.6%; AM-Parser 70.8%.
  - On deeper recursion beyond the training output length, **every Transformer (scratch, T5, LLaMA) scores 0.0 and the AM-Parser 100.0**.
  - The AM-Parser fails long-movement questions because it predicts only projective trees.
  - The PSCFG generator is released, under MIT.
  - Sources: [arXiv 2310.15040](https://arxiv.org/abs/2310.15040); [GitHub](https://github.com/bingzhilee/SLOG)
- **CFQ** (Keysers et al., ICLR 2020). 239,357 question–SPARQL pairs; MCD splits maximize compound divergence. LSTM+attention scores 97.4 on the random split vs 28.9 / 5.0 / 10.8 on MCD1–3. — [arXiv 1912.09713](https://arxiv.org/abs/1912.09713); [GitHub](https://github.com/google-research/google-research/tree/master/cfq)
- **PCFG SET** (Hupkes, Dankers, Mul & Bruni, *JAIR* 67:757–795, 2020; [Crossref doi](https://doi.org/10.1613/jair.1.11674)).
  - Inputs are string-edit function compositions (copy, reverse, shift, swap, repeat, echo, append, …) from a recursive PCFG. About 100k pairs.
  - Five tests: systematicity, productivity, substitutivity, localism and overgeneralisation.
  - Transformer scores: productivity 0.50, systematicity 0.72. License: MIT.
  - Sources: [arXiv 1908.08351](https://arxiv.org/pdf/1908.08351); [repo](https://github.com/i-machine-think/am-i-compositional)
- **gSCAN** (Ruis et al., NeurIPS 2020) is grounded SCAN in a 2D grid world: 367,933 training examples, MIT. — [arXiv 2003.05161](https://arxiv.org/abs/2003.05161)

**Symbolic and grammar-based learners**

| System | What it is | Results |
|---|---|---|
| Rule synthesis (Nye et al., NeurIPS 2020) | Synthesizes an explicit rule system | 100% on all four SCAN splits ([arXiv 2003.05562](https://arxiv.org/abs/2003.05562); [Kim 2021](https://arxiv.org/abs/2109.01135)) |
| NeSS (Chen et al. 2020) | Neural-symbolic stack machine | 100% on SCAN and on "CFG parsing tasks" ([arXiv 2008.06662](https://arxiv.org/abs/2008.06662)) |
| LANE (Liu et al. 2020) | Analytical-expression learner | 100% on SCAN ([arXiv 2006.10627](https://arxiv.org/abs/2006.10627)) |
| NQG (Shaw et al., ACL 2021) | Induced synchronous grammar with a codelength objective, described as "high-precision grammar-based" | 100 on SCAN jump / turn-left / length / MCD, but only 76.8 / 61.9 / 37.4 / 41.1 on GeoQuery standard / template / length / TMCD. NQG-T5 falls back to T5 when NQG gives no output ([arXiv 2010.12725](https://arxiv.org/abs/2010.12725)) |
| Neural QCFG (Kim, NeurIPS 2021) | Latent source and target trees | SCAN 96.9 / 96.8 / 98.7 / 95.7 ([arXiv 2109.01135](https://arxiv.org/abs/2109.01135)) |
| AM parser (Weißenhorn, Donatelli & Koller, *SEM 2022, pp. 44–54) | Compositional parser | COGS generalization 59.9 (78.3±22.9 with BERT and distance features). No seq2seq model exceeds 40% on structural types ([PDF](https://aclanthology.org/2022.starsem-1.4.pdf)) |
| CSL (Qiu et al., NAACL 2022) | Induced quasi-synchronous CFG | COGS 99.5. Grammar **coverage** of test inputs: SCAN 100%, COGS 99.9%, GeoQuery standard 76.3%, template 61.0% ([arXiv 2112.07610](https://arxiv.org/abs/2112.07610)) |
| Multiset tagging + permutations (Lindemann, Koller & Titov, ACL 2023) | No trees | COGS obj→subj PP 9±13 vs CP 79 vs PP 85 ([arXiv 2305.16954](https://arxiv.org/abs/2305.16954)) |

- **Caution from flat programs.** A RASP program reaches near-perfect COGS structural accuracy with flat pattern-matching. Success on COGS structural cases therefore does not prove a recursive grammar was learned. — [Bruns 2025, arXiv 2504.15349](https://arxiv.org/abs/2504.15349)

**Human data**
- **Lake, Linzen & Baroni (CogSci 2019).** 30 participants learned pseudoword→colour-sequence functions.
  - Accuracy: 76.0% on complex instructions; 72.5% on longer-than-studied items.
  - Error patterns: "one-to-one" made up 24.4% of errors; "iconic concatenation" 23.3% of third-function errors.
  - No seq2seq model exceeded 2.5%.
  - Source: [arXiv 1901.04587](https://arxiv.org/abs/1901.04587)
- **Lake & Baroni (2023)**, *Nature* 623:115–121 ([doi](https://doi.org/10.1038/s41586-023-06668-3)).
  - Humans matched the algebraic answer 80.7% of the time; MLC 82.4%.
  - In open-ended responses, 62.1% / 79.3% / 93.1% of people showed one-to-one / iconic concatenation / mutual exclusivity biases.
  - MLC has **100% error on the SCAN length split and on three COGS structural types**.
  - Source: [full text, Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10620072/fullTextXML)

### Inferences
- **Suitability for a semantics-free learner** that learns the input grammar, from gold unlabeled trees (re-derived from the released generators) or from raw strings:
  - **Free if the grammar is induced correctly.** COGS PP/CP recursion depth; SLOG deeper PP, tail CP and center embedding; PCFG SET productivity. These test recursion depth beyond training, the canonical "use the internal grammar out of distribution" claim.
  - **Free only if categories are position-independent.** COGS obj→subj PP; SLOG PP/RC on new grammatical roles; PCFG SET systematicity and substitutivity; SCAN around-right and MCD. These test whether TRELLIS's context taxonomy over-splits by position: one NP concept for subject and object, or two.
  - **Needs a non-adjacent co-indexing relation.** SLOG gap and long-movement cases. A TRELLIS without gap tracking should over-generate filled-gap strings, giving a ready-made commission probe.
  - **Not licensed by the input distribution.** SCAN add-jump ("jump" appears only alone) and SCAN length, which is defined on output length.
- **How to score omission and commission:**
  - **Omission** = the share of legal held-out strings (by depth and by case) that get no single spanning root, plus the share of gold constituents missed. CSL's "grammar coverage" is a published omission-side analogue.
  - **Commission** = the share of K generated strings rejected by an exact CKY/Earley recognizer for the generator grammar, plus the share of near-miss illegal strings the parser accepts.
  - Published COGS/SLOG numbers are logical-form exact match, so TRELLIS's input-side numbers must not be compared to them directly.
- **The Lake/Baroni few-shot task is a possible stretch human comparison.** Treat each study item as a joint word+colour experience and complete the colour part by Cobweb prediction. It needs a pairing (two-stream) representation that v1 lacks.

### Gaps
- No verified 2024–2026 GPT-4-class results on SLOG, COGS or ReCOGS were found.
- Whether the COGS repository ships its grammar file, and whether CFQ's generating grammar is released, were not checked.
- The MLC repository shows no license.

## Q7. Context-sensitive and mildly context-sensitive synthetic languages as a step beyond CFGs: suites, learners, human data, and what TRELLIS would need

### Takeaway
**FLaRe** (ICLR 2025, MIT) is the best off-the-shelf recognition suite. Its features:
- positive strings plus edit-perturbed negatives;
- length generalization from training strings of length 0–40 to test strings up to 500;
- tiers from regular to context-sensitive, including marked reversal (nested) vs marked copy (cross-serial).

The symbolic theory that fits TRELLIS best is **distributional learning**:
- Clark & Eyraud's substitutability for CFLs;
- Yoshinaka's **multidimensional substitutability** for multiple context-free languages (MCFLs).

TRELLIS's context taxonomy is, in effect, an incremental and probabilistic version of their substitutability classes. To move beyond CFGs, TRELLIS needs:
- **discontinuous chunks** (tuples of spans, fan-out ≥2);
- **multi-hole contexts**;
- a **co-indexing relation**.

Human artificial-grammar data show crossed dependencies are **not** harder than nested ones, and sometimes easier. A cognitively plausible TRELLIS should reproduce that.

### Cited Findings
- **Target languages and formal background.**
  - Shieber (1985), "Evidence against the context-freeness of natural language," *Linguistics and Philosophy* 8(3):333–343 ([Crossref doi](https://doi.org/10.1007/bf00630917)): cross-serial dependencies in Swiss German.
  - Joshi (1985) introduced tree adjoining grammars and mild context-sensitivity ([doi](https://doi.org/10.1017/cbo9780511597855.007)).
  - MIX is a 2-MCFL ([Salvati 2015, *JCSS* 81:1252–1277](https://doi.org/10.1016/j.jcss.2015.03.004)).
  - Öttl et al. frame nested dependencies as context-free and cross-serial ones as mildly context-sensitive ([PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4401728/fullTextXML)).
- **Delétang et al., "Neural Networks and the Chomsky Hierarchy"** (transduction tasks). Source for all points below: [arXiv 2207.02098](https://arxiv.org/abs/2207.02098); [code, Apache-2.0](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy)
  - 20,910 models on 15 tasks; training length ≤40, test lengths 41–500.
  - Context-sensitive tasks: Duplicate String, Missing Duplicate, Odds First, Binary Addition/Multiplication, Compute Sqrt, Bucket Sort.
  - Tape-RNN reaches 100 on the first four. Transformers sit near chance, e.g. 52.8 on Duplicate String.
- **FLaRe** (Butoi et al., ICLR 2025, "Training Neural Networks as Recognizers of Formal Languages"). Source for all points below: [arXiv 2411.07107](https://arxiv.org/abs/2411.07107); [flare](https://github.com/rycolab/flare)
  - **Languages by tier:**
    - Regular: Even Pairs, Repeat 01, Parity, Cycle Navigation, Modular Arithmetic, Dyck-(2,3), First.
    - Deterministic CF: Majority, Stack Manipulation, Marked Reversal.
    - CF: Unmarked Reversal.
    - CS: Marked Copy, Missing Duplicate, Odds First, Binary Addition/Multiplication, Compute Sqrt, Bucket Sort.
  - **Data:** 10k training strings of length 0–40; tests up to 500. Negatives are made by K random edits, with K drawn from a geometric distribution.
  - **Mean accuracy:**

    | Language | Transformer | RNN | LSTM |
    |---|---|---|---|
    | Marked Copy | 0.63 | 0.76 | 0.69 |
    | Marked Reversal | 0.64 | 0.70 | 0.74 |

- **Bhattamishra, Ahuja & Goyal (EMNLP 2020).** Counter languages (Dyck-1, Shuffle-Dyck, aⁿbⁿ, aⁿbⁿcⁿ) plus regular languages; Transformers handle a subclass of counter languages. — [arXiv 2009.11264](https://arxiv.org/abs/2009.11264)
- **MLRegTest.** 1,800 regular languages organized by logical complexity; useful only for tier-based non-adjacent dependencies. — [arXiv 2304.07687](https://arxiv.org/abs/2304.07687)
- **Mildly CS languages with neural models.** Transformers do fine in distribution but extrapolate worse than LSTMs. — [Wang & Steinert-Threlkeld 2023, arXiv 2309.00857](https://arxiv.org/abs/2309.00857)
- **Omphalos (correction to the brief).** The CFG-learning competition was organized by Starkie, Coste & van Zaanen (ICGI 2004; [doi](https://doi.org/10.1007/978-3-540-30195-0_3)). Clark described his winning learner in *Machine Learning* 66:93–110 ([doi](https://doi.org/10.1007/s10994-006-9592-9)).
- **Distributional learning.**
  - Clark & Eyraud formalized substitutability for polynomial identification in the limit from positive data. It is "not necessary to identify constituents… sufficient to identify the syntactic congruence." — [*JMLR* 8](https://www.jmlr.org/papers/v8/clark07a.html)
  - Clark, Eyraud & Habrard's contextual binary feature grammars model "the lattice structure of the distribution of a set of substrings" and cover some CS languages. — [*JMLR* 11](https://www.jmlr.org/papers/v11/clark10a.html)
  - Yoshinaka (2011), "Efficient learning of multiple context-free languages with multidimensional substitutability from positive data," *TCS* 412(19):1821–1831. — [Crossref doi](https://doi.org/10.1016/j.tcs.2010.12.058)
  - Clark & Yoshinaka extend this to parallel MCFGs ([*ML* 96:5–31](https://doi.org/10.1007/s10994-013-5403-2)) and survey the area ([2016 chapter](https://doi.org/10.1007/978-3-662-48395-4_6)).
- **MDL-RNNs** (Lan, Geyer, Chemla & Katzir, *TACL* 10:785–799, 2022; [Crossref doi](https://doi.org/10.1162/tacl_a_00489)). They learned aⁿbⁿ, aⁿbⁿcⁿ, aⁿbⁿcⁿdⁿ, aⁿb²ⁿ, aⁿbᵐcⁿ⁺ᵐ and Dyck-1, often perfectly, with proofs of correctness for any input. — [arXiv 2111.00600](https://arxiv.org/abs/2111.00600)
- **Discontinuous grammar induction on natural data.** Unsupervised probabilistic LCFRS-2 induction for German and Dutch had to drop O(n⁶) rules to get O(n⁵) parsing. — [Yang, Levy & Kim, ACL 2023, arXiv 2212.09140](https://arxiv.org/abs/2212.09140)
- **Human artificial-grammar data: crossed vs nested.**
  - **Bach, Brown & Marslen-Wilson (1986)**, *LCP* 1:249–262 ([doi](https://doi.org/10.1080/01690968608404677)). As summarized by Öttl et al., Dutch cross-serial sentences were judged more comprehensible than German nested ones.
  - **de Vries et al. (2012)**, *Phil. Trans. R. Soc. B* 367:2065–2076 ([Crossref doi](https://doi.org/10.1098/rstb.2011.0414); [PubMed](https://pubmed.ncbi.nlm.nih.gov/22688641/)). With three dependencies, participants struggled with the *middle* dependency in nested but not crossed sequences; there was no difference with two dependencies.
  - **Uddén, Ingvar, Hagoort & Petersson (2012)**, *Cognitive Science* 36(6):1078–1101 ([Crossref doi](https://doi.org/10.1111/j.1551-6709.2012.01235.x)). The push-down stack model was only "partly supported," and crossed dependencies held an advantage. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/22452530/)
  - **Öttl, Jäger & Kaup (2015)**, *PLOS ONE* 10(4):e0123059 ([Crossref doi](https://doi.org/10.1371/journal.pone.0123059)). Both dependency types were learned, with no difference. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/25885790/)

### Inferences
- **What v2 needs for MCFL-class languages:**
  - **Fan-out-2 chunks** whose content is (components + a linearization template). Example: a copy language rule `A(x1 a, x2 a) ← A(x1, x2)`.
  - **Contexts with three parts**, (left, middle, right), which is Yoshinaka's multidimensional substitutability.
  - **A co-indexing or "paired-with" relation.**
- **This same machinery serves the other v2 domains.**
  - SLOG filler-gap cases need it.
  - Chess pins and x-rays are discontinuous relational chunks: the pinned piece lies between attacker and target.
- **A greedy parser fits.** Greedy bottom-up merging restricted to fan-out ≤2 and well-nested combinations, gated by mature three-part context concepts, is consistent with the user's stay-greedy design. Its adequacy here is untested.
- **Recommended protocol.** Run FLaRe (positives only for training) and score:
  - **omission** = positive test strings rejected;
  - **commission** = edit-perturbed negatives accepted (by edit distance), plus illegal strings among 1,000 generated samples (checked by FLaRe's membership code).
- **Predicted results.**
  - v1-style CFG chunks handle the reversal and Dyck languages.
  - On Marked Copy they should either fail (omission) or fall back to a context-free over-approximation (commission). This makes it a crisp diagnostic for the discontinuous-chunk extension.
- **A cognitive-plausibility check.** Replicate the de Vries (2012) and Öttl (2015) crossed/nested designs. TRELLIS should not penalize crossed dependencies more than nested ones, and ideally should show the middle-dependency error under triple nesting.

### Gaps
- Bach et al.'s (1986) result is known here only through Öttl et al.'s summary.
- Kanazawa & Salvati's "MIX is not a TAL" was not found.
- The MDL-RNN repository was not checked.
- Whether a greedy TRELLIS parser can handle fan-out-2 chunks without a chart is an open design question; no source addresses it.

## Q8. Cross-domain comparison: which domains best satisfy the selection criteria?

### Takeaway
Weighing the six criteria gives this ranking:
1. **Chess positions.** The best fit to the literature, to human data, and to omission/commission scoring of reconstruction.
2. **Chinese characters via IDS.** The best discrete 2D domain with gold trees, plus human data on legal novel items.
3. **Action sequences and hierarchical plans.** HVM variable-transfer sequences, then HTN traces with a VAL commission oracle. This is the arena the TRELLIS paper itself names, after boards and scenes.

Two alternatives sit alongside the top three:
- **Music (Jazz Harmony Treebank)** is the cheapest non-language domain with gold trees. It is a strong alternative third choice, or a fast sanity domain.
- **The formal-language and structural-generalization suites** (FLaRe, SLOG/COGS/PCFG SET input side) are diagnostic harnesses rather than domains.

### Cited Findings
The ratings below rest on these facts, documented in the domain sections above.

**Chess**
- Human recall is scored as omission and commission errors. Published targets exist by skill, presentation time and position type. — [Gobet & Simon 2000](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)
- Data: about 8.2B CC0 games and about 6.2M CC0 motif-tagged puzzles. — [Lichess](https://database.lichess.org/)

**IDS characters**
- About 89k gold IDS trees under GPLv2. — [cjkvi-ids](https://github.com/cjkvi/cjkvi-ids)
- 4,864 structurally generated pseudocharacters with lexical-decision data. — [Wang et al. 2025](https://doi.org/10.3758/s13428-025-02701-7)
- Radical representations are position-specific. — [Taft et al. 1999](https://doi.org/10.1006/jmla.1998.2625)
- Sorting shifts from components to configuration with literacy. — [Yeh et al. 2003](https://doi.org/10.1080/13506280344000077)

**Plans and action sequences**
- Langley's three HTN subproblems: structure, method heads, conditions. — [Langley 2025 AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35091)
- VAL is BSD-3. — [VAL](https://github.com/KCL-Planning/VAL)
- HVM human recall-time fit: R=0.86 in training, 0.70 in transfer. — [Wu et al. 2025](https://arxiv.org/html/2410.21332v2)
- "people prefer programs with reuse over and above the predictions of MDL." — [Correa et al. 2025](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22hierarchical%20structure%20of%20human%20plans%22&format=json&resultType=core)

**Music**
- JHT: 150 binary trees (CC BY-NC-SA). — [JHT](https://github.com/DCMLab/JazzHarmonyTreebank)
- Unsupervised and supervised parse baselines exist. — [Cartuyvels et al. 2024](https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/11/Unsupervised-Induction-of-Harmonic-Syntax.pdf)
- Two-level human boundary data. — [Pearce et al. 2010](https://research.gold.ac.uk/id/eprint/5380/1/p6507.pdf)

**Orbán grids**
- Exact human familiarity patterns and a Bayesian chunk-learner competitor. — [Orbán et al. 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2268207)

**L-systems**
- Recursion-depth generalization with human classification and generation data. — [Lake & Piantadosi](https://doi.org/10.1007/s42113-019-00053-y)

**Benchmarks**
- SLOG deeper recursion: every Transformer scores 0.0, the AM-Parser 100.0. — [SLOG](https://arxiv.org/abs/2310.15040)
- FLaRe: positive and negative strings across Chomsky tiers. — [FLaRe](https://arxiv.org/abs/2411.07107)

**ARC**
- Rich human data, but no fixed grammar to score against. — [H-ARC](https://doi.org/10.1038/s41597-025-05687-1); [ARC-AGI-2](https://arxiv.org/abs/2505.11831)

### Inferences
Ratings are H/M/L; this is my synthesis.

| Domain | Hierarchy matters for parse | Concept AND chunk facets | Data & license | Human comparison data | Feasibility (discrete, incremental; correspondence problem) | Omission/commission measurable | Relations beyond "before" |
|---|---|---|---|---|---|---|---|
| **Chess positions** | H: POS → pawn chains, castled king → templates → position | H: template slots = roles filled by different pieces | H: CC0, billions of games | **H**: recall % correct, omission and commission by skill, time, random vs game | M: board coordinates give canonical order; relation graph is dense; no gold trees | H for reconstruction; M for grammar legality (status() is a necessary condition only) | H: attack, defend, adjacency, line, pin |
| **Chinese IDS characters** | H: depth 2–5, operator and position matter (呆/杏) | H: component variants (胡=⿰古月 vs ⿰古⺼) and slot classes | H: 89k trees, GPLv2 | H: lexical decision, pseudocharacters, position specificity, novice→expert | **H**: gold trees, canonical slot order, small trees | H: attestation and human pseudocharacters | H: 12+ spatial operators, n-ary |
| **Synthetic 2D And-Or (IDS-operator) grammars** | Configurable | Configurable | Unlimited (generated) | L | H | **Exact** | H |
| **Orbán / Fiser–Aslin grids** | L–M: pairs and triplets | M | Stimuli recreatable from papers | **H**: exact 2AFC patterns | H | M: familiarity of true chunks vs foils | M: grid offsets |
| **L-system figures** | H: recursion | M | Recreatable | M: classification 64.9%; generation | M: continuous angles | H: depth extrapolation | M |
| **Omniglot / BPL** | M: 3 levels | H | H: MIT | H: one-shot error, visual Turing tests | L: continuous strokes need quantizing | M | M: start/end/along |
| **Block towers (McCarthy 2021)** | H | M | M: no license detected | M | M | M | M |
| **ARC** | M | M | H: Apache-2.0 | H | L | **L**: no fixed grammar | M |
| **Music: JHT harmony** | H: average height 7.57 | H: T/S/D functions, substitutions | M: 150 trees, NC license | M: segmentation, Koelsch ERP | **H**: 1-D, v1 machinery applies | H for trees; M for generation | L–M: interval, metre |
| **Music: melodic segmentation** | M: 2 levels | M | M: Essen license unverified | H: Pearce 2010 boundaries | H | H: boundary precision/recall | L |
| **HVM variable sequences** | M | **H**: context-defined variables | M: stimuli from paper | **H**: 112 participants, recall time | H | M | L |
| **Hierarchical plans (HTN + VAL)** | H | H: method heads = concepts; methods = chunks | H: generated traces, BSD-3 validator | L–M: Correa 2025, Solway 2014 | L–M: partial orders and variable binding | **H**: VAL is an exact commission oracle | **H**: enables, co-reference |
| **Programs / code idioms** | H: ASTs | H: plans with slots and context | H: The Stack (gated) | M: program recall, Adelson, Soloway | M | M | H: slot-labelled AST edges, data flow |
| **COGS/SLOG/PCFG SET (input side)** | H: recursion depth | M | H: MIT | L for structure | H: v1 regime | **Exact** | L, except gap co-indexing |
| **CS languages (FLaRe, copy)** | H | L | H: MIT | M: crossed vs nested AGL | M: needs fan-out-2 chunks | **Exact** | H: co-indexing |

- **Chess** dominates on the criteria the user prioritizes: the literature, human comparison, concept + chunk, and non-linear relations. It is weakest on gold structure, so it depends on v2's unsupervised chunk formation.
- **IDS characters** dominate on feasibility while still adding n-ary spatial relations. That makes them the best de-risking step for the representation changes chess needs.
- **Plans** dominate on commission measurability and on Langley's own lineage, but they carry the hardest correspondence problem (variable binding, partial orders).
- **Music** is the lowest-risk non-language domain but adds the fewest new relations.
- **The benchmark suites** should be treated as harnesses for testing out-of-distribution use of the grammar and context-sensitive capacity, not as the "more than language" domain the user asked for.

### Gaps
- The H/M/L ratings are judgment calls synthesized from the sources above. No source compares these domains for a Cobweb-style learner.
- Chess's low rating on gold structure rests on a negative finding (see Q2 Gaps): no public trial-level human recall data, and no gold parse trees for positions, were found.
- Feasibility ratings assume relation types are precomputed symbolically (e.g., python-chess attack maps; IDS operators). They do not cover learning relations from pixels.

## Implications for TRELLIS v2

### Ranked recommendation
1. **Chess positions.** The flagship domain, and the user's stated priority.
2. **Chinese characters via Ideographic Description Sequences (IDS).** The 2D domain, piloted with synthetic IDS-operator grammars and an Orbán grid replication.
3. **Action sequences → hierarchical plans.** First the HVM variable-transfer task, then HTN traces scored with VAL. Alternative third choice: jazz harmony on the Jazz Harmony Treebank (JHT).
4. **Running alongside all of these: an out-of-distribution / context-sensitivity diagnostic harness.** It uses the input side of SLOG/COGS/PCFG SET plus FLaRe and copy languages.

**Suggested order of work, by risk rather than value:**
- **Stage 0: synthetic IDS-operator grammars.** These have gold trees, exact omission/commission, n-ary arity and typed spatial relations. They are the smallest step from v1.
- **Stage 1: real IDS characters.**
- **Stage 2: chess.** It needs the unsupervised chunk formation that v2 is building anyway (inside-outside + compression).
- **Stage 3: plans.** Partial orders and variable binding make this the hardest correspondence problem.

The rationale is the cross-domain comparison in Q8:
- Chess wins on literature, human data and relation richness, but has no gold trees.
- IDS wins on feasibility while still exercising the same representational changes chess needs.
- Plans have the cleanest commission oracle and are the arena the TRELLIS paper itself names.

### Representation changes shared by all three proposals
Everything in this subsection is my design inference from the sources cited in Q1–Q7.

**1. N-ary content with typed relations.** This replaces v1's binary LEFT/RIGHT content.
- A content instance holds:
  - the arity;
  - per-slot attributes (a context-concept pointer, a complexity/depth tag, and slot-specific attributes);
  - pairwise **relation attributes** between slots, such as `rel(s1,s2)=defends`.
- This extends v1's (child-id, complexity) encoding described in the TRELLIS paper. It is close to TRESTLE's relational tuples ([arXiv 2410.10588](https://arxiv.org/pdf/2410.10588)), but without flattening the whole hierarchy.

**2. Assigning constituents to slots (role assignment).**
- Use a **canonical slot order supplied by the domain**:
  - the IDS operator's slot order;
  - square order on the board, or roles defined by relations (attacker/target);
  - temporal or causal order for plans.
- This avoids general structure mapping.
- Fall back to TRESTLE-style greedy partial matching only where no canonical order exists.

**3. Context = relations that cross the chunk boundary**, each encoded as (relation type, neighbour's concept id, relative position or slot).
- **Chess:** CHREST's ±2-square visual field is a principled default window ([Gobet & Simon 2000](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)).
- **IDS:** parent operator, slot index and sibling.
- **Sequences:** preceding and succeeding chunks. These are exactly the sets HVM uses to define its variables ([Wu et al. 2025](https://arxiv.org/html/2410.21332v2)).

**4. Location coding.** Keep both an **absolute anchor** and **relative offsets** as content attributes, and let Cobweb's attribute distributions decide.
- A peaked anchor distribution gives a location-bound chunk.
- A diffuse one gives a location-general chunk.
- The human chess data favour location-specific chunks ([Gobet & Simon 1996, M&C](https://pubmed.ncbi.nlm.nih.gov/8757497/)), and so do the IDS data, where radical representations are position-specific ([Taft et al. 1999](https://doi.org/10.1006/jmla.1998.2625)).

**5. Candidate chunks restricted by relations**, to keep inside-outside parsing tractable.
- **Chess:** only connected subgraphs of the relation graph.
- **IDS:** groupings defined by the operators.
- **Plans:** contiguous or causally linked subsequences.
- Without this restriction, a 25-piece position has exponentially many candidate subsets.

**6. Chunk minimality.** Add a competitive-exclusion or compression step that prunes chunks subsumed by better ones.
- Zhu et al. (2008) found exclusion is "the main factor that causes the bottom-up process to stop" ([PDF](https://people.csail.mit.edu/leozhu/paper/usl_eccv_2008.pdf)).
- Humans do not find embedded sub-chunks familiar ([Orbán et al. 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2268207)).

**7. A later extension: chunks with fan-out 2 plus a co-indexing relation.**
- Chess pins and x-rays: the attacker, the pinned piece and the target are discontinuous in line order.
- Filler-gap structures (SLOG).
- Cross-serial and copy languages ([Yoshinaka 2011](https://doi.org/10.1016/j.tcs.2010.12.058)).

### Proposal 1: Chess positions (flagship)
**Why.**
- Both founding documents name it. Langley's essay talks of pieces "that threaten or defend each other, but different types of pieces might occupy the same role" ([Langley 2025](http://www.cogsys.org/proceedings/2025/paper-2025-3.pdf)). The TRELLIS paper says "Chess is the natural first target" ([arXiv 2609.30414](https://arxiv.org/pdf/2609.30414)).
- The human-memory literature already scores reconstruction with **errors of omission and commission**.
- The strongest symbolic competitor, CHREST, fits human omissions but **mis-fits commissions** ([Gobet & Simon 2000](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)). That leaves an opening for a probabilistic slot model.
- The data are CC0 at scale ([Lichess](https://database.lichess.org/)).

**Element** (a piece on a square; primitive):
`{type: R, color: W, file: f, rank: 1}`. Optionally add a side-relative rank, so White and Black chunks can share concepts.

**Relations** (directed and typed, computed from python-chess attack maps; [docs](https://python-chess.readthedocs.io/en/latest/core.html)):
- `attacks(Bb5→Nc6)`;
- `defends(Kg1→Pf2)` (attacks a square occupied by a same-colour piece);
- `adjacent(Kg1,Rf1)` (Chase & Simon's proximity);
- `same-line(Rf1,Pf2,file)`;
- later, with fan-out 2: `pins(Bb5; Nc6; Ke8)` and x-ray.

Same colour and same type, Chase & Simon's other two relations ([CS73](http://matt.colorado.edu/teaching/highcog/fall8/cs73.pdf)), come from comparing slot attributes.

**Content instance** (a castled-king chunk, before generalization):
```
{arity: 5, anchor: g1,
 s1: ctx#K, s1.sq: g1,  s2: ctx#P, s2.sq: f2,  s3: ctx#P, s3.sq: g2,
 s4: ctx#P, s4.sq: h2,  s5: ctx#R, s5.sq: f1,
 rel(s1,s2)=defends, rel(s1,s3)=defends, rel(s1,s4)=defends,
 rel(s1,s5)=adjacent, rel(s5,s2)=defends+same-line, cplx: 1}
```
- The Cobweb concept that covers many such instances is a **template** in Gobet & Simon's sense:
  - **core:** low-entropy attributes, here K on g1 and pawns on f2/g2;
  - **slots:** high-entropy attributes, e.g. P(s5 = R) = 0.7, P(s5 = Q) = 0.1, P(empty) = 0.2, or P(h-pawn on h2) vs P(h-pawn on h3);
  - **defaults:** the modal values.
- So "different pieces occupy the same role" becomes a filler distribution on a slot. Gobet & Simon's own template example likewise has square-slots (`g1:<white king>`) and piece-slots (`white rook:<e1>`) ([GS00](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)).

**Context instance** (for the same chunk; relations crossing its boundary within ±2 squares):
```
{in:  [defends(Nf3→h2), defends(Nf3→g1)],
 out: [defends(Rf1→Qd1)],                  # e1 empty
 opp: [attacks(Bg4→Nf3)],                  # later, with fan-out 2: pins(Bg4; Nf3; Qd1)
 neighbor-concepts: [ctx#knight-f3-defender, ctx#queen-d1],
 side-to-move: W, phase: middlegame}
```
In this hypothetical position, White has Kg1, Rf1, pawns f2/g2/h2, Nf3 and Qd1, and Black has a bishop on g4.
Context concepts would group chunks that play the same role in positions. One example is "kingside shelter under attack" vs "quiet shelter".

**Parse** (bottom-up and greedy, or inside-outside over relation-connected candidates):
- level 1: pieces on squares;
- level 2: local chunks such as pawn chains, the castled king, batteries, fianchettos;
- level 3: templates (opening-family structures);
- level 4: the position.

The human data say static same-colour structures dominate: 47 of the master's 77 chunks were pawn chains ([SC73](https://www.Gwern.net/doc/psychology/chess/1973-simon.pdf)). Tactical cross-colour chunks such as forks and pins are a later level.

**Generate (recall).** Hold about 3–7 chunk identifiers, matching the STM limits used by CHREST and human data ([Gobet & Clarkson 2004](https://pubmed.ncbi.nlm.nih.gov/15724362/)).
- Decompose them top-down.
- Fill slots with their modal values.
- Resolve conflicts with one piece per square and one king per side, as CHREST does.

**Evaluation**
- **E1. Recall vs humans** (the main result). Stimuli mirror Chase & Simon: Lichess middle-games near White's move 21 with about 24–26 pieces, endgames near move 41, and random placements of the same pieces ([CS73](http://matt.colorado.edu/teaching/highcog/fall8/cs73.pdf)). Proxies:
  - **Skill** = training-set size, analogous to CHREST's 500 / 10k / 300k-node nets.
  - **Presentation time** = the number of recognition cycles allowed.

  Score % correct, **omission** (pieces in stimulus − pieces placed) and **commission** (pieces placed wrongly), as in Gobet & Simon. Targets for game positions:

  | Measure | Masters | Experts | Class A |
  |---|---|---|---|
  | Omission | 0.6 | 5.8 | 9.5 |
  | Commission | 2.2 | 4.2 | 4.5 |

  Also report the random-position effect and the largest-chunk and number-of-chunks profiles ([GS00](https://bura.brunel.ac.uk/bitstream/2438/811/1/Five%20Seconds%20or%20Sixty%20.pdf)).
- **E2. Relational signature of chunk boundaries.** Within-chunk relation profiles should resemble Chase & Simon's (.89 copy–recall correlation), and between-chunk profiles should look random ([SC73](https://www.Gwern.net/doc/psychology/chess/1973-simon.pdf)).
- **E3. Concept validity.**
  - Purity or adjusted Rand index of frontier-cut concepts against Lichess puzzle themes (align them to the position after the first move).
  - Correlation between ease of recognition and puzzle Glicko ratings ([Lichess](https://database.lichess.org/)).
- **E4. Grammar-level omission and commission.**
  - **Omission** = held-out real positions that get no complete parse (no single root covering every piece).
  - **Commission** = generated positions that fail `Board.status()`. This is a necessary condition only, since "reachability is not guaranteed."
  - Baseline: the prototype-position result ("better than 44%").
- **E5. Out-of-distribution tests.**
  - Mirror-reflected and translated positions; humans drop ([M&C 1996](https://pubmed.ncbi.nlm.nih.gov/8757497/)).
  - Chess960; a 270M transformer drops 10–20% ([Lomasov et al. 2025](https://arxiv.org/abs/2510.26025)).
  - Recall of up to 5 boards ([Gobet & Simon 1996](https://doi.org/10.1006/cogp.1996.0011)).

**Risks**
- **No gold trees.** Bootstrap from proxy segmentations (connected components of relations, or CHREST-style chunks) only for diagnostics, never as training targets.
- **The relation graph is dense.**
- **Legality is global**, so generation needs a constraint-repair step.

### Proposal 2: Chinese characters via IDS (the 2D domain)
**Why.**
- These are real, discrete, recursive 2D grammars with **gold relation-labelled trees and no category labels**, which is v1's regime.
- About 89k entries; expanded depth is mostly 2–5 ([cjkvi-ids](https://github.com/cjkvi/cjkvi-ids)).
- Arrangement matters: 呆=⿱口木 and 杏=⿱木口 ([ids.txt](https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt)).
- There is human data on **legal novel items**: 4,864 pseudocharacters with lexical decisions ([Wang et al. 2025](https://doi.org/10.3758/s13428-025-02701-7)). This lets commission be compared against human acceptability.
- It is the 2D analogue of Langley's "letters/words with fonts and spellings" example. Regional variants such as 胡 = ⿰古月 [GJK] vs ⿰古⺼ [T] are literally different "spellings" of the same slot ([ids.txt](https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt)).

**Element:** a primitive component, `{id: 氵}` (optionally with stroke count).

**Relations:** the IDS operators as typed relations between ordered slots ([Unicode](https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt)):
- ⿰ left-of;
- ⿱ above;
- ⿲/⿳ ternary;
- ⿴ surround (國=⿴囗或);
- ⿵ surround-from-above (問=⿵門口);
- ⿻ overlay.

**Content instances:**
- 胡 → `{op: ⿰, arity: 2, s1: ctx#古, s1.cplx: 1, s2: ctx#月, s2.cplx: 0}`.
- 街 → `{op: ⿲, arity: 3, s1: ctx#彳, s2: ctx#圭, s3: ctx#亍}`.
- 古 → `{op: ⿱, arity: 2, s1: ctx#十, s2: ctx#口}`.

**Context instances:**
- 胡 inside 湖 (=⿰氵胡) → `{parent_op: ⿰, slot: 2/2, sibling(s1): ctx#氵}`.
- 古 inside 胡 → `{parent_op: ⿰, slot: 1/2, sibling(s2): ctx#月, grandparent_op: ⿰, grandparent_slot: 2/2}`.

**Concepts that should emerge:**
- **Context concepts** such as "components that fill the right slot of ⿰ next to 氵" and "left-slot radicals."
- **Content concepts** such as the ⿰氵X family, i.e. a template whose X slot is a filler distribution.
- **Variant classes** such as {月, ⺼}.

**Evaluation**
- **Omission:** the share of held-out real characters (including rare Extension-A+ characters) that parse to a single root from known components.
- **Commission:**
  - the share of generated novel characters with an unattested (component, operator, slot) triple, i.e. a radical in an illegal position;
  - a small rating study, or comparison with the SCLP pseudocharacter set, to check that legal-looking generations are judged character-like.
- **Concept tests:**
  - **Position specificity:** no transposition benefit ([Taft et al. 1999](https://doi.org/10.1006/jmla.1998.2625)).
  - **A shift from component-based to configuration-based similarity** as the content taxonomy matures, matching novices vs literate readers ([Yeh et al. 2003](https://doi.org/10.1080/13506280344000077)).
- **Out-of-distribution:**
  - zero-shot recognition of unseen characters assembled from known parts, the symbolic analogue of RAN ([doi](https://doi.org/10.1109/ICME.2018.8486456));
  - generalization to deeper nesting.

**Pilots**
- **(a) Synthetic grammars** using the same operators, scored exactly with Tu-style "generated samples accepted by the true grammar" (commission) and "true-grammar samples covered" (omission) ([Tu et al. 2013](https://papers.neurips.cc/paper_files/paper/2013/file/24681928425f5a9133504de568f5f6df-Paper.pdf)).
- **(b) An Orbán et al. replication:** 12 shapes on 3×3/5×5 grids, with grid-offset relations. Targets:
  - true combos familiar;
  - embedded sub-chunks at chance;
  - a fit of r≈0.88–0.92, the level the Bayesian chunk learner reaches ([PMC2268207](https://pmc.ncbi.nlm.nih.gov/articles/PMC2268207)).

**Risks**
- The IDS data are GPLv2, which is fine for research use.
- There is no exact "illegal" oracle; attestation statistics and human ratings stand in for one.
- Glyph appearance is abstracted away. A later step could ground primitives in Cobweb/4V image concepts ([arXiv 2402.16933](https://arxiv.org/abs/2402.16933)).

### Proposal 3: Action sequences → hierarchical plans
**Why.**
- The TRELLIS paper lists "plans" among the arenas that test generality.
- Langley's HTN-learning decomposition maps onto TRELLIS's three facets ([AAAI-25](https://ojs.aaai.org/index.php/AAAI/article/view/35091)):
  - structure → content;
  - method heads → concepts;
  - conditions → context.
- VAL gives an exact commission oracle ([VAL](https://github.com/KCL-Planning/VAL)).
- HVM is the closest published chunk-plus-variable competitor, with human data ([Wu et al. 2025](https://arxiv.org/html/2410.21332v2)).

**Stage A: the HVM variable-transfer sequences.** Stimuli such as `B X D F` with X∈{A,C,E}.
- **Element:** `{token: B}`.
- **Relation:** before, exactly as in v1.
- **Content:** `{arity: 3, s1: ctx#B, s2: ctx#{A,C,E}, s3: ctx#D}`.
- **Context:** `{prev: ctx#…, next: ctx#F}`.
- **Evaluation:**
  - parse cost against human recall time (targets R=0.86 in training and 0.70 in transfer);
  - the transfer advantage of the variable group;
  - baselines HVM, HCM and LZ78.

**Stage B: HTN traces** (Blocksworld/Logistics plans from a planner).
- **Element:** a grounded action, `{op: pickup, arg1: C, arg1.type: block}`.
- **Relations:**
  - `precedes(a1,a2)`;
  - `enables(a1→a2)`: an effect of a1 (`holding(C)`) satisfies a precondition of a2 (`stack(C,D)`);
  - `coref(a1.arg1, a2.arg1)`.
- **Content** (method "put C on D"): `{arity: 2, s1: ctx#pickup|unstack, s2: ctx#stack, rel(s1,s2)=enables, bind: s1.arg1=s2.arg1}`.
- **Context:** `{prev: ctx#clear-D chunk, next: ctx#…, state: {clear(D), handempty}}`. Method conditions live in context.
- **Evaluation:**
  - **Omission:** held-out valid plans that get no complete parse.
  - **Commission:** generated decompositions or plans that VAL rejects.
  - **Out-of-distribution:** more blocks and taller towers (cf. LOVE's 3→5-object transfer; [arXiv 2212.04590](https://arxiv.org/pdf/2212.04590)).
  - **Baselines:** HTN-MAKER, Nejati et al. (2006), and the pHTN grammar induction of Li et al. (2009).
  - **Human comparisons:** the reuse bias in Correa et al. (2025), and Solway et al. (2014).

**Alternative third choice: jazz harmony on the JHT** ([GitHub](https://github.com/DCMLab/JazzHarmonyTreebank)).
- **Element:** a chord with its root relative to the key.
- **Relations:** before, root interval, metrical strength.
- **Content:** (head, dependent, relation). For example, a preparation by a falling fifth.
- **Context:** neighbouring chords and metrical position.
- **Evaluation:**
  - span omission and commission against the gold trees, with combined-score baselines 0.39–0.48 (unsupervised) and 0.62 (supervised) ([Cartuyvels et al.](https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/11/Unsupervised-Induction-of-Harmonic-Syntax.pdf));
  - out-of-distribution: transposition and held-out tritone substitutions.

### Running alongside: the out-of-distribution / context-sensitivity harness
- **SLOG/COGS/PCFG SET input side.** Re-derive gold trees from the released generators and produce per-depth curves (training depth ≤4, testing 5–12, for SLOG). Use position-independence tests for categories, and filled-gap minimal pairs as commission probes ([SLOG](https://arxiv.org/abs/2310.15040)).
- **FLaRe.** Train on positives only.
  - **Omission** = positives rejected.
  - **Commission** = edit-perturbed negatives accepted, plus illegal generations.
  - Marked Copy diagnoses whether fan-out-2 chunks are needed ([FLaRe](https://arxiv.org/abs/2411.07107)).
- **Crossed vs nested artificial-grammar simulation.** Crossed dependencies should not be penalized relative to nested ones ([de Vries et al. 2012](https://doi.org/10.1098/rstb.2011.0414); [Öttl et al. 2015](https://doi.org/10.1371/journal.pone.0123059)).

### One evaluation template for every domain

| Domain | Parse / recognition **omission** | Generation **commission** | Human comparison | Out-of-distribution split |
|---|---|---|---|---|
| Chess | Held-out positions without a complete parse; recall omissions | Recall commissions; generated positions failing `status()` | Gobet & Simon 1996/2000 recall tables; Chase & Simon chunk statistics | Mirror and translated positions; Chess960; multiple boards |
| IDS characters | Held-out (rare) characters not parsed to one root | Novel characters with unattested slot/operator use; human-rated non-characters | SCLP pseudocharacters; Taft 1999; Yeh 2003 | Unseen characters from known parts; deeper nesting |
| Plans | Valid held-out plans without a parse | Generated plans rejected by VAL | HVM recall times; Correa 2025 | More objects; deeper recursion |
| Harness | Positive strings rejected; gold constituents missed | Negatives accepted; generations rejected by the reference grammar | Crossed/nested artificial-grammar results | Depth 5–12; length 41–500 |

### Decisions to codify before starting
- **Gold vs induced structure, per domain.**
  - IDS and synthetic grammars: start with gold unlabeled trees, as in v1, then remove them.
  - Chess: requires v2's unsupervised chunk formation from the outset.
- **Role assignment:** canonical domain order by default; TRESTLE-style matching only as a fallback ([TRESTLE](https://arxiv.org/pdf/2410.10588)).
- **Relation inventory:** fixed and symbolic per domain (python-chess attack maps; IDS operators; STRIPS preconditions and effects). Learning relations themselves is out of scope for v2.
- **Licensing:**
  - Lichess data is CC0.
  - python-chess is GPL-3.0+.
  - IDS data is GPLv2.
  - JHT is CC BY-NC-SA 4.0.
  - FLaRe, SLOG and COGS are MIT.
  - VAL is BSD-3.

  All allow academic research. Check redistribution of any derived datasets against GPL, NC and ND terms.
