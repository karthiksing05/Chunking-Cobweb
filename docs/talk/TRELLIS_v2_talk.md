# TRELLIS v2 talk: slide content

Same format as TRELLIS-ACS-2026-talk-v5 (Drive id 1tSl8rTk01KTo7LToUqgKuzigc44wVbGW): academic talk, one message per slide, story titles, speaker notes on every slide, slide number bottom right, backup slides at the end.
Framing throughout: **chunking is a method of compression**. Each chunk lets memory refer to many parts with one symbol; each concept lets one description cover many instances. Good compression and good generation are two readings of one model.
"[nod: ...]" marks a slide that answers a question the author raised; the nod is in the content, not printed as a label.

---

## 1. Title
**TRELLIS v2: Chunking as Compression**
Subtitle: Learning concepts and chunks from raw text and chess, one experience at a time
Karthik Singaravadivelan and Pat Langley
Georgia Institute of Technology, Atlanta, Georgia · Institute for the Study of Learning and Expertise, Palo Alto, California
Footer: Follows Singaravadivelan and Langley (2026), "A Unified Account of Concepts and Chunks," Advances in Cognitive Systems.
Notes: The first TRELLIS unified concepts and chunks and learned toy grammars from sample parse trees. This talk covers version 2, which learns from raw tokens with no parse trees, parses and generates by inside-outside over its memory, and runs on children's stories and chess. The thread through the talk is compression: chunking is how a limited memory holds more.

## 2. Limited memory forces compression, and chunks are how people do it
Left: "Short-term memory holds about 7 items (Miller, 1956)" with a row of 7 boxes holding single digits.
Right: "Chess masters recall a whole board after 5 seconds; on random boards their advantage disappears (Chase & Simon, 1973)."
Bottom line, bold: A chunk is one symbol for many parts. Expertise is a better code, not a bigger memory.
Notes: Miller's span applies to chunks, not raw items. Chase and Simon showed masters perceive familiar configurations as single units, so the same seven slots hold far more pieces. That makes chunking a compression scheme: it trades a stored dictionary of familiar patterns for shorter descriptions of new experience.

## 3. Two operations compress experience: composition and generalization
Table (no fill header):
| | Composition (chunks) | Generalization (concepts) |
| What it saves | One symbol for parts that occur together | One description for instances that behave alike |
| Language example | "once upon a time" stored as one unit | dog, cat, bird can all fill "the ___ ran" |
| Chess example | castled king with pawn shield: f2 g1 g2 h2 | a pawn chain on any file |
| Alone it fails because | it cannot say what a new chunk resembles | it has only single tokens to group |
Bottom: TRELLIS makes each operation the other's vocabulary.
Notes: A chunk's parts are named by concepts of where those parts occur; a concept's description includes what an element is made of. A better category sharpens the chunks built from it, and the reverse.

## 4. A good compressor is also a good generator
Diagram: x (context) -> general concept -> specialized by x -> y (new output). Caption: y = G(x).
Bullets:
- A memory that compresses well gives high probability to what it has not seen but could have.
- A general concept covers many instances; the context narrows it to one output.
- Grammatical novelty comes from that step: combinations no single instance had.
Notes: This is why we judge TRELLIS by generation as well as by bits. If chunking is compression, then sampling from the code should produce new but well-formed experience.

## 5. Version 2 removes the supervision and the greedy steps of version 1
Table:
| | TRELLIS v1 (ACS 2026) | TRELLIS v2 |
| Input | sentences with sample parse trees | raw token stream only |
| Parsing | greedy, bottom up, recognition threshold | inside-outside over every span and split |
| Generation | top-down expansion with recall thresholds | one word at a time, from what the whole sentence allows |
| Learning | committed chunks only | every candidate chunk, weighted by its posterior; no offline phase |
| Domains | synthetic context-free grammars | toy grammars, TinyStories, chess |
Notes: These answer the open questions from the v1 talk: a less greedy parser, chunks without sample parse trees, natural corpora and chess. The two linked Cobweb taxonomies stay.

## 6. Two linked Cobweb trees still hold everything TRELLIS knows
Shape diagram: left box "Context tree: where an element occurs" (groups dog/cat/bird); right box "Content tree: what an element is made of" (stores [the + animal], [animal + action]). Arrows: "names parts" (context -> content), "names wholes" (content -> context). Caption under right box: "the dog" and "a cat" share a chunk though they share no word.
Notes: The context tree takes every word and composite; the content tree takes composites. A composite's left and right parts are named by context concepts, so the content tree never sees a raw word. The names keep changing as the trees restructure.

## 7. Parsing sums over every bracketing instead of committing to one
Left: chart diagram of spans over "the fish was not happy anymore" (triangle of cells).
Right bullets:
- Inside pass: probability of each span from its sub-spans.
- Outside pass: probability of everything around a span.
- Output: the bracketing with the most expected correct chunks.
Illustrative bracketing (label it "illustrative"): ((the fish) ((was not) (happy anymore)))
Notes: Version 1 committed the best pair at each step and could not recover. Inside-outside keeps every analysis alive and weights it, which is also what learning needs: every candidate chunk is stored with its posterior.

## 8. Generation writes one word at a time from what the whole sentence allows
Four-step flow, left to right: 1 Read the grammar from memory -> 2 Read the prompt -> 3 Choose the length -> 4 Fill each word from all ways the rest could finish.
Caption: The outside score of a position sums over every chunk that could surround it, so each word sees chunk context in both directions.
Notes: A pair gate keeps word pairs that memory has seen side by side. Without it grammaticality collapses; we come back to this on the abstraction slide.

## 9. Learning is one pass: every candidate chunk is stored, weighted by how likely it is
Bullets:
- No labels, no parse trees, no offline phase; gold labels only score.
- Each composite enters both trees with weight equal to its posterior.
- Memory used to parse sentence n is the memory built from sentences 1 to n-1.
Notes: Learning stays incremental, as in Cobweb. There is no separate chunk-creation routine; parsing proposes chunks and learning treats them as data.

## 10. Section divider: Does chunking compress, and what does the code produce?

## 11. We test on toy grammars, TinyStories and chess against matched baselines
Table:
| Domain | Data | Measures | Baselines |
| Toy grammars | 3 to 8 nonterminals | grammatical and new generations | grammar known exactly |
| TinyStories | 125 to 4,000 sentences | judged grammatical and new; parse success | n-gram, LSTM, GPT sized for the data |
| Chess | 300 to 27,459 positions from Lichess | bits per position; Chase & Simon recall | BPE chunkers, trigram |
Footnote: 8 seeds for generation, 4 for parsing; paired 95% intervals.
Notes: Neural baselines are trained on exactly the same sentences and sized by held-out loss, then sampled at tuned temperatures.

## 12. The judge separates syntax from meaning  [nod: the judge must tell syntactic from semantic grammaticality]
Table (share passing, held-out half, Qwen3-8B + phi-4 ensemble):
| Control sentence | Example | Syntax pass | Meaning pass |
| Real | the fish was not happy anymore | 100% | 95% |
| Content words swapped | the hard door looked hard | 88% | 13% |
| Two adjacent words swapped | little the bird looked sad | 13% | n/a |
| Agreement error | they is not friends anymore | 4% | n/a |
| Scrambled | not anymore was happy the fish | 2% | n/a |
Notes: We calibrated two thresholds on one half of the controls and report the other half. Content swaps keep their syntax and lose their meaning, so the judge can tell a syntax-only regime from a meaningful one.

## 13. With little data, TRELLIS writes as well as matched neural models
Table:
| Sentences learned | vs LSTM | vs GPT |
| 250 | level | TRELLIS ahead, 8 of 8 seeds (+.035) |
| 500 | level | TRELLIS ahead, 8 of 8 seeds (+.024) |
| 1,000 | level | level |
| 4,000 | LSTM ahead | GPT ahead |
Footnote: Measure: share of generations judged grammatical and new. "Level" = paired 95% interval includes 0 over 8 seeds. No model reaches 95% grammatical.
Notes: TRELLIS uses no gradients. Its advantage is at small data, where a compressed memory of chunks generalizes before a neural model has enough examples.

## 14. Every learner pays for novelty with grammaticality; neural models pay less  [nod: novelty vs abstraction trade-off]
Content: description of Figure S4: x = share of new sentences with an unstored trigram; y = judged grammatical. Lines for TRELLIS (windows 3 to 7), LSTM, GPT, n-gram.
Bullets:
- From novelty .3 to .7, LSTM and GPT lie above every TRELLIS line.
- Among novel generations, TRELLIS is grammatical .50 to .63 of the time, LSTM .80 to .93 (250 to 1,000 sentences).
- Shorter context windows copy more; longer windows generalize more and err more.
Notes: This is the cost of a coarser code. A more general concept produces more new combinations, and some of them are wrong. The trade-off curve is where the right level of abstraction has to be chosen.

## 15. Parse trees alone check syntax; the generator's probabilities check meaning  [nod: parsing evaluation curve, grammatical vs ungrammatical parse success]
Table (share accepted at 4,000 sentences; threshold at the 10th percentile of real sentences):
| Score | Content swapped accepted |
| TRELLIS parse only | 91% |
| TRELLIS generator | 58% |
| LSTM | 41% |
| GPT | 40% |
| 5-gram | 39% |
Caption under table: Real sentences are accepted 91-93% by every score. The parse-only score also accepts 77% of scrambled sentences.
Bullets:
- Across 125 to 4,000 sentences, TRELLIS keeps up with the baselines at 125 to 500 on scrambles, swaps and agreement, then falls behind.
- It is weakest on meaning at every size.
Notes: The parse score accepts content swaps as readily as real sentences because their syntax is intact, which is the behaviour of a syntax-only parser. Adding the generator's conditionals inside chunks gives a small gain on swaps and agreement and changes the trees: F1 against gold rises from .32-.36 to .34-.38, still below right-branching (.42).

## 16. Chunks fall out of raw text, and most parses are not right-branching  [nod: chunk consistency; non-right-branching examples]
Left: a parse from the model that is neither right- nor left-branching (r216, n = 4,000): (little (((the bird) looked) sad)). Draw it as a small tree from shapes.
Right bullets:
- 82% of parses are not right-branching; 66% are neither right- nor left-branching.
- Top chunks are less consistent across seeds than raw frequency: top-50 Jaccard .33, against .51 for frequency and .43 for BPE.
Notes: TRELLIS does not fall back on the English right-branching default. Its chunks vary more across seeds than frequency counts, which is the honest cost of learning structure one sentence at a time.

## 17. In chess, the chunks are the ones players name
Bullets with mono notation:
- Most common chunk: castled king with pawn shield, f2 g1 g2 h2.
- Pawn chains are 35% of the top chunks; the b2-c3 shape recurs on 16 different squares.
- Batteries and castled kings appear; attacks do not, as Chase and Simon found.
Notes: Nothing in TRELLIS knows chess rules. These chunks fall out because the same configurations recur across games.

## 18. Chunks compress chess positions when familiar shapes move to new squares
Table (bits per held-out position, 4,000 learned; lower is better):
| Coder | Control positions | Shifted positions |
| TRELLIS | 127.7 | 147.2 |
| BPE over squares | 108.8 | 205.5 |
| BPE over shapes | 111.3 | 149.7 |
Notes: On ordinary positions co-occurrence chunkers code better. When shapes appear on squares they never occupied in training, TRELLIS's chunks, named by relations rather than absolute squares, transfer and save about 58 bits a position against square-based BPE.

## 19. Recall grows with experience as Chase and Simon found
Table (pieces recalled, middle game):
| | Beginner | Club player | Master |
| Chase & Simon (people) | 4 | 8 | 16 |
| TRELLIS, 7 chunks, after 300 / 1,000 / 4,000 positions | 7.2 | 9.7 | 12.9 |
Bullets:
- With 7 chunks, recall reaches the master's 16 pieces at about 8,000 positions and levels off near 17.
- Random-board recall stays flat, so the gain is specific to real positions.
- Recall is capped by chunks times pieces per chunk: better compression is what raises it.
Notes: With 4 chunks (the exact protocol), beginner and random-board recall match the human numbers with nothing fitted; the master level would need chunks of about 4 pieces.

## 20. Can one dial move TRELLIS between meaning, syntax-only and nonsense?  [nod: level of abstraction; three regimes; context length]
Table:
| Setting | Expected output | Example | Reached |
| Exact words | grammatical and meaningful | he liked to play with his ball | yes, 63% |
| Word classes | grammatical, meaningless | the hard door looked hard | not yet, at most 26% |
| No classes or no pair gate | word salad | give it is shiny | yes, over 90% |
Bullets:
- Shortening the context window to 2 raises meaningful output to 72%, but by copying.
- With 240 word types, a swap within a class still makes sense, and the pair gate restores word-level pairings.
Notes: We tried emitting through word classes, abstracting only content words, and classes built from function-word neighbours. None reached the middle regime. A run on 4,000 sentences, with a larger vocabulary, and a class-level gate are in progress. In chess, recognition does not encode legality, so a legal-but-pointless regime does not appear either.

## 21. Related and future research
Table:
| Framework | Relation |
| Cobweb, LABYRINTH | no context taxonomy |
| EPAM, CHREST | top-down parsing; chunks without concepts |
| Compression-based grammar induction (Wolff, 1982) | same principle, fixed code |
| Neural language models | strong at scale, no explicit chunks |
Open questions:
- Can memory choose the level of abstraction itself, instead of a hand-set cut?
- Can TRELLIS keep up with neural models past 4,000 sentences?
- Can a chess generator compose positions from chunks?

## 22. Concluding remarks
Bullets:
- Chunking is compression: one symbol for many parts, and expertise is a better code.
- TRELLIS v2 learns chunks and concepts from raw text and chess with no labels and no gradients.
- With little data it writes as well as matched LSTM and GPT models, and its chess memory behaves like Chase and Simon's experts.
- Meaning lags syntax, and choosing the right level of abstraction remains open.
Visual: 7 primitive slots ("the big dog chased the black cat") vs 3 chunk slots, as in v1's last slide.

## 23. References
Chase, W. G., & Simon, H. A. (1973). Perception in chess. Cognitive Psychology, 4, 55-81.
Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? arXiv:2305.07759.
Fisher, D. H. (1987). Knowledge acquisition via incremental conceptual clustering. Machine Learning, 2, 139-172.
Gobet, F., & Lane, P. C. R. (2012). CHREST. Encyclopedia of the sciences of learning, 556-559.
Langley, P. (2025). Concepts and chunks in cognitive systems. Advances in Cognitive Systems, 11, 33-44.
Miller, G. A. (1956). The magical number seven, plus or minus two. Psychological Review, 63, 81-97.
Singaravadivelan, K., & Langley, P. (2026). A unified account of concepts and chunks. Advances in Cognitive Systems.
Thompson, K., & Langley, P. (1991). Concept formation in structured domains.
Wolff, J. G. (1982). Language acquisition, data compression, and generalization. Language and Communication.

## 24. Backup: what we tried for the syntax-only regime
| Attempt | Best syntax-only share |
| Emit through word classes | .249 |
| Keep frequent words exact, abstract the rest | .255 |
| Classes from function-word neighbours | .264 |
| Gate at class level; vocabulary at 4,000 | in progress |

## 25. Backup: parsing evaluation along the learning curve
Bullets: AUC grammatical vs each broken class, n = 125 to 4,000, 4 seeds; TRELLIS generator vs 5-gram, LSTM, GPT. At n = 125 scramble AUC .886 (5-gram .880, LSTM .807, GPT .888). At n = 4,000 content-swap AUC .81 vs .88-.89.
