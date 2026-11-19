# TRELLIS v2 talk: slide content

Deck: https://docs.google.com/presentation/d/161jduQT2dR5sZmeBV9EgKDq_Xm7nIUGlsjTDBlHKb2A. Style follows TRELLIS-ACS-2026-talk-v5 (Drive id 1tSl8rTk01KTo7LToUqgKuzigc44wVbGW): short title-case titles, plain declarative bullets, examples and diagrams over curves, speaker notes on every slide, slide number bottom right, backups at the end.
Framing: chunking is compression (one symbol for many parts; expertise is a better code). Results are mostly case studies drawn from real model output; only slides 12, 14, 16, 21 (and the table on 22) are metric tables.
Drawn natively from real data: trees (slides 7, 15, 17, 18) from experiments/v3/results/story_parse_eval/r216_parse_fix.md and story_gen/r213/cases.txt; concepts and chunk history (6, 19) from r213_chunks.md and r213/cases.txt; generations (13) from story_gen/r212-n1000-w7 and r212-n1000-refs, seed 29; boards (20, 22) from docs/site/frag_chess.html and chess_cs1973_exact/cs73x_s13.json.

## 1. TRELLIS v2: Chunking as Compression
- Learning concepts and chunks from raw text and chess, one experience at a time
- Karthik Singaravadivelan and Pat Langley
- Georgia Institute of Technology, Atlanta, Georgia · Institute for the Study of Learning and Expertise, Palo Alto, California
- Follows Singaravadivelan and Langley (2026), "A Unified Account of Concepts and Chunks," Advances in Cognitive Systems.
Notes: At ACS we presented TRELLIS as a theory that unifies concepts and chunks, and we tested it on small synthetic grammars where it learned from sample parse trees. This talk covers version 2. It learns from raw tokens with no parse trees, it parses and generates by summing over every analysis its memory allows, and it runs on children's stories and on chess. The thread through the talk is compression: a chunk lets memory refer to many parts with one symbol, and a concept lets one description cover many instances.

## 2. Some Empirical Phenomena
- Limited Memory Span
- People hold about seven items at once (Miller, 1956).
- 7 digits in 7 slots
- Mitigated by Expertise
- Chess masters recall a whole board after five seconds; on random boards the advantage disappears (Chase & Simon, 1973).
- 26 pieces in 7 chunks
- A chunk is one symbol for many parts, so chunking compresses what memory must hold. Expertise is a better code, not a bigger memory.
Notes: Two phenomena motivate the talk. People have a limited ability to recall presented items, about seven digits or words, which is the left row of slots. Expertise in a domain like chess effectively expands that span, because experts perceive familiar configurations as chunks (Chase & Simon, 1973). The right row shows one real middle-game position that our model reads into seven chunks that together hold 26 pieces; we draw it on a board later. Miller's span applies to chunks, not raw items, so we can see chunking as a compression scheme: it spends a dictionary of familiar patterns to shorten the description of new experience.

## 3. Concepts and Chunks as Compression
Table:  | Chunks (composition) | Concepts (generalization) / What it saves | One symbol for parts that occur together | One description for instances that behave alike / Language | "one day" stored as one unit | dog, cat and bird can all fill "the ___ ran" / Chess | castled king and pawn shield: f2 g1 g2 h2 | a pawn chain on any file / Alone it fails because | it cannot say what a new chunk resembles | it has only single tokens to group
- These are two ways of saving symbols. TRELLIS uses both, and each one names the other's parts.
Notes: Researchers have treated concepts and chunks as distinct, with concepts emphasizing generalization and chunks emphasizing composition. Read as compression they are two ways of saving symbols. A chunk saves symbols by naming parts that occur together; a concept saves description by grouping instances that behave alike. Neither works alone: a system with chunks and no concepts cannot say what a new chunk resembles, and a system with concepts and no chunks has only single tokens to group. In TRELLIS a chunk's parts are named by concepts, and a concept's description includes what an element is made of.

## 4. Compression and Generation
- context x
- general concept
- narrowed by x
- new output y
- A memory that compresses well gives high probability to experiences it has not seen but could have seen. / A general concept covers many instances, and the context narrows it to one output. / Grammatical novelty comes from that step, which recombines parts no single instance had.
Notes: We judge TRELLIS by what it generates as well as by bits, because the two readings go together. If chunking is compression, then sampling from the code should produce new but well-formed experience. A general concept covers many instances, and the context in which it is used narrows it to a single output; that narrowing is where grammatical novelty comes from. It is also where errors come from, and the novelty results later in the talk measure that cost.

## 5. What Changes in Version 2
Table:  | TRELLIS v1 (ACS 2026) | TRELLIS v2 / Input | sentences with sample parse trees | raw token stream only / Parsing | greedy, bottom up, recognition threshold | inside-outside over every span and split / Generation | top-down expansion with recall thresholds | one word at a time, from what the whole sentence allows / Learning | committed chunks only | every candidate chunk, weighted by its posterior / Domains | synthetic context-free grammars | toy grammars, TinyStories, chess
- The two linked Cobweb taxonomies stay as they were.
Notes: These changes answer the open questions at the end of the first talk: can a less greedy parser handle the choices it gets wrong, can we learn chunks without sample parse trees, and can the theory handle natural corpora and chess. Version 2 reads a raw token stream, so gold labels are used only for scoring. Parsing no longer commits to one pair at a time, and learning no longer keeps only the chunks the parser committed to. The representation and organization postulates are unchanged.

## 6. Representation and Organization
- Context taxonomy / Groups elements that occur in like surroundings, words and chunks alike.
- names parts
- names wholes
- Content taxonomy: one concept
- concept
- One content concept after 1,000 TinyStories sentences (seed 13).
- Both taxonomies are Cobweb trees. A composite's two parts are named by context concepts, so the content taxonomy never sees a raw word.
Notes: Long-term memory holds two Cobweb taxonomies. The context taxonomy takes every word and every composite and groups them by the surroundings in which they occur. The content taxonomy takes composites only and groups them by what they are made of. They are linked: a composite's left and right parts are named by context concepts, so the content taxonomy never sees a raw word. The concept on the right is a real one from a model trained on 1,000 children's sentences. It groups "he wanted" with "she wanted", "tom wanted" and "he liked", so one description covers four chunks that no single sentence contained together.

## 7. Parsing in TRELLIS v2
- A parse of a held-out sentence, seed 13, after 4,000 sentences. Each internal node is a chunk.
- Parsing sums over every bracketing of the sentence instead of committing to one pair at a time. / The inside pass scores each span from its sub-spans, and the outside pass scores everything around it. / The tree shown is the bracketing with the most expected correct chunks.
Notes: In version 1 parsing was greedy: it proposed candidate chunks over adjacent pairs, admitted those that cleared a recognition threshold, and committed the best one. A wrong early choice could not be undone. Version 2 sums over every bracketing with an inside pass and an outside pass over the spans of the sentence, and then reports the bracketing with the most expected correct chunks. This is a real parse of the held-out sentence "the fish was not happy anymore" from a model trained on 4,000 sentences. We do not claim it is the linguist's parse; the evaluation later compares such trees with a shallow-phrase reference and finds them rather different. The same quantities that parse a sentence are what learning uses, since every candidate chunk is stored with its posterior.

## 8. Generation in TRELLIS v2
- 1  Read the grammar from memory
- 2  Read the prompt
- 3  Choose the length
- 4  Fill each word from all ways the rest could finish
- The outside score of a position sums over every chunk that could surround it, so each word sees chunk context in both directions. A pair gate keeps word pairs that memory has seen side by side.
- Prompt: tim was very happy
- TRELLIS wrote: they wanted to play near the water
Notes: Generation in version 2 writes one word at a time, but each word is chosen using a score for what the whole sentence could still become. The outside score of a position sums over every chunk that could surround it, which is how a word sees chunk context on both sides. A pair gate keeps word pairs that the memory has seen side by side; without it grammaticality collapses, a point we return to when we discuss how abstract the words can be. The example is a real sentence written by a model trained on 1,000 sentences from the prompt shown, and the judge scored it as grammatical.

## 9. Learning in TRELLIS v2
- Learning needs no labels, no parse trees and no offline phase; gold labels only score. / Each composite enters both taxonomies with a weight equal to its posterior. / The memory that parses sentence n is the memory built from sentences 1 to n-1.
- Sentence n arrives
- Parse it with the memory so far
- Store every candidate chunk, weighted by its posterior
Notes: Learning stays incremental, as in Cobweb. There is no separate routine for creating chunks: parsing proposes candidates and learning treats them as data for updating the two taxonomies. What changes in version 2 is that every candidate chunk is stored, weighted by how probable the parse says it is, instead of only the chunks a greedy parser committed to. Learning is interleaved with performance, so the memory that parses sentence n is exactly the memory built from the sentences before it.

## 10. Experimental Design
- Language needs both generalization and composition, so we test on children's stories and compare with models trained on the same sentences:
- Training sentences / 125 to 4,000
- Real held-out sentences, broken in controlled ways
- TRELLIS learns one sentence at a time
- TRELLIS parse or generator score
- New sentences it writes
- Accepted or rejected
- Judge: grammatical? new?
- Compare with n-gram, LSTM, GPT
- Baselines train on exactly the same sentences and are sized by held-out loss. Eight seeds for generation, four for parsing.
Notes: TinyStories sentences of three to eight words from a 250-word vocabulary make the language task. Each model sees n training sentences, with n from 125 to 4,000, and the held-out stories are never used for training. The top row is the generation protocol: the model writes new sentences and a language model judges whether each is grammatical and whether it is new, meaning not copied from the training set. The bottom row is the parsing protocol: we take real held-out sentences, break them in controlled ways, and ask whether a score accepts them. The neural baselines are an LSTM and a small GPT trained on exactly the same sentences, sized by held-out loss and sampled at tuned temperatures, plus n-gram models. We use eight seeds for generation and four for parsing, and we report paired differences.

## 11. Judging Syntax and Meaning
- Two language models, Qwen3-8B and phi-4, each answer two questions about a sentence. Thresholds were set on one half of the controls and the table reports the other half.
Table: Control sentence | Example | Syntax pass | Meaning pass / Real | the fish was not happy anymore | 100% | 95% / Content words swapped | the hard door looked hard | 88% | 13% / Two adjacent words swapped | little the bird looked sad | 13% | 6% / Agreement error | they is not friends anymore | 4% | 15% / Scrambled | not anymore was happy the fish | 2% | 0%
- A content swap keeps its syntax and loses its meaning, so the judge can tell the two apart.
Notes: We needed a judge that separates syntax from meaning, because TRELLIS may be grammatical without being sensible, and the reverse. We built control sets from held-out sentences: real sentences; sentences with every content word swapped for another of the same part of speech, which are grammatical but nonsense; sentences with two adjacent words swapped; agreement errors; and scrambles. Each of two models answers a syntax question and a meaning question, and the ensemble's thresholds were set on one half of the controls. On the other half it passes 88 percent of content swaps on syntax and only 13 percent on meaning, while scrambles and agreement errors fail both. One caveat is that a few of the agreement errors are actually grammatical, such as plural uses of fish.

## 12. Experimental Results: Generation
- Share of generations judged grammatical and new, TRELLIS against matched models:
Table: Sentences learned | vs LSTM | vs GPT / 250 | level | TRELLIS ahead, 8 of 8 seeds (+.035) / 500 | level | TRELLIS ahead, 8 of 8 seeds (+.024) / 1,000 | level | level / 4,000 | LSTM ahead | GPT ahead
- With little data TRELLIS writes as well as matched neural models. As the data grow they pull ahead.
- Level means the paired 95% interval over 8 seeds includes 0. No model reaches 95% grammatical.
Notes: The first result is on generation. The measure is the share of generations that the judge accepts as grammatical and that are new. At 250 and 500 sentences TRELLIS is level with the LSTM and ahead of the GPT on all eight seeds; at 1,000 sentences it is level with both; at 4,000 sentences both neural models are ahead. Level means that the paired 95 percent interval over the eight seeds includes zero. TRELLIS uses no gradients, and its advantage lies where a compressed memory of chunks can generalize before a neural model has enough examples. No model reaches 95 percent grammatical on this judge. The next slide shows what these generations look like.

## 13. What the Three Models Wrote
- Each model learned the same 1,000 sentences. The first two draws per prompt, with the judge's probability that the sentence is grammatical. All three write new sentences, and all three also write broken ones.
Table: Prompt | TRELLIS | LSTM | GPT / tom said sorry cat | it had a dog named spot (1.00) | the bird saw a big dog named max (1.00) | sara says good tim (0.00) /  | can play with it (0.04) | she is the best of you (0.01) | they were very happy (1.00) / he told his friends | tom and max were all day long (0.32) | mom said yes we can help (0.68) | tim and learned but sam was scared (0.03) /  | sue thought it and a big fish (0.05) | tim was happy and said thank you tim (0.00) | they are very are friends (0.00) / tim was very happy | it liked to help people (0.85) | max was not happy (1.00) | they were very happy (1.00) /  | let's go get your food together (0.99) | the bird felt sad (1.00) | tom and tim are friends (1.00)
Notes: Here are real generations from seed 29 after 1,000 training sentences, each model sampled at the temperature we tuned for it. I took the first two draws for each of three prompts in the order they were sampled, so these are not the best ones. The number in parentheses is the judge's probability that the sentence is grammatical. All three models write new sentences with similar fluency, and all three also produce word salad now and then, such as "sara says good tim" and "they are very are friends." Over the whole run the share judged grammatical is about 0.59 for TRELLIS, 0.58 for the LSTM and 0.61 for the GPT, which is the level result in the table.

## 14. Novelty and Grammaticality
- Models compared at about the same share of generations with a trigram unseen in training (1,000 sentences, 4 seeds). Real sentences are judged grammatical .89 of the time.
Table: Model | New trigram | Grammatical / TRELLIS, window 3 | .34 | .76 / LSTM | .32 | .92 / TRELLIS, window 4 | .52 | .71 / LSTM | .52 | .81 / TRELLIS, window 5 | .57 | .67 / GPT | .57 | .80
- Every learner pays for novelty with grammaticality. / Neural models pay less at this size. / Shorter context windows copy more, and longer windows generalize more and err more.
Notes: This is the novelty against abstraction trade-off. A model that copies the training set is grammatical but not new; a model that recombines parts more freely produces more new sentences and more errors. To compare fairly we look at models at the same level of novelty, measured as the share of generations that contain a trigram not stored in training. At about a third novel, TRELLIS with a three-word window is grammatical .76 of the time and the LSTM .92. At about half, the LSTM and GPT are at .81 and .80 while TRELLIS with windows of four and five words is at .71 and .67. Shorter windows copy more and longer windows generalize more and make more errors. This is the price of a coarser code, and it shows where the level of abstraction has to be chosen.

## 15. Parsing Real and Broken Sentences
- We broke real held-out sentences and asked whether each score accepts them. Thresholds were fixed on real sentences alone (seed 13, 4,000 sentences learned).
- Real
- Parse score: accepted
- Generator score: accepted
- Two words swapped
- Parse score: rejected
- Generator score: rejected
- Content words swapped
- Parse score: accepted
- Generator score: rejected
- The parse accepts the content swap, as a syntax checker should. The generator's word probabilities reject it.
Notes: This is the parsing protocol made concrete. We take a real held-out sentence and break it: swap two adjacent words, or swap content words for others of the same part of speech so the sentence stays grammatical but stops making sense. The trees are the parses TRELLIS chose after 4,000 sentences. We fix an operating threshold on real sentences only and ask whether the parse score and the generator score accept each sentence. All three trees look equally reasonable, and the parse score accepts the real sentence and the content swap while rejecting the adjacent swap. The generator's word probabilities reject the content swap too, because the words do not belong together. So the parse alone behaves like a syntax checker, and the meaning information lives in the generator.

## 16. Parse Success by Training Size
- Share of sentences each score accepts, as training sentences grow (4 seeds). Lower is better on broken sentences.
Table: Broken sentences | Score | 125 | 500 | 1,000 | 4,000 / Scrambled | TRELLIS parse | .88 | .81 | .78 | .77 /  | TRELLIS generator | .39 | .19 | .13 | .05 /  | LSTM / GPT | .65 / .40 | .16 / .09 | .08 / .04 | .01 / .01 / Content swapped | TRELLIS parse | .94 | .93 | .92 | .91 /  | TRELLIS generator | .86 | .79 | .71 | .58 /  | LSTM / GPT | .80 / .80 | .66 / .70 | .62 / .64 | .41 / .40
- Real sentences are accepted .90 to .94 by every score. Threshold fixed on real sentences.
Notes: This is the parse success curve along the learning curve. For each training size we fix the threshold on grammatical sentences and report how many broken sentences each score still accepts, so lower is better here. The parse score alone accepts most scrambles at every size, which is why we added the generator's conditional probabilities. The generator keeps up with the n-gram, LSTM and GPT baselines at 125 to 500 sentences on scrambles, adjacent swaps and agreement errors, then falls behind. The content swaps are where it is weakest: at 4,000 sentences it still accepts 58 percent against about 40 percent for the neural models, and the parse alone accepts 91 percent. In short, TRELLIS is better at telling syntax apart than at telling meaning apart.

## 17. The Same Sentence Under Four Seeds
- seed 13
- the
- tree
- had
- no
- friends
- seed 17
- the
- tree
- had
- no
- friends
- seed 19
- the
- tree
- had
- no
- friends
- seed 23
- the
- tree
- had
- no
- friends
- Four learners, each trained on its own 1,000 sentences, parse the same held-out sentence.
- Phrase reference: [the tree] [had [no friends]]
- Only the chunk (the tree) is bracketed by all four seeds.
Notes: This is the chunk consistency question. Four learners, each trained on its own thousand sentences, parse the same held-out sentence, "the tree had no friends." The trees differ: seed 13 groups "had no", seed 17 attaches the subject to the verb first, seed 19 happens to match the phrase reference, and seed 23 builds a left-leaning tree. What all four agree on is the chunk "the tree," a determiner and a noun. Across the held-out set, independent learners agree at bracket F1 of about .65 to .72, where random binary trees give .34, so the structure is real but not identical from seed to seed.

## 18. Parses That Are Not Right-Branching
- one day tom went to the park
- one
- day
- tom
- went
- to
- the
- park
- All four seeds give this tree. Reference: [one day] [[tom went] [to [the park]]]
- i will help you said tom
- i
- will
- help
- you
- said
- tom
- 82% of held-out parses differ from right-branching, and 66% are neither right- nor left-branching. / Where they differ they score .28 to .32 F1 against the phrase reference, below right-branching (.45 to .48). / The lower tree is a mistake: it joins "you said tom" across two clauses.
Notes: TRELLIS does not fall back on the English right-branching default. After 1,000 sentences, 82 percent of the held-out parses differ from right-branching and 66 percent are neither right- nor left-branching. Where TRELLIS departs from right-branching it scores .28 to .32 F1 against a shallow-phrase reference, below right-branching on the same sentences, so we do not claim the departures are better; the reference prefers right-branching trees. The top tree, "one day tom went to the park," is the same in all four seeds. It treats "one day" and "the park" as chunks, as the reference does, but attaches the subject to the verb before the prepositional phrase. The bottom tree is a failure we want to show: the sentence has two clauses, "i will help you" and "said tom," and the tree joins "you said tom." The two examples come from a block we drew at random from the non-right-branching parses, not from the best ones.

## 19. Chunks Reused Across Contexts
- The chunk "the bird" (seed 13)
Table: Sentences seen | 100 | 250 | 500 | 750 | 1,000 / Times bracketed | 2 | 5 | 8 | 13 | 14
- First sentences where it was a chunk: / the bird saw a big cat / the big cat walked away with the bird / they all laughed and played with the bird
- The concept it sits under changes
- after 250 sentences / all day; had fun; very happy; their toy
- after 500 / he was; he liked; yes you; they played
- after 1,000 / happy too; the animals; was inside; the tree
- 14 chunks are in the top 50 of every seed. Seeds overlap on the top 50 at .33, below raw frequency (.51) and BPE (.43).
Notes: A chunk is reused across contexts, and its concept keeps changing as the taxonomies restructure. On the left is the chunk "the bird" in one seed: it first appeared as a chunk at sentence 27 and was bracketed 14 times by 1,000 sentences. Its first sentences show it as a subject and as the end of a prepositional phrase. On the right are some of the other chunks that share its concept at three checkpoints; the group is different each time because the tree is rebuilt as data arrive, so what a concept contains depends on how much has been seen. Across seeds, 14 chunks are in the top 50 of all four, including "one day," "he wanted" and "the dog." The honest cost is that the overlap of the top 50 between seeds is .33 at 1,000 sentences, lower than a raw frequency count (.51) or BPE (.43), so in language chunking adds no cross-seed agreement beyond frequency.

## 20. Chess Chunks on the Board
- white king g1 with pawns f2 g2 h2
- black rooks a8 f8, knight b8, pawn a7
- black king g8 with pawns f7 g6 h7
- white pawns b2 c3 d4, knight f3
- black pawns b7 c6 d5, bishop e6
- white queen c2, bishop d3, rooks e1 e2
- black queen d6, knight d7
- 7 chunks hold 26 of 27 pieces.
- Capitals are white, lower case black. A held-out position, seed 13, 4,000 positions learned.
Notes: The same machinery runs on chess. Nothing in TRELLIS knows the rules of chess; a position is a set of pieces with their relations, and chunks are groups of pieces with relative offsets, capped at four pieces here. This is the first held-out position as the parser reads it after 4,000 positions: seven chunks hold 26 of its 27 pieces. The chunks are ones players would name: a castled king with its pawn shield, a pawn chain, a fianchetto, two rooks that defend each other. The most common four-piece chunk in memory is the castled king with pawns f2, g2 and h2, stored 861 times. Long-range and mixed-colour chunks also occur, and we have not measured how many are meaningful.

## 21. Compressing Shifted Chess Positions
- Bits to code a held-out position after 4,000 positions learned (lower is better). Shifted positions put familiar shapes on squares never seen in training.
Table: Coder | Control positions | Shifted positions / TRELLIS, squares in the description | 127.7 | - / TRELLIS, offsets only | - | 147.2 / BPE over squares | 108.8 | 205.5 / BPE over shapes | 111.3 | 149.7
- Co-occurrence chunkers code ordinary positions better. TRELLIS's chunks transfer when the squares change.
- The two TRELLIS settings are separate runs, each best on one set. Older recipe.
Notes: This is where chess tests the laws of compression and where they transfer. We code each held-out position with the learned chunks and count the bits, comparing with byte-pair chunkers over squares and over shapes. On ordinary positions the BPE coders are better. When the same shapes appear on squares they never occupied in training, BPE over squares degrades to 205.5 bits and TRELLIS's offset-only chunks stay at 147.2, a saving of about 58 bits a position, because TRELLIS names chunks by relations between pieces rather than by absolute squares. Two caveats. The two TRELLIS figures come from two runs of the older recipe, each best on one of the sets, so there is no single TRELLIS setting that wins both columns. BPE over shapes, which also abstracts away from squares, is within about 2 bits of TRELLIS on the shifted set.

## 22. A Chase and Simon Recall Trial
- Seven chunks held after the first look at a 26-piece middle-game position. The red square was recalled twice.
- Pieces correct after trials 1 to 4: 12, 18, 25, 26 of 26
Table: Pieces recalled | Beginner | Club | Master / People | 4 | 8 | 16 / TRELLIS, 7 chunks | 7.2 | 9.7 | 12.9
- TRELLIS after 300, 1,000 and 4,000 positions. With 7 chunks recall reaches 16 at about 8,000 positions and levels off near 17. Recall on random boards stays flat.
Notes: Chase and Simon showed people a board for five seconds and asked them to rebuild it. Masters recalled far more than beginners, but only on real positions, because they see familiar groups of pieces. We give TRELLIS a short-term memory of seven chunks and run the same task. The board shows one real position of 26 pieces and the seven chunks the model held after its first look. Twelve of its pieces were right; the red square was recalled twice, as a pawn and as a bishop, so one of the two is a commission error. Over repeated trials it reaches 12, 18, 25 and 26 correct. Across positions its recall rises with experience, from 7.2 pieces after 300 positions to 12.9 after 4,000, the same ordering as people at 4, 8 and 16, and it reaches 16 at about 8,000 positions. Random-board recall stays flat, so the gain is specific to real positions. Recall is capped by the number of chunks times the pieces per chunk, which is the compression argument again: a better code is what raises it.

## 23. Meaning, Syntax Only, and Nonsense
- We varied how abstract the words are in the generator's context, and hoped to get three kinds of output.
Table: Setting | Expected output | Example | Reached / Exact words | grammatical, meaningful | he liked to play with his ball | yes, 63% / Word classes | grammatical, meaningless | the hard door looked hard | not yet, at most 26% / No classes or no pair gate | word salad | give it is shiny | yes, over 90%
- With about 240 word types in 1,000 sentences, nearly any swap within a class still makes sense. / In chess, recognition does not encode legality, so a legal-but-poor regime does not appear either.
Notes: The question is whether one setting, the abstraction of the words in the generator's context, moves TRELLIS between three regimes: grammatical and meaningful, grammatical but meaningless, and word salad. The first and last are easy: exact words give meaningful output about 63 percent of the time, and removing the classes or the pair gate gives word salad over 90 percent of the time. The middle regime we did not reach; the best share of syntax-only output was 26 percent. The reason seems to be that with about 240 word types in 1,000 children's sentences, nearly any swap within a class still makes sense, and TRELLIS's classes group words by meaning as well as by grammar. We also varied the context window, which moves output between copying and recombining but not between these regimes. Chess shows no middle regime either: of the legal moves, the share that lose material stays near the .33 of a random legal move. The backup slides give the attempts.

## 24. Related and Future Research
- Other frameworks address parts of the problem:
Table: Framework | Relation to our theory / Cobweb variants: LABYRINTH (Thompson & Langley, 1991) | lacks a context taxonomy / EPAM and CHREST (Feigenbaum, 1961; Gobet & Lane, 2012) | parse from the top down; chunks without concepts / Compression-based grammar induction (Wolff, 1982) | same principle, with a fixed code / Neural language models | strong at scale, no explicit chunks
- Open questions for future work:
- Can memory choose the level of abstraction itself? / Can TRELLIS keep up with neural models past 4,000 sentences? / Can a chess generator compose positions from chunks?
Notes: Other frameworks address parts of the problem. Labyrinth encoded both primitive and composite concepts but lacked a context taxonomy and suffered substantial order effects. EPAM and CHREST parse structures from the top down, and treat chunks without concepts. Grammar induction systems that minimize description length, such as Wolff's, follow the same compression principle with a fixed code instead of a learned memory. Neural language models are strong at scale but create no explicit chunks. For future work, we would like memory to choose its own level of abstraction rather than rely on a hand-set cut, we need TRELLIS to keep up with neural models past 4,000 sentences, and we want a chess generator that composes positions from chunks.

## 25. Concluding Remarks
- TRELLIS v2 treats chunking as compression:
- One symbol for many parts, with expertise as a better code / Chunks and concepts learned from raw text and chess, with no labels and no gradients / Level with matched LSTM and GPT models on little data, and chess recall that follows Chase and Simon / Meaning lags syntax, and the right level of abstraction remains open
- 7 slots
- 3 slots
Notes: Our account treats chunking as a compression device. Once a memory has introduced a unit for some configuration it refers to it with a single symbol, so the context it has to carry shrinks from a list of primitives to a handful of chunks, as in the bottom row. Version 2 learns chunks and concepts from raw text and chess without labels or gradients. With little data it writes as well as matched LSTM and GPT models, and its chess memory behaves like the experts of Chase and Simon. It falls behind neural models as data grow, and it is weaker on meaning than on syntax. We do not claim that it competes with such systems at their scale, but a memory that builds and refers to its own units offers a better way to handle context than retaining the whole past.

## 26. References
- Chase, W. G., & Simon, H. A. (1973). Perception in chess. Cognitive Psychology, 4, 55-81. / Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? arXiv:2305.07759. / Fisher, D. H. (1987). Knowledge acquisition via incremental conceptual clustering. Machine Learning, 2, 139-172. / Gobet, F., & Lane, P. C. R. (2012). CHREST. Encyclopedia of the sciences of learning, 556-559. / Langley, P. (2025). Concepts and chunks in cognitive systems. Advances in Cognitive Systems, 11, 33-44. / Miller, G. A. (1956). The magical number seven, plus or minus two. Psychological Review, 63, 81-97. / Singaravadivelan, K., & Langley, P. (2026). A unified account of concepts and chunks. Advances in Cognitive Systems. / Thompson, K., & Langley, P. (1991). Concept formation in structured domains. / Wolff, J. G. (1982). Language acquisition, data compression, and generalization. Language and Communication.
Notes: References cited in the talk, including the backup slides.

## 27. Backup: Why the Middle Regime Failed
- We looked for a level of the learned taxonomy where slots are syntactic: pure (purity at least .8) and still varied (at least 3 bits of filler entropy).
Table: Level in the tree | Slot purity | Filler entropy / Leaf | .8 and above | 0.03 bits / 8 bits | .39 to .40 | 4 bits or more
- No level passes both tests, on seeds 13 and 17 with 1,000 sentences. The generation run that would have used such a level was not launched.
Notes: Backup. To produce grammatical but meaningless output we looked for a band of the learned content tree whose slots are syntactic, meaning they are pure in part of speech and still have varied fillers. The stopping rule required purity of at least .8 and filler entropy of at least 3 bits. No band meets it on either seed: purity reaches .8 only at the leaf, where the filler entropy is 0.03 bits, and at the band where entropy is still at least 4 bits purity is .39 to .40. So reading the tree at a coarser level does not pass through a syntactic-slot regime, and we did not run the generation arms.

## 28. Backup: Parsing Along the Learning Curve
- AUC for real against each broken class, 125 to 4,000 sentences, 4 seeds, TRELLIS generator against 5-gram, LSTM and GPT. / At 125 sentences scramble AUC is .886 (5-gram .880, LSTM .807, GPT .888). / At 4,000 sentences content-swap AUC is .81 against .88 to .89.
Notes: Backup. AUC compares grammatical sentences against each broken class along the learning curve over four seeds. At 125 sentences the generator scores scrambles at .886 against .880, .807 and .888 for the 5-gram, LSTM and GPT; at 4,000 sentences its content-swap AUC is .81 against .88 to .89.

## 29. Backup: Chess Moves Without Rules
- TRELLIS plays the move after which memory recognizes the resulting position best. We judged 200 positions: legal moves that lose material count as the middle regime.
Table: Candidates | Loses material | Illegal / Legal moves, best recognized (1,000 positions learned) | .38 | 0 / Legal moves, uniform random | .35 | 0 / Any displacement, best recognized | .11 | .77 / Any displacement, uniform random | .10 | .70
- Recognition barely separates poor legal moves from reasonable ones, and it prefers legal moves only slightly over chance.
Notes: Backup. In chess we asked the same regime question about moves. TRELLIS has no move generator, so it plays the legal move after which memory recognizes the resulting position best. Among legal moves, the share that lose at least two pawns of material stays between .27 and .42 across every setting we tried, around the .35 of a uniformly random legal move, so recognition does not separate poor moves from reasonable ones. When the model is not told the rules and chooses among legal moves and random displacements, about three quarters of candidates are illegal, and its best-recognized picks are illegal .77 of the time against .70 for a random pick. The judge is a material criterion, not an engine, so it catches hung pieces and lost exchanges but not positional sense.
