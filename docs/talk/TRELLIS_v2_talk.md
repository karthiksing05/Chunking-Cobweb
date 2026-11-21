# TRELLIS v2 talk: slide content

Deck: https://docs.google.com/presentation/d/161jduQT2dR5sZmeBV9EgKDq_Xm7nIUGlsjTDBlHKb2A (44 slides: 39 main, 5 backups). Style follows TRELLIS-ACS-2026-talk-v5 (Drive id 1tSl8rTk01KTo7LToUqgKuzigc44wVbGW): plain title-case titles, short declarative text, one figure per slide, speaker notes on every slide, slide number bottom right, backups at the end. No pull-quote or slogan lines (removed 2026-11-20 on the user's feedback).

## Structure (revision 3, 2026-11-20): 39 main slides and 5 backups

The user found the earlier 61-slide deck good but long and some pictures hard to read. Revision 3 merged or removed 17 slides and replaced small-font pictures with talk versions (confs/trellis-v2/make_talk_figures_rev3.py, PNGs *_talk.png in confs/trellis-v2/talk_figures/). Prediction sentences moved onto the first slide of each section and the speaker notes were extended to carry the removed slides' content.

| # | Title | Notes |
|---|---|---|
| 1 | TRELLIS v2: Chunking as Compression | title |
| 2 | Some Empirical Phenomena | motivation |
| 3 | Concepts and Chunks as Compression | merged old 3 and 4 (generation sentence moved into the closing line) |
| 4 | Representation and Learning Postulates | R1, R2, L1, L2 above one image (instances_talk.png); merged old representation, learning v1 |
| 5 | Organization Postulates | O1, O2 above hierarchies_talk.png (full width, regenerated) |
| 6 | Parsing in Version 1 | P1-P3, infographic |
| 7 | Generation in Version 1 | P4-P6, infographic |
| 8 | What Changes in Version 2 | overview table (old "Representation and Organization" removed) |
| 9 | Parsing: Version 1 to Version 2 | postulate |
| 10 | Steps of Parsing | |
| 11 | Parsing Over Every Span | infographic; the discarded "dog eats" (.11) node removed from panel 3 |
| 12 | A Parse of a Held-Out Sentence | |
| 13 | Parsing Within a Window | |
| 14 | Generation: Version 1 to Version 2 | |
| 15 | Steps of Generation | |
| 16 | Generation One Element at a Time | infographic |
| 17 | A Generated Sentence | |
| 18 | Learning: Version 1 to Version 2 | merged postulate and steps (one sentence under the two columns) |
| 19 | Learning From Every Candidate | infographic, larger cards |
| 20 | Experimental Design | prediction sentence and judge description folded in (old Predictions: Children's Stories, Judging Syntax and Meaning) |
| 21 | Parsing Real and Broken Sentences | |
| 22 | Parsing Along Training | table "Parse Success by Training Size" folded into this figure |
| 23 | Parses That Are Not Right-Branching | old "Same Sentence Under Four Seeds" dropped |
| 24 | Story Chunks Through Experience | two sentences at 250 and 4,000 (story_chunks_cases_talk.png); phrase result in caption, plot in backup |
| 25 | Chunks Reused Across Contexts | |
| 26 | What the Three Models Wrote | |
| 27 | Experimental Results: Generation | figure replaces the table slide |
| 28 | Chess Chunks on the Board | prediction folded in (old Predictions: Chess); old "Frequent Chess Chunks" dropped |
| 29 | Chess Chunks Through Experience | one position at 300, 1,000, 4,000 (chess_ideas_cases_talk.png) |
| 30 | One Position, Real and Scrambled | chess_case_talk.png (case 6 of case_s13.json) |
| 31 | Compression Across Experience | chess_chunkers_talk.png; table "Compressing Shifted Chess Positions" folded in |
| 32 | Chase and Simon Trials | cs_trials_talk.png; recall-trial slide folded in |
| 33 | Recall Across Experience | chess_cs1973_recall_talk.png |
| 34 | Removing Chunks and Concepts | |
| 35 | Abstraction Costs Meaning Before Grammar | prediction folded in |
| 36 | With an Exact Judge, One Dial Gives All Three | controlled grammar |
| 37 | Related and Future Research | |
| 38 | Concluding Remarks | |
| 39 | References | |
| 40 | Backup: Phrase Chunks Over Experience | story_chunks_phrases_talk.png |
| 41 | Novelty and Grammaticality | moved to backups |
| 42 | Backup: Chess Ideas Over Experience | chess_ideas_curves_talk.png |
| 43 | Backup: Why the Middle Regime Failed | |
| 44 | Backup: Chess Moves Without Rules | |

Deleted in revision 3: Compression and Generation (merged into 3), Learning in Version 1 (merged into 4), Representation and Organization, Learning postulate slide (merged into 18), four "Predictions:" dividers, Judging Syntax and Meaning, Parse Success by Training Size (table), The Same Sentence Under Four Seeds, Frequent Chess Chunks, Compressing Shifted Chess Positions, A Chase and Simon Recall Trial, the Learning Curves section divider, the duplicate "Backup: Removing Chunks and Concepts", and the backup "Parsing Along the Learning Curve". The earlier 61-slide outline is in git history.

Revision 3 Drive talk images (all anyoneWithLink readers): hierarchies_talk_v2 1D7QTXrPa0XyMQvJAvECpaoUbRZZ6-TVI, parse_v2_infographic_talk 1JeZ0SPZgwpxO7w-pjN6uEG1-BLUGmqys, story_chunks_cases_talk 1pQTsbwAI5Dqy43Zl9HsQCW9XabVXx3XD, story_chunks_phrases_talk 1dfYbOqRP5UzGJK9AvxgnuPFzfk6VZ41j, chess_ideas_cases_talk 1nkM9SL86oh-FB48ojEz4wCJ0l6eYUkFW, chess_ideas_curves_talk 1RYd3FJuB5HJypbdxrx_V4Zn5Rfkq3pRA, chess_case_talk 1_H_GNfAnAs0Fq_WqtgzRq0AoHy9Ihoje, cs_trials_talk_v2 1lcbqOpMjxbR0xELEctgHJO0g5mMCaWEH, chess_cs1973_recall_talk 1yoOa37pb0EPsPPHCZu3cpe-xi5WhNvKs, chess_chunkers_talk 1UGyJByxScEgRAvrOmCLhvfCaizPQd5Tr, instances_talk 1rpE1z0uN2R25ky5ubuMie4DSIKZMFh5P, learning_v2_infographic_talk 1QqYMNDzN846y6E0jRua6QlFj4Yb-6E5B.

## New slides: text and notes

The infographics are drawn by confs/trellis-v2/make_talk_infographics.py (helpers taken from confs/acs-26/make_paper_figures.py), PNGs in confs/trellis-v2/talk_figures/. Their numbers are schematic, as in the ACS infographics, and the speaker notes say so. Slides 5-9 reuse confs/acs-26/paper/graphics/{instances,hierarchies,parse_infographic,generation_infographic}.png. Postulate wording on slides 5-9 is copied from the ACS paper (R5, R6 -> R1, R2 here; O4, O5 -> O1, O2; P5-P7 -> P1-P3; P8-P10 -> P4-P6; the two new-learning statements -> L1, L2). Wording on slides 12, 17, 21 (version 2 column) is copied from confs/trellis-v2/paper/main.tex lines 102-176.

Version 1 -> version 2 postulates (changed words are bold on the slides):
- Parsing: "iteratively ... bottom up ... selects the best recognized chunk" -> "sums over every binary segmentation ... inside pass ... outside pass ... most expected correct chunks".
- Generation: "from the top down, decomposing chunks" -> "one element at a time ... sums over every chunk that could surround the position ... both directions".
- Learning: "composite instances ... primitive instances only into the context taxonomy" -> "Every candidate chunk ... weighted by its posterior probability".

Slide 16 (r218): table of trees for "when they got home something unexpected happened" (seed 13, n = 1000) from experiments/v3/results/story_parse_eval/r218_cases.md; composites evaluated 56 (full), 13 (greedy), 15 (W4), 25 (W4 stride 2), 29 (chart cap 4); gold F1 .43 (W3s2), .41 (W4s2), .35 (full) from r218_window_parse.md.

Speaker notes for slides 5-9, 12-14, 16-19, 21-24, 37, 46, 49 are in the deck.

## Retained slides: text as of the earlier outline

Headings below carry the OLD slide numbers (before the 2026-11-20 restructure). Slides 2, 3, 4, 7, 8, 25 (old 10 kept) and the concluding slide were edited; slide text on the deck is authoritative.

### TRELLIS v2: Chunking as Compression (old 1)
- Learning concepts and chunks from raw text and chess, one experience at a time
- Karthik Singaravadivelan and Pat Langley
- Georgia Institute of Technology, Atlanta, Georgia · Institute for the Study of Learning and Expertise, Palo Alto, California
- Follows Singaravadivelan and Langley (2026), "A Unified Account of Concepts and Chunks," Advances in Cognitive Systems.
Notes: At ACS we presented TRELLIS as a theory that unifies concepts and chunks, and we tested it on small synthetic grammars where it learned from sample parse trees. This talk covers version 2. It learns from raw tokens with no parse trees, it parses and generates by summing over every analysis its memory allows, and it runs on children's stories and on chess. The thread through the talk is compression: a chunk lets memory refer to many parts with one symbol, and a concept lets one description cover many instances.

### Some Empirical Phenomena (old 2)
- Limited Memory Span
- People hold about seven items at once (Miller, 1956).
- 7 digits in 7 slots
- Mitigated by Expertise
- Chess masters recall a whole board after five seconds; on random boards the advantage shrinks (Chase & Simon, 1973).
- 26 pieces in 7 chunks
- A chunk is one symbol for many parts, so chunking compresses what memory must hold. Expertise is a better code, not a bigger memory.
Notes: Two phenomena motivate the talk. People have a limited ability to recall presented items, about seven digits or words, which is the left row of slots. Expertise in a domain like chess effectively expands that span, because experts perceive familiar configurations as chunks (Chase & Simon, 1973). The right row shows one real middle-game position that our model reads into seven chunks that together hold 26 pieces; we draw it on a board later. Miller's span applies to chunks, not raw items, so we can see chunking as a compression scheme: it spends a dictionary of familiar patterns to shorten the description of new experience.

### Concepts and Chunks as Compression (old 3)
Table:  | Chunks (composition) | Concepts (generalization) / What it saves | One symbol for parts that occur together | One description for instances that behave alike / Language | "one day" stored as one unit | dog, cat and bird can all fill "the ___ ran" / Chess | castled king and pawn shield: f2 g1 g2 h2 | a pawn chain on any file / Alone it fails because | it cannot say what a new chunk resembles | it has only single tokens to group
- These are two ways of saving symbols. TRELLIS uses both, and each one names the other's parts.
Notes: Researchers have treated concepts and chunks as distinct, with concepts emphasizing generalization and chunks emphasizing composition. Read as compression they are two ways of saving symbols. A chunk saves symbols by naming parts that occur together; a concept saves description by grouping instances that behave alike. Neither works alone: a system with chunks and no concepts cannot say what a new chunk resembles, and a system with concepts and no chunks has only single tokens to group. In TRELLIS a chunk's parts are named by concepts, and a concept's description includes what an element is made of.

### Compression and Generation (old 4)
- context x
- general concept
- narrowed by x
- new output y
- A memory that compresses well gives high probability to experiences it has not seen but could have seen. / A general concept covers many instances, and the context narrows it to one output. / Grammatical novelty comes from that step, which recombines parts no single instance had.
Notes: We judge TRELLIS by what it generates as well as by bits, because the two readings go together. If chunking is compression, then sampling from the code should produce new but well-formed experience. A general concept covers many instances, and the context in which it is used narrows it to a single output; that narrowing is where grammatical novelty comes from. It is also where errors come from, and the novelty results later in the talk measure that cost.

### What Changes in Version 2 (old 5)
Table:  | TRELLIS v1 (ACS 2026) | TRELLIS v2 / Input | sentences with sample parse trees | raw token stream only / Parsing | greedy, bottom up, recognition threshold | inside-outside over every span and split / Generation | top-down expansion with recall thresholds | one word at a time, from what the whole sentence allows / Learning | committed chunks only | every candidate chunk, weighted by its posterior / Domains | synthetic context-free grammars | toy grammars, TinyStories, chess
- The two linked Cobweb taxonomies stay as they were.
Notes: These changes answer the open questions at the end of the first talk: can a less greedy parser handle the choices it gets wrong, can we learn chunks without sample parse trees, and can the theory handle natural corpora and chess. Version 2 reads a raw token stream, so gold labels are used only for scoring. Parsing no longer commits to one pair at a time, and learning no longer keeps only the chunks the parser committed to. The representation and organization postulates are unchanged.

### Representation and Organization (old 6)
- Context taxonomy / Groups elements that occur in like surroundings, words and chunks alike.
- names parts
- names wholes
- Content taxonomy: one concept
- concept
- One content concept after 1,000 TinyStories sentences (seed 13).
- Both taxonomies are Cobweb trees. A composite's two parts are named by context concepts, so the content taxonomy never sees a raw word.
Notes: Long-term memory holds two Cobweb taxonomies. The context taxonomy takes every word and every composite and groups them by the surroundings in which they occur. The content taxonomy takes composites only and groups them by what they are made of. They are linked: a composite's left and right parts are named by context concepts, so the content taxonomy never sees a raw word. The concept on the right is a real one from a model trained on 1,000 children's sentences. It groups "he wanted" with "she wanted", "tom wanted" and "he liked", so one description covers four chunks that no single sentence contained together.

### Parsing in TRELLIS v2 (old 7)
- A parse of a held-out sentence, seed 13, after 4,000 sentences. Each internal node is a chunk.
- Parsing sums over every bracketing of the sentence instead of committing to one pair at a time. / The inside pass scores each span from its sub-spans, and the outside pass scores everything around it. / The tree shown is the bracketing with the most expected correct chunks.
Notes: In version 1 parsing was greedy: it proposed candidate chunks over adjacent pairs, admitted those that cleared a recognition threshold, and committed the best one. A wrong early choice could not be undone. Version 2 sums over every bracketing with an inside pass and an outside pass over the spans of the sentence, and then reports the bracketing with the most expected correct chunks. This is a real parse of the held-out sentence "the fish was not happy anymore" from a model trained on 4,000 sentences. We do not claim it is the linguist's parse; the evaluation later compares such trees with a shallow-phrase reference and finds them rather different. The same quantities that parse a sentence are what learning uses, since every candidate chunk is stored with its posterior.

### Generation in TRELLIS v2 (old 8)
- 1  Read the grammar from memory
- 2  Read the prompt
- 3  Choose the length
- 4  Fill each word from all ways the rest could finish
- The outside score of a position sums over every chunk that could surround it, so each word sees chunk context in both directions. A pair gate keeps word pairs that memory has seen side by side.
- Prompt: tim was very happy
- TRELLIS wrote: they wanted to play near the water
Notes: Generation in version 2 writes one word at a time, but each word is chosen using a score for what the whole sentence could still become. The outside score of a position sums over every chunk that could surround it, which is how a word sees chunk context on both sides. A pair gate keeps word pairs that the memory has seen side by side; without it grammaticality collapses, a point we return to when we discuss how abstract the words can be. The example is a real sentence written by a model trained on 1,000 sentences from the prompt shown, and the judge scored it as grammatical.

### Learning in TRELLIS v2 (old 9)
- Learning needs no labels, no parse trees and no offline phase; gold labels only score. / Each composite enters both taxonomies with a weight equal to its posterior. / The memory that parses sentence n is the memory built from sentences 1 to n-1.
- Sentence n arrives
- Parse it with the memory so far
- Store every candidate chunk, weighted by its posterior
Notes: Learning stays incremental, as in Cobweb. There is no separate routine for creating chunks: parsing proposes candidates and learning treats them as data for updating the two taxonomies. What changes in version 2 is that every candidate chunk is stored, weighted by how probable the parse says it is, instead of only the chunks a greedy parser committed to. Learning is interleaved with performance, so the memory that parses sentence n is exactly the memory built from the sentences before it.

### Experimental Design (old 10)
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
Notes: TinyStories sentences of three to eight words from a 250-word vocabulary make the language task. Each model sees n training sentences, with n from 125 to 4,000, and the held-out stories are never used for training. The top row is the generation protocol: the model writes new sentences, two language models (Qwen3-8B and phi-4) judge whether each is grammatical and meaningful, and the training stream tells us whether it is new, meaning it contains a trigram the training set never stored. The bottom row is the parsing protocol: we take real held-out sentences, break them in controlled ways, and ask whether a score accepts them. The neural baselines are an LSTM and a small GPT trained on exactly the same sentences, sized by held-out loss and sampled at tuned temperatures, plus n-gram models. We use eight seeds for generation and four for parsing, and we report paired differences.

### Judging Syntax and Meaning (old 11)
- Two language models, Qwen3-8B and phi-4, each answer a syntax question and a meaning question, and we average their logits. We set thresholds on one half of the controls and the table reports the other half.
Table: Control sentence | Example | Syntax pass | Meaning pass / Real | the fish was not happy anymore | 100% | 95% / Content words swapped | the hard door looked hard | 88% | 13% / Two adjacent words swapped | little the bird looked sad | 13% | 6% / Agreement error | they is not friends anymore | 4% | 15% / Scrambled | not anymore was happy the fish | 2% | 0%
- A content swap keeps its syntax and loses its meaning, so the judge can tell the two apart. The syntax threshold is permissive on purpose.
Notes: We needed a judge that separates syntax from meaning, because TRELLIS may be grammatical without being sensible, and the reverse. We built control sets from held-out sentences: real sentences; sentences with every content word swapped for another of the same part of speech, which are grammatical but nonsense; sentences with two adjacent words swapped; agreement errors; and scrambles. Each of two models, Qwen3-8B and phi-4, answers a syntax question and a meaning question, we average their logits, and we set the thresholds on one half of the controls. On the other half, 150 sentences per class, the judge passes 88 percent of content swaps on syntax and only 13 percent on meaning, while scrambles and agreement errors fail both. The syntax threshold is permissive by design, so that well-formed nonsense passes, and the judge agrees with our earlier Qwen2.5-3B judge on 79 to 81 percent of sentences. We call a sentence meaningful when it passes both questions, and we hand-labeled no generated sentences. Two caveats: a few agreement errors are actually grammatical, such as plural uses of fish, and a single adjacent swap passes syntax at 13 percent. We validated the judge on constructed controls only.

### Experimental Results: Generation (old 12)
- Fair count of new, judged-grammatical sentences, TRELLIS minus each matched model:
Table: Sentences learned | vs LSTM | vs GPT / 250 | slightly ahead (+.015) | slightly ahead (+.020) / 500 | level; behind on 4 confirmation seeds | level / 1,000 | LSTM ahead on meaning (-.026) | GPT ahead on syntax (-.019) / 4,000 | LSTM ahead (-.057) | GPT ahead (-.055)
- Level or slightly ahead at 250 sentences, behind from 1,000.
- Judge: Qwen3-8B and phi-4, syntax pass, 8 seeds (2 at 4,000). Level means the paired 95% interval includes 0. The earlier Qwen2.5-3B judge chose the settings.
Notes: The first result is on generation. The measure is the fair count: distinct sentences that are new, contain a trigram the training stream never stored, and pass the judge, divided by attempted draws. Two models, Qwen3-8B and phi-4, judge each sentence, and the table gives the syntax pass pooled over eight seeds. At 250 sentences TRELLIS is slightly ahead of both neural models (+.015 over the LSTM and +.020 over the GPT, with intervals that reach zero). At 500 sentences the three tie to within .002 on syntax when we pool the eight seeds, but on the four confirmation seeds the LSTM leads on all four (syntax -.012, meaningful -.022). At 1,000 sentences the LSTM leads on the meaningful count by .026 and the GPT leads on syntax by .019. At 4,000 sentences, on two seeds, both neural models lead by .055 to .057 on syntax. Level means that the paired 95 percent interval includes zero and has a lower bound above -.03. With the earlier Qwen2.5-3B judge TRELLIS was ahead of the GPT on eight of eight seeds at 250 and 500 sentences, and that result does not survive the two-model judge because the GPT's count rises more under it. The earlier judge also chose every sampling temperature and window, which limits the comparison. TRELLIS uses no gradients, and no model reaches 95 percent grammatical. The next slide plots these curves.

### What the Three Models Wrote (old 13)
- Same 1,000 training sentences for each model. First two draws per prompt, labeled by the judge: M meaningful, S syntax-only, N nonsense. All three write new sentences, and all three also write broken ones.
Table: Prompt | TRELLIS | LSTM | GPT / tom said sorry cat | it had a dog named spot (M) | the bird saw a big dog named max (M) | sara says good tim (N) /  | can play with it (N) | she is the best of you (S) | they were very happy (M) / he told his friends | tom and max were all day long (N) | mom said yes we can help (M) | tim and learned but sam was scared (N) /  | sue thought it and a big fish (N) | tim was happy and said thank you tim (S) | they are very are friends (N) / tim was very happy | it liked to help people (M) | max was not happy (M) | they were very happy (M) /  | let's go get your food together (M) | the bird felt sad (M) | tom and tim are friends (M)
Notes: Here are real generations from seed 29 after 1,000 training sentences, each model sampled at the temperature we tuned for it. We took the first two draws for each of three prompts in the order they were sampled, so these are not the best ones. The letters are the labels of the two-model judge: M meaningful, S syntax-only, N nonsense. All three models write fluent sentences next to word salad such as "sara says good tim" and "they are very are friends." Of the six draws, three are meaningful for TRELLIS, four for the LSTM and three for the GPT, and the LSTM has two syntax-only draws. The outputs show little visible dependence on the prompt.

### Novelty and Grammaticality (old 14)
- Models compared at about the same share of generations with a trigram unseen in training (1,000 sentences, 4 seeds). Real sentences are judged grammatical .89 of the time (earlier Qwen2.5-3B judge).
Table: Model | New trigram | Grammatical / TRELLIS, window 3 | .34 | .76 / LSTM | .32 | .92 / TRELLIS, window 4 | .52 | .71 / LSTM | .52 | .81 / TRELLIS, window 5 | .57 | .67 / GPT | .57 | .80
- Every learner pays for novelty with grammaticality. / Neural models pay less at this size. / Shorter context windows copy more, and longer windows generalize more and err more.
Notes: This is the novelty against abstraction trade-off, judged by the earlier Qwen2.5-3B model, which also chose the sampling settings. A model that copies the training set is grammatical but not new; a model that recombines parts more freely produces more new sentences and more errors. To compare fairly we look at models at the same level of novelty, measured as the share of generations that contain a trigram not stored in training. At about a third novel, TRELLIS with a three-word window is grammatical .76 of the time and the LSTM .92. At about half, the LSTM and GPT are at .81 and .80 while TRELLIS with windows of four and five words is at .71 and .67. Shorter windows copy more and longer windows generalize more and make more errors. From 1,000 to 4,000 sentences only the new-trigram sentences of TRELLIS become less grammatical, which we suspect means it innovates too little after rare histories and too much after frequent ones; we have not tested that. This is the price of a coarser code, and it shows where the level of abstraction has to be chosen.

### Parsing Real and Broken Sentences (old 15)
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

### Parse Success by Training Size (old 16)
- Share of sentences each score accepts, as training sentences grow (4 seeds). Lower is better on broken sentences.
Table: Broken sentences | Score | 125 | 500 | 1,000 | 4,000 / Scrambled | TRELLIS parse | .88 | .81 | .78 | .77 /  | TRELLIS generator | .39 | .19 | .13 | .05 /  | LSTM / GPT | .65 / .40 | .16 / .09 | .08 / .04 | .01 / .01 / Content swapped | TRELLIS parse | .94 | .93 | .92 | .91 /  | TRELLIS generator | .86 | .79 | .71 | .58 /  | LSTM / GPT | .80 / .80 | .66 / .70 | .62 / .64 | .41 / .40
- Real sentences are accepted .90 to .94 by every score. Threshold fixed on real sentences.
Notes: This is the parse success curve along the learning curve. For each training size we fix the threshold on grammatical sentences and report how many broken sentences each score still accepts, so lower is better here. The parse score alone accepts most scrambles at every size, which is why we added the generator's conditional probabilities. The generator keeps up with the n-gram, LSTM and GPT baselines at 125 to 500 sentences on scrambles, adjacent swaps and agreement errors, then falls behind. The content swaps are where it is weakest: at 4,000 sentences it still accepts 58 percent against about 40 percent for the neural models, and the parse alone accepts 91 percent. In short, TRELLIS is better at telling syntax apart than at telling meaning apart.

### The Same Sentence Under Four Seeds (old 17)
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

### Parses That Are Not Right-Branching (old 18)
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

### Chunks Reused Across Contexts (old 19)
- The chunk "the bird" (seed 13)
Table: Sentences seen | 100 | 250 | 500 | 750 | 1,000 / Times bracketed | 2 | 5 | 8 | 13 | 14
- First sentences where it was a chunk: / the bird saw a big cat / the big cat walked away with the bird / they all laughed and played with the bird
- The concept it sits under changes
- after 250 sentences / all day; had fun; very happy; their toy
- after 500 / he was; he liked; yes you; they played
- after 1,000 / happy too; the animals; was inside; the tree
- 14 chunks are in the top 50 of every seed. Seeds overlap on the top 50 at .33, below raw frequency (.51) and BPE (.43).
Notes: A chunk is reused across contexts, and its concept keeps changing as the taxonomies restructure. On the left is the chunk "the bird" in one seed: it first appeared as a chunk at sentence 27 and was bracketed 14 times by 1,000 sentences. Its first sentences show it as a subject and as the end of a prepositional phrase. On the right are some of the other chunks that share its concept at three checkpoints; the group is different each time because the tree is rebuilt as data arrive, so what a concept contains depends on how much has been seen. Across seeds, 14 chunks are in the top 50 of all four, including "one day," "he wanted" and "the dog." The honest cost is that the overlap of the top 50 between seeds is .33 at 1,000 sentences, lower than a raw frequency count (.51) or BPE (.43), so in language chunking adds no cross-seed agreement beyond frequency.

### Chess Chunks on the Board (old 20)
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

### Compressing Shifted Chess Positions (old 21)
- Bits to code a held-out position after 4,000 positions learned (lower is better). Shifted positions put familiar shapes on squares never seen in training.
Table: Coder | Control positions | Shifted positions / TRELLIS, best variant | 127.7 | 147.2 / TRELLIS, standard | 130.6 | 153.6 / BPE over squares | 108.8 | 205.5 / BPE over shapes | 111.3 | 149.7
- Byte-pair coders code ordinary positions better. On shifted positions TRELLIS trails byte-pair over shapes (151.5 against 149.7 bits, three seeds).
- Best variant: better of two variants per column; shifted value uses scoring rule C7. Seed 13, one run.
Notes: This is where chess tests chunking as compression. We code each held-out position with the learned chunks, count the bits, and compare with byte-pair coders over squares and over shapes (seed 13, one run, model cost not charged). On ordinary positions byte-pair over squares is best at 108.8 bits, against 130.6 for the standard TRELLIS configuration and 127.7 for the better of two variants. When the same shapes appear one file over, on squares never seen in training, byte-pair over squares degrades to 205.5 bits. The standard configuration codes them in 153.6 bits, 52 bits better, and its three-seed mean is 151.5, which still trails byte-pair over shapes at 149.7. Only the C7 variant, which scores a new single-position concept as one correct guess as the original Cobweb does, reaches 147.2 and edges ahead by 2.5 bits. Chunks that describe pieces by their relations transfer when the squares change, and a frequency-based miner over shapes does the same, so we claim that TRELLIS compresses chess about as well as byte-pair over shapes and never better under the standard configuration. The stronger evidence that the chunks do the compressing comes from removing them: the code grows by 47 to 59 bits per position over three seeds, lands within 1.4 to 2.4 bits of a list of independent pieces, and the recall gap between game and random positions disappears.

### A Chase and Simon Recall Trial (old 22)
- Seven chunks held after the first look at a 26-piece middle-game position. The red square was recalled twice.
- Pieces correct after trials 1 to 4: 12, 18, 25, 26 of 26
Table: Pieces recalled | Beginner | Club | Master / People | 4 | 8 | 16 / TRELLIS, 7 chunks | 7.2 | 9.7 | 12.9
- TRELLIS after 300, 1,000 and 4,000 positions (3 seeds; subjects share one memory). Recall reaches 15.6 at about 8,000 positions and levels off near 16.7. Random-board recall rises (4.2 to 6.0) where people's falls.
Notes: Chase and Simon showed people a board for five seconds and asked them to rebuild it. Masters recalled far more than beginners, but only on real positions, because they see familiar groups of pieces. We give TRELLIS a short-term memory of seven chunks and run the same task. The board shows one real position of 26 pieces and the seven chunks the model held after its first look. Twelve of its pieces were right; the red square was recalled twice, as a pawn and as a bishop, so one of the two is a commission error. Over repeated trials it reaches 12, 18, 25 and 26 correct. Across positions its recall rises with experience, from 7.2 pieces after 300 positions to 12.9 after 4,000, the same ordering as people at 4, 8 and 16. On a longer stream it reaches 15.6 at about 8,000 positions and levels off near 16.7. These are means over three seeds with ten simulated subjects each. Each seed learns one memory that its ten subjects share, so the subjects differ only in the 28 test positions they see. Random-board recall rises too, from 4.2 to 6.0 pieces, while people's falls from 3.0 to 2.0, so the model departs from the human pattern there. Any learner that stores more chunks with more data would recall more, and we ran no chunker baseline, so the ordering by experience is weak evidence for the account. Recall is capped by the number of chunks times the pieces per chunk, which is the compression argument again: a better code is what raises it.

### Abstraction Costs Meaning Before Grammar (old 23)
- We varied how abstract the words are in the generator's context, and hoped to get three kinds of output.
Table: Setting | Expected output | Example | Reached / Exact words | grammatical, meaningful | he liked to play with his ball | yes, 54 to 63% / Word classes | grammatical, meaningless | the hard door looked hard | no, peak .21 to .27 (target .5) / One class (root) | word salad | this had the this day be | yes, 87 to 89%
- The meaningless share of grammatical output rises with abstraction: .25 to .29 at the leaf, .42 to .45 at b = 7, .72 to .74 at the root. / The target of .5 equals an oracle's ceiling (.45 to .52). The context window moves output between meaningful (.68) and nonsense (.85). / In chess, file-only abstraction keeps legality longer than realism (at most .25 legal but unrealistic); the judge is partly file-tolerant (mirror image: 79% real).
Notes: The question is whether one setting, the abstraction of the words in the generator's context, moves TRELLIS between three regimes: grammatical and meaningful, grammatical but meaningless, and word salad. The two-model judge defines them: a sentence is meaningful if it passes both the syntax and the meaning question, syntax-only if it passes syntax and fails meaning, and nonsense if it fails syntax. The ends are easy. With exact words 54 to 63 percent of the sentences are meaningful, and with one class 87 to 89 percent are nonsense. In between, the meaningful share falls and the nonsense share rises steadily. The regularity is that abstraction costs meaning before it costs grammar: the meaningless share of grammatical output rises from .25 to .29 at the leaf, through .42 to .45 at level b = 7, to .72 to .74 at the root. The syntax-only regime peaks at .21 to .27 at b = 7, far below our target of .5, and an oracle that swaps every noun, verb and adjective within exact morphosyntactic cells reaches only .45 to .52, so the target equals the ceiling of a perfect swapper and missing it separates no mechanism from another. The context window is a separate dial between meaningful (.68 at a reach of three words) and nonsense (.85 at a reach of zero). We call this a regularity because it rests on three runs and one permissive judge, and a random class-free swap gives a similar share (.32 to .45). In chess the same ordering appears with locations in place of words. Redrawing only the file of each chunk keeps legality longer than realism, but the legal-unrealistic share never exceeds .25. The realism judge is partly file-tolerant: the mirror image of a real position scores real 79 percent of the time, against 96 percent unchanged, so the judge shapes that cap. The backup slides give the attempts.

### Related and Future Research (old 24)
- Other frameworks address parts of the problem:
Table: Framework | Relation to our theory / Cobweb variants: LABYRINTH (Thompson & Langley, 1991) | lacks a context taxonomy / EPAM and CHREST (Feigenbaum, 1961; Gobet & Lane, 2012) | parse from the top down; chunks without concepts / Compression-based grammar induction (Wolff, 1982) | same principle, with a fixed code / Neural language models | strong at scale, no explicit chunks
- Open questions for future work:
- Can memory choose the level of abstraction itself? / Can TRELLIS keep up with neural models past 4,000 sentences? / Can a chess generator compose positions from chunks?
Notes: Other frameworks address parts of the problem. Labyrinth encoded both primitive and composite concepts but lacked a context taxonomy and suffered substantial order effects. EPAM and CHREST parse structures from the top down, and treat chunks without concepts. Grammar induction systems that minimize description length, such as Wolff's, follow the same compression principle with a fixed code instead of a learned memory. Neural language models are strong at scale but create no explicit chunks. For future work, we would like memory to choose its own level of abstraction rather than rely on a hand-set cut, we need TRELLIS to keep up with neural models past 4,000 sentences, and we want a chess generator that composes positions from chunks.

### Concluding Remarks (old 25)
- TRELLIS v2 treats chunking as compression:
- One symbol for many parts, with expertise as a better code / Chunks and concepts learned with no labels or gradients; chunks compress chess, concepts drive stories / Level with LSTM and GPT models at 250 sentences, behind from 1,000; chess recall follows Chase and Simon / Abstraction costs meaning before grammar; the right level remains open
- 7 slots
- 3 slots
Notes: Our account treats chunking as a compression device. Once a memory has introduced a unit for some configuration it refers to it with a single symbol, so the context it has to carry shrinks from a list of primitives to a handful of chunks, as in the bottom row. Version 2 learns chunks and concepts from raw text and chess without labels or gradients. The evidence for chunking as compression comes from chess: removing the chunks adds 47 to 59 bits per position and erases the recall gap between game and random positions, and the chunks code shifted positions in slightly more bits than byte-pair coding over shapes. In short stories we found no effect of the chunks. The concepts drive generation, and the word-window generator, not the chunks, earns the 2-bit margin over byte-pair coding. With 250 sentences the system writes new, judged-grammatical sentences about as often as matched LSTM and GPT models, and it trails them from 1,000 sentences. Abstraction costs meaning before grammar, and the grammatical but meaningless regime stays thin. We suspect that chess rewards chunks because positions repeat multi-piece structures while short sentences offer few multi-word units, but that remains untested. We do not claim that it competes with neural systems at their scale, but a memory that builds and refers to its own units offers a better way to handle context than retaining the whole past.

### References (old 26)
- Chase, W. G., & Simon, H. A. (1973). Perception in chess. Cognitive Psychology, 4, 55-81. / Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? arXiv:2305.07759. / Fisher, D. H. (1987). Knowledge acquisition via incremental conceptual clustering. Machine Learning, 2, 139-172. / Gobet, F., & Lane, P. C. R. (2012). CHREST. Encyclopedia of the sciences of learning, 556-559. / Langley, P. (2025). Concepts and chunks in cognitive systems. Advances in Cognitive Systems, 11, 33-44. / Miller, G. A. (1956). The magical number seven, plus or minus two. Psychological Review, 63, 81-97. / Singaravadivelan, K., & Langley, P. (2026). A unified account of concepts and chunks. Advances in Cognitive Systems. / Thompson, K., & Langley, P. (1991). Concept formation in structured domains. / Wolff, J. G. (1982). Language acquisition, data compression, and generalization. Language and Communication.
Notes: References cited in the talk, including the backup slides.

### Backup: Why the Middle Regime Failed (old 27)
- We looked for a level of the learned taxonomy where slots are syntactic: pure (purity at least .8) and still varied (at least 3 bits of filler entropy).
Table: Level in the tree | Slot purity | Filler entropy / Leaf | .8 and above | 0.03 bits / 8 bits | .39 to .40 | 4 bits or more
- No level passes both tests, on seeds 13 and 17 with 1,000 sentences. The generation run that would have used such a level was not launched.
Notes: Backup. To produce grammatical but meaningless output we looked for a band of the learned content tree whose slots are syntactic, meaning they are pure in part of speech and still have varied fillers. The stopping rule required purity of at least .8 and filler entropy of at least 3 bits. No band meets it on either seed: purity reaches .8 only at the leaf, where the filler entropy is 0.03 bits, and at the band where entropy is still at least 4 bits purity is .39 to .40. So reading the tree at a coarser level does not pass through a syntactic-slot regime, and we did not run the generation arms.

### Backup: Parsing Along the Learning Curve (old 28)
- AUC for real against each broken class, 125 to 4,000 sentences, 4 seeds, TRELLIS generator against 5-gram, LSTM and GPT. / At 125 sentences scramble AUC is .886 (5-gram .880, LSTM .807, GPT .888). / At 4,000 sentences content-swap AUC is .81 against .88 to .89.
Notes: Backup. AUC compares grammatical sentences against each broken class along the learning curve over four seeds. At 125 sentences the generator scores scrambles at .886 against .880, .807 and .888 for the 5-gram, LSTM and GPT; at 4,000 sentences its content-swap AUC is .81 against .88 to .89.

### Backup: Chess Moves Without Rules (old 29)
- TRELLIS plays the move after which memory recognizes the resulting position best. We judged 200 positions: legal moves that lose material count as the middle regime.
Table: Candidates | Loses material | Illegal / Legal moves, best recognized (1,000 positions learned) | .38 | 0 / Legal moves, uniform random | .35 | 0 / Any displacement, best recognized | .11 | .77 / Any displacement, uniform random | .10 | .70
- Recognition barely separates poor legal moves from reasonable ones, and it prefers legal moves only slightly over chance.
Notes: Backup. In chess we asked the same regime question about moves. TRELLIS has no move generator, so it plays the legal move after which memory recognizes the resulting position best. Among legal moves, the share that lose at least two pawns of material stays between .27 and .42 across every setting we tried, around the .35 of a uniformly random legal move, so recognition does not separate poor moves from reasonable ones. When the model is not told the rules and chooses among legal moves and random displacements, about three quarters of candidates are illegal, and its best-recognized picks are illegal .77 of the time against .70 for a random pick. The judge is a material criterion, not an engine, so it catches hung pieces and lost exchanges but not positional sense.

## Figure slides added 2026-11-19 (deck is now 41 slides after the 2026-11-20 paper alignment)
The sections above keep the original 1-29 numbering. Eleven figure slides were added from the detailed site (docs/site/trellis_full.html), and later slides shifted: old 13-16 are now 14-17, old 17-18 are 19-20, old 19-20 are 23-24, old 21 is 28, old 22 is 30, old 23-29 are 34-40. Titles, takeaways and speaker notes are in the deck; figures are PNGs under docs/talk/build/img/ (not committed), uploaded to the Drive image folder and shown by URL. New slide, figure, source:
- 13 Generation Across Experience: generation_fair_count_judge2.png, copied from confs/trellis-v2/paper/figures (paper Figure 4; Qwen3-8B and phi-4 judge, 8-seed means, 2 seeds at 4,000), uploaded to the Drive image folder as 1Ft88JWvS7AnkNTbJCj54b1_fYrt4vP4G and shared anyone-with-link reader. It replaced generation_fair_count.png (Qwen2.5-3B judge, r212_seeds.md).
- 18 Parsing Along Training: parse_success_curve.png, fig_talk_curves.py from experiments/v3/results/story_parse_eval/r216_figure.tsv (p_succ_set_mean; same numbers as slide 17).
- 21 Story Chunks Through Experience: docs/site/story_chunks_cases.png.
- 22 Phrase Chunks Over Experience: docs/site/story_chunks_curves.png.
- 25 Frequent Chess Chunks: docs/site/chess_chunk_types.png.
- 26 Chess Chunks Through Experience: docs/site/chess_ideas_cases.png.
- 27 One Position, Real and Scrambled: docs/site/chess_case.png.
- 29 Compression Across Experience: docs/site/chess_chunkers.png.
- 31 Recall Across Experience: docs/site/chess_cs1973_curve.png (data experiments/v3/results/chess_cs1973_exact/; 3-seed means, ten subjects per seed share one memory).
- 32 Chase and Simon Trials: docs/site/cs_fig1.png.
- 33 Chess Ideas Over Experience: docs/site/chess_ideas_curves.png.
Build scripts: docs/talk/build/fig_talk_curves.py (matplotlib figures), gen3.py and gen4.py (slide text, notes and request batches).

## Alignment with the paper (2026-11-20, revision 4, commit 2943564a)
The deck follows confs/trellis-v2/paper/main.tex, supplement.tex and CHANGES_3.md. Changed slides: 2 (random-board wording), 10 and 11 (notes: judge is Qwen3-8B and phi-4, permissive syntax threshold, 79 to 81 percent agreement with the Qwen2.5-3B judge), 12 (table rebuilt from paper Table 4: slightly ahead at 250, level at 500 with the LSTM ahead on the four confirmation seeds, behind at 1,000 and 4,000), 13 (judge-2 figure), 14 (judge-2 M/S/N labels from paper Table 3), 15 (earlier judge named), 28 and 29 (standard 130.6/153.6 and best variant 127.7/147.2 against byte-pair over shapes 149.7; three-seed mean 151.5; no "as compactly as" claim), 30 and 31 (three-seed means, subjects share one memory, random-board recall rises, 15.6 at 8,000 and about 16.7 at 16,000), 34 (retitled; meaningless share .25-.29 / .42-.45 / .72-.74; peak .21-.27 against target .5; oracle ceiling .45-.52; context window .68 and .85; chess file-only legality and the file-tolerant judge), 36 (conclusion).

### Backup: Removing Chunks and Concepts (new slide 41 in the deck) (old 30)
- Bits to code held-out sentences (1,000 sentences learned, 2 seeds) and shifted chess positions (4,000 positions learned, 3 seeds).
Table: Variant | Language, bits per sentence | Chess, bits per position / Full system | 30.8 | 151.5 / Chunks off | 30.8 | 210.7 / Concepts off | 28.8 | 152.1 / Both off | 28.6 | -
- In chess, removing the chunks adds 47 to 59 bits per position. In stories it changes nothing, and removing the concepts shortens the code.
- Without chunks, chess recall falls from 12.9 to 7.0 pieces, with no game-random gap.
Notes: Backup. We removed the parts of TRELLIS to see which ones carry the compression. In language, removing every stored chunk leaves the code length at 30.8 bits per held-out sentence and leaves the generation counts unchanged within the seed spread. Removing the concepts from the generator's reads shortens the code by about 2 bits, to 28.8, and removing both gives 28.6, shorter than the full system at every size. So the 2-bit margin over byte-pair coding belongs to the word-window generator with its pair gate, the chunks add nothing to it, and the concepts cost about 2 bits of code while they help generation. The code stays 5 to 7 bits longer than the trigram and neural codes. Chunks reach the generator only through the context tree they sit in, so the null result for chunks in generation holds partly by construction. In chess the picture reverses. With the chunks capped at one piece, the code grows by 47 to 59 bits per position over three seeds (151.5 to 210.7 on shifted positions), within 1.4 to 2.4 bits of a list of independent pieces, and middle-game recall at seven slots falls from 12.9 to 7.0 pieces with no gap between game and random positions. Removing the concepts costs under one bit, and it removes recall entirely by construction, so that control only bounds the effect. We suspect that chess rewards chunks because positions repeat multi-piece structures, but we have not tested that.

## Revision 3b (readability pass, 2026-11-20)
Slides 6, 7, 16, 22 and 36 use larger-font talk figures from confs/trellis-v2/make_talk_figures_rev3b.py (parse_v1_infographic_talk, generation_v1_infographic_talk, generation_v2_infographic_talk, parse_success_curve_talk, controlled_regimes_talk in confs/trellis-v2/talk_figures/). Infographic text is scaled about 1.5x and a few labels are shortened (panel titles and the admit, rank, pair gate and sentence-boundary boxes). Drive ids: parse_v1 1SPkvq35RmDwN1c6_N8ldQrFFitzOM1yH, generation_v1 1OvMDeHgsWGmBzU9b2jSMe9Oj_L8lU2qV, generation_v2 1Q7hvZWIgcyslUaoTbEMwisCkGRm3vcNF, parse_success 1lw0p-AmySI00RCA5e0FXRk1bgJOqVmrI, controlled_regimes 1uEVmtiUYGN2MZTksQ1Txg_3iCLX_bWMu.

## Revision 4 (2026-11-21): 31 main slides, 13 backups (44 in all)
Slide 7 now uses a three-panel generation figure with 20 pt labels (generation_v1_infographic_talk.png from make_talk_figures_rev3b.py v1_gen3; Drive 1y7gCIBYkbiNjIGQXhDIMyL0v_02XxeL6). Slide 10 uses parse_v2_infographic_talk.png (Drive 19nHJT0iMluJDr7TF79GlCWYYFG_c8NJ6). Slide 28 was resized so its caption clears the page number.
Moved to backups, in this order after References: What Changes in Version 2, Parsing Within a Window, Parsing Along Training, Parses That Are Not Right-Branching, Chunks Reused Across Contexts, What the Three Models Wrote, One Position Real and Scrambled, Recall Across Experience (titles prefixed "Backup:").

Main talk: 1 TRELLIS v2: Chunking as Compression; 2 Some Empirical Phenomena; 3 Concepts and Chunks as Compression; 4 Representation and Learning Postulates; 5 Organization Postulates; 6 Parsing in Version 1; 7 Generation in Version 1; 8 Parsing: Version 1 to Version 2; 9 Steps of Parsing; 10 Parsing Over Every Span; 11 A Parse of a Held-Out Sentence; 12 Generation: Version 1 to Version 2; 13 Steps of Generation; 14 Generation One Element at a Time; 15 A Generated Sentence; 16 Learning: Version 1 to Version 2; 17 Learning From Every Candidate; 18 Experimental Design; 19 Parsing Real and Broken Sentences; 20 Story Chunks Through Experience; 21 Experimental Results: Generation; 22 Chess Chunks on the Board; 23 Chess Chunks Through Experience; 24 Compression Across Experience; 25 Chase and Simon Trials; 26 Removing Chunks and Concepts; 27 Abstraction Costs Meaning Before Grammar; 28 With an Exact Judge, One Dial Gives All Three; 29 Related and Future Research; 30 Concluding Remarks; 31 References.
Backups: 32 What Changes in Version 2; 33 Parsing Within a Window; 34 Parsing Along Training; 35 Parses That Are Not Right-Branching; 36 Chunks Reused Across Contexts; 37 What the Three Models Wrote; 38 One Position, Real and Scrambled; 39 Recall Across Experience; 40 Phrase Chunks Over Experience; 41 Novelty and Grammaticality; 42 Chess Ideas Over Experience; 43 Why the Middle Regime Failed; 44 Chess Moves Without Rules.

## Revision 4b (2026-11-21): 14 pt picture text on slides 6, 10, 14, 17, 25, 28
confs/trellis-v2/make_talk_figures_rev4.py redraws the four infographics (three-word example, large labelled boxes instead of mini-trees, 22 pt in a 13.5 in figure shown 8.6 in wide) and the two plots (16 pt). Slide 25 now shows the middle-game panels only (the random-position panels are in the notes) and a two-line caption. Drive ids (v4): parse_v1 139JWJwl4NRX1KOLII0fYEs76QfCD_kmM, parse_v2 1mn4zpTBH_-dGunZwja5N77oKun196bDQ, generation_v2 1w-zut3bXwy26cQKxLyGarYmrPvVwE8Ny, learning_v2 1-Q3h6YxgJxg5my6AfEZ3j_UQALy6I5xy, cs_trials 15YMY1z52pkpEQDvzBhTtyd30rjChrxZk, controlled_regimes 1r2naJPJZMD-vxkxZ6EozqbBA0VrhbRDj.
