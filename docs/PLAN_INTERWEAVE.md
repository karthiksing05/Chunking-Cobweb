# PLAN_INTERWEAVE: showing why concepts and chunks are interwoven (2026-11-25)

User goal (2026-11-25): keep the thesis (two linked taxonomies; a composite names its
parts by context concepts) and iterate until the paper has a clean, defensible result that
shows the value of interweaving composition (chunks) with generalization (concepts),
read through generation.

## The gap this closes
The paper's removal studies show a division of labour, not an interaction:
chess compression = chunks (concepts add 0.4-0.7 bits); story generation = concepts
(chunks reach the word-window generator only through the context tree, so "chunks off"
is null partly by construction). No result shows a case where the combination does
something neither part can.

## Claim to test (fixed before running)
A composite whose parts are named by concepts is a template that
(a) keeps the structure of the stored instances (what a chunk alone gives) and
(b) admits new fillers that behave alike (what a concept alone gives).
So generation that expands stored composites, filling each part from its concept, should
produce sentences that are both NEW and CORRECT more often than either part alone:
- chunks only (parts named by exact fillers, leaf reads): correct but copies;
- concepts only (no composites, class sequence): new but breaks long-range constraints;
- both: new and correct.

## Testbed: controlled grammar with an exact judge (r230 grammar, 147 words)
Exact judge: R-sem (parses, all selectional restrictions hold), R-syn, R-non. The known
defect of the word-window generator here: about a quarter of leaf generations violate an
OBJECT restriction (verb ... Det (Adj) N, distance 2-3), which a window/pair gate cannot
enforce but a VP composite [V NP] with concept-named parts can.

Primary metric (fixed in advance): **new-correct rate** = share of draws that are R-sem
AND not a training sentence AND contain a word bigram or trigram never stored in training
(the fair count of the paper, with the exact judge in place of LM judges).
Secondary: R-sem precision; object-restriction violation rate; code length of held-out
sentences, split by whether they contain an unseen combination.

Arms (same learner, same memory, only the read-out changes):
1. W: current word-window generator (C6a/C6b leaf, the paper's generator).
2. E-full: expansive generator, composites expanded top-down, parts filled from their
   concept at level b (b chosen on seeds 13/17, confirmed on 19/23/29/31).
3. E-chunks-only: expansive, parts read at the leaf (exact stored fillers).
4. E-concepts-only: no composites; words drawn from concepts in sequence (class n-gram).
5. Baselines: KN 3/4-gram, LSTM, GPT on the same n (existing harness), and a class
   bigram over the same word taxonomy.

Held-out combination split (second test): withhold a set of legal verb-object type
pairings' surface combinations from training (every withheld bigram's words still
appear elsewhere); measure how often each arm generates and how cheaply it codes held-out
sentences that use them.

Seeds: 13, 17 select; 19, 23, 29, 31 confirm. n = 250, 1,000, 4,000.
Decision rule: E-full beats both single-part arms on new-correct with paired t interval
excluding zero on the confirmation seeds -> GO, write into the paper; otherwise report.

## Then TinyStories
Same arms with judge 2 and the fair count (C1 protocol), to see whether the expansive
read-out carries over. Reported either way.
