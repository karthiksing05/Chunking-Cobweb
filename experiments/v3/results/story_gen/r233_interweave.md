# r233: does interweaving chunks and concepts give correct NEW combinations? (controlled grammar, exact judge)

Plan: docs/PLAN_INTERWEAVE.md. Grammar and judge: r230 (147 words; agreement, frames, selectional restrictions). n = 1000, seeds 13/17,
1600 draws per arm. Code: experiments/v3/diag/r233_{p1.sh,analyse.py,newonly.py,fact.py,purity.py,leafprobe.py,score.py,chunkctx.py,goldparse.py};
src/trellis2/trellis3.py additions (all opt-in, default behaviour unchanged): `_generate_tree(choose="bits", bits=..)`,
`word_bits=` (word parts filled from the WordTree class of the stored word), `compose()` (content-taxonomy composer, via
generate_tree(compose={...})), word fillers in sibling mode stand for a stored occurrence (bug fix: bare Elem had no ctx).
Driver r230_story.py: an arm may set fill_name (default unchanged).

Metric NCh = distinct, R-sem draws containing a head pair (subject-verb, verb-object, adjective-noun) never in training = correct new
combinations. Fresh grammar samples: NCh .137 over all draws (.157 among new). P = among draws with an unseen head pair (adjectives
that accept every noun excluded), share R-sem; a random same-category swap is legal ~.34.

## Results
1. Paper generator W (window 5, exact emission, stored-pair gate): sem .72-.73, fair count .29-.30, NCh .004 (both seeds). A
   stored-adjacent-pair generator cannot form an unseen adjacent head pair by construction.
2. Tree generator (v1-style expansion, SENT -> LPART/RPART -> FILL) at the context leaf: new share .47-.48; among new draws sem .61-.66,
   fair count .29-.31, NCh .034-.036 (9x W). Every read above the leaf (hops 1-6, bits 4-14, best, match, backoff, kind, sibling 1-4,
   sibling pair, ctx_fill) gives 40-100% nonsense. Content-taxonomy composer: the same (leaf fine, nonsense above ~12 bits).
3. Purity (r233_purity, r233_leafprobe): token context leaves hold distributionally interchangeable occurrences ("the ___ </s>" = all
   nouns), but the window never carries the verb, so concepts above the leaf mix semantic types; content concepts mix category
   signatures above ~12 bits. Window 1-3 does not fix it.
4. WordTree classes in the tree generator (word_bits): nonsense .2-.65, NCh <= .029.
5. Parse score as a critic of W (offline, r233_score): AUC sem vs object violation .73 full, .76 chunks off, trigram .85; best of 4:
   .894 / .892 / .957 (random .728). Composition is no better a critic than a trigram.
6. Factorial on "all" WordTree classes (r233_fact; emission exact/b7/b6 x gate none/word pairs/class pairs): best NCh .038-.044
   (class emission + class gate), P <= .135 in every arm, i.e. below a random same-category swap. No clean interaction.
7. Chunk context for word concepts (r233_chunkctx): learned trees bracket the object NP in 100% of training sentences (both seeds).
   Semantic purity of type classes at b=7: chunk context .679/.692 vs windows .58-.64, right-branching .60-.62, random trees .59-.67.

## Verdict
NO-GO for a generation showcase of interweaving on this grammar with the current representations: TRELLIS's own concepts do not
carry selectional structure above the leaf, so no read-out that generalizes through them is both new and correct, and the best
class-based arms make new combinations less accurately than random same-category substitution.
Kept: the learned parser recovers the object NP exactly (gold-parse curves in r233_goldparse, pending); the tree generator at the
leaf makes 9x more correct new head pairs than the window generator at equal fair count.

## Gold-parse results, FIRST PASS (SUPERSEDED: held set overlapped training; see "Review" below)
Gold constituents of the grammar: subject NP, VP, PP, object NP (flat NPs, so [Adj N] is not gold and costs nothing in recall; it
counts as a non-crossing extra bracket). Recall / exact at n = 125 ... 4000, seeds 13/17/19/23 (29/31 running):

| arm | n=125 recall | n=1000 recall | n=4000 recall | n=4000 exact |
|---|---|---|---|---|
| TRELLIS (full) | .77/.79/.92/.84 | .92/.93/.97/.85 | .96/.93/.97/.92 | .85/.72/.90/.70 |
| chunks off (store none) | .34/.67/.34/.70 | .61/.74/.61/.64 | .59/.85/.60/.65 | .20/.73/.27/.50 |
| concepts >= 3 bits only | .54/.60/.48/.35 | .75/.78/.79/.55 | .84/.85/.85/.83 | .56/.62/.69/.64 |
| concepts >= 6 bits only | .32/.33/-/.31 | .38/.37/-/.36 | .62/.65/-/.58 | .28/.38/-/.30 |
| concepts >= 9 bits only | .33/.33/-/- | .33/.33/-/- | .33/.36/-/- | 0 |
| concepts off (leaf only) | .30-.33 | .30-.33 | .30-.33 | 0 (constant tie-break tree) |
| right-branching | | | .52-.57 | 0 |
| left-branching | | | .43-.48 | .18-.25 |
| random binary | | | .40-.42 | .13-.17 |

Readings.
- Composition recovers the grammar's constituents without supervision: 92-97% of gold constituents, 70-90% of held-out sentences
  exactly right at n = 4000; right-branching gets none exactly right.
- Stored chunks matter: storing nothing costs .08-.37 recall and .00-.63 exact at n = 4000 (s17 is the near-tie).
- General concepts matter, and they matter as data efficiency: with all concepts TRELLIS is at .77-.92 after 125 sentences; with only
  concepts at >= 6 bits it stays at the tie-break floor until n = 1000 and reaches .58-.65 at 4000; >= 9 bits never leaves the floor.
  Each step of generality removed costs roughly an order of magnitude of data. Concepts off is the limit: no candidate ever repeats
  exactly, every score ties, and the parser returns a constant shape.
- Learned trees bracket the object NP in 100% of training sentences (r233_chunkctx).

## Review (Opus, 2026-11-26) and fixes
Three confirmed problems with the first pass:
1. The held-out set overlapped training: r230_grammar.load_corpus computes `seen` but never filters on it. 22% (s13) / 17% (s17)
   of held sentences occur in the 4,000-sentence stream, 95-98% of three-word ones. TRELLIS at n=500 s13: seen sentences exact .976,
   unseen .366. Fix: r233_goldparse.held_unseen (first 500 held sentences absent from the whole 4,000 stream); every r233 parse
   score is now on that set, outputs in results/story_gen/r233/unseen/. Pooled bracket precision and F1 added.
2. The CCM baseline was initialised uniformly over bracketings (all ratios 1), not with Klein & Manning's split-uniform prior,
   and fell into an anti-gold optimum ([V Det], [V Prep]). Fixed in r233_ccm.split_stats. Fixed CCM on words reaches recall
   .90-.92 at n=4000 (review run, s13/17), so it is a strong baseline, not a weak one. The uniform init's MAP objective is higher
   than the split init's on words (-224573 vs -224859 at n=1000), so the initialisation is reported.
3. noconcept, neither and minbits >= 9 return one constant tree: every candidate gain is exactly 0 (the leaf's share term costs
   ~11 bits and the gain is clamped at 0), so the posterior is uniform and MBR returns a fixed shape that never brackets the subject.
   These arms measure a parser with no signal; minbits:6 is constant up to n=1000. They show that the parse signal is carried by
   general concepts (only they save more bits than they cost to name), but they are not a fair "memory without concepts".
   Added a non-degenerate chunks-without-concepts baseline: r233_exactchunk.py (CKY over exact word-sequence counts, or PMI).
Smoke run, s13, unseen set: exact-chunk count recall .59/.53/.52 at n=125/1000/4000 (falls with data), PMI .21/.18/.15; fixed CCM
.75 at 125, .89 at 1000; TRELLIS full .77 at 125.

## Gold-parse results on unseen sentences (FINAL; results/story_gen/r233/unseen; 6 seeds 13/17/19/23/29/31; 500 held sentences each,
none in the 4,000-sentence stream). Range over seeds (mean). R = recall of gold constituents, X = exact sentences, F1 = pooled bracket F1.

| parser | n=125 R / X / F1 | n=1000 R / X / F1 | n=4000 R / X / F1 |
|---|---|---|---|
| TRELLIS | .73-.90 (.78) / .42 / .70 | .83-.97 (.91) / .68 / .81 | .90-.97 (.93) / .63-.89 (.75) / .79-.88 (.83) |
| TRELLIS, chunks off | (.67) / .30 / .63 | (.71) / .40 / .67 | .51-.84 (.72) / .45 / .68 |
| TRELLIS, concepts >= 3 bits | (.59) / .22 / .57 | (.74) / .38 / .71 | .80-.87 (.83) / .57 / .78 |
| TRELLIS, concepts >= 6 bits | (.36) / 0 / .34 | (.42) / .02 / .40 | .68-.75 (.72) / .38 / .69 |
| TRELLIS, concepts >= 9 bits | (.37) / 0 / .34 | (.37) / 0 / .34 | (.39) / 0 / .37 |
| TRELLIS, concepts off | (.37) / 0 / .34 constant tree | same | same |
| exact-chunk memory (count CKY) | (.58) / .21 / .54 | (.53) / .15 / .49 | .50-.54 (.52) / .10 / .48 |
| exact-chunk memory (PMI CKY) | (.20) / .01 / .18 | (.17) / 0 / .16 | (.16) / .01 / .14 |
| CCM, words (split init, Viterbi) | .78-.84 (.80) / .51 / .75 | .88-.90 (.89) / .64 / .79 | .89-.91 (.90) / .63-.67 (.65) / .79-.81 (.80) |
| CCM, words, MBR | (.80) / .51 / .74 | (.89) / .64 / .79 | (.90) / .65 / .80 |
| CCM, gold POS tags | (.92) / .69 / .82 | (.92) / .69 / .82 | (.92) / .69 / .82 |
| right-branching | | | .62-.65 (.64) / 0 / .62 |
| left-branching | | | (.36) / .08 / .29 |
| random binary | | | (.39) / .10 / .35 |

Paired per seed (F1, TRELLIS minus CCM on words): n=125 -.10..+.04 (CCM ahead 5/6); n=500 -.05..+.02; n=1000 -.04..+.08 (TRELLIS
ahead 5/6); n=2000 5/6; n=4000 +.005..+.081 (6/6). Exact: n=4000 -.03..+.23 (5/6). TRELLIS at 4,000 on words matches CCM on gold tags.
Chunks (full minus chunks off, F1 at 4000): +.026..+.384, 6/6 seeds.

Readings.
- One online pass of TRELLIS recovers the constituents of sentences it has never seen: .93 recall, 75% exactly right, F1 .83 at
  4,000 sentences, ahead of CCM (20 EM passes) from 1,000 sentences on, and level with CCM given gold tags. CCM learns faster early.
- Chunks without concepts do not parse: an exact-sequence memory scores below right-branching and gets worse as data grow (.58 ->
  .52), because new sentences rarely repeat their constituents exactly while frequent cross-constituent pairs (V Det) do recur.
- Concepts without chunks parse, but worse: removing the stored chunks costs .22 recall and .15 F1 (6/6 seeds).
- Inside TRELLIS the signal comes from general concepts: restricting the parse to concepts at >= 3 bits costs .10 recall, >= 6 bits
  leaves the parser at the zero-signal floor until 1,000 sentences, >= 9 bits never leaves it. The zero-signal floor (every gain 0,
  constant tree) is what a code-saving score gives when no concept is both cheap to name and predictive.
