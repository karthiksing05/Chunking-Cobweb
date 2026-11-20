# Round 219: agreement-aware abstraction (oracle ceiling first). Verdict: NO-GO on the go rule; no TRELLIS mechanism was implemented

Everything below is diagnostic. spaCy (`en_core_web_sm`, env cobint) is used only to build oracles and purities; it is never part of the model.
Code: `experiments/v3/diag/r219_oracle.py` (+ `r219_oracle*.sh`, `r219_trees.sh`, `r219_wordtree.py`), `src/trellis2/trellis3_r219.py` (WordTree219, feature sets only).
Judge: r214_judge2 ensemble (Qwen3-8B + phi-4; syntax logit > -5.25, semantics logit > 2.5), thresholds unchanged. Inputs: the R-sem sentences of the gated token leaf (tau .7) arm
(n=1000 seed 13: 468 sentences; n=4000 seed 13: 697; n=4000 seed 17: 679), 3 draws each. R-sem/R-syn/R-non are shares of the oracle sentences.
Mapping to the architecture (user constraint, 2026-11-19): oracles map to nothing (they are cheat tests). The only code that could enter the model is `WordTree219`,
a different feature set for the existing word-category tree (same Cobweb tree, same level b read, same class emission `emit_bits`); no generation machinery was added.

## 1. Judge control (hand-built template sentences, no LLM; `r219/controls/`)
100 syntax-only (right frame, number agreement, a/an, absurd selection, e.g. "the boxes ate the flowers", "lily and anna ate the house"), 100 sensible, 99 scrambled/reversed/article-dropped.
| class | n | R-sem | R-syn | R-non | syntax pass | semantics pass |
|---|---|---|---|---|---|---|
| SYN | 100 | .50 | .50 | .00 | **1.00** | .50 |
| SEM | 100 | 1.00 | .00 | .00 | 1.00 | 1.00 |
| NON | 99 | .01 | .20 | .79 | .21 | .01 |
The judge passes syntax on every clean syntax-only sentence. R-syn is .5 only because half of my random templates came out sensible by chance ("lily took a sad dog", sem_p .64). So the judge is not
the blocker: it awards R-syn when syntax is clean and meaning absurd (already seen in r214_judge2: C2full R-syn .75 on real held-out sentences with every content word swapped inside tag + frame). Its known slack is on NON (.21 syntax pass of scrambled/article-dropped items; adjacent-swap C3adj .127).

## 2. Oracle table (class-aware swaps of the R-sem sentences)
Variants: pos = UPOS (r214's oracle, K=50 kept); fine = PTB tag_; morph = tag + Number/Tense/VerbForm + verb frame (dobj/dative/prt/ccomp/xcomp/acomp/attr/prep/...) + adjective role + noun determiner habit;
morph_ar = + a/an repair; morph1 = one swapped word; chunk = every spaCy noun chunk swapped for a stored chunk of the same category (root tag, role subj/obj/pobj, determiner kind, pronoun morph) + a/an repair;
_c = content words only (NOUN/VERB/ADJ/ADV; chunk_c: common-noun chunks only). K = number of most frequent training types kept unchanged. Candidates are non-top-K types with the same type-majority signature.
R-sem / R-syn / R-non (share of all oracle sentences; seed 13 n=1000 | seed 13 n=4000 | seed 17 n=4000):

| variant (K=50 unless stated) | n=1000 s13 | n=4000 s13 | n=4000 s17 |
|---|---|---|---|
| pos | .31 / .22 / .47 | .25 / .23 / .52 | .25 / .23 / .52 |
| fine | .37 / .26 / .38 | .31 / .29 / .40 | .29 / .30 / .42 |
| morph | .48 / .31 / .21 | .41 / .33 / .26 | .40 / .34 / .28 |
| morph_ar (c) | .50 / .29 / .21 | .43 / .33 / .24 | .39 / .35 / .27 |
| morph1 | .66 / .23 / .11 | .61 / .25 / .14 | .58 / .29 / .13 |
| chunk (spaCy NP) | .71 / .25 / .04 | .67 / .25 / .08 | .66 / .28 / .06 |
| morph_c (content only) | .52 / .31 / .17 | .47 / .33 / .19 | .46 / .36 / .18 |
| chunk_c | .80 / .17 / .03 | .74 / .20 / .06 | .76 / .20 / .05 |
| K=0, morph_c, N/V/A/ADV | .17 / .40 / .43 | .13 / .45 / .42 | .13 / .48 / .39 |
| **K=0, morph_c, N/V/ADJ only** | .29 / **.45** / .27 | .23 / **.50** / .28 | .24 / **.52** / .24 |
| K=0, morph_c1 (one word) | .57 / .32 / .11 | .54 / .35 / .11 | .55 / .35 / .10 |

Reading. (i) Syntax survives class-aware swaps: morph_c (K=0, N/V/ADJ) keeps syntax in .73-.76 of sentences (R-sem + R-syn). (ii) Finer classes help monotonically (pos .22 -> fine .26 -> morph .31 at n=1000), a/an repair is within noise.
(iii) The only oracle at the go-rule value is the one that swaps ALL nouns, verbs and adjectives (no frequent words kept; adverbs, function words and names kept): R-syn .45 / .50 / .52 (n=1000 / 4000 s13 / 4000 s17), with R-non .24-.28 and R-sem .23-.29.
Keeping the 50 most frequent types, or swapping a single word or chunk, leaves the sentence semantically acceptable in about half of the cases (a swapped "happy -> shiny" or "bird -> boy" is still sensible), so R-syn stays at .23-.36.
Chunk oracle: chunks are good at preserving meaning/syntax (R-sem .66-.80, R-non <= .08) and cannot produce R-syn (.17-.28 at most), because pronoun/name/common-noun swaps rarely break the semantic judge. The adverb swaps in the K=0 variant add the residual R-non (and ADV is position-free, so spaCy frames are no help).
(iv) Failure sample (K=0, ADV included): "tim and once found then all time", "she were high that he could find her", "at the fish i'm put a little toy" (adverbs, interjections and conjunction structure).
Files: `r219/o_*`, `r219c/o_*` (K=50 and content variants), `r219k0/`, `r219k0n/` (K=0), each with report.md, scores.tsv, variants.json.

## 3. Can an unsupervised word tree supply those classes? (tree classes inside the same oracle, content words N/V/ADJ, K=0, `r219t/`, `r219t2/`)
Classes = TRELLIS word-category tree (Cobweb over left/right neighbour bags, r214 `WordTree`) at level b, swapping within the class. n=4000, R-syn, seed 13 | seed 17:
| tree features | b=5 | b=6 | b=7 |
|---|---|---|---|
| all neighbours (r214) | .29 / .30 | .30 / .34 | .32 / .34 |
| func:100 (r214) | .24 / .27 | .32 / .29 | .32 / .32 |
| fn:25 | .26 / .30 | .29 / .33 | .27 / .32 |
| fn:25 + joint (left,right) function-word pair facet ("+lr") | .27 / .27 | .32 / .34 | .33 / .29 |
| fn:40 + two-away facets + pair facet | .19 / .30 | .31 / .38 | .21 / .33 |
vs the morph-signature oracle .50 / .52. The new feature sets (WordTree219: fn:K, +d2 = words two away, +lr = joint left/right function-word pair) do not beat the r214 trees beyond seed noise
(best cell .38, one seed). Morph-signature purity of class members without the judge (`r219w/purity_s*_n4000.md`, b=6, candidates sharing the token's signature): all .40, func100 .39, fn15 .48, fn25 .50, fn25+lr .51 (seed 17; seed 13: .37, .38, .36, .39, .44).
Purity ~.4-.5 is the same number at every feature set, i.e. the tree levels (25 classes at b=5, ~45 at b=6) mix tags and frames; the classes the oracle needs are about 50-100 morphosyntactic cells of a few words, which none of the b levels isolates.
Moving from the r214 emission (K=50, all tokens, R-syn .25-.27) to content-words-only with no frequency cut already lifts the tree oracle from .18-.21 to .30-.34 (the same trees); the K=0 content oracle (.50) is the ceiling.

## 4. Go / no-go
- R-sem >= .6: met earlier (gated token leaf tau .7: .63). R-non >= .6: met (root/coarse emission .87-.99). R-syn >= .5: **not met by any generator** (best generated .265, Addendum 7). The oracle reaches .50-.52 only with the full content-word swap inside exact morphosyntactic cells; every unsupervised tree class tried reaches .30-.38.
- Steps 2 (mechanism inside TRELLIS) and 3 (W' x b grid, learning curves) and 4 (showcase) were NOT run: the tree-feature oracle (Step 2's gate) stayed below .5, so there is no mechanism for the W' ablation to act on; running W' on the existing mechanism would reproduce Addendum 3/7 (R-syn flat .08-.27 from leaf to root).
- What the ceiling says: R-syn is reachable by the judge (controls) and by a signature-exact oracle (.5), so the blocker is the class quality (tree levels do not isolate agreement/frame cells; best purity .5) and the need to replace ALL content words (not frequent-type exceptions). A representation that gives classes at the grain of ~50 morphosyntactic cells (e.g. word-tree features that include the word's own affix/ending, which a raw-token model could get from sub-word characters, or a type-level feature of the typical following function word with the preceding one) is the next test, and would be a feature change to WordTree only. Not run.
- Caveat: the oracle sentences are the R-sem set of one generator; a chunk oracle with TRELLIS constituents (not spaCy NP chunks) was not built; K=0 swaps all content words so R-sem at that setting is only .13-.29.
