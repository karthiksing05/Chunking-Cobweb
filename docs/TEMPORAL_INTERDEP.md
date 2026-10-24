# Temporal and interdependent representations

Goal (user, 23 October 2026): remove two inconsistencies of incremental representation learning in TRELLIS v2, and keep iterating until both are confidently satisfied.

1. **Representations are not temporal.** A stored description reflects the trees at the time it was stored. It can be brought up to date by reassignment, which may require broad restructuring of the tree. Reassignment must stay incremental: no night, no batch rebuild.
2. **Representations are not interdependent.** A word's context is its raw neighbour tokens, so its representation does not use what has been learned about its neighbours. Isolated context means slower convergence.

## How each is measured (`logs/v3_diag/t3_props.py`)

| Property | Measure | Satisfied when |
|---|---|---|
| temporal | **stale**: share of stored records whose stored description differs from the current one (content; context) | low, and not growing with the corpus |
| temporal | **drift**: mean total-variation distance between stored and current descriptions | near 0 |
| temporal | **misplaced**: share of stored leaves not under the concept their current description sorts to | low |
| temporal | **name churn**: TV distance of the 30 most frequent words' names between checkpoints (20, 40, …, 640 experiences) | falls as learning proceeds (names settle) |
| interdependent | toy a/the (`t3_athe.py`): dog (only after *a*) and cat (only after *the*) share a representation; the noun~verb control does not | dog~cat high, dog~saw near 0 |
| interdependent | **convergence**: a word's nearest word by name overlap has its gold category (gold only scores), at checkpoints and by how often the word was seen | high early and for rarely seen words |
| (guard) | generation, grammatical and new (`t3_genbench.py`, leaf rule) | not worse |

## Round 1: stable symbols (information bands), sampled restructuring, reassignment (seed 13)

`bands`: a path is named by the first concept reaching each of 1, 2, 4, 6, 8, 12 bits (−log2 share of the tree), instead of every concept on the path (`ipath`). `rs`: sampled restructuring. `refit`: reassignment (unfit + ifit with the current description) for the newest records and 300 older ones per experience, instead of relabelling in place.

| med | base | rs | bands + rs | bands + rs + refit |
|---|---|---|---|---|
| stale content / context | 0.96 / 0.75 | 0.87 / 0.25 | 0.77 / **0.12** | 0.75 / 0.17 |
| drift content / context | 0.117 / 0.034 | 0.054 / 0.005 | 0.062 / 0.008 | 0.060 / 0.014 |
| misplaced context / content | 0.70 / 0.87 | 0.61 / 0.62 | 0.62 / 0.66 | **0.48 / 0.34** |
| name churn (40 → 320) | 0.55 → 0.59 | 0.47 → 0.57 | **0.38 → 0.41** | 0.44 → 0.38 |
| word category accuracy (20 / 80 / 320 exp.) | 0.71 / 0.77 / 0.73 | 0.81 / 0.73 / 0.73 | 0.86 / 0.68 / 0.77 | 0.91 / 0.73 / 0.86 |
| learning seconds | 847 | 355 | **209** | 1204 |
| generation, grammatical / gram+new | 1.00 / 0.88 | 0.96 / 0.86 | 0.98 / **0.92** | see below |

Toy a/the, name overlap (seeds 13 / 17): base dog~cat 0.26 / 0.15; bands + rs 0.31 / 0.48 (control dog~saw 0.20 / 0.00); bands + rs + refit 0.48 / 0.38 (control 0.15 / 0.00). Tree sharing two levels up stays about 0.

Readings:
- Bands make names much less volatile: context staleness 0.75 → 0.12 and churn 0.55 → 0.38 on `med`, learning 4× faster, generation as good or better on `med` (0.92 new) and `large` (0.99 / 0.78), lower novelty on `small` (0.39 against 0.50).
- Content descriptions stay 75–77% stale by exact comparison, but their drift is small (TV 0.06): they change a little, often.
- Reassignment halves the misplaced leaves (content 0.66 → 0.34) but costs 6× the time.
- **Bug found:** with refit, generation produced nothing. Generation statistics (FILL, LPART/RPART, SENT) named parts by leaf id and sat on the path of the leaf where they were added, so moving a record orphaned them. Fixed: statistics now name records (`rid`, append-only and stable), and each record keeps the passive counts it contributed (`Elem.pas`), which `_move` carries to its new leaf. `t3_case.py` and `t3_export.py` still read leaf ids and need updating.
- Neither property is satisfied yet: misplacement stays at 0.3–0.6 and the context window is still raw tokens.

## Round 2: interdependent context (seed 13; toy seeds 13 / 17)

On the bands + sampled-restructuring base. `nt`: the context window's adjacent slots (`lc`/`rc`) carry each neighbour's learned name, its type bag of context concepts read through the current tree (`neighbour_types`). `nocc`: a word's own context description no longer carries its token (`word_cc` off); the token still identifies it as a word in generation (FILL).

| toy a/the | dog~cat overlap | dog~cat shared at u2 | dog~saw control |
|---|---|---|---|
| current setup | 0.26 / 0.15 | 0.01 / 0.01 | 0.00 / 0.02 |
| bands + rs | 0.31 / 0.48 | 0.00 / 0.02 | 0.20 / 0.00 |
| bands + rs + nocc | 0.25 / 0.48 | 0.28 / 0.04 | 0.04 / 0.08 |
| nt | 0.34 / 0.50 | 0.00 / 0.01 | 0.20 / 0.00 |
| **nt + nocc** | **0.52 / 0.50** | **0.21 / 0.16** | **0.00 / 0.00** |

| | small | med | large |
|---|---|---|---|
| grammatical / gram+new, current | 1.00 / 0.50 | 1.00 / 0.88 | 1.00 / 0.79 |
| nt | 1.00 / 0.38 | 0.94 / 0.82 | 0.99 / 0.78 |
| bands + rs + nocc | 1.00 / 0.47 | 0.86 / 0.82 | 0.98 / 0.94 |
| nt + nocc | 1.00 / 0.52 | 0.85 / 0.80 | 0.96 / 0.90 |
| temporal, nt + nocc (small / med): stale context | 0.009 / 0.14 | | |
| temporal, nt + nocc: name churn 40 → 320 | 0.54 → 0.32 / 0.44 → 0.36 (settling) | | |

Readings:
- **Interdependence on the toy is reached by `nt + nocc`:** dog and cat, never seen after the same word, share concepts two levels up on both seeds, and the noun~verb control is 0. Neighbour names alone (`nt`) do not do it, because the word's own token still dominates its context leaf; removing the token alone does it on one seed.
- Generation keeps the trade-off seen with `word_cc` off: large gains novelty (0.79 → 0.90), med loses grammaticality (1.00 → 0.85) to whole chunks in the wrong role.
- The convergence measure is uninformative on these corpora (`med` has 22 word types, all seen 8+ times by experience 160); a larger-vocabulary condition is needed.

## Round 3: lazy repair (seed 13), on `nt + nocc`

`refresh="drift"` (from the notes' lazy repair, drift as the signal): a stored record is reassigned (unfit and re-inserted with its current description, its passive counts moving with it) when its description drifted by more than τ (mean total variation over attributes); smaller drift is relabelled in place. `refit` reassigns on any change.

| | small: learn s / misplaced ctx, cnt / gram, gram+new | med: learn s / misplaced ctx, cnt / gram, gram+new |
|---|---|---|
| relabel | 42 / 0.34, 0.54 / 1.00, 0.52 | 360 / 0.60, 0.64 / 0.85, 0.80 |
| drift τ = 0.15 | 102 / 0.59, 0.46 / 1.00, 0.59 | 655 / 0.65, 0.50 / 0.81, 0.76 |
| drift τ = 0.05 | 378 / 0.26, 0.48 / 1.00, 0.55 | 1547 / 0.55, 0.56 / 0.85, 0.81 |
| refit | 198 / **0.16, 0.09** / 1.00, 0.55 | 1296 / 0.53, 0.32 / 0.84, 0.81 |

Large (gram / gram+new): relabel 0.96 / 0.90, τ 0.15 0.94 / 0.89, τ 0.05 0.95 / 0.91, refit 0.92 / 0.86.

- Reassignment makes `small` nearly current (refit: 0.16 / 0.09 misplaced) and raises novelty there (0.52 → 0.55–0.59).
- On `med`, even refit leaves half of the context leaves misplaced: refit fires on description changes, while most misplacement comes from the tree restructuring under unchanged descriptions.
- The misplacement measure (one level above the leaf) is stricter than what representations use. `t3_props.py` now also reports **misplacement at the information bands**: the share of band levels at which a stored leaf's concept differs from the one its current description sorts to. That is the criterion for "representations follow the current trees", since names are read at the bands.

## The med grammaticality cost: words at shared leaves, and words named by their representation

`logs/v3_diag/t3_purity.py` (`med`, seed 13): share of stored records (weighted) at a leaf whose records do not all share their gold category (gold only scores). Generation's leaf rule draws a slot's fillers from the part's context leaf, so a mixed leaf lets one category stand in for another.

| | context, composites | context, words | content |
|---|---|---|---|
| current | 0.006 | 0.000 | 0.000 |
| bands + rs | 0.013 | 0.000 | 0.001 |
| bands + rs + nocc | 0.009 | **0.044** | 0.000 |
| nt + nocc | 0.012 | **0.043** | 0.000 |
| name | 0.008 | 0.000 | 0.001 |
| nt + name | 0.014 | 0.000 | 0.001 |

Without the token, 4% of words share an exact-match leaf with a word of another category (identical two-word surroundings). Over the six or so word slots of a sentence that accounts for much of the 15-point drop.

`word_cc="name"`: a word's own-content facet is its learned type name (the bag of context concepts where its type stood, read through the current tree), not its token. Words are identified by their representation; the name is relabelled as the tree changes.

| seed 13 | toy dog~cat overlap / shared at u2 / control (s13, s17) | small | med | large |
|---|---|---|---|---|
| current | 0.26, 0.15 / 0.01, 0.01 / 0.00, 0.02 | 1.00 / 0.50 | 1.00 / 0.88 | 1.00 / 0.79 |
| nt + nocc | 0.52, 0.50 / 0.21, 0.16 / 0.00, 0.00 | 1.00 / 0.52 | 0.85 / 0.80 | 0.96 / 0.90 |
| name | 0.43, 0.50 / 0.01, 0.00 / 0.00, 0.13 | 1.00 / 0.53 | 1.00 / 0.79 | 0.90 / 0.68 |
| nt + name | 0.46, 0.48 / 0.00, 0.00 / 0.00, 0.00 | 1.00 / 0.44 | 0.97 / 0.78 | 0.99 / 0.80 |

- Naming words by their representation removes the mixed word leaves and restores `med` grammaticality (0.85 → 0.97–1.00), and dog and cat still share most of their names with a clean control.
- They no longer share a tree concept two levels up, and `med` novelty is lower than the current setup's (0.78 against 0.88). Seed 17 is running to separate this from noise.

## Round 4: two seeds, and misplacement at the information bands

Generation, grammatical / gram+new, seed 13 then seed 17:

| | small | med | large | learn s (med) |
|---|---|---|---|---|
| current | 1.00/0.50, 1.00/0.47 | 1.00/0.88, 0.99/0.73 | 1.00/0.79, 1.00/0.83 | 1114 |
| **bands + rs** | 1.00/0.39, 1.00/0.45 | **0.98/0.92, 1.00/0.91** | 0.99/0.78, 0.97/0.87 | **347** |
| nt + nocc | 1.00/0.52, 1.00/0.47 | 0.85/0.80, 0.85/0.79 | 0.96/0.90, 0.94/0.90 | 341 |
| name | 1.00/0.53, 1.00/0.41 | 1.00/0.79, 0.81/0.69 | 0.90/0.68, 0.98/0.81 | 502 |
| nt + name | 1.00/0.44, 1.00/0.45 | 0.97/0.78, 0.98/0.75 | 0.99/0.80, 1.00/0.81 | 480 |

Misplacement at the information bands (share of band levels where a stored leaf's concept is not the one its current description sorts to), context / content, seed 13:

| | small | med | learn s (med) |
|---|---|---|---|
| current | 0.13 / 0.40 | 0.14 / 0.40 | 1179 |
| bands + rs | 0.13 / 0.46 | 0.17 / 0.41 | 279 |
| nt + nocc | 0.13 / 0.37 | 0.16 / 0.34 | 353 |
| nt + name | 0.13 / 0.32 | 0.20 / 0.42 | 358 |
| nt + nocc + refit | **0.07 / 0.06** | **0.13 / 0.19** | 1422 |
| nt + name + refit | 0.03 / 0.18 | 0.12 / 0.26 | 945 |

- Bands + sampled restructuring is a gain on its own: best `med` generation on both seeds, learning 3–4× faster.
- Read at the bands, context names are about 85% current in every variant; content names about 60%. Reassignment brings them to 94% (small) and 81% (med) at 3–4× the cost. nt + name + refit collapses novelty on `large` (0.95 / 0.25) and is set aside.
- The interdependence trade-off holds across seeds: nt + nocc links dog and cat in the tree but costs `med` grammaticality (0.85); nt + name keeps grammaticality and loses the tree-level link.
- Hypothesis for nt + name: the name bag carries the fine bands (8–12 bits), specific to each word. Round 5 caps a word's own-content name at its coarse bands (`word_name_bits` 4 or 6), its category-level representation.

## Round 5: a word's identity at the category level (coarse bands only)

`word_name_bits` caps a word's own-content name at the bands up to 4 or 6 bits. Seeds 13 / 17.

| | toy dog~cat overlap / shared at u2 / control | small | med | large |
|---|---|---|---|---|
| nt + name4 | 0.38, 0.49 / 0.00, 0.00 / 0.00, 0.00 | 1.00/0.46, 1.00/0.44 | 0.99/0.89, 0.95/0.82 | 0.97/0.81, 0.98/0.81 |
| nt + name6 | 0.34, 0.38 / 0.00, 0.00 / 0.00, 0.09 | 1.00/0.42, 1.00/0.44 | 0.97/0.83, 0.94/0.79 | 1.00/0.87, 1.00/0.77 |

Word leaves stay almost pure (`med`, mixed 0.002 / 0.000).

**Why the tree-level link is lost, whatever the resolution:** any facet specific to a word separates its context instances. Its token does, and so does its own name, because dog's name is built from the leaves where dog stood, which steers dog's new instances back to those leaves (a self-reinforcing loop). In nt + nocc nothing word-specific is in the description, and the neighbours' learned names (a ≈ the) carry the link.

**So the remaining confusion belongs to the choice, not to placement.** The 4% of words at mixed leaves in nt + nocc have identical surroundings two words on each side: within that window they are truly ambiguous. Round 6 keeps the interdependent tree (nt + nocc) and disambiguates when generation fills a word slot: each candidate word drawn from the leaf is weighted by how much its learned name overlaps the name of the word stored in the slot (`generate_tree(fill_name=True)`, genbench option `fillname`). The tree, and with it the toy results, are unchanged.

## Round 6: what actually goes wrong in `med` (nt + nocc)

- Name-weighted word filling (`fill_name`, context names): `med` 0.86 / 0.78 and 0.87 / 0.80 (seeds 13 / 17) against 0.85 / 0.80 and 0.85 / 0.79 without. No effect.
- Content leaves do not mix differently built composites (`t3_purity.py`, chosen composites at content leaves with mixed part categories: 0.0 in every variant).
- `logs/v3_diag/t3_generr.py` traces ungrammatical generations step by step (slot record, its words and gold label; what was drawn). Nearly every failure is **a verb slot filled with a preposition or the reverse**: slot [saw]:V ← "under", "on", "in"; slot [under]:P ← "found". Verbs and prepositions both stand between two noun phrases ("the man [saw] the woman", "the man [in] the park"): identical two-word surroundings and near-identical context names, so neither the context tree nor the context name can tell them apart.
- What separates them is what they build (the AND side): a verb heads the predicate beside the subject, a preposition heads a phrase attached inside a noun or verb phrase. This is the user's "identified by their representation in content and context": context alone cannot do it.
- Round 7: keep the interdependent tree (nt + nocc), maintain each word's build-role reservoir without putting it in any description (`word_content="track"`), and weight a word slot's candidates by the overlap of their build roles with the slot word's (`fill_name="build"`, genbench option `fillbuild`).

Discrete tree profile (40 sentences of `small`): 10.8 s Python tree, 11.3 s discrete. The wrapper is not the bottleneck at that scale; the 12× slowdown over the full 320 sentences comes from how the discrete tree grows. To profile at a longer run.

## Rounds 7–8: disambiguating word slots by representation (nt + nocc tree, unchanged)

`word_content="track"` keeps each word's build-role reservoir (content leaves of the composites it is a direct part of) without writing it into any description. `fill_name`: a word slot's candidates (drawn from the part's context leaf) are weighted by overlap with the slot word's representation: `"build"` what it builds, `"both"` the product of build-role and context-name overlaps.

| gram / gram+new, seeds 13, 17 | small | med | large |
|---|---|---|---|
| current | 1.00/0.50, 1.00/0.47 | 1.00/0.88, 0.99/0.73 | 1.00/0.79, 1.00/0.83 |
| nt + nocc | 1.00/0.52, 1.00/0.47 | 0.85/0.80, 0.85/0.79 | 0.96/0.90, 0.94/0.90 |
| nt + nocc, fill by build role | 1.00/0.49, 1.00/0.51 | 0.90/0.85, 0.92/0.85 | 0.96/0.88, 0.98/0.91 |
| nt + nocc, fill by both | 1.00/0.43, 1.00/0.54 | **0.92/0.86, 0.94/0.83** | 0.97/0.87, 0.98/0.92 |

The build role recovers about half the `med` loss. The remaining errors (28 of 300, `t3_generr.py` with `build`) are still verb/preposition swaps. The learner analyses *saw a man* and *under a red cat* alike (a word plus a noun phrase), so the two words build the same things one level up, and the composites [saw the man] and [under the man] have the same two-word surroundings (*the man ___ </s>*). They differ three words back: the verb phrase follows the subject at the start of the sentence, the prepositional phrase follows a verb. It is a two-word-window ambiguity that the token used to hide by memorising which word stands in each slot. Round 9 tests a three-word window (`window=3`) on nt + nocc with both-fill, and the current setup at `window=3` for comparison.

## Round 9: a three-word window (rejected)

| gram / gram+new, seeds 13, 17 | small | med | large |
|---|---|---|---|
| nt + nocc, both-fill, window 3 | 1.00/0.54, 1.00/0.39 | 1.00/0.59, 1.00/0.61 | 1.00/0.24, 1.00/0.23 |
| current setup, window 3 | | 1.00/0.36, 0.98/0.30 | |

Toy dog~cat at window 3: overlap 0.42 / 0.36, shared at u2 0.00 / 0.00. A wider window removes the verb/preposition swaps but makes contexts nearly unique, so the trees memorise and the interdependent link is lost. The window stays at two words.

## Round 10: reassignment only where the record is misplaced (`refresh="verify"`), seed 13

On nt + nocc + `word_content="track"` + both-fill. A record is moved (unfit, re-ifit with its passive counts) only when the concepts at its information bands differ from where its current description sorts now. `verifyN`: up to N records checked per sentence.

| seed 13 | misplaced at bands ctx / cnt, small | med | learn s small / med | gram / gram+new small | med | large |
|---|---|---|---|---|---|---|
| relabel only (round 4) | ~0.13–0.16 / 0.34–0.37 | | ~50 / ~120 | 1.00/0.43 | 0.92/0.86 | 0.97/0.87 |
| verify 300 | 0.02 / 0.14 | 0.065 / 0.27 | 1010 / 1331 | 1.00/0.51 | 0.89/0.85 | 0.996/0.89 |
| verify 100 | 0.07 / 0.28 | 0.08 / 0.24 | 458 / 907 | 1.00/0.52 | 0.90/0.82 | 0.986/0.85 |

Verify reassignment makes context representations temporal (2–8% of records sit where they would no longer sort) without hurting generation; content misplacement halves. The cost is 10–20× the learning time on `small`; it is the knob that trades currency for speed.

## Characters (seed 13): the candidates on a second domain

well formed / attested / real / novel:

| | 250 | 500 | time at 500 |
|---|---|---|---|
| current | 0.994 / 0.980 / 0.824 / 0.227 | 0.928 / 0.908 / 0.688 / 0.357 | 7523 s |
| bands + sampled restructuring | 0.972 / 0.940 / 0.801 / 0.244 | 0.930 / 0.897 / 0.618 / 0.429 | 1491 s |
| nt + nocc + both-fill | 0.985 / 0.982 / 0.753 / 0.290 | 0.921 / 0.904 / 0.635 / 0.430 | 1180 s |

Both candidates keep well-formedness and attestedness, generate more novel characters at 500, and run 5–6× faster. They produce fewer real (dictionary) characters at 500 (0.62–0.64 against 0.69).

## Where this stands

- Temporal: yes, with `refresh="verify"` (band-level misplacement 2–8% context); cheap modes (relabel) keep names current but leave 13–16% of records misplaced.
- Interdependent: yes, with neighbour names and no word identity in context (toy dog~cat share a concept at u2, rare nouns cohere after 50 sentences against 400). Cost: `med` grammaticality 0.89–0.94 from verb/preposition swaps that a two-word window cannot separate.
