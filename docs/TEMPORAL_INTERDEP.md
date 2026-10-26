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

## Correction (2026-10-24): the neighbour-type facet was blank in rounds 2–10

`neighbour_types` read the full type aggregate (`type_bag`), which is only filled when `type_names` is not `"sample"`. Every configuration here used `type_names="sample"`, so `lc`/`rc` were blank for every record (`logs/v3_diag/t3_ntcheck.py`: 1000 of 1000 blank). Everything labelled "nt + nocc" above is in fact **nocc**: the word's identity left out of its context, with bands and sampled restructuring. The dog~cat link and the faster rare-noun cohesion come from that. Dog and cat are described only by their surface windows, which they share. Neighbours' learned representations played no part, so these results do not show interdependence as defined in the criteria.

Fix: `_nb_name` reads a neighbour's sampled type name (its reservoir of context leaves through the current tree), and `_refresh` rewrites `lc`/`rc` from the stored neighbour words, so the facet stays current. Round 11 reruns the toy, rare-word, generation and temporal tests with the facet on (`logs/v3_nbfix_submit.sh`).

## Round 11: the neighbour facet actually on

gram / gram+new, seeds 13, 17 (bands + sampled restructuring + nocc + track + both-fill; `nb_bits` keeps a neighbour's name to its coarse bands):

| | small | med | large |
|---|---|---|---|
| current setup | 1.00/0.50, 1.00/0.47 | 1.00/0.88, 0.99/0.73 | 1.00/0.79, 1.00/0.83 |
| nocc (rounds 2–10 "nt + nocc") | 1.00/0.43, 1.00/0.54 | 0.92/0.86, 0.94/0.83 | 0.97/0.87, 0.98/0.92 |
| neighbour names, all bands | 1.00/0.05, 1.00/0.06 | 1.00/0.08, 1.00/0.00 | 0.99/0.07, 1.00/0.02 |
| neighbour names, ≤ 4 bits | 1.00/0.10, 1.00/0.12 | 1.00/0.09, 1.00/0.06 | 0.99/0.17, 1.00/0.08 |
| neighbour names, ≤ 2 bits | 1.00/0.36, 1.00/0.36 | 0.99/0.41, 1.00/0.58 | 1.00/0.62, 1.00/0.44 |

Toy: dog~cat sharing at u2 is 0.000–0.003 with neighbour names, against 0.21 / 0.16 for nocc; the two words share only at u3. In the rare-noun test every variant puts 93–100% of nouns nearest a noun from 50 sentences on, so that test no longer separates them.

Learned neighbour names remove the `med` verb/preposition swaps: grammaticality returns to 0.99–1.00. They also make contexts more specific, so the learner memorises more. Novelty drops sharply with finer names; at 2 bits it is 0.36–0.62, against 0.43–0.92 for nocc. Interdependence by construction costs novelty, and the toy link loses its u2 level. The trade-off is still open.

## Round 12: neighbour names for rare types only (brief T6)

`nb_rare=n`: an element gets its neighbours' learned names only while its own type has been seen fewer than n times; after that its own window describes it. The facet follows refresh, so it switches off as a type becomes frequent.

gram / gram+new, seeds 13, 17:

| | small | med | large |
|---|---|---|---|
| current setup | 1.00/0.50, 1.00/0.47 | 1.00/0.88, 0.99/0.73 | 1.00/0.79, 1.00/0.83 |
| neighbour names for every word, ≤ 2 bits | 1.00/0.36, 1.00/0.36 | 0.99/0.41, 1.00/0.58 | 1.00/0.62, 1.00/0.44 |
| n = 10, ≤ 2 bits | 1.00/0.41, 1.00/0.40 | 0.99/0.71, 0.99/0.73 | 0.99/0.72, 1.00/0.65 |
| n = 10, all bands | 1.00/0.42, 1.00/0.39 | 0.99/0.65, 0.94/0.60 | –, 1.00/0.53 |
| n = 3, ≤ 2 bits | 1.00/0.48, 1.00/0.19 | 0.94/0.79, 0.95/0.74 | 0.98/0.73, 1.00/0.72 |

Word-category accuracy on `med` (nearest word by name has the same preterminal), mean over checkpoints 20–320 (22 word types, so one word is 0.045). `small` is 1.0 almost everywhere and does not separate the arms.

| med | seed 13 | seed 17 | misplaced at bands ctx / cnt (13) |
|---|---|---|---|
| control (no neighbour facet) | 0.78 | 0.73 | 0.18 / 0.33 |
| neighbour names for every word, ≤ 2 bits | 0.80 | 0.78 | 0.23 / 0.41 |
| n = 10, ≤ 2 bits | 0.92 | 0.83 | 0.23 / 0.43 |
| n = 10, all bands | 0.84 | 0.80 | 0.25 / 0.48 |
| n = 3, ≤ 2 bits | 0.83 | 0.73 | 0.23 / 0.41 |
| n = 3, all bands | 0.89 | 0.78 | 0.26 / 0.37 |

n = 10 with coarse names is the most consistent arm: it has the best word categories on both seeds (+0.14 and +0.10 over the control), grammaticality within 2 points, and novelty below the current setup on 4 of 6 runs (equal on med seed 17). The `med` vocabulary is too small for these differences to be firm, so TinyStories (250 words, `experiments/v3/diag/t3_stories.py`) decides. Under relabel-only refresh, misplacement is still 18–35% context and 33–53% content; temporal currency needs `refresh="verify"` on top.

## Rounds 12–18 on TinyStories (`experiments/v3/diag/t3_stories.py`)

Sentences of 3–8 words using the 250 most frequent words, 2000-sentence stream. Gold coarse POS (NLTK) only scores. Word-category accuracy: a word's nearest other word by learned name has the same tag. The raw-count baseline is cosine over left/right neighbour counts.

What the toy grammars hid:

- **Neighbour facets hurt on real text.** Any version that writes neighbours' learned names into the stored context lowers category accuracy, novelty and currency against the control. This covers a facet for every word, a facet for rare types (T6), and a window written in learned names; it holds for both band names and parent names. Examples at 2000 sentences: novelty 0.01–0.28 against the control's 0.37–0.41, content misplacement 0.27–0.51 against 0.19–0.23.
- **Band names lose category information the tree holds** (`t3_names.py`, 800 sentences). Built from the same trees, a word named by the parent concept of every one of its leaves scores 0.60 / 0.59. The current name (bands over 8 sampled leaves) scores 0.48 / 0.54, and the raw-count baseline 0.54 / 0.59.

The representation that works:

1. `name_level="parent"`, `type_k=32`: a word is named by the parent concepts of its sampled instances, read from the current tree.
2. `name_mix_k=4`: interdependence happens at readout only. A word's name is its own name with weight n/(n+4), plus the side-tagged names of the words that stood next to it. The stored descriptions do not change.
3. `part_names="bands"`: chunks name their word parts by the coarse band names, so the content tree generalises. Generation fills word slots by these coarse names too, because a slot accepts the members of its category.

| 2000 sentences, seeds 13 / 17 | category accuracy | misplaced ctx / cnt | novel | tag sequence attested | attested + novel |
|---|---|---|---|---|---|
| current setup (original) | 0.48 / 0.50 (at 1600) | | | | |
| control (bands + sampled restructuring, nocc) | 0.46 / 0.54 | 0.15 / 0.19, 0.14 / 0.23 | 0.37 / 0.41 | | 0.31 / 0.35 |
| parent names | 0.61 / 0.59 | 0.13 / 0.37, 0.16 / 0.35 | 0.26 / 0.27 | | 0.23 / 0.22 |
| parent + coarse parts + coarse fill | 0.61 / 0.59 | 0.15 / 0.24, 0.14 / 0.25 | 0.46 / 0.36 | 0.92 / 0.94 | 0.38 / 0.30 |
| **mixed (k=4) + coarse parts + coarse fill** | **0.59 / 0.65** | 0.15 / 0.22, 0.15 / 0.26 | 0.41 / 0.36 | 0.93 / 0.95 | 0.34 / 0.31 |

The mixed name converges fastest. Its category accuracy is 0.43 on both seeds at 50 sentences, against 0.43 / 0.27 for the control, and 0.57 / 0.54 at 400, above the raw-count baseline (0.53). This is interdependence that speeds convergence. Currency is at the control's level, but context misplacement is still 15%, against the ≤ 5% target, which needs cheaper reassignment. Generation is level with the control: attested + novel is higher on seed 13 and lower on seed 17.

## Rounds 19–26: grammaticality under pooled word categories (2026-10-24)

Grammars: `t3_evolve.py`, checkpoints 40–320, seeds 13 / 17, grammatical / grammatical and novel. TinyStories: `t3_stories.py`, 2000 sentences. Base config: the round-18 representation (parent names, `name_mix_k` 4, coarse parts).

| round | change | grammars at 320 (gram / gram+new) | TinyStories attested + novel | category accuracy (TS) | misplaced ctx / cnt (TS) |
|---|---|---|---|---|---|
| original | (words as tokens, band names) | 1.00 / 0.73–0.88 | 0.19 | 0.48 / 0.51 | 0.12 / 0.26 |
| 20 | fill: fine names, slot word capped at its best peer (`both+peer`) | med 0.88 / 0.80 | 0.37 / 0.38 | 0.59 / 0.65 | 0.15 / 0.22–0.26 |
| 21 | window 3 / 4 (raw window) | gram 1.0, slow novelty (med 0.76 / 0.68) | 0.13–0.19 / 0.07–0.11 | 0.62 / 0.65 | |
| 22 | fill one or two concepts above the leaf | grammaticality collapses (cancelled) | | | |
| 22b | window 3, far slots as one categorical learned label (`window_names="far"`) | med 0.87–0.91, large 0.92 | 0.42 / 0.46 | 0.67 / – | 0.22–0.24 / 0.30–0.36 |
| 23 | a word's own context slot = its learned category (`word_cc="cat"`), window 2 / window 3 far | 0.89–0.98 / 0.82–0.89; 0.86–0.99 / 0.77–0.90 | 0.33 (w2, s13) | 0.58 (w2, s13) | 0.17 / 0.30 |
| 24 | a word's own context slot = its token (`word_cc=true`) | 0.996–1.00 / 0.62–0.78 (large 13: 0.996 / 0.78) | | 0.52 / 0.48 at 800 | |
| 25 | fill weighted by P(candidate's category \| composite part) (`+predict`) | identical to round 23 (no-op) | | | |
| 26 | category label at 3 / 4 bits (`nb_bits`) | 3 bits identical to 2; 4 bits: running | | | |

**Where the grammaticality goes** (`t3_generr.py`, med, round-23 setup): every ungrammatical sentence under window 2 is a single verb↔preposition swap (33/33 on seed 13, 13/13 on seed 17). Under window 3 the same swap moves one level up: VP↔PP chunks (41 of 48 chunk errors on large 13). With two words of context, verbs and prepositions stand in the same contexts ("the dog eats the dog" / "the dog in the dog"); they share the coarse 2-bit category, so (a) their contexts pool at the same leaves, (b) chunks naming them as parts coincide ([V NP] = [P NP]), and (c) the round-25 prediction over that category cannot separate them. Their fine names do separate them (word-category accuracy 0.91–0.97 on the grammars), so the merge is the coarseness of the shared label, not the representation.

**Reassignment cost at 2000 sentences** (round-18 config): `refresh="verify"` (budget 100) halves context misplacement to 0.07 (both seeds) and content to 0.15–0.16, at 13 200–14 200 s of learning (about 3–4× relabel at 400 sentences). Still above the ≤ 5% target.

## Rounds 27–32: speed, the compiled tree, and what the misplacement measure counts (2026-10-25/26)

Base: the round-28 representation (parent names, `type_k` 32, `name_mix_k` 4, `word_cc` true, bands, sampled restructuring, fill `both+peer`).

- **Rounds 29–30** (`verify-events`: check records under a concept a merge or split just changed; `verify-drift`: check placement only once a description has drifted by more than `drift_tau`) both failed the grammar guard (med-13 0.783, large-17 0.804; round 30 failed 2 of 4) and were slower. Cancelled.
- **Profile:** refresh was 65% of the learning time and `_ec_with` dominated it. The Python fast path (`_sq`, dot product over the smaller side) gives about 1.5×.
- **Compiled Fisher tree** (`tree="cu"`, cobweb-private `trellis-discrete` b2fb112, module `cobweb_cu_trellis`): reference arithmetic is bit-identical to the Python tree (40 / 40 random streams; TRELLIS end-to-end on med-40 and stories-300). Fast mode matches within 1e-9 except at near-ties. On stories-300, learning takes 33 s (cu fast), 104 s (cu ref), 313 s (Python fast) and 517 s (Python ref).

**Round 31** (cu, `refresh="verify"`, budget sweep):

| | grammars at 320, gram / gram+new | TinyStories 2000, misplaced ctx / cnt (weight-1 measure) | TS learn s |
|---|---|---|---|
| original | med 1.0 / 0.868, 0.996 / 0.745; large 1.0 / 0.799, 0.993 / 0.833 | 0.123 / 0.252, 0.114 / 0.294 | 3 994 / 5 150 |
| verify 100 | fails large-13 (0.975) | 0.055 / 0.258, 0.069 / 0.23 | 3 179 / 3 283 |
| verify 300 | fails med-13 gram+new (0.854) | 0.056 / 0.218, 0.058 / 0.213 | 6 478 / 6 152 |
| verify 1000 | passes all 4 (1.0 / 0.871, 1.0 / 0.895, 0.986 / 0.858, 0.993 / 0.848) | running | > 10 000 |
| verify 3000 | passes all 4 | running | |

Content misplacement did not respond to the budget. Name churn (`t3_namechurn.py`, 1000 sentences): after one sentence, 25% of word names change which concepts they name, and 91% of those changes go to concepts that already existed.

**Round 32: the measure** (`t3_misplace_why.py`, 600 sentences, seed 13). The TEMPORAL measure sorts a record's stored description at weight 1, while `ifit` placed the record at its posterior weight (often below 0.1). Measured on records just stored:

| | ctx at weight 1 | ctx at own weight | cnt at weight 1 | cnt at own weight |
|---|---|---|---|---|
| fresh, no reassignment | 0.141 | 0.007 | 0.21 | 0.015 |
| end, no reassignment | 0.169 | 0.099 | 0.327 | 0.242 |
| end, verify 300 at weight 1 (445 s) | 0.094 | 0.026 | 0.173 | 0.052 |
| end, verify 100 at own weight (190 s) | 0.098 | 0.024 | 0.317 | 0.119 |
| end, verify 300 at own weight (344 s) | 0.102 | 0.028 | 0.31 | 0.072 |
| end, verify 1000 at own weight (805 s) | 0.086 | 0.010 | 0.257 | 0.055 |

So about 0.2 of the reported content misplacement is a mismatch of weights at insertion: a record sits where its own weight sorts it. Real drift still builds up without reassignment (0.24 content at 600 sentences), and verification removes most of it. Measured at own weight, verify 300 is near the 5% target at 600 sentences. `t3_stories.py` now reports both measures, and `verify_weight="own"` makes the check sort at the record's own weight. Round 33 re-measures at 2000 sentences.

## Rounds 33–36: the target met at verify 700 (2026-10-26)

Measures at 2000 TinyStories sentences (seeds 13 / 17). "Own weight" sorts each record's stored description at the weight `ifit` placed it with. "Share of stored weight" weights each record by its posterior weight, i.e. how much of an instance it put into the tree. Grammars: gram / gram+new at 320, against the original's med 1.0 / 0.868 and 0.996 / 0.745, and large 1.0 / 0.799 and 0.993 / 0.833.

| run (cu tree) | misplaced ctx / cnt, own weight | share of stored weight misplaced, ctx / cnt | learn s | attested + novel | category acc. | grammar guard |
|---|---|---|---|---|---|---|
| original | 0.088 / 0.213, 0.081 / 0.229 | | 4 091 / 5 207 | 0.218 / 0.225 | 0.455 / 0.504 | reference |
| verify 300 | 0.020 / 0.062, 0.015 / 0.076 | | 6 873 / 6 545 | 0.300 / 0.226 | 0.520 / 0.560 | fails med-13 (0.854) |
| verify 500 | 0.019 / 0.065, 0.025 / 0.076 | 0.011 / 0.028, 0.010 / 0.037 | 10 180 / 9 047 | 0.275 / 0.230 | 0.549 / 0.488 | fails med-13 (0.850) |
| **verify 700** | **0.005 / 0.076, 0.005 / 0.062** | **0.001 / 0.026, 0.003 / 0.029** | 12 818 / 11 941 | **0.290 / 0.277** | **0.533 / 0.548** | **passes all 4**: 1.0 / 0.871, 1.0 / 0.892, 0.996 / 0.846, 0.993 / 0.839 |
| verify 300 at own weight | 0.021 / 0.078, 0.014 / 0.124 | 0.018 / 0.045, – | 5 931 / 4 835 | 0.259 / 0.260 | 0.524 / 0.524 | fails med-13 (0.836) |
| verify 1000 at own weight | 0.013 / 0.067, 0.010 / 0.083 | | 14 558 / 12 624 | 0.236 / 0.275 | 0.520 / 0.532 | fails med-13 (0.841) |

**Verify 700** (`refresh="verify"`, `refresh_budget` 700, weight-1 check) meets the temporal target:
- context misplacement is 0.5%;
- content misplacement is 2.6–2.9% of stored weight (6–8% of records, almost all of weight below 0.1);
- learning time is 2.3–3.1× the original's.

It also passes the grammar guard on all four grammars, and it is up on TinyStories generation (attested + novel 0.29 / 0.28 against 0.22 / 0.23) and on category accuracy (interdependent names: 0.53 / 0.55 against 0.46 / 0.50). Checking at the record's own weight moves fewer records, but repairs less per check: the weight-1 check re-inserts almost every low-weight record it visits, which acts as a refit.

**Update on read (rounds 34–35, `ref_names`, `src/trellis2/ref_cobweb.py`, 600 sentences, Python tree).** Reference-valued descriptions, a versioned name table, and nodes that sync when a sort reaches them. With these, stored descriptions are current: drift 0.000–0.01, against 0.07–0.27 for copies. Test `tests/trellis2/test_ref_cobweb.py`: identical to the base tree when names are fixed; when names change, synced counts equal the current names summed. Placement is no better, and the cost is no lower:

| 600 sentences, against current descriptions | cnt misplaced | ctx misplaced | s |
|---|---|---|---|
| copies, verify 300 / 1000 | 0.087 / 0.061 | 0.024 / 0.011 | 4 772 / 10 916 |
| references, verify 300 / 1000 | 0.096 / 0.050 | 0.018 / 0.010 | 5 119 / 11 433 |
| references, no reassignment | 0.349 | 0.095 | 1 529 |
| references + repair beside reached leaves (200 / sentence) | 0.226 | 0.090 | 5 468 |

Name churn changes what records should sit with, not only how they are described. So synced counts still need records moved, and moving records (unfit + ifit) is what costs time. Update on read is kept as an option for exact descriptions; it is not needed for the target.
