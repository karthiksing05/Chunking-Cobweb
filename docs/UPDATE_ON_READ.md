# Update on read: names computed when a sort reaches a node (plan, 2026-10-26)

## Why

Round 31 (compiled Fisher tree, TinyStories 2000) shows that a larger reassignment budget does not bring content records current: stale content stays at 0.21–0.26 for budgets 0, 100 and 300, while stale context falls to about 0.06. Measured name churn (`t3_namechurn.py`, 1000 sentences): after one sentence, 25% of word names change which concepts they name, 91% of those to concepts that already existed (splits, moved leaves, reservoir turnover, neighbours' names through the k=4 mix) and 9% to a newly created node. The names are computed live; what goes stale are the copies of them in stored descriptions and in the trees' counts.

Which description fields copy names (round-28 representation):

| tree | field | holds | stale-prone |
|---|---|---|---|
| context | `l2 l1 r1 r2` | raw neighbouring tokens | no |
| context | `cc` | a word's own token; a chunk's own content concept ids | chunks only |
| content | `L R` | the parts' names (word: mixed name; chunk: its type's reservoir name) | yes |
| content | `LC RC` | the parts' content concept ids (`-` for words) | yes |

## Design

1. **Reference values.** A stale-prone field stores a reference, not a name: a word part by its token, a chunk part (and a chunk's own `cc`) by its record id. References never change.
2. **Name table.** A shared table maps each reference to its current name (a bag of concept ids), with a version number per entry and a global log of changes. TRELLIS updates the entries whose names changed after each experience.
3. **Nodes count references.** Each node keeps counts per reference for these fields, plus the translated counts per concept id (and their sums of squares), and the version it was last synced at.
4. **Sync on visit, save the result.** When `ifit` or `categorize` reaches a node, the node and each child it scores apply the logged name changes since their stamp (count of the reference × (new name − old name)), update their sums of squares, and save the result with the new stamp. Nodes no sort visits wait until one does.
5. **The instance** is translated with the current table at the start of its sort.
6. **Records at a reached leaf** are checked and moved if they no longer belong, so placement is repaired where there is activity.
7. **No refresh sweep:** `refresh_budget` and relabelling go.

## Stages

- **A. Python prototype** on `IncrementalCobwebTree` (reference fields, name table, sync on visit), run on the four grammars and 300 TinyStories sentences: staleness, grammar, the count of synced deltas per sort. Confirms the idea before compiled work.
- **B. Compiled** in the CU tree (opt-in reference attributes, versioned table, per-node stamps), checked bit-for-bit against stage A in reference arithmetic.
- **C. Full runs:** TinyStories 2000 both seeds plus grammars, against round 31's best budget on staleness, cost, grammar and category accuracy.

## Measure of success

Stale content and context at or below 5% at no more than about 3× the original setup's learning time on the same tree, with the grammar guard (grammaticality within 2 points of the original, grammatical + novel up).

## Stage A result (2026-10-26)

Built: `src/trellis2/ref_cobweb.py` (`RefCobwebTree`, `NameTable`, `Ref`) and `Trellis3(ref_names=True)`, tested in `tests/trellis2/test_ref_cobweb.py`. Descriptions are exact (drift ≤ 0.01). Placement and cost match copies with the same reassignment budget; see TEMPORAL_INTERDEP.md, rounds 33–36. The misplacement the plan set out to fix was mostly a measurement artefact: categorize sorted at weight 1, while ifit placed records at their posterior weight. The rest was structural drift that only moving records repairs. Stage B (compiled) is on hold: verify 700 on the compiled tree already meets the target.
