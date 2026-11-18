# Round 208: stopped by user decision: n=4000 dropped

The user dropped n=4000 (n=1000 is enough) while the round was running, so the generation runs were cancelled before any produced a result.
No fair-count numbers exist for any arm; no pass or kill verdict is given.

## What exists

- Code: `src/trellis2/trellis3_r208.py` (`Trellis3R208(Trellis3R207)`, `store_same` learner flag, `repeat_read` read flag). Stored word occurrences get
  `s_k` = [own word == l_k]; the read groups committed records by their `s_k` pattern against the query's window words and runs the parent read per group.
  Without `store_same` the class is Trellis3R207 byte for byte. Hooks in `experiments/v3/diag/t3_story.py`: cfg key `"same_k": true`, arm key
  `"repeat": "same"` (the edge / escape class swaps now skip a learner that already is a subclass). `experiments/v3/diag/r208_fair.py` reports repeat arms as the edge family.
- Tests: `tests/trellis2/test_r208_same.py` (6 tests: byte-identical defaults, flag-off read equals parent, `s_k` counts equal direct counting on a toy,
  pattern marks the matching occurrences, repeat read lowers "dog dog" against "dog ran", works with edge_exact). Full `tests/trellis2` suite: 127 passed (sbatch 4788183).
- Smoke (n=250 seed 13 w5 t2.0, sbatch 4788121): edge, edge+same and edge+escape+same all ran to completion; no judge, no scores.

## M3 kill pre-test (passed, counts only, seed 13, sbatch 4788079)

Of ungrammatical sentences that repeat a word, the share whose repeat lies within the window (Qwen judge, P(yes) <= .5):

| dump | window | ungrammatical repeat sentences | within window |
|---|---|---|---|
| r200 n=1000 w6, t1.6 (design cell) | 6 | 94 | .979 |
| r200 n=1000 w6, all temps | 6 | 276 | .993 |
| r204 n=4000 w5, all temps | 5 | 324 | .944 |
| r207 n=4000 w5 escape, t2.0 | 5 | 123 | .967 |

Kill threshold .80: not triggered; M3 survives the pre-test.

## Not run (cancelled)

- Long shot 1: edge + escape at w5 t1.0 / 1.3 / 1.6, n=4000 seed 13 (jobs 4788073-77).
- Long shot 2: edge + same at w5 t1.6 / 2.0, n=4000 seed 13 (jobs 4788184-87).
- Seed 17, n=1000 confirmations, escape+same combination.

## Rule slips

- One `python3` empty-heredoc command was run on the login node by mistake (no script, nothing computed; the permission classifier refused it).
- `.sh` scripts and logs are git-ignored (logs/v3_r208_*.sh) and not committed.
