# Chess: the read's context for the top-level labels (2026-10-11)

Each top-level element's label is coded given what the read has already placed nearby: the backward half of the star as a
chain (S; S+W; S+W+SW+SE; plus the four backward knight squares), through `mdl.backoff_chain_coder`, grouped by
(anchor kind, square), in reading order. Description length chooses the depth (0–4) and one β of `CONTEXT_WEIGHTS`
(13 options, log2 13 bits); depth 0 is the old code exactly. Used in the search (moves scored exactly) and in the
grammar's top level (code, `log_prob`, `sample`). α 1e-3, counts on, split of `run_chess_chunks.py` (seed 13). Bits per
position. "Current" rows: `results/chess_hier/runs`.

| setting | learner | train: search | TRELLIS total | held-out: read alone | held-out: grammar | held-out chunk gain | chunk types | top-level chunks by depth | pieces in chunks |
|---|---|---|---|---|---|---|---|---|---|
| 1800+/16, 4k | current | 53.184 | 52.972 | 51.824 | 51.550 | +0.27 | 2 | 1: 2606, 2: 721 | 6.1% |
| | context | 52.447 | **52.416** | 51.823 | **50.551** | **+1.27** | 5 | 1: 3356, 2: 732, 3: 494 | 9.1% |
| 1800+/16, 8k | current | 52.151 | 52.009 | 50.655 | 50.418 | +0.24 | 3 | 1: 5616, 2: 1420 | 6.5% |
| | context | 50.994 | **51.047** | 50.655 | **49.070** | **+1.59** | 9 | 1: 9254, 2: 2787, 3: 1011 | 12.9% |
| 1500+/30, 4k | current | 79.242 | 79.118 | 76.199 | 76.105 | +0.09 | 0 | – | 0 |
| | context | 79.238 | 79.115 | 76.197 | 76.105 | +0.09 | 0 | – | 0 |
| 1500+/30, 16k | current | 76.316 | 76.276 | 75.477 | 75.372 | +0.11 | 1 | 1: 10799 | 5.1% |
| | context | 76.252 | **76.204** | 75.476 | **75.101** | **+0.38** | 1 | 1: 11245 | 5.4% |

Context with greedy merges: identical at 4k in both settings (no merge pays).

Chunk types under context, 1800/16 at 8k: `[wP NW1 wP]` (white fianchetto pawns h2/g3; f2/e3), `[wP NNW wP]`,
`[[wP NW1 wP] NW2 wP]` (f2-e3-d4 chain), `[wK E1 wB]`, `[bP NW1 [bP E1 bB]]` (black kingside / queenside fianchetto),
`[[bP NW1 [bP E1 bB]] WNW bP]` (level 3), `[wN E2 wP]`, `[[wN E2 wP] NE2 bB]`.

Generation (1,000 samples, current → context): passes every check 0.991 → 0.964 (1800/16 4k), 0.996 → 0.949 (8k),
1.000 → 1.000 (1500/30 4k), 0.999 → 0.990 (16k); rejected samples 0–0.5% → 5.6–9.5% where chunks grew; generated
chunks found in held-out games 0.99 → 0.92–0.95 (1800/16). Random control (same pieces at random, 4k): no chunks under
either code; held-out grammar code unchanged.

Against the transformer (`results/chess_memory`, 1500+/30): 4k 65.18 vs 76.11 (current) / 76.11 (context); 16k 60.00
vs 75.37 / 75.10.

## Findings (confirmed unless marked)

1. On 1800/16 context makes chunks pay and the gain grows with N (held-out +0.27 → +1.27 at 4k, +0.24 → +1.59 at 8k;
   total −0.56 / −0.96 bits per position), with fianchetto structures and pawn chains up to three levels.
2. On 1500/30 context changes nothing at 4k; at 16k the held-out gain rises 0.11 → 0.38.
3. Random control: zero chunks, so the gain is from real positions only.
4. Generation is somewhat worse where chunks grew (rejected samples 5.6–9.5%): the contextual top level draws chunk
   labels where their parts are no longer free.
5. The gap to the transformer barely moves (about 15 bits at 16k).
6. Step 1 rerun under the context code (`diag/`): whole configurations (castled kings, fianchettos with the king, pawn
   chains) still do not pay as single joint moves (0 of 55–57; median +1.23 / +2.72 bits per occurrence); what pays is a
   step-by-step decomposition (`[bP E1 bB]` → `[bP NW1 [bP E1 bB]]` → `+ WNW bP`). The binding cost is the top-level
   announcement of a whole template.

Suspected: most of the remaining gap is in the read's own yes/no answers (about 95% of the code), which see only square
and count; the same back-off on the read's answers, given the backward star, is the next candidate. The context search
is slow (3–13 h at 4k–16k against 0.2–4 h).

## Code

`src/trellis2/chess.py`: `BoardSearch(merges=False, merge_kinds, patience)` with merge moves; `ContextBoardSearch`;
`BoardMemory(context=False)` with the contextual top level; `ChessLearner(merges=False, patience=0, context=False)`
(defaults reproduce the earlier results exactly: 1800/16 4k 53.184 / 52.972 / 51.550). Tests: `test_chess_search.py` (7),
`test_chess_context.py` (4); full suite 124 passed.
