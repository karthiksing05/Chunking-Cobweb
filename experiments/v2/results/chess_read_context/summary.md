# Chess: the read's answers in context (2026-10-12)

Each question the read asks ("is an element anchored here on a piece of kind k?") is coded given its counted row and
what the read has already placed around the square: the backward half of the star as a chain (S; S+W; S+W+SW+SE; plus
the backward knight squares), through `mdl.backoff_chain_coder`, 13 depth/β options chosen by description length
(choice paid); depth 0 is the counted read exactly. Used in the search (exact, incremental per square), the grammar's
code, `log_prob` and `sample` (`ChessLearner(read_context=True)`; off by default). α 1e-3, counts on, split of
`run_chess_chunks.py` (seed 13). Bits per position.

| setting | learner | TRELLIS total | held-out: read alone | held-out: grammar | held-out chunk gain | chunk types | pieces in chunks | time |
|---|---|---|---|---|---|---|---|---|
| 1800/16 4k | old (no context) | 52.972 | 51.824 | 51.550 | +0.27 | 2 | 6.1% | 675 s |
| | label context | 52.416 | 51.823 | 50.551 | +1.27 | 5 | 9.1% | 14,349 s |
| | + read context | **44.230** | 41.058 | **40.847** | +0.21 | 1 | 4.0% | 3,995 s |
| 1800/16 8k | old | 52.009 | 50.655 | 50.418 | +0.24 | 3 | 6.5% | 2,522 s |
| | label context | 51.047 | 50.655 | 49.070 | +1.59 | 9 | 12.9% | 47,681 s |
| | + read context | **41.854** | 38.542 | **38.342** | +0.20 | 2 | 6.7% | 10,151 s |
| 1500/30 4k | old | 79.118 | 76.199 | 76.105 | +0.09 | 0 | 0 | 502 s |
| | label context | 79.115 | 76.197 | 76.105 | +0.09 | 0 | 0 | 1,189 s |
| | + read context | **74.357** | 70.044 | **69.952** | +0.09 | 0 | 0 | 1,693 s |
| 1500/30 16k | old | 76.276 | 75.477 | 75.372 | +0.11 | 1 | 5.1% | 15,677 s |
| | label context | 76.204 | 75.476 | 75.101 | +0.38 | 1 | 5.4% | 10,856 s |
| | + read context | **69.575** | 66.964 | **66.959** | +0.005 | 0 | 0 | 12,305 s |

Against the transformer (`results/chess_memory`, 1500/30, held-out): 4k 65.18 vs 69.95 (gap 10.9 → 4.8); 16k 60.00 vs
66.96 (gap 15.4 → 7.0).

Options chosen: 1800/16 read (4, β16), all eight backward neighbours; 1500/30 read (2, β16) at 4k, (3, β16) at 16k;
random positions: depth 0 (no context), held-out code unchanged (135.325 / 127.929), no chunks.

Generation (1,000 samples): passes every check 0.996–1.000, rejected 0–0.3%, generated chunks found in held-out 0.99
(the rejections label context introduced are gone).

Findings: (1) most of the code was in the read, and context recovers 6–11 held-out bits per position, more with more
data; (2) random positions unaffected; (3) generation improves; (4) chunks mostly vanish: none pays on 1500/30, on
1800/16 only `[wN N1 wP]` and `[wQ E1 wK]` (0.2 held-out bits); the fianchetto and pawn-chain chunks are gone.
Inferred (not tested): those chunks joined pieces whose later member lies in the backward star of the earlier one, so
the read in context now predicts it and a chunk saves too little to pay for its announcement.

Code: `src/trellis2/chess.py` (`read_context`, `backoff_terms`), tests in `test_chess_context.py`, `test_unified.py`,
`test_domains.py`; 131 tests pass.
