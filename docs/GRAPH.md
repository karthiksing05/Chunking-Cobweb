# TRELLIS v2 on relational experiences (chess boards)

Code: `src/trellis2/trellis_graph.py` (`GraphTrellis`, `ChessDomain`, `SentenceDomain`).

## The generalisation

| | sentences (`Trellis3`) | relational (`GraphTrellis`) |
|---|---|---|
| primitive | a word | a primitive in the domain's read order (chess: a piece, squares read a1 b1 … h8) |
| element | a span (i, j) | a set of primitives, connected by the joins that built it |
| composite | split k: [i, k) + [k, j) | A + B where B's anchor (first primitive) comes after A's and some primitive of A sees it under a relation |
| relations | one (next word) | the domain's (chess: forward rays N, NE, NW, E at distance 1–7, forward knight jumps; A's own pieces do not block) |
| content | parts' names (+ donated, heights) | the same, plus `rel`: the bag of relations by which A sees B |
| surface context | two words either side | the domain's window around the anchor, own primitives left out (chess: the star, i.e. a bag of the first piece on each of the 8 rays, a bag of the 8 knight squares, and the anchor's square) |
| neighbour facet | `lc`, `rc`: the learned names of the words either side | `nb`: the mean learned name of the pieces in the star |
| analysis | one binary tree over the sentence | a forest of chunks of at most `max_size` primitives; pieces not joined stay single |
| posterior | inside-outside over spans | estimated from `samples` analyses drawn bottom-up |

With one relation the composites are exactly the spans and splits, so a sentence is the special case. `SentenceDomain` exists to check the estimator against inside-outside.

## Parsing

A board has too many forests to sum over. Each analysis is drawn bottom-up. At every step, either one possible join of the current top-level elements is taken, with weight 2^(gain/τ), or the analysis stops, with weight 1. A join that does not pay is therefore less likely than stopping. The stored analysis is the greedy one: take the best-paying join while one pays (v1's greedy chunking). Chunk context and the generation statistics come from it. Every composite that appears in any sample is stored with its sample frequency as weight, as `store="posterior"` does for sentences. This is the "beam between v1 greedy and v2 inside-outside" direction (docs/FAILURE_MODES.md, memory note on beam search).

## What learning keeps from `Trellis3`

The two hierarchies, donation, information-band names, sampled restructuring, refresh and reassignment, type reservoirs, and the word-content reservoirs are all used unchanged. Each top-level chunk of the stored analysis adds a passive `top` count, the board-level statistic that generation will read.

## Evaluation plan

1. **Chunks carry structure:** the share of stored chunks whose joins are tactical relations (`chess.interaction`: attack or defence) against a random-board control.
2. **Downstream (representation learning):** Lichess puzzle themes (fork, pin, skewer, mate in 2, back-rank mate, discovered attack, …). Learning sees only positions. A linear probe on each test position's bag of concepts (band names from both hierarchies, over its words and stored-analysis chunks) predicts each theme. Baselines are a piece-square bag and a piece-relation bag (the same relations, no learning). Accuracy is reported against the number of labelled examples.
3. **Memorisation:** Chase–Simon reconstruction, as in the v1 chess runs.
4. **Generation** (later): chunks from content concepts, checked for board consistency (`chess.plausibility`, and that every relation is realised).

## First probe result (2026-10-24, paused afterwards)

Run with `store="best"`, interdependent config (`nb_bits` 2), max chunk size 4, seed 13. Mean AUC over 16 Lichess themes, 2000 held-out puzzles, by number of labelled positions:

| | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|
| piece-square | 0.60 | 0.60 | 0.62 | 0.63 |
| piece-relation | 0.57 | 0.60 | 0.61 | 0.63 |
| TRELLIS concepts, 300 positions learned | 0.57 | 0.58 | 0.58 | 0.60 |
| TRELLIS concepts, 1000 positions learned | 0.57 | 0.57 | 0.58 | 0.60 |
| TRELLIS + piece-square (1000 learned) | 0.57 | 0.58 | 0.59 | 0.63 |

The learned concepts add nothing over the raw features, and learning 1000 positions instead of 300 does not help. This run used the band names that later proved lossy on TinyStories (docs/TEMPORAL_INTERDEP.md). Chess is paused until the representation is settled.
