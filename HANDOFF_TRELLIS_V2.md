# TRELLIS v2: handoff for running on a cluster (2026-10-08)

Branch `inside-outside` of this repository. Start with [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) (the system end to end) and [`docs/V2_DESIGN.md`](docs/V2_DESIGN.md) (the design log: every decision with its evidence, in dated entries). The report with figures is the claude.ai artifact *Concepts and Chunks from Raw Experience: Extending Cobweb from Composition to Compression* (https://claude.ai/artifact/NRNV4sV26WoV8N4cBkTZG9, private until shared).

## 1. Where things stand

TRELLIS v2 keeps every element of an experience in two Cobweb hierarchies: the **representation hierarchy** (how the element behaves) and the **composition hierarchy** (what it is made of). One factored probabilistic grammar is read off cuts through the two, chosen by description length; it parses (inside-outside, minimum-risk tree), generates, completes prompts, and prices the data. From experiences alone the learner alternates day (perceive and store) and night. A night now has eight steps (`src/trellis2/unsupervised.py`):

1. word classes; 2. structure search (chunk and merge moves, plain code); 3. consolidation into the two hierarchies; 4. re-analysis (Viterbi, hard EM);
5. the three best search results are consolidated with and without their forests joined (`join_forests=True`), the full code choosing;
6. re-analysis without the read's context (`context_free_steps=10`);
7. **sampled re-analysis** (stochastic EM, `sampling=(1, 1, .8, .8, .6, .6, .4, .4, .2)`), new on 2026-10-08;
8. rewrite the stored analyses in the new categories.

Every step is kept only if it shortens the full code. 90 tests pass (`python -m pytest tests/trellis2 -q`, about a minute).

**Results in `experiments/v2/results/` and which night made them.** `unsupervised/` (synthetic, two seeds) has the full night, sampling included. Every other directory was made on 2026-10-07 with steps 1–6 but not step 7: the reruns with sampling were stopped when this handoff was written. **Rerun everything on the cluster** (section 3) to have all results from one night.

| Domain | Chunks | Coherence |
|---|---|---|
| Synthetic corpora (paper's six grammars), from sentences alone | complete analyses; code within 0.2% of the gold-tree grammar's | commission 0.1–7.4%; unseen prompts completed grammatically 91–100% (n-grams 29–53%) |
| Chinese characters, structures given | categories of position (left radicals, tops, frames); recurring parts | 100% well formed; 84% of components in attested slots (93% from 6,000) |
| Chinese characters from token sequences | 38% of training characters analysed exactly as gold | 53% well formed (59% with sampling, in a refit test); gold structures still code 2.7% shorter |
| Chess (16,000 Lichess positions) | weak: one chunk type pays (blocked pawns); the read's counting does the rest | 100% of generated boards pass every check |
| English, TinyStories 3–5 words over 100 words | every sentence one tree | 70–75% of samples real; at the trigram's novelty as often real as the trigram (87%); held-out code below n-grams |
| English, 3–8 words over 250 words, 10,000 sentences | every sentence one tree | 37% of samples real (bigram 24%, trigram 52%); held-out 20.9 bits (n-grams 23.9–24.2) |
| Penn Treebank tags (up to 20 tags) | base phrases; sentences stay forests (the code prefers them) | 94% of samples derived as one tree have every tag triple attested |

**Blockers**, in order: (1) speed: the search, grammar read-out and charts are Python, charts are cubic in sentence length, and a night on 10,000 eight-word sentences took 7.6 hours (more with sampling); (2) sentence structure does not pay for itself in this grammar family on real text (joins are generic, the treebank prefers forests); (3) the structure search falls short where gold is known (characters); (4) generation is n-gram-quality at matched novelty; (5) richer real prose is too sparse at the sizes we can afford (Gutenberg failed). Details: `docs/FRAMEWORK.md` section 14.

## 2. Setting up

**Code.** Clone this repository (branch `inside-outside`) and cobweb-private (branch `karthik-experimental`, already pushed, with `cobweb_cu`). Build the compiled Cobweb and make it importable:

```
cd cobweb-private
pip install nanobind                      # and a C++17 compiler, CMake
cmake -S . -B build && cmake --build build --target cobweb_cu
export PYTHONPATH=$PWD/build:$PYTHONPATH  # or symlink build/cobweb_cu*.so into site-packages
```

`src/trellis2/cobweb.py` imports `cobweb.cobweb_cu` or `cobweb_cu`. Python 3.12 with numpy, scipy, matplotlib and pytest; fontTools draws the characters figure; python-chess and zstd extract chess positions.

**Data** (`data/` is gitignored; never commit it):

| Data | Where | How |
|---|---|---|
| the paper's six synthetic corpora | `data/cfg_grammar_experiment_{small,med,large}`, `data/cfg_terminal_{low,med,high}` (about 1.6 MB each) | not in any repository: copy them from this machine (`rsync`); `default_data_root()` reads `../trellis_v1/data` if it exists, else `data/` |
| TinyStories validation file | `data/tinystories/TinyStoriesV2-GPT4-valid.txt` | `curl` line in `src/trellis2/stories.py` |
| CJKVI IDS | `data/ids/ids.txt` | `curl` line in `src/trellis2/characters.py` |
| Penn Treebank sample (NLTK) | `data/ptb_sample/treebank/combined` | `curl` + `unzip` lines in `src/trellis2/treebank.py` |
| Lichess, January 2013 | `data/chess/positions_1500_ply30.fen` | `curl` the `.pgn.zst`, then `run_chess.py --extract` (and `--min-elo 1500`), see `experiments/v2/run_chess.py` |

Smoke test: `python -m pytest tests/trellis2 -q`.

## 3. Rerunning everything

`experiments/v2/run_all.sh` runs the whole suite in order, one job after another, logging to `logs/`; on a cluster, submit its lines as separate jobs instead, and give the long English nights their own nodes. Timings are from this laptop (12 cores shared among several runs), without sampling unless noted, so treat them as upper bounds for a dedicated node:

| Command | Writes | Time here |
|---|---|---|
| `run_unsupervised.py --seeds 13,17 --workers 6` | `results/unsupervised` | 5 min (with sampling) |
| `run_incremental.py --seeds 13,17 --workers 6` then `plot_incremental.py results/incremental` | `results/incremental` | 10–20 min |
| `run_prompts_synthetic.py --workers 6` | `results/prompts_synthetic` | 10–20 min |
| `run_characters.py` | `results/characters` | about 1 hour (the unsupervised night runs in one process) |
| `run_treebank.py --train-max-len {10,15,20} --out results/treebank/wsj{10,15,20}` | treebank | 2 min, 20 min, 1 hour |
| `run_stories.py --train 2500 --out results/stories_2500` | | 20–40 min |
| `run_stories.py --train 5000 --out results/stories`, then `run_prompts.py` | `results/stories`, `results/prompts` | 1.5–3 hours |
| `run_stories.py --train 9000 --out results/stories_9000` | | 4–6 hours |
| `run_stories.py --train 2500 --vocab 250 --max-len 8 --out results/stories_250` | | 20–40 min |
| `run_stories.py --train 10000 --vocab 250 --max-len 8 --out results/stories_250_10000` | | 8–20 hours |
| `run_synthetic.py`, `run_chess.py` | `results/main`, chess | unaffected by the night (supervised; chess has its own learner) |

Gotchas: `run_incremental.py` and `run_characters.py` default to seed 13 only (the committed incremental results use 13 and 17); every run overwrites its results directory; `grammar.pkl` is gitignored; nights parallelize their searches and consolidations (`--workers`), but steps 6–7 are sequential; zsh's `time` prints a line ending in `total`, not `real`.

## 4. Next steps

### Track 1. Parse and generate complex language: grammatical, and novel

The goal: a parser that parses and generates complex grammatical structure, grammatical and sufficiently novel at once, which needs representations and chunks built for generation.

1. **Speed first.** Profile a night (`cProfile`) and move the hot loops to C++ next to `cobweb_cu`: the chart's inside, Viterbi and sampling passes (`chart.py`, loops over span lengths, starts and split points), the search's move scoring (`mdl_search._State.scored_moves`), and the cut searches' code-length calls (`grammar.py`). The brute-force chart tests and the exact-scoring search tests check that a compiled version gives identical results. Expect 10–100x on those parts.
2. **An exact testbed for complex grammar.** Write larger synthetic CFGs with what real syntax has and the paper's six lack: agreement, center embedding, nested relative clauses, coordination, sentences of 10–30 words. Grammaticality is then exact (`trellis2.evaluation.CFG`), and novelty is measured against training sentences. Report grammatical, novel, and grammatical-and-novel per 1,000 samples, bracket omission and commission, and prompting (`run_prompts_synthetic.py`). This is where representation changes should be judged first.
3. **Real English at scale.** TinyStories with longer sentences and real vocabularies (3–12 words over 1,000 words, 50,000+ sentences), then the treebank's words instead of its tags. Measure realness, every-triple attestation, and the novelty-versus-realness curve against n-grams at matched novelty (temperature sweep; see `docs/V2_DESIGN.md`, "Coherent new sentences").
4. **Representations for generation.** Heads recorded in both hierarchies are the leading candidate (describing a chunk by its left head was tried on characters and was worse: `docs/V2_DESIGN.md`, characters section); parent-annotated categories; categories that keep rare words apart. Every change so far that acted only at generation time traded novelty for coherence along the same curve, so the lever is the categories and the structure.

### Track 2. Chess: storage against forgetting

A memorization test in the spirit of Chase and Simon (1973), whose experts recalled real positions far better than random ones because they stored chunks.

- **Storage.** Learn N positions (1,000 to 16,000). The storage cost of TRELLIS is its description length: the grammar's model bits plus each position's code (`Memory.log_prob`), which arithmetic coding reconstructs exactly. Compare bits per position against: 64 squares at log2 13 bits; FEN strings with gzip, xz and zstd (per position and as a corpus); an independent per-square model; the board read alone, without chunks (already reported by `run_chess.py`); and a learned embedder, an autoencoder with a quantized k-dimensional latent (latent bits plus decoder parameter bits) and a small transformer language model over squares (cross-entropy plus parameter bits).
- **Real against random.** The same comparison on random positions with the same piece counts: chunks should help real positions only, and the gap between them is the measure of what was chunked.
- **Forgetting.** Recall under a budget: coarsen the grammar (coarser cuts, fewer chunk types and categories) and plot reconstruction accuracy (squares correct) against bits stored, a rate-distortion curve, against an autoencoder with a shrinking latent. Then interference: learn N more positions, and test recall of the first N against a network trained on them in sequence.
- Reuse `src/trellis2/chess.py` (the read, `BoardMemory`, `ChessLearner`) and `run_chess.py`; a new `experiments/v2/run_chess_memory.py` would hold the baselines and the curves.

### Track 3. Chinese characters: are the chunks strong?

- **Memorization and compression.** Bits per character for TRELLIS (structures given, and sequences alone) against gzip and xz of the IDS strings, token n-grams and a small neural sequence model; and on shuffled characters (components placed at random in their operators' slots), where chunks should not help.
- **Chunk validity.** Strong chunks should be units the script reuses: the share of learned composite chunk types that are themselves characters or standard components in the IDS database (胡 = 古 + 月), and how many of the database's frequent components the composition hierarchy recovers.
- **Completion and cued recall.** Given an operator and its first part, complete the character (prompting); mask one component and fill it in; compare with n-grams. Rediscovering held-out real characters is already measured (5.3% of samples).

## 5. Rules of this repository

Push as karthiksing05 (`GH_TOKEN="$(gh auth token --user karthiksing05)" git push`). Never commit `INSIDE_OUTSIDE.md` (the user's working note) or anything under `data/`. cobweb-private is pushed by the user only, with no Claude co-author trailers there. Never write "F1": report omission and commission (Langley and Stromsten).
