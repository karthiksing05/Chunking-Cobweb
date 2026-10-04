# TRELLIS (Chunking Cobweb)

Work with Dr. Pat Langley and Dr. Chris Maclellan on ISLE Internship to model attributes of the psychological principle of cognitive chunking through a Cobweb-backed framework.

## TRELLIS v2

TRELLIS v2 keeps concepts and chunks in two Cobweb hierarchies, the representation hierarchy (how elements behave) and the composition hierarchy (what they are made of). It reads one probabilistic grammar off them by minimum description length, and learns from analysed sentences or from sentences alone, by day and by night. The two hierarchies are the core of v2: every other part is read off them or feeds them.

- [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md): the framework explained end to end, with figures.
- [`docs/V2_DESIGN.md`](docs/V2_DESIGN.md): design decisions, evidence and result tables.
- Code: `src/trellis2/`; experiments: `experiments/v2/`; tests: `python -m pytest tests/trellis2 -q`.
