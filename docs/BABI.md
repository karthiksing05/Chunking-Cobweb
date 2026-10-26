# bAbI: a question-answering corpus beyond TinyStories (2026-10-26)

Data: `data/babi/babi_{train,valid,test}.jsonl` (Hugging Face Muennighoff/babi, 20 tasks), read by `src/trellis2/babi.py`. The stream holds each story's sentences once plus `<question> ? <answer>` lines: 76,610 sentences, 4,114 distinct, 150 words. Every measurement script runs on it with `CORPUS=babi`: `t3_stories`, `t3_parse_consistency`, `t3_prompt` and `t3_chunk_probe`.

## Question answering as gap filling (`experiments/v3/diag/t3_babi_qa.py`)

The prompt is the question plus `?`, and the answer is the generated gap. Measures:
- **type accuracy:** the answer is one the task ever has;
- **exact accuracy;**
- **story:** fillers restricted to the story's words, or closed answers such as yes/no or numbers;
- **sibling:** the answer generated as the right part of a composite whose left part matches the question's right-edge chunk. QA lines parse as `[where [is [[mary ?] bathroom]]]`, so the answer is the sibling of `mary ?`.

The baseline draws an answer from the training answers by frequency.

## First results (600 sentences, 4 questions per task)

Window-gap type accuracy is 0.55, against 0.23 for the baseline. Story-restricted exact accuracy is 0.16, against 0.06. Long questions fail: "where was the apple before the bathroom ?" is answered yes/no, because the window sees only "bathroom ?". Sibling mode now conditions on the parsed chunk; full runs are in `logs/babi-*`.

## Next

Story-conditioned answering by analogy: align the question's chunks with the story sentence that shares its entity and role, and answer with the filler in the aligned slot. This needs chunk-level binding, not a bag of story words.
