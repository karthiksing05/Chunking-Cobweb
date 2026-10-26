"""bAbI (Weston et al. 2015): 20 question-answering tasks over short English stories,
from the Hugging Face copy Muennighoff/babi (jsonl rows: passage, question, answer,
task) in ``data/babi``:

    curl -L -o data/babi/babi_train.jsonl \\
        https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_train.jsonl

A row's passage repeats its story up to the question, so a story's sentences are
taken once, in order, with each question written after the sentences it follows as
"<question words> ? <answer words>". Words are lowercased; "?" is kept as a token.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|\?")


def default_path(split: str = "train") -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "data", "babi", f"babi_{split}.jsonl"))


def words(text: str) -> List[str]:
    return WORD.findall(text.lower().replace(",", " "))


def babi_rows(split: str = "train", tasks: Optional[set] = None) -> List[Dict]:
    """One row per question: the story so far (sentences), question, answer, task."""
    out = []
    with open(default_path(split), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if tasks is not None and r["task"] not in tasks:
                continue
            story = [words(s) for s in r["passage"].split("\n") if s.strip()]
            out.append({"story": [s for s in story if s], "question": words(r["question"]),
                        "answer": words(r["answer"]), "task": r["task"]})
    return out


def babi_sentences(split: str = "train", tasks: Optional[set] = None) -> List[List[str]]:
    """The stream: each story's sentences once, in order, each question after the
    sentences it follows as question + ["?"] + answer (the question's "?" kept once)."""
    out: List[List[str]] = []
    prev: List[List[str]] = []
    for r in babi_rows(split, tasks):
        story = r["story"]
        k = len(prev) if story[:len(prev)] == prev else 0
        out.extend(story[k:])
        q = [w for w in r["question"] if w != "?"]
        out.append(q + ["?"] + r["answer"])
        prev = story
    return out
