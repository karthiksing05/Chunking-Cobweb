"""Simple English: sentences from children's stories.

TinyStories (Eldan & Li 2023) are short stories written with the words a
three- or four-year-old knows, made to test whether small models can produce
coherent English (CDLA-Sharing-1.0). The validation file, about 27,600
stories and 4.4 million words, is enough here and goes into
``data/tinystories``:

    curl -L -o data/tinystories/TinyStoriesV2-GPT4-valid.txt \\
        https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

A sentence ends at ``.``, ``!`` or ``?``; it is lowercased and split into
words, and punctuation is dropped. ``simple_sentences`` keeps the sentences
of a given length that use only the most frequent words.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import List, Optional, Sequence

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def default_tinystories_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "data", "tinystories",
                                        "TinyStoriesV2-GPT4-valid.txt"))


def read_sentences(path: Optional[str] = None) -> List[List[str]]:
    """Every sentence of the file, as lowercase words."""
    with open(path or default_tinystories_path(), encoding="utf-8") as f:
        text = f.read().replace("<|endoftext|>", " ")
    text = re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'"))
    out = []
    for m in re.finditer(r"[^.!?]+[.!?]", text):
        words = WORD.findall(m.group(0).lower())
        if words:
            out.append(words)
    return out


def simple_sentences(sentences: Sequence[Sequence[str]], vocab_size: int = 250,
                     min_len: int = 3, max_len: int = 8) -> List[List[str]]:
    """The sentences of ``min_len``–``max_len`` words that use only the
    ``vocab_size`` most frequent words of ``sentences``, in their order."""
    vocab = {w for w, _ in Counter(w for s in sentences for w in s).most_common(vocab_size)}
    return [list(s) for s in sentences if min_len <= len(s) <= max_len and all(w in vocab for w in s)]
