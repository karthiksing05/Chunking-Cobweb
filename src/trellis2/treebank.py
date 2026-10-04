"""Penn Treebank sentences as experiences (WSJ10 with gold part-of-speech tags).

The standard setting for unsupervised parsing (Klein & Manning 2002): the
tokens are the gold part-of-speech tags; empty elements and punctuation are
removed, and so are the constituents left empty; sentences of at most ten
tags are kept. Unary chains collapse into one span. Brackets are compared
without labels, ignoring single tags and the whole-sentence span, which no
parser can get wrong.

The data are NLTK's public sample of the treebank (about 3,900 sentences of
the Wall Street Journal, for non-commercial use), unpacked into
``data/ptb_sample``:

    curl -L -o treebank.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip
    unzip treebank.zip -d data/ptb_sample
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import FrozenSet, List, Sequence

from .data import Span, Tree

PUNCTUATION = {",", ".", ":", "``", "''", "-LRB-", "-RRB-", "#", "$", "-NONE-"}


@dataclass
class TreebankSentence:
    tags: List[str]
    words: List[str]
    brackets: FrozenSet[Span]          # every constituent span of length >= 2 (n-ary gold)
    tree: Tree                         # the gold tree, right-binarized (for supervised training)

    @property
    def tokens(self) -> List[str]:
        return self.tags


def _parse_sexpr(text: str):
    """Parse '( (S (NP ...) ...) )' blocks into nested lists [label, children...]."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def node():
        nonlocal pos
        assert tokens[pos] == "("
        pos += 1
        label = None
        if tokens[pos] not in ("(", ")"):
            label = tokens[pos]
            pos += 1
        kids = []
        while tokens[pos] != ")":
            if tokens[pos] == "(":
                kids.append(node())
            else:
                kids.append(tokens[pos])
                pos += 1
        pos += 1
        return [label] + kids

    out = []
    while pos < len(tokens):
        out.append(node())
    return out


def _clean(t, tags, words):
    """Drop punctuation and empty elements, collapse unary nodes. A cleaned
    node is a leaf index or a list of at least two cleaned children."""
    label, kids = t[0], t[1:]
    if len(kids) == 1 and isinstance(kids[0], str):           # preterminal
        if label in PUNCTUATION:
            return None
        tags.append(label)
        words.append(kids[0])
        return len(tags) - 1
    cleaned = [c for c in (_clean(k, tags, words) for k in kids) if c is not None]
    if not cleaned:
        return None
    return cleaned[0] if len(cleaned) == 1 else cleaned


def _span(node) -> Span:
    if isinstance(node, int):
        return (node, node + 1)
    return (_span(node[0])[0], _span(node[-1])[1])


def _brackets(node, out: set) -> set:
    if not isinstance(node, int):
        out.add(_span(node))
        for child in node:
            _brackets(child, out)
    return out


def _right_binarized(node, split: dict) -> dict:
    """Children c1 ... ck become c1 (c2 (... ck))."""
    if isinstance(node, int):
        return split
    left, right = (node[0], node[1]) if len(node) == 2 else (node[0], node[1:])
    i, k = _span(left)
    split[(i, _span(right)[1])] = k
    _right_binarized(left, split)
    _right_binarized(right, split)
    return split


def load_wsj(root: str, max_len: int = 10, min_len: int = 2) -> List[TreebankSentence]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*.mrg"))):
        with open(path) as f:
            for block in _parse_sexpr(f.read()):
                t = block[1] if block[0] is None and len(block) == 2 else block
                tags, words = [], []
                node = _clean(t, tags, words)
                n = len(tags)
                if node is None or not (min_len <= n <= max_len):
                    continue
                out.append(TreebankSentence(tags, words, frozenset(_brackets(node, set())),
                                            Tree(n, _right_binarized(node, {}))))
    return out


def default_ptb_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "data", "ptb_sample", "treebank", "combined"))


def split(sentences: Sequence[TreebankSentence], seed: int, test_share: float = 0.2):
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    n_test = int(round(test_share * len(order)))
    test = [sentences[i] for i in order[:n_test]]
    train = [sentences[i] for i in order[n_test:]]
    return train, test


def evaluable(brackets, n: int) -> set:
    """Brackets that a parser can get wrong: span length >= 2, not the whole sentence."""
    return {(i, j) for i, j in brackets if j - i >= 2 and not (i == 0 and j == n)}
