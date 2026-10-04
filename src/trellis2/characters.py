"""Chinese characters as experiences: a first domain beyond language.

An Ideographic Description Sequence (IDS) describes a character as an
operator that places two (or three) parts, for example 湖 = ⿰氵胡 (water to
the left of 胡) and 胡 = ⿰古月. Expanding every part down to atomic
components gives a prefix sequence of operators and components,
``⿰ 氵 ⿰ 古 月``, which is the character's token sequence here. Its gold tree
groups an operator with its first part, so that a component *in position*
is a chunk: ``[[⿰ 氵] [[⿰ 古] 月]]`` (a ternary operator: ``[[[⿲ A] B] C]``).

Because prefix notation with known arities is unambiguous, any well-formed
generated sequence reads back as exactly one character structure, which is
how generated characters are checked:

* well formed: a composed character whose operators all have their parts;
* positions: every atomic component sits only in positions (operator and
  slot) where some real character places it;
* real: the sequence is the decomposition of an existing character.

The data are the CJKVI IDS database (based on CHISE; GPL), unpacked into
``data/ids/ids.txt``:

    curl -L -o data/ids/ids.txt https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .data import Tree

ARITY = {"⿰": 2, "⿱": 2, "⿴": 2, "⿵": 2, "⿶": 2, "⿷": 2, "⿸": 2, "⿹": 2, "⿺": 2,
         "⿻": 2, "⿲": 3, "⿳": 3}
OPERATORS = set(ARITY)


@dataclass
class Character:
    char: str
    tokens: List[str]      # prefix sequence of operators and atomic components
    tree: Tree             # gold structure: an operator grouped with its first part


def default_ids_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "data", "ids", "ids.txt"))


def read_ids(path: str) -> Dict[str, str]:
    """Character -> its first IDS, with region tags such as [GTKV] removed."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out[parts[1]] = re.sub(r"\[[^\]]*\]", "", parts[2])
    return out


def expand(char: str, ids: Dict[str, str], depth: int = 0) -> Optional[List[str]]:
    """The prefix sequence of atomic components and operators, or None if the
    decomposition uses an unencoded component or does not terminate."""
    s = ids.get(char, char)
    if s == char or depth > 12:
        # Atomic components are CJK characters, radicals or strokes; the circled
        # numbers and other symbols the database uses for unencoded parts are not.
        return [char] if len(char) == 1 and ord(char) >= 0x2E80 else None
    out: List[str] = []
    for ch in s:
        if ch in OPERATORS:
            out.append(ch)
        else:
            sub = expand(ch, ids, depth + 1)
            if sub is None:
                return None
            out.extend(sub)
    return out


def parse_prefix(tokens: Sequence[str]):
    """Read a prefix sequence back into a structure: a component, or
    (operator, part, part[, part]). Returns None if it is not well formed."""
    pos = 0

    def node():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError
        t = tokens[pos]
        pos += 1
        if t in OPERATORS:
            return (t,) + tuple(node() for _ in range(ARITY[t]))
        return t
    try:
        out = node()
    except (ValueError, RecursionError):
        return None
    return out if pos == len(tokens) else None


SLOT_NAMES = {("⿰", 0): "left", ("⿰", 1): "right", ("⿱", 0): "top", ("⿱", 1): "bottom",
              ("⿲", 0): "left", ("⿲", 1): "middle", ("⿲", 2): "right",
              ("⿳", 0): "top", ("⿳", 1): "middle", ("⿳", 2): "bottom",
              ("⿴", 0): "full frame", ("⿴", 1): "enclosed", ("⿵", 0): "frame open below",
              ("⿵", 1): "enclosed", ("⿶", 0): "frame open above", ("⿶", 1): "enclosed",
              ("⿷", 0): "frame open right", ("⿷", 1): "enclosed", ("⿸", 0): "upper-left frame",
              ("⿸", 1): "enclosed", ("⿹", 0): "upper-right frame", ("⿹", 1): "enclosed",
              ("⿺", 0): "lower-left frame", ("⿺", 1): "enclosed", ("⿻", 0): "overlaid",
              ("⿻", 1): "overlaid"}


def token_slots(tokens: Sequence[str]) -> List[Optional[Tuple[str, int]]]:
    """For each token, the (operator, slot) it fills directly; None for the
    root and for operators."""
    out: List[Optional[Tuple[str, int]]] = [None] * len(tokens)
    pos = 0

    def node(slot):
        nonlocal pos
        t = tokens[pos]
        if t not in OPERATORS:
            out[pos] = slot
            pos += 1
            return
        pos += 1
        for k in range(ARITY[t]):
            node((t, k))
    node(None)
    return out


def gold_tree(tokens: Sequence[str]) -> Tree:
    """Binary tree over the prefix sequence: [[op A] B], [[[op A] B] C]."""
    split = {}
    pos = 0

    def node() -> Tuple[int, int]:
        nonlocal pos
        start = pos
        t = tokens[pos]
        pos += 1
        if t not in OPERATORS:
            return (start, pos)
        end_prev = pos                                   # the operator alone spans [start, start+1)
        for _ in range(ARITY[t]):
            i, j = node()
            split[(start, j)] = end_prev
            end_prev = j
        return (start, pos)
    node()
    return Tree(len(tokens), split)


def placements(structure, out: Optional[Set] = None) -> Set[Tuple[str, int, str]]:
    """(operator, slot, atomic component) for every component placed directly
    in a slot of an operator."""
    out = set() if out is None else out
    if isinstance(structure, tuple):
        op = structure[0]
        for slot, part in enumerate(structure[1:]):
            if isinstance(part, tuple):
                placements(part, out)
            else:
                out.add((op, slot, part))
    return out


def load_characters(path: str, max_tokens: int = 11, seed: int = 0,
                    first: int = 0x4E00, last: int = 0x9FFF) -> Tuple[List[Character], Dict[str, str]]:
    """Every character of the block with a decomposition of at most
    ``max_tokens`` tokens, in a seeded random order."""
    ids = read_ids(path)
    chars = []
    for c, s in ids.items():
        if not (first <= ord(c) <= last) or s == c:
            continue
        tokens = expand(c, ids)
        if tokens is None or len(tokens) > max_tokens or parse_prefix(tokens) is None:
            continue
        chars.append(Character(c, tokens, gold_tree(tokens)))
    random.Random(seed).shuffle(chars)
    return chars, ids
