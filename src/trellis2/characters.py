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

**Operators as relations.** The same structure can be read as parts joined
by typed relations, as pieces are joined on a board: 湖 = [氵 ⿰ [古 ⿰ 月]].
The operator is then the relation of a composite, not a token, and what the
representation hierarchy sees of a part includes its slot (the operator
that places it and which part it is), so that categories of components form
by where they go (``CharacterMemory``). A three-part operator is written as
two joins of its two-part counterpart, which lays the parts out the same way
(⿲ A B C = ⿰ A ⿰ B C), so that every generated tree is a well-formed
character.

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

import numpy as np

from .data import Tree
from .grammar import UNK
from .memory import ROOT, Memory, chunk_attrs

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


# --------------------------------------------------------------------------- #
# Characters as relational trees
# --------------------------------------------------------------------------- #
RELATIONS = ["⿰", "⿱", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻"]
TWO_PART = {"⿲": "⿰", "⿳": "⿱"}


def to_relational(structure):
    """(op, a, b[, c]) as nested binary (relation, first, second) nodes."""
    if isinstance(structure, str):
        return structure
    op, parts = structure[0], [to_relational(p) for p in structure[1:]]
    if len(parts) == 2:
        return (op, parts[0], parts[1])
    rel = TWO_PART[op]
    return (rel, parts[0], (rel, parts[1], parts[2]))


def from_relational(node):
    """The IDS structure of a relational tree (two-part operators only)."""
    if isinstance(node, str):
        return node
    return (node[0], from_relational(node[1]), from_relational(node[2]))


def canonical(structure):
    """An IDS structure with three-part operators written as two joins."""
    return from_relational(to_relational(structure))


def structure_tokens(structure) -> List[str]:
    """The prefix sequence of an IDS structure."""
    if isinstance(structure, str):
        return [structure]
    return [structure[0]] + [t for p in structure[1:] for t in structure_tokens(p)]


class CharacterMemory(Memory):
    """Element records for characters as relational trees. What the
    representation hierarchy sees of an element: its slot (the operator that
    places it and which part it is; ROOT for the whole character), its first
    and last components, its own operator (P for a component), and its chunk
    context."""

    relations = RELATIONS

    def __init__(self, spine_depth: int = 2, granularities: int = 2):
        super().__init__(spine_depth=spine_depth, granularities=granularities)
        self.attrs = ["slot", "f", "e", "k"] + chunk_attrs(spine_depth, granularities)
        self.first: List[str] = []
        self.last: List[str] = []

    def add_structure(self, node, weight: float = 1.0) -> None:
        """Record every element of a character's relational tree, bottom up."""
        sid = len(self.sentences)
        self.sentences.append(node)

        def record(n) -> int:
            if isinstance(n, str):
                a = b = -1
                first = last = n
            else:
                a, b = record(n[1]), record(n[2])
                first, last = self.first[a], self.last[b]
            e = len(self.kind)
            self.kind.append(self.PRIMITIVE if a < 0 else self.COMPOSITE)
            self.token.append(n if a < 0 else None)
            self.left.append(a)
            self.right.append(b)
            self.relation.append(None if a < 0 else n[0])
            if a >= 0:
                self.parent[a] = self.parent[b] = e
            self.first.append(first)
            self.last.append(last)
            self.parent.append(-1)
            self.top_left.append(-1)
            self.top_right.append(-1)
            self.is_root.append(False)
            self.weight.append(weight)
            self.sentence_of.append(sid)
            self.span.append(None)
            return e

        self.is_root[record(node)] = True

    def surface(self, e: int):
        p = self.parent[e]
        slot = ROOT if p < 0 else f"{self.relation[p]}:{0 if self.left[p] == e else 1}"
        return {"slot": slot, "f": self.first[e], "e": self.last[e],
                "k": "P" if self.kind[e] == self.PRIMITIVE else self.relation[e]}

    def describe(self, e: int) -> str:
        if self.kind[e] == self.PRIMITIVE:
            return self.token[e]
        return f"{self.relation[e]}{self.describe(self.left[e])}{self.describe(self.right[e])}"


def structure_log_prob(g, node) -> float:
    """ln P(relational tree) under a grammar: the inside pass over the known
    structure, summing over every assignment of categories."""
    rel_index = {r: i for i, r in enumerate(g.relations)}
    unk = g.tok_index[UNK]

    def inside(n):
        if isinstance(n, str):
            v = g.U @ (g.pk * g.E[:, g.tok_index.get(n, unk)])
            scale = 0.0
        else:
            lx, vx = inside(n[1])
            ly, vy = inside(n[2])
            v = g.U @ (g.qk * g.Rel[:, rel_index[n[0]]] * (g.Lt @ vx) * (g.Rt @ vy))
            scale = lx + ly
        total = v.sum()
        return scale + np.log(total), v / total

    scale, v = inside(node)
    return float(scale + np.log(v @ g.S) + g.log_whole)


def sample_structure(g, rng: np.random.Generator, max_depth: int = 12):
    """A relational tree read off the grammar: one tree from the start row."""
    def expand(sym, depth):
        c = int(rng.choice(g.M, p=g.U[sym]))
        if rng.random() < g.pk[c] or depth >= max_depth:
            return g.vocab[int(rng.choice(len(g.vocab), p=g.E[c]))]
        b, d = int(rng.choice(g.K, p=g.Lt[c])), int(rng.choice(g.K, p=g.Rt[c]))
        rel = g.relations[int(rng.choice(len(g.relations), p=g.Rel[c]))]
        return (rel, expand(b, depth + 1), expand(d, depth + 1))
    return expand(int(rng.choice(g.K, p=g.S)), 0)
