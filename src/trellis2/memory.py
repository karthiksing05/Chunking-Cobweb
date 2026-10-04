"""Long-term memory: element records and the representation hierarchy.

Every element of a learned analysis is recorded: its span, its kind, its
token (primitives) or its two children (composites), and its parent. The
representation hierarchy is a Cobweb tree over the elements' representation
instances, which describe how each element behaves:

* surface context: the token on either side, its first and last token, and
  its kind (primitive or composite, postulate R5);
* chunk context, in category terms: the categories of its children (what it
  is made of) and a spine of ancestors with the sibling chunk beside the path
  at each level (default depth 2), each written at two granularities (the
  grammar's symbol and a finer node of the hierarchy).

Chunk context needs categories, and categories come from the hierarchy. So
consolidation iterates: describe every element with the categories of the
previous round, rebuild the hierarchy by replaying the elements in their
original order, read off new categories (see ``model.Trellis2.consolidate``).
In the first round the chunk attributes are blank and only surface context
is used.
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from .cobweb import CobwebNode, CobwebTree, Instance
from .data import Span, Tree

BOS, EOS = "<s>", "</s>"
BLANK = "-"
ROOT = "<root>"


def chunk_attrs(spine_depth: int, granularities: int) -> List[str]:
    """Children categories, then for each level d of the spine the ancestor
    category (a<d>) and the sibling chunk on its left or right (sl<d>, sr<d>);
    each written once per granularity (suffix .0 = grammar symbol, .1 = a
    finer node of the representation hierarchy)."""
    base = ["cl", "cr"]
    for d in range(1, spine_depth + 1):
        base += [f"a{d}", f"sl{d}", f"sr{d}"]
    return [f"{b}.{g}" for g in range(granularities) for b in base]


def representation_attrs(width: int, spine_depth: int, granularities: int,
                         composition_ref: bool, sentence_bags: bool = False) -> List[str]:
    return ([f"l{d}" for d in range(1, width + 1)]
            + [f"r{d}" for d in range(1, width + 1)]
            + ["f", "e", "k"] + chunk_attrs(spine_depth, granularities)
            + ([f"c.{g}" for g in range(granularities)] if composition_ref else [])
            + (["bl", "br"] if sentence_bags else []))


def _bag(tokens: Sequence[str]) -> Dict[str, float]:
    if not tokens:
        return {BLANK: 1.0}
    out: Dict[str, float] = {}
    for t in tokens:
        out[t] = out.get(t, 0.0) + 1.0 / len(tokens)
    return out


class Memory:
    """Element records plus the code that turns them into representation
    instances and a representation hierarchy."""

    PRIMITIVE, COMPOSITE = 0, 1

    def __init__(self, context_width: int = 1, spine_depth: int = 2,
                 granularities: int = 2, composition_ref: bool = False,
                 sentence_bags: bool = False):
        self.context_width = context_width
        self.spine_depth = spine_depth
        self.granularities = granularities
        self.composition_ref = composition_ref
        # Whole-sentence context: a bag of every token before the element and
        # one of every token after it ("all levels of content before and after").
        self.sentence_bags = sentence_bags
        self.attrs = representation_attrs(context_width, spine_depth, granularities,
                                          composition_ref, sentence_bags)
        self.kind: List[int] = []
        self.token: List[Optional[str]] = []
        self.left: List[int] = []
        self.right: List[int] = []
        # The relation that joins a composite's parts: None for a primitive,
        # and for concatenation (sequences have a single relation).
        self.relation: List[Optional[Hashable]] = []
        self.parent: List[int] = []
        # Top-level elements of a partial analysis: the neighbouring top-level
        # chunks act as their level-1 siblings (-1 if none).
        self.top_left: List[int] = []
        self.top_right: List[int] = []
        self.is_root: List[bool] = []
        self.weight: List[float] = []
        self.sentence_of: List[int] = []
        self.span: List[Span] = []
        self.sentences: List[List[str]] = []

    def __len__(self) -> int:
        return len(self.kind)

    def add_tree(self, tokens: Sequence[str], tree: Tree,
                 weight: float = 1.0) -> Dict[Span, int]:
        """Record every element of an analysed experience, bottom up.
        Returns the element id of each span."""
        tokens = list(tokens)
        if tree.n != len(tokens):
            raise ValueError("tree and token sequence disagree in length")
        sid = len(self.sentences)
        self.sentences.append(tokens)
        eid_of: Dict[Span, int] = {}
        for (i, j) in tree.bottom_up():
            eid = len(self.kind)
            eid_of[(i, j)] = eid
            if j - i == 1:
                self.kind.append(self.PRIMITIVE)
                self.token.append(tokens[i])
                self.left.append(-1)
                self.right.append(-1)
            else:
                k = tree.split[(i, j)]
                left, right = eid_of[(i, k)], eid_of[(k, j)]
                self.kind.append(self.COMPOSITE)
                self.token.append(None)
                self.left.append(left)
                self.right.append(right)
                self.parent[left] = eid
                self.parent[right] = eid
            self.relation.append(None)
            self.parent.append(-1)
            self.top_left.append(-1)
            self.top_right.append(-1)
            self.is_root.append((i, j) in tree.roots)
            self.weight.append(weight)
            self.sentence_of.append(sid)
            self.span.append((i, j))
        tops = [eid_of[r] for r in tree.roots]
        for a, b in zip(tops, tops[1:]):
            self.top_right[a] = b
            self.top_left[b] = a
        return eid_of

    def vocabulary(self) -> List[str]:
        return sorted({t for t in self.token if t is not None})

    # ------------------------------------------------------------------ #
    def instance(self, e: int, labels: Optional[Sequence[np.ndarray]] = None,
                 rules: Optional[Sequence[np.ndarray]] = None) -> Instance:
        """The representation instance of element ``e``. ``labels`` holds, per
        granularity, the category of every element from the previous round,
        and ``rules`` its composition class (None = blank, as in the first
        round). The optional composition reference (off by default; it made
        results worse and less stable) points an element's representation at
        the concept describing what it is made of."""
        x = self.surface(e)
        for g in range(self.granularities):
            lab = None if labels is None else labels[g]

            def cat(other: int) -> str:
                return BLANK if lab is None or other < 0 else f"S{int(lab[other])}"

            x[f"cl.{g}"] = cat(self.left[e])
            x[f"cr.{g}"] = cat(self.right[e])
            # The spine: walk up the analysis, recording each ancestor and the
            # sibling chunk beside the path at that level.
            node = e
            for d in range(1, self.spine_depth + 1):
                p = self.parent[node] if node >= 0 else -1
                if p < 0:
                    # ROOT marks the whole experience. A piece of a partial
                    # analysis (a forest) is not the sentence: its parent is
                    # unknown.
                    whole = node >= 0 and self.top_left[node] < 0 and self.top_right[node] < 0
                    x[f"a{d}.{g}"] = ROOT if (whole and lab is not None) else BLANK
                    # A top-level chunk's siblings are its top-level neighbours.
                    top = node >= 0
                    x[f"sl{d}.{g}"] = cat(self.top_left[node]) if top else BLANK
                    x[f"sr{d}.{g}"] = cat(self.top_right[node]) if top else BLANK
                    node = -1
                    continue
                is_left = self.left[p] == node
                x[f"a{d}.{g}"] = cat(p)
                x[f"sl{d}.{g}"] = BLANK if is_left else cat(self.left[p])
                x[f"sr{d}.{g}"] = cat(self.right[p]) if is_left else BLANK
                node = p
            if self.composition_ref:
                x[f"c.{g}"] = BLANK if rules is None else f"R{int(rules[g][e])}"
        return x

    def surface(self, e: int) -> Instance:
        """The element's surface context: the tokens on either side, its first
        and last token, and its kind."""
        tokens = self.sentences[self.sentence_of[e]]
        i, j = self.span[e]
        n = len(tokens)
        x: Instance = {}
        for d in range(1, self.context_width + 1):
            x[f"l{d}"] = tokens[i - d] if i - d >= 0 else BOS
            x[f"r{d}"] = tokens[j + d - 1] if j + d - 1 < n else EOS
        x["f"] = tokens[i]
        x["e"] = tokens[j - 1]
        x["k"] = "P" if self.kind[e] == self.PRIMITIVE else "C"
        if self.sentence_bags:
            x["bl"] = _bag(tokens[:i])
            x["br"] = _bag(tokens[j:])
        return x

    def describe(self, e: int) -> str:
        """A short rendering of the element's content (its tokens)."""
        i, j = self.span[e]
        toks = self.sentences[self.sentence_of[e]][i:j]
        return " ".join(toks) if len(toks) <= 4 else " ".join(toks[:2] + ["…"] + toks[-1:])

    def build_hierarchy(self, labels: Optional[Sequence[np.ndarray]] = None,
                        rules: Optional[Sequence[np.ndarray]] = None,
                        seed: int = 0) -> Tuple[CobwebTree, List[CobwebNode]]:
        """Replay every element, in learning order, into a fresh Cobweb tree.
        Returns the tree and the leaf that stores each element."""
        tree = CobwebTree(self.attrs, seed=seed)
        leaves = [tree.ifit(self.instance(e, labels, rules), self.weight[e])
                  for e in range(len(self.kind))]
        return tree, leaves
