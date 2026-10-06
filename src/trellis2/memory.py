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
from .mdl import dm_code, rows_nats

BOS, EOS = "<s>", "</s>"
BLANK = "-"
ROOT = "<root>"
SENTENCE = "<sentence>"


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
    instances and a representation hierarchy.

    A domain is a subclass that says what an experience is. It supplies

    * ``add``: record every element of an analysed experience (its kind,
      token or two parts, the relation joining them, its parent, whether it is
      a top-level element, and the analysis' own label for it);
    * ``relations``: how two parts can be joined (None: concatenation only);
    * ``surface``: what the representation hierarchy sees of an element
      besides its chunk context (its context window), and ``describe``;
    * the top level, how an experience's top-level elements are laid out and
      transmitted: ``top_level_nats``, ``layout_nats``, ``top_level_tables``;
    * ``contexts``: what the read has seen just before each element, in whose
      light its rule class is chosen (None: nothing);
    * ``sample`` and ``log_prob``: draw an experience from the grammar and
      code one, in the same reading order.

    The chunk context, the representation hierarchy, the cuts, the
    composition hierarchy and the grammar are shared. This class is the
    domain of sentences: tokens in a row, read left to right, each rule
    choice in the light of the word before the element.
    """

    PRIMITIVE, COMPOSITE = 0, 1

    def __init__(self, context_width: int = 1, spine_depth: int = 2,
                 granularities: int = 2, composition_ref: bool = False,
                 sentence_bags: bool = False, previous_word: bool = True,
                 sentence_parent: bool = False, fresh_pieces: bool = False, read_words: int = 2):
        self.context_width = context_width
        self.spine_depth = spine_depth
        self.granularities = granularities
        self.composition_ref = composition_ref
        # Whole-sentence context: a bag of every token before the element and
        # one of every token after it ("all levels of content before and after").
        self.sentence_bags = sentence_bags
        # Whether each rule choice is made in the light of the word read just
        # before the element (``contexts``).
        self.previous_word = previous_word
        # Whether the sentence itself is the parent recorded above a whole
        # tree's top parts and above a forest's pieces alike (``instance``).
        self.sentence_parent = sentence_parent
        # Whether each piece of a forest is read afresh: an element at a
        # piece's start is read in the light of BOS, not of the previous
        # piece's last word (``contexts``).
        self.fresh_pieces = fresh_pieces
        # How many of the words just read the memory offers as each rule
        # choice's context (1 or 2); description length decides how many the
        # grammar uses.
        if read_words not in (1, 2) or (read_words == 2 and fresh_pieces):
            raise ValueError("read_words is 1 or 2, and pieces are read afresh with one word only")
        self.read_words = read_words
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
        self.label: List[Hashable] = []      # the analysis' own label of each element
        self.experience_of: List[int] = []
        self.span: List[Span] = []
        self.experiences: List = []

    def __len__(self) -> int:
        return len(self.kind)

    # ------------------------------------------------------------------ #
    # The domain's experiences: record one, draw one, code one. A sentence
    # is its tokens; its analysis is a Tree (a forest when partial).
    # ------------------------------------------------------------------ #
    def add(self, experience: Sequence[str], analysis: Tree, weight: float = 1.0) -> None:
        """Record every element of an analysed sentence, bottom up."""
        tokens, tree = list(experience), analysis
        if tree.n != len(tokens):
            raise ValueError("tree and token sequence disagree in length")
        sid = len(self.experiences)
        self.experiences.append(tokens)
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
            self.label.append(tree.label.get((i, j)))
            self.experience_of.append(sid)
            self.span.append((i, j))
        tops = [eid_of[r] for r in tree.roots]
        for a, b in zip(tops, tops[1:]):
            self.top_right[a] = b
            self.top_left[b] = a

    def sample(self, grammar, rng: np.random.Generator, **kw):
        """An experience and its analysis drawn from the grammar: (tokens,
        Tree), or None if longer than ``max_len`` (callers resample)."""
        return grammar.sample(rng, **kw)

    def log_prob(self, grammar, experience, analysis=None) -> float:
        """ln P(experience) under the grammar, summed over every analysis
        (inside-outside)."""
        from .chart import Chart
        return Chart(grammar, experience).log_prob

    def contexts(self) -> Optional[List[Hashable]]:
        """What the read has seen just before each element, in whose light its
        rule class is chosen: for a sentence, the word before the element's
        span (BOS at the start), or the two words before it (``read_words=2``;
        the grammar then chooses by description length between none, one and
        two). None: every choice is made the same way."""
        if not self.previous_word:
            return None
        fresh = set()
        if self.fresh_pieces:
            fresh = {(self.experience_of[e], self.span[e][0]) for e in range(len(self.span))
                     if self.is_root[e] and (self.top_left[e] >= 0 or self.top_right[e] >= 0)}
        if self.read_words == 2:
            return [(self.experiences[s][i - 2] if i > 1 else BOS, self.experiences[s][i - 1] if i > 0 else BOS)
                    for s, (i, _) in zip(self.experience_of, self.span)]
        return [self.experiences[s][i - 1] if i > 0 and (s, i) not in fresh else BOS
                for s, (i, _) in zip(self.experience_of, self.span)]

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
            if self.sentence_parent:
                self._sentence_spine(x, e, g, lab, cat)
                if self.composition_ref:
                    x[f"c.{g}"] = BLANK if rules is None else f"R{int(rules[g][e])}"
                continue
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

    def _whole_root(self, n: int) -> bool:
        return n >= 0 and self.parent[n] < 0 and self.top_left[n] < 0 and self.top_right[n] < 0

    def _sentence_spine(self, x: Instance, e: int, g: int, lab, cat) -> None:
        """The spine with the sentence as a parent: above a whole tree's top
        parts and above a forest's pieces alike, the parent is SENTENCE (whose
        join is known in the one case and unknown in the other), and above it
        ROOT. A piece's siblings are its neighbouring pieces."""
        known = lab is not None
        node: object = e
        for d in range(1, self.spine_depth + 1):
            a = sl = sr = BLANK
            if node == SENTENCE:
                a, node = (ROOT if known else BLANK), None
            elif node is not None:
                p = self.parent[node]
                if p < 0 and self._whole_root(node):
                    a, node = (ROOT if known else BLANK), None        # the element is the sentence
                elif p < 0:
                    a = SENTENCE if known else BLANK                  # a piece of a forest
                    sl, sr = cat(self.top_left[node]), cat(self.top_right[node])
                    node = SENTENCE
                else:
                    is_left = self.left[p] == node
                    top = self._whole_root(p)
                    a = (SENTENCE if known else BLANK) if top else cat(p)
                    sl = BLANK if is_left else cat(self.left[p])
                    sr = cat(self.right[p]) if is_left else BLANK
                    node = SENTENCE if top else p
            x[f"a{d}.{g}"], x[f"sl{d}.{g}"], x[f"sr{d}.{g}"] = a, sl, sr

    def surface(self, e: int) -> Instance:
        """The element's surface context: the tokens on either side, its first
        and last token, and its kind."""
        tokens = self.experiences[self.experience_of[e]]
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
        toks = self.experiences[self.experience_of[e]][i:j]
        return " ".join(toks) if len(toks) <= 4 else " ".join(toks[:2] + ["…"] + toks[-1:])

    # ------------------------------------------------------------------ #
    # The top level: how an experience's top-level elements are laid out.
    # A sentence is one tree, or a forest of two or more pieces when the
    # grammar cannot derive it whole; roots and pieces have rows of their own.
    # ------------------------------------------------------------------ #
    def _top_level(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(the elements that are the root of an experience analysed as one
        tree, those that are a piece of a forest, every element's weight).
        The cut searches ask for it at every step, so it is kept until an
        element is added."""
        n = len(self.is_root)
        kept = self.__dict__.get("_top_level_kept")
        if kept is None or kept[0] != n:
            root = np.array(self.is_root, dtype=bool)
            alone = (np.array(self.top_left) < 0) & (np.array(self.top_right) < 0)
            kept = (n, np.flatnonzero(root & alone), np.flatnonzero(root & ~alone),
                    np.array(self.weight, dtype=float))
            self._top_level_kept = kept
        return kept[1:]

    def _layout_counts(self) -> Tuple[np.ndarray, np.ndarray]:
        """(one tree, a forest) per experience, and (stop, go on) after each
        piece of a forest from the second on."""
        root = np.array(self.is_root, dtype=bool)
        w = np.array(self.weight, dtype=float)
        sid = np.array(self.experience_of)
        m = np.bincount(sid[root], minlength=len(self.experiences))
        sent_w = np.zeros(len(self.experiences))
        sent_w[sid[root]] = w[root]
        forest = m >= 2
        return (np.array([sent_w[m == 1].sum(), sent_w[forest].sum()]),
                np.array([sent_w[forest].sum(), (sent_w * np.maximum(m - 2, 0)).sum()]))

    def top_level_nats(self, s: np.ndarray, K: int, alpha: float) -> float:
        """Code (nats) of the top-level elements' symbols ``s`` (one per
        element, K of them): the part of the top level that depends on the
        categories."""
        whole, piece, w = self._top_level()
        return (dm_code(np.zeros(len(whole), dtype=np.int64), s[whole], w[whole], K, alpha)
                + dm_code(np.zeros(len(piece), dtype=np.int64), s[piece], w[piece], K, alpha))

    def layout_nats(self, alpha: float) -> float:
        """Code (nats) of the layout that does not depend on the categories:
        whether each experience is one tree or a forest, and where a forest ends."""
        n_mode, n_stop = self._layout_counts()
        return rows_nats(np.stack([n_mode, n_stop]), alpha)

    def top_level_tables(self, s: np.ndarray, K: int, alpha: float):
        """The grammar's top-level fields, and the count tables behind them
        (for the model/data split)."""
        whole, piece, w = self._top_level()
        n_start = np.bincount(s[whole], weights=w[whole], minlength=K)
        n_piece = np.bincount(s[piece], weights=w[piece], minlength=K)
        n_mode, n_stop = self._layout_counts()
        fields = {"S": (n_start + alpha) / (n_start + alpha).sum(),
                  "S_piece": (n_piece + alpha) / (n_piece + alpha).sum(),
                  "p_whole": float((n_mode[0] + alpha) / (n_mode.sum() + 2 * alpha)),
                  "p_stop": float((n_stop[0] + alpha) / (n_stop.sum() + 2 * alpha))}
        return fields, [(n_start, K), (n_piece, K), (n_mode, 2), (n_stop, 2)]

    def build_hierarchy(self, labels: Optional[Sequence[np.ndarray]] = None,
                        rules: Optional[Sequence[np.ndarray]] = None,
                        seed: int = 0) -> Tuple[CobwebTree, List[CobwebNode]]:
        """Replay every element, in learning order, into a fresh Cobweb tree.
        Returns the tree and the leaf that stores each element."""
        tree = CobwebTree(self.attrs, seed=seed)
        leaves = [tree.ifit(self.instance(e, labels, rules), self.weight[e])
                  for e in range(len(self.kind))]
        return tree, leaves
