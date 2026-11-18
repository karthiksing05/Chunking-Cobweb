"""Round 203: parse-readout and boundary-count options for Trellis3, as a subclass (trellis3.py is untouched).

Trellis3R203(edge_w=None, goodman=False, ...) is Trellis3 unless an option is set:

* edge_w (a): count a sentence edge ONCE in the context gain, without dropping slots. When the instance's nearest
  slot and the ones beyond it all hold the boundary marker (l1 = l2 = .. = BOS, or r1 = r2 = .. = EOS: a span at the
  sentence edge), those m >= 2 slots are one observation of "the sentence ends here", so each slot's log-ratio term
  log P(v | C) / P(v | root) is weighted edge_w / m (edge_w = 1: the edge counts once; the default 2-slot window
  counts it twice, weight 1 + 1 = 2). The instance, its sorting path and storage are exactly the default's (round
  196b's edge_once removed the slot from the scored instance, which changed the path and the instance the compiled
  tree sees; here only the weights of the log-ratio terms change). The factor's share term is unchanged.
* goodman (b): the chosen tree maximises the expected number of correct spans, a CKY over span posterior sums
  (Goodman's labelled recall) WITHOUT the split-posterior tie-break the default _mbr adds when picking a split.
* goodman_nary(n, post, lam): an n-ary Goodman readout from the same posteriors: the non-crossing set of spans
  maximising sum (P(span) - lam) over included spans; lam = 0.5 keeps spans more likely than not, a smaller lam
  keeps more of them (lam <= 0 includes every span of the best binary tree).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .data import Tree
from .trellis3 import BOS, EOS, LN2, Trellis3


def span_posteriors(post) -> Dict[Tuple[int, int], float]:
    sp: Dict[Tuple[int, int], float] = {}
    for (i, k, j), p in post.items():
        sp[(i, j)] = sp.get((i, j), 0.0) + p
    return sp


def goodman_binary(n: int, span_post) -> Dict[Tuple[int, int], int]:
    """Binary tree maximising sum of span posteriors; returns split[(i, j)] = k."""
    best = {(i, i + 1): 0.0 for i in range(n)}
    arg: Dict[Tuple[int, int], int] = {}
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            k = max(range(i + 1, j), key=lambda k: best[(i, k)] + best[(k, j)])
            arg[(i, j)] = k
            best[(i, j)] = best[(i, k)] + best[(k, j)] + span_post.get((i, j), 0.0)
    split: Dict[Tuple[int, int], int] = {}

    def walk(i, j):
        if j - i < 2:
            return
        split[(i, j)] = arg[(i, j)]
        walk(i, arg[(i, j)])
        walk(arg[(i, j)], j)
    walk(0, n)
    return split


def goodman_nary(n: int, span_post, lam: float) -> Set[Tuple[int, int]]:
    """Non-crossing spans (2..n-1 words) maximising sum_{included} (P(span) - lam). Spans are laminar, so a binary
    CKY where every node may be included for free-or-gain (max(0, P - lam)) represents every such set."""
    best = {(i, i + 1): 0.0 for i in range(n)}
    arg: Dict[Tuple[int, int], int] = {}
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            k = max(range(i + 1, j), key=lambda k: best[(i, k)] + best[(k, j)])
            arg[(i, j)] = k
            gain = 0.0 if length == n else max(0.0, span_post.get((i, j), 0.0) - lam)
            best[(i, j)] = best[(i, k)] + best[(k, j)] + gain
    out: Set[Tuple[int, int]] = set()

    def walk(i, j):
        if j - i < 2:
            return
        if j - i < n and span_post.get((i, j), 0.0) - lam > 0.0:
            out.add((i, j))
        walk(i, arg[(i, j)])
        walk(arg[(i, j)], j)
    walk(0, n)
    return out


class Trellis3R203(Trellis3):
    def __init__(self, *args, edge_w: Optional[float] = None, goodman: bool = False, **kw):
        super().__init__(*args, **kw)
        self.edge_w = edge_w
        self.goodman = goodman

    # -- (a) the sentence edge counted once, slots kept ---------------------------------------------------------
    def _edge_weights(self, x: dict) -> Optional[Dict[str, float]]:
        """Per-attribute weights for the context gain, or None when no edge group of >= 2 slots is present."""
        w: Optional[Dict[str, float]] = None
        for side, mark in (("l", BOS), ("r", EOS)):
            m = 0
            while x.get(f"{side}{m + 1}") == mark:
                m += 1
            if m >= 2:
                w = w if w is not None else {}
                for k in range(1, m + 1):
                    w[f"{side}{k}"] = self.edge_w / m
        return w

    def _wlog(self, node, x, root, wts) -> float:
        tot = 0.0
        for a, v in x.items():
            d = node.av.get(a, {})
            size = len(root.av.get(a, {})) + 1
            al = self.alpha if self.alpha >= 0 else -self.alpha / size
            wa = wts.get(a, 1.0)
            if isinstance(v, dict):
                t = sum(p * math.log((d.get(u, 0.0) + al) / (node.count + al * size)) for u, p in v.items())
            else:
                t = math.log((d.get(v, 0.0) + al) / (node.count + al * size))
            tot += wa * t
        return tot

    def _gain(self, tree, x: dict) -> float:
        if self.edge_w is None or tree is not self.ctx or self.gain_alpha is not None:
            return super()._gain(tree, x)
        wts = self._edge_weights(x)
        if wts is None:
            return super()._gain(tree, x)
        root = tree.root
        if root.count <= 0:
            return 0.0
        base = self._wlog(root, x, root, wts)
        best = 0.0
        for node in tree.categorize(x)[1:]:
            if node.count <= 0:
                continue
            best = max(best, (math.log(node.count / root.count) + self._wlog(node, x, root, wts) - base) / LN2)
        return best

    # -- (b) Goodman labelled-recall binary readout ---------------------------------------------------------------
    def _mbr(self, n: int, post):
        if not self.goodman:
            return super()._mbr(n, post)
        split = goodman_binary(n, span_posteriors(post))
        chosen: List[Tuple[int, int, int]] = []

        def walk(i, j):
            if j - i < 2:
                return
            k = split[(i, j)]
            chosen.append((i, k, j))
            walk(i, k)
            walk(k, j)
        walk(0, n)
        return Tree(n, split, {}, [(0, n)]), chosen
