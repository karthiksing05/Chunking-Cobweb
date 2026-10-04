"""Reading a probabilistic grammar out of the two hierarchies.

Symbols are the nodes of a cut through the representation tree (possibly
joined by Bayesian model merging); rule classes are the nodes of a cut
through the composition tree. The grammar is the factored PCFG

    P(A -> w)   = sum_c U[A,c] pk[c] E[c,w]
    P(A -> B C) = sum_c U[A,c] (1 - pk[c]) Lt[c,B] Rt[c,C]

In a domain whose parts are joined by typed relations (a board, where the
second part sits in a given direction and distance from the first), a
composite also draws its relation, Rel[c,r], and the composition hierarchy
describes compositions by their relation too. Sequences have one relation
(concatenation), and the formulas above are unchanged.

with every table the Dirichlet-multinomial posterior predictive (concentration
``alpha``) of counts aggregated through the cuts. Cuts are chosen by minimum
description length: the code length of the training derivations is the sum of
Dirichlet-multinomial marginal likelihoods of those tables (a prequential code
that does not depend on presentation order).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

from .cobweb import CobwebNode, CobwebTree
from .mdl import elias_delta_bits, split_bits
from .memory import Memory

LN2 = float(np.log(2.0))
UNK = "<unk>"
COMPOSITION_ATTRS = ("kind", "tok", "L", "R")
RELATIONAL_COMPOSITION_ATTRS = ("kind", "tok", "L", "rel", "R")


# --------------------------------------------------------------------------- #
# Code lengths
# --------------------------------------------------------------------------- #
def dm_code(groups: np.ndarray, keys: np.ndarray, weights: np.ndarray,
            alphabet: int, alpha: float) -> float:
    """-log marginal likelihood (nats) of ``keys`` under one Dirichlet-multinomial
    per group, each over ``alphabet`` outcomes with concentration ``alpha``."""
    if groups.size == 0:
        return 0.0
    stride = np.int64(alphabet)
    gk = groups.astype(np.int64) * stride + keys.astype(np.int64)
    uniq, inv = np.unique(gk, return_inverse=True)
    cnt = np.bincount(inv, weights=weights)
    _, ginv = np.unique(uniq // stride, return_inverse=True)
    totals = np.bincount(ginv, weights=cnt)
    a_tot = alphabet * alpha
    return float(np.sum(gammaln(totals + a_tot) - gammaln(a_tot))
                 - np.sum(gammaln(cnt + alpha) - gammaln(alpha)))


# --------------------------------------------------------------------------- #
# Cuts through a Cobweb tree
# --------------------------------------------------------------------------- #
class TreeIndex:
    """Array view of a Cobweb tree. Leaves are numbered in pre-order, so the
    leaves under any node form a contiguous range ``[lo, hi)``."""

    def __init__(self, tree: CobwebTree):
        self.tree = tree
        self.nodes: List[CobwebNode] = list(tree.nodes())
        pos = {n.id: i for i, n in enumerate(self.nodes)}
        n_nodes = len(self.nodes)
        self.children = [[pos[c.id] for c in n.children] for n in self.nodes]
        self.parent = np.full(n_nodes, -1, dtype=np.int64)
        for i, ch in enumerate(self.children):
            for c in ch:
                self.parent[c] = i
        self.node_of_leaf = np.array([i for i in range(n_nodes) if not self.children[i]],
                                     dtype=np.int64)
        self.leafpos_of_id: Dict[int, int] = {
            self.nodes[i].id: p for p, i in enumerate(self.node_of_leaf)}
        self.lo = np.zeros(n_nodes, dtype=np.int64)
        self.hi = np.zeros(n_nodes, dtype=np.int64)
        for i in reversed(range(n_nodes)):  # children come after parents in pre-order
            if not self.children[i]:
                p = self.leafpos_of_id[self.nodes[i].id]
                self.lo[i], self.hi[i] = p, p + 1
            else:
                self.lo[i] = min(self.lo[c] for c in self.children[i])
                self.hi[i] = max(self.hi[c] for c in self.children[i])

    @property
    def n_leaves(self) -> int:
        return len(self.node_of_leaf)

    def assign(self, cut: Sequence[int]) -> np.ndarray:
        """Leaf position -> index of the cut node above it."""
        arr = np.full(self.n_leaves, -1, dtype=np.int64)
        for x in cut:
            arr[self.lo[x]:self.hi[x]] = x
        covered = int(sum(self.hi[x] - self.lo[x] for x in cut))
        if covered != self.n_leaves or (arr < 0).any():
            raise ValueError("cut does not cover every leaf exactly once")
        return arr


def evidence_cut(index: TreeIndex, alpha: float = 0.5, beta: float = 0.5) -> List[int]:
    """Bottom-up DP for the cut that best codes the tree's own instances.

    Keeping node x as one cluster costs -log DM of its attribute counts;
    descending costs the children's best codes plus -log DM of which child each
    instance belongs to. Used as the starting point of the grammar search.
    """
    nodes = index.nodes
    n_nodes = len(nodes)
    own = np.asarray(index.tree.concept_codes(alpha))
    best = np.zeros(n_nodes)
    keep = np.zeros(n_nodes, dtype=bool)
    for i in reversed(range(n_nodes)):
        ch = index.children[i]
        if not ch:
            best[i], keep[i] = own[i], True
            continue
        cnt = np.array([nodes[c].count for c in ch])
        b_tot = len(ch) * beta
        member = (gammaln(cnt.sum() + b_tot) - gammaln(b_tot)
                  - np.sum(gammaln(cnt + beta) - gammaln(beta)))
        descend = member + best[ch].sum()
        if own[i] <= descend:
            best[i], keep[i] = own[i], True
        else:
            best[i] = descend
    cut, stack = [], [0]
    while stack:
        i = stack.pop()
        if keep[i]:
            cut.append(i)
        else:
            stack.extend(index.children[i])
    return sorted(cut)


def fine_cut(index: TreeIndex, max_count: float) -> List[int]:
    """The coarsest cut whose nodes each hold at most ``max_count`` instances.

    A specific starting point for the search: merging (collapse) moves can
    then reach any coarser grammar one subtree at a time, whereas refining a
    coarse mixed node pays off only after several refinements in a row.
    """
    cut, stack = [], [0]
    while stack:
        i = stack.pop()
        if not index.children[i] or index.nodes[i].count <= max_count:
            cut.append(i)
        else:
            stack.extend(index.children[i])
    return sorted(cut)


def search_cut(index: TreeIndex, cost: Callable[[np.ndarray], float],
               starts: Dict[str, List[int]]) -> Tuple[List[int], float, str, int]:
    """Hill-climb from several starting cuts; keep the shortest code."""
    best = None
    for name, start in starts.items():
        cut, value, moves = hill_climb(index, start, cost)
        if best is None or value < best[1]:
            best = (cut, value, name, moves)
    return best


def merge_symbols(arr: np.ndarray, cost: Callable[[np.ndarray], float],
                  tol: float = 1e-6) -> Tuple[np.ndarray, float, int]:
    """Bayesian model merging over symbols (Stolcke & Omohundro, 1994).

    ``arr`` maps each leaf to the cut node that stands for its symbol. Merging
    two symbols relabels one node's leaves with the other's. Repeatedly apply
    the merge that shortens the code most, until none does. Unlike a collapse,
    a merge may join symbols that are not siblings in the hierarchy.
    """
    current = cost(arr)
    merges = 0
    while True:
        syms = np.unique(arr)
        masks = {s: arr == s for s in syms}
        best = None
        for i, a in enumerate(syms):
            for b in syms[i + 1:]:
                trial = arr.copy()
                trial[masks[b]] = a
                value = cost(trial)
                if value < current - tol and (best is None or value < best[0]):
                    best = (value, a, b)
        if best is None:
            return arr, current, merges
        value, a, b = best
        arr = arr.copy()
        arr[arr == b] = a
        current = value
        merges += 1


def hill_climb(index: TreeIndex, cut: Sequence[int],
               cost: Callable[[np.ndarray], float],
               max_passes: int = 50, tol: float = 1e-6) -> Tuple[List[int], float, int]:
    """Minimise ``cost(leaf -> cut node)`` over cuts by first-improvement passes.

    Moves: *refine* replaces a cut node by its children; *collapse* replaces
    every cut node below a node by that node. Returns (cut, cost, #moves).
    """
    cutset = set(cut)
    arr = index.assign(sorted(cutset))
    current = cost(arr)
    n_moves = 0
    for _ in range(max_passes):
        improved = False
        candidates = [("refine", x) for x in sorted(cutset) if index.children[x]]
        parents = sorted({int(index.parent[x]) for x in cutset if index.parent[x] >= 0})
        candidates += [("collapse", p) for p in parents]
        for move, x in candidates:
            lo, hi = index.lo[x], index.hi[x]
            if move == "refine":
                if x not in cutset:
                    continue
                trial = arr.copy()
                for c in index.children[x]:
                    trial[index.lo[c]:index.hi[c]] = c
            else:
                if x in cutset:
                    continue
                inside = np.unique(arr[lo:hi])
                # Skip if x lies under a cut node (that node is not inside x).
                if np.any(index.lo[inside] < lo) or np.any(index.hi[inside] > hi):
                    continue
                trial = arr.copy()
                trial[lo:hi] = x
            value = cost(trial)
            if value < current - tol:
                if move == "refine":
                    cutset.discard(x)
                    cutset.update(index.children[x])
                else:
                    cutset -= set(inside.tolist())
                    cutset.add(x)
                arr, current = trial, value
                improved = True
                n_moves += 1
        if not improved:
            break
    return sorted(cutset), current, n_moves


# --------------------------------------------------------------------------- #
# The grammar
# --------------------------------------------------------------------------- #
@dataclass
class Grammar:
    vocab: List[str]
    S: np.ndarray       # (K,)   symbol of a sentence analysed as one tree
    U: np.ndarray       # (K, M) P(rule class | symbol)
    pk: np.ndarray      # (M,)   P(primitive | rule class)
    Lt: np.ndarray      # (M, K) P(left child symbol | rule class, composite)
    Rt: np.ndarray      # (M, K) P(right child symbol | rule class, composite)
    E: np.ndarray       # (M, V) P(token | rule class, primitive)
    alpha: float
    # A sentence is one tree (probability p_whole) or a partial analysis: a
    # forest of two or more pieces. The first piece's symbol comes from
    # S_piece and each next one's from T_piece[previous piece's symbol];
    # from the second piece on, the forest ends after a piece of symbol B
    # with probability stop_piece[B]. (Without T_piece and stop_piece every
    # piece is drawn from S_piece and the forest ends with p_stop.)
    p_stop: float = 1.0
    p_whole: float = 1.0
    S_piece: Optional[np.ndarray] = None
    T_piece: Optional[np.ndarray] = None
    stop_piece: Optional[np.ndarray] = None
    # The representation-tree nodes that make up each symbol (one node unless
    # model merging joined several).
    symbol_nodes: List[List[CobwebNode]] = field(default_factory=list)
    rule_nodes: List[CobwebNode] = field(default_factory=list)
    info: Dict[str, float] = field(default_factory=dict)
    # Training-set statistics kept for inspection.
    symbol_yields: List[Counter] = field(default_factory=list)
    rule_keys: List[Counter] = field(default_factory=list)
    # Symbol of every training element, and a finer node of the representation
    # hierarchy above it (used to re-describe the elements next round).
    elem_symbol: Optional[np.ndarray] = None
    elem_fine: Optional[np.ndarray] = None
    # The node above every training element in the representation tree's
    # evidence-optimal cut (the tree's own clusters; used to propose chunks).
    elem_evidence: Optional[np.ndarray] = None
    # The ancestor of every training element at depths 1..D of the
    # representation tree (column d-1 = depth d): every node is a candidate
    # category when proposing chunks.
    elem_depth: Optional[np.ndarray] = None
    # Rule class of every training element, and its composition leaf.
    elem_rule: Optional[np.ndarray] = None
    elem_rule_fine: Optional[np.ndarray] = None
    # Typed relations (None for sequences): Rel[c, r] = P(relation r | rule
    # class c, composite), over ``relations``. A board's top level is read
    # square by square: Q[q, A] = P(top-level symbol A anchored at square q),
    # with column K for an empty square.
    relations: Optional[List[Hashable]] = None
    Rel: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None

    def __post_init__(self):
        self.tok_index = {t: i for i, t in enumerate(self.vocab)}
        self.qk = 1.0 - self.pk
        if self.S_piece is None:
            self.S_piece = self.S
        if self.T_piece is None:
            self.T_piece = np.tile(self.S_piece, (len(self.S_piece), 1))
        if self.stop_piece is None:
            self.stop_piece = np.full(len(self.S_piece), self.p_stop)
        with np.errstate(divide="ignore"):
            self.log_stop = float(np.log(self.p_stop))
            self.log_cont = float(np.log1p(-self.p_stop))
            self.log_whole = float(np.log(self.p_whole))
            self.log_forest = float(np.log1p(-self.p_whole))
            self.log_T = np.log(self.T_piece)
            self.log_stop_v = np.log(self.stop_piece)
            self.log_cont_v = np.log1p(-self.stop_piece)

    @property
    def K(self) -> int:
        return self.S.shape[0]

    @property
    def M(self) -> int:
        return self.pk.shape[0]

    def token_ids(self, tokens: Sequence[str]) -> np.ndarray:
        unk = self.tok_index[UNK]
        return np.array([self.tok_index.get(t, unk) for t in tokens], dtype=np.int64)

    def tempered(self, temperature: float) -> "Grammar":
        """The same grammar with every probability raised to 1/T (unnormalised).
        Charts built from it sample trees from P(tree | sentence)^(1/T)."""
        if temperature == 1.0:
            return self
        p = 1.0 / temperature
        g = Grammar(vocab=self.vocab, S=self.S ** p, U=self.U ** p, pk=self.pk ** p,
                    Lt=self.Lt ** p, Rt=self.Rt ** p, E=self.E ** p, alpha=self.alpha,
                    p_stop=self.p_stop, p_whole=self.p_whole, S_piece=self.S_piece ** p,
                    T_piece=self.T_piece ** p, stop_piece=self.stop_piece, info=dict(self.info))
        g.qk = self.qk ** p
        g.log_stop, g.log_cont = self.log_stop * p, self.log_cont * p
        g.log_whole, g.log_forest = self.log_whole * p, self.log_forest * p
        g.log_stop_v, g.log_cont_v = self.log_stop_v * p, self.log_cont_v * p
        return g

    def lexical(self, token_id: int) -> np.ndarray:
        """Inside probabilities of a single token, one per symbol."""
        return self.U @ (self.pk * self.E[:, token_id])

    def sample(self, rng: np.random.Generator, max_len: int = 40, whole_only: bool = False):
        """Sample (tokens, Tree with symbol labels) from the grammar; None if the
        sentence exceeds ``max_len`` tokens (callers resample). With
        ``whole_only`` the sentence is one tree: the grammar's distribution
        given that it derives the whole sentence."""
        from .data import Tree
        if whole_only or rng.random() < self.p_whole:
            tops = [int(rng.choice(self.K, p=self.S))]
        else:
            tops = [int(rng.choice(self.K, p=self.S_piece))]
            tops.append(int(rng.choice(self.K, p=self.T_piece[tops[-1]])))
            while rng.random() >= self.stop_piece[tops[-1]]:
                tops.append(int(rng.choice(self.K, p=self.T_piece[tops[-1]])))
        # Expand depth-first, left to right; spans are filled in afterwards.
        tokens: List[str] = []
        nodes: List[list] = []  # [symbol, rule, token or None, left idx, right idx]
        roots = []
        for top in tops:
            stack = [(top, -1, 0)]  # (symbol, parent node idx, side)
            while stack:
                sym, parent, side = stack.pop()
                c = int(rng.choice(self.M, p=self.U[sym]))
                idx = len(nodes)
                if parent < 0:
                    roots.append(idx)
                else:
                    nodes[parent][3 + side] = idx
                if rng.random() < self.pk[c]:
                    w = int(rng.choice(len(self.vocab), p=self.E[c]))
                    nodes.append([sym, c, self.vocab[w], -1, -1])
                    tokens.append(self.vocab[w])
                    if len(tokens) > max_len:
                        return None
                else:
                    nodes.append([sym, c, None, -1, -1])
                    b = int(rng.choice(self.K, p=self.Lt[c]))
                    d = int(rng.choice(self.K, p=self.Rt[c]))
                    stack.append((d, idx, 1))
                    stack.append((b, idx, 0))
        # Recover spans: leaves were emitted left to right in DFS order.
        split, label = {}, {}
        cursor = [0]

        def span_of(idx):
            sym, _, tok, li, ri = nodes[idx]
            if tok is not None:
                i = cursor[0]
                cursor[0] += 1
                label[(i, i + 1)] = sym
                return i, i + 1
            i, k = span_of(li)
            _, j = span_of(ri)
            split[(i, j)] = k
            label[(i, j)] = sym
            return i, j

        root_spans = [span_of(r) for r in roots]
        return tokens, Tree(len(tokens), split, label, root_spans)

    def describe(self, top: int = 6) -> str:
        """Human-readable summary of symbols and rule classes."""
        lines = [f"Grammar: {self.K} symbols, {self.M} rule classes, "
                 f"{len(self.vocab)} tokens, alpha={self.alpha:g}"]
        for k, v in self.info.items():
            lines.append(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append("Symbols (most frequent yields):")
        for a, yields in enumerate(self.symbol_yields):
            total = sum(yields.values())
            shown = ", ".join(f"{y} ({n:g})" for y, n in yields.most_common(top))
            lines.append(f"  S{a:<3d} n={total:<6g} start={self.S[a]:.3f}  {shown}")
        lines.append("Rule classes (most frequent compositions):")
        for c, keys in enumerate(self.rule_keys):
            total = sum(keys.values())
            shown = ", ".join(f"{k} ({n:g})" for k, n in keys.most_common(top))
            lines.append(f"  R{c:<3d} n={total:<6g} prim={self.pk[c]:.2f}  {shown}")
        return "\n".join(lines)


@dataclass
class _Elements:
    """Element records as arrays (one row per learned element)."""
    leafpos: np.ndarray
    prim: np.ndarray
    tok: np.ndarray       # token id for primitives, -1 otherwise
    left: np.ndarray
    right: np.ndarray
    root: np.ndarray
    whole: np.ndarray     # a top-level element with no top-level neighbours
    prev: np.ndarray      # the previous top-level element (-1 if none)
    last: np.ndarray      # no top-level element follows
    w: np.ndarray
    rel: np.ndarray       # relation id of composites (0 for primitives)
    n_rel: int            # 1 for sequences


def _elements(mem: Memory, leaves: Sequence[CobwebNode], rindex: TreeIndex,
              tok_index: Dict[str, int]) -> _Elements:
    relations = getattr(mem, "relations", None)
    rel_index = {r: i for i, r in enumerate(relations or [])}
    return _Elements(
        leafpos=np.array([rindex.leafpos_of_id[l.id] for l in leaves], dtype=np.int64),
        prim=np.array([k == Memory.PRIMITIVE for k in mem.kind]),
        tok=np.array([tok_index[t] if t is not None else -1 for t in mem.token], dtype=np.int64),
        left=np.array(mem.left, dtype=np.int64),
        right=np.array(mem.right, dtype=np.int64),
        root=np.array(mem.is_root),
        whole=np.array([r and a < 0 and b < 0
                        for r, a, b in zip(mem.is_root, mem.top_left, mem.top_right)], dtype=bool),
        prev=np.array(mem.top_left, dtype=np.int64),
        last=np.array([b < 0 for b in mem.top_right], dtype=bool),
        w=np.array(mem.weight, dtype=float),
        rel=np.array([rel_index.get(r, 0) for r in mem.relation], dtype=np.int64),
        n_rel=max(len(rel_index), 1),
    )


def _compact(nodes_of_elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(nodes_of_elements, return_inverse=True)
    return uniq, inv.astype(np.int64)


def _plain_pcfg_code(el: _Elements, V: int, alpha: float, mem: Memory
                     ) -> Callable[[np.ndarray], float]:
    """Code length of the derivations under a plain PCFG over the candidate symbols."""
    prim, comp = el.prim, ~el.prim
    cl, cr = el.left[comp], el.right[comp]
    board = hasattr(mem, "top_level_nats")

    def cost(leaf_to_node: np.ndarray) -> float:
        _, s = _compact(leaf_to_node[el.leafpos])
        K = int(s.max()) + 1
        key = np.empty_like(s)
        key[prim] = el.tok[prim]
        if el.n_rel == 1:
            key[comp] = V + s[cl] * K + s[cr]
        else:
            key[comp] = V + (s[cl] * K + s[cr]) * el.n_rel + el.rel[comp]
        top = mem.top_level_nats(s, K, alpha) if board else _sentence_top_nats(el, s, K, alpha)
        return top + dm_code(s, key, el.w, V + K * K * el.n_rel, alpha)
    return cost


def _sentence_top_nats(el: _Elements, s: np.ndarray, K: int, alpha: float) -> float:
    """The symbols at a sentence's top level: the roots of sentences analysed
    as one tree; the first piece of each forest; each later piece given the
    previous piece's symbol; and after each later piece, whether the forest
    ends, given its symbol."""
    whole, piece = el.whole, el.root & ~el.whole
    first, later = piece & (el.prev < 0), piece & (el.prev >= 0)
    return (dm_code(np.zeros(int(whole.sum()), dtype=np.int64), s[whole], el.w[whole], K, alpha)
            + dm_code(np.zeros(int(first.sum()), dtype=np.int64), s[first], el.w[first], K, alpha)
            + dm_code(s[el.prev[later]], s[later], el.w[later], K, alpha)
            + dm_code(s[later], el.last[later].astype(np.int64), el.w[later], 2, alpha))


def _beta_nats(n: np.ndarray, alpha: float) -> float:
    """Code of a two-outcome row (Beta-binomial, concentration alpha each)."""
    return float(gammaln(n.sum() + 2 * alpha) - gammaln(2 * alpha)
                 - np.sum(gammaln(n + alpha) - gammaln(alpha)))


def _factored_code(el: _Elements, s: np.ndarray, K: int, V: int, alpha: float,
                   cleafpos: np.ndarray) -> Callable[[np.ndarray], float]:
    """Code length of the derivations under the factored grammar (start term
    omitted: it does not depend on the composition cut)."""
    prim, comp = el.prim, ~el.prim
    sl, sr = s[el.left[comp]], s[el.right[comp]]
    kind = comp.astype(np.int64)

    def cost(leaf_to_node: np.ndarray) -> float:
        _, c = _compact(leaf_to_node[cleafpos])
        M = int(c.max()) + 1
        nats = (dm_code(s, c, el.w, M, alpha)
                + dm_code(c, kind, el.w, 2, alpha)
                + dm_code(c[prim], el.tok[prim], el.w[prim], V, alpha)
                + dm_code(c[comp], sl, el.w[comp], K, alpha)
                + dm_code(c[comp], sr, el.w[comp], K, alpha))
        if el.n_rel > 1:
            nats += dm_code(c[comp], el.rel[comp], el.w[comp], el.n_rel, alpha)
        return nats
    return cost


def _composition_instance(key: tuple) -> Dict[str, Hashable]:
    relational = len(key) == 4 or (key[0] == "P" and len(key) == 3)
    if key[0] == "P":
        x = {"kind": "P", "tok": key[1], "L": "-", "R": "-"}
    else:
        x = {"kind": "C", "tok": "-", "L": key[1], "R": key[2]}
    if relational:
        x["rel"] = "-" if key[0] == "P" else str(key[3])
    return x


def compile_grammar(mem: Memory, rtree: CobwebTree, leaves: Sequence[CobwebNode],
                    alpha: float = 0.001, seed: int = 0, search: bool = True,
                    merge: bool = True, evidence_alpha: float = 0.5,
                    fine_divisor: float = 200.0) -> Grammar:
    """Consolidate memory into a grammar (cuts chosen by description length).

    The representation-cut search starts both from the evidence cut and from
    a fine cut (nodes holding at most 1/``fine_divisor`` of the elements) and
    keeps the shorter code; Bayesian model merging may then join symbols that
    are not siblings. The composition-cut search starts from its evidence cut
    and from its leaves.
    """
    if len(mem) == 0:
        raise ValueError("memory is empty")
    vocab = mem.vocabulary() + [UNK]
    tok_index = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    info: Dict[str, float] = {}

    # Stage A: symbols = a cut through the representation tree.
    rindex = TreeIndex(rtree)
    el = _elements(mem, leaves, rindex, tok_index)
    rcost = _plain_pcfg_code(el, V, alpha, mem)
    starts = {"evidence": evidence_cut(rindex, alpha=evidence_alpha),
              "fine": fine_cut(rindex, max(2.0, el.w.sum() / fine_divisor))}
    info["R leaves"] = rindex.n_leaves
    for name, start in starts.items():
        info[f"R {name} start: symbols"] = len(start)
        info[f"R {name} start: bits"] = rcost(rindex.assign(start)) / LN2
    rcut = starts["evidence"]
    if search:
        rcut, value, name, moves = search_cut(rindex, rcost, starts)
        info["R search start"] = name
        info["R search moves"] = moves
        info["bits (plain PCFG, chosen cut)"] = value / LN2
    rarr = rindex.assign(rcut)
    if search and merge:
        rarr, value, merges = merge_symbols(rarr, rcost)
        info["R merges"] = merges
        info["bits (plain PCFG, after merges)"] = value / LN2
    _, fine = _compact(rindex.assign(starts["fine"])[el.leafpos])
    _, evidence = _compact(rindex.assign(starts["evidence"])[el.leafpos])
    depth_of = np.zeros(len(rindex.nodes), dtype=np.int64)
    for i in range(1, len(rindex.nodes)):          # parents precede children
        depth_of[i] = depth_of[rindex.parent[i]] + 1
    max_depth = 8
    leaf_anc = np.zeros((rindex.n_leaves, max_depth), dtype=np.int64)
    for p, x in enumerate(rindex.node_of_leaf):
        path = [x]
        while rindex.parent[path[-1]] >= 0:
            path.append(int(rindex.parent[path[-1]]))
        path.reverse()                              # root ... leaf
        for d in range(1, max_depth + 1):
            leaf_anc[p, d - 1] = path[min(d, len(path) - 1)]
    elem_depth = leaf_anc[el.leafpos]
    sym_reps, s = _compact(rarr[el.leafpos])
    K = len(sym_reps)
    members = defaultdict(list)
    for x in rcut:
        members[int(rarr[rindex.lo[x]])].append(rindex.nodes[x])
    symbol_nodes = [members[int(r)] for r in sym_reps]

    # Stage B: rebuild the composition tree over compositions in symbol terms.
    relational = el.n_rel > 1
    keys: List[tuple] = []
    for e in range(len(s)):
        if el.prim[e]:
            keys.append(("P", vocab[el.tok[e]]) + (("-",) if relational else ()))
        else:
            keys.append(("C", int(s[el.left[e]]), int(s[el.right[e]]))
                        + ((mem.relations[el.rel[e]],) if relational else ()))
    weight_of: Dict[tuple, float] = defaultdict(float)
    for key, w in zip(keys, el.w):
        weight_of[key] += w
    C = CobwebTree(RELATIONAL_COMPOSITION_ATTRS if relational else COMPOSITION_ATTRS, seed=seed)
    leaf_of_key = {}
    for key in sorted(weight_of, key=lambda k: (-weight_of[k], str(k))):
        leaf_of_key[key] = C.ifit(_composition_instance(key), weight_of[key])
    cindex = TreeIndex(C)
    cleafpos = np.array([cindex.leafpos_of_id[leaf_of_key[k].id] for k in keys], dtype=np.int64)
    ccost = _factored_code(el, s, K, V, alpha, cleafpos)
    cstarts = {"evidence": evidence_cut(cindex, alpha=evidence_alpha),
               "leaves": [int(i) for i in cindex.node_of_leaf]}
    ccut = cstarts["evidence"]
    info["C leaves"] = cindex.n_leaves
    if search:
        ccut, value, name, moves = search_cut(cindex, ccost, cstarts)
        info["C search start"] = name
        info["C search moves"] = moves
    rule_nodes, c = _compact(cindex.assign(ccut)[cleafpos])
    M = len(rule_nodes)

    # Tables: Dirichlet-multinomial posterior predictives.
    w = el.w
    prim, comp = el.prim, ~el.prim
    board = hasattr(mem, "top_level_nats")
    whole, piece = (el.root, el.root & False) if board else (el.whole, el.root & ~el.whole)
    n_start = np.bincount(s[whole], weights=w[whole], minlength=K)
    first, later = piece & (el.prev < 0), piece & (el.prev >= 0)
    n_piece = np.bincount(s[first], weights=w[first], minlength=K)
    n_T = np.zeros((K, K))
    np.add.at(n_T, (s[el.prev[later]], s[later]), w[later])
    n_end = np.zeros((K, 2))                       # columns: the forest ends, continues
    np.add.at(n_end, (s[later], (~el.last[later]).astype(np.int64)), w[later])
    # Top level of a sentence: one tree or a forest; a forest of m pieces has
    # m - 2 continues and one stop.
    sent = np.array(mem.sentence_of)
    m = np.bincount(sent[el.root], minlength=len(mem.sentences))
    sent_w = np.zeros(len(mem.sentences))
    sent_w[sent[el.root]] = w[el.root]
    forest = m >= 2
    n_mode = np.array([sent_w[(m == 1)].sum(), sent_w[forest].sum()])
    n_stop_cont = np.array([sent_w[forest].sum(), (sent_w * np.maximum(m - 2, 0)).sum()])
    n_U = np.zeros((K, M))
    np.add.at(n_U, (s, c), w)
    n_prim = np.bincount(c[prim], weights=w[prim], minlength=M)
    n_comp = np.bincount(c[comp], weights=w[comp], minlength=M)
    n_E = np.zeros((M, V))
    np.add.at(n_E, (c[prim], el.tok[prim]), w[prim])
    n_L = np.zeros((M, K))
    n_R = np.zeros((M, K))
    np.add.at(n_L, (c[comp], s[el.left[comp]]), w[comp])
    np.add.at(n_R, (c[comp], s[el.right[comp]]), w[comp])

    def normalize(n, axis=-1):
        n = n + alpha
        return n / n.sum(axis=axis, keepdims=True)

    n_Rel = np.zeros((M, el.n_rel))
    np.add.at(n_Rel, (c[comp], el.rel[comp]), w[comp])
    if board:
        # A board is read square by square: the top level is the scan code.
        total_code = mem.top_level_nats(s, K, alpha) + ccost(cindex.assign(ccut))
        info["bits (factored grammar)"] = total_code / LN2
    else:
        total_code = _sentence_top_nats(el, s, K, alpha) + ccost(cindex.assign(ccut))
        info["bits (factored grammar)"] = total_code / LN2
        info["bits (factored grammar)"] += _beta_nats(n_mode, alpha) / LN2
    # The receiver also needs the grammar's size; Elias codes make the total
    # an actual message length.
    info["structure bits"] = elias_delta_bits(K) + elias_delta_bits(M)
    info["total bits"] = info["bits (factored grammar)"] + info["structure bits"]
    info["bits per sentence"] = info["total bits"] / max(len(mem.sentences), 1)
    rows = lambda table: [dict(enumerate(r)) for r in np.atleast_2d(table)]
    tables = [(rows(n_U), M), (rows(np.stack([n_prim, n_comp], axis=1)), 2),
              (rows(n_E), V), (rows(n_L), K), (rows(n_R), K)]
    if board:
        n_Q = mem.scan_counts(s, K)
        tables = [(rows(n_Q), K + 1), (rows(n_Rel), el.n_rel)] + tables
    else:
        tables = [(rows(n_start), K), (rows(n_piece), K), (rows(n_T), K), (rows(n_end), 2),
                  (rows(n_mode), 2)] + tables
    model_bits, data_bits = split_bits(tables, alpha)
    info["model bits"] = model_bits + info["structure bits"]
    info["data bits"] = data_bits
    info["symbols"] = K
    info["rule classes"] = M
    info["composite rule classes"] = int(np.sum(n_comp > 0))
    info["chunk types"] = len({(int(s[e]), int(s[el.left[e]]), int(el.rel[e]), int(s[el.right[e]]))
                               for e in np.flatnonzero(comp)})

    symbol_yields = [Counter() for _ in range(K)]
    for e in range(len(s)):
        symbol_yields[s[e]][mem.describe(e)] += w[e]
    rule_keys = [Counter() for _ in range(M)]
    for key, cc, ww in zip(keys, c, w):
        if key[0] == "P":
            label = key[1]
        elif relational:
            label = f"S{key[1]} {key[3]} S{key[2]}"
        else:
            label = f"S{key[1]} S{key[2]}"
        rule_keys[cc][label] += ww

    pk = (n_prim + alpha) / (n_prim + n_comp + 2 * alpha)
    return Grammar(
        vocab=vocab,
        S=normalize(n_start),
        U=normalize(n_U),
        pk=pk,
        p_stop=float((n_stop_cont[0] + alpha) / (n_stop_cont.sum() + 2 * alpha)),
        p_whole=float((n_mode[0] + alpha) / (n_mode.sum() + 2 * alpha)),
        S_piece=normalize(n_piece),
        T_piece=normalize(n_T) if not board else None,
        stop_piece=((n_end[:, 0] + alpha) / (n_end.sum(axis=1) + 2 * alpha)) if not board else None,
        Lt=normalize(n_L),
        Rt=normalize(n_R),
        E=normalize(n_E),
        alpha=alpha,
        symbol_nodes=symbol_nodes,
        rule_nodes=[cindex.nodes[i] for i in rule_nodes],
        info=info,
        symbol_yields=symbol_yields,
        elem_symbol=s,
        elem_fine=fine,
        elem_evidence=evidence,
        elem_depth=elem_depth,
        elem_rule=c,
        elem_rule_fine=_compact(cleafpos)[1],
        rule_keys=rule_keys,
        relations=list(mem.relations) if relational else None,
        Rel=normalize(n_Rel) if relational else None,
        Q=normalize(mem.scan_counts(s, K)) if board else None,
    )
