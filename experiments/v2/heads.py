"""Heads in both hierarchies (domain-neutral), without editing ``src/trellis2``.

A composite is described by one designated part, its **head**, and its
**valence** (on which sides it has taken dependents along its head chain).

* Representation hierarchy (``HeadMixin`` over any domain's Memory): a
  composite's instance also carries its head part's category (each
  granularity) and the surface attributes of its lexical head (the primitive
  at the end of its head chain) exactly as the domain's ``surface`` supplies
  them (sequences: neighbouring tokens, first/last token, kind; characters:
  slot, first/last component, operator); its kind attribute is refined by
  its valence.
* Composition hierarchy: a composite is (head category, dependent category,
  side) instead of (left, right).
* Grammar (``HeadGrammar``): a composite rule class c chooses a side
  sigma in {left head, right head, none}; then the head's category h, then the
  dependent's category given the head's,

      P(B, C | c) = sum_sigma P(sigma | c) H[c, sigma, h] D[c, sigma, h, d],

  with (h, d) = (B, C) for a left head and (C, B) for a right head; "none" is
  the current factorization Lt[c, B] Rt[c, C]. The tables are coded exactly
  (prequentially, events in learning order): P(h | c, sigma) and
  P(d | c, sigma) are Dirichlet-multinomial predictives (alpha), and
  P(d | c, sigma, h) backs off to P(d | c, sigma) with weight beta, as the
  read's context backs off elsewhere. Which part is the head is chosen per
  rule class by description length (``side="learned"``), with no domain
  knowledge; ``side="oracle"`` takes it per element from ``Memory.oracle_side``
  (a diagnostic upper bound). Description length also chooses beta, or no
  heads at all, which is exactly the current grammar.

``install()`` substitutes, in this process only, the names the night uses
(listed in ``SUBSTITUTED``), so the full unsupervised night runs with heads.
"""
from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

import trellis2.chart as _chart_mod
import trellis2.grammar as _grammar_mod
import trellis2.memory as _memory_mod
import trellis2.model as _model_mod
import trellis2.unsupervised as _unsup_mod
from trellis2.chart import Chart as _Chart
from trellis2.cobweb import CobwebTree
from trellis2.data import Tree
from trellis2.grammar import (CONTEXT_WEIGHTS, LN2, Grammar, TreeIndex, _compact, _pair_rows, evidence_cut,
                              search_cut)
from trellis2.mdl import _before, backoff_chain_coder, backoff_coder, dm_code, elias_delta_bits
from trellis2.memory import BLANK, Memory, chunk_attrs

LEFT, RIGHT, NONE = 0, 1, 2
SIDES = ("L", "R", "N")
HEAD_BETAS = CONTEXT_WEIGHTS          # how strongly the head's category is trusted (DL chooses)
_ORIGINAL_COMPILE = _grammar_mod.compile_grammar


# --------------------------------------------------------------------------- #
# The grammar

@dataclass
class HeadGrammar(Grammar):
    # P2[c, B, C] = P(left part B, right part C | composite rule class c).
    P2: Optional[np.ndarray] = None
    # Per rule class: P(sigma | c) over (left head, right head, none).
    side_p: Optional[np.ndarray] = None

    def log_binary(self, context=None) -> np.ndarray:
        cache = self.__dict__.setdefault("_log_binary_h", {})
        if isinstance(context, tuple) and self.Uc2 is not None and context in self.Uc2:
            key = context
        else:
            key = context[-1] if isinstance(context, tuple) else context
            key = key if key in self.context_index else None
        if key not in cache:
            with np.errstate(divide="ignore"):
                cache[key] = np.log(self.rule_tensor(key))
        return cache[key]

    def rule_tensor(self, context=None) -> np.ndarray:
        """(K, K, K): P(A -> B C) in the light of the read's context."""
        return np.einsum("ac,c,cbd->abd", self.rules(context), self.qk, self.P2)

    def tempered(self, temperature: float) -> "HeadGrammar":
        if temperature == 1.0:
            return self
        p = 1.0 / temperature
        base = Grammar.tempered(self, temperature)
        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(Grammar)}
        g = HeadGrammar(**fields, P2=self.P2 ** p, side_p=self.side_p)
        g.qk = base.qk
        g.log_stop, g.log_cont, g.log_whole, g.log_forest = base.log_stop, base.log_cont, base.log_whole, base.log_forest
        return g

    def sample(self, rng: np.random.Generator, max_len: int = 40, whole_only: bool = False):
        """As ``Grammar.sample``, with a composite's two parts drawn jointly."""
        if whole_only or rng.random() < self.p_whole:
            tops = [int(rng.choice(self.K, p=self.S))]
        else:
            tops = [int(rng.choice(self.K, p=self.S_piece)) for _ in range(2)]
            while rng.random() >= self.p_stop:
                tops.append(int(rng.choice(self.K, p=self.S_piece)))
        tokens: List[str] = []
        nodes: List[list] = []
        roots = []
        KK = self.K * self.K
        for top in tops:
            stack = [(top, -1, 0)]
            start = len(tokens) if self.fresh_pieces else 0
            while stack:
                sym, parent, side = stack.pop()
                c = int(rng.choice(self.M, p=self.rules(self.read(tokens[start:], len(tokens) - start))[sym]))
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
                    pair = self.P2[c].ravel()
                    b, d = divmod(int(rng.choice(KK, p=pair / pair.sum())), self.K)
                    stack.append((d, idx, 1))
                    stack.append((b, idx, 0))
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


# --------------------------------------------------------------------------- #
# The chart

class HeadChart(_Chart):
    """Inside-outside, Viterbi and sampling for a ``HeadGrammar``: the binary
    rule is the full tensor R[A, B, C] (no Lt x Rt factorization). Viterbi is
    inherited (it reads ``log_binary``)."""

    def __init__(self, grammar: HeadGrammar, tokens: Sequence[str]):
        g = grammar
        if g.fresh_pieces:
            raise NotImplementedError("pieces read afresh are not supported with heads")
        self.g = g
        self.tokens = list(tokens)
        ids = g.token_ids(tokens)
        n = self.n = len(ids)
        K = g.K
        a = self.a = np.zeros((n + 1, n + 1, K))
        la = self.la = np.full((n + 1, n + 1), -np.inf)
        self.lam = self.rho = None
        self.context = [g.read(self.tokens, i) for i in range(n)]
        by_context = {x: g.rules(x) for x in set(self.context)}
        self.U = [by_context[x] for x in self.context]
        tensors = {x: g.rule_tensor(x) for x in set(self.context)}
        self.R = [tensors[x] for x in self.context]
        for i in range(n):
            v = g.lexical(int(ids[i]), self.context[i])
            s = v.sum()
            if s > 0:
                a[i, i + 1] = v / s
                la[i, i + 1] = np.log(s)
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                scale = la[i, i + 1:j] + la[i + 1:j, j]
                top = scale.max()
                if not np.isfinite(top):
                    continue
                wts = np.exp(scale - top)
                v = np.einsum("abd,sb,sd,s->a", self.R[i], a[i, i + 1:j], a[i + 1:j, j], wts, optimize=True)
                s = v.sum()
                if s <= 0:
                    continue
                a[i, j] = v / s
                la[i, j] = top + np.log(s)
        self.fresh = False
        self.b, self.lb, self.lamb = a, la, None
        with np.errstate(divide="ignore"):
            self.top = self.lb + np.log(np.einsum("ijk,k->ij", self.b, g.S_piece))
            whole = la[0, n] + np.log(a[0, n] @ g.S) if n else -np.inf
        self.whole = g.log_whole + whole
        F1 = self.F1 = np.full(n + 1, -np.inf)
        H = self.H = np.full(n + 1, -np.inf)
        if n:
            F1[1:n] = g.log_forest + self.top[0, 1:n]
        for j in range(2, n + 1):
            H[j] = logsumexp(np.logaddexp(F1[1:j], g.log_cont + H[1:j]) + self.top[1:j, j])
        G = self.G = np.full(n + 1, -np.inf)
        G1 = self.G1 = np.full(n + 1, -np.inf)
        if n:
            G[n] = g.log_stop
        for j in range(n - 1, 0, -1):
            G1[j] = logsumexp(self.top[j, j + 1:] + G[j + 1:])
            G[j] = g.log_cont + G1[j]
        self.forest = H[n] + g.log_stop if n else -np.inf
        self.log_prob = float(np.logaddexp(self.whole, self.forest)) if n else 0.0
        self._mu = None

    def _down(self, mu, mu_left, mu_right, a, la, lam, U, i, j):
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(a[i, j] > 0, mu[i, j] / a[i, j], 0.0)
        G = np.einsum("a,abd->bd", w, self.R[i])
        rel = np.exp(la[i, i + 1:j] + self.la[i + 1:j, j] - la[i, j])
        al, ar = a[i, i + 1:j], self.a[i + 1:j, j]
        mu_left[i, i + 1:j] += al * (ar @ G.T) * rel[:, None]
        mu_right[i + 1:j, j] += ar * (al @ G) * rel[:, None]

    def sample_tree(self, rng: np.random.Generator) -> Tree:
        g, n, a, la = self.g, self.n, self.a, self.la
        if n == 0:
            return Tree(0, {})

        def pick(logits):
            p = np.exp(logits - logsumexp(logits))
            return int(rng.choice(len(p), p=p / p.sum()))

        if rng.random() < np.exp(self.whole - self.log_prob):
            w = g.S * a[0, n]
            roots, stack = [(0, n)], [(0, n, int(rng.choice(g.K, p=w / w.sum())))]
        else:
            ends = [1 + pick(self.F1[1:n] + self.G1[1:n])]
            while ends[-1] < n:
                i = ends[-1]
                ends.append(i + 1 + pick(self.top[i, i + 1:] + self.G[i + 1:]))
            roots, stack = [], []
            for i, j in zip([0] + ends[:-1], ends):
                w = g.S_piece * self.b[i, j]
                roots.append((i, j))
                stack.append((i, j, int(rng.choice(g.K, p=w / w.sum()))))
        split: Dict = {}
        label: Dict = {}
        K = g.K
        while stack:
            i, j, A = stack.pop()
            label[(i, j)] = A
            if j - i < 2:
                continue
            rel = np.exp(la[i, i + 1:j] + la[i + 1:j, j] - la[i, j])
            weight = (self.R[i][A][None, :, :] * a[i, i + 1:j][:, :, None] * a[i + 1:j, j][:, None, :]
                      * rel[:, None, None])
            flat = weight.ravel()
            pk_ = int(rng.choice(flat.size, p=flat / flat.sum()))
            kk, rest = divmod(pk_, K * K)
            B, C = divmod(rest, K)
            k = i + 1 + kk
            split[(i, j)] = k
            stack.append((k, j, int(C)))
            stack.append((i, k, int(B)))
        return Tree(n, split, label, roots)


def make_chart(grammar, tokens):
    """The chart for a grammar: ``HeadChart`` for a ``HeadGrammar``."""
    if isinstance(grammar, HeadGrammar):
        return HeadChart(grammar, tokens)
    return _Chart(grammar, tokens)


# --------------------------------------------------------------------------- #
# The code of the composites' parts, per rule class and side

def _phi_sum(n, alpha):
    return gammaln(n + alpha) - gammaln(alpha)


def side_codes(c: np.ndarray, M: int, left: np.ndarray, right: np.ndarray, w: np.ndarray, K: int,
               alpha: float, beta: float, groups: Optional[np.ndarray] = None,
               sides: Sequence[int] = (LEFT, RIGHT, NONE)) -> np.ndarray:
    """Nats of the composites' parts per rule class (rows) under each side
    (columns: left head, right head, none), coded prequentially in the given
    order. ``groups`` (default: the classes) are the rows the tables are kept
    per; the output is summed per class."""
    grp = c if groups is None else groups
    G = int(grp.max()) + 1 if grp.size else 1
    g64, L, R = grp.astype(np.int64), left.astype(np.int64), right.astype(np.int64)
    n_g = _before(g64, w)
    out = np.zeros((M, 3))
    if c.size == 0:
        return out
    norm = n_g + K * alpha
    for side in sides:
        if side == NONE:
            pl = (_before(g64 * K + L, w) + alpha) / norm
            pr = (_before(g64 * K + R, w) + alpha) / norm
            nats = -w * (np.log(pl) + np.log(pr))
        else:
            h, d = (L, R) if side == LEFT else (R, L)
            gh = g64 * K + h
            n_gh = _before(gh, w)
            ph = (n_gh + alpha) / norm
            pb = (_before(g64 * K + d, w) + alpha) / norm
            pd = (_before(gh * K + d, w) + beta * pb) / (n_gh + beta)
            nats = -w * (np.log(ph) + np.log(pd))
        out[:, side] = np.bincount(c, weights=nats, minlength=M)
    return out


def side_row_nats(c: np.ndarray, sigma: np.ndarray, w: np.ndarray, M: int, alpha: float) -> float:
    """Dirichlet-multinomial code of each composite's side given its class
    (alphabet 3)."""
    if c.size == 0:
        return 0.0
    return dm_code(c, sigma, w, 3, alpha)


@dataclass
class _HeadElements:
    s: np.ndarray
    c_of: np.ndarray        # composition leaf of each element (cleafpos)
    prim: np.ndarray
    tok: np.ndarray
    left: np.ndarray
    right: np.ndarray
    rel: np.ndarray
    n_rel: int
    w: np.ndarray
    ctx: Optional[np.ndarray]
    ctx2: Optional[np.ndarray]


def head_cost(el, s, K, V, alpha, cleafpos, weight, beta: Optional[float], oracle: Optional[np.ndarray] = None):
    """The factored code with heads, as a function of the composition cut
    (leaf -> node). ``beta`` None: no heads (exactly the current code of the
    parts, Lt and Rt). Returns cost(arr) and detail(arr) -> (nats, class
    sides, element sides)."""
    prim = el.prim
    comp = ~prim
    w = el.w
    kind = comp.astype(np.int64)
    comp_idx = np.flatnonzero(comp)
    Ls, Rs = s[el.left[comp_idx]], s[el.right[comp_idx]]
    wc = w[comp_idx]
    if weight is None:
        in_context = None
    elif isinstance(weight, tuple):
        in_context = backoff_chain_coder(s, [el.ctx, el.ctx2], w)
    else:
        in_context = backoff_coder(s, el.ctx, w)

    def detail(arr):
        _, c = _compact(arr[cleafpos])
        M = int(c.max()) + 1
        rules = dm_code(s, c, w, M, alpha) if in_context is None else in_context(c, M, alpha, weight)
        nats = (rules + dm_code(c, kind, w, 2, alpha)
                + dm_code(c[prim], el.tok[prim], w[prim], V, alpha))
        cc = c[comp_idx]
        if el.n_rel > 1:
            nats += dm_code(cc, el.rel[comp_idx], wc, el.n_rel, alpha)
        if beta is None:
            nats += float(side_codes(cc, M, Ls, Rs, wc, K, alpha, 1.0, sides=(NONE,))[:, NONE].sum())
            return nats, np.full(M, NONE), np.full(len(comp_idx), NONE)
        if oracle is not None:
            sig = oracle[comp_idx]
            groups = cc * 2 + sig
            # Each element's own side: sum over its side's column.
            part = np.zeros(M)
            for side in (LEFT, RIGHT):
                m = sig == side
                if m.any():
                    part += side_codes(cc[m], M, Ls[m], Rs[m], wc[m], K, alpha, beta,
                                       groups=groups[m], sides=(side,))[:, side]
            nats += float(part.sum()) + side_row_nats(cc, sig, wc, M, alpha)
            cls = np.full(M, -1)
            return nats, cls, sig
        codes = side_codes(cc, M, Ls, Rs, wc, K, alpha, beta)
        cls = codes.argmin(axis=1)
        sig = cls[cc]
        nats += float(codes.min(axis=1).sum()) + side_row_nats(cc, sig, wc, M, alpha)
        return nats, cls, sig

    return (lambda arr: detail(arr)[0]), detail


# --------------------------------------------------------------------------- #
# Compile: the original stage A, then stage B with heads

HEAD_COMPOSITION_ATTRS = ("kind", "tok", "H", "D", "side")
HEAD_RELATIONAL_ATTRS = ("kind", "tok", "H", "D", "side", "rel")


def _comp_instance(key):
    if key[0] == "P":
        x = {"kind": "P", "tok": key[1], "H": "-", "D": "-", "side": "-"}
        if len(key) == 3:
            x["rel"] = "-"
    else:
        x = {"kind": "C", "tok": "-", "H": key[1], "D": key[2], "side": key[3]}
        if len(key) == 5:
            x["rel"] = str(key[4])
    return x


# side: "learned" (per rule class, by description length) or "oracle" (Memory.oracle_side);
# representation_sides: False withholds the sides from the representation
# hierarchy (a control: the head attributes are present but blank).
CONFIG = {"side": "learned", "betas": HEAD_BETAS, "passes": 2, "representation_sides": True}


def compile_grammar_heads(mem: Memory, rtree, leaves, alpha: float = 0.001, seed: int = 0,
                          search: bool = True, merge: bool = True, **kw) -> Grammar:
    """``compile_grammar`` with heads in the composition hierarchy and the
    grammar. Stage A (the representation cut and symbols) is the original's."""
    g0 = _ORIGINAL_COMPILE(mem, rtree, leaves, alpha=alpha, seed=seed, search=search, merge=merge, **kw)
    g = compile_with_symbols_heads(mem, g0.elem_symbol, alpha=alpha, seed=seed, search=search,
                                   side=CONFIG["side"], betas=CONFIG["betas"], passes=CONFIG["passes"])
    g.symbol_nodes, g.elem_fine, g.elem_evidence, g.elem_depth = (g0.symbol_nodes, g0.elem_fine,
                                                                  g0.elem_evidence, g0.elem_depth)
    g.symbol_yields = g0.symbol_yields
    g.info["chunk types"] = g0.info["chunk types"]
    g.info["plain grammar total bits"] = g0.info["total bits"]
    if hasattr(rtree, "nodes"):
        g._rtree, g._leaves = rtree, leaves
    return g


def _elements(mem: Memory, tok_index):
    """``grammar._elements`` without the representation leaves (stage B does
    not read them)."""
    from trellis2.grammar import _Elements
    values = mem.contexts()
    pairs = None
    if values is not None and values and isinstance(values[0], tuple):
        pairs, values = values, [x[-1] for x in values]
    contexts = sorted(set(values)) if values is not None else None
    ctx_index = {x: i for i, x in enumerate(contexts or [])}
    contexts2 = sorted(set(pairs)) if pairs is not None else None
    pair_index = {x: i for i, x in enumerate(contexts2 or [])}
    el = _Elements(
        leafpos=np.zeros(len(mem), dtype=np.int64),
        prim=np.array([k == Memory.PRIMITIVE for k in mem.kind]),
        tok=np.array([tok_index[t] if t is not None else -1 for t in mem.token], dtype=np.int64),
        left=np.array(mem.left, dtype=np.int64), right=np.array(mem.right, dtype=np.int64),
        root=np.array(mem.is_root), w=np.array(mem.weight, dtype=float),
        rel=np.zeros(len(mem), dtype=np.int64), n_rel=1,
        ctx=None if values is None else np.array([ctx_index[x] for x in values], dtype=np.int64),
        contexts=contexts,
        ctx2=None if pairs is None else np.array([pair_index[x] for x in pairs], dtype=np.int64),
        contexts2=contexts2)
    relations = getattr(mem, "relations", None)
    if relations:
        rel_index = {r: i for i, r in enumerate(relations)}
        el.rel = np.array([rel_index.get(r, 0) for r in mem.relation], dtype=np.int64)
        el.n_rel = max(len(rel_index), 1)
    return el


def compile_with_symbols_heads(mem: Memory, s: np.ndarray, alpha: float = 0.001, seed: int = 0,
                               search: bool = True, side: str = "learned", betas=HEAD_BETAS,
                               passes: int = 2) -> HeadGrammar:
    from trellis2.grammar import UNK
    vocab = mem.vocabulary() + [UNK]
    tok_index = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    el = _elements(mem, tok_index)
    _, s = _compact(np.asarray(s, dtype=np.int64))
    K = int(s.max()) + 1
    relational = el.n_rel > 1
    oracle = getattr(mem, "oracle_side", None) if side == "oracle" else None
    elem_side = np.full(len(s), -1, dtype=np.int64)
    if oracle is not None:
        elem_side = np.where(el.prim, -1, oracle)
    beta0 = betas[len(betas) // 2] if betas else None
    best = None
    for p in range(passes):
        # The composition hierarchy over (head, dependent, side); before the
        # sides are known, (left, right, ?).
        keys = []
        for e in range(len(s)):
            if el.prim[e]:
                keys.append(("P", vocab[el.tok[e]]) + (("-",) if relational else ()))
                continue
            l, r, sd = int(s[el.left[e]]), int(s[el.right[e]]), int(elem_side[e])
            h, d = (r, l) if sd == RIGHT else (l, r)
            keys.append(("C", h, d, SIDES[sd] if sd >= 0 else "?")
                        + ((str(mem.relation[e]),) if relational else ()))
        weight_of: Dict[tuple, float] = defaultdict(float)
        for key, ww in zip(keys, el.w):
            weight_of[key] += ww
        C = CobwebTree(HEAD_RELATIONAL_ATTRS if relational else HEAD_COMPOSITION_ATTRS, seed=seed)
        leaf_of_key = {}
        for key in sorted(weight_of, key=lambda k: (-weight_of[k], str(k))):
            leaf_of_key[key] = C.ifit(_comp_instance(key), weight_of[key])
        cindex = TreeIndex(C)
        cleafpos = np.array([cindex.leafpos_of_id[leaf_of_key[k].id] for k in keys], dtype=np.int64)
        cstarts = {"evidence": evidence_cut(cindex, alpha=0.5), "leaves": [int(i) for i in cindex.node_of_leaf]}
        for wctx in [None] + ([CONTEXT_WEIGHTS[1]] if el.ctx is not None else []):
            for b in ([None] + ([beta0] if betas else [])):
                cost, _ = head_cost(el, s, K, V, alpha, cleafpos, wctx, b, oracle)
                cut = cstarts["evidence"]
                if search:
                    cut, _, _, _ = search_cut(cindex, cost, cstarts)
                ctx_options = [wctx] if wctx is None else list(CONTEXT_WEIGHTS)
                if wctx is not None and el.ctx2 is not None:
                    ctx_options += [(b1, b2) for b1 in CONTEXT_WEIGHTS for b2 in CONTEXT_WEIGHTS]
                beta_options = [None] if b is None else list(betas)
                for w_ in ctx_options:
                    for b_ in beta_options:
                        _, detail = head_cost(el, s, K, V, alpha, cleafpos, w_, b_, oracle)
                        value, cls, sig = detail(cindex.assign(cut))
                        if best is None or value < best[0] - 1e-9:
                            best = (value, w_, b_, cut, cindex, cleafpos, cls, sig, p)
        # The sides found become each composite's side in the next pass.
        _, _, b_, _, _, _, cls, sig, _ = best
        if b_ is None or oracle is not None:
            break
        new_side = np.full(len(s), -1, dtype=np.int64)
        new_side[np.flatnonzero(~el.prim)] = sig
        if np.array_equal(new_side, elem_side):
            break
        elem_side = new_side
    value, weight, beta, ccut, cindex, cleafpos, cls, sig, _ = best
    g = _tables(mem, el, s, K, V, vocab, alpha, weight, beta, ccut, cindex, cleafpos, sig, value,
                head_options=bool(betas))
    # The sides of every element, for the representation hierarchy's next
    # round. Where the grammar does not condition on heads, each class's head
    # is still the part (left or right) whose head code is shorter.
    rep_sig = sig
    if oracle is None and beta is None and betas:
        _, c_ = _compact(cindex.assign(ccut)[cleafpos])
        comp_idx = np.flatnonzero(~el.prim)
        cc = c_[comp_idx]
        codes = side_codes(cc, int(c_.max()) + 1, s[el.left[comp_idx]], s[el.right[comp_idx]], el.w[comp_idx],
                           K, alpha, beta0, sides=(LEFT, RIGHT))
        rep_sig = codes[:, :2].argmin(axis=1)[cc]
        g.info["representation head sides (L, R) by class"] = [int(np.sum(codes[np.unique(cc), :2].argmin(1) == k))
                                                              for k in (0, 1)]
    es = np.full(len(s), -1, dtype=np.int64)
    es[np.flatnonzero(~el.prim)] = rep_sig
    mem.elem_head_side = es
    g.elem_head_side = es
    return g


def _tables(mem, el, s, K, V, vocab, alpha, weight, beta, ccut, cindex, cleafpos, sig, value,
            head_options: bool = True) -> Grammar:
    """The tables; a ``HeadGrammar`` where the rules condition on heads, else
    the current ``Grammar`` (and, without head options, exactly its code)."""
    info: Dict[str, float] = {}
    two = isinstance(weight, tuple)
    offered2 = el.ctx2 is not None
    contexts_offered = el.contexts is not None
    if weight is None:
        el.ctx = None
    if not two:
        el.ctx2 = None
    beta1 = weight[0] if two else weight
    _, c = _compact(cindex.assign(ccut)[cleafpos])
    M = int(c.max()) + 1
    w = el.w
    prim, comp = el.prim, ~el.prim
    comp_idx = np.flatnonzero(comp)
    n_U = np.zeros((K, M))
    np.add.at(n_U, (s, c), w)
    if el.ctx is not None:
        n_Uc = np.zeros((K, len(el.contexts), M))
        np.add.at(n_Uc, (s, el.ctx, c), w)
    if el.ctx2 is not None:
        cell, row_of = np.unique(s * len(el.contexts2) + el.ctx2, return_inverse=True)
        n_Uc2 = np.zeros((len(cell), M))
        np.add.at(n_Uc2, (row_of.reshape(-1), c), w)
    n_prim = np.bincount(c[prim], weights=w[prim], minlength=M)
    n_comp = np.bincount(c[comp], weights=w[comp], minlength=M)
    n_E = np.zeros((M, V))
    np.add.at(n_E, (c[prim], el.tok[prim]), w[prim])
    cc, Ls, Rs, wc = c[comp_idx], s[el.left[comp_idx]], s[el.right[comp_idx]], w[comp_idx]

    def normalize(n):
        n = n + alpha
        return n / n.sum(axis=-1, keepdims=True)

    n_L = np.zeros((M, K))
    n_R = np.zeros((M, K))
    np.add.at(n_L, (cc, Ls), wc)
    np.add.at(n_R, (cc, Rs), wc)
    # P2 and P(side | class) from the counts: Dirichlet predictives, with the
    # dependent's row backing off to its class-and-side row.
    n_side = np.zeros((M, 3))
    np.add.at(n_side, (cc, sig), wc)
    side_p = normalize(n_side)
    P2 = np.zeros((M, K, K))
    if beta is None:
        P2 = normalize(n_L)[:, :, None] * normalize(n_R)[:, None, :]
        side_p = np.tile(np.array([0.0, 0.0, 1.0]), (M, 1))
    else:
        for side in (LEFT, RIGHT, NONE):
            m = sig == side
            nh = np.zeros((M, K))
            nd = np.zeros((M, K))
            nhd = np.zeros((M, K, K))
            if m.any():
                h, d = (Ls[m], Rs[m]) if side != RIGHT else (Rs[m], Ls[m])
                np.add.at(nh, (cc[m], h), wc[m])
                np.add.at(nd, (cc[m], d), wc[m])
                np.add.at(nhd, (cc[m], h, d), wc[m])
            H = normalize(nh)
            Dbase = normalize(nd)
            if side == NONE:
                part = H[:, :, None] * Dbase[:, None, :]
            else:
                D = (nhd + beta * Dbase[:, None, :]) / (nhd.sum(axis=2, keepdims=True) + beta)
                part = H[:, :, None] * D                        # [c, h, d]
                if side == RIGHT:
                    part = part.transpose(0, 2, 1)              # [c, left = d, right = h]
            P2 += side_p[:, side][:, None, None] * part
    top_fields, _ = mem.top_level_tables(s, K, alpha)
    total = mem.top_level_nats(s, K, alpha) + mem.layout_nats(alpha) + value
    info["bits (factored grammar)"] = total / LN2
    info["structure bits"] = elias_delta_bits(K) + elias_delta_bits(M)
    if contexts_offered:
        info["structure bits"] += float(np.log2(1 + len(CONTEXT_WEIGHTS) + (len(CONTEXT_WEIGHTS) ** 2 if offered2 else 0)))
    if head_options:
        info["structure bits"] += float(np.log2(1 + len(HEAD_BETAS)))  # no heads, or one of the betas
    info["total bits"] = info["bits (factored grammar)"] + info["structure bits"]
    info["model bits"] = float("nan")
    info["data bits"] = float("nan")
    info["symbols"], info["rule classes"] = K, M
    info["head beta"] = 0.0 if beta is None else beta
    info["read context weight"] = 0.0 if weight is None else (weight[0] if two else weight)
    comp_classes = np.flatnonzero(n_comp > 0)
    info["composite classes"] = int(comp_classes.size)
    info["classes by side (L, R, none)"] = [int(np.sum((side_p[comp_classes].argmax(1)) == k)) for k in range(3)] \
        if beta is not None else [0, 0, int(comp_classes.size)]
    info["chunk types"] = len({(int(s[e]), int(s[el.left[e]]), int(s[el.right[e]])) for e in comp_idx})
    pk = (n_prim + alpha) / (n_prim + n_comp + 2 * alpha)
    extra = {} if beta is None else {"P2": P2, "side_p": side_p}
    cls = Grammar if beta is None else HeadGrammar
    g = cls(vocab=vocab, U=normalize(n_U), pk=pk, **top_fields, Lt=normalize(n_L), Rt=normalize(n_R),
                    E=normalize(n_E), alpha=alpha, info=info, elem_symbol=s, elem_rule=c,
                    elem_rule_fine=_compact(cleafpos)[1],
                    relations=list(mem.relations) if el.n_rel > 1 else None,
                    Rel=None,
                    contexts=None if el.ctx is None else el.contexts,
                    Uc=(None if el.ctx is None else
                        (n_Uc + beta1 * normalize(n_U)[:, None, :]) / (n_Uc.sum(axis=2, keepdims=True) + beta1)),
                    Uc2=None if el.ctx2 is None else _pair_rows(n_Uc2, cell, el, n_Uc, normalize(n_U), beta1, weight[1]),
                    fresh_pieces=False, **extra)
    if el.n_rel > 1:
        n_Rel = np.zeros((M, el.n_rel))
        np.add.at(n_Rel, (c[comp], el.rel[comp]), w[comp])
        g.Rel = normalize(n_Rel)
    return g


# --------------------------------------------------------------------------- #
# The representation hierarchy: heads and valence (any domain's Memory)

class HeadMixin:
    """Over a domain's Memory: each composite's instance carries its head
    part's category, its lexical head's surface attributes (as the domain's
    ``surface`` gives them) and its valence. Sides come from the last
    consolidation (``elem_head_side``) or an oracle (``oracle_side``)."""

    elem_head_side: Optional[np.ndarray] = None
    oracle_side: Optional[np.ndarray] = None

    def _head_init(self):
        chunk = set(chunk_attrs(self.spine_depth, self.granularities))
        self._surface_keys = [a for a in self.attrs if a not in chunk]
        self.attrs = list(self.attrs) + [f"lh.{a}" for a in self._surface_keys] + \
            [f"hc.{g}" for g in range(self.granularities)]

    def _side(self, e: int) -> int:
        if not CONFIG.get("representation_sides", True):
            return -1
        src = self.oracle_side if self.oracle_side is not None else self.elem_head_side
        if src is None or e >= len(src) or self.kind[e] == self.PRIMITIVE:
            return -1
        return int(src[e])

    def _head_child(self, e: int) -> int:
        sd = self._side(e)
        if sd == LEFT:
            return self.left[e]
        if sd == RIGHT:
            return self.right[e]
        return -1

    def _lexical_head(self, e: int) -> int:
        while self.kind[e] != self.PRIMITIVE:
            h = self._head_child(e)
            if h < 0:
                return -1
            e = h
        return e

    def _valence(self, e: int) -> str:
        sides = set()
        while self.kind[e] != self.PRIMITIVE:
            sd = self._side(e)
            if sd not in (LEFT, RIGHT):
                return "?"
            sides.add("R" if sd == LEFT else "L")     # a left head has taken a dependent on its right
            e = self.left[e] if sd == LEFT else self.right[e]
        return "".join(sorted(sides))

    def instance(self, e, labels=None, rules=None):
        x = super().instance(e, labels, rules)
        if self.kind[e] == self.PRIMITIVE:
            surf = self.surface(e)
            for a in self._surface_keys:
                x[f"lh.{a}"] = surf.get(a, BLANK)
            for g in range(self.granularities):
                x[f"hc.{g}"] = BLANK
            return x
        lh = self._lexical_head(e)
        surf = self.surface(lh) if lh >= 0 else {}
        for a in self._surface_keys:
            x[f"lh.{a}"] = surf.get(a, BLANK)
        h = self._head_child(e)
        for g in range(self.granularities):
            lab = None if labels is None else labels[g]
            x[f"hc.{g}"] = BLANK if (lab is None or h < 0) else f"S{int(lab[h])}"
        val = self._valence(e)
        if val != "?":
            x["k"] = f"{x['k']}|{val}"
        return x


def head_memory_class(base):
    """``base`` (a Memory subclass) with heads in its representation."""
    def __init__(self, *a, **k):
        base.__init__(self, *a, **k)
        self._head_init()
    return type(f"Head{base.__name__}", (HeadMixin, base), {"__init__": __init__})


HeadMemory = head_memory_class(Memory)

# --------------------------------------------------------------------------- #
# In-process substitution

SUBSTITUTED: List[str] = []
_ORIGINALS: Dict[str, object] = {}


def install(representation: bool = True, composition: bool = True, side: str = "learned",
            representation_sides: bool = True) -> List[str]:
    """Substitute, in this process only, the names the night and the models
    use: ``trellis2.model.compile_grammar`` (composition side),
    ``trellis2.model.Memory`` (representation side), and ``Chart`` in
    ``trellis2.chart``, ``trellis2.model`` and ``trellis2.unsupervised``
    (``make_chart``: ``HeadChart`` for a ``HeadGrammar``)."""
    CONFIG["side"] = side
    CONFIG["representation_sides"] = representation_sides
    names = []

    def put(mod, name, value):
        key = f"{mod.__name__}.{name}"
        _ORIGINALS.setdefault(key, getattr(mod, name))
        setattr(mod, name, value)
        names.append(key)

    if composition:
        put(_model_mod, "compile_grammar", compile_grammar_heads)
        for mod in (_chart_mod, _model_mod, _unsup_mod):
            put(mod, "Chart", make_chart)
    if representation:
        put(_model_mod, "Memory", HeadMemory)
    SUBSTITUTED[:] = names
    return names


def uninstall():
    import importlib
    for key, value in _ORIGINALS.items():
        modname, name = key.rsplit(".", 1)
        setattr(importlib.import_module(modname), name, value)
    _ORIGINALS.clear()
    SUBSTITUTED.clear()
